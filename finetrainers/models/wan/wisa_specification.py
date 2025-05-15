import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
)
import math
from .pipeline_wan_wisa import WanPipeline
from .transformer_wan_wisa import WanTransformer3DModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL.Image import Image
from transformers import AutoModel, AutoTokenizer, UMT5EncoderModel
from peft import LoraConfig, get_peft_model_state_dict

from ... import data
from ... import functional as FF
from ...logging import get_logger
from ...processors import ProcessorMixin, T5Processor
from ...typing import ArtifactType, SchedulerType
from ...utils import get_non_null_items
from ..modeling_utils import ModelSpecification


logger = get_logger()


class WanLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Wan VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 3

    def forward(
        self,
        vae: AutoencoderKLWan,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if image is not None:
            video = image.unsqueeze(1)

        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"
        video = video.to(device=device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            # TODO(aryan): refactor in diffusers to have use_slicing attribute
            # if vae.use_slicing and video.shape[0] > 1:
            #     encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
            #     moments = torch.cat(encoded_slices)
            # else:
            #     moments = vae._encode(video)
            moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std)

        return {self.output_names[0]: latents, self.output_names[1]: latents_mean, self.output_names[2]: latents_std}


class WISAWanModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )

        if condition_model_processors is None:
            condition_model_processors = [T5Processor(["encoder_hidden_states", "prompt_attention_mask"])]
        if latent_model_processors is None:
            latent_model_processors = [WanLatentEncodeProcessor(["latents", "latents_mean", "latents_std"])]

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors

    @property
    def _resolution_dim_keys(self):
        return {"latents": (2, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id, revision=self.revision, cache_dir=self.cache_dir
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )

        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id,
                torch_dtype=self.text_encoder_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        else:
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        if self.vae_id is not None:
            vae = AutoencoderKLWan.from_pretrained(
                self.vae_id,
                torch_dtype=self.vae_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        else:
            vae = AutoencoderKLWan.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="vae",
                torch_dtype=self.vae_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )

        return {"vae": vae}

    def load_diffusion_models(self, ckpt_path=None) -> Dict[str, torch.nn.Module]:
        if self.transformer_id is not None:
            transformer = WanTransformer3DModel.from_pretrained(
                self.transformer_id,
                torch_dtype=self.transformer_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        elif ckpt_path is not None:
            phys_weights_path = os.path.join(ckpt_path, "phys_weights.safetensors")
            transformer = WanTransformer3DModel.from_pretrained_load_phys(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                phys_weights_path=phys_weights_path,
                torch_dtype=self.transformer_dtype,
            )
        else:
            transformer = WanTransformer3DModel.from_pretrained_load_phys(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                phys_weights_path=None,
                torch_dtype=self.transformer_dtype
            )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        text_encoder: Optional[UMT5EncoderModel] = None,
        transformer: Optional[WanTransformer3DModel] = None,
        vae: Optional[AutoencoderKLWan] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> WanPipeline:
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
        }
        components = get_non_null_items(components)

        pipe = WanPipeline.from_pretrained(
            self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
        )
        
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
        pipe.scheduler = scheduler

        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)

        if not training:
            pipe.transformer.to(self.transformer_dtype)

        # TODO(aryan): add support in diffusers
        # if enable_slicing:
        #     pipe.vae.enable_slicing()
        # if enable_tiling:
        #     pipe.vae.enable_tiling()
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        return pipe

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        caption: str,
        max_sequence_length: int = 512,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "caption": caption,
            "max_sequence_length": max_sequence_length,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_conditions(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        conditions.pop("prompt_attention_mask", None)
        return conditions

    @torch.no_grad()
    def prepare_priori_or_quantify_priori(
        self,
        priori: torch.Tensor,
        quantify_priori: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "priori": priori,
            "quantify_priori": quantify_priori,
            # "caption": caption,
            # "max_sequence_length": 500,
            # **kwargs,
        }
        return conditions

    @torch.no_grad()
    def prepare_latents(
        self,
        vae: AutoencoderKLWan,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        conditions = {
            "vae": vae,
            "image": image,
            "video": video,
            "generator": generator,
            # We must force this to False because the latent normalization should be done before
            # the posterior is computed. The VAE does not handle this any more:
            # https://github.com/huggingface/diffusers/pull/10998
            "compute_posterior": False,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_latents(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        return conditions

    def forward(
        self,
        transformer: WanTransformer3DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        compute_posterior = False  # See explanation in prepare_latents
        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
        else:
            latents = latent_model_conditions.pop("latents")
            latents_mean = latent_model_conditions.pop("latents_mean")
            latents_std = latent_model_conditions.pop("latents_std")

            mu, logvar = torch.chunk(latents, 2, dim=1)
            mu = self._normalize_latents(mu, latents_mean, latents_std)
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)
            latents = torch.cat([mu, logvar], dim=1)

            posterior = DiagonalGaussianDistribution(latents)
            latents = posterior.sample(generator=generator)
            del posterior

        latents = latents.to(next(transformer.parameters()).device)
        sigmas = sigmas.to(next(transformer.parameters()).device)

        noise = torch.zeros_like(latents).normal_(generator=generator)
        noisy_latents = FF.flow_match_xt(latents, noise, sigmas)
        timesteps = (sigmas.flatten() * 1000.0).long()

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)

        if "priori" in condition_model_conditions.keys() and transformer.whether_classifier:
            pred = transformer(
                **latent_model_conditions,
                **condition_model_conditions,
                timestep=timesteps,
                return_dict=False,
            )[0:2]
        else:
            pred = transformer(
                **latent_model_conditions,
                **condition_model_conditions,
                timestep=timesteps,
                return_dict=False,
            )[0]

        target = FF.flow_match_target(noise, latents)

        return pred, target, sigmas

    def validation(
        self,
        pipeline: WanPipeline,
        prompt: str,
        image: Optional[Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[ArtifactType]:
        if image is not None:
            pipeline = WanImageToVideoPipeline.from_pipe(pipeline)

        priori, quantify_priori = self.obtain_pri(kwargs)
        prompt = prompt + kwargs["phys_law"]
        generation_kwargs = {
            "prompt": prompt,
            "image": image,
            "height": height,
            "width": width,
            "priori": priori,
            "quantify_priori": quantify_priori,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        generation_kwargs = get_non_null_items(generation_kwargs)
        video = pipeline(**generation_kwargs).frames[0]
        return [data.VideoArtifact(value=video)]

    def obtain_pri(self, sample):
        priori_number = [7, 7, 6, 2, 7]
        quantify_priori_obj = 5
        priori_obj = [
            ["collision", "rigid body motion", "elastic motion", "liquid motion", "gas motion", "deformation", "no obvious dynamic phenomenon"],
            ["melting", "solidification", "vaporization", "liquefaction", "explosion", "combustion", "no obvious thermodynamic phenomenon"],
            ["reflection", "refraction", "scattering", "interference and diffraction", "unnatural light source", "no obvious optical phenomenon"],
            ["yes", "no"],
            ["liquids objects appearance", "solid objects appearance", "gas objects appearance", "object decomposition and splitting", "mixing of multiple objects", "object disappearance", "no change"]
        ]

        density = sample['n0']
        densitys = density.split("g/cm³")
        quantify_priori_den = []
        for den in densitys[:-1]:
            phy_den = den.split(" ")
            for index, dstr in enumerate(phy_den):
                if dstr == "to":
                    try:
                        quantify_priori_den.append([float(phy_den[index-1]), float(phy_den[index+1])])
                    except:
                        pass
                    break
                elif index == len(phy_den) - 2:
                    try:
                        quantify_priori_den.append([float(phy_den[index]), float(phy_den[index])])
                    except:
                        pass

        sample["quantify_n0"] = quantify_priori_den

        phy_time = sample["n1"].split(" ")
        for index, t in enumerate(phy_time):
            if t == "to":
                try:
                    quantify_priori_t = [float(phy_time[index-1]), float(phy_time[index+1])]
                except:
                    pass
                break
        sample["quantify_n1"] = quantify_priori_t
                
        phy_tem = sample["n2"].split(" ")
        quantify_priori_tem = None
        for index, t in enumerate(phy_tem):
            if t == "to":
                try:
                    quantify_priori_tem = [float(phy_tem[index-1]), float(phy_tem[index+1])]
                except:
                    pass
                break
        if quantify_priori_tem is None:
            quantify_priori_tem = [20.0, 25.0]
        sample["quantify_n2"] = quantify_priori_tem


        priori = torch.zeros(sum(priori_number))
        quantify_priori = torch.zeros([4 + quantify_priori_obj, 2])

        # priori
        head_index = 0
        for index, objs in enumerate(priori_obj):
            key_name = f"q{index}"
            phy_res = sample[key_name]
            for obj in objs:
                if obj in phy_res:
                    priori[head_index] = 1
                head_index += 1

        # quantify_priori
        for index, density in enumerate(sample["quantify_n0"]):
            if quantify_priori_obj > index:
                quantify_priori[index, 0] = float(density[0])
                quantify_priori[index, 1] = float(density[1])

        quantify_priori[quantify_priori_obj, 0], quantify_priori[quantify_priori_obj, 1] \
                = split_to_coefficient_and_exponent(sample["quantify_n1"][0])

        quantify_priori[quantify_priori_obj + 1, 0], quantify_priori[quantify_priori_obj + 1, 1] \
                = split_to_coefficient_and_exponent(sample["quantify_n1"][1])

        quantify_priori[quantify_priori_obj + 2, 0], quantify_priori[quantify_priori_obj + 2, 1] \
                = split_to_coefficient_and_exponent(sample["quantify_n2"][0])

        quantify_priori[quantify_priori_obj + 3, 0], quantify_priori[quantify_priori_obj + 3, 1] \
                = split_to_coefficient_and_exponent(sample["quantify_n2"][1])

        quantify_priori = quantify_priori.reshape((4 + quantify_priori_obj) * 2)

        return priori.unsqueeze(0), quantify_priori.unsqueeze(0)

    def _save_lora_weights(
        self,
        directory: str,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        *args,
        **kwargs,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            WanPipeline.save_lora_weights(directory, transformer_state_dict, safe_serialization=True)
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_lora_wisa_weights(
        self,
        directory: str,
        checkpointing_steps: int,
        transformer: None,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        *args,
        **kwargs,
    ) -> None:
        # TODO(aryan): this needs refactoring
        save_path = os.path.join(directory, f"checkpoint-{checkpointing_steps}")
        state_dict = get_peft_model_state_dict(transformer, transformer_state_dict)
        # if transformer_state_dict is not None:
        WanPipeline.save_lora_weights(save_path, state_dict, safe_serialization=True)

        phys_weights_to_save = {}
        phys_weights_to_save.update({
            name: param.detach().cpu()
            for name, param in transformer.state_dict().items()
            if "phys" in name
        })
        from safetensors.torch import save_file
        phys_weights_path = f"{save_path}/phys_weights.safetensors"
        save_file(phys_weights_to_save, phys_weights_path)
        print(f"Saved 'phys' weights to {phys_weights_path}")

        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_model(
        self,
        directory: str,
        transformer: WanTransformer3DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = WanTransformer3DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents


def split_to_coefficient_and_exponent(x):
    if x == 0:
        return x, x
    else:
        exponent = math.floor(math.log10(abs(x))) 
        coefficient = x / (10 ** exponent)         
        return coefficient, exponent