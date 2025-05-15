import argparse
from typing import Literal, Optional
import os
import json
import torch
import math
from diffusers.utils import export_to_video
import imageio
import numpy as np
import ffmpeg


RESOLUTION_MAP = {
    # wanx 2.1 *
    "wanx2_1-14b": (832, 480),
    # cogvideox *
    "cogvideox-5b": (720, 480),
}

def split_to_coefficient_and_exponent(x):
    if x == 0:
        return x, x
    else:
        exponent = math.floor(math.log10(abs(x)))  
        coefficient = x / (10 ** exponent)         
        return coefficient, exponent

def load_prompt_json(json_path, dtype, generate_type):
    priori_number = [7, 7, 6, 2, 7]
    quantify_priori_obj = 5
    priori_obj = [
        ["collision", "rigid body motion", "elastic motion", "liquid motion", "gas motion", "deformation", "no obvious dynamic phenomenon"],
        ["melting", "solidification", "vaporization", "liquefaction", "explosion", "combustion", "no obvious thermodynamic phenomenon"],
        ["reflection", "refraction", "scattering", "interference and diffraction", "unnatural light source", "no obvious optical phenomenon"],
        ["yes", "no"],
        ["liquids objects appearance", "solid objects appearance", "gas objects appearance", "object decomposition and splitting", "mixing of multiple objects", "object disappearance", "no change"]
    ]
    sample_data = []
    with open(json_path, "r", encoding="utf-8") as file:
        datas = json.load(file)
    for data in datas:
        data_dict = {}
        if generate_type == "wisa":
            try:
                data_dict["caption"] = data['detail_caption'] + data['phys_law']
            except:
                data_dict["caption"] = data['en_caption'] + data['phys_law']
        else:
            data_dict["caption"] = data['en_caption']

        if "n0" not in data.keys():
            continue

        density = data['n0']
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

        data["quantify_n0"] = quantify_priori_den

        phy_time = data["n1"].split(" ")
        for index, t in enumerate(phy_time):
            if t == "to":
                try:
                    quantify_priori_t = [float(phy_time[index-1]), float(phy_time[index+1])]
                except:
                    pass
                break
        data["quantify_n1"] = quantify_priori_t
                
        phy_tem = data["n2"].split(" ")
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
        data["quantify_n2"] = quantify_priori_tem


        priori = torch.zeros(sum(priori_number))
        quantify_priori = torch.zeros([4 + quantify_priori_obj, 2])

        # ---------------priori--------------------
        head_index = 0
        for index, objs in enumerate(priori_obj):
            key_name = f"q{index}"
            phy_res = data[key_name]
            for obj in objs:
                if obj in phy_res:
                    priori[head_index] = 1
                head_index += 1

        # ---------------quantify_priori--------------------
        for index, density in enumerate(data["quantify_n0"]):
            if quantify_priori_obj > index:
                quantify_priori[index, 0] = float(density[0])
                quantify_priori[index, 1] = float(density[1])

        quantify_priori[quantify_priori_obj, 0], quantify_priori[quantify_priori_obj, 1] \
                = split_to_coefficient_and_exponent(data["quantify_n1"][0])

        quantify_priori[quantify_priori_obj + 1, 0], quantify_priori[quantify_priori_obj + 1, 1] \
                = split_to_coefficient_and_exponent(data["quantify_n1"][1])

        quantify_priori[quantify_priori_obj + 2, 0], quantify_priori[quantify_priori_obj + 2, 1] \
                = split_to_coefficient_and_exponent(data["quantify_n2"][0])

        quantify_priori[quantify_priori_obj + 3, 0], quantify_priori[quantify_priori_obj + 3, 1] \
                = split_to_coefficient_and_exponent(data["quantify_n2"][1])

        quantify_priori = quantify_priori.reshape((4 + quantify_priori_obj) * 2)

        data_dict['priori'] = priori.to(dtype=dtype)
        data_dict['quantify_priori'] = quantify_priori.to(dtype=dtype)

        sample_data.append(data_dict)

    return sample_data


def generate_video(
    model_path: str, 
    prompts: list[str, dict],
    lora_path: Optional[str] = None, 
    lora_name: str = "lora_adapter",
    lora_rank: int = 128,
    lora_alpha: int = 64,
    num_frames: int = 81,
    output_file: str = "./outputs/sample/",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0, 
    phys_guidance_scale: float = 6.0, 
    generate_type: str = Literal["lora", "wisa", "baseline"],
    model_type: str = Literal["cogvideox", "wanx2_1"],  
    fps: int = 16,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
):
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    if model_type == "wanx2_1":
        model_name = "wanx2_1-14b"
    elif model_type == "cogvideox":
        model_name = "cogvideox-5b"
    else:
        ValueError(f"Eorror: please enter a valid model_type. Supported values: ['wanx2_1', 'cogvideox']")

    transformer = None
    if model_type == "wanx2_1":
        model_name = "wanx2_1-14b"
        if generate_type == "wisa":
            # do not use physical cfg
            from finetrainers.models.wan.pipeline_wan_wisa import WanPipeline as generation_pipeline

            # use physical cfg
            # from finetrainers.models.wan.pipeline_wan_wisa_multicfg import WanPipeline as generation_pipeline

            from finetrainers.models.wan.transformer_wan_wisa import WanTransformer3DModel as diffusion_transfomer
            phys_weight_path = os.path.join(lora_path, "phys_weights.safetensors")
            transformer = diffusion_transfomer.from_pretrained_load_phys(
                pretrained_model_path=model_path,
                subfolder="transformer",
                phys_weights_path=phys_weight_path,
                torch_dtype=dtype,
            )
        else:
            from diffusers import WanPipeline as generation_pipeline
    elif model_type == "cogvideox":
        model_name = "cogvideox-5b"
        if generate_type == "wisa":
            # do not use physical cfg
            from finetrainers.models.cogvideox.pipeline_cogvideox_wisa import CogVideoXPipeline as generation_pipeline

            # use physical cfg
            # from finetrainers.models.cogvideox.pipeline_cogvideox_wisa_multicfg import CogVideoXPipeline as generation_pipeline

            from finetrainers.models.cogvideox.transformer_cogvideox_wisa import CogVideoXTransformer3DModel as diffusion_transfomer
            phys_weight_path = os.path.join(lora_path, "phys_weights.safetensors")
            transformer = diffusion_transfomer.from_pretrained_load_phys(
                pretrained_model_path=model_path,
                subfolder="transformer",
                phys_weights_path=phys_weight_path,
                torch_dtype=dtype,
            )
        else:
            from diffusers import CogVideoXPipeline as generation_pipeline


    desired_resolution = RESOLUTION_MAP[model_name]
    width, height = desired_resolution

    if transformer is not None:
        pipe = generation_pipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=dtype).to("cuda")
    else:
        pipe = generation_pipeline.from_pretrained(model_path, torch_dtype=dtype).to("cuda")

    if lora_path and (generate_type == "lora" or generate_type == "wisa"):
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name=lora_name)
        pipe.set_adapters([lora_name], [lora_alpha / lora_rank])

    if model_type == "cogvideox":
        from diffusers import CogVideoXDPMScheduler
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif model_type == "wanx2_1":
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
        pipe.scheduler = scheduler
        pipe.to("cuda")

    pipe.enable_model_cpu_offload()

    generation_kwargs = {}
    if model_type == "cogvideox":
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        generation_kwargs = {
            "use_dynamic_cfg": True,
        }
        if generate_type == "wisa":
            generation_kwargs["max_sequence_length"] = 500

    print(generation_kwargs)

    for index, prompt in enumerate(prompts):
        print(prompt)
        video_name = f"video_{index}.mp4"
        print(f"generation idx {index} video")
        video_save_path = os.path.join(output_file, video_name)

        if generate_type == "wisa":
            text_caption = prompt["caption"]
            video = pipe(
                prompt=text_caption,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                phys_guidance_scale=phys_guidance_scale,
                priori=prompt["priori"],
                quantify_priori=prompt["quantify_priori"],
                generator=torch.Generator().manual_seed(seed),
                **generation_kwargs
            ).frames[0]
        else:
            if isinstance(prompt, dict):
                prompt = prompt["caption"]
            video = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
                **generation_kwargs
            ).frames[0]
        frames = [np.array(img) for img in video]
        if model_type == "wanx2_1":
            frames = [(frame * 255).astype(np.uint8) for frame in frames]
        imageio.mimsave(video_save_path, frames, fps=fps, codec="libx264")
        print(f"save generated video at {video_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate video using CogVideoX and LoRA weights")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the video generation")
    parser.add_argument("--prompt_path", type=str, default="prompts_file", help="The description of the video to be generated")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-5B", help="Base Model path or HF ID")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA weights")
    parser.add_argument("--lora_name", type=str, default="lora_adapter", help="Name of the LoRA adapter")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument("--lora_alpha", type=int, default=2, help="The rank of the LoRA weights")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--phys_guidance_scale", type=float, default=3.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of steps for the inference process")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--generate_type", type=str, default="lora", help="The type of video generation")
    parser.add_argument("--model_type", type=str, default="cogvideox", help="The type of video generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output video")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    if os.path.exists(args.prompt_path):
        prompts = load_prompt_json(args.prompt_path, dtype, args.generate_type)
    else:
        prompts = [args.prompt]
    
    generate_video(
        model_path=args.model_path,
        prompts=prompts, 
        lora_path=args.lora_path, 
        lora_name=args.lora_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_frames=args.num_frames,
        output_file=args.output_file,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        phys_guidance_scale=args.phys_guidance_scale,
        generate_type=args.generate_type,
        model_type=args.model_type,
        fps=args.fps,
        seed=args.seed,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
