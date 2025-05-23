import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import datasets.data_files
import datasets.distributed
import datasets.exceptions
import huggingface_hub
import huggingface_hub.errors
import numpy as np
import PIL.Image
import torch
import torch.distributed.checkpoint.stateful
from diffusers.utils import load_image, load_video
from huggingface_hub import list_repo_files, repo_exists, snapshot_download
from tqdm.auto import tqdm
import math
from pathlib import Path
import hashlib

from .. import constants
from .. import functional as FF
from ..logging import get_logger
from . import utils


import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger()


# fmt: off
MAX_PRECOMPUTABLE_ITEMS_LIMIT = 5000000
COMMON_CAPTION_FILES = ["prompt.txt", "prompts.txt", "caption.txt", "captions.txt"]
COMMON_VIDEO_FILES = ["video.txt", "videos.txt"]
COMMON_IMAGE_FILES = ["image.txt", "images.txt"]
COMMON_WDS_CAPTION_COLUMN_NAMES = ["txt", "text", "caption", "captions", "short_caption", "long_caption", "prompt", "prompts", "short_prompt", "long_prompt", "description", "descriptions", "alt_text", "alt_texts", "alt_caption", "alt_captions", "alt_prompt", "alt_prompts", "alt_description", "alt_descriptions", "image_description", "image_descriptions", "image_caption", "image_captions", "image_prompt", "image_prompts", "image_alt_text", "image_alt_texts", "image_alt_caption", "image_alt_captions", "image_alt_prompt", "image_alt_prompts", "image_alt_description", "image_alt_descriptions", "video_description", "video_descriptions", "video_caption", "video_captions", "video_prompt", "video_prompts", "video_alt_text", "video_alt_texts", "video_alt_caption", "video_alt_captions", "video_alt_prompt", "video_alt_prompts", "video_alt_description"]
# fmt: on


class ImageCaptionFilePairDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        caption_files = sorted(utils.find_files(self.root.as_posix(), "*.txt", depth=0))
        for caption_file in caption_files:
            data_file = self._find_data_file(caption_file)
            if data_file:
                data.append(
                    {
                        "caption": (self.root / caption_file).as_posix(),
                        "image": (self.root / data_file).as_posix(),
                    }
                )

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("image", datasets.Image(mode="RGB"))

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["caption"] = _read_caption_from_file(sample["caption"])
                sample["image"] = _preprocess_image(sample["image"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}

    def _find_data_file(self, caption_file: str) -> str:
        caption_file = pathlib.Path(caption_file)
        data_file = None
        found_data = 0

        for extension in constants.SUPPORTED_IMAGE_FILE_EXTENSIONS:
            image_filename = caption_file.with_suffix(f".{extension}")
            if image_filename.exists():
                found_data += 1
                data_file = image_filename

        if found_data == 0:
            return False
        elif found_data > 1:
            raise ValueError(
                f"Multiple data files found for caption file {caption_file}. Please ensure there is only one data "
                f"file per caption file. The following extensions are supported:\n"
                f"  - Images: {constants.SUPPORTED_IMAGE_FILE_EXTENSIONS}\n"
            )

        return data_file.as_posix()


class VideoCaptionFilePairDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        caption_files = sorted(utils.find_files(self.root.as_posix(), "*.txt", depth=0))
        for caption_file in caption_files:
            data_file = self._find_data_file(caption_file)
            if data_file:
                data.append(
                    {
                        "caption": (self.root / caption_file).as_posix(),
                        "video": (self.root / data_file).as_posix(),
                    }
                )

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("video", datasets.Video())

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["caption"] = _read_caption_from_file(sample["caption"])
                sample["video"] = _preprocess_video(sample["video"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}

    def _find_data_file(self, caption_file: str) -> str:
        caption_file = pathlib.Path(caption_file)
        data_file = None
        found_data = 0

        for extension in constants.SUPPORTED_VIDEO_FILE_EXTENSIONS:
            video_filename = caption_file.with_suffix(f".{extension}")
            if video_filename.exists():
                found_data += 1
                data_file = video_filename

        if found_data == 0:
            return False
        elif found_data > 1:
            raise ValueError(
                f"Multiple data files found for caption file {caption_file}. Please ensure there is only one data "
                f"file per caption file. The following extensions are supported:\n"
                f"  - Videos: {constants.SUPPORTED_VIDEO_FILE_EXTENSIONS}\n"
            )

        return data_file.as_posix()


class ImageFileCaptionFileListDataset(
    torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful
):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        VALID_CAPTION_FILES = ["caption.txt", "captions.txt", "prompt.txt", "prompts.txt"]
        VALID_IMAGE_FILES = ["image.txt", "images.txt"]

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        existing_caption_files = [file for file in VALID_CAPTION_FILES if (self.root / file).exists()]
        existing_image_files = [file for file in VALID_IMAGE_FILES if (self.root / file).exists()]

        if len(existing_caption_files) == 0:
            raise FileNotFoundError(
                f"No caption file found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_image_files) == 0:
            raise FileNotFoundError(
                f"No image file found in {self.root}. Must have exactly one of {VALID_IMAGE_FILES}"
            )
        if len(existing_caption_files) > 1:
            raise ValueError(
                f"Multiple caption files found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_image_files) > 1:
            raise ValueError(
                f"Multiple image files found in {self.root}. Must have exactly one of {VALID_IMAGE_FILES}"
            )

        caption_file = existing_caption_files[0]
        image_file = existing_image_files[0]

        with open((self.root / caption_file).as_posix(), "r") as f:
            captions = f.read().splitlines()
        with open((self.root / image_file).as_posix(), "r") as f:
            images = f.read().splitlines()
            images = [(self.root / image).as_posix() for image in images]

        if len(captions) != len(images):
            raise ValueError(f"Number of captions ({len(captions)}) must match number of images ({len(images)})")

        for caption, image in zip(captions, images):
            data.append({"caption": caption, "image": image})

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("image", datasets.Image(mode="RGB"))

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["image"] = _preprocess_image(sample["image"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoFileCaptionFileListDataset(
    torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful
):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        VALID_CAPTION_FILES = ["caption.txt", "captions.txt", "prompt.txt", "prompts.txt"]
        VALID_VIDEO_FILES = ["video.txt", "videos.txt"]

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        existing_caption_files = [file for file in VALID_CAPTION_FILES if (self.root / file).exists()]
        existing_video_files = [file for file in VALID_VIDEO_FILES if (self.root / file).exists()]

        if len(existing_caption_files) == 0:
            raise FileNotFoundError(
                f"No caption file found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_video_files) == 0:
            raise FileNotFoundError(
                f"No video file found in {self.root}. Must have exactly one of {VALID_VIDEO_FILES}"
            )
        if len(existing_caption_files) > 1:
            raise ValueError(
                f"Multiple caption files found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_video_files) > 1:
            raise ValueError(
                f"Multiple video files found in {self.root}. Must have exactly one of {VALID_VIDEO_FILES}"
            )

        caption_file = existing_caption_files[0]
        video_file = existing_video_files[0]

        with open((self.root / caption_file).as_posix(), "r") as f:
            captions = f.read().splitlines()
        with open((self.root / video_file).as_posix(), "r") as f:
            videos = f.read().splitlines()
            videos = [(self.root / video).as_posix() for video in videos]

        if len(captions) != len(videos):
            raise ValueError(f"Number of captions ({len(captions)}) must match number of videos ({len(videos)})")

        for caption, video in zip(captions, videos):
            data.append({"caption": caption, "video": video})

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("video", datasets.Video())

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["video"] = _preprocess_video(sample["video"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class WISA_VideoFileCaptionFileListDataset(
    torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful
):
    def __init__(self, root: str, infinite: bool = False, model_name: str = None, enable_precomputation: bool = False, precompute_root: str = None) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.model = model_name

        self.enable_precomputation = enable_precomputation
        # if self.enable_precomputation:
        if precompute_root is not None:
            self.precompute_root = pathlib.Path(precompute_root)
            self.precompute_root_exist = self.precompute_root.exists()
        else:
            raise ValueError("lack precompute_root path")

        self.prompts, self.videos, self.phys_prioris, self.video_id = self.get_data_form_json(self.root)
        self.infinite = infinite
        

        self.priori_number = [7, 7, 6, 2, 7]
        self.quantify_priori_obj = 5

        self.priori_obj = [
            ["collision", "rigid body motion", "elastic motion", "liquid motion", "gas motion", "deformation", "no obvious dynamic phenomenon"],
            ["melting", "solidification", "vaporization", "liquefaction", "explosion", "combustion", "no obvious thermodynamic phenomenon"],
            ["reflection", "refraction", "scattering", "interference and diffraction", "unnatural light source", "no obvious optical phenomenon"],
            ["yes", "no"],
            ["liquids objects appearance", "solid objects appearance", "gas objects appearance", "object decomposition and splitting", "mixing of multiple objects", "object disappearance", "no change"]
        ]

        data = []

        if len(self.prompts) != len(self.videos):
            raise ValueError(f"Number of captions ({len(self.prompts)}) must match number of videos ({len(self.videos)})")

        print(f"training data number: {len(self.videos)}")


        if self.enable_precomputation or not self.precompute_root.exists():
            for caption, video, phys_prioris, video_id in zip(self.prompts, self.videos, self.phys_prioris, self.video_id):
                data.append({"caption": caption, "video": video, "priori": phys_prioris, "video_id": video_id})
        else:
            for caption, video, phys_prioris, video_id in zip(self.prompts, self.videos, self.phys_prioris, self.video_id):
                data.append({"caption": caption, "priori": phys_prioris, "video_id": video_id})

        data = datasets.Dataset.from_list(data)
        if self.enable_precomputation or not self.precompute_root.exists():
            data = data.cast_column("video", datasets.Video())

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def get_data_form_json(self, json_path):
        import json
        with open(json_path, "r") as file:
            data = json.load(file)

        import random
        random.shuffle(data)

        prompts = []
        video_path = []
        video_id = []
        prioris = []
        for sample in data:
            
            id_token = "PIKA_CRUSH"
            if not Path(sample['file']).is_file():
                continue

            prompt = sample['captions'] + sample['physical_annotation']['phys_law']


            if "duration" in sample.keys() and float(sample['duration']) < 3:
                continue
  
            if not self.enable_precomputation:
                if "duration" in sample.keys() and (float(sample['duration']) > 10 or float(sample['duration']) < 2):
                    continue

                if "width" in sample.keys() and (float(sample['width']) > 3000 or float(sample['width']) < 720):
                    continue

                if "text_bbox_num" in sample.keys() and float(sample['text_bbox_num']) > 1.0:
                    continue

                if self.precompute_root.exists():
                    prompt_for_hash = FF.remove_prefix(prompt, constants.COMMON_LLM_START_PHRASES)

                    prompt_for_hash = f"{id_token} {prompt_for_hash}"

                    hash_id = str(hashlib.sha256(prompt_for_hash.encode()).hexdigest()) + ".pt"
                    
                    condition_path = self.precompute_root / "condition" / self.model / hash_id
                    latent_path = self.precompute_root / "latent" / self.model / hash_id
                    priori_path = self.precompute_root / "priori" / hash_id
                    q_priori_path = self.precompute_root / "quantify_priori" / hash_id
                    if not (condition_path.is_file() and latent_path.is_file() \
                        and priori_path.is_file() and q_priori_path.is_file()):
                        continue

            prompts.append(prompt)

            video_id.append(sample['video_id'])

            video_path.append(sample['file'])

            del sample['physical_annotation']["phys_law"]
            priori = sample['physical_annotation']
            prioris.append(priori)
        return prompts, video_path, prioris, video_id


    def process_phys_priori(self, sample):
        priori = torch.zeros(sum(self.priori_number))
        quantify_priori = torch.zeros([4 + self.quantify_priori_obj, 2])

        # priori
        head_index = 0
        for index, objs in enumerate(self.priori_obj):
            key_name = f"q{index}"
            phy_res = sample[key_name]
            for obj in objs:
                if obj in phy_res:
                    priori[head_index] = 1
                head_index += 1

        # quantify_priori
        for index, density in enumerate(sample["quantify_n0"]):
            if self.quantify_priori_obj > index:
                quantify_priori[index, 0] = float(density[0])
                quantify_priori[index, 1] = float(density[1])

        quantify_priori[self.quantify_priori_obj, 0], quantify_priori[self.quantify_priori_obj, 1] \
                = self.split_to_coefficient_and_exponent(sample["quantify_n1"][0])

        quantify_priori[self.quantify_priori_obj + 1, 0], quantify_priori[self.quantify_priori_obj + 1, 1] \
                = self.split_to_coefficient_and_exponent(sample["quantify_n1"][1])

        quantify_priori[self.quantify_priori_obj + 2, 0], quantify_priori[self.quantify_priori_obj + 2, 1] \
                = self.split_to_coefficient_and_exponent(sample["quantify_n2"][0])

        quantify_priori[self.quantify_priori_obj + 3, 0], quantify_priori[self.quantify_priori_obj + 3, 1] \
                = self.split_to_coefficient_and_exponent(sample["quantify_n2"][1])

        quantify_priori = quantify_priori.reshape((4 + self.quantify_priori_obj) * 2)

        return priori, quantify_priori

    def split_to_coefficient_and_exponent(self, x):
        if x == 0:
            return x, x
        else:
            exponent = math.floor(math.log10(abs(x)))  
            coefficient = x / (10 ** exponent)       
            return coefficient, exponent

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                if self.enable_precomputation or not self.precompute_root_exist:
                    sample["video"] = _preprocess_video(sample["video"])
                priori, quantify_priori = self.process_phys_priori(sample["priori"])
                sample["priori"] = priori
                sample["quantify_priori"] = quantify_priori
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class ImageFolderDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = datasets.load_dataset("imagefolder", data_dir=self.root.as_posix(), split="train")

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["image"] = _preprocess_image(sample["image"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoFolderDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = datasets.load_dataset("videofolder", data_dir=self.root.as_posix(), split="train")

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["video"] = _preprocess_video(sample["video"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class ImageWebDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(
        self,
        dataset_name: str,
        infinite: bool = False,
        column_names: Union[str, List[str]] = "__auto__",
        weights: Dict[str, float] = -1,
        **kwargs,
    ) -> None:
        super().__init__()

        assert weights == -1 or isinstance(
            weights, dict
        ), "`weights` must be a dictionary of probabilities for each caption column"

        self.dataset_name = dataset_name
        self.infinite = infinite

        data = datasets.load_dataset(dataset_name, split="train", streaming=True)

        if column_names == "__auto__":
            if weights == -1:
                caption_columns = [column for column in data.column_names if column in COMMON_WDS_CAPTION_COLUMN_NAMES]
                if len(caption_columns) == 0:
                    raise ValueError(
                        f"No common caption column found in the dataset. Supported columns are: {COMMON_WDS_CAPTION_COLUMN_NAMES}"
                    )
                weights = [1] * len(caption_columns)
            else:
                caption_columns = list(weights.keys())
                weights = list(weights.values())
                if not all(column in data.column_names for column in caption_columns):
                    raise ValueError(
                        f"Caption columns {caption_columns} not found in the dataset. Available columns are: {data.column_names}"
                    )
        else:
            if isinstance(column_names, str):
                if column_names not in data.column_names:
                    raise ValueError(
                        f"Caption column {column_names} not found in the dataset. Available columns are: {data.column_names}"
                    )
                caption_columns = [column_names]
                weights = [1] if weights == -1 else [weights.get(column_names)]
            elif isinstance(column_names, list):
                if not all(column in data.column_names for column in column_names):
                    raise ValueError(
                        f"Caption columns {column_names} not found in the dataset. Available columns are: {data.column_names}"
                    )
                caption_columns = column_names
                weights = [1] if weights == -1 else [weights.get(column) for column in column_names]
            else:
                raise ValueError(f"Unsupported type for column_name: {type(column_names)}")

        for column_names in constants.SUPPORTED_IMAGE_FILE_EXTENSIONS:
            if column_names in data.column_names:
                data = data.cast_column(column_names, datasets.Image(mode="RGB"))
                data = data.rename_column(column_names, "image")
                break

        self._data = data
        self._sample_index = 0
        self._precomputable_once = False
        self._caption_columns = caption_columns
        self._weights = weights

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                caption_column = random.choices(self._caption_columns, weights=self._weights, k=1)[0]
                sample["caption"] = sample[caption_column]
                sample["image"] = _preprocess_image(sample["image"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_index = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoWebDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(
        self,
        dataset_name: str,
        infinite: bool = False,
        column_names: Union[str, List[str]] = "__auto__",
        weights: Dict[str, float] = -1,
        **kwargs,
    ) -> None:
        super().__init__()

        assert weights == -1 or isinstance(
            weights, dict
        ), "`weights` must be a dictionary of probabilities for each caption column"

        self.dataset_name = dataset_name
        self.infinite = infinite

        data = datasets.load_dataset(dataset_name, split="train", streaming=True)

        if column_names == "__auto__":
            if weights == -1:
                caption_columns = [column for column in data.column_names if column in COMMON_WDS_CAPTION_COLUMN_NAMES]
                if len(caption_columns) == 0:
                    raise ValueError(
                        f"No common caption column found in the dataset. Supported columns are: {COMMON_WDS_CAPTION_COLUMN_NAMES}"
                    )
                weights = [1] * len(caption_columns)
            else:
                caption_columns = list(weights.keys())
                weights = list(weights.values())
                if not all(column in data.column_names for column in caption_columns):
                    raise ValueError(
                        f"Caption columns {caption_columns} not found in the dataset. Available columns are: {data.column_names}"
                    )
        else:
            if isinstance(column_names, str):
                if column_names not in data.column_names:
                    raise ValueError(
                        f"Caption column {column_names} not found in the dataset. Available columns are: {data.column_names}"
                    )
                caption_columns = [column_names]
                weights = [1] if weights == -1 else [weights.get(column_names)]
            elif isinstance(column_names, list):
                if not all(column in data.column_names for column in column_names):
                    raise ValueError(
                        f"Caption columns {column_names} not found in the dataset. Available columns are: {data.column_names}"
                    )
                caption_columns = column_names
                weights = [1] if weights == -1 else [weights.get(column) for column in column_names]
            else:
                raise ValueError(f"Unsupported type for column_name: {type(column_names)}")

        for column_names in constants.SUPPORTED_VIDEO_FILE_EXTENSIONS:
            if column_names in data.column_names:
                data = data.cast_column(column_names, datasets.Video())
                data = data.rename_column(column_names, "video")
                break

        self._data = data
        self._sample_index = 0
        self._precomputable_once = False
        self._caption_columns = caption_columns
        self._weights = weights

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                caption_column = random.choices(self._caption_columns, weights=self._weights, k=1)[0]
                sample["caption"] = sample[caption_column]
                sample["video"] = _preprocess_video(sample["video"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_index = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class ValidationDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename: str):
        super().__init__()

        self.filename = pathlib.Path(filename)

        if not self.filename.exists():
            raise FileNotFoundError(f"File {self.filename.as_posix()} does not exist")

        if self.filename.suffix == ".csv":
            data = datasets.load_dataset("csv", data_files=self.filename.as_posix(), split="train")
        elif self.filename.suffix == ".json":
            data = datasets.load_dataset("json", data_files=self.filename.as_posix(), split="train", field="data")
        elif self.filename.suffix == ".parquet":
            data = datasets.load_dataset("parquet", data_files=self.filename.as_posix(), split="train")
        elif self.filename.suffix == ".arrow":
            data = datasets.load_dataset("arrow", data_files=self.filename.as_posix(), split="train")
        else:
            _SUPPORTED_FILE_FORMATS = [".csv", ".json", ".parquet", ".arrow"]
            raise ValueError(
                f"Unsupported file format {self.filename.suffix} for validation dataset. Supported formats are: {_SUPPORTED_FILE_FORMATS}"
            )

        self._data = data.to_iterable_dataset()

    def __iter__(self):
        for sample in self._data:
            # For consistency reasons, we mandate that "caption" is always present in the validation dataset.
            # However, since the model specifications use "prompt", we create an alias here.
            sample["prompt"] = sample["caption"]

            # Load image or video if the path is provided
            # TODO(aryan): need to handle custom columns here for control conditions
            sample["image"] = None
            sample["video"] = None

            if sample.get("image_path", None) is not None:
                image_path = pathlib.Path(sample["image_path"])
                if not image_path.is_file():
                    logger.warning(f"Image file {image_path.as_posix()} does not exist.")
                else:
                    sample["image"] = load_image(sample["image_path"])

            if sample.get("video_path", None) is not None:
                video_path = pathlib.Path(sample["video_path"])
                if not video_path.is_file():
                    logger.warning(f"Video file {video_path.as_posix()} does not exist.")
                else:
                    sample["video"] = load_video(sample["video_path"])

            sample = {k: v for k, v in sample.items() if v is not None}
            yield sample


class IterableDatasetPreprocessingWrapper(
    torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful
):
    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        dataset_type: str,
        id_token: Optional[str] = None,
        image_resolution_buckets: List[Tuple[int, int]] = None,
        video_resolution_buckets: List[Tuple[int, int, int]] = None,
        reshape_mode: str = "bicubic",
        remove_common_llm_caption_prefixes: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.dataset = dataset
        self.dataset_type = dataset_type
        self.id_token = id_token
        self.image_resolution_buckets = image_resolution_buckets
        self.video_resolution_buckets = video_resolution_buckets
        self.reshape_mode = reshape_mode
        self.remove_common_llm_caption_prefixes = remove_common_llm_caption_prefixes

        logger.info(
            f"Initializing IterableDatasetPreprocessingWrapper for the dataset with the following configuration:\n"
            f"  - Dataset Type: {dataset_type}\n"
            f"  - ID Token: {id_token}\n"
            f"  - Image Resolution Buckets: {image_resolution_buckets}\n"
            f"  - Video Resolution Buckets: {video_resolution_buckets}\n"
            f"  - Reshape Mode: {reshape_mode}\n"
            f"  - Remove Common LLM Caption Prefixes: {remove_common_llm_caption_prefixes}\n"
        )

    def __iter__(self):
        logger.info("Starting IterableDatasetPreprocessingWrapper for the dataset")
        for sample in iter(self.dataset):
            if self.dataset_type == "image":
                if self.image_resolution_buckets:
                    sample["_original_num_frames"] = 1
                    sample["_original_height"] = sample["image"].size(1)
                    sample["_original_width"] = sample["image"].size(2)
                    sample["image"] = FF.resize_to_nearest_bucket_image(
                        sample["image"], self.image_resolution_buckets, self.reshape_mode
                    )
            elif self.dataset_type == "video":
                # pass
                if getattr(self.dataset, 'enable_precomputation', True) or not getattr(self.dataset, "precompute_root_exist", True):
                    if self.video_resolution_buckets:
                        sample["_original_num_frames"] = sample["video"].size(0)
                        sample["_original_height"] = sample["video"].size(2)
                        sample["_original_width"] = sample["video"].size(3)
                        sample["video"], _first_frame_only = FF.resize_to_nearest_bucket_video(
                            sample["video"], self.video_resolution_buckets, self.reshape_mode
                        )
                        if _first_frame_only:
                            msg = (
                                "The number of frames in the video is less than the minimum bucket size "
                                "specified. The first frame is being used as a single frame video. This "
                                "message is logged at the first occurence and for every 128th occurence "
                                "after that."
                            )
                            logger.log_freq("WARNING", "BUCKET_TEMPORAL_SIZE_UNAVAILABLE", msg, frequency=128)
                            sample["video"] = sample["video"][0]

            if self.remove_common_llm_caption_prefixes:
                sample["caption"] = FF.remove_prefix(sample["caption"], constants.COMMON_LLM_START_PHRASES)

            if self.id_token is not None:
                sample["caption"] = f"{self.id_token} {sample['caption']}"

            yield sample

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {"dataset": self.dataset.state_dict()}


class IterableCombinedDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, datasets: List[torch.utils.data.IterableDataset], buffer_size: int, shuffle: bool = False):
        super().__init__()

        self.datasets = datasets
        self.buffer_size = buffer_size
        self.shuffle = shuffle

        logger.info(
            f"Initializing IterableCombinedDataset with the following configuration:\n"
            f"  - Number of Datasets: {len(datasets)}\n"
            f"  - Buffer Size: {buffer_size}\n"
            f"  - Shuffle: {shuffle}\n"
        )

    def __iter__(self):
        logger.info(f"Starting IterableCombinedDataset with {len(self.datasets)} datasets")
        iterators = [iter(dataset) for dataset in self.datasets]
        buffer = []
        per_iter = max(1, self.buffer_size // len(iterators))

        for index, it in enumerate(iterators):
            for _ in tqdm(range(per_iter), desc=f"Filling buffer from data iterator {index}"):
                try:
                    buffer.append((it, next(it)))
                except StopIteration:
                    continue

        while len(buffer) > 0:
            idx = 0
            if self.shuffle:
                idx = random.randint(0, len(buffer) - 1)
            current_it, sample = buffer.pop(idx)
            yield sample
            try:
                buffer.append((current_it, next(current_it)))
            except StopIteration:
                pass

    def load_state_dict(self, state_dict):
        for dataset, dataset_state_dict in zip(self.datasets, state_dict["datasets"]):
            dataset.load_state_dict(dataset_state_dict)

    def state_dict(self):
        return {"datasets": [dataset.state_dict() for dataset in self.datasets]}


# TODO(aryan): maybe write a test for this
def initialize_dataset(
    dataset_name_or_root: str,
    dataset_type: str = "video",
    streaming: bool = True,
    infinite: bool = False,
    model_name: str = None,
    enable_precomputation: bool = False, 
    precompute_root: str = None,
    *,
    _caption_options: Optional[Dict[str, Any]] = None,
) -> torch.utils.data.IterableDataset:
    assert dataset_type in ["image", "video"]

    try:
        does_repo_exist_on_hub = repo_exists(dataset_name_or_root, repo_type="dataset")
    except huggingface_hub.errors.HFValidationError:
        does_repo_exist_on_hub = False

    if does_repo_exist_on_hub:
        return _initialize_hub_dataset(dataset_name_or_root, dataset_type, infinite, _caption_options=_caption_options)
    else:
        return _initialize_local_dataset(dataset_name_or_root, dataset_type, infinite, model_name, enable_precomputation, precompute_root)


def combine_datasets(
    datasets: List[torch.utils.data.IterableDataset], buffer_size: int, shuffle: bool = False
) -> torch.utils.data.IterableDataset:
    return IterableCombinedDataset(datasets=datasets, buffer_size=buffer_size, shuffle=shuffle)


def wrap_iterable_dataset_for_preprocessing(
    dataset: torch.utils.data.IterableDataset, dataset_type: str, config: Dict[str, Any]
) -> torch.utils.data.IterableDataset:
    return IterableDatasetPreprocessingWrapper(dataset, dataset_type, **config)


def _initialize_local_dataset(
        dataset_name_or_root: str, 
        dataset_type: str, 
        infinite: bool = False, 
        model_name: str = None, 
        enable_precomputation: bool = False, 
        precompute_root: str = None,):
    root = pathlib.Path(dataset_name_or_root)
    supported_metadata_files = ["metadata.json", "metadata.jsonl", "metadata.csv"]
    metadata_files = [root / metadata_file for metadata_file in supported_metadata_files]
    metadata_files = [metadata_file for metadata_file in metadata_files if metadata_file.exists()]

    if len(metadata_files) > 1:
        raise ValueError("Found multiple metadata files. Please ensure there is only one metadata file.")

    if len(metadata_files) == 1:
        if dataset_type == "image":
            dataset = ImageFolderDataset(root.as_posix(), infinite=infinite)
        else:
            dataset = VideoFolderDataset(root.as_posix(), infinite=infinite)
        return dataset

    if root.name.endswith(".json") and dataset_type == "video":
        dataset = WISA_VideoFileCaptionFileListDataset(
            root, 
            infinite=infinite, 
            model_name=model_name, 
            enable_precomputation=enable_precomputation, 
            precompute_root=precompute_root
        )
    elif _has_data_caption_file_pairs(root, remote=False):
        if dataset_type == "image":
            dataset = ImageCaptionFilePairDataset(root.as_posix(), infinite=infinite)
        else:
            dataset = VideoCaptionFilePairDataset(root.as_posix(), infinite=infinite)
    elif _has_data_file_caption_file_lists(root, remote=False):
        if dataset_type == "image":
            dataset = ImageFileCaptionFileListDataset(root.as_posix(), infinite=infinite)
        else:
            dataset = VideoFileCaptionFileListDataset(root.as_posix(), infinite=infinite)
    else:
        raise ValueError(
            f"Could not find any supported dataset structure in the directory {root}. Please open an issue at "
            f"https://github.com/a-r-r-o-w/finetrainers with information about your dataset structure and we will "
            f"help you set it up."
        )

    return dataset


def _initialize_hub_dataset(
    dataset_name: str, dataset_type: str, infinite: bool = False, *, _caption_options: Optional[Dict[str, Any]] = None
):
    repo_file_list = list_repo_files(dataset_name, repo_type="dataset")
    if _has_data_caption_file_pairs(repo_file_list, remote=True):
        return _initialize_data_caption_file_dataset_from_hub(dataset_name, dataset_type, infinite)
    elif _has_data_file_caption_file_lists(repo_file_list, remote=True):
        return _initialize_data_file_caption_file_dataset_from_hub(dataset_name, dataset_type, infinite)

    has_tar_files = any(file.endswith(".tar") or file.endswith(".parquet") for file in repo_file_list)
    if has_tar_files:
        return _initialize_webdataset(dataset_name, dataset_type, infinite, _caption_options=_caption_options)

    # TODO(aryan): This should be improved
    caption_files = [pathlib.Path(file).name for file in repo_file_list if file.endswith(".txt")]
    if len(caption_files) < MAX_PRECOMPUTABLE_ITEMS_LIMIT:
        try:
            dataset_root = snapshot_download(dataset_name, repo_type="dataset")
            if dataset_type == "image":
                dataset = ImageFolderDataset(dataset_root, infinite=infinite)
            else:
                dataset = VideoFolderDataset(dataset_root, infinite=infinite)
            return dataset
        except Exception:
            pass

    raise ValueError(f"Could not load dataset {dataset_name} from the HF Hub")


def _initialize_data_caption_file_dataset_from_hub(
    dataset_name: str, dataset_type: str, infinite: bool = False
) -> torch.utils.data.IterableDataset:
    logger.info(f"Downloading dataset {dataset_name} from the HF Hub")
    dataset_root = snapshot_download(dataset_name, repo_type="dataset")
    if dataset_type == "image":
        return ImageCaptionFilePairDataset(dataset_root, infinite=infinite)
    else:
        return VideoCaptionFilePairDataset(dataset_root, infinite=infinite)


def _initialize_data_file_caption_file_dataset_from_hub(
    dataset_name: str, dataset_type: str, infinite: bool = False
) -> torch.utils.data.IterableDataset:
    logger.info(f"Downloading dataset {dataset_name} from the HF Hub")
    dataset_root = snapshot_download(dataset_name, repo_type="dataset")
    if dataset_type == "image":
        return ImageFileCaptionFileListDataset(dataset_root, infinite=infinite)
    else:
        return VideoFileCaptionFileListDataset(dataset_root, infinite=infinite)


def _initialize_webdataset(
    dataset_name: str, dataset_type: str, infinite: bool = False, _caption_options: Optional[Dict[str, Any]] = None
) -> torch.utils.data.IterableDataset:
    logger.info(f"Streaming webdataset {dataset_name} from the HF Hub")
    _caption_options = _caption_options or {}
    if dataset_type == "image":
        return ImageWebDataset(dataset_name, infinite=infinite, **_caption_options)
    else:
        return VideoWebDataset(dataset_name, infinite=infinite, **_caption_options)


def _has_data_caption_file_pairs(root: Union[pathlib.Path, List[str]], remote: bool = False) -> bool:
    # TODO(aryan): this logic can be improved
    if not remote:
        caption_files = utils.find_files(root.as_posix(), "*.txt", depth=0)
        for caption_file in caption_files:
            caption_file = pathlib.Path(caption_file)
            for extension in [*constants.SUPPORTED_IMAGE_FILE_EXTENSIONS, *constants.SUPPORTED_VIDEO_FILE_EXTENSIONS]:
                data_filename = caption_file.with_suffix(f".{extension}")
                if data_filename.exists():
                    return True
        return False
    else:
        caption_files = [file for file in root if file.endswith(".txt")]
        for caption_file in caption_files:
            caption_file = pathlib.Path(caption_file)
            for extension in [*constants.SUPPORTED_IMAGE_FILE_EXTENSIONS, *constants.SUPPORTED_VIDEO_FILE_EXTENSIONS]:
                data_filename = caption_file.with_suffix(f".{extension}").name
                if data_filename in root:
                    return True
        return False


def _has_data_file_caption_file_lists(root: Union[pathlib.Path, List[str]], remote: bool = False) -> bool:
    # TODO(aryan): this logic can be improved
    if not remote:
        file_list = {x.name for x in root.iterdir()}
        has_caption_files = any(file in file_list for file in COMMON_CAPTION_FILES)
        has_video_files = any(file in file_list for file in COMMON_VIDEO_FILES)
        has_image_files = any(file in file_list for file in COMMON_IMAGE_FILES)
        return has_caption_files and (has_video_files or has_image_files)
    else:
        has_caption_files = any(file in root for file in COMMON_CAPTION_FILES)
        has_video_files = any(file in root for file in COMMON_VIDEO_FILES)
        has_image_files = any(file in root for file in COMMON_IMAGE_FILES)
        return has_caption_files and (has_video_files or has_image_files)


def _read_caption_from_file(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read().strip()


def _preprocess_image(image: PIL.Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).contiguous() / 127.5 - 1.0
    return image


def _preprocess_video(video: decord.VideoReader) -> torch.Tensor:
    video = video.get_batch(list(range(len(video))))
    video = video.permute(0, 3, 1, 2).contiguous()
    video = video.float() / 127.5 - 1.0
    return video
