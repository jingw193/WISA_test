# WISA

This is the official reproduction of [WISA](https://360cvgroup.github.io/WISA/), designed to enhance Text-to-Video models by improving their ability to simulate the real world. 

**[WISA: World Simulator Assistant for Physics-Aware Text-to-Video Generation](https://arxiv.org/pdf/2503.08153)**
</br>
Jing Wang*, Ao Ma*, Ke Cao*, Jun Zheng, Zhanjie Zhang, Jiasong Feng, Shanyuan Liu, Yuhang Ma, Bo Cheng, Dawei Leng‡, Yuhui Yin, Xiaodan Liang‡(*Equal Contribution, ‡Corresponding Authors)
</br>
[![arXiv](https://img.shields.io/badge/arXiv-2503.08153-b31b1b.svg)](https://arxiv.org/pdf/2503.08153)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://360cvgroup.github.io/WISA/)


## 📰 News
- **[2025.05.15]** 🚀 We are excited to announce the official release of WISA's codebase and model weights on GitHub! This implementation is built upon the powerful [finetrains](https://github.com/a-r-r-o-w/finetrainers) framework.
- **[2025.03.28]** 🔥 We have uploaded the [WISA-80K](https://huggingface.co/datasets/qihoo360/WISA-80K) dataset to Hugging Face, including processed video clips and annotations. 
- **[2025.03.12]** We have released our paper [WISA](https://arxiv.org/pdf/2503.08153) and created a dedicated [project homepage](https://360cvgroup.github.io/WISA/). 


<table align="center">
<tr>
  <th align="center">Wan2.1-14B</th>
  <th align="center">WISA</th>
  <th align="center">Prompt</th>
</tr>
<tr>
  <td align="center"><video src="assets/videos/wan_1.mp4">Your browser does not support the video tag.</video></td>
  <td align="center"><video src="assets/videos/wisa_wan_1.mp4">Your browser does not support the video tag.</video></td>
  <th align="center">A dry clump of soil rests on a flat surface, with fine details of its texture and cracks visible. As a heavy object presses down slowly, the surface first releases a faint dust. With increasing pressure, the cracks spread, and the soil splits into uneven fragments. The fragments' surfaces display irregular patterns, while fine powder disperses into the surrounding air, creating a thin veil of dust. The entire scene captures the irreversible transformation of the soil's structure under compression.</th>
</tr>
<tr>
  <td align="center"><video src="assets/videos/wan_3.mp4">Your browser does not support the video tag.</video></td>
  <td align="center"><video src="assets/videos/wisa_wan_3.mp4">Your browser does not support the video tag.</video></td>
  <th align="center">A bowl of clear water sits in the center of a freezer. As the temperature gradually drops, tiny ice crystals begin to form on the surface, resembling a thin layer of frost. The crystals spread rapidly, connecting to create a delicate solid ice film. Over time, the film thickens and eventually covers the entire surface, while the water beneath slowly freezes. Finally, the entire bowl of water solidifies into a transparent block of ice, reflecting the faint light of the freezer and illustrating the transformation from liquid to solid.</th>
</tr>
<tr>
  <td align="center"><video src="/assets/videos/wan_2.mp4">Your browser does not support the video tag.</video></td>
  <td align="center"><video src="/assets/videos/wisa_wan_2.mp4">Your browser does not support the video tag.</video></td>
  <th align="center">The camera focuses on a toothpaste tube on the bathroom countertop. As a finger gently applies pressure to one end of the tube, a steady stream of toothpaste begins to emerge from the opening. At first, the toothpaste flows slowly, but as the pressure increases, the flow speeds up, forming a smooth, white line. The surface of the toothpaste is smooth with delicate textures, and the air around the nozzle is filled with a fresh, minty scent. Eventually, the toothpaste forms a uniform line, neatly resting on the bristles of a toothbrush, ready for brushing.</th>
</tr>
</table>


##  🚀 Quick Started

### 1. Environment Set Up
Clone this repository and install packages.!
```bash
git clone https://github.com/360CVGroup/WISA.git
cd WISA
conda create -n wisa python=3.10
conda activate wisa
pip install -r requirements.txt

```

### 2. Download Pretrained Weights

#### 1. Download Text-to-Video Pretrained Models
Please download CogvideoX and Wan2.1 checkpoints from [ModelScope](https://www.modelscope.cn/home) and put it in `./pretrain_models/`.

```bash
mkdir ./pretrain_models
cd ./pretrain_models
pip install modelscope
modelscope download Wan-AI/Wan2.1-T2V-14B-Diffusers --local_dir ./Wan2.1-T2V-14B-Diffusers
modelscope download ZhipuAI/CogVideoX-5b --local_dir ./CogVideoX-5b-Diffusers
```

#### 2. Download WISA Pretrained Lora and Physical-block Weight
Please download weight from [Huggingface](https://huggingface.co/datasets/qihoo360/WISA-80K) and put it in `./pretrain_models/WISA/`.

```bash
git lfs install
git clone https://huggingface.co/qihoo360/WISA
cd ..
```

### 3. Generate Video

You can revise the `MODEL_TYPE`, `GEN_TYPE`, `PROMPT_PATH`, `OUTPUT_FILE` and `LORA_PATH` in `inference.sh` for different inference settings.
Then run
```bash 
sh inference.sh
```


## ✨ Training

### 1. Download WISA-80K

Download the WISA-80K dataset from [huggingface](https://huggingface.co/datasets/qihoo360/WISA-80K).

### 2. Precomputing Latents and Text Embeddings (Optional)
This project supports precomputing and saving the latent codes of videos and text embeddings to avoid loading the VAE and Text Encoder onto the GPU during training, thereby reducing GPU memory usage. This operation is essential when training Wan2.1-14B; otherwise, it will result in an out-of-memory (OOM) error.

**Step 1**: you need to add the following parameters to the `dataset_cmd` in your training script (like `examples/training/sft/wan/crush_smol_lora/train_wisa.sh`), and ensure you have sufficient storage space available.

```bash
dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
  --dataset_shuffle_buffer_size 10
  --precomputation_items 2000        # Number of samples to precompute
  --enable_precomputation            # Flag to activate precomputation
  --precomputation_once
  --precomputation_dir ./cache/path  # Directory for cached outputs
  --hash_save                        # Enable hash-based filename storage
  --first_samples
)
```
**Step 2**: Configure dataset paths in file `examples/training/sft/cogvideox/crush_smol_lora/training_wisa.json` and execute
```bash
sh examples/training/sft/wan/crush_smol_lora/train_wisa.sh
```

"Note: Process data in batches to prevent CPU cache overload (recommended maximum: 12,000 samples per batch)."

**Step 3**: Disable --enable_precomputation flag
```bash
dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
  --dataset_shuffle_buffer_size 10
  --precomputation_items 2000        # Number of samples to precompute
  # --enable_precomputation            # Flag to activate precomputation
  --precomputation_once
  --precomputation_dir ./cache/path  # Directory for cached outputs
  --hash_save                        # Enable hash-based filename storage
  --first_samples
)
```

### 3. Start Training

```bash
sh examples/training/sft/wan/crush_smol_lora/train_wisa.sh
```
Due to quality issues in the validation phase (bug-induced video generation artifacts causing significant deviation from test-phase results), we have disabled validation.

## 👍 **Acknowledgement**
This work stands on the shoulders of groundbreaking research and open-source contributions. We extend our deepest gratitude to the authors and contributors of the following projects:

* [CogVideoX](https://github.com/THUDM/CogVideo) - For their pioneering work in video generation
* [Wan2.1](https://github.com/Wan-Video/Wan2.1) - For their foundational contributions to large-scale video models


Special thanks to the [finetrains](https://github.com/a-r-r-o-w/finetrainers) framework for enabling efficient model training - your excellent work has been invaluable to this project.


## BibTeX
```
@misc{wang2025wisa,
                title={WISA: World Simulator Assistant for Physics-Aware Text-to-Video Generation}, 
                author={Jing Wang and Ao Ma and Ke Cao and Jun Zheng and Zhanjie Zhang and Jiasong Feng and Shanyuan Liu and Yuhang Ma and Bo Cheng and Dawei Leng and Yuhui Yin and Xiaodan Liang},
                year={2025},
                eprint={2502.08153},
                archivePrefix={arXiv},
                primaryClass={cs.CV},
                url={https://arxiv.org/abs/2502.08153}, 
}
```
