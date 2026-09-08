# Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation

### Joy is dopamine’s handiwork—whether in humans or in robotics.

         





## 🗞️ News

- `**2026-05-13**`: 🤗 We released [Robo-Dopamine-GRM-Dataset](https://huggingface.co/datasets/tanhuajie2001/Robo-Dopamine-GRM-Dataset), fully 35M training dataset for Robo-Dopamine-GRM series.
- `**2026-04-05**`: 🤗 We released [Robo-Dopamine-GRM-2.0-4B-Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview) model in HF. This is a ***lightweight*** GRM-2.0 preview version, achieving approximately 80% accuracy of [Robo-Dopamine-GRM-2.0-8B-Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview).
- `**2026-03-23`**: 📊 We employ [Robo-Dopamine-GRM](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview) as a ***Judge*** to evaluate existing VLAs on the [RoboChallenge](https://robochallenge.ai/) benchmark across multiple dimensions and metrics. Please refer to the  [📊 **Leaderboard](https://prm-as-a-judge.github.io/leaderboard.html)** and [📑 **Blog](https://prm-as-a-judge.github.io/blog.html)** for details.
- `**2026-03-05`**: 🔥🔥🔥 We released [Robo-Dopamine-GRM-2.0-8B-Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview) model in HF. **Highly recommend trying the more versatile and stable GRM-2.0-preview version**. It currently supports ***single-view/multi-view*** use cases, ***both with and without reference target images***, please refer to [Quick Start](https://github.com/FlagOpen/Robo-Dopamine/tree/main?tab=readme-ov-file#-simple-usage) for details.
- `**2026-03-02`**: 🤗 We released [Robo-Dopamine-GRM-8B](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-8B) model in HF.
- `**2026-02-22**`: 🔥🔥🔥 **Robo-Dopamine** gets accepted to **CVPR 2026**! See you in Denver, Colorado, USA!
- `**2026-02-10`**: ⚡  We released data generation pipeline and finetune codes. ***Try to finetune with your own data***.
- `**2026-01-26`**: 🔍 We released [Robo-Dopamine-Bench](https://huggingface.co/datasets/tanhuajie2001/Robo-Dopamine-Bench) benchmark and evaluation codes.
- `**2026-01-08**`: 🤗 We released [Robo-Dopamine-GRM-3B](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-3B) model and inference codes.
- `**2025-12-30**`: ✨ Codes, Dataset and Weights are coming soon! Stay tuned for updates.
- `**2025-12-30**`: 🔥 We released our [Project Page](https://robo-dopamine.github.io/) of **Robo-Dopamine**.

## 🎯 TODO

- Release Robo-Dopamine-GRM-3B model and inference codes.
- Release Robo-Dopamine-Bench benchmark and evaluation codes.
- Release data generation pipeline and finetune codes.
- Release Robo-Dopamine-GRM-8B model.
- Release more powerful and stable Robo-Dopamine-GRM-2.0-8B-Preview model.
- Release a lightweight Robo-Dopamine-GRM-2.0-4B-Preview model.
- Release full GRM dataset and GRM pre-training codes.

## 🤖 Overview

**Robo-Dopamine** is composed of two core components: ***(a) Dopamine-Reward Modeling Method --*** At the heart of our reward modeling is to build the General Reward Model (GRM), a vision-language model that is prompted with a task description and conditioned on multi-view images of initial, goal, "BEFORE," and "AFTER" states to predict a relative progress or regress hop. To ensure a stable and accurate signal, we employ *Multi-Perspective Progress Fusion*, which combines incremental, forward-anchored, and backward-anchored predictions into a final fused reward. And ***(b) Dopamine-RL Training Framework --*** The Dopamine-RL framework first adapts the pre-trained GRM to a novel task using a single demonstration, i.e., *One-Shot GRM Adaptation*. Subsequently, it uses a theoretically-sound *Policy-Invariant Reward Shaping* method to convert the GRM's dense output into a reward signal that accelerates learning without altering the optimal policy. 
This approach is universally compatible with a wide range of RL algorithms.





## 🤗 Model Zoo


| Models                | Checkpoint                                                                                                                 | Description                                                                                                                             |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| GRM-3B                | [🤗 tanhuajie2001/Robo-Dopamine-GRM-3B](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-3B)                         | Full-trained GRM from RoboBrain-2.0-3B                                                                                                  |
| GRM-8B                | [🤗 tanhuajie2001/Robo-Dopamine-GRM-8B](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-8B)                         | Full-trained GRM from RoboBrain-2.0-8B                                                                                                  |
| 🔥 GRM-2.0-4B-Prewiew | [🤗 tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview) | *A Lightweight GRM with ST Modeling, supporting single-view/multi-view cases, both with and without reference target images*            |
| 🔥 GRM-2.0-8B-Prewiew | [🤗 tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview) | *More Powerful and Stable GRM with ST Modeling, supporting single-view/multi-view cases, both with and without reference target images* |


## 🛠️ Setup

```bash
# clone repo.
git clone https://github.com/FlagOpen/Robo-Dopamine.git
cd Robo-Dopamine

# build conda env., and require `cuda >=12.8`
conda create -n robo-dopamine python=3.10
conda activate robo-dopamine
pip install -r requirements.txt
```

## 💡 Simple Usage

The following are simple and practical examples of the three inference modes (Incremental-Mode, Forward-Mode, and Backward-Mode). In practice, to predict the task state reward more accurately, ***we highly recommend averaging the inference reward results from all three modes to use as the final reward in RL***.

```python
import os
from examples.inference import GRMInference

model = GRMInference("tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview")

TASK_INSTRUCTION = "organize the table"
BASE_DEMO_PATH = "./examples/demo_table"
OUTPUT_ROOT = "./results"

## Note: If no reference/goal image is provided, 
## please replace `GOAL_IMAGE_PATH` with the blank image "./examples/blank_goal.png".
GOAL_IMAGE_PATH = "./examples/demo_table/goal_image.png" # "./examples/blank_goal.png"

# select prediction model: Forward-Mode, Incremental-Mode or Backward-Mode
PREDICTION_MODE = "forward" # "incremental" or "backward"

# multi-view usage:
output_dir = model.run_pipeline(
    cam_high_path  = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
    cam_left_path  = os.path.join(BASE_DEMO_PATH, "cam_left_wrist.mp4"),
    cam_right_path = os.path.join(BASE_DEMO_PATH, "cam_right_wrist.mp4"),
    out_root       = OUTPUT_ROOT,
    task           = TASK_INSTRUCTION,
    frame_interval = 10, # modify frame_interval as desired, but it shouldn't be set too small if using 'incremental'.
    batch_size     = 1, # please increase batch_size > 1, if you have enough GPU memory.
    goal_image     = GOAL_IMAGE_PATH,
    eval_mode      = PREDICTION_MODE,
    visualize      = True
)
print(f"Episode ({BASE_DEMO_PATH}) processed with multi-view {PREDICTION_MODE}-mode. Output at: {output_dir}")

# single-view usage:
output_dir = model.run_pipeline(
    cam_high_path  = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
    cam_left_path  = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"), # repeat cam_high
    cam_right_path = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"), # repeat cam_high
    out_root       = OUTPUT_ROOT,
    task           = TASK_INSTRUCTION,
    frame_interval = 10, # modify frame_interval as desired, but it shouldn't be set too small if using 'incremental'.
    batch_size     = 1, # please increase batch_size > 1, if you have enough GPU memory.
    goal_image     = GOAL_IMAGE_PATH,
    eval_mode      = PREDICTION_MODE,
    visualize      = True
)
print(f"Episode ({BASE_DEMO_PATH}) processed with single-view {PREDICTION_MODE}-mode. Output at: {output_dir}")

```

## ✨ More Cases for Testing

Many thanks to [Robometer](https://github.com/robometer/robometer) for providing more interesting test examples 🤗. To better demonstrate the usage of **'single-view without goal image'** with our latest [Robo-Dopamine-GRM-2.0-8B-Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview) model, we also provide the following reference script for your easy tests.

```python
import os
from examples.inference import GRMInference

model = GRMInference("tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview")

## Note: If no target/goal image is provided, 
## please replace `GOAL_IMAGE_PATH` with the blank image!
GOAL_IMAGE_PATH = "./examples/blank_goal.png" 

# select prediction mode: Forward-Mode, Incremental-Mode or Backward-Mode
PREDICTION_MODE =  "forward" # "incremental" or "backward"

OUTPUT_ROOT = "./results"

# 1. open red drawer
output_dir = model.run_pipeline(
    cam_high_path  = "./examples/more_demos/open_red_drawer_wrist.mp4",
    cam_left_path  = "./examples/more_demos/open_red_drawer_wrist.mp4",
    cam_right_path = "./examples/more_demos/open_red_drawer_wrist.mp4",
    out_root       = OUTPUT_ROOT,
    task           = "open red drawer",
    frame_interval = 5, 
    batch_size     = 1, 
    goal_image     = GOAL_IMAGE_PATH,
    eval_mode      = PREDICTION_MODE,
    visualize      = True
)
print(f"Episode processed with single-view {PREDICTION_MODE}-mode. Output at: {output_dir}")

# 2. put marker in cup (fail case)
output_dir = model.run_pipeline(
    cam_high_path  = "./examples/more_demos/put_marker_in_cup_fail.mp4",
    cam_left_path  = "./examples/more_demos/put_marker_in_cup_fail.mp4",
    cam_right_path = "./examples/more_demos/put_marker_in_cup_fail.mp4",
    out_root       = OUTPUT_ROOT,
    task           = "put marker in cup",
    frame_interval = 5, 
    batch_size     = 1, 
    goal_image     = GOAL_IMAGE_PATH,
    eval_mode      = PREDICTION_MODE,
    visualize      = True
)
print(f"Episode processed with single-view {PREDICTION_MODE}-mode. Output at: {output_dir}")

# 3. push green block in green bowl
output_dir = model.run_pipeline(
    cam_high_path  = "./examples/more_demos/push_green_block_in_green_bowl.mp4",
    cam_left_path  = "./examples/more_demos/push_green_block_in_green_bowl.mp4",
    cam_right_path = "./examples/more_demos/push_green_block_in_green_bowl.mp4",
    out_root       = OUTPUT_ROOT,
    task           = "push green block in green bowl",
    frame_interval = 5, 
    batch_size     = 1, 
    goal_image     = GOAL_IMAGE_PATH,
    eval_mode      = PREDICTION_MODE,
    visualize      = True
)
print(f"Episode processed with single-view {PREDICTION_MODE}-mode. Output at: {output_dir}")

# 4. put apple in tray
output_dir = model.run_pipeline(
    cam_high_path  = "./examples/more_demos/put_apple_in_tray.mp4",
    cam_left_path  = "./examples/more_demos/put_apple_in_tray.mp4",
    cam_right_path = "./examples/more_demos/put_apple_in_tray.mp4",
    out_root       = OUTPUT_ROOT,
    task           = "put apple in tray",
    frame_interval = 5, 
    batch_size     = 1, 
    goal_image     = GOAL_IMAGE_PATH,
    eval_mode      = PREDICTION_MODE,
    visualize      = True
)
print(f"Episode processed with single-view {PREDICTION_MODE}-mode. Output at: {output_dir}")

```

We have attached the visualization results from these tests below. ***Please feel free to open an issue if you have any questions when testing the provided examples or your own test cases.*** 🤗 🤗 🤗

[Demo Result 1](https://github.com/user-attachments/assets/23474262-e61a-4f86-b2e5-c3b426292536) | [Demo Result 2](https://github.com/user-attachments/assets/85adb149-0589-4237-907a-faedc9034058) | [Demo Result 3](https://github.com/user-attachments/assets/3e1db271-688e-4e2b-9847-076e0d358f70) | [Demo Result 4](https://github.com/user-attachments/assets/46588f92-6004-4aec-91c6-a0775a0890d2)

## 🔍 Evaluation

### 0. Download `Robo-Dopamine-Bench` from huggingface.

```bash
# download benchmark
huggingface-cli download --repo-type dataset --resume-download tanhuajie2001/Robo-Dopamine-Bench --local-dir ./Robo-Dopamine-Bench

# unzip images
cd Robo-Dopamine-Bench
unzip image.zip
cd ..
```

### 1. Evaluate local GRM with vLLM.

```bash
# GRM-3B
export CUDA_VISIBLE_DEVICES=0 
python -m eval.evaluation_grm \
  --model_path tanhuajie2001/Robo-Dopamine-GRM-3B \
  --input_json_dir ./Robo-Dopamine-Bench/jsons \
  --base_dir ./Robo-Dopamine-Bench/images \
  --out_root_dir ./eval_results/results_Robo-Dopamine-GRM-3B \
  --batch_size 16

# GRM-8B
export CUDA_VISIBLE_DEVICES=0 
python -m eval.evaluation_grm \
  --model_path tanhuajie2001/Robo-Dopamine-GRM-8B \
  --input_json_dir ./Robo-Dopamine-Bench/jsons \
  --base_dir ./Robo-Dopamine-Bench/images \
  --out_root_dir ./eval_results/results_Robo-Dopamine-GRM-8B \
  --batch_size 16
```

### 2. Evaluate other models with API.

```bash
python -m eval.evaluation_api \
  --model_name <MODEL-NAME, e.g., gpt-4o, gemini-3-pro> \
  --api_key <OPENAI-API-KEY> \
  --base_url <OPENAI-BASE-URL> \
  --input_json_dir ./Robo-Dopamine-Bench/jsons \
  --base_dir ./Robo-Dopamine-Bench/images \
  --out_root_dir ./eval_results/results_{MODEL-NAME} \
  --max_workers 16
```

***EVALUATION RESULTS***



## ⚡ Fine-Tuning

### Step 1. Reconstruct Your Own Dataset

***Raw Data Directory Structure***: The `[dataset/example_raw_data](https://github.com/FlagOpen/Robo-Dopamine/tree/main/dataset/example_raw_data)` directory serves as **an EXAMPLE** to demonstrate the required structure for your own raw data, ensuring compatibility with our provided data processing scripts.

```
example_raw_data/
├── episode_001/
│   ├── annotated_keyframes.json   # Keyframe annotations for subtask segmentation
│   ├── cam_high.mp4               # Video from the high-mounted camera
│   ├── cam_left_wrist.mp4         # Video from the left wrist-mounted camera
│   └── cam_right_wrist.mp4        # Video from the right wrist-mounted camera
├── episode_002/
│   ├── annotated_keyframes.json
│   ├── cam_high.mp4
│   ├── cam_left_wrist.mp4
│   └── cam_right_wrist.mp4
├── episode_003/
│   ├── annotated_keyframes.json
│   ├── cam_high.mp4
│   ├── cam_left_wrist.mp4
│   └── cam_right_wrist.mp4
├── ...
├── episode_xxx/                   # Generalized episode directory (xxx = episode number)
│   ├── annotated_keyframes.json
│   ├── cam_high.mp4
│   ├── cam_left_wrist.mp4
│   └── cam_right_wrist.mp4
└── task_instruction.json          # Natural language task instructions (shared across all episodes)
```

### Step 2. Process Your Own Dataset

Here, we use `[dataset/example_raw_data](https://github.com/FlagOpen/Robo-Dopamine/tree/main/dataset/example_raw_data)` as an example.

```bash
cd dataset

# first, pre-process the raw data with sample_factor
python -m utils.0_preprocess_data \
  --raw_dir ./example_raw_data \
  --cvt_dir ./train_data \
  --sample_factor 20

# then, generate training data with bin-sampling strategy
python -m utils.1_generate_data \
  --base-dir ./train_data \
  --score-bins 25 \
  --gap-bins 4 \
  --oversample-factor 100 \
  --zero-ratio 0.05 \
  --max_sample_num 1000

# finally, post-process the sampled data for fine-tuning
python -m utils.2_posprocess_data \
  --root-dir ./train_data \
  --merged-json ./train_data/train_jsons/finetune_data_wo_replace.json \
  --final-json ./train_data/train_jsons/finetune_data_final.json \
  --replace-prob 0.75

```

### Step 3. Fine-Tune GRM with Your Own Dataset

**Add the meta-info of your own dataset to `train/qwenvl/data/__init__.py`**

```python
# modified here
EXAMPLE_GRM_FINETUNE = {
    "annotation_path": "./dataset/train_data/train_jsons/finetune_data_final.json",
    "data_path": "./dataset",
}

# modified here
data_dict = {
    "example_grm_finetune": EXAMPLE_GRM_FINETUNE,
}
```

**Modify the path of training script `train/scripts/finetune_grm.sh`**

```python
# ======================
# Path Configuration
# ======================
MODEL_PATH="tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview" # modified here
OUTPUT_DIR="./checkpoints/example_grm_finetune"             # modified here
DATASETS=example_grm_finetune                               # modified here
```

**Launch the training script**

*Before you launch training, make sure that you have installed the packages below:*

```
pip install flash-attn --no-build-isolation
pip install deepspeed
pip install opencv-python-headless<4.12
pip install numpy<2.0
```

*Then, run the script below:*

```bash
cd ../train
bash scripts/finetune_grm.sh
```

## 🐛 Fine-Tuning Troubleshooting

The following are common issues and solutions encountered during fine-tuning.

### 1. `CUDA_HOME does not exist` — DeepSpeed Cannot Find CUDA

**Error:**
```
deepspeed.ops.op_builder.builder.MissingCUDAException: CUDA_HOME does not exist, unable to compile CUDA op(s)
```

**Cause:** DeepSpeed requires `CUDA_HOME` environment variable and `nvcc` to check CUDA compatibility at import time. If your system doesn't have a system-level CUDA toolkit installed (only PyTorch's bundled CUDA), this error occurs.

**Solution:** Create a fake `nvcc` that returns the correct version, and set `CUDA_HOME`:

```bash
# 1. Create fake nvcc (replace <CONDA_ENV_PATH> with your conda env path)
FAKE_CUDA_DIR="<CONDA_ENV_PATH>/lib/python3.10/site-packages/nvidia"
mkdir -p $FAKE_CUDA_DIR/bin

cat > $FAKE_CUDA_DIR/bin/nvcc << 'EOF'
#!/bin/bash
if [[ "$1" == "-V" || "$1" == "--version" ]]; then
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "Cuda compilation tools, release 12.8, V12.8.93"
else
    echo "Fake nvcc for DeepSpeed compatibility" >&2
    exit 1
fi
EOF
chmod +x $FAKE_CUDA_DIR/bin/nvcc

# 2. Add to your training script:
export CUDA_HOME=$FAKE_CUDA_DIR
```

### 2. `do not find <dataset_name>` — Dataset Not Registered

**Error:**
```
ValueError: do not find ../dataset/my_train_data
```

**Cause:** The `--dataset_use` parameter expects a **key name** registered in `train/qwenvl/data/__init__.py`'s `data_dict`, not a file path.

**Solution:** Use the registered key name in your training script:
```bash
# Wrong:
DATASETS=../dataset/my_train_data

# Correct (must match the key in data_dict):
DATASETS=my_carrot_dataset
```

### 3. Image Path Mismatch — Symbolic Link Required

**Error:**
```
ValueError: Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image file, or a base64 encoded string. Got .../train_data/episode_001/cam_high/frame_000000.jpg. Failed with Incorrect padding
```

**Cause:** The data generation script (`utils/1_generate_data.py`) hardcodes the image path prefix as `train_data/` in `build_images_rel()`. If you used a different `--cvt_dir` name (e.g., `my_train_data`), the JSON paths won't match the actual directory.

**Solution (choose one):**

Option A — Use the default `--cvt_dir train_data` when running `0_preprocess_data.py`:
```bash
python -m utils.0_preprocess_data --raw_dir ./my_raw_data --cvt_dir ./train_data
```

Option B — Create a symbolic link to bridge the path:
```bash
cd dataset
ln -sfn my_train_data train_data
```

Option C — Fix the JSON paths with `sed`:
```bash
sed -i 's|train_data/|my_train_data/|g' ./my_train_data/train_jsons/finetune_data_final.json
```

### 4. CUDA Out of Memory (OOM)

**Error:**
```
CUDA out of memory. Tried to allocate 10.00 MiB. GPU 0 has a total capacity of 79.25 GiB of which 9.88 MiB is free.
```

**Key parameters to reduce memory usage:**

| Parameter | Default | Description | Reduction Strategy |
|-----------|---------|-------------|-------------------|
| `model_max_length` | 32768 | Maximum sequence length (tokens). Each sample has 8 images + prompt, typically ~4K tokens | Lower to `8192` — sufficient for 8-image samples |
| `max_pixels` | 76800 | Maximum pixels per image (~277x277). Affects image token count and attention memory | Lower to `40401` (~201x201) or `22500` (~150x150) |
| `video_max_frame_pixels` | 1304576 | Maximum pixels per video frame (~1142x1142) | Lower to `262144` (~512x512) |
| `per_device_train_batch_size` | 2 | Batch size per GPU | Lower to `1` |
| `gradient_accumulation_steps` | 4 | Accumulate gradients over N steps to simulate larger batch | Increase to compensate for smaller batch size |

**Recommended configurations by hardware:**

| Hardware | `nproc_per_node` | `per_device_train_batch_size` | `gradient_accumulation_steps` | `model_max_length` | `max_pixels` | `deepspeed` |
|----------|-------------------|-------------------------------|-------------------------------|--------------------|-------------|-------------|
| 8×A100-80G | 8 | 2 | 2 | 32768 | 76800 | zero3.json |
| 4×A100-80G | 4 | 1 | 4 | 8192 | 76800 | zero3.json |
| 2×A100-80G | 2 | 1 | 8 | 8192 | 40401 | zero3.json |
| 1×A100-80G | 1 | 1 | 16 | 8192 | 22500 | zero3_offload.json |

**Formula:** Effective Batch Size = `nproc_per_node` × `per_device_train_batch_size` × `gradient_accumulation_steps` ≈ 32

If still OOM with 2 GPUs, consider using the **4B model** (`Robo-Dopamine-GRM-2.0-4B-Preview`) which requires ~half the memory.

### 5. wandb Permission Warning in Kubernetes

**Warning:**
```
wandb: WARNING Unable to read the token file at /var/run/secrets/kubernetes.io/serviceaccount/token due to permission error
```

**Solution:** Configure wandb environment variables in your training script:

```bash
# Option A: Disable wandb (no logging)
export WANDB_MODE=disabled
export WANDB_DIR=/tmp/wandb

# Option B: Enable wandb logging
export WANDB_MODE=online
export WANDB_DIR=/tmp/wandb
export WANDB_API_KEY=your_api_key_here
export WANDB_ENTITY=your_username
export WANDB_PROJECT=robo-dopamine
```

Get your API key from: https://wandb.ai/authorize

### 6. GPU Selection

Control which GPUs to use with `CUDA_VISIBLE_DEVICES` and `--nproc_per_node`:

```bash
# Use GPU 0 and 1
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 ...

# Use GPU 2 and 3
export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 ...

# Use all 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 ...
```

**Important:** `--nproc_per_node` must equal the number of GPUs visible to the program (i.e., the number of GPUs in `CUDA_VISIBLE_DEVICES`).


## 📑 Citation

If you find our work helpful, feel free to cite it:

```
@article{tan2025robo,
  title={Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation},
  author={Tan, Huajie and Chen, Sixiang and Xu, Yijie and Wang, Zixiao and Ji, Yuheng and Chi, Cheng and Lyu, Yaoxu and Zhao, Zhongxia and Chen, Xiansheng and Co, Peterson and others},
  journal={arXiv preprint arXiv:2512.23703},
  year={2025}
}
```

