#!/bin/bash
export PYTHONPATH=$(pwd)
export WANDB_MODE=offline
# export WANDB_MODE=online
export WANDB_DIR=/tmp/wandb
# export WANDB_API_KEY="your_wandb_api_key"
# export WANDB_ENTITY='test180'
# export WANDB_PROJECT="robo-dopamine"
# export http_proxy=http://127.0.0.1:7898 
# export https_proxy=http://127.0.0.1:7898
export CUDA_HOME=/mnt/public1/dais/miniconda3/envs/robo-dopamine-train/lib/python3.10/site-packages/nvidia
# export WANDB_ENTITY=你的用户名
# ======================
# Path Configuration
# ======================
MODEL_PATH="../pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview"
OUTPUT_DIR="./checkpoints/my_carrot_finetune_big"
DATASETS=my_carrot_dataset

CACHE_DIR="./.cache"
mkdir -p $OUTPUT_DIR

# ======================
# Training Hyperparameters (4x A100-80G)
# ======================
# export CUDA_VISIBLE_DEVICES=2,3

MODEL_MAX_LENGTH=32768 #32768
MAX_PIXELS=76800 #76800
VIDEO_MAX_FRAME_PIXELS=1304576 #1304576

torchrun --nproc_per_node=4 --nnodes=1 --master_port=29515 \
    qwenvl/train/train_qwen.py \
    --model_name_or_path $MODEL_PATH \
    --tune_mm_llm True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --dataset_use $DATASETS \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 5e-7 \
    --optim adamw_torch \
    --model_max_length $MODEL_MAX_LENGTH \
    --data_flatten False \
    --data_packing False \
    --max_pixels $MAX_PIXELS \
    --min_pixels 12544 \
    --base_interval 2 \
    --video_max_frames 8 \
    --video_min_frames 4 \
    --video_max_frame_pixels $VIDEO_MAX_FRAME_PIXELS \
    --video_min_frame_pixels 200704 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 3 \
    --eval_strategy "no" \
    --deepspeed ./scripts/zero3.json \
2>&1 | tee ${OUTPUT_DIR}/train.log
