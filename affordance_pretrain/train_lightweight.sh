#!/bin/bash

# Affordance Pre-training with Lightweight VLM (CLIP)
# Use this if you don't have enough GPU memory for LLaVA (~20GB+)
# CLIP-based VLM uses only ~4GB VRAM

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Dataset paths
DAMON_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/DAMON"
PIAD_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/PIAD"

# Training settings
BATCH_SIZE=16         # Can use larger batch with CLIP
EPOCHS=50
NUM_WORKERS=4
SAVE_DIR="${SCRIPT_DIR}/checkpoints_lightweight"

# wandb settings
WANDB_PROJECT="affordance-pretrain"
RUN_NAME="clip_bs${BATCH_SIZE}_ep${EPOCHS}"

# Create save directory
mkdir -p "${SAVE_DIR}"

# Create a temporary config with lightweight VLM
CONFIG_TMP="${SCRIPT_DIR}/configs/config_lightweight.yaml"
cat > "${CONFIG_TMP}" << 'EOF'
# Lightweight Affordance Pre-training Configuration (CLIP-based)

model:
  d_tr: 256
  num_human_points: 10475
  use_lightweight_vlm: true   # Use CLIP instead of LLaVA
  vlm_model_name: "openai/clip-vit-base-patch32"
  freeze_vlm: true
  dropout: 0.3

data:
  image_size: 224
  num_object_points: 2048
  num_human_points: 10475

training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 5
  lambda_human: 1.0
  lambda_object: 1.0
  use_focal: true
  use_dice: true
  focal_alpha: 0.25
  focal_gamma: 2.0
  lambda_dice: 1.0
  lambda_focal: 1.0
  lr_vlm_scale: 0.1
  lr_encoder_scale: 0.5
  lr_decoder_scale: 1.0

logging:
  log_interval: 20
  save_interval: 5
EOF

# Run training
cd "${PROJECT_ROOT}"

python affordance_pretrain/train.py \
    --config "${CONFIG_TMP}" \
    --damon_root "${DAMON_ROOT}" \
    --piad_root "${PIAD_ROOT}" \
    --piad_setting Seen \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --save_dir "${SAVE_DIR}" \
    --wandb_project "${WANDB_PROJECT}" \
    --run_name "${RUN_NAME}"
