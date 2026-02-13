#!/bin/bash

# Affordance Pre-training with LoRA
# This script trains the affordance model with LLaVA + LoRA

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Dataset paths
DAMON_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/DAMON"
PIAD_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/PIAD"

# Training settings
BATCH_SIZE=8          # Reduced for LLaVA (uses ~20GB+ VRAM)
EPOCHS=200
NUM_WORKERS=4
SAVE_DIR="${SCRIPT_DIR}/checkpoints"

# wandb settings
WANDB_PROJECT="affordance-pretrain-lora"
RUN_NAME="lora_r16_bs${BATCH_SIZE}_ep${EPOCHS}"

# Create save directory
mkdir -p "${SAVE_DIR}"

# Run training
cd "${PROJECT_ROOT}"

python affordance_pretrain/train.py \
    --config affordance_pretrain/configs/config.yaml \
    --damon_root "${DAMON_ROOT}" \
    --piad_root "${PIAD_ROOT}" \
    --piad_setting Seen \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --save_dir "${SAVE_DIR}" \
    --wandb_project "${WANDB_PROJECT}" \
    --run_name "${RUN_NAME}"
