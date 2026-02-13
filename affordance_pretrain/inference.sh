#!/bin/bash

# Inference with 3D output (.ply files)
# Run from affordance_pretrain directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CHECKPOINT="${SCRIPT_DIR}/checkpoints/epoch_089.pth"
DAMON_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/DAMON"
PIAD_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/PIAD"
SMPLX_PATH="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/smpl_models/SMPLX_NEUTRAL.npz"
OUTPUT_DIR="${SCRIPT_DIR}/vis_results"

python "${SCRIPT_DIR}/inference.py" \
    --checkpoint "${CHECKPOINT}" \
    --damon_root "${DAMON_ROOT}" \
    --piad_root "${PIAD_ROOT}" \
    --piad_setting Seen \
    --smplx_path "${SMPLX_PATH}" \
    --num_samples 20 \
    --output_dir "${OUTPUT_DIR}" \
    --dataset both
