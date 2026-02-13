#!/bin/bash

# Visualize DAMON GT contact labels in 3D
# Run from affordance_pretrain directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

DAMON_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/DAMON"
SMPLX_PATH="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/smpl_models/SMPLX_NEUTRAL.npz"
OUTPUT_DIR="${SCRIPT_DIR}/vis_results/damon_gt_3d"

python "${SCRIPT_DIR}/visualize_damon_gt.py" \
    --damon_root "${DAMON_ROOT}" \
    --smplx_path "${SMPLX_PATH}" \
    --split train \
    --num_samples 20 \
    --output_dir "${OUTPUT_DIR}" \
    --start_idx 0
