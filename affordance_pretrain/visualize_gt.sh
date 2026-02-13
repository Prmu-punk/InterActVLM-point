#!/bin/bash

# Visualize DAMON and PIAD GT data in 3D
# Saves .ply files with vertex colors

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

DAMON_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/DAMON"
PIAD_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/PIAD"
SMPLX_PATH="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/smpl_models/SMPLX_NEUTRAL.npz"

echo "=== Visualizing DAMON GT (Human Contact) ==="
python "${SCRIPT_DIR}/visualize_damon_gt.py" \
    --damon_root "${DAMON_ROOT}" \
    --smplx_path "${SMPLX_PATH}" \
    --split train \
    --num_samples 10 \
    --output_dir "${SCRIPT_DIR}/vis_results/damon_gt_3d" \
    --format ply

echo ""
echo "=== Visualizing PIAD GT (Object Affordance) ==="
python "${SCRIPT_DIR}/visualize_piad_gt.py" \
    --piad_root "${PIAD_ROOT}" \
    --setting Seen \
    --split train \
    --num_samples 10 \
    --output_dir "${SCRIPT_DIR}/vis_results/piad_gt_3d"
