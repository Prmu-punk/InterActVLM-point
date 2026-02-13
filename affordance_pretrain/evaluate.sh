#!/bin/bash

# Evaluate affordance model with proper metrics
# Human: F1, Precision, Recall, Accuracy, Geodesic
# Object: SIM, MAE, AUC, IOU

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CHECKPOINT="${SCRIPT_DIR}/checkpoints/best.pth"
DAMON_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/DAMON"
PIAD_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/PIAD"
SMPLX_PATH="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/smpl_models/SMPLX_NEUTRAL.npz"

python "${SCRIPT_DIR}/evaluate.py" \
    --checkpoint "${CHECKPOINT}" \
    --damon_root "${DAMON_ROOT}" \
    --piad_root "${PIAD_ROOT}" \
    --piad_setting Seen \
    --smplx_path "${SMPLX_PATH}" \
    --dataset both
