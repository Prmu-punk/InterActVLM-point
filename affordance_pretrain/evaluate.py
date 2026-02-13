"""
Evaluation script for Affordance Pre-training
Human: F1, Precision, Recall, Accuracy, Geodesic Distance
Object: SIM, MAE, AUC, IOU
"""

import argparse
import os
import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from affordance_pretrain.aff_models.unified_aff_model import build_affordance_model
from affordance_pretrain.data.piad_dataset import PIADDataset, PIADTransform
from affordance_pretrain.data.damon_dataset import DAMONDataset, DAMONTransform


# ============ Geodesic Distance Matrix ============

# Priority: SMPLX (10475 vertices) > SMPL (6890 vertices)
GEODESIC_DIST_MATRIX = None
_geodesic_paths = [
    # SMPLX geodesic (10475 vertices) - preferred for our model
    '/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/data/smplx_geodesic_dist.npy',
    '/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/affordance_pretrain/data/smplx_geodesic_dist.npy',
    './data/smplx_geodesic_dist.npy',
    # SMPL geodesic (6890 vertices) - fallback but may not match
    '/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/affordance_pretrain/InteractVLM/data/smpl_neutral_geodesic_dist.npy',
    '/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/data/smpl_neutral_geodesic_dist.npy',
    './data/smpl_neutral_geodesic_dist.npy',
]
for _gpath in _geodesic_paths:
    if os.path.exists(_gpath):
        try:
            GEODESIC_DIST_MATRIX = np.load(_gpath)
            print(f"Loaded geodesic distance matrix from: {_gpath}")
            break
        except Exception as e:
            print(f"Failed to load geodesic matrix from {_gpath}: {e}")

if GEODESIC_DIST_MATRIX is None:
    print("Warning: Geodesic distance matrix not found. Will use Euclidean approximation.")


# ============ Metrics ============

def compute_f1_precision_recall(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5):
    """Compute F1, Precision, Recall for binary classification."""
    pred_binary = (pred >= threshold).astype(float)
    gt_binary = (gt > 0).astype(float)

    tp = (pred_binary * gt_binary).sum()
    pred_pos = pred_binary.sum()
    gt_pos = gt_binary.sum()

    precision = tp / (pred_pos + 1e-10)
    recall = tp / (gt_pos + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (pred_binary == gt_binary).mean()

    return f1, precision, recall, accuracy


def compute_geodesic_error(pred: np.ndarray, gt: np.ndarray, vertices: np.ndarray = None, threshold: float = 0.5):
    """
    Compute geodesic error for human contact prediction.
    Following InteractVLM/utils/eval_utils.py get_h_geo_metric method.

    If geodesic distance matrix is available, use it.
    Otherwise, fall back to Euclidean distance approximation.

    Returns:
        fp_dist: Mean geodesic/Euclidean distance of predictions to nearest GT contact
    """
    pred_binary = (pred >= threshold).astype(float)
    gt_binary = (gt > 0).astype(float)

    if GEODESIC_DIST_MATRIX is not None:
        # Use precomputed geodesic distance matrix (same as eval_utils.py)
        dist_matrix = GEODESIC_DIST_MATRIX

        # Get columns corresponding to GT contact points
        gt_contact_mask = gt_binary == 1
        if gt_contact_mask.any():
            gt_columns = dist_matrix[:, gt_contact_mask]
        else:
            return 0.0

        # Get rows corresponding to predicted contact points
        pred_contact_mask = pred_binary >= threshold
        if pred_contact_mask.any():
            error_matrix = gt_columns[pred_contact_mask, :]
        else:
            return 0.0

        # False positive distance: for each predicted point, min distance to any GT point
        if error_matrix.size > 0:
            fp_dist = error_matrix.min(axis=1).mean()
            return fp_dist
        return 0.0

    else:
        # Fallback: Euclidean distance approximation using SMPLX vertices
        if vertices is None:
            return 0.0

        pred_contact_idx = np.where(pred_binary)[0]
        gt_contact_idx = np.where(gt_binary)[0]

        if len(pred_contact_idx) == 0 or len(gt_contact_idx) == 0:
            return 0.0

        # Get vertex positions
        pred_vertices = vertices[pred_contact_idx]  # (N_pred, 3)
        gt_vertices = vertices[gt_contact_idx]  # (N_gt, 3)

        # Compute pairwise distances (Euclidean approximation)
        distances = []
        for pv in pred_vertices:
            dist = np.sqrt(((gt_vertices - pv) ** 2).sum(axis=1))
            distances.append(dist.min())

        return np.mean(distances) if distances else 0.0


def compute_sim(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12):
    """Compute Similarity metric."""
    pred_norm = pred / (pred.sum() + eps)
    gt_norm = gt / (gt.sum() + eps)
    intersection = np.minimum(pred_norm, gt_norm)
    return intersection.sum()


def compute_mae(pred: np.ndarray, gt: np.ndarray):
    """Compute Mean Absolute Error."""
    return np.abs(pred - gt).mean()


def compute_auc(pred: np.ndarray, gt: np.ndarray):
    """Compute Area Under ROC Curve."""
    gt_binary = (gt >= 0.5).astype(int)
    if len(np.unique(gt_binary)) < 2:
        return float('nan')
    try:
        return roc_auc_score(gt_binary, pred)
    except ValueError:
        return float('nan')


def compute_iou(pred: np.ndarray, gt: np.ndarray, thresholds: np.ndarray = None):
    """Compute average IoU across multiple thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 20)

    gt_binary = (gt >= 0.5).astype(int)
    ious = []

    for thresh in thresholds:
        pred_binary = (pred >= thresh).astype(int)
        intersection = (pred_binary & gt_binary).sum()
        union = (pred_binary | gt_binary).sum()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(0.0)

    return np.mean(ious)


# ============ Model Loading ============

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    model_config['device'] = device

    model = build_affordance_model(model_config).to(device)

    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items()
                      if k in model_state and model_state[k].shape == v.shape}

    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
    return model


def load_smplx_template(smplx_path: str):
    """Load SMPLX template vertices."""
    data = np.load(smplx_path, allow_pickle=True)
    return data['v_template']


# ============ Evaluation Functions ============

@torch.no_grad()
def evaluate_damon(model, damon_root, smplx_path, device, num_samples=None):
    """Evaluate on DAMON dataset (Human Contact)."""
    print(f"\n{'='*60}")
    print("Evaluating DAMON (Human Contact)")
    print(f"{'='*60}")

    # Load SMPLX vertices for geodesic calculation
    smplx_vertices = load_smplx_template(smplx_path)

    transform = DAMONTransform(image_size=224)
    dataset = DAMONDataset(
        damon_root=damon_root, split='val',
        transform=transform, num_human_points=10475
    )

    if num_samples:
        indices = np.linspace(0, len(dataset) - 1, min(num_samples, len(dataset)), dtype=int)
    else:
        indices = range(len(dataset))

    # Metrics accumulators
    f1_list, precision_list, recall_list, acc_list = [], [], [], []
    geo_list = []

    for idx in tqdm(indices, desc="DAMON"):
        sample = dataset[idx]
        rgb_image = sample['rgb_image'].unsqueeze(0).to(device)

        outputs = model(rgb_image=rgb_image, object_points=None,
                        compute_human=True, compute_object=False)

        pred = outputs['human_affordance'].squeeze(0).cpu().numpy()
        gt = sample['human_mask_gt'].numpy()

        # Compute metrics
        f1, precision, recall, acc = compute_f1_precision_recall(pred, gt)
        geo = compute_geodesic_error(pred, gt, smplx_vertices)

        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        acc_list.append(acc)
        geo_list.append(geo)

    # Aggregate results
    results = {
        'F1': np.mean(f1_list),
        'Precision': np.mean(precision_list),
        'Recall': np.mean(recall_list),
        'Accuracy': np.mean(acc_list),
        'Geodesic': np.mean(geo_list),
    }

    print(f"\n[DAMON Results]")
    print(f"  F1:        {results['F1']:.4f}")
    print(f"  Precision: {results['Precision']:.4f}")
    print(f"  Recall:    {results['Recall']:.4f}")
    print(f"  Accuracy:  {results['Accuracy']:.4f}")
    print(f"  Geodesic:  {results['Geodesic']:.4f}")

    return results


@torch.no_grad()
def evaluate_piad(model, piad_root, setting, device, num_samples=None):
    """Evaluate on PIAD dataset (Object Affordance)."""
    print(f"\n{'='*60}")
    print(f"Evaluating PIAD {setting} (Object Affordance)")
    print(f"{'='*60}")

    transform = PIADTransform(image_size=224)
    dataset = PIADDataset(
        piad_root=piad_root, setting=setting, split='test',
        transform=transform, num_object_points=2048
    )

    if num_samples:
        indices = np.linspace(0, len(dataset) - 1, min(num_samples, len(dataset)), dtype=int)
    else:
        indices = range(len(dataset))

    # Metrics accumulators
    sim_list, mae_list, auc_list, iou_list = [], [], [], []

    for idx in tqdm(indices, desc="PIAD"):
        sample = dataset[idx]
        rgb_image = sample['rgb_image'].unsqueeze(0).to(device)
        object_points = sample['object_points'].unsqueeze(0).to(device)

        outputs = model(rgb_image=rgb_image, object_points=object_points,
                        compute_human=False, compute_object=True)

        pred = outputs['object_affordance'].squeeze(0).cpu().numpy()
        gt = sample['object_mask_gt'].numpy()

        # Compute metrics
        sim = compute_sim(pred, gt)
        mae = compute_mae(pred, gt)
        auc = compute_auc(pred, gt)
        iou = compute_iou(pred, gt)

        sim_list.append(sim)
        mae_list.append(mae)
        if not np.isnan(auc):
            auc_list.append(auc)
        iou_list.append(iou)

    # Aggregate results
    results = {
        'SIM': np.mean(sim_list),
        'MAE': np.mean(mae_list),
        'AUC': np.mean(auc_list) if auc_list else 0.0,
        'IOU': np.mean(iou_list),
    }

    print(f"\n[PIAD {setting} Results]")
    print(f"  SIM: {results['SIM']:.4f}")
    print(f"  MAE: {results['MAE']:.4f}")
    print(f"  AUC: {results['AUC']:.4f}")
    print(f"  IOU: {results['IOU']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--damon_root', type=str, default=None)
    parser.add_argument('--piad_root', type=str, default=None)
    parser.add_argument('--piad_setting', type=str, default='Seen')
    parser.add_argument('--smplx_path', type=str,
                        default='/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/smpl_models/SMPLX_NEUTRAL.npz')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None=all)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['piad', 'damon', 'both'])
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    model = load_model(args.checkpoint, args.device)

    all_results = {}

    if args.dataset in ['damon', 'both'] and args.damon_root:
        results = evaluate_damon(
            model, args.damon_root, args.smplx_path,
            args.device, args.num_samples
        )
        all_results['damon'] = results

    if args.dataset in ['piad', 'both'] and args.piad_root:
        results = evaluate_piad(
            model, args.piad_root, args.piad_setting,
            args.device, args.num_samples
        )
        all_results['piad'] = results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if 'damon' in all_results:
        r = all_results['damon']
        print(f"\nDAMON (Human Contact):")
        print(f"  F1={r['F1']:.4f}, Prec={r['Precision']:.4f}, "
              f"Recall={r['Recall']:.4f}, Acc={r['Accuracy']:.4f}, Geo={r['Geodesic']:.4f}")

    if 'piad' in all_results:
        r = all_results['piad']
        print(f"\nPIAD (Object Affordance):")
        print(f"  SIM={r['SIM']:.4f}, MAE={r['MAE']:.4f}, "
              f"AUC={r['AUC']:.4f}, IOU={r['IOU']:.4f}")


if __name__ == '__main__':
    main()
