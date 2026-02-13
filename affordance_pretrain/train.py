"""
Training script for Unified Affordance Pre-training
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path FIRST (before any other imports)
# Insert at position 0 to ensure root models/ is found before affordance_pretrain/models/
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm import tqdm
import yaml
from sklearn.metrics import roc_auc_score

# Import from affordance_pretrain's local models using relative import style
from affordance_pretrain.aff_models.unified_aff_model import UnifiedAffordanceModel, AffordanceLoss, build_affordance_model
from affordance_pretrain.data.unified_aff_dataset import create_affordance_dataloaders
from affordance_pretrain.data.damon_dataset import create_damon_dataloaders
from affordance_pretrain.data.piad_dataset import create_piad_dataloaders

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============ Metrics Functions ============

# Try to load geodesic distance matrix for human metrics
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
            GEODESIC_DIST_MATRIX = torch.tensor(np.load(_gpath))
            print(f"Loaded geodesic distance matrix from: {_gpath}")
            break
        except Exception as e:
            print(f"Failed to load geodesic matrix from {_gpath}: {e}")

if GEODESIC_DIST_MATRIX is None:
    print("Warning: Geodesic distance matrix not found. Will use Euclidean approximation.")


def compute_geodesic_error(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5):
    """
    Compute geodesic error for human contact prediction.
    Following the same method as InteractVLM/utils/eval_utils.py get_h_geo_metric.

    Args:
        pred: (B, N) predicted probabilities
        gt: (B, N) ground truth binary labels
        threshold: classification threshold

    Returns:
        fp_dist_avg: Mean geodesic distance of false positives to nearest GT contact
    """
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()

    batch_size = pred.shape[0]
    fp_dist_list = []

    if GEODESIC_DIST_MATRIX is not None:
        # Use precomputed geodesic distance matrix
        dist_matrix = GEODESIC_DIST_MATRIX
        for b in range(batch_size):
            gt_b = (gt[b] > 0).float()
            pred_b = (pred[b] >= threshold).float()

            # Get columns corresponding to GT contact points
            gt_contact_mask = gt_b == 1
            if gt_contact_mask.any():
                gt_columns = dist_matrix[:, gt_contact_mask]
            else:
                continue

            # Get rows corresponding to predicted contact points
            pred_contact_mask = pred_b >= threshold
            if pred_contact_mask.any():
                error_matrix = gt_columns[pred_contact_mask, :]
            else:
                continue

            # False positive distance: for each predicted point, min distance to any GT point
            if error_matrix.numel() > 0:
                fp_dist = error_matrix.min(dim=1)[0].mean().item()
                fp_dist_list.append(fp_dist)
    else:
        # Fallback: use simple ratio of false positives (no geodesic available)
        for b in range(batch_size):
            pred_binary = (pred[b] >= threshold).float()
            gt_binary = (gt[b] > 0).float()

            # Count false positives as a simple metric
            fp = ((pred_binary == 1) & (gt_binary == 0)).float().sum()
            total_pred = pred_binary.sum()
            if total_pred > 0:
                fp_ratio = (fp / total_pred).item()
                fp_dist_list.append(fp_ratio)

    return np.mean(fp_dist_list) if fp_dist_list else 0.0


def compute_human_metrics(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5):
    """
    Compute human contact metrics: F1, Precision, Recall, Accuracy, Geodesic.
    Following InteractVLM/utils/eval_utils.py methods.

    Args:
        pred: (B, N) predicted probabilities
        gt: (B, N) ground truth binary labels
        threshold: classification threshold

    Returns:
        dict with f1, precision, recall, accuracy, geodesic
    """
    pred_binary = (pred >= threshold).float()
    gt_binary = (gt > 0).float()

    tp = (pred_binary * gt_binary).sum()
    pred_pos = pred_binary.sum()
    gt_pos = gt_binary.sum()

    precision = tp / (pred_pos + 1e-10)
    recall = tp / (gt_pos + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (pred_binary == gt_binary).float().mean()

    # Geodesic error
    geodesic = compute_geodesic_error(pred, gt, threshold)

    return {
        'f1': f1.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': accuracy.item(),
        'geodesic': geodesic
    }


def compute_object_metrics(pred: torch.Tensor, gt: torch.Tensor):
    """
    Compute object affordance metrics: SIM, MAE, AUC, IOU.

    Args:
        pred: (B, N) predicted probabilities
        gt: (B, N) ground truth values

    Returns:
        dict with sim, mae, auc, iou
    """
    batch_size = pred.shape[0]
    eps = 1e-12
    thresholds = np.linspace(0, 1, 20)

    sim_total, mae_total, auc_total, iou_total = 0.0, 0.0, 0.0, 0.0
    valid_auc_samples = 0

    for b in range(batch_size):
        pred_b = pred[b]
        gt_b = gt[b]

        # SIM (Similarity)
        pred_norm = pred_b / (pred_b.sum() + eps)
        gt_norm = gt_b / (gt_b.sum() + eps)
        sim = torch.minimum(pred_norm, gt_norm).sum()
        sim_total += sim.item()

        # MAE
        mae = torch.abs(pred_b - gt_b).mean()
        mae_total += mae.item()

        # AUC and IOU
        gt_binary = (gt_b >= 0.5).int()
        unique_vals = torch.unique(gt_binary)

        if len(unique_vals) >= 2:
            try:
                auc = roc_auc_score(gt_binary.cpu().numpy(), pred_b.cpu().numpy())
                auc_total += auc
                valid_auc_samples += 1
            except ValueError:
                pass

            # IOU across thresholds
            ious = []
            for thresh in thresholds:
                pred_binary = (pred_b >= thresh).int()
                intersection = (pred_binary & gt_binary).sum()
                union = (pred_binary | gt_binary).sum()
                if union > 0:
                    ious.append((intersection / union).item())
                else:
                    ious.append(0.0)
            iou_total += np.mean(ious)

    return {
        'sim': sim_total / batch_size,
        'mae': mae_total / batch_size,
        'auc': auc_total / max(1, valid_auc_samples),
        'iou': iou_total / batch_size
    }


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    loss_fn: AffordanceLoss,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
    use_wandb: bool = False
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_human_loss = 0.0
    total_object_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        # Move data to device
        rgb_image = batch['rgb_image'].to(device)
        human_mask_valid = batch['human_mask_valid'].to(device)
        object_mask_valid = batch['object_mask_valid'].to(device)

        # Optional data
        object_points = batch.get('object_points')
        if object_points is not None:
            object_points = object_points.to(device)

        human_mask_gt = batch.get('human_mask_gt')
        if human_mask_gt is not None:
            human_mask_gt = human_mask_gt.to(device)

        object_mask_gt = batch.get('object_mask_gt')
        if object_mask_gt is not None:
            object_mask_gt = object_mask_gt.to(device)

        # Forward pass
        compute_human = human_mask_valid.any()
        compute_object = object_mask_valid.any() and object_points is not None

        outputs = model(
            rgb_image=rgb_image,
            object_points=object_points if compute_object else None,
            compute_human=compute_human,
            compute_object=compute_object
        )

        # Compute loss
        targets = {
            'human_mask_gt': human_mask_gt,
            'human_mask_valid': human_mask_valid,
            'object_mask_gt': object_mask_gt,
            'object_mask_valid': object_mask_valid
        }

        losses = loss_fn(outputs, targets)
        loss = losses['total_loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_human_loss += losses['human_loss'].item()
        total_object_loss += losses['object_loss'].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'h_loss': f"{losses['human_loss'].item():.4f}",
            'o_loss': f"{losses['object_loss'].item():.4f}"
        })

        # Log to wandb
        if use_wandb and HAS_WANDB and (step + 1) % log_interval == 0:
            wandb.log({
                'train/step_loss': loss.item(),
                'train/step_human_loss': losses['human_loss'].item(),
                'train/step_object_loss': losses['object_loss'].item(),
                'train/step': epoch * len(loader) + step
            })

    return {
        'train/loss': total_loss / max(1, num_batches),
        'train/human_loss': total_human_loss / max(1, num_batches),
        'train/object_loss': total_object_loss / max(1, num_batches)
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    loss_fn: AffordanceLoss,
    device: torch.device,
    epoch: int
):
    """Validate the model with comprehensive metrics."""
    model.eval()

    total_loss = 0.0
    total_human_loss = 0.0
    total_object_loss = 0.0
    num_batches = 0

    # Human metrics accumulators
    human_f1_list, human_prec_list, human_rec_list, human_acc_list, human_geo_list = [], [], [], [], []

    # Object metrics accumulators
    object_sim_list, object_mae_list, object_auc_list, object_iou_list = [], [], [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for batch in pbar:
        rgb_image = batch['rgb_image'].to(device)
        human_mask_valid = batch['human_mask_valid'].to(device)
        object_mask_valid = batch['object_mask_valid'].to(device)

        object_points = batch.get('object_points')
        if object_points is not None:
            object_points = object_points.to(device)

        human_mask_gt = batch.get('human_mask_gt')
        if human_mask_gt is not None:
            human_mask_gt = human_mask_gt.to(device)

        object_mask_gt = batch.get('object_mask_gt')
        if object_mask_gt is not None:
            object_mask_gt = object_mask_gt.to(device)

        compute_human = human_mask_valid.any()
        compute_object = object_mask_valid.any() and object_points is not None

        outputs = model(
            rgb_image=rgb_image,
            object_points=object_points if compute_object else None,
            compute_human=compute_human,
            compute_object=compute_object
        )

        targets = {
            'human_mask_gt': human_mask_gt,
            'human_mask_valid': human_mask_valid,
            'object_mask_gt': object_mask_gt,
            'object_mask_valid': object_mask_valid
        }

        losses = loss_fn(outputs, targets)

        total_loss += losses['total_loss'].item()
        total_human_loss += losses['human_loss'].item()
        total_object_loss += losses['object_loss'].item()
        num_batches += 1

        # Compute human metrics (F1, Precision, Recall, Accuracy, Geodesic)
        if compute_human and 'human_affordance' in outputs:
            valid_idx = human_mask_valid
            if valid_idx.any():
                pred_valid = outputs['human_affordance'][valid_idx]
                gt_valid = human_mask_gt[valid_idx]
                h_metrics = compute_human_metrics(pred_valid, gt_valid)
                human_f1_list.append(h_metrics['f1'])
                human_prec_list.append(h_metrics['precision'])
                human_rec_list.append(h_metrics['recall'])
                human_acc_list.append(h_metrics['accuracy'])
                human_geo_list.append(h_metrics['geodesic'])

        # Compute object metrics (SIM, MAE, AUC, IOU)
        if compute_object and 'object_affordance' in outputs:
            valid_idx = object_mask_valid
            if valid_idx.any():
                pred_valid = outputs['object_affordance'][valid_idx]
                gt_valid = object_mask_gt[valid_idx]
                o_metrics = compute_object_metrics(pred_valid, gt_valid)
                object_sim_list.append(o_metrics['sim'])
                object_mae_list.append(o_metrics['mae'])
                object_auc_list.append(o_metrics['auc'])
                object_iou_list.append(o_metrics['iou'])

    # Aggregate metrics
    metrics = {
        'val/loss': total_loss / max(1, num_batches),
        'val/human_loss': total_human_loss / max(1, num_batches),
        'val/object_loss': total_object_loss / max(1, num_batches),
    }

    # Human metrics
    if human_f1_list:
        metrics['val/human_f1'] = np.mean(human_f1_list)
        metrics['val/human_precision'] = np.mean(human_prec_list)
        metrics['val/human_recall'] = np.mean(human_rec_list)
        metrics['val/human_accuracy'] = np.mean(human_acc_list)
        metrics['val/human_geodesic'] = np.mean(human_geo_list)
    else:
        metrics['val/human_f1'] = 0.0
        metrics['val/human_precision'] = 0.0
        metrics['val/human_recall'] = 0.0
        metrics['val/human_accuracy'] = 0.0
        metrics['val/human_geodesic'] = 0.0

    # Object metrics
    if object_sim_list:
        metrics['val/object_sim'] = np.mean(object_sim_list)
        metrics['val/object_mae'] = np.mean(object_mae_list)
        metrics['val/object_auc'] = np.mean(object_auc_list)
        metrics['val/object_iou'] = np.mean(object_iou_list)
    else:
        metrics['val/object_sim'] = 0.0
        metrics['val/object_mae'] = 0.0
        metrics['val/object_auc'] = 0.0
        metrics['val/object_iou'] = 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Unified Affordance Pre-training")
    parser.add_argument('--config', type=str, default='configs/config.yaml')

    # Data arguments (override config)
    parser.add_argument('--human_data', type=str, default=None)
    parser.add_argument('--object_data', type=str, default=None)
    parser.add_argument('--intercap_data', type=str, default=None)
    parser.add_argument('--annot_data', type=str, default=None)
    parser.add_argument('--damon_root', type=str, default=None,
                        help="Root directory of DAMON dataset for human affordance")
    parser.add_argument('--piad_root', type=str, default=None,
                        help="Root directory of PIAD dataset for object affordance")
    parser.add_argument('--piad_setting', type=str, default='Seen',
                        help="PIAD setting: 'Seen' or 'Unseen'")

    # Training arguments (override config)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=4)

    # Logging
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb_project', type=str, default='affordance-pretrain')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')

    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {'model': {}, 'data': {}, 'training': {}, 'logging': {}}

    # Override config with command line arguments
    if args.human_data:
        config['data']['human_data_root'] = args.human_data
    if args.object_data:
        config['data']['object_data_root'] = args.object_data
    if args.intercap_data:
        config['data']['intercap_data'] = args.intercap_data
    if args.annot_data:
        config['data']['annot_data'] = args.annot_data
    if args.damon_root:
        config['data']['damon_root'] = args.damon_root
    if args.piad_root:
        config['data']['piad_root'] = args.piad_root
    if args.piad_setting:
        config['data']['piad_setting'] = args.piad_setting
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        os.environ.setdefault('WANDB_MODE', 'offline')
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=config
        )

    # Create dataloaders
    print("Creating dataloaders...")

    damon_root = config['data'].get('damon_root')
    piad_root = config['data'].get('piad_root')

    # Determine which dataset(s) to use
    if damon_root and piad_root:
        # Joint training: DAMON (human) + PIAD (object)
        print(f"Joint training mode:")
        print(f"  - DAMON (human): {damon_root}")
        print(f"  - PIAD (object): {piad_root}")

        from torch.utils.data import ConcatDataset
        from data.damon_dataset import DAMONDataset, DAMONTransform, damon_collate_fn
        from data.piad_dataset import PIADDataset, PIADTransform

        batch_size = config['training'].get('batch_size', 8)
        image_size = config['data'].get('image_size', 224)

        # Create DAMON datasets
        damon_transform = DAMONTransform(image_size=image_size)
        damon_train = DAMONDataset(damon_root, 'train', damon_transform)
        damon_val = DAMONDataset(damon_root, 'val', damon_transform)

        # Create PIAD datasets
        piad_transform = PIADTransform(image_size=image_size)
        piad_setting = config['data'].get('piad_setting', 'Seen')
        piad_train = PIADDataset(piad_root, piad_setting, 'train', piad_transform)
        piad_val = PIADDataset(piad_root, piad_setting, 'test', piad_transform)

        # Combine datasets
        from data.unified_aff_dataset import collate_fn
        train_dataset = ConcatDataset([damon_train, piad_train])
        val_dataset = ConcatDataset([damon_val, piad_val])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    elif damon_root:
        # DAMON only (human affordance)
        print(f"Using DAMON dataset from: {damon_root}")
        train_loader, val_loader = create_damon_dataloaders(
            damon_root=damon_root,
            batch_size=config['training'].get('batch_size', 8),
            num_workers=args.num_workers,
            image_size=config['data'].get('image_size', 224),
            num_human_points=config['data'].get('num_human_points', 10475)
        )

    elif piad_root:
        # PIAD only (object affordance)
        print(f"Using PIAD dataset from: {piad_root}")
        train_loader, val_loader = create_piad_dataloaders(
            piad_root=piad_root,
            setting=config['data'].get('piad_setting', 'Seen'),
            batch_size=config['training'].get('batch_size', 8),
            num_workers=args.num_workers,
            image_size=config['data'].get('image_size', 224),
            num_object_points=config['data'].get('num_object_points', 2048)
        )

    else:
        # Use unified dataset
        train_loader, val_loader = create_affordance_dataloaders(
            human_data_root=config['data'].get('human_data_root'),
            object_data_root=config['data'].get('object_data_root'),
            intercap_data=config['data'].get('intercap_data'),
            annot_data=config['data'].get('annot_data'),
            batch_size=config['training'].get('batch_size', 8),
            num_workers=args.num_workers,
            image_size=config['data'].get('image_size', 224),
            num_object_points=config['data'].get('num_object_points', 1024),
            num_human_points=config['data'].get('num_human_points', 10475)
        )

    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Build model
    print("Building model...")
    model_config = config.get('model', {})
    model_config['device'] = str(device)
    model = build_affordance_model(model_config).to(device)

    # Freeze VLM if specified
    if model_config.get('freeze_vlm', True):
        model.freeze_vlm()
        print("VLM backbone frozen")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer with different learning rates
    train_config = config.get('training', {})
    base_lr = train_config.get('learning_rate', 1e-4)

    param_groups = model.get_trainable_parameters()
    optimizer_params = [
        {'params': param_groups['vlm'], 'lr': base_lr * train_config.get('lr_vlm_scale', 0.1)},
        {'params': param_groups['encoder'], 'lr': base_lr * train_config.get('lr_encoder_scale', 0.5)},
        {'params': param_groups['decoder'], 'lr': base_lr * train_config.get('lr_decoder_scale', 1.0)},
        {'params': param_groups['human_branch'], 'lr': base_lr},
        {'params': param_groups['object_branch'], 'lr': base_lr}
    ]
    # Filter out empty param groups
    optimizer_params = [p for p in optimizer_params if len(p['params']) > 0]

    optimizer = AdamW(
        optimizer_params,
        lr=base_lr,
        weight_decay=train_config.get('weight_decay', 1e-4)
    )

    # Setup scheduler - use warm restarts to escape local minima
    num_epochs = train_config.get('num_epochs', 50)
    warmup_epochs = train_config.get('warmup_epochs', 5)
    use_warm_restarts = train_config.get('use_warm_restarts', True)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    if use_warm_restarts:
        # CosineAnnealingWarmRestarts: restart every T_0 epochs
        # T_mult=2 means restart periods double each time (10, 20, 40, ...)
        restart_period = train_config.get('restart_period', 10)
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=restart_period,
            T_mult=2,
            eta_min=1e-6
        )
    else:
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-7
        )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )

    # Setup loss
    loss_fn = AffordanceLoss(
        use_focal=train_config.get('use_focal', True),
        use_dice=train_config.get('use_dice', True),
        focal_alpha=train_config.get('focal_alpha', 0.25),
        focal_gamma=train_config.get('focal_gamma', 2.0),
        lambda_human=train_config.get('lambda_human', 1.0),
        lambda_object=train_config.get('lambda_object', 1.0),
        lambda_dice=train_config.get('lambda_dice', 1.0),
        lambda_focal=train_config.get('lambda_focal', 1.0)
    )

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    log_interval = config.get('logging', {}).get('log_interval', 20)
    save_interval = config.get('logging', {}).get('save_interval', 5)

    print("Starting training...")
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            log_interval=log_interval, use_wandb=use_wandb
        )

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device, epoch)

        # Update scheduler
        scheduler.step()

        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch, 'lr': current_lr}

        print(f"Epoch {epoch}: train_loss={train_metrics['train/loss']:.4f}, "
              f"val_loss={val_metrics['val/loss']:.4f}")
        print(f"  Human: F1={val_metrics['val/human_f1']:.4f}, "
              f"Prec={val_metrics['val/human_precision']:.4f}, "
              f"Recall={val_metrics['val/human_recall']:.4f}, "
              f"Acc={val_metrics['val/human_accuracy']:.4f}, "
              f"Geo={val_metrics['val/human_geodesic']:.4f}")
        print(f"  Object: SIM={val_metrics['val/object_sim']:.4f}, "
              f"MAE={val_metrics['val/object_mae']:.4f}, "
              f"AUC={val_metrics['val/object_auc']:.4f}, "
              f"IOU={val_metrics['val/object_iou']:.4f}")

        if use_wandb:
            wandb.log(metrics)

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_path = save_dir / f'epoch_{epoch:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        # Save best model
        if val_metrics['val/loss'] < best_val_loss:
            best_val_loss = val_metrics['val/loss']
            best_path = save_dir / 'best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, best_path)
            print(f"Saved best model (val_loss={best_val_loss:.4f})")

    # Save final model
    final_path = save_dir / 'final.pth'
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    print(f"Training complete. Final model saved to {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
