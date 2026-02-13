import argparse
import os

import torch

import wandb

from data.dataset import create_dataloaders
from utils.keypoints import KeypointManager
from models import build_model
from tqdm import tqdm
# import torch.multiprocessing as mp

# mp.set_start_method("spawn", force=True)
# torch.multiprocessing.set_sharing_strategy("file_system")

def _align_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[-1] < target_len:
        pad = torch.zeros(
            mask.shape[0],
            target_len - mask.shape[-1],
            dtype=mask.dtype,
            device=mask.device
        )
        mask = torch.cat([mask, pad], dim=-1)
    elif mask.shape[-1] > target_len:
        mask = mask[..., :target_len]
    return mask.float()


def train_one_epoch(model, loader, optimizer, device, epoch, log_interval=20):
    model.train()
    total = 0.0
    total_human = 0.0
    total_obj = 0.0
    total_human_aff = 0.0
    total_obj_aff = 0.0
    total_obj_aff_cons = 0.0
    total_human_aff_cons = 0.0

    pbar = tqdm(enumerate(loader), desc=f"Training epoch {epoch}", total=len(loader))
    for step, batch in pbar:
        rgb = batch['rgb_image'].to(device)
        obj_pts = batch['object_points'].to(device)
        human_labels = batch['human_labels'].to(device)
        object_indices = batch['object_indices'].to(device)

        outputs = model(rgb, obj_pts, return_aux=False)

        human_mask = batch.get('human_contact_mask', torch.zeros_like(outputs['human_affordance_logits']))
        object_mask = batch.get('object_contact_mask', torch.zeros_like(outputs['object_affordance_logits']))

        # human_mask = _align_mask(human_mask, outputs['human_affordance_logits'].shape[-1]).to(device)
        # object_mask = _align_mask(object_mask, outputs['object_affordance_logits'].shape[-1]).to(device)

        predictions = {
            'human_logits': outputs['human_logits'],
            'human_contact': outputs['human_contact'],
            'object_index_logits': outputs['object_index_logits'],
            'object_index_probs': outputs['object_index_probs'],
            'human_mask': outputs['human_affordance_logits'],
            'object_mask': outputs['object_affordance_logits'],
            'object_affordance_logits': outputs['object_affordance_logits'],
            'human_affordance_logits': outputs['human_affordance_logits']
        }
        human_kp_idx = batch.get('human_keypoint_indices')
        if isinstance(human_kp_idx, torch.Tensor):
            human_kp_idx = human_kp_idx.to(device)
        targets = {
            'human_labels': human_labels,
            'object_indices': object_indices,
            'object_points': obj_pts,
            'human_keypoint_indices': human_kp_idx,
            'human_mask_gt': human_mask,
            'object_mask_gt': object_mask
        }

        losses = model.compute_loss(predictions, targets, compute_aux=True)
        loss = losses['total_loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        total_human += losses['human_loss'].item()
        total_obj += losses['object_loss'].item()
        total_human_aff += losses['human_mask_loss'].item()
        total_obj_aff += losses['object_mask_loss'].item()
        total_obj_aff_cons += losses['object_aff_consistency_loss'].item()
        total_human_aff_cons += losses['human_aff_consistency_loss'].item()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'human': f"{losses['human_loss'].item():.4f}",
            'obj': f"{losses['object_loss'].item():.4f}",
            'h_mask': f"{losses['human_mask_loss'].item():.4f}",
            'o_mask': f"{losses['object_mask_loss'].item():.4f}",
            'h_aff_cons': f"{losses['human_aff_consistency_loss'].item():.4f}",
            'o_aff_cons': f"{losses['object_aff_consistency_loss'].item():.4f}",
        })

        if (step + 1) % log_interval == 0:
            wandb.log({
                'train/step_loss': loss.item(),
                'train/step_human_loss': losses['human_loss'].item(),
                'train/step_object_loss': losses['object_loss'].item(),
                'train/step_human_aff_loss': losses['human_mask_loss'].item(),
                'train/step_object_aff_loss': losses['object_mask_loss'].item(),
                'train/step_object_aff_consistency_loss': losses['object_aff_consistency_loss'].item(),
                'train/step_human_aff_consistency_loss': losses['human_aff_consistency_loss'].item(),
            })

    n = max(1, len(loader))
    return {
        'train/loss': total / n,
        'train/human_loss': total_human / n,
        'train/object_loss': total_obj / n,
        'train/human_aff_loss': total_human_aff / n,
        'train/object_aff_loss': total_obj_aff / n,
        'train/object_aff_consistency_loss': total_obj_aff_cons / n,
        'train/human_aff_consistency_loss': total_human_aff_cons / n,
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = 0.0
    total_human = 0.0
    total_obj = 0.0
    total_human_aff = 0.0
    total_obj_aff = 0.0
    total_obj_aff_cons = 0.0
    total_human_aff_cons = 0.0

    for batch in loader:
        rgb = batch['rgb_image'].to(device)
        obj_pts = batch['object_points'].to(device)
        human_labels = batch['human_labels'].to(device)
        object_indices = batch['object_indices'].to(device)

        outputs = model(rgb, obj_pts, return_aux=False)

        human_mask = batch.get('human_contact_mask', torch.zeros_like(outputs['human_affordance_logits']))
        object_mask = batch.get('object_contact_mask', torch.zeros_like(outputs['object_affordance_logits']))

        human_mask = _align_mask(human_mask, outputs['human_affordance_logits'].shape[-1]).to(device)
        object_mask = _align_mask(object_mask, outputs['object_affordance_logits'].shape[-1]).to(device)

        predictions = {
            'human_logits': outputs['human_logits'],
            'human_contact': outputs['human_contact'],
            'object_index_logits': outputs['object_index_logits'],
            'object_index_probs': outputs['object_index_probs'],
            'human_mask': outputs['human_affordance_logits'],
            'object_mask': outputs['object_affordance_logits'],
            'object_affordance_logits': outputs['object_affordance_logits'],
            'human_affordance_logits': outputs['human_affordance_logits']
        }
        human_kp_idx = batch.get('human_keypoint_indices')
        if isinstance(human_kp_idx, torch.Tensor):
            human_kp_idx = human_kp_idx.to(device)
        targets = {
            'human_labels': human_labels,
            'object_indices': object_indices,
            'object_points': obj_pts,
            'human_keypoint_indices': human_kp_idx,
            'human_mask_gt': human_mask,
            'object_mask_gt': object_mask
        }

        losses = model.compute_loss(predictions, targets, compute_aux=True)
        loss = losses['total_loss']

        total += loss.item()
        total_human += losses['human_loss'].item()
        total_obj += losses['object_loss'].item()
        total_human_aff += losses['human_mask_loss'].item()
        total_obj_aff += losses['object_mask_loss'].item()
        total_obj_aff_cons += losses['object_aff_consistency_loss'].item()
        total_human_aff_cons += losses['human_aff_consistency_loss'].item()

    n = max(1, len(loader))
    return {
        'val/loss': total / n,
        'val/human_loss': total_human / n,
        'val/object_loss': total_obj / n,
        'val/human_aff_loss': total_human_aff / n,
        'val/object_aff_loss': total_obj_aff / n,
        'val/object_aff_consistency_loss': total_obj_aff_cons / n,
        'val/human_aff_consistency_loss': total_human_aff_cons / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intercap_data', type=str, required=True)
    parser.add_argument('--annot_data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--project', type=str, default='ivd-point')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--train_vlm', action='store_true', help='Enable LoRA/projection training in VLM')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.environ.setdefault('WANDB_MODE', 'offline')
    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    keypoint_manager = KeypointManager('data/part_kp.json')
    train_loader, val_loader, _ = create_dataloaders(
        intercap_data=args.intercap_data,
        annot_data=args.annot_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        keypoint_manager=keypoint_manager,
        load_contact_masks=True
    )

    model = build_model({
        'd_tr': 256,
        'num_body_points': 87,
        'num_object_queries': 87,
        'use_lightweight_vlm': False,
        'device': str(device)
    }).to(device)

    if hasattr(model, 'vlm') and hasattr(model.vlm, 'print_trainable_params'):
        model.vlm.print_trainable_params()

    if args.train_vlm:
        # Unfreeze input embeddings so newly added special tokens can learn
        emb = model.vlm.model.get_input_embeddings()
        emb.weight.requires_grad = True

    else:
        model.freeze_vlm()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
        wandb.log(metrics)
        if epoch % 2 == 0:
            ckpt_path = os.path.join(args.save_dir, f'epoch_{epoch:03d}.pth')
            torch.save({'model_state_dict': model.state_dict()}, ckpt_path)


if __name__ == '__main__':
    main()
