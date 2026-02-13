import argparse
import os
from typing import Tuple

import numpy as np
import torch
import trimesh
try:
    from scipy.spatial import cKDTree
    _HAS_CKDTREE = True
except Exception:
    _HAS_CKDTREE = False

from data.dataset import create_dataloaders
from models import build_model
from utils.keypoints import KeypointManager


def save_ply(points: np.ndarray, path: str) -> None:
    if points.size == 0:
        points = np.zeros((0, 3), dtype=np.float32)
    pc = trimesh.PointCloud(points.astype(np.float32))
    pc.export(path)


def align_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    if mask.ndim == 2:
        mask = mask[0]
    if mask.shape[0] < target_len:
        pad = np.zeros((target_len - mask.shape[0],), dtype=mask.dtype)
        mask = np.concatenate([mask, pad], axis=0)
    elif mask.shape[0] > target_len:
        mask = mask[:target_len]
    return mask


def snap_to_cloud(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    if points.size == 0 or cloud.size == 0:
        return points
    if _HAS_CKDTREE:
        tree = cKDTree(cloud)
        _, idx = tree.query(points, k=1)
        return cloud[idx]
    pts = torch.from_numpy(points.astype(np.float32))
    cld = torch.from_numpy(cloud.astype(np.float32))
    dist = torch.cdist(pts.unsqueeze(0), cld.unsqueeze(0)).squeeze(0)
    idx = torch.argmin(dist, dim=1)
    return cloud[idx.numpy()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intercap_data', type=str, default="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/INTERCAP_train")
    parser.add_argument('--annot_data', type=str, default="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/annotations")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/vis_sample')
    parser.add_argument('--keypoints_json', type=str, default='data/part_kp.json')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, _, test_loader = create_dataloaders(
        intercap_data=args.intercap_data,
        annot_data=args.annot_data,
        batch_size=1,
        num_workers=0,
        load_contact_masks=True
    )
    it = iter(test_loader)
    _ = next(it)
    _ = next(it)
    batch = next(it)

    rgb = batch['rgb_image'].to(device)
    obj_pts = batch['object_points'].to(device)

    # Keep inference config consistent with scripts/train_point_wandb.py
    model = build_model({
        'd_tr': 256,
        'num_body_points': 87,
        'num_object_queries': 87,
        'use_lightweight_vlm': False,
        'device': str(device)
    }).to(device)

    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        outputs = model(rgb, obj_pts, return_aux=False)

    # Object full and masked (model affordance)
    object_points = batch['object_points'][0].numpy()
    object_aff = outputs['object_affordance'][0].cpu().numpy()
    object_mask = (object_aff > args.threshold).astype(np.uint8)
    object_mask = align_mask(object_mask, object_points.shape[0])

    save_ply(object_points, os.path.join(args.output_dir, 'object_points.ply'))
    save_ply(object_points[object_mask > 0], os.path.join(args.output_dir, 'object_points_contact.ply'))

    # Human SMPL-X vertices (full 10475)
    human_vertices = batch['human_vertices'][0].numpy()
    human_full = human_vertices

    human_aff = outputs['human_affordance'][0].cpu().numpy()
    human_mask = (human_aff > args.threshold).astype(np.uint8)
    human_mask = align_mask(human_mask, human_full.shape[0])

    save_ply(human_full, os.path.join(args.output_dir, 'human_points_10475.ply'))
    save_ply(human_full[human_mask > 0], os.path.join(args.output_dir, 'human_points_10475_contact.ply'))

    # Predicted discrete contacts (human labels + object point indices)
    pred_human = outputs['human_contact'][0].cpu().numpy()
    pred_human_mask = pred_human > args.threshold
    pred_object_idx = outputs['object_index_logits'][0].argmax(dim=-1).cpu().numpy()

    keypoints = KeypointManager(args.keypoints_json)
    kp_idx = keypoints.get_vertex_indices()
    human_87 = human_vertices[kp_idx]
    human_87_contact = human_87[:87][pred_human_mask]
    selected_obj_idx = pred_object_idx[pred_human_mask]
    selected_obj_idx = selected_obj_idx[(selected_obj_idx >= 0) & (selected_obj_idx < object_points.shape[0])]
    object_coords_contact = object_points[selected_obj_idx] if selected_obj_idx.size > 0 else np.zeros((0, 3), dtype=np.float32)

    save_ply(human_87_contact, os.path.join(args.output_dir, 'human_keypoints_contact.ply'))
    save_ply(object_coords_contact, os.path.join(args.output_dir, 'object_coords_contact.ply'))

    # np.savez(
    #     os.path.join(args.output_dir, 'discrete_contacts_pred.npz'),
    #     human_contact_labels=pred_human,
    #     human_contact_points=human_87_contact,
    #     object_contact_points=object_coords_contact
    # )

    print(f"Saved outputs to {args.output_dir}")


if __name__ == '__main__':
    main()
