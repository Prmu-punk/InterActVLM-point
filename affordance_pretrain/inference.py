"""
Inference and 3D Visualization for Unified Affordance Pre-training
Saves predictions as .ply/.obj files
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
from PIL import Image
from tqdm import tqdm

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Error: trimesh required. Install: pip install trimesh")

from affordance_pretrain.aff_models.unified_aff_model import build_affordance_model
from affordance_pretrain.data.piad_dataset import PIADDataset, PIADTransform, AFFORDANCE_CLASSES
from affordance_pretrain.data.damon_dataset import DAMONDataset, DAMONTransform


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    model_config['device'] = device

    model = build_affordance_model(model_config).to(device)

    # Filter VLM weights for compatibility
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items()
                      if k in model_state and model_state[k].shape == v.shape}

    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
    return model


def load_smplx_template(smplx_path: str):
    """Load SMPLX template."""
    data = np.load(smplx_path, allow_pickle=True)
    return data['v_template'], data['f']


def save_mesh_ply(vertices, faces, values, save_path, threshold=0.5):
    """Save mesh with vertex colors based on values."""
    colors = np.zeros((len(vertices), 4), dtype=np.uint8)
    colors[:, 3] = 255
    colors[:, :3] = 180  # Gray

    mask = values > threshold
    colors[mask, 0] = 255
    colors[mask, 1] = 0
    colors[mask, 2] = 0

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    mesh.export(save_path)


def save_pointcloud_ply(points, values, save_path, threshold=0.5):
    """Save point cloud with vertex colors based on values."""
    colors = np.zeros((len(points), 4), dtype=np.uint8)
    colors[:, 3] = 255
    colors[:, :3] = 180  # Gray

    mask = values > threshold
    colors[mask, 0] = 255
    colors[mask, 1] = 0
    colors[mask, 2] = 0

    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    cloud.export(save_path)


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from ImageNet normalization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    image = image * std + mean
    return np.clip(image * 255, 0, 255).astype(np.uint8)


@torch.no_grad()
def run_inference_piad(model, piad_root, setting, device, num_samples, output_dir):
    """Run inference on PIAD and save 3D results."""
    print(f"\n{'='*50}")
    print("PIAD Inference (Object Affordance)")
    print(f"{'='*50}")

    transform = PIADTransform(image_size=224)
    dataset = PIADDataset(
        piad_root=piad_root, setting=setting, split='test',
        transform=transform, num_object_points=2048
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_samples = min(num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    total_correct, total_points = 0, 0

    for i, idx in enumerate(tqdm(indices)):
        sample = dataset[idx]

        rgb_image = sample['rgb_image'].unsqueeze(0).to(device)
        object_points = sample['object_points'].unsqueeze(0).to(device)

        outputs = model(rgb_image=rgb_image, object_points=object_points,
                        compute_human=False, compute_object=True)

        pred = outputs['object_affordance'].squeeze(0).cpu().numpy()
        gt = sample['object_mask_gt'].numpy()
        points = sample['object_points'].numpy()

        # Accuracy
        total_correct += ((pred > 0.5) == (gt > 0.5)).sum()
        total_points += len(pred)

        # Save files
        obj_type = sample['object_type']
        aff_type = AFFORDANCE_CLASSES[sample['affordance_index']]
        base_name = f"{i:04d}_{obj_type}_{aff_type}"

        # Prediction .ply
        save_pointcloud_ply(points, pred, str(output_path / f"{base_name}_pred.ply"))
        # GT .ply
        save_pointcloud_ply(points, gt, str(output_path / f"{base_name}_gt.ply"))
        # Image
        img_np = denormalize_image(sample['rgb_image'].numpy())
        Image.fromarray(img_np).save(output_path / f"{base_name}.jpg")

    acc = total_correct / total_points * 100
    print(f"\nPIAD Accuracy: {acc:.2f}%")
    print(f"Saved to: {output_path}")
    return acc


@torch.no_grad()
def run_inference_damon(model, damon_root, smplx_path, device, num_samples, output_dir):
    """Run inference on DAMON and save 3D results."""
    print(f"\n{'='*50}")
    print("DAMON Inference (Human Affordance)")
    print(f"{'='*50}")

    # Load SMPLX template
    smplx_verts, smplx_faces = load_smplx_template(smplx_path)
    print(f"  SMPLX: {smplx_verts.shape[0]} vertices")

    transform = DAMONTransform(image_size=224)
    dataset = DAMONDataset(
        damon_root=damon_root, split='val',
        transform=transform, num_human_points=10475
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_samples = min(num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    total_correct, total_points = 0, 0

    for i, idx in enumerate(tqdm(indices)):
        sample = dataset[idx]

        rgb_image = sample['rgb_image'].unsqueeze(0).to(device)

        outputs = model(rgb_image=rgb_image, object_points=None,
                        compute_human=True, compute_object=False)

        pred = outputs['human_affordance'].squeeze(0).cpu().numpy()
        gt = sample['human_mask_gt'].numpy()

        # Accuracy
        total_correct += ((pred > 0.5) == (gt > 0.5)).sum()
        total_points += len(pred)

        # Save files
        sample_id = sample['sample_id']
        base_name = f"{i:04d}_{sample_id}"

        # Prediction .ply (mesh)
        save_mesh_ply(smplx_verts, smplx_faces, pred,
                      str(output_path / f"{base_name}_pred.ply"))
        # GT .ply (mesh)
        save_mesh_ply(smplx_verts, smplx_faces, gt,
                      str(output_path / f"{base_name}_gt.ply"))
        # Image
        img_np = denormalize_image(sample['rgb_image'].numpy())
        Image.fromarray(img_np).save(output_path / f"{base_name}.jpg")

    acc = total_correct / total_points * 100
    print(f"\nDAMON Accuracy: {acc:.2f}%")
    print(f"Saved to: {output_path}")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--damon_root', type=str, default=None)
    parser.add_argument('--piad_root', type=str, default=None)
    parser.add_argument('--piad_setting', type=str, default='Seen')
    parser.add_argument('--smplx_path', type=str,
                        default='/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/smpl_models/SMPLX_NEUTRAL.npz')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='vis_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='both', choices=['piad', 'damon', 'both'])
    args = parser.parse_args()

    if not HAS_TRIMESH:
        print("trimesh required for 3D output")
        return

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    model = load_model(args.checkpoint, args.device)

    results = {}

    if args.dataset in ['piad', 'both'] and args.piad_root:
        results['piad'] = run_inference_piad(
            model, args.piad_root, args.piad_setting, args.device,
            args.num_samples, os.path.join(args.output_dir, 'piad_3d')
        )

    if args.dataset in ['damon', 'both'] and args.damon_root:
        results['damon'] = run_inference_damon(
            model, args.damon_root, args.smplx_path, args.device,
            args.num_samples, os.path.join(args.output_dir, 'damon_3d')
        )

    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}%")
    print(f"\n3D files saved to: {args.output_dir}")
    print("  - *_pred.ply: model prediction (red=contact)")
    print("  - *_gt.ply: ground truth (red=contact)")
    print("  - *.jpg: input image")


if __name__ == '__main__':
    main()
