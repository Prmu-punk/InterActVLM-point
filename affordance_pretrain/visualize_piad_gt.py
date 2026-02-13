"""
Visualize PIAD dataset ground truth affordance labels in 3D
Saves point clouds as .ply files with vertex colors
Matches training data format: use_all_affordances=True
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Error: trimesh required. Install: pip install trimesh")


def pc_normalize(pc: np.ndarray):
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def save_pointcloud_ply(points: np.ndarray, affordance_mask: np.ndarray, save_path: str):
    """Save point cloud with affordance as vertex colors."""
    colors = np.zeros((len(points), 4), dtype=np.uint8)
    colors[:, 3] = 255
    colors[:, :3] = 180  # Gray base

    # Red for affordance
    colors[affordance_mask, 0] = 255
    colors[affordance_mask, 1] = 0
    colors[affordance_mask, 2] = 0

    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    cloud.export(save_path)


def load_piad_pointcloud(point_path: str):
    """Load PIAD point cloud file."""
    coordinates = []
    with open(point_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            data = [float(x) for x in parts[2:]]
            coordinates.append(data)

    data_array = np.array(coordinates, dtype=np.float32)
    points = data_array[:, 0:3]
    affordance_labels = data_array[:, 3:]  # (N, 17)

    # Normalize (same as training)
    points = pc_normalize(points)

    # Create mask same as training: any affordance > 0.5
    affordance_mask = affordance_labels.max(axis=1) > 0.5

    return points, affordance_mask, affordance_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--piad_root', type=str,
                        default='/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/PIAD')
    parser.add_argument('--setting', type=str, default='Seen')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='vis_results/piad_gt_3d')
    parser.add_argument('--start_idx', type=int, default=0)
    args = parser.parse_args()

    if not HAS_TRIMESH:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    piad_root = Path(args.piad_root)
    data_dir = piad_root / args.setting
    split_suffix = 'Train' if args.split == 'train' else 'Test'

    with open(data_dir / f'Img_{split_suffix}.txt', 'r') as f:
        img_list = [l.strip() for l in f if l.strip()]
    with open(data_dir / f'Point_{split_suffix}.txt', 'r') as f:
        point_list = [l.strip() for l in f if l.strip()]

    print(f"PIAD {args.setting}/{args.split}: {len(img_list)} images, {len(point_list)} point clouds")

    end_idx = min(args.start_idx + args.num_samples, len(point_list))

    for idx in range(args.start_idx, end_idx):
        # Load point cloud
        point_rel = point_list[idx]
        parts = Path(point_rel).parts
        point_path = piad_root / '/'.join(parts[1:]) if parts[0] == 'Data' else piad_root / point_rel

        if not point_path.exists():
            print(f"  [{idx}] Not found: {point_path}")
            continue

        points, aff_mask, _ = load_piad_pointcloud(str(point_path))

        # Get image
        img_idx = idx % len(img_list)
        img_rel = img_list[img_idx]
        parts = Path(img_rel).parts
        img_path = piad_root / '/'.join(parts[1:]) if parts[0] == 'Data' else piad_root / img_rel

        # Extract object/affordance type from image path
        filename = Path(img_rel).stem
        fname_parts = filename.split('_')
        obj_type = fname_parts[-3] if len(fname_parts) >= 3 else 'unknown'
        aff_type = fname_parts[-2] if len(fname_parts) >= 2 else 'unknown'

        sample_name = Path(point_rel).stem

        # Save .ply
        ply_path = output_dir / f"{idx:04d}_{sample_name}.ply"
        save_pointcloud_ply(points, aff_mask, str(ply_path))

        # Save image
        if img_path.exists():
            img_dst = output_dir / f"{idx:04d}_{sample_name}.jpg"
            Image.open(img_path).convert('RGB').save(img_dst)

        print(f"  [{idx}] {obj_type}/{aff_type}: {aff_mask.sum()}/{len(points)} affordance points")

    print(f"\nDone! Saved to: {output_dir}")


if __name__ == '__main__':
    main()
