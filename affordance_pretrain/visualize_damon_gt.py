"""
Visualize DAMON dataset ground truth contact labels in 3D
Saves as .ply mesh files with vertex colors
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Warning: trimesh not installed. Install with: pip install trimesh")


def load_smplx_from_npz(npz_path: str):
    """Load SMPLX template directly from npz file."""
    data = np.load(npz_path, allow_pickle=True)

    if 'v_template' in data:
        vertices = data['v_template']
    else:
        raise KeyError(f"v_template not found in {npz_path}. Keys: {list(data.keys())}")

    if 'f' in data:
        faces = data['f']
    else:
        faces = None

    return vertices, faces


def save_mesh_with_contact(
    vertices: np.ndarray,
    faces: np.ndarray,
    contact_labels: np.ndarray,
    save_path: str,
    colormap: str = 'hot'
):
    """
    Save mesh with contact labels as vertex colors.

    Args:
        vertices: (N, 3) mesh vertices
        faces: (F, 3) mesh faces
        contact_labels: (N,) contact labels per vertex [0, 1]
        save_path: Path to save (.ply or .obj)
        colormap: Colormap for contact visualization
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh is required for 3D export")

    # Create vertex colors based on contact labels
    # Non-contact: light gray, Contact: red gradient
    colors = np.zeros((len(vertices), 4), dtype=np.uint8)
    colors[:, 3] = 255  # Alpha

    # Base color: light gray
    colors[:, 0] = 180
    colors[:, 1] = 180
    colors[:, 2] = 180

    # Contact regions: red with intensity based on contact value
    contact_mask = contact_labels > 0.1
    colors[contact_mask, 0] = 255  # Red
    colors[contact_mask, 1] = ((1 - contact_labels[contact_mask]) * 100).astype(np.uint8)
    colors[contact_mask, 2] = 0

    # High contact (>0.5): bright red
    high_contact = contact_labels > 0.5
    colors[high_contact, 0] = 255
    colors[high_contact, 1] = 0
    colors[high_contact, 2] = 0

    # Create mesh
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors
    )

    # Export
    mesh.export(save_path)
    return mesh


def load_damon_data(damon_root: str, split: str = 'train'):
    """Load DAMON dataset."""
    damon_root = Path(damon_root)

    if split == 'train':
        npz_path = damon_root / 'Release_Datasets' / 'damon' / 'hot_dca_trainval.npz'
    else:
        npz_path = damon_root / 'Release_Datasets' / 'damon' / 'hot_dca_test.npz'

    print(f"Loading DAMON data from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    return data, damon_root / 'HOT-Annotated' / 'images'


def main():
    parser = argparse.ArgumentParser(description="Visualize DAMON GT contact labels - save as 3D mesh")

    parser.add_argument('--damon_root', type=str,
                        default='/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/DAMON')
    parser.add_argument('--smplx_path', type=str,
                        default='/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/smpl_models/SMPLX_NEUTRAL.npz')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='vis_results/damon_gt_3d')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--format', type=str, default='ply', choices=['ply', 'obj'])

    args = parser.parse_args()

    if not HAS_TRIMESH:
        print("Error: trimesh is required. Install with: pip install trimesh")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SMPLX template
    print(f"Loading SMPLX template from: {args.smplx_path}")
    smplx_vertices, smplx_faces = load_smplx_from_npz(args.smplx_path)
    print(f"  SMPLX vertices: {smplx_vertices.shape}")
    print(f"  SMPLX faces: {smplx_faces.shape}")

    # Load DAMON data
    data, image_root = load_damon_data(args.damon_root, args.split)

    imgnames = data['imgname']
    contact_labels = data['contact_label_smplx']

    print(f"Total samples: {len(imgnames)}")

    # Visualize samples
    end_idx = min(args.start_idx + args.num_samples, len(imgnames))

    print(f"\nSaving 3D meshes for samples {args.start_idx} to {end_idx-1}...")

    for idx in range(args.start_idx, end_idx):
        imgname = imgnames[idx]
        contact = contact_labels[idx]

        # Statistics
        contact_ratio = (contact > 0.5).mean() * 100
        num_contact = (contact > 0.5).sum()

        sample_name = Path(imgname).stem

        # Save mesh
        mesh_path = output_dir / f"{idx:04d}_{sample_name}.{args.format}"
        save_mesh_with_contact(
            vertices=smplx_vertices,
            faces=smplx_faces,
            contact_labels=contact,
            save_path=str(mesh_path)
        )

        # Also copy the corresponding image
        img_filename = Path(imgname).name
        img_src = image_root / img_filename
        if img_src.exists():
            img_dst = output_dir / f"{idx:04d}_{sample_name}.jpg"
            Image.open(img_src).save(img_dst)

        print(f"  [{idx}] {sample_name}: {num_contact} contacts ({contact_ratio:.1f}%) -> {mesh_path.name}")

    print(f"\nDone! Results saved to: {output_dir}")
    print(f"  - .{args.format} files: 3D mesh with contact colors (red=contact, gray=no contact)")
    print(f"  - .jpg files: corresponding RGB images")


if __name__ == '__main__':
    main()
