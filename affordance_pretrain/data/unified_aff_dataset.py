"""
Unified Affordance Dataset for Pre-training
Supports mixed data with partial annotations (human-only or object-only)
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import trimesh


class HumanAffordanceDataset(Dataset):
    """
    Dataset for human affordance annotations.

    Expected directory structure:
        human_data_root/
        ├── images/
        │   ├── sample1.jpg
        │   └── sample2.jpg
        ├── masks/
        │   ├── sample1.npy  # (10475,) binary mask
        │   └── sample2.npy
        └── splits/
            ├── train.txt
            └── val.txt
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform=None,
        num_human_points: int = 10475
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.num_human_points = num_human_points

        self.image_dir = self.data_root / 'images'
        self.mask_dir = self.data_root / 'masks'

        self.sample_ids = self._load_split()

    def _load_split(self) -> List[str]:
        """Load sample IDs for the split."""
        split_file = self.data_root / 'splits' / f'{self.split}.txt'

        if split_file.exists():
            with open(split_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            # Fallback: use all mask files
            if self.mask_dir.exists():
                return [p.stem for p in self.mask_dir.glob('*.npy')]
            return []

    def _load_image(self, sample_id: str) -> np.ndarray:
        """Load RGB image."""
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = self.image_dir / f'{sample_id}{ext}'
            if img_path.exists():
                image = Image.open(img_path).convert('RGB')
                return np.array(image)
        raise FileNotFoundError(f"Image not found for {sample_id}")

    def _load_mask(self, sample_id: str) -> np.ndarray:
        """Load human contact mask."""
        mask_path = self.mask_dir / f'{sample_id}.npy'
        if mask_path.exists():
            mask = np.load(mask_path).astype(np.float32)
            # Pad or truncate to num_human_points
            if len(mask) < self.num_human_points:
                mask = np.pad(mask, (0, self.num_human_points - len(mask)))
            elif len(mask) > self.num_human_points:
                mask = mask[:self.num_human_points]
            return mask
        return np.zeros(self.num_human_points, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.sample_ids[idx]

        rgb_image = self._load_image(sample_id)
        human_mask = self._load_mask(sample_id)

        sample = {
            'sample_id': sample_id,
            'data_type': 'human',
            'rgb_image': rgb_image,
            'human_mask_gt': human_mask,
            'human_mask_valid': True,
            'object_mask_valid': False,
            'object_points': None,
            'object_mask_gt': None
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class ObjectAffordanceDataset(Dataset):
    """
    Dataset for object affordance annotations.

    Expected directory structure:
        object_data_root/
        ├── images/
        │   ├── sample1.jpg
        │   └── sample2.jpg
        ├── point_clouds/
        │   ├── sample1.ply  # or .npy
        │   └── sample2.ply
        ├── masks/
        │   ├── sample1.npy  # (N,) binary mask
        │   └── sample2.npy
        └── splits/
            ├── train.txt
            └── val.txt
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform=None,
        num_object_points: int = 1024
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.num_object_points = num_object_points

        self.image_dir = self.data_root / 'images'
        self.pc_dir = self.data_root / 'point_clouds'
        self.mask_dir = self.data_root / 'masks'

        self.sample_ids = self._load_split()

    def _load_split(self) -> List[str]:
        """Load sample IDs for the split."""
        split_file = self.data_root / 'splits' / f'{self.split}.txt'

        if split_file.exists():
            with open(split_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            if self.mask_dir.exists():
                return [p.stem for p in self.mask_dir.glob('*.npy')]
            return []

    def _load_image(self, sample_id: str) -> np.ndarray:
        """Load RGB image."""
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = self.image_dir / f'{sample_id}{ext}'
            if img_path.exists():
                image = Image.open(img_path).convert('RGB')
                return np.array(image)
        raise FileNotFoundError(f"Image not found for {sample_id}")

    def _load_point_cloud(self, sample_id: str) -> np.ndarray:
        """Load object point cloud."""
        # Try .ply first
        ply_path = self.pc_dir / f'{sample_id}.ply'
        if ply_path.exists():
            mesh = trimesh.load(ply_path)
            points = np.array(mesh.vertices, dtype=np.float32)
        else:
            # Try .npy
            npy_path = self.pc_dir / f'{sample_id}.npy'
            if npy_path.exists():
                points = np.load(npy_path).astype(np.float32)
            else:
                return np.zeros((self.num_object_points, 3), dtype=np.float32)

        # Subsample or pad to num_object_points
        if len(points) > self.num_object_points:
            indices = np.random.choice(len(points), self.num_object_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_object_points:
            padding = np.zeros((self.num_object_points - len(points), 3), dtype=np.float32)
            points = np.concatenate([points, padding], axis=0)

        return points

    def _load_mask(self, sample_id: str, num_points: int) -> np.ndarray:
        """Load object contact mask."""
        mask_path = self.mask_dir / f'{sample_id}.npy'
        if mask_path.exists():
            mask = np.load(mask_path).astype(np.float32)
            # Align with point cloud size
            if len(mask) < num_points:
                mask = np.pad(mask, (0, num_points - len(mask)))
            elif len(mask) > num_points:
                mask = mask[:num_points]
            return mask
        return np.zeros(num_points, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.sample_ids[idx]

        rgb_image = self._load_image(sample_id)
        object_points = self._load_point_cloud(sample_id)
        object_mask = self._load_mask(sample_id, len(object_points))

        sample = {
            'sample_id': sample_id,
            'data_type': 'object',
            'rgb_image': rgb_image,
            'object_points': object_points,
            'object_mask_gt': object_mask,
            'object_mask_valid': True,
            'human_mask_valid': False,
            'human_mask_gt': None
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class InterCapAffordanceDataset(Dataset):
    """
    Dataset using existing InterCap format with contact masks.
    Uses the same structure as IVDDataset but focuses on affordance.
    """

    def __init__(
        self,
        intercap_data: str,
        annot_data: str,
        split: str = 'train',
        transform=None,
        num_object_points: int = 1024,
        num_human_points: int = 10475,
        load_human_mask: bool = True,
        load_object_mask: bool = True
    ):
        self.intercap_data = Path(intercap_data)
        self.annot_data = Path(annot_data)
        self.split = split
        self.transform = transform
        self.num_object_points = num_object_points
        self.num_human_points = num_human_points
        self.load_human_mask = load_human_mask
        self.load_object_mask = load_object_mask

        self.output_dir = self.intercap_data / 'Output'
        if not self.output_dir.exists():
            self.output_dir = self.intercap_data

        self.human_mask_dir = self.annot_data / 'human_mask'
        self.object_mask_dir = self.annot_data / 'object_mask'

        self.sample_ids = self._load_split()

    def _load_split(self) -> List[str]:
        """Load sample IDs."""
        split_file = self.intercap_data / 'splits' / f'{self.split}.txt'

        if split_file.exists():
            with open(split_file, 'r') as f:
                sample_ids = [line.strip() for line in f if line.strip()]
        else:
            if self.annot_data.exists():
                sample_ids = [p.stem for p in self.annot_data.glob('*.json')]
            else:
                sample_ids = []

        # Filter by available masks
        valid_ids = []
        for sid in sample_ids:
            has_human = (self.human_mask_dir / f'{sid}.npy').exists() if self.load_human_mask else False
            has_object = (self.object_mask_dir / f'{sid}.npy').exists() if self.load_object_mask else False
            if has_human or has_object:
                valid_ids.append(sid)

        return sorted(valid_ids)

    def _get_sample_dir(self, sample_id: str) -> Path:
        return self.output_dir / sample_id

    def _load_image(self, sample_id: str) -> np.ndarray:
        sample_dir = self._get_sample_dir(sample_id)
        img_path = sample_dir / 'image.jpg'
        if img_path.exists():
            image = Image.open(img_path).convert('RGB')
            return np.array(image)
        raise FileNotFoundError(f"Image not found for {sample_id}")

    def _load_object_points(self, sample_id: str) -> np.ndarray:
        sample_dir = self._get_sample_dir(sample_id)
        obj_path = sample_dir / 'obj_pcd_origin.ply'

        if obj_path.exists():
            mesh = trimesh.load(obj_path)
            points = np.array(mesh.vertices, dtype=np.float32)

            if len(points) > self.num_object_points:
                indices = np.random.choice(len(points), self.num_object_points, replace=False)
                points = points[indices]

            return points
        return np.zeros((self.num_object_points, 3), dtype=np.float32)

    def _load_masks(self, sample_id: str, num_obj_points: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        human_mask = None
        object_mask = None

        if self.load_human_mask:
            human_path = self.human_mask_dir / f'{sample_id}.npy'
            if human_path.exists():
                human_mask = np.load(human_path).astype(np.float32)
                if len(human_mask) < self.num_human_points:
                    human_mask = np.pad(human_mask, (0, self.num_human_points - len(human_mask)))
                elif len(human_mask) > self.num_human_points:
                    human_mask = human_mask[:self.num_human_points]

        if self.load_object_mask:
            object_path = self.object_mask_dir / f'{sample_id}.npy'
            if object_path.exists():
                object_mask = np.load(object_path).astype(np.float32)
                if len(object_mask) < num_obj_points:
                    object_mask = np.pad(object_mask, (0, num_obj_points - len(object_mask)))
                elif len(object_mask) > num_obj_points:
                    object_mask = object_mask[:num_obj_points]

        return human_mask, object_mask

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.sample_ids[idx]

        rgb_image = self._load_image(sample_id)
        object_points = self._load_object_points(sample_id)
        human_mask, object_mask = self._load_masks(sample_id, len(object_points))

        sample = {
            'sample_id': sample_id,
            'data_type': 'mixed',
            'rgb_image': rgb_image,
            'object_points': object_points,
            'human_mask_gt': human_mask if human_mask is not None else np.zeros(self.num_human_points, dtype=np.float32),
            'human_mask_valid': human_mask is not None,
            'object_mask_gt': object_mask if object_mask is not None else np.zeros(len(object_points), dtype=np.float32),
            'object_mask_valid': object_mask is not None
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class AffordanceTransform:
    """Transform for affordance dataset."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample: Dict) -> Dict:
        # Process RGB image
        if 'rgb_image' in sample and sample['rgb_image'] is not None:
            image = sample['rgb_image']

            # Resize
            image = Image.fromarray(image).resize(
                (self.image_size, self.image_size),
                Image.BILINEAR
            )
            image = np.array(image).astype(np.float32) / 255.0

            # Normalize
            image = (image - self.mean) / self.std

            # HWC -> CHW
            image = image.transpose(2, 0, 1)
            sample['rgb_image'] = torch.from_numpy(image).float()

        # Convert masks to tensors
        if sample.get('human_mask_gt') is not None:
            sample['human_mask_gt'] = torch.from_numpy(sample['human_mask_gt']).float()

        if sample.get('object_mask_gt') is not None:
            sample['object_mask_gt'] = torch.from_numpy(sample['object_mask_gt']).float()

        # Convert object points to tensor
        if sample.get('object_points') is not None:
            sample['object_points'] = torch.from_numpy(sample['object_points']).float()

        # Convert valid flags to tensors
        sample['human_mask_valid'] = torch.tensor(sample['human_mask_valid']).bool()
        sample['object_mask_valid'] = torch.tensor(sample['object_mask_valid']).bool()

        return sample


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for mixed batch."""
    result = {
        'sample_id': [s['sample_id'] for s in batch],
        'data_type': [s['data_type'] for s in batch]
    }

    # Stack RGB images
    result['rgb_image'] = torch.stack([s['rgb_image'] for s in batch])

    # Stack valid flags
    result['human_mask_valid'] = torch.stack([s['human_mask_valid'] for s in batch])
    result['object_mask_valid'] = torch.stack([s['object_mask_valid'] for s in batch])

    # Stack human masks (handle None values for samples without human annotations)
    human_masks = []
    has_any_human = any(s.get('human_mask_gt') is not None for s in batch)
    if has_any_human:
        # Get the size from a valid sample
        human_size = next(s['human_mask_gt'].shape[0] for s in batch if s.get('human_mask_gt') is not None)
        for s in batch:
            if s.get('human_mask_gt') is not None:
                human_masks.append(s['human_mask_gt'])
            else:
                # Fill with zeros for samples without human annotation
                human_masks.append(torch.zeros(human_size))
        result['human_mask_gt'] = torch.stack(human_masks)

    # Stack object points and masks (need to handle variable sizes)
    has_any_object = any(s.get('object_points') is not None for s in batch)
    if has_any_object:
        # Pad to max size in batch
        valid_pts = [s['object_points'] for s in batch if s.get('object_points') is not None]
        max_pts = max(p.shape[0] for p in valid_pts)

        padded_points = []
        padded_masks = []

        for s in batch:
            pts = s.get('object_points')
            if pts is not None:
                if pts.shape[0] < max_pts:
                    pad = torch.zeros(max_pts - pts.shape[0], 3)
                    pts = torch.cat([pts, pad], dim=0)
                padded_points.append(pts)

                mask = s.get('object_mask_gt')
                if mask is not None:
                    if mask.shape[0] < max_pts:
                        mask = torch.cat([mask, torch.zeros(max_pts - mask.shape[0])], dim=0)
                    padded_masks.append(mask)
                else:
                    padded_masks.append(torch.zeros(max_pts))
            else:
                padded_points.append(torch.zeros(max_pts, 3))
                padded_masks.append(torch.zeros(max_pts))

        result['object_points'] = torch.stack(padded_points)
        result['object_mask_gt'] = torch.stack(padded_masks)

    return result


def create_affordance_dataloaders(
    human_data_root: Optional[str] = None,
    object_data_root: Optional[str] = None,
    intercap_data: Optional[str] = None,
    annot_data: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 224,
    num_object_points: int = 1024,
    num_human_points: int = 10475
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Can use:
    1. Separate human/object data roots
    2. InterCap format with masks
    3. Any combination
    """
    transform = AffordanceTransform(image_size=image_size)

    train_datasets = []
    val_datasets = []

    # Add human dataset if provided
    if human_data_root and Path(human_data_root).exists():
        train_datasets.append(HumanAffordanceDataset(
            human_data_root, split='train', transform=transform,
            num_human_points=num_human_points
        ))
        val_datasets.append(HumanAffordanceDataset(
            human_data_root, split='val', transform=transform,
            num_human_points=num_human_points
        ))

    # Add object dataset if provided
    if object_data_root and Path(object_data_root).exists():
        train_datasets.append(ObjectAffordanceDataset(
            object_data_root, split='train', transform=transform,
            num_object_points=num_object_points
        ))
        val_datasets.append(ObjectAffordanceDataset(
            object_data_root, split='val', transform=transform,
            num_object_points=num_object_points
        ))

    # Add InterCap dataset if provided
    if intercap_data and annot_data:
        train_datasets.append(InterCapAffordanceDataset(
            intercap_data, annot_data, split='train', transform=transform,
            num_object_points=num_object_points, num_human_points=num_human_points
        ))
        val_datasets.append(InterCapAffordanceDataset(
            intercap_data, annot_data, split='val', transform=transform,
            num_object_points=num_object_points, num_human_points=num_human_points
        ))

    # Combine datasets
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    elif len(train_datasets) == 1:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
    else:
        raise ValueError("No valid data sources provided")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader
