"""
DAMON/DECO Dataset Adapter for Human Affordance Pre-training

This module provides data loading for the DAMON dataset (HOT-DCA) which contains:
- RGB images
- SMPLX contact labels (10475 vertices)
- SMPL/SMPLX body parameters

Reference: DECO: Dense Estimation of 3D Human-Scene Contact In The Wild
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image


class DAMONDataset(Dataset):
    """
    Dataset for DAMON/DECO human contact annotations.

    Expected data structure:
        damon_root/
        ├── Release_Datasets/
        │   └── damon/
        │       ├── hot_dca_trainval.npz
        │       └── hot_dca_test.npz
        └── HOT-Annotated/
            └── images/
                └── *.jpg

    NPZ file contains:
        - imgname: (N,) image paths (relative)
        - contact_label_smplx: (N, 10475) contact labels for SMPLX vertices
        - pose: (N, 72) SMPL pose parameters
        - shape: (N, 10) SMPL shape parameters
        - transl: (N, 3) translation
    """

    def __init__(
        self,
        damon_root: str,
        split: str = 'train',
        transform=None,
        num_human_points: int = 10475,
        contact_threshold: float = 0.5,
        image_size: int = 224
    ):
        """
        Initialize DAMON dataset.

        Args:
            damon_root: Root directory of DAMON dataset
            split: 'train' or 'val'/'test'
            transform: Optional transform
            num_human_points: Number of human vertices (10475 for SMPLX)
            contact_threshold: Threshold to binarize contact labels
            image_size: Target image size
        """
        self.damon_root = Path(damon_root)
        self.split = split
        self.transform = transform
        self.num_human_points = num_human_points
        self.contact_threshold = contact_threshold
        self.image_size = image_size

        # Locate npz file
        if split == 'train':
            npz_path = self.damon_root / 'Release_Datasets' / 'damon' / 'hot_dca_trainval.npz'
        else:
            npz_path = self.damon_root / 'Release_Datasets' / 'damon' / 'hot_dca_test.npz'

        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        # Load data
        print(f"Loading DAMON dataset from: {npz_path}")
        self.data = np.load(npz_path, allow_pickle=True)

        self.imgnames = self.data['imgname']
        self.contact_labels = self.data['contact_label_smplx']  # (N, 10475)

        # Image root directory
        self.image_root = self.damon_root / 'HOT-Annotated' / 'images'

        # Validate and filter samples with existing images
        self.valid_indices = self._validate_samples()

        print(f"DAMON {split} split: {len(self.valid_indices)} valid samples "
              f"(out of {len(self.imgnames)} total)")

    def _validate_samples(self) -> List[int]:
        """Filter samples with existing images."""
        valid = []
        for i, imgname in enumerate(self.imgnames):
            # Extract filename from relative path
            # e.g., "datasets/HOT-Annotated/images/vcoco_000000000589.jpg" -> "vcoco_000000000589.jpg"
            filename = Path(imgname).name
            img_path = self.image_root / filename

            if img_path.exists():
                valid.append(i)

        return valid

    def _get_image_path(self, imgname: str) -> Path:
        """Get actual image path from relative path in npz."""
        filename = Path(imgname).name
        return self.image_root / filename

    def _load_image(self, img_path: Path) -> np.ndarray:
        """Load and preprocess image."""
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(image)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict:
        data_idx = self.valid_indices[idx]

        # Load image
        imgname = self.imgnames[data_idx]
        img_path = self._get_image_path(imgname)
        rgb_image = self._load_image(img_path)

        # Get contact label
        contact_label = self.contact_labels[data_idx].astype(np.float32)

        # Ensure correct size
        if len(contact_label) < self.num_human_points:
            contact_label = np.pad(
                contact_label,
                (0, self.num_human_points - len(contact_label))
            )
        elif len(contact_label) > self.num_human_points:
            contact_label = contact_label[:self.num_human_points]

        # Binarize if needed (labels are already in [0, 1])
        # For training, we can use soft labels or binarize
        human_mask = (contact_label > self.contact_threshold).astype(np.float32)

        sample = {
            'sample_id': Path(imgname).stem,
            'data_type': 'human',
            'rgb_image': rgb_image,
            'human_mask_gt': human_mask,
            'human_mask_soft': contact_label,  # Keep soft labels for optional use
            'human_mask_valid': True,
            'object_mask_valid': False,
            'object_points': None,
            'object_mask_gt': None
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class DAMONTransform:
    """Transform for DAMON dataset."""

    def __init__(self, image_size: int = 224, normalize: bool = True):
        self.image_size = image_size
        self.normalize = normalize
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample: Dict) -> Dict:
        # Process RGB image
        if 'rgb_image' in sample and sample['rgb_image'] is not None:
            image = sample['rgb_image'].astype(np.float32) / 255.0

            if self.normalize:
                image = (image - self.mean) / self.std

            # HWC -> CHW
            image = image.transpose(2, 0, 1)
            sample['rgb_image'] = torch.from_numpy(image).float()

        # Convert masks to tensors
        if sample.get('human_mask_gt') is not None:
            sample['human_mask_gt'] = torch.from_numpy(
                sample['human_mask_gt']
            ).float()

        if sample.get('human_mask_soft') is not None:
            sample['human_mask_soft'] = torch.from_numpy(
                sample['human_mask_soft']
            ).float()

        # Convert valid flags
        sample['human_mask_valid'] = torch.tensor(sample['human_mask_valid']).bool()
        sample['object_mask_valid'] = torch.tensor(sample['object_mask_valid']).bool()

        return sample


def damon_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DAMON dataset."""
    result = {
        'sample_id': [s['sample_id'] for s in batch],
        'data_type': [s['data_type'] for s in batch]
    }

    # Stack tensors
    result['rgb_image'] = torch.stack([s['rgb_image'] for s in batch])
    result['human_mask_gt'] = torch.stack([s['human_mask_gt'] for s in batch])
    result['human_mask_valid'] = torch.stack([s['human_mask_valid'] for s in batch])
    result['object_mask_valid'] = torch.stack([s['object_mask_valid'] for s in batch])

    if batch[0].get('human_mask_soft') is not None:
        result['human_mask_soft'] = torch.stack([s['human_mask_soft'] for s in batch])

    return result


def create_damon_dataloaders(
    damon_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 224,
    num_human_points: int = 10475,
    contact_threshold: float = 0.5
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DAMON train and validation dataloaders.

    Args:
        damon_root: Root directory of DAMON dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        num_human_points: Number of human vertices
        contact_threshold: Threshold for binarizing contact labels

    Returns:
        Tuple of (train_loader, val_loader)
    """
    transform = DAMONTransform(image_size=image_size)

    train_dataset = DAMONDataset(
        damon_root=damon_root,
        split='train',
        transform=transform,
        num_human_points=num_human_points,
        contact_threshold=contact_threshold,
        image_size=image_size
    )

    val_dataset = DAMONDataset(
        damon_root=damon_root,
        split='val',
        transform=transform,
        num_human_points=num_human_points,
        contact_threshold=contact_threshold,
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=damon_collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=damon_collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--damon_root', type=str, required=True)
    args = parser.parse_args()

    train_loader, val_loader = create_damon_dataloaders(
        damon_root=args.damon_root,
        batch_size=4,
        num_workers=0
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v)}")

    # Check contact label distribution
    mask = batch['human_mask_gt']
    print(f"\nContact stats in batch:")
    print(f"  Contact ratio: {mask.mean():.4f}")
    print(f"  Per-sample contacts: {mask.sum(dim=1).tolist()}")
