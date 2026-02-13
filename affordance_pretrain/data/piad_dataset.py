"""
PIAD (Point-Image Affordance Dataset) Adapter for Object Affordance Pre-training

This module provides data loading for the PIAD dataset which contains:
- Interaction images (human-object interaction scenes)
- Object point clouds with 17-class affordance annotations
- Bounding boxes for subject and object

Reference: Grounding 3D Object Affordance from 2D Interactions in Images (ICCV 2023)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image


# 17 affordance categories in PIAD
AFFORDANCE_CLASSES = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab'
]


def pc_normalize(pc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


class PIADDataset(Dataset):
    """
    Dataset for PIAD object affordance annotations.

    Expected data structure:
        piad_root/
        ├── Seen/
        │   ├── Img/
        │   │   ├── Train/
        │   │   │   └── ObjectType/AffordanceType/*.jpg
        │   │   └── Test/
        │   ├── Point/
        │   │   ├── Train/
        │   │   │   └── ObjectType/*.txt
        │   │   └── Test/
        │   ├── Bounding_Box/
        │   │   ├── Train/
        │   │   └── Test/
        │   ├── Img_Train.txt
        │   ├── Img_Test.txt
        │   ├── Point_Train.txt
        │   ├── Point_Test.txt
        │   ├── Box_Train.txt
        │   └── Box_Test.txt
        └── Unseen/
            └── ...

    Point cloud file format:
        Each line: uuid ObjectType x y z label1 label2 ... label17
        - 2048 points per file
        - 17 affordance labels per point
    """

    def __init__(
        self,
        piad_root: str,
        setting: str = 'Seen',  # 'Seen' or 'Unseen'
        split: str = 'train',   # 'train' or 'test'
        transform=None,
        num_object_points: int = 2048,
        image_size: int = 224,
        use_all_affordances: bool = True,  # Use all 17 affordances or just the one from image
        normalize_pc: bool = True
    ):
        """
        Initialize PIAD dataset.

        Args:
            piad_root: Root directory of PIAD dataset
            setting: 'Seen' or 'Unseen' split
            split: 'train' or 'test'
            transform: Optional transform
            num_object_points: Number of points to use (2048 default)
            image_size: Target image size
            use_all_affordances: If True, use all 17 affordance labels;
                                 If False, use only the affordance from image filename
            normalize_pc: Whether to normalize point cloud
        """
        self.piad_root = Path(piad_root)
        self.setting = setting
        self.split = split
        self.transform = transform
        self.num_object_points = num_object_points
        self.image_size = image_size
        self.use_all_affordances = use_all_affordances
        self.normalize_pc = normalize_pc

        self.data_dir = self.piad_root / setting

        # Load file lists
        split_suffix = 'Train' if split == 'train' else 'Test'
        self.img_list = self._load_file_list(self.data_dir / f'Img_{split_suffix}.txt')
        self.point_list = self._load_file_list(self.data_dir / f'Point_{split_suffix}.txt')
        self.box_list = self._load_file_list(self.data_dir / f'Box_{split_suffix}.txt')

        # Build object-wise index for training (match points to images by object type)
        if split == 'train':
            self.object_to_points = self._build_object_index()

        print(f"PIAD {setting}/{split}: {len(self.img_list)} images, {len(self.point_list)} point clouds")

    def _load_file_list(self, filepath: Path) -> List[str]:
        """Load file list from txt file."""
        with open(filepath, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        return files

    def _build_object_index(self) -> Dict[str, List[int]]:
        """Build index mapping object types to point cloud indices."""
        object_to_points = {}
        for idx, point_path in enumerate(self.point_list):
            # Extract object type from path: .../Point_Train_ObjectType_N.txt
            obj_type = Path(point_path).stem.split('_')[-2]
            if obj_type not in object_to_points:
                object_to_points[obj_type] = []
            object_to_points[obj_type].append(idx)
        return object_to_points

    def _get_actual_path(self, relative_path: str) -> Path:
        """Convert relative path in txt to actual path."""
        # Paths in txt are like: Data/Seen/Img/Train/...
        # We need to map to: piad_root/Seen/Img/Train/...
        parts = Path(relative_path).parts
        if parts[0] == 'Data':
            # Remove 'Data' prefix and join with piad_root
            actual_path = self.piad_root / '/'.join(parts[1:])
        else:
            actual_path = self.piad_root / relative_path
        return actual_path

    def _load_image(self, img_path: Path) -> np.ndarray:
        """Load and preprocess image."""
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(image)

    def _load_point_cloud(self, point_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load point cloud and affordance labels.

        Returns:
            points: (N, 3) xyz coordinates
            affordance_labels: (N, 17) affordance labels for each point
        """
        coordinates = []
        with open(point_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # Format: uuid ObjectType x y z label1...label17
                data = [float(x) for x in parts[2:]]
                coordinates.append(data)

        data_array = np.array(coordinates, dtype=np.float32)
        points = data_array[:, 0:3]  # (N, 3)
        affordance_labels = data_array[:, 3:]  # (N, 17)

        # Normalize point cloud if needed
        if self.normalize_pc:
            points, _, _ = pc_normalize(points)

        return points, affordance_labels

    def _get_affordance_index(self, img_path: str) -> int:
        """Extract affordance index from image path."""
        # Path format: .../Img_Train_ObjectType_AffordanceType_N.jpg
        filename = Path(img_path).stem
        parts = filename.split('_')
        affordance_type = parts[-2]  # e.g., 'grasp'
        return AFFORDANCE_CLASSES.index(affordance_type)

    def _get_object_type(self, img_path: str) -> str:
        """Extract object type from image path."""
        filename = Path(img_path).stem
        parts = filename.split('_')
        return parts[-3]  # e.g., 'Chair'

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int) -> Dict:
        # Load image
        img_rel_path = self.img_list[idx]
        img_path = self._get_actual_path(img_rel_path)
        rgb_image = self._load_image(img_path)

        # Get affordance index from image filename
        affordance_idx = self._get_affordance_index(img_rel_path)
        object_type = self._get_object_type(img_rel_path)

        # Load point cloud
        if self.split == 'train':
            # For training, sample a point cloud of the same object type
            point_indices = self.object_to_points.get(object_type, [])
            if point_indices:
                point_idx = np.random.choice(point_indices)
            else:
                point_idx = idx % len(self.point_list)
        else:
            # For testing, use corresponding point cloud
            point_idx = idx % len(self.point_list)

        point_rel_path = self.point_list[point_idx]
        point_path = self._get_actual_path(point_rel_path)
        points, affordance_labels = self._load_point_cloud(point_path)

        # Handle point count
        if len(points) > self.num_object_points:
            indices = np.random.choice(len(points), self.num_object_points, replace=False)
            points = points[indices]
            affordance_labels = affordance_labels[indices]
        elif len(points) < self.num_object_points:
            # Pad with zeros
            pad_size = self.num_object_points - len(points)
            points = np.pad(points, ((0, pad_size), (0, 0)))
            affordance_labels = np.pad(affordance_labels, ((0, pad_size), (0, 0)))

        # Create affordance mask
        if self.use_all_affordances:
            # Use all 17 affordance labels (multi-label)
            # Combine into single mask: any affordance > 0.5
            object_mask = (affordance_labels.max(axis=1) > 0.5).astype(np.float32)
        else:
            # Use only the affordance corresponding to the image
            object_mask = (affordance_labels[:, affordance_idx] > 0.5).astype(np.float32)

        sample = {
            'sample_id': Path(img_rel_path).stem,
            'data_type': 'object',
            'rgb_image': rgb_image,
            'object_points': points.astype(np.float32),
            'object_mask_gt': object_mask,
            'object_mask_valid': True,
            'human_mask_valid': False,
            'human_mask_gt': None,
            'affordance_labels_full': affordance_labels.astype(np.float32),  # Keep full labels
            'affordance_index': affordance_idx,
            'object_type': object_type
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class PIADTransform:
    """Transform for PIAD dataset."""

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

        # Convert point cloud to tensor
        if sample.get('object_points') is not None:
            sample['object_points'] = torch.from_numpy(sample['object_points']).float()

        # Convert masks to tensors
        if sample.get('object_mask_gt') is not None:
            sample['object_mask_gt'] = torch.from_numpy(sample['object_mask_gt']).float()

        if sample.get('affordance_labels_full') is not None:
            sample['affordance_labels_full'] = torch.from_numpy(
                sample['affordance_labels_full']
            ).float()

        # Convert valid flags
        sample['human_mask_valid'] = torch.tensor(sample['human_mask_valid']).bool()
        sample['object_mask_valid'] = torch.tensor(sample['object_mask_valid']).bool()

        return sample


def piad_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for PIAD dataset."""
    result = {
        'sample_id': [s['sample_id'] for s in batch],
        'data_type': [s['data_type'] for s in batch],
        'object_type': [s['object_type'] for s in batch],
        'affordance_index': torch.tensor([s['affordance_index'] for s in batch])
    }

    # Stack tensors
    result['rgb_image'] = torch.stack([s['rgb_image'] for s in batch])
    result['object_points'] = torch.stack([s['object_points'] for s in batch])
    result['object_mask_gt'] = torch.stack([s['object_mask_gt'] for s in batch])
    result['human_mask_valid'] = torch.stack([s['human_mask_valid'] for s in batch])
    result['object_mask_valid'] = torch.stack([s['object_mask_valid'] for s in batch])

    if batch[0].get('affordance_labels_full') is not None:
        result['affordance_labels_full'] = torch.stack(
            [s['affordance_labels_full'] for s in batch]
        )

    return result


def create_piad_dataloaders(
    piad_root: str,
    setting: str = 'Seen',
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 224,
    num_object_points: int = 2048,
    use_all_affordances: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PIAD train and test dataloaders.

    Args:
        piad_root: Root directory of PIAD dataset
        setting: 'Seen' or 'Unseen'
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        num_object_points: Number of points per object
        use_all_affordances: Whether to use all 17 affordance labels

    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = PIADTransform(image_size=image_size)

    train_dataset = PIADDataset(
        piad_root=piad_root,
        setting=setting,
        split='train',
        transform=transform,
        num_object_points=num_object_points,
        image_size=image_size,
        use_all_affordances=use_all_affordances
    )

    test_dataset = PIADDataset(
        piad_root=piad_root,
        setting=setting,
        split='test',
        transform=transform,
        num_object_points=num_object_points,
        image_size=image_size,
        use_all_affordances=use_all_affordances
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=piad_collate_fn,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=piad_collate_fn,
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--piad_root', type=str, required=True)
    parser.add_argument('--setting', type=str, default='Seen')
    args = parser.parse_args()

    train_loader, test_loader = create_piad_dataloaders(
        piad_root=args.piad_root,
        setting=args.setting,
        batch_size=4,
        num_workers=0
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v).__name__}")

    # Check affordance stats
    mask = batch['object_mask_gt']
    print(f"\nAffordance stats in batch:")
    print(f"  Contact ratio: {mask.mean():.4f}")
    print(f"  Per-sample contacts: {mask.sum(dim=1).int().tolist()}")
    print(f"  Object types: {batch['object_type']}")
    print(f"  Affordance indices: {batch['affordance_index'].tolist()}")
