"""
Dataset for InterActVLM-Discrete (IVD)
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from PIL import Image
import trimesh
import smplx
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.keypoints import KeypointManager
from utils.renderer import MultiViewRenderer
from .transforms import IVDTransform, get_train_transforms, get_val_transforms


class IVDDataset(Dataset):    
    def __init__(
        self,
        intercap_data: str,
        annot_data: str,
        split: str = 'train',
        transform: Optional[IVDTransform] = None,
        keypoint_manager: Optional[KeypointManager] = None,
        renderer: Optional[MultiViewRenderer] = None,
        num_views: int = 4,
        render_on_fly: bool = True,
        num_object_points: int = 1024,
        num_object_queries: int = 87,  # K object contact queries (matches num_body_points)
        cache_renders: bool = False,
        render_size: int = 256,
        load_contact_masks: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            intercap_data: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Data transforms
            keypoint_manager: Keypoint manager for 87 body points (SMPL-X)
            renderer: Multi-view renderer
            num_views: Number of rendered views
            render_on_fly: Whether to render views on-the-fly
            num_object_points: Number of points in object point cloud
            num_object_queries: Number of object contact queries (K)
            cache_renders: Whether to cache rendered views
            render_size: Render size for fallback projections
            load_contact_masks: Whether to load per-point contact masks
        """
        self.intercap_data = Path(intercap_data)
        self.annot_data = Path(annot_data)
        self.split = split
        self.transform = transform
        self.keypoint_manager = keypoint_manager
        self.renderer = renderer
        self.num_views = num_views
        self.render_on_fly = render_on_fly
        self.num_object_points = num_object_points
        self.num_object_queries = num_object_queries
        self.cache_renders = cache_renders
        self.render_size = render_size
        self.load_contact_masks = load_contact_masks
        candidate_output = self.intercap_data / 'Output'
        self.output_dir = candidate_output if candidate_output.exists() else self.intercap_data
        self.human_mask_dir = self.annot_data / 'human_mask'
        self.object_mask_dir = self.annot_data / 'object_mask'

        
        # Load sample IDs
        self.sample_ids = self._load_split()
        
        # Render cache
        self.render_cache = {} if cache_renders else None
        self.model_path = "smpl_models/SMPLX_NEUTRAL.npz"
        self.smplx_model = smplx.create(
            self.model_path,
            model_type='smplx',
            gender='neutral',
            use_pca=False,
            num_betas=10
        )
        
    
    def _get_sample_dir(self, sample_id: str) -> Path:
        if self.output_dir.exists():
            return self.output_dir / sample_id
        return self.intercap_data / sample_id

    def _load_split(self) -> List[str]:
        """Load sample IDs for the current split."""
        split_file = self.intercap_data / 'splits' / f'{self.split}.txt'
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                sample_ids = [line.strip() for line in f if line.strip()]
        else:
            # Fallback: prefer annotation ids, then output folder names
            if self.annot_data.exists():
                sample_ids = [p.stem for p in self.annot_data.glob('*.json')]
            elif self.output_dir.exists():
                sample_ids = [p.name for p in self.output_dir.iterdir() if p.is_dir()]
            else:
                sample_ids = []

        if self.output_dir.exists():
            sample_ids = [
                sid for sid in sample_ids
                if (self.output_dir / sid).exists()
            ]
        
        return sorted(sample_ids)
    
    def _load_image(self, sample_id: str) -> np.ndarray:
        """Load RGB image."""
        sample_dir = self._get_sample_dir(sample_id)
        img_path = sample_dir / 'image.jpg'
        if img_path.exists():
            image = Image.open(img_path).convert('RGB')
            return np.array(image)

        raise FileNotFoundError(f"Image not found for sample {sample_id}")

    def _load_human_data(self, sample_id: str) -> Dict:
        """Load human mesh data."""
        sample_dir = self._get_sample_dir(sample_id)
        human_path = sample_dir / 'smplx_parameters_new.json'
        with open(human_path, 'r') as f:
            smplx_params = json.load(f)

        body_pose = torch.tensor(smplx_params['body_pose'], dtype=torch.float32).reshape(1, -1)
        root_pose = torch.tensor(smplx_params['root_pose'], dtype=torch.float32).reshape(1, 3)
        lhand_pose = torch.tensor(smplx_params['lhand_pose'], dtype=torch.float32).reshape(1, -1)
        rhand_pose = torch.tensor(smplx_params['rhand_pose'], dtype=torch.float32).reshape(1, -1)
        
        betas = torch.zeros(1, 10, dtype=torch.float32)
        if 'shape' in smplx_params:
            shape = smplx_params['shape'][:10] if len(smplx_params['shape']) >= 10 else smplx_params['shape']
            betas[0, :len(shape)] = torch.tensor(shape, dtype=torch.float32)
        
        transl = torch.tensor(smplx_params.get('cam_trans', [0, 0, 0]), dtype=torch.float32).reshape(1, 3)
        
        # 前向传播
        with torch.no_grad():
            output = self.smplx_model(
                global_orient=root_pose,
                body_pose=body_pose,
                left_hand_pose=lhand_pose,
                right_hand_pose=rhand_pose,
                betas=betas,
                transl=transl
            )
        
        vertices = output.vertices.numpy().squeeze(0)
        if vertices is not None:
            return {
                'vertices': vertices,  # (V, 3)
                'faces': np.array(self.smplx_model.faces, dtype=np.int32),  # (F, 3)
            }
        else:
            # Return dummy data
            return {
                'vertices': np.zeros((10475, 3), dtype=np.float32),
                'faces': None,
            }
    
    def _load_object_data(self, sample_id: str) -> Dict:
        """Load object point cloud data."""
        sample_dir = self._get_sample_dir(sample_id)
        object_path = sample_dir / 'obj_template.ply'
        
        mesh = trimesh.load(object_path)    
        points = np.array(mesh.vertices, dtype=np.float32)  # (N, 3)
        centroid = points.mean(axis=0).astype(np.float32) if len(points) > 0 else np.zeros(3, dtype=np.float32)
        orig_len = len(points)
        indices = None
        
        # Subsample if needed
        if len(points) > self.num_object_points:
            indices = np.random.choice(
                len(points), self.num_object_points, replace=False
            )
            points = points[indices]
        elif len(points) < self.num_object_points:
            pad = np.zeros((self.num_object_points - len(points), 3), dtype=np.float32)
            points = np.concatenate([points, pad], axis=0)
        
        # Normalize: center + scale by max radius
        points = (points - centroid).astype(np.float32)
        max_radius = 1.0
        if len(points) > 0:
            max_radius = np.linalg.norm(points, axis=1).max()
            if max_radius > 0:
                points = points / max_radius

        return {
            'points': points,
            'indices': indices,
            'orig_len': orig_len,
            'centroid': centroid,
            'max_radius': float(max_radius),
        }
    
    def _load_annotation(self, sample_id: str) -> Dict:
        """Load contact annotations."""
        ann_path = self.annot_data / f'{sample_id}.json'
        
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            return ann
        else:
            # Return default annotations
            return {
                'human_contact_labels': np.zeros(87, dtype=np.float32).tolist(),
                'object_contact_coords': np.zeros(
                    (self.num_object_queries, 3), dtype=np.float32
                ).tolist()
            }

    def _load_contact_masks(
        self,
        sample_id: str,
        human_vertices: np.ndarray,
        object_points: np.ndarray,
        object_indices: Optional[np.ndarray] = None,
        object_orig_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load human/object contact masks if available."""
        human_mask = None
        object_mask = None

        human_path = self.human_mask_dir / f'{sample_id}.npy'
        human_mask = np.load(human_path)
        object_path = self.object_mask_dir / f'{sample_id}.npy'
        object_mask = np.load(object_path)
        if object_indices is not None:
            # Align mask to the same subsampling indices used for object_points.
            if object_mask.shape[0] != object_points.shape[0]:
                object_mask = object_mask[object_indices]
        # Pad or truncate mask to match object_points length
        if object_mask.shape[0] < object_points.shape[0]:
            pad = np.zeros((object_points.shape[0] - object_mask.shape[0],), dtype=object_mask.dtype)
            object_mask = np.concatenate([object_mask, pad], axis=0)
        elif object_mask.shape[0] > object_points.shape[0]:
            object_mask = object_mask[:object_points.shape[0]]
        return human_mask, object_mask
    
    def _load_or_generate_renders(
        self,
        sample_id: str,
        human_vertices: np.ndarray,
        human_faces: Optional[np.ndarray],
        object_points: np.ndarray
    ) -> torch.Tensor:
        """Load pre-rendered views or generate on-the-fly."""
        # Check cache first
        if self.cache_renders and sample_id in self.render_cache:
            return self.render_cache[sample_id]
        
        # Generate renders on-the-fly
        if self.renderer is not None and human_faces is not None:
            human_verts = torch.from_numpy(human_vertices).unsqueeze(0)
            faces = torch.from_numpy(human_faces)
            obj_pts = torch.from_numpy(object_points).unsqueeze(0)
            
            with torch.no_grad():
                render_dict = self.renderer(human_verts, faces, obj_pts)
            
            # Combine human and object renders
            human_renders = render_dict['human_renders']  # (J, 3, H, W)
            object_renders = render_dict['object_renders']  # (J, 3, H, W)
            
            # Concatenate along channel dimension or stack
            renders = torch.cat([human_renders, object_renders], dim=0)  # (2J, 3, H, W)
            
            if self.cache_renders:
                self.render_cache[sample_id] = renders
            
            return renders
        
        # Fallback: simple point cloud projection renders
        renders = self._render_pointcloud_views(object_points)
        if self.cache_renders:
            self.render_cache[sample_id] = renders
        return renders

    def _render_pointcloud_views(self, object_points: np.ndarray) -> torch.Tensor:
        """Render simple orthographic projections from point cloud."""
        h = w = int(self.render_size)
        views = []
        if object_points.shape[0] == 0:
            return torch.zeros(self.num_views * 2, 3, h, w)

        pts = object_points.astype(np.float32)
        for view_idx in range(self.num_views):
            angle = (2.0 * np.pi * view_idx) / max(self.num_views, 1)
            rot = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            rotated = pts @ rot.T
            xy = rotated[:, :2]
            xy_min = xy.min(axis=0)
            xy_max = xy.max(axis=0)
            denom = np.maximum(xy_max - xy_min, 1e-6)
            norm_xy = (xy - xy_min) / denom
            pix = (norm_xy * (w - 1)).astype(np.int32)
            img = np.zeros((h, w), dtype=np.float32)
            img[pix[:, 1], pix[:, 0]] = 1.0
            img = np.stack([img, img, img], axis=0)  # (3, H, W)
            views.append(torch.from_numpy(img))

        object_renders = torch.stack(views, dim=0)
        human_renders = torch.zeros_like(object_renders)
        return torch.cat([human_renders, object_renders], dim=0)
    
    def _get_human_labels(
        self,
        human_data: Dict,
        annotation: Dict
    ) -> np.ndarray:
        """Get binary contact labels for 87 keypoints."""
        # Priority: annotation > vertex contacts > zeros
        if 'human_contact_labels' in annotation:
            labels = np.array(annotation['human_contact_labels'], dtype=np.float32)
        elif 'keypoint_object_pairs' in annotation:
            labels = np.zeros(87, dtype=np.float32)
            for pair in annotation.get('keypoint_object_pairs', []):
                idx = pair.get('keypoint_idx')
                if isinstance(idx, int) and 0 <= idx < 87:
                    labels[idx] = 1.0
        elif human_data.get('contacts') is not None and self.keypoint_manager is not None:
            # Convert vertex contacts to keypoint labels
            vertex_contacts = human_data['contacts']
            indices = self.keypoint_manager.get_vertex_indices()
            labels = vertex_contacts[indices].astype(np.float32)
        else:
            labels = np.zeros(87, dtype=np.float32)

        if labels.shape[0] < 87:
            pad = np.zeros(87 - labels.shape[0], dtype=np.float32)
            labels = np.concatenate([labels, pad], axis=0)
        elif labels.shape[0] > 87:
            labels = labels[:87]

        return labels
    
    def _get_object_coords(self, annotation: Dict) -> Tuple[np.ndarray, int]:
        """Get object contact coordinates."""
        coords_list = []
        if 'object_contact_coords' in annotation:
            coords_list = annotation['object_contact_coords']
        elif 'keypoint_object_pairs' in annotation:
            coords_list = [p.get('object_coord') for p in annotation.get('keypoint_object_pairs', [])]

        if coords_list:
            num_valid = len(coords_list)
            coords = np.array(coords_list, dtype=np.float32)
            if len(coords) < self.num_object_queries:
                padding = np.zeros((self.num_object_queries - len(coords), 3), dtype=np.float32)
                coords = np.concatenate([coords, padding], axis=0)
            elif len(coords) > self.num_object_queries:
                coords = coords[:self.num_object_queries]
                num_valid = self.num_object_queries
            return coords, num_valid
        return np.zeros((self.num_object_queries, 3), dtype=np.float32), 0
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a data sample.
        
        Returns:
            Dictionary containing:
                - sample_id: Sample identifier
                - rgb_image: (3, 224, 224) original scene image
                - render_views: (2J, 3, 256, 256) multi-view renders
                - human_vertices: (V, 3) human mesh vertices
                - object_points: (N, 3) object point cloud
                - human_labels: (87,) binary contact labels
                - object_coords: (K, 3) object contact coordinates
        """
        sample_id = self.sample_ids[idx]
        
        # Load data
        rgb_image = self._load_image(sample_id)
        human_data = self._load_human_data(sample_id)
        object_data = self._load_object_data(sample_id)
        annotation = self._load_annotation(sample_id)
        human_labels = self._get_human_labels(human_data, annotation)
        object_coords, object_coords_valid = self._get_object_coords(annotation)
        sample = {
            'sample_id': sample_id,
            'rgb_image': rgb_image,
            'human_vertices': human_data['vertices'],
            'object_points': object_data['points'],
            'human_labels': human_labels,
            'object_coords': object_coords
        }

        if self.load_contact_masks:
            human_mask, object_mask = self._load_contact_masks(
                sample_id,
                human_data['vertices'],
                object_data['points'],
                object_indices=object_data.get('indices'),
                object_orig_len=object_data.get('orig_len')
            )
            sample['human_contact_mask'] = human_mask
            sample['object_contact_mask'] = object_mask
        if self.transform is not None:
            sample = self.transform(sample)

        if 'object_coords' not in sample:
            raise KeyError(
                f"Missing object_coords in sample {sample_id}. "
                f"Check annotation at {self.annot_data / f'{sample_id}.json'}"
            )
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for IVD dataset.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    if any('object_coords' not in s for s in batch):
        missing = [s.get('sample_id', '<unknown>') for s in batch if 'object_coords' not in s]
        raise KeyError(f"Missing object_coords for samples: {missing}")

    result = {}
    
    # String fields
    result['sample_id'] = [s['sample_id'] for s in batch]
    
    # Tensor fields
    tensor_keys = [
        'rgb_image', 'human_vertices',
        'object_points', 'human_labels', 'object_coords',
        'human_contact_mask', 'object_contact_mask'
    ]
    
    for key in tensor_keys:
        if key not in batch[0]:
            continue
        values = [s[key] for s in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            result[key] = torch.from_numpy(np.stack(values))
    
    return result


def create_dataloaders(
    intercap_data: str,
    annot_data: str,
    batch_size: int = 8,
    num_workers: int = 6,
    keypoint_manager: Optional[KeypointManager] = None,
    renderer: Optional[MultiViewRenderer] = None,
    image_size: int = 224,
    render_size: int = 256,
    load_contact_masks: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        intercap_data: Dataset root directory
        annot_data: Contact annotation directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        keypoint_manager: Keypoint manager
        renderer: Multi-view renderer
        image_size: RGB image size
        render_size: Render view size
        load_contact_masks: Whether to load per-point contact masks
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_transform = get_train_transforms(image_size, render_size)
    val_transform = get_val_transforms(image_size, render_size)
    
    train_dataset = IVDDataset(
        intercap_data=intercap_data,
        annot_data=annot_data,
        split='train',
        transform=train_transform,
        keypoint_manager=keypoint_manager,
        renderer=renderer,
        render_on_fly=True,
        render_size=render_size,
        load_contact_masks=load_contact_masks
    )
    
    val_dataset = IVDDataset(
        intercap_data=intercap_data,
        annot_data=annot_data,
        split='val',
        transform=val_transform,
        keypoint_manager=keypoint_manager,
        renderer=renderer,
        render_on_fly=True,
        render_size=render_size,
        load_contact_masks=load_contact_masks
    )
    
    test_dataset = IVDDataset(
        intercap_data=intercap_data,
        annot_data=annot_data,
        split='test',
        transform=val_transform,
        keypoint_manager=keypoint_manager,
        renderer=renderer,
        render_on_fly=True,
        render_size=render_size,
        load_contact_masks=load_contact_masks
    )
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
