"""
InterCap Dataset for IVD
基于重建结果生成人体-物体接触标注
"""

import os
import sys
import json
import glob
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from scipy.spatial import cKDTree
# import open3d as o3d
import smplx

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.keypoints import KeypointManager


class ContactAnnotationGenerator:
    """
    从 InterCap 重建结果生成接触标注
    
    流程:
    1. 加载人体点云和物体点云
    2. 计算人体点云每个点到物体的最近距离
    3. 小于阈值的点标记为接触区域
    4. 检查 87 个关键点是否在接触区域内
    5. 对于接触的关键点，找到最近的物体点作为配对
    """
    
    def __init__(
        self,
        keypoint_manager: KeypointManager,
        contact_threshold: float = 0.05,
        smplx_model_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            keypoint_manager: 87 个关键点管理器
            contact_threshold: 接触距离阈值（米）
            smplx_model_path: SMPL-X 模型路径
            device: 计算设备
        """
        self.keypoint_manager = keypoint_manager
        self.contact_threshold = contact_threshold
        self.device = device
        
        self.model_path = "/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/smpl_models/SMPLX_NEUTRAL.npz"
        self.smplx_model = smplx.create(self.model_path, model_type='smplx', gender='neutral', 
                        use_pca=False, num_betas=10, batch_size=1).to(device)
        if smplx_model_path and SMPLX_AVAILABLE:
            self.smplx_model = smplx.create(
                smplx_model_path,
                model_type='smplx',
                gender='neutral',
                use_pca=False,
                num_betas=10
            )

    def _farthest_point_sample_indices(self, points: np.ndarray, num_samples: int) -> np.ndarray:
        """Uniformly sample points via farthest point sampling."""
        num_points = points.shape[0]
        if num_samples >= num_points:
            return np.arange(num_points, dtype=np.int64)
        pts = torch.from_numpy(points.astype(np.float32))
        indices = torch.zeros(num_samples, dtype=torch.long)
        distances = torch.full((num_points,), float('inf'))
        farthest = 0
        for i in range(num_samples):
            indices[i] = farthest
            centroid = pts[farthest].unsqueeze(0)
            dist = torch.sum((pts - centroid) ** 2, dim=1)
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances).item()
        return indices.numpy()
    
    def load_point_cloud(self, ply_path: str) -> np.ndarray:
        """加载 PLY 点云文件"""
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required to load PLY files")
        
        mesh = trimesh.load(ply_path)
        
        if hasattr(mesh, 'vertices'):
            return np.array(mesh.vertices, dtype=np.float32)
        else:
            return np.array(mesh, dtype=np.float32)
    
    def load_smplx_params(self, json_path: str) -> Dict:
        """加载 SMPL-X 参数"""
        with open(json_path, 'r') as f:
            params = json.load(f)
        return params
    
    def get_human_vertices_from_params(self, smplx_params: Dict) -> np.ndarray:
        """从 SMPL-X 参数生成人体顶点"""
        if self.smplx_model is None:
            raise RuntimeError("SMPL-X model not loaded")
        
        # 准备参数
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
        
        vertices = output.vertices.numpy().squeeze(0)  # (V, 3)
        # human_mesh = o3d.geometry.PointCloud()
        # human_mesh.points = o3d.utility.Vector3dVector(vertices)
        # o3d.io.write_point_cloud(f'./human_mesh.ply', human_mesh)
        return vertices
    
    def compute_contact_regions(
        self,
        human_points: np.ndarray,
        object_points: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算人体点云的接触区域
        
        Args:
            human_points: (N_h, 3) 人体点云
            object_points: (N_o, 3) 物体点云
            threshold: 接触阈值，默认使用 self.contact_threshold
            
        Returns:
            contact_mask: (N_h,) 布尔数组，True 表示该点在接触区域
            nearest_distances: (N_h,) 每个人体点到最近物体点的距离
        """
        if threshold is None:
            threshold = self.contact_threshold
        
        # 构建物体点云的 KD-Tree
        object_tree = cKDTree(object_points)
        
        # 查询每个人体点到最近物体点的距离
        nearest_distances, nearest_indices = object_tree.query(human_points, k=1)
        
        # 标记接触区域
        contact_mask = nearest_distances < threshold
        
        return contact_mask, nearest_distances
    
    def extract_keypoint_contacts(
        self,
        human_vertices: np.ndarray,
        object_points_contact: np.ndarray,
        contact_mask: np.ndarray,
        threshold: Optional[float] = None,
        object_points_output: Optional[np.ndarray] = None
    ) -> Dict:
        """
        提取 87 个关键点的接触信息
        
        Args:
            human_vertices: (V, 3) 人体网格顶点
            object_points_contact: (N_o, 3) 物体点云（用于接触计算）
            contact_mask: (V,) 顶点级别的接触 mask
            threshold: 关键点判定阈值
            object_points_output: (N_o, 3) 输出用物体点云（可选）
            
        Returns:
            字典包含:
                - human_contact_labels: (87,) 二分类标签
                - object_contact_coords: (K, 3) 接触物体点坐标
                - keypoint_object_pairs: List[(keypoint_idx, object_coord)]
        """
        if threshold is None:
            threshold = self.contact_threshold
        
        # 获取 87 个关键点的顶点索引
        keypoint_indices = self.keypoint_manager.get_vertex_indices()
        
        # 提取关键点位置
        keypoint_positions = human_vertices[keypoint_indices]  # (87, 3)
        
        # 获取关键点对应的接触状态
        keypoint_contact_from_mask = contact_mask[keypoint_indices]  # (87,)
        
        # 同时计算关键点到物体的直接距离（双重验证）
        object_tree = cKDTree(object_points_contact)
        kp_distances, kp_nearest_obj_indices = object_tree.query(keypoint_positions, k=1)
        
        # 综合判断：顶点在接触区域 OR 关键点直接距离小于阈值
        human_contact_labels = np.logical_or(
            keypoint_contact_from_mask,
            kp_distances < threshold
        ).astype(np.float32)

        ## visulize of human contact
        # human_contact_point = keypoint_positions[human_contact_labels > 0.5]
        # human_contact_point_mesh = o3d.geometry.PointCloud()
        # human_contact_point_mesh.points = o3d.utility.Vector3dVector(human_contact_point)
        # o3d.io.write_point_cloud(f'./human_contact_point.ply', human_contact_point_mesh)

        # object_contact_point = object_points[kp_nearest_obj_indices[human_contact_labels > 0.5]]
        # object_contact_point_mesh = o3d.geometry.PointCloud()
        # object_contact_point_mesh.points = o3d.utility.Vector3dVector(object_contact_point)
        # o3d.io.write_point_cloud(f'./object_contact_point.ply', object_contact_point_mesh)
        
        # 收集接触的关键点及其对应的物体点
        keypoint_object_pairs = []
        object_contact_coords = []
        object_contact_index = []
        
        for kp_idx in range(keypoint_positions.shape[0]):
            if human_contact_labels[kp_idx] > 0.5:
                # 该关键点是接触点
                obj_point_idx = kp_nearest_obj_indices[kp_idx]
                if object_points_output is None:
                    object_points_output = object_points_contact
                obj_coord = object_points_output[obj_point_idx]
                
                keypoint_object_pairs.append({
                    'keypoint_idx': int(kp_idx),
                    'keypoint_name': self.keypoint_manager.idx_to_name.get(kp_idx, f'point_{kp_idx}'),
                    'keypoint_coord': keypoint_positions[kp_idx].tolist(),
                    'object_index': int(obj_point_idx),
                    'object_coord': obj_coord.tolist(),
                    'distance': float(kp_distances[kp_idx])
                })
                
                object_contact_coords.append(obj_coord)
                object_contact_index.append(int(obj_point_idx))
        # 如果没有接触点，添加一个占位符
        if len(object_contact_coords) == 0:
            object_contact_coords = [np.zeros(3)]
        
        return {
            'human_contact_labels': human_contact_labels,
            'object_contact_coords': np.array(object_contact_coords, dtype=np.float32),
            'object_contact_index': object_contact_index,
            'keypoint_object_pairs': keypoint_object_pairs,
            'num_contacts': int(human_contact_labels.sum())
        }
    
    def process_sample(
        self,
        sample_dir: str,
        use_combined_pcd: bool = True,
        return_masks: bool = False
    ) -> Dict:
        """
        处理单个样本，生成完整标注
        
        Args:
            sample_dir: 样本目录路径
            use_combined_pcd: 是否使用组合点云 (hum&obj_pcd.ply)
            
        Returns:
            标注字典
        """
        sample_dir = Path(sample_dir)
        sample_id = sample_dir.name
        
        # 加载物体点云
        obj_optimize_path = sample_dir / 'obj_optimize.ply'
        obj_template_path = sample_dir / 'obj_template.ply'
        
        if not obj_optimize_path.exists():
            raise FileNotFoundError(f"Object optimized mesh not found in {sample_dir}")
        if not obj_template_path.exists():
            raise FileNotFoundError(f"Object template mesh not found in {sample_dir}")
        
        object_points_optimize = self.load_point_cloud(str(obj_optimize_path))
        object_points_template = self.load_point_cloud(str(obj_template_path))
        # Normalize template points: center + scale by max radius
        if object_points_template.size > 0:
            centroid = object_points_template.mean(axis=0)
            object_points_template = object_points_template - centroid
            max_radius = np.linalg.norm(object_points_template, axis=1).max()
            if max_radius > 0:
                object_points_template = object_points_template / max_radius
        # obj_mesh = o3d.geometry.PointCloud()
        # obj_mesh.points = o3d.utility.Vector3dVector(object_points)
        # o3d.io.write_point_cloud(f'./obj_mesh.ply', obj_mesh)
        
        # 加载人体数据
        smplx_path = sample_dir / 'smplx_parameters.json'
        if not smplx_path.exists():
            smplx_path = sample_dir / 'smplx_parameters_new.json'
        
        if smplx_path.exists() and self.smplx_model is not None:
            # 从 SMPL-X 参数生成人体顶点
            smplx_params = self.load_smplx_params(str(smplx_path))
            human_vertices = self.get_human_vertices_from_params(smplx_params)
        elif use_combined_pcd:
            # 从组合点云提取（需要分离人体和物体）
            print(sample_dir)
            raise NotImplementedError("Not implemented")
            combined_path = sample_dir / 'hum&obj_pcd.ply'
            if combined_path.exists():
                # 这里简化处理，实际可能需要更复杂的分离逻辑
                combined_points = self.load_point_cloud(str(combined_path))
                # 假设前 N 个点是人体（需要根据实际情况调整）
                human_vertices = combined_points
            else:
                raise FileNotFoundError(f"No human mesh data found in {sample_dir}")
        else:
            raise FileNotFoundError(f"No human mesh data found in {sample_dir}")
        
        # 计算接触区域（人体点）
        contact_mask_full, distances = self.compute_contact_regions(human_vertices, object_points_optimize)

        # 计算接触区域（物体点）
        human_tree = cKDTree(human_vertices)
        obj_distances, _ = human_tree.query(object_points_optimize, k=1)
        object_mask = obj_distances < self.contact_threshold
        
        # 提取关键点接触信息
        contact_info = self.extract_keypoint_contacts(
            human_vertices,
            object_points_optimize,
            contact_mask_full,
            object_points_output=object_points_template
        )
        
        # 加载原始标注（如果存在）
        ann_path = sample_dir / 'annotations.json'
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                original_ann = json.load(f)
        else:
            original_ann = {}
        
        # 加载人体部位标注（如果存在）
        human_part_path = sample_dir / 'contact_label' / 'human_part.json'
        if human_part_path.exists():
            with open(human_part_path, 'r') as f:
                human_parts = json.load(f)
        else:
            human_parts = []
        
        # 组合结果
        result = {
            'sample_id': sample_id,
            'human_contact_labels': contact_info['human_contact_labels'].tolist(),
            'object_contact_coords': contact_info['object_contact_coords'].tolist(),
            'object_contact_index': contact_info['object_contact_index'],
            'keypoint_object_pairs': contact_info['keypoint_object_pairs'],
            'num_contacts': contact_info['num_contacts'],
            'contact_threshold': self.contact_threshold,
            'original_human_parts': human_parts,
            'original_annotation': original_ann
        }

        if return_masks:
            return result, contact_mask_full.astype(np.uint8), object_mask.astype(np.uint8)

        return result


class InterCapDataset(torch.utils.data.Dataset):
    """
    InterCap 数据集用于 IVD 训练
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        keypoints_json: str = None,
        contact_threshold: float = 0.05,
        transform = None,
        precompute_annotations: bool = False,
        annotation_cache_dir: str = None,
        max_samples: int = None
    ):
        """
        Args:
            data_root: InterCap Output 目录
            split: 'train' 或 'val'
            keypoints_json: 87 个关键点定义文件
            contact_threshold: 接触阈值
            transform: 数据变换
            precompute_annotations: 是否预计算标注
            annotation_cache_dir: 标注缓存目录
            max_samples: 最大样本数
        """
        self.data_root = Path(data_root)
        self.split = split
        self.contact_threshold = contact_threshold
        self.transform = transform
        
        # 初始化关键点管理器
        if keypoints_json:
            self.keypoint_manager = KeypointManager(keypoints_json)
        else:
            self.keypoint_manager = KeypointManager()
        
        # 查找所有样本
        self.sample_dirs = self._find_samples()
        
        if max_samples:
            self.sample_dirs = self.sample_dirs[:max_samples]
        
        print(f"Found {len(self.sample_dirs)} samples for {split}")
        
        # 标注缓存
        self.annotation_cache_dir = annotation_cache_dir
        if annotation_cache_dir:
            os.makedirs(annotation_cache_dir, exist_ok=True)
        
        # 标注生成器
        self.annotation_generator = ContactAnnotationGenerator(
            keypoint_manager=self.keypoint_manager,
            contact_threshold=contact_threshold
        )
    
    def _find_samples(self) -> List[Path]:
        """查找所有样本目录"""
        # 查找包含必要文件的目录
        pattern = str(self.data_root / '*')
        all_dirs = glob.glob(pattern)
        
        valid_dirs = []
        for d in all_dirs:
            d = Path(d)
            if d.is_dir():
                # 检查必要文件
                has_image = (d / 'image.jpg').exists() or (d / 'image.png').exists()
                has_obj = (d / 'obj_pcd.ply').exists() or (d / 'obj_pcd_origin.ply').exists()
                has_smplx = (d / 'smplx_parameters_new.json').exists() or (d / 'smplx_parameters.json').exists()
                
                if has_image and has_obj:
                    valid_dirs.append(d)
        
        return sorted(valid_dirs)
    
    def _load_annotation(self, sample_dir: Path) -> Dict:
        """加载或生成标注"""
        sample_id = sample_dir.name
        
        # 检查缓存
        if self.annotation_cache_dir:
            cache_path = Path(self.annotation_cache_dir) / f'{sample_id}.json'
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    return json.load(f)
        
        # 检查样本目录中是否有预计算的标注
        ivd_ann_path = sample_dir / 'ivd_annotation.json'
        if ivd_ann_path.exists():
            with open(ivd_ann_path, 'r') as f:
                return json.load(f)
        
        # 生成标注
        try:
            annotation = self.annotation_generator.process_sample(str(sample_dir))
            
            # 保存到缓存
            if self.annotation_cache_dir:
                cache_path = Path(self.annotation_cache_dir) / f'{sample_id}.json'
                with open(cache_path, 'w') as f:
                    json.dump(annotation, f)
            
            return annotation
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            # 返回默认标注
            return {
                'sample_id': sample_id,
                'human_contact_labels': [0.0] * 87,
                'object_contact_coords': [[0.0, 0.0, 0.0]],
                'num_contacts': 0
            }
    
    def _load_image(self, sample_dir: Path) -> np.ndarray:
        """加载图像"""
        from PIL import Image
        
        for ext in ['.jpg', '.png', '.jpeg']:
            img_path = sample_dir / f'image{ext}'
            if img_path.exists():
                return np.array(Image.open(img_path).convert('RGB'))
        
        raise FileNotFoundError(f"Image not found in {sample_dir}")
    
    def _load_object_points(self, sample_dir: Path) -> np.ndarray:
        """加载物体点云"""
        for name in ['obj_pcd.ply', 'obj_pcd_origin.ply']:
            pcd_path = sample_dir / name
            if pcd_path.exists():
                return self.annotation_generator.load_point_cloud(str(pcd_path))
        
        raise FileNotFoundError(f"Object point cloud not found in {sample_dir}")
    
    def __len__(self) -> int:
        return len(self.sample_dirs)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_dir = self.sample_dirs[idx]
        sample_id = sample_dir.name
        
        # 加载图像
        rgb_image = self._load_image(sample_dir)

        annotation = self._load_annotation(sample_dir)
        object_points = self._load_object_points(sample_dir)
        
        # 构建样本
        sample = {
            'sample_id': sample_id,
            'rgb_image': rgb_image,
            'object_points': object_points,
            'human_labels': np.array(annotation['human_contact_labels'], dtype=np.float32),
            'object_coords': np.array(annotation['object_contact_coords'], dtype=np.float32),
            'num_contacts': annotation.get('num_contacts', 0)
        }
        
        # 应用变换
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def precompute_all_annotations(
    data_root: str,
    output_dir: str,
    keypoints_json: str,
    contact_threshold: float = 0.05,
    smplx_model_path: str = None
):
    """
    预计算所有样本的标注
    
    Args:
        data_root: InterCap Output 目录
        output_dir: 标注输出目录
        keypoints_json: 关键点定义文件
        contact_threshold: 接触阈值
        smplx_model_path: SMPL-X 模型路径
    """
    os.makedirs(output_dir, exist_ok=True)
    human_mask_dir = os.path.join(output_dir, 'human_mask')
    object_mask_dir = os.path.join(output_dir, 'object_mask')
    os.makedirs(human_mask_dir, exist_ok=True)
    os.makedirs(object_mask_dir, exist_ok=True)
    
    # 初始化
    keypoint_manager = KeypointManager(keypoints_json)
    generator = ContactAnnotationGenerator(
        keypoint_manager=keypoint_manager,
        contact_threshold=contact_threshold,
        smplx_model_path=smplx_model_path
    )
    
    # 查找所有样本
    sample_dirs = glob.glob(os.path.join(data_root, '*'))
    sample_dirs = [d for d in sample_dirs if os.path.isdir(d)]
    
    print(f"Processing {len(sample_dirs)} samples...")
    
    results = []
    for sample_dir in tqdm(sample_dirs):
        annotation, human_mask, object_mask = generator.process_sample(
            sample_dir,
            return_masks=True
        )
        
        # 保存单个标注
        sample_id = os.path.basename(sample_dir)
        ann_path = os.path.join(output_dir, f'{sample_id}.json')
        with open(ann_path, 'w') as f:
            json.dump(annotation, f, indent=2)

        # 保存接触 mask
        np.save(os.path.join(human_mask_dir, f'{sample_id}.npy'), human_mask)
        np.save(os.path.join(object_mask_dir, f'{sample_id}.npy'), object_mask)
        
        results.append({
            'sample_id': sample_id,
            'num_contacts': annotation['num_contacts'],
            'success': True
        })
        
    
    # 保存汇总
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'total_samples': len(sample_dirs),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'contact_threshold': contact_threshold,
            'results': results
        }, f, indent=2)
    
    print(f"Done! Results saved to {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate IVD annotations from InterCap')
    parser.add_argument('--data_root', type=str, default='/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/DATASET/INTERCAP_train/Output',
                        help='InterCap Output directory')
    parser.add_argument('--output_dir', type=str, default='./annotations',
                        help='Output directory for annotations')
    parser.add_argument('--keypoints', type=str, default='data/part_kp.json',
                        help='Keypoints JSON file')
    parser.add_argument('--threshold', type=float, default=0.025,
                        help='Contact threshold in meters')
    parser.add_argument('--smplx_model', type=str, default=None,
                        help='SMPL-X model path')
    
    args = parser.parse_args()
    
    precompute_all_annotations(
        data_root=args.data_root,
        output_dir=args.output_dir,
        keypoints_json=args.keypoints,
        contact_threshold=args.threshold,
        smplx_model_path=args.smplx_model
    )
