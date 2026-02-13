"""
Multi-View Renderer using PyTorch3D
Renders human mesh and object point cloud from multiple viewpoints
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict

try:
    import pytorch3d
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        PointsRasterizationSettings,
        PointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
        look_at_view_transform,
        TexturesVertex,
        PointLights
    )
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: PyTorch3D not available. Rendering will use dummy outputs.")


class MultiViewRenderer(nn.Module):
    """
    Multi-view renderer for human mesh and object point cloud.
    
    Renders canonical pose human and object from J viewpoints for 
    extracting geometry-aware visual features.
    """
    
    def __init__(
        self,
        image_size: int = 256,
        num_views: int = 4,
        view_angles: List[float] = [0, 90, 180, 270],
        elevation: float = 0,
        distance: float = 2.5,
        use_nocs: bool = True,
        use_normal: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize the multi-view renderer.
        
        Args:
            image_size: Output image resolution
            num_views: Number of rendered views (J)
            view_angles: Azimuth angles in degrees for each view
            elevation: Camera elevation angle in degrees
            distance: Camera distance from origin
            use_nocs: Whether to render NOCS (Normalized Object Coordinate Space) maps
            use_normal: Whether to render normal maps
            device: Compute device
        """
        super().__init__()
        
        self.image_size = image_size
        self.num_views = num_views
        self.view_angles = view_angles[:num_views]
        self.elevation = elevation
        self.distance = distance
        self.use_nocs = use_nocs
        self.use_normal = use_normal
        self.device = device
        
        if PYTORCH3D_AVAILABLE:
            self._setup_renderer()
    
    def _setup_renderer(self):
        """Setup PyTorch3D rendering components."""
        # Create cameras for each view
        self.cameras_list = []
        
        for azim in self.view_angles:
            R, T = look_at_view_transform(
                dist=self.distance,
                elev=self.elevation,
                azim=azim,
                device=self.device
            )
            camera = FoVPerspectiveCameras(
                R=R, T=T, 
                fov=60,
                device=self.device
            )
            self.cameras_list.append(camera)
        
        # Mesh rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        
        # Point cloud rasterization settings
        self.point_raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius=0.01,
            points_per_pixel=10
        )
        
        # Lights for shading
        self.lights = PointLights(
            device=self.device,
            location=[[0.0, 2.0, 2.0]]
        )
    
    def render_mesh(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_colors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Render mesh from multiple viewpoints.
        
        Args:
            vertices: (B, V, 3) mesh vertices
            faces: (F, 3) mesh face indices
            vertex_colors: (B, V, 3) optional vertex colors (for NOCS)
            
        Returns:
            rendered_images: (B*J, 3, H, W) rendered images from all views
        """
        if not PYTORCH3D_AVAILABLE:
            B = vertices.shape[0]
            return torch.zeros(
                B * self.num_views, 3, self.image_size, self.image_size,
                device=vertices.device
            )
        
        B = vertices.shape[0]
        all_renders = []
        
        for camera in self.cameras_list:
            # Expand camera for batch
            cameras = camera.extend(B)
            
            # Create mesh with textures
            if vertex_colors is None:
                # Use NOCS coordinates as colors
                if self.use_nocs:
                    # Normalize vertices to [0, 1] for NOCS
                    v_min = vertices.min(dim=1, keepdim=True)[0]
                    v_max = vertices.max(dim=1, keepdim=True)[0]
                    vertex_colors = (vertices - v_min) / (v_max - v_min + 1e-8)
                else:
                    vertex_colors = torch.ones_like(vertices) * 0.7
            
            textures = TexturesVertex(verts_features=vertex_colors)
            
            # Create batch of meshes
            meshes = Meshes(
                verts=vertices,
                faces=faces.unsqueeze(0).expand(B, -1, -1),
                textures=textures
            )
            
            # Create renderer
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=self.raster_settings
                ),
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=cameras,
                    lights=self.lights
                )
            )
            
            # Render
            images = renderer(meshes)  # (B, H, W, 4)
            images = images[..., :3].permute(0, 3, 1, 2)  # (B, 3, H, W)
            all_renders.append(images)
        
        # Concatenate all views: (B*J, 3, H, W)
        renders = torch.cat(all_renders, dim=0)
        
        # Reorder to (B, J, 3, H, W) then reshape to (B*J, 3, H, W)
        renders = renders.view(self.num_views, B, 3, self.image_size, self.image_size)
        renders = renders.permute(1, 0, 2, 3, 4).contiguous()
        renders = renders.view(B * self.num_views, 3, self.image_size, self.image_size)
        
        return renders
    
    def render_point_cloud(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Render point cloud from multiple viewpoints.
        
        Args:
            points: (B, N, 3) point cloud coordinates
            colors: (B, N, 3) optional point colors
            
        Returns:
            rendered_images: (B*J, 3, H, W) rendered images from all views
        """
        if not PYTORCH3D_AVAILABLE:
            B = points.shape[0]
            return torch.zeros(
                B * self.num_views, 3, self.image_size, self.image_size,
                device=points.device
            )
        
        B = points.shape[0]
        all_renders = []
        
        for camera in self.cameras_list:
            cameras = camera.extend(B)
            
            if colors is None:
                if self.use_nocs:
                    # Normalize points to [0, 1] for NOCS
                    p_min = points.min(dim=1, keepdim=True)[0]
                    p_max = points.max(dim=1, keepdim=True)[0]
                    colors = (points - p_min) / (p_max - p_min + 1e-8)
                else:
                    colors = torch.ones_like(points) * 0.7
            
            # Create point cloud
            point_cloud = Pointclouds(points=points, features=colors)
            
            # Create renderer
            rasterizer = PointsRasterizer(
                cameras=cameras,
                raster_settings=self.point_raster_settings
            )
            renderer = PointsRenderer(
                rasterizer=rasterizer,
                compositor=AlphaCompositor()
            )
            
            # Render
            images = renderer(point_cloud)  # (B, H, W, 4)
            images = images[..., :3].permute(0, 3, 1, 2)  # (B, 3, H, W)
            all_renders.append(images)
        
        # Concatenate and reshape
        renders = torch.cat(all_renders, dim=0)
        renders = renders.view(self.num_views, B, 3, self.image_size, self.image_size)
        renders = renders.permute(1, 0, 2, 3, 4).contiguous()
        renders = renders.view(B * self.num_views, 3, self.image_size, self.image_size)
        
        return renders
    
    def render_normal_map(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> torch.Tensor:
        """
        Render normal maps from multiple viewpoints.
        
        Args:
            vertices: (B, V, 3) mesh vertices
            faces: (F, 3) mesh face indices
            
        Returns:
            normal_maps: (B*J, 3, H, W) normal maps from all views
        """
        if not PYTORCH3D_AVAILABLE:
            B = vertices.shape[0]
            return torch.zeros(
                B * self.num_views, 3, self.image_size, self.image_size,
                device=vertices.device
            )
        
        B = vertices.shape[0]
        
        # Compute vertex normals
        vertex_normals = self._compute_vertex_normals(vertices, faces)
        
        # Normalize to [0, 1] for visualization
        normal_colors = (vertex_normals + 1) / 2
        
        return self.render_mesh(vertices, faces, normal_colors)
    
    def _compute_vertex_normals(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute vertex normals from mesh.
        
        Args:
            vertices: (B, V, 3) mesh vertices
            faces: (F, 3) mesh face indices
            
        Returns:
            normals: (B, V, 3) vertex normals
        """
        B, V, _ = vertices.shape
        F = faces.shape[0]
        
        # Get face vertices
        v0 = vertices[:, faces[:, 0]]  # (B, F, 3)
        v1 = vertices[:, faces[:, 1]]
        v2 = vertices[:, faces[:, 2]]
        
        # Compute face normals
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Accumulate to vertex normals
        vertex_normals = torch.zeros_like(vertices)
        
        for i in range(3):
            vertex_normals.scatter_add_(
                1,
                faces[:, i].unsqueeze(0).unsqueeze(-1).expand(B, -1, 3),
                face_normals
            )
        
        # Normalize
        vertex_normals = vertex_normals / (vertex_normals.norm(dim=-1, keepdim=True) + 1e-8)
        
        return vertex_normals
    
    def forward(
        self,
        human_vertices: torch.Tensor,
        human_faces: torch.Tensor,
        object_points: torch.Tensor,
        render_type: str = 'nocs'
    ) -> Dict[str, torch.Tensor]:
        """
        Render both human mesh and object point cloud.
        
        Args:
            human_vertices: (B, V, 3) human mesh vertices
            human_faces: (F, 3) human mesh faces
            object_points: (B, N, 3) object point cloud
            render_type: 'nocs', 'normal', or 'rgb'
            
        Returns:
            Dictionary with:
                - 'human_renders': (B*J, 3, H, W)
                - 'object_renders': (B*J, 3, H, W)
        """
        if render_type == 'normal':
            human_renders = self.render_normal_map(human_vertices, human_faces)
        else:
            human_renders = self.render_mesh(human_vertices, human_faces)
        
        object_renders = self.render_point_cloud(object_points)
        
        return {
            'human_renders': human_renders,
            'object_renders': object_renders
        }
