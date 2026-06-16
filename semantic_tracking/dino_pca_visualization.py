"""
DINO PCA Visualization for Mesh Correspondence

This module provides functionality to visualize mesh vertices using PCA-reduced
DINO features. Instead of using position-based coloring, we extract semantic
features from DINOv2 for each vertex and use the first 3 PCA components as RGB.

This provides a powerful way to visualize semantic correspondence - vertices
with similar semantic meaning will have similar colors, regardless of their
spatial position.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from sklearn.decomposition import PCA
import nvdiffrast.torch as dr

from .dino_features import DINOv2FeatureExtractor


class DINOPCAColorizer:
    """
    Color mesh vertices based on PCA-reduced DINO features.
    
    This creates semantic-based colors where vertices that are semantically
    similar (e.g., all belong to "leg" parts) will have similar colors,
    even if they're spatially distant.
    """
    
    def __init__(
        self,
        dino_model_name: str = 'dinov2_vits14_reg',
        n_views: int = 8,
        image_size: int = 512,
        device: str = 'cuda'
    ):
        """
        Initialize DINO PCA colorizer.
        
        Args:
            dino_model_name: DINOv2 model variant
            n_views: Number of viewpoints to render from
            image_size: Size of rendered images
            device: Device to run on
        """
        self.device = device
        self.n_views = n_views
        self.image_size = image_size
        
        # Initialize DINO feature extractor
        self.dino = DINOv2FeatureExtractor(
            model_name=dino_model_name,
            device=device,
            frozen=True
        )
        
        # PCA model
        self.pca = None
        self.pca_fg = None  # Second-step PCA for foreground (2-step PCA)
        self.feature_dim = self.dino.embed_dim
        
    def generate_camera_poses(self, n_views: int, radius: float = 3.0) -> torch.Tensor:
        """
        Generate camera poses around the object.
        
        Args:
            n_views: Number of viewpoints
            radius: Camera distance from origin
        
        Returns:
            Camera-to-world matrices, shape (n_views, 4, 4)
        """
        poses = []
        for i in range(n_views):
            azimuth = 2 * np.pi * i / n_views
            elevation = np.pi / 12  # ~15 degrees
            
            # Spherical to Cartesian
            cam_x = radius * np.cos(elevation) * np.cos(azimuth)
            cam_y = radius * np.sin(elevation)
            cam_z = radius * np.cos(elevation) * np.sin(azimuth)
            cam_pos = np.array([cam_x, cam_y, cam_z])
            
            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)
            
            # Compute right and up vectors
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(world_up, forward)
            right = right / (np.linalg.norm(right) + 1e-8)
            up = np.cross(forward, right)
            
            # Build camera-to-world matrix
            c2w = np.eye(4)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = -forward  # Forward points in -Z direction
            c2w[:3, 3] = cam_pos
            
            poses.append(c2w)
        
        return torch.from_numpy(np.stack(poses)).float().to(self.device)
    
    def render_mesh(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        mvp: torch.Tensor,
        resolution: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render mesh from a camera viewpoint using nvdiffrast.
        
        Args:
            vertices: Mesh vertices, shape (V, 3)
            faces: Mesh faces, shape (F, 3)
            mvp: Model-view-projection matrix, shape (4, 4)
            resolution: Output image resolution
        
        Returns:
            Tuple of:
                - rendered_image: RGB image, shape (1, 3, H, W)
                - vertex_ids: Per-pixel vertex IDs, shape (H, W)
        """
        # Initialize nvdiffrast context
        glctx = dr.RasterizeGLContext()
        
        # Homogeneous coordinates
        vertices_homo = torch.cat([
            vertices,
            torch.ones(vertices.shape[0], 1, device=vertices.device)
        ], dim=-1)
        
        # Project vertices
        vertices_clip = vertices_homo @ mvp.T
        
        # Rasterize
        rast_out, _ = dr.rasterize(
            glctx,
            vertices_clip.unsqueeze(0),
            faces.int(),
            resolution=[resolution, resolution]
        )
        
        # Extract vertex IDs from rasterization
        # rast_out[..., 3] contains triangle IDs (0 = background)
        tri_ids = rast_out[0, :, :, 3].long() - 1  # -1 because 0 is background
        
        # Get barycentric coordinates
        bary = rast_out[0, :, :, :3]
        
        # Create a simple white rendering (we only need DINO features, not appearance)
        rendered = torch.ones(1, resolution, resolution, 3, device=self.device)
        
        # Mask where mesh is visible
        mask = tri_ids >= 0
        
        # Convert to (B, C, H, W) format
        rendered_image = rendered.permute(0, 3, 1, 2)
        
        # Get vertex IDs per pixel
        vertex_ids = torch.full((resolution, resolution), -1, dtype=torch.long, device=self.device)
        valid_mask = mask
        valid_tris = tri_ids[valid_mask]
        valid_bary = bary[valid_mask]
        
        # For each pixel, get the dominant vertex (highest barycentric weight)
        if valid_tris.numel() > 0:
            tri_verts = faces[valid_tris]  # (N, 3)
            # Get vertex with max barycentric weight
            max_bary_idx = valid_bary.argmax(dim=-1)  # (N,)
            vertex_ids[valid_mask] = tri_verts[torch.arange(len(tri_verts), device=self.device), max_bary_idx]
        
        return rendered_image, vertex_ids
    
    def build_projection_matrix(
        self,
        fov: float = 60.0,
        aspect: float = 1.0,
        near: float = 0.1,
        far: float = 100.0
    ) -> torch.Tensor:
        """Build perspective projection matrix."""
        fov_rad = fov * np.pi / 180.0
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        proj = torch.zeros(4, 4, device=self.device)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0
        
        return proj
    
    @torch.no_grad()
    def extract_vertex_features(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        normalize_mesh: bool = True
    ) -> np.ndarray:
        """
        Extract DINO features for each vertex by rendering from multiple views.
        
        Args:
            vertices: Mesh vertices, shape (V, 3)
            faces: Mesh faces, shape (F, 3)
            normalize_mesh: Whether to normalize mesh to unit sphere
        
        Returns:
            Per-vertex DINO features, shape (V, D)
        """
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).float().to(self.device)
        if isinstance(faces, np.ndarray):
            faces = torch.from_numpy(faces).long().to(self.device)
        
        n_vertices = vertices.shape[0]
        
        # Normalize mesh to fit in unit sphere
        if normalize_mesh:
            center = vertices.mean(dim=0)
            vertices = vertices - center
            scale = vertices.norm(dim=-1).max()
            vertices = vertices / scale
        
        # Generate camera poses (radius=3 for normalized mesh)
        c2w_matrices = self.generate_camera_poses(self.n_views, radius=3.0)
        
        # Build projection matrix
        proj = self.build_projection_matrix()
        
        # Storage for features per vertex (we'll average across views)
        vertex_features = torch.zeros(n_vertices, self.feature_dim, device=self.device)
        vertex_counts = torch.zeros(n_vertices, device=self.device)
        
        print(f"Extracting DINO features from {self.n_views} views...")
        
        for view_idx in range(self.n_views):
            # Get camera pose
            c2w = c2w_matrices[view_idx]
            w2c = torch.inverse(c2w)
            
            # Build MVP matrix (projection @ view)
            mvp = proj @ w2c  # Both are 4x4 matrices
            
            # Render mesh
            try:
                rendered_img, vertex_ids = self.render_mesh(
                    vertices, faces, mvp, self.image_size
                )
                
                # Debug: Check if any vertices were rendered
                visible_count = (vertex_ids >= 0).sum().item()
                if visible_count == 0:
                    print(f"  View {view_idx}: WARNING - No vertices visible!")
                    continue
                else:
                    print(f"  View {view_idx}: {visible_count} pixels with visible vertices")
                    
            except Exception as e:
                print(f"Warning: Failed to render view {view_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Extract DINO patch features
            dino_features = self.dino.extract_patch_features(rendered_img)  # (1, H', W', D)
            dino_features = dino_features[0]  # (H', W', D)
            
            # Upsample DINO features to match image resolution
            H_dino, W_dino = dino_features.shape[:2]
            dino_features = dino_features.permute(2, 0, 1).unsqueeze(0)  # (1, D, H', W')
            dino_features = F.interpolate(
                dino_features,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
            dino_features = dino_features[0].permute(1, 2, 0)  # (H, W, D)
            
            # For each visible vertex, accumulate its DINO feature
            visible_mask = vertex_ids >= 0
            if visible_mask.sum() == 0:
                continue
            
            visible_vertex_ids = vertex_ids[visible_mask]
            visible_features = dino_features[visible_mask]  # (N_visible, D)
            
            # Accumulate features for each vertex
            unique_verts = visible_vertex_ids.unique()
            for vert_id in unique_verts:
                vert_mask = visible_vertex_ids == vert_id
                vert_feat = visible_features[vert_mask].mean(dim=0)
                vertex_features[vert_id] += vert_feat
                vertex_counts[vert_id] += 1
        
        # Average features across views
        valid_mask = vertex_counts > 0
        
        if valid_mask.sum() == 0:
            # No vertices were visible in any view - this shouldn't happen
            # Fall back to random features
            print("ERROR: No vertices were visible in any view! Using random features as fallback.")
            vertex_features = torch.randn(n_vertices, self.feature_dim, device=self.device)
            return vertex_features.cpu().numpy()
        
        vertex_features[valid_mask] /= vertex_counts[valid_mask].unsqueeze(-1)
        
        # For vertices never seen, use nearest neighbor features
        if (~valid_mask).any():
            print(f"Warning: {(~valid_mask).sum()} vertices were never visible. Using k-NN interpolation.")
            unseen_verts = torch.where(~valid_mask)[0]
            seen_verts = torch.where(valid_mask)[0]
            
            for unseen_v in unseen_verts:
                # Find nearest seen vertex
                dists = (vertices[seen_verts] - vertices[unseen_v]).norm(dim=-1)
                if dists.numel() > 0:
                    nearest_idx = dists.argmin()
                    nearest_v = seen_verts[nearest_idx]
                    vertex_features[unseen_v] = vertex_features[nearest_v]
        
        return vertex_features.cpu().numpy()
    
    def fit_pca_and_colorize(
        self,
        vertex_features: np.ndarray,
        n_components: int = 3,
        use_two_step_pca: bool = True,
        fg_bg_threshold: float = 0.35
    ) -> np.ndarray:
        """
        Apply 2-step PCA to vertex features and map to RGB colors.
        
        This implements the algorithm from purnasai/Dino_V2:
        1. First PCA: Separate foreground (mesh) from background using first component
        2. Second PCA: Apply PCA only to foreground for better semantic coloring
        
        Args:
            vertex_features: Per-vertex DINO features, shape (V, D)
            n_components: Number of PCA components (should be 3 for RGB)
            use_two_step_pca: Whether to use 2-step PCA (True) or simple PCA (False)
            fg_bg_threshold: Threshold for first PCA component to separate fg/bg
        
        Returns:
            RGB colors per vertex, shape (V, 3) in [0, 1]
        """
        print(f"Fitting PCA to reduce {vertex_features.shape[1]}D features to {n_components}D...")
        
        if not use_two_step_pca:
            # Simple single-step PCA (legacy behavior)
            self.pca = PCA(n_components=n_components)
            pca_features = self.pca.fit_transform(vertex_features)
            
            print(f"PCA explained variance: {self.pca.explained_variance_ratio_}")
            
            # Normalize PCA features to [0, 1] for RGB
            pca_min = pca_features.min(axis=0, keepdims=True)
            pca_max = pca_features.max(axis=0, keepdims=True)
            pca_range = pca_max - pca_min
            pca_range[pca_range == 0] = 1
            
            colors = (pca_features - pca_min) / pca_range
            colors = np.clip(colors, 0, 1).astype(np.float32)
            return colors
        
        # =====================================================================
        # 2-STEP PCA (from purnasai/Dino_V2)
        # =====================================================================
        print("Using 2-step PCA for better semantic coloring...")
        
        n_vertices = vertex_features.shape[0]
        
        # Step 1: First PCA to separate foreground from background
        # For mesh vertices, all should be "foreground" but this step still
        # helps identify outlier vertices with atypical features
        self.pca = PCA(n_components=n_components)
        pca_features_step1 = self.pca.fit_transform(vertex_features)  # (V, 3)
        
        print(f"Step 1 PCA explained variance: {self.pca.explained_variance_ratio_}")
        
        # Min-max scale the first component
        pc1 = pca_features_step1[:, 0]
        pc1_scaled = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
        
        # For mesh vertices, we use adaptive thresholding since all vertices
        # are "foreground" - we identify the main cluster vs outliers
        # Use median-based threshold for robustness
        adaptive_threshold = np.median(pc1_scaled)
        
        # Identify foreground (main vertices) vs potential outliers  
        # For meshes, we keep more vertices as "foreground"
        fg_mask = pc1_scaled <= adaptive_threshold + 0.3  # More permissive for mesh
        bg_mask = ~fg_mask
        
        n_fg = fg_mask.sum()
        n_bg = bg_mask.sum()
        print(f"  Step 1: {n_fg} foreground vertices, {n_bg} background/outlier vertices")
        
        # If almost all vertices are in one category, use simple PCA
        if n_fg < 10 or n_bg < 3:
            print("  Most vertices in same cluster, using standard min-max scaling")
            # All vertices treated as foreground - just do min-max scaling
            colors = np.zeros((n_vertices, 3), dtype=np.float32)
            for i in range(3):
                col = pca_features_step1[:, i]
                col_min, col_max = col.min(), col.max()
                if col_max - col_min > 1e-8:
                    colors[:, i] = (col - col_min) / (col_max - col_min)
                else:
                    colors[:, i] = 0.5
            return np.clip(colors, 0, 1).astype(np.float32)
        
        # Step 2: Second PCA only on foreground vertices for better coloring
        print("  Step 2: Fitting PCA on foreground vertices...")
        self.pca_fg = PCA(n_components=n_components)
        fg_features = vertex_features[fg_mask]
        pca_features_fg = self.pca_fg.fit_transform(fg_features)  # (N_fg, 3)
        
        print(f"Step 2 PCA explained variance: {self.pca_fg.explained_variance_ratio_}")
        
        # Min-max scale each component separately
        for i in range(n_components):
            col = pca_features_fg[:, i]
            col_min, col_max = col.min(), col.max()
            if col_max - col_min > 1e-8:
                pca_features_fg[:, i] = (col - col_min) / (col_max - col_min)
            else:
                pca_features_fg[:, i] = 0.5
        
        # Build final colors
        colors = np.zeros((n_vertices, 3), dtype=np.float32)
        
        # Background/outlier vertices get dark color (not pure black for visibility)
        colors[bg_mask] = 0.1
        
        # Foreground vertices get the scaled PCA features as RGB
        colors[fg_mask] = pca_features_fg
        
        colors = np.clip(colors, 0, 1).astype(np.float32)
        
        print(f"  Final color range: [{colors.min():.3f}, {colors.max():.3f}]")
        
        return colors
    
    def colorize_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normalize_mesh: bool = True
    ) -> np.ndarray:
        """
        Generate DINO PCA-based colors for mesh vertices.
        
        This is the main entry point: given a mesh, it extracts DINO features,
        applies PCA, and returns RGB colors.
        
        Args:
            vertices: Mesh vertices, shape (V, 3)
            faces: Mesh faces, shape (F, 3)
            normalize_mesh: Whether to normalize mesh before rendering
        
        Returns:
            RGB colors per vertex, shape (V, 3) in [0, 1]
        """
        # Extract DINO features from multi-view renders
        if isinstance(vertices, torch.Tensor):
            vertices_np = vertices.detach().cpu().numpy()
        else:
            vertices_np = vertices
            
        if isinstance(faces, torch.Tensor):
            faces_np = faces.detach().cpu().numpy()
        else:
            faces_np = faces
        
        vertex_features = self.extract_vertex_features(
            torch.from_numpy(vertices_np).float().to(self.device),
            torch.from_numpy(faces_np).long().to(self.device),
            normalize_mesh=normalize_mesh
        )
        
        # Apply PCA and convert to colors
        colors = self.fit_pca_and_colorize(vertex_features, n_components=3)
        
        return colors


def create_dino_pca_visualization(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    dino_model: str = 'dinov2_vits14_reg',
    n_views: int = 8,
    normalize_mesh: bool = True,
    device: str = 'cuda'
):
    """
    Convenience function to create DINO PCA visualization and save to PLY.
    
    Args:
        vertices: Mesh vertices, shape (V, 3)
        faces: Mesh faces, shape (F, 3)
        output_path: Output PLY file path
        dino_model: DINOv2 model name
        n_views: Number of viewpoints for feature extraction
        normalize_mesh: Whether to normalize mesh
        device: Device to run on
    """
    from .vertex_color_tracking import export_ply
    
    # Create colorizer
    colorizer = DINOPCAColorizer(
        dino_model_name=dino_model,
        n_views=n_views,
        device=device
    )
    
    # Generate DINO PCA colors
    colors = colorizer.colorize_mesh(vertices, faces, normalize_mesh)
    
    # Export to PLY
    export_ply(output_path, vertices, faces, colors)
    print(f"Saved DINO PCA visualization to {output_path}")
    
    return colors
