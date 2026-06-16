"""
Multi-angle PCA Visualization for MeshUp

This module renders the mesh from multiple fixed viewpoints and saves
DINO PCA colored images at regular intervals during training.

Useful for tracking semantic correspondence preservation across epochs.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple
import nvdiffrast.torch as dr

try:
    from nvdiffmodeling.src import mesh, render
    from utilities.camera import get_camera_params
    from utilities.helpers import create_scene
    RENDER_AVAILABLE = True
except ImportError:
    RENDER_AVAILABLE = False


class MultiAnglePCAVisualizer:
    """
    Renders mesh from multiple fixed angles with DINO PCA colors.
    """
    
    def __init__(
        self,
        output_dir: str,
        n_angles: int = 8,
        elevation: float = 30.0,
        distance: float = 3.0,
        resolution: int = 512,
        device: str = 'cuda'
    ):
        self.output_dir = Path(output_dir)
        self.n_angles = n_angles
        self.elevation = elevation
        self.distance = distance
        self.resolution = resolution
        self.device = device
        
        # Fixed azimuth angles
        self.azimuths = [i * (360.0 / n_angles) for i in range(n_angles)]
        
        # Create output directories
        self.pca_dir = self.output_dir / "pca_multiview"
        self.pca_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"MultiAnglePCAVisualizer initialized:")
        print(f"  - Angles: {self.azimuths}")
        print(f"  - Output: {self.pca_dir}")
    
    def render_multiview(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_colors: torch.Tensor,
        glctx,
        epoch: int
    ) -> List[np.ndarray]:
        """
        Render mesh from all fixed angles with vertex colors.
        
        Args:
            vertices: Mesh vertices (V, 3)
            faces: Mesh faces (F, 3)
            vertex_colors: Per-vertex colors (V, 3) in [0, 1]
            glctx: OpenGL context
            epoch: Current epoch number
        
        Returns:
            List of rendered images as numpy arrays
        """
        if not RENDER_AVAILABLE:
            print("Warning: Rendering modules not available")
            return []
        
        images = []
        
        # Create epoch directory
        epoch_dir = self.pca_dir / f"epoch_{epoch:05d}"
        epoch_dir.mkdir(exist_ok=True)
        
        for i, azim in enumerate(self.azimuths):
            # Get camera parameters
            cam_params = get_camera_params(
                self.elevation, azim, self.distance, 
                self.resolution, fov=60.0
            )
            
            # Move to device
            for k, v in cam_params.items():
                if isinstance(v, torch.Tensor):
                    cam_params[k] = v.to(self.device)
            
            # Render with vertex colors
            img = self._render_with_colors(
                vertices, faces, vertex_colors, 
                cam_params, glctx
            )
            
            images.append(img)
            
            # Save image
            img_path = epoch_dir / f"view_{i:02d}_azim{int(azim):03d}.png"
            Image.fromarray((img * 255).astype(np.uint8)).save(img_path)
        
        # Create combined grid image
        self._save_grid(images, epoch_dir / "grid.png")
        
        return images
    
    def _render_with_colors(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        colors: torch.Tensor,
        cam_params: dict,
        glctx
    ) -> np.ndarray:
        """Render mesh with vertex colors using nvdiffrast."""
        vertices = vertices.to(self.device)
        faces = faces.to(self.device).int()
        colors = colors.to(self.device)
        
        # Normalize vertices
        center = vertices.mean(dim=0)
        vertices = vertices - center
        scale = vertices.norm(dim=-1).max()
        vertices = vertices / scale
        
        # Add homogeneous coordinate
        v_homo = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
        
        # Transform vertices
        mvp = cam_params['mvp'].to(self.device)
        v_clip = torch.matmul(v_homo, mvp.T)
        
        # Rasterize
        rast, _ = dr.rasterize(glctx, v_clip[None], faces, resolution=[self.resolution, self.resolution])
        
        # Interpolate colors
        colors_out, _ = dr.interpolate(colors[None], rast, faces)
        
        # Apply alpha mask
        alpha = (rast[..., 3:4] > 0).float()
        colors_out = colors_out * alpha + (1 - alpha)  # White background
        
        # Convert to numpy
        img = colors_out[0].detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        
        return img
    
    def _save_grid(self, images: List[np.ndarray], path: Path):
        """Save images as a grid."""
        n = len(images)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        h, w = images[0].shape[:2]
        grid = np.ones((rows * h, cols * w, 3))
        
        for i, img in enumerate(images):
            r, c = i // cols, i % cols
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
        
        Image.fromarray((grid * 255).astype(np.uint8)).save(path)


def create_multiangle_visualizer(
    output_dir: str,
    n_angles: int = 8,
    device: str = 'cuda'
) -> MultiAnglePCAVisualizer:
    """Factory function to create multi-angle visualizer."""
    return MultiAnglePCAVisualizer(
        output_dir=output_dir,
        n_angles=n_angles,
        device=device
    )
