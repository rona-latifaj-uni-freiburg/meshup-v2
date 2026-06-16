"""
Target Mesh DINO Guidance for Semantic Correspondence-Preserving Mesh Deformation

This module provides DINO-based guidance that deforms a source mesh towards a 
specific target mesh while preserving semantic correspondence. Unlike text-guided
deformation or source-reference DINO loss, this module:

1. Loads a target mesh (the mesh you want to arrive at)
2. Extracts DINO features from multi-view renders of the target
3. Guides deformation so the source mesh's DINO features match the target
4. Maintains semantic correspondence: tail→tail, head→head, limbs→limbs

Key Insight:
-----------
DINOv2 features encode semantic meaning regardless of visual appearance.
By matching DINO features between source renders and target renders,
we ensure that semantically equivalent regions (even between different
identities like two different dinosaurs) are mapped correctly.

Usage:
------
    # Create the guidance
    guidance = TargetMeshDINOGuidance(
        target_mesh_path='./data/.../dinosaur_014/Aligned.obj',
        device='cuda',
        weight=0.5,
    )
    
    # In training loop:
    dino_loss = guidance(current_rendered_images, epoch)
    total_loss = ... + dino_loss

Example: Transforming dinosaur_032 → dinosaur_014
-------------------------------------------------
Both dinosaurs have similar semantics (head, body, tail, legs) but
differ in specific shape/pose. The guidance extracts DINO features
from renders of dinosaur_014 and encourages dinosaur_032's deformation
to produce matching features, ensuring the final mesh looks like
dinosaur_014 while vertex correspondences are preserved.

Author: MeshUp Enhanced Pipeline
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pymeshlab
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union, Callable
import math

# Import rendering utilities (same as main loop uses)
try:
    import nvdiffrast.torch as dr
    from nvdiffmodeling.src import obj, mesh, render, texture
    from utilities.camera import get_camera_params
    from utilities.helpers import create_scene
    RENDERING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Rendering modules not available: {e}")
    RENDERING_AVAILABLE = False

# Import DINO extractor from the existing module
try:
    from .dino_correspondence_loss import DINOv2Extractor
    DINO_EXTRACTOR_AVAILABLE = True
except ImportError:
    DINO_EXTRACTOR_AVAILABLE = False


class TargetMeshDINOGuidance(nn.Module):
    """
    DINO-based guidance for mesh-to-mesh transformation.
    
    This guides deformation from a source mesh toward a specific target mesh,
    using DINOv2 features to maintain semantic correspondence.
    
    The key difference from DINOCorrespondenceLoss is:
    - DINOCorrespondenceLoss: Uses SOURCE mesh features as reference (prevents drift)
    - TargetMeshDINOGuidance: Uses TARGET mesh features as goal (guides transformation)
    
    Parameters:
    -----------
    target_mesh_path : str
        Path to the target mesh file (.obj)
    device : str
        Device for computation ('cuda' or 'cpu')
    weight : float
        Overall loss weight
    model_name : str
        DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', etc.)
    n_views : int
        Number of viewpoints for feature extraction
    warmup_epochs : int
        Epochs before loss fully activates
    warmup_type : str
        Warmup schedule ('linear', 'cosine', 'step')
    global_weight : float
        Weight for global (CLS) feature matching
    spatial_weight : float
        Weight for spatial (patch) feature matching
    use_soft_matching : bool
        Use soft assignment vs hard matching for patches
    temperature : float
        Softmax temperature for soft matching
    """
    
    def __init__(
        self,
        target_mesh_path: str,
        device: str = 'cuda',
        weight: float = 0.5,
        model_name: str = 'dinov2_vits14_reg',
        n_views: int = 8,
        warmup_epochs: int = 50,
        warmup_type: str = 'linear',
        global_weight: float = 0.3,
        spatial_weight: float = 0.7,
        use_soft_matching: bool = True,
        temperature: float = 0.1,
        image_resolution: int = 224,
    ):
        super().__init__()
        
        self.target_mesh_path = target_mesh_path
        self.device = device
        self.weight = weight
        self.n_views = n_views
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.global_weight = global_weight
        self.spatial_weight = spatial_weight
        self.use_soft_matching = use_soft_matching
        self.temperature = temperature
        self.image_resolution = image_resolution
        
        # Initialize DINO extractor
        if DINO_EXTRACTOR_AVAILABLE:
            self.dino = DINOv2Extractor(model_name=model_name, device=device)
        else:
            print("Loading DINOv2 directly...")
            self.dino = self._create_dino_extractor(model_name, device)
        
        # Target mesh features (will be computed during initialization)
        self.target_cls_features = None      # (N_views, D)
        self.target_patch_features = None    # (N_views, H, W, D)
        self.target_global_mean = None       # (D,)
        self.target_patch_bank = None        # (N*H*W, D)
        
        self.initialized = False
        
        print(f"TargetMeshDINOGuidance created:")
        print(f"  - Target mesh: {target_mesh_path}")
        print(f"  - Weight: {weight}")
        print(f"  - Views: {n_views}")
        print(f"  - Warmup: {warmup_epochs} epochs ({warmup_type})")
    
    def _create_dino_extractor(self, model_name: str, device: str):
        """Fallback DINO extractor if main module not available."""
        class SimpleDINOExtractor(nn.Module):
            def __init__(self, model_name, device):
                super().__init__()
                self.patch_size = 14
                self.model = torch.hub.load('facebookresearch/dinov2', model_name)
                self.model = self.model.to(device).eval()
                for p in self.model.parameters():
                    p.requires_grad = False
                self.embed_dim = self.model.embed_dim
                self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
                self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
            
            def preprocess(self, images):
                if images.max() > 1.5:
                    images = images / 255.0
                images = (images - self.mean.to(images.device)) / self.std.to(images.device)
                H, W = images.shape[2:]
                new_H = (H // self.patch_size) * self.patch_size
                new_W = (W // self.patch_size) * self.patch_size
                if new_H != H or new_W != W:
                    images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
                return images
            
            def forward(self, images, return_cls=True, return_patches=True):
                # Differentiable with respect to current renders; callers that
                # cache target/reference features wrap the call in no_grad.
                images = self.preprocess(images)
                features = self.model.forward_features(images)
                result = {}
                if return_cls:
                    result['cls'] = features['x_norm_clstoken']
                if return_patches:
                    B = images.shape[0]
                    H = images.shape[2] // self.patch_size
                    W = images.shape[3] // self.patch_size
                    patch_tokens = features['x_norm_patchtokens']
                    result['patches'] = patch_tokens.reshape(B, H, W, -1)
                return result
        
        return SimpleDINOExtractor(model_name, device)
    
    def get_warmup_factor(self, epoch: int) -> float:
        """Compute warmup factor for gradual loss introduction."""
        if epoch < self.warmup_epochs:
            if self.warmup_type == 'linear':
                return epoch / self.warmup_epochs
            elif self.warmup_type == 'cosine':
                return 0.5 * (1 - math.cos(math.pi * epoch / self.warmup_epochs))
            elif self.warmup_type == 'step':
                return 0.0
        return 1.0
    
    def _get_canonical_viewpoints(self) -> List[Dict]:
        """Generate canonical viewpoints for multi-view feature extraction."""
        viewpoints = []
        elevation = 30.0
        distance = 3.0
        resolution = self.image_resolution
        fov = 60.0
        
        for i in range(self.n_views):
            azimuth = i * (360.0 / self.n_views)
            cam_params = get_camera_params(elevation, azimuth, distance, resolution, fov)
            for k, v in cam_params.items():
                if isinstance(v, torch.Tensor):
                    cam_params[k] = v.to(self.device)
            viewpoints.append(cam_params)
        
        return viewpoints
    
    @torch.no_grad()
    def initialize_from_target_mesh(
        self,
        glctx=None,
        texture_path: Optional[str] = None,
    ):
        """
        Initialize target features by loading and rendering the target mesh.
        
        This extracts DINO features from multiple viewpoints of the target mesh,
        which will serve as the goal for the deformation.
        
        Args:
            glctx: OpenGL context for rendering (if None, creates one)
            texture_path: Optional path to texture (uses Scan.jpg by default)
        """
        if not RENDERING_AVAILABLE:
            raise RuntimeError("Rendering modules not available. Cannot initialize from mesh.")
        
        print(f"Initializing target mesh features from: {self.target_mesh_path}")
        
        # Create GL context if needed
        if glctx is None:
            glctx = dr.RasterizeGLContext()
        
        # Determine texture path
        mesh_dir = Path(self.target_mesh_path).parent
        if texture_path is None:
            texture_path = mesh_dir / "Scan.jpg"
            if not texture_path.exists():
                texture_path = None
        
        # Load target mesh using pymeshlab and nvdiffmodeling
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(self.target_mesh_path))
        
        # Ensure mesh has UV coordinates
        if not ms.current_mesh().has_wedge_tex_coord():
            ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
        
        # Save to temp and reload with nvdiffmodeling
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_obj = os.path.join(tmpdir, "target.obj")
            ms.save_current_mesh(tmp_obj)
            target_mesh = obj.load_obj(tmp_obj)
        
        # Unit size normalization
        target_mesh = mesh.unit_size(target_mesh)
        
        # Create material (flat gray for consistent features)
        kd = texture.Texture2D(torch.full((1, 512, 512, 3), 0.5, device=self.device))
        ks = texture.Texture2D(torch.zeros(1, 512, 512, 3, device=self.device))
        nm = texture.Texture2D(torch.tensor([[[0., 0., 1.]]]).expand(1, 512, 512, 3).to(self.device))
        
        target_mesh = mesh.Mesh(
            v_pos=target_mesh.v_pos.to(self.device),
            t_pos_idx=target_mesh.t_pos_idx.to(self.device),
            v_tex=target_mesh.v_tex.to(self.device) if target_mesh.v_tex is not None else None,
            t_tex_idx=target_mesh.t_tex_idx.to(self.device) if target_mesh.t_tex_idx is not None else None,
            material={"bsdf": "diffuse", "kd": kd, "ks": ks, "normal": nm},
        )
        
        # Create scene
        scene = create_scene([target_mesh.eval()], sz=512)
        scene = mesh.compute_tangents(mesh.auto_normals(scene))
        
        # Get viewpoints
        viewpoints = self._get_canonical_viewpoints()
        
        # Render and extract features
        all_cls_features = []
        all_patch_features = []
        
        for i, cam_params in enumerate(viewpoints):
            # Render from this viewpoint
            final_mesh = scene.eval(cam_params)
            rendered = render.render_mesh(
                glctx, final_mesh,
                cam_params["mvp"], cam_params["campos"], cam_params["lightpos"],
                5.0,  # light power
                self.image_resolution, spp=1, num_layers=1, msaa=False,
                background=torch.ones(1, self.image_resolution, self.image_resolution, 3, device=self.device)
            )  # (1, H, W, 3)
            
            # Convert to (B, C, H, W)
            if rendered.shape[-1] == 3:
                rendered = rendered.permute(0, 3, 1, 2)
            
            # Extract DINO features
            features = self.dino(rendered, return_cls=True, return_patches=True)
            all_cls_features.append(features['cls'])
            all_patch_features.append(features['patches'])
        
        # Stack features
        self.target_cls_features = torch.cat(all_cls_features, dim=0)  # (N_views, D)
        self.target_patch_features = torch.stack(
            [f.squeeze(0) for f in all_patch_features], dim=0
        )  # (N_views, H, W, D)
        
        # Compute aggregated features
        self.target_global_mean = self.target_cls_features.mean(dim=0)  # (D,)
        
        # Build patch feature bank
        N_views, H, W, D = self.target_patch_features.shape
        self.target_patch_bank = self.target_patch_features.reshape(-1, D)  # (N*H*W, D)
        
        # Normalize features for cosine similarity
        self.target_global_mean = F.normalize(self.target_global_mean, dim=0)
        self.target_patch_bank = F.normalize(self.target_patch_bank, dim=1)
        self.target_cls_features = F.normalize(self.target_cls_features, dim=1)
        
        self.initialized = True
        
        print(f"Target mesh features initialized:")
        print(f"  - CLS features: {self.target_cls_features.shape}")
        print(f"  - Patch features: {self.target_patch_features.shape}")
        print(f"  - Feature bank size: {self.target_patch_bank.shape}")
    
    @torch.no_grad()
    def initialize_from_renders(
        self,
        rendered_images: torch.Tensor,
    ):
        """
        Initialize target features from pre-rendered images.
        
        This is useful if you already have renders of the target mesh
        or want to use images from a different source.
        
        Args:
            rendered_images: Target images (N, C, H, W) or (N, H, W, C) in [0, 1]
        """
        print("Initializing target features from provided renders...")
        
        # Ensure correct format
        if rendered_images.shape[-1] == 3:
            rendered_images = rendered_images.permute(0, 3, 1, 2)
        
        rendered_images = rendered_images.to(self.device)
        
        # Extract features
        features = self.dino(rendered_images, return_cls=True, return_patches=True)
        
        self.target_cls_features = features['cls']  # (N, D)
        self.target_patch_features = features['patches']  # (N, H, W, D)
        
        # Compute aggregates
        self.target_global_mean = self.target_cls_features.mean(dim=0)
        N, H, W, D = self.target_patch_features.shape
        self.target_patch_bank = self.target_patch_features.reshape(-1, D)
        
        # Normalize
        self.target_global_mean = F.normalize(self.target_global_mean, dim=0)
        self.target_patch_bank = F.normalize(self.target_patch_bank, dim=1)
        self.target_cls_features = F.normalize(self.target_cls_features, dim=1)
        
        self.initialized = True
        
        print(f"Target features initialized from {rendered_images.shape[0]} renders")
    
    def _compute_global_loss(self, current_cls: torch.Tensor) -> torch.Tensor:
        """Compute global feature similarity loss."""
        current_cls = F.normalize(current_cls, dim=1)  # (B, D)
        similarity = torch.mm(current_cls, self.target_global_mean.unsqueeze(1))  # (B, 1)
        loss = (1 - similarity.mean())
        return loss
    
    def _compute_spatial_loss(self, current_patches: torch.Tensor) -> torch.Tensor:
        """Compute spatial patch feature matching loss."""
        B, H, W, D = current_patches.shape
        
        current_flat = current_patches.reshape(B * H * W, D)
        current_flat = F.normalize(current_flat, dim=1)
        
        # Similarity to target patch bank
        similarity_matrix = torch.mm(current_flat, self.target_patch_bank.T)  # (B*H*W, N*H*W)
        
        if self.use_soft_matching:
            weights = F.softmax(similarity_matrix / self.temperature, dim=1)
            weighted_sim = (weights * similarity_matrix).sum(dim=1)
            loss = (1 - weighted_sim.mean())
        else:
            max_sim, _ = similarity_matrix.max(dim=1)
            loss = (1 - max_sim.mean())
        
        return loss
    
    def _compute_view_matching_loss(self, current_cls: torch.Tensor) -> torch.Tensor:
        """
        Compute view-matching loss for better pose alignment.
        
        This finds the best matching target view for each current view
        and encourages similarity, helping with pose alignment.
        """
        current_cls = F.normalize(current_cls, dim=1)  # (B, D)
        
        # Similarity to all target views
        sim_to_views = torch.mm(current_cls, self.target_cls_features.T)  # (B, N_views)
        
        # Best match per current view
        best_sim, _ = sim_to_views.max(dim=1)  # (B,)
        loss = (1 - best_sim.mean())
        
        return loss
    
    def forward(
        self,
        rendered_images: torch.Tensor,
        epoch: int = 0,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute target mesh DINO guidance loss.
        
        Args:
            rendered_images: Current rendered images (B, C, H, W) in [0, 1]
            epoch: Current epoch (for warmup)
            return_components: Whether to return loss components
        
        Returns:
            Total loss, or dict with components if return_components
        """
        if not self.initialized:
            zero = torch.tensor(0.0, device=self.device)
            if return_components:
                return {'total': zero, 'global': zero, 'spatial': zero, 
                        'view_match': zero, 'warmup': 0.0}
            return zero
        
        # Get warmup factor
        warmup = self.get_warmup_factor(epoch)
        if warmup == 0:
            zero = torch.tensor(0.0, device=self.device)
            if return_components:
                return {'total': zero, 'global': zero, 'spatial': zero,
                        'view_match': zero, 'warmup': 0.0}
            return zero
        
        # Ensure correct format
        if rendered_images.shape[-1] == 3:
            rendered_images = rendered_images.permute(0, 3, 1, 2)
        
        # Extract features
        features = self.dino(rendered_images, return_cls=True, return_patches=True)
        
        # Compute losses
        global_loss = self._compute_global_loss(features['cls'])
        spatial_loss = self._compute_spatial_loss(features['patches'])
        view_match_loss = self._compute_view_matching_loss(features['cls'])
        
        # Combine losses
        total_loss = (
            self.global_weight * global_loss +
            self.spatial_weight * spatial_loss +
            0.1 * view_match_loss  # Small weight for view matching
        ) * self.weight * warmup
        
        if return_components:
            return {
                'total': total_loss,
                'global': global_loss * self.global_weight * self.weight * warmup,
                'spatial': spatial_loss * self.spatial_weight * self.weight * warmup,
                'view_match': view_match_loss * 0.1 * self.weight * warmup,
                'warmup': warmup
            }
        
        return total_loss


def create_target_mesh_guidance(
    target_mesh_path: str,
    device: str = 'cuda',
    weight: float = 0.5,
    warmup_epochs: int = 50,
    **kwargs
) -> TargetMeshDINOGuidance:
    """
    Convenience function to create target mesh DINO guidance.
    
    Args:
        target_mesh_path: Path to target .obj file
        device: Device for computation
        weight: Overall loss weight
        warmup_epochs: Warmup period
        **kwargs: Additional arguments
    
    Returns:
        Configured TargetMeshDINOGuidance instance
    """
    return TargetMeshDINOGuidance(
        target_mesh_path=target_mesh_path,
        device=device,
        weight=weight,
        warmup_epochs=warmup_epochs,
        **kwargs
    )
