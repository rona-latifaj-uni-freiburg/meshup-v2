"""
DINOv2 Feature Consistency Loss for Semantic Correspondence in MeshUp

This module provides a robust DINOv2-based feature consistency loss that helps
maintain semantic correspondence during mesh deformation. It works by:

1. Capturing reference features from multiple canonical viewpoints at initialization
2. During optimization, encouraging rendered views to maintain similar semantic structure
3. Using spatial feature matching to preserve local semantic correspondence

This is a general-purpose solution that works across diverse transformations
(hound→human, bird→plane, hippo→eagle) without requiring manual annotations.

Key Features:
- Multi-view reference features for robust correspondence
- Spatial patch-level matching for local semantic preservation
- Global (CLS) feature matching for overall shape coherence
- Gradual warm-up to not interfere with early optimization
- View-agnostic matching that handles novel viewpoints

Usage:
    loss_fn = DINOCorrespondenceLoss(device='cuda')
    loss_fn.initialize_reference(render_fn, mesh, glctx, device)
    # In training loop:
    dino_loss = loss_fn(rendered_images, epoch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Callable
import math


class DINOv2Extractor(nn.Module):
    """
    Efficient DINOv2 feature extractor optimized for MeshUp.
    
    This handles the preprocessing, feature extraction, and provides
    utilities for computing correspondence.
    """
    
    def __init__(
        self,
        model_name: str = 'dinov2_vits14',
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.patch_size = 14
        
        # Load DINOv2
        print(f"Loading DINOv2 model: {model_name}")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.embed_dim = self.model.embed_dim
        
        # ImageNet normalization
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for DINOv2."""
        # Ensure images are in [0, 1]
        if images.max() > 1.5:
            images = images / 255.0
        
        # Normalize
        images = (images - self.mean.to(images.device)) / self.std.to(images.device)
        
        # Resize to be divisible by patch size
        H, W = images.shape[2:]
        new_H = (H // self.patch_size) * self.patch_size
        new_W = (W // self.patch_size) * self.patch_size
        
        if new_H != H or new_W != W:
            images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        return images
    
    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        return_cls: bool = True,
        return_patches: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from images.
        
        Returns:
            Dict with 'cls' (B, D) and/or 'patches' (B, H, W, D)
        """
        images = self.preprocess(images)
        features = self.model.forward_features(images)
        
        result = {}
        
        if return_cls:
            result['cls'] = features['x_norm_clstoken']  # (B, D)
        
        if return_patches:
            patch_tokens = features['x_norm_patchtokens']  # (B, N, D)
            B = images.shape[0]
            H = images.shape[2] // self.patch_size
            W = images.shape[3] // self.patch_size
            result['patches'] = patch_tokens.reshape(B, H, W, -1)  # (B, H, W, D)
        
        return result


class DINOCorrespondenceLoss(nn.Module):
    """
    DINOv2-based feature consistency loss for semantic correspondence.
    
    This loss encourages the deformed mesh to maintain the same semantic
    structure as the original mesh by matching DINOv2 features.
    
    The key insight is that DINOv2 features capture semantic meaning:
    - Head regions have similar features regardless of species
    - Limb regions have similar features
    - Body/torso regions cluster together
    
    By encouraging feature consistency, we preserve which parts correspond
    to which, even as the shape transforms.
    """
    
    def __init__(
        self,
        model_name: str = 'dinov2_vits14',
        device: str = 'cuda',
        weight: float = 0.1,
        # Loss composition weights
        global_weight: float = 0.3,      # Weight for global (CLS) feature matching
        spatial_weight: float = 0.7,     # Weight for spatial (patch) feature matching
        # Warm-up settings
        warmup_epochs: int = 100,        # Epochs before DINO loss kicks in
        warmup_type: str = 'linear',     # 'linear', 'cosine', or 'step'
        # Multi-view settings
        n_reference_views: int = 8,      # Number of canonical views to capture
        # Matching settings
        use_soft_matching: bool = True,  # Use soft assignment vs hard matching
        temperature: float = 0.1,        # Temperature for soft matching
    ):
        super().__init__()
        
        self.device = device
        self.weight = weight
        self.global_weight = global_weight
        self.spatial_weight = spatial_weight
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.n_reference_views = n_reference_views
        self.use_soft_matching = use_soft_matching
        self.temperature = temperature
        
        # Initialize DINOv2
        self.dino = DINOv2Extractor(model_name=model_name, device=device)
        
        # Reference features (will be set during initialization)
        self.reference_cls_features = None      # (N_views, D)
        self.reference_patch_features = None    # (N_views, H, W, D)
        self.reference_viewpoints = None        # (N_views, ...) camera params
        
        # Aggregated reference features for view-agnostic matching
        self.global_reference_mean = None       # (D,)
        self.global_reference_cov = None        # For Mahalanobis distance (optional)
        self.patch_feature_bank = None          # (N_patches_total, D)
        
        self.initialized = False
    
    def get_warmup_factor(self, epoch: int) -> float:
        """
        Compute warmup factor for gradual loss introduction.
        
        This prevents DINO loss from interfering with early optimization
        when the mesh is still finding its rough shape.
        """
        if epoch < self.warmup_epochs:
            if self.warmup_type == 'linear':
                return epoch / self.warmup_epochs
            elif self.warmup_type == 'cosine':
                return 0.5 * (1 - math.cos(math.pi * epoch / self.warmup_epochs))
            elif self.warmup_type == 'step':
                return 0.0
        return 1.0
    
    @torch.no_grad()
    def initialize_reference(
        self,
        render_function: Callable,
        mesh,
        glctx,
        device: torch.device,
        camera_params_list: Optional[List[Dict]] = None,
    ):
        """
        Initialize reference features from the source mesh.
        
        This captures DINOv2 features from multiple canonical viewpoints
        of the original mesh to establish the semantic reference.
        
        Args:
            render_function: Function that renders mesh given camera params
            mesh: The source mesh object
            glctx: OpenGL context for rendering
            device: Torch device
            camera_params_list: Optional list of camera parameters. If None,
                               uses canonical viewpoints.
        """
        print("Initializing DINOv2 reference features...")
        
        if camera_params_list is None:
            camera_params_list = self._get_canonical_viewpoints(device)
        
        all_cls_features = []
        all_patch_features = []
        
        for i, cam_params in enumerate(camera_params_list):
            # Render from this viewpoint
            rendered = render_function(mesh, cam_params, glctx, device)
            
            # Handle different render output formats
            if isinstance(rendered, dict):
                img = rendered.get('image', rendered.get('rgb'))
            else:
                img = rendered
            
            # Ensure correct format (B, C, H, W)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            if img.shape[-1] == 3:  # (B, H, W, C) -> (B, C, H, W)
                img = img.permute(0, 3, 1, 2)
            
            # Extract features
            features = self.dino(img, return_cls=True, return_patches=True)
            
            all_cls_features.append(features['cls'])
            all_patch_features.append(features['patches'])
        
        # Stack features
        self.reference_cls_features = torch.cat(all_cls_features, dim=0)  # (N_views, D)
        self.reference_patch_features = torch.stack(
            [f.squeeze(0) for f in all_patch_features], dim=0
        )  # (N_views, H, W, D)
        
        # Compute aggregated features for view-agnostic matching
        self.global_reference_mean = self.reference_cls_features.mean(dim=0)  # (D,)
        
        # Build patch feature bank for spatial matching
        N_views, H, W, D = self.reference_patch_features.shape
        self.patch_feature_bank = self.reference_patch_features.reshape(-1, D)  # (N*H*W, D)
        
        # Normalize features for cosine similarity
        self.global_reference_mean = F.normalize(self.global_reference_mean, dim=0)
        self.patch_feature_bank = F.normalize(self.patch_feature_bank, dim=1)
        self.reference_cls_features = F.normalize(self.reference_cls_features, dim=1)
        
        self.initialized = True
        print(f"DINOv2 reference initialized with {len(camera_params_list)} views")
        print(f"  - CLS features: {self.reference_cls_features.shape}")
        print(f"  - Patch features: {self.reference_patch_features.shape}")
        print(f"  - Feature bank size: {self.patch_feature_bank.shape}")
    
    def _get_canonical_viewpoints(self, device: torch.device) -> List[Dict]:
        """
        Generate canonical viewpoints for capturing reference features.
        
        Returns camera parameters for multiple views around the object.
        """
        from utilities.camera import get_camera_params
        
        viewpoints = []
        n_azimuth = self.n_reference_views
        elevation = 30.0  # Fixed elevation
        distance = 3.0
        resolution = 224
        fov = 60.0
        
        for i in range(n_azimuth):
            azimuth = i * (360.0 / n_azimuth)
            cam_params = get_camera_params(elevation, azimuth, distance, resolution, fov)
            # Move tensors to device
            for k, v in cam_params.items():
                if isinstance(v, torch.Tensor):
                    cam_params[k] = v.to(device)
            viewpoints.append(cam_params)
        
        return viewpoints
    
    def _compute_global_loss(self, current_cls: torch.Tensor) -> torch.Tensor:
        """
        Compute global feature consistency loss.
        
        This encourages the overall semantic content to remain similar.
        """
        # Normalize current features
        current_cls = F.normalize(current_cls, dim=1)  # (B, D)
        
        # Compute similarity to reference mean
        similarity = torch.mm(current_cls, self.global_reference_mean.unsqueeze(1))  # (B, 1)
        
        # Loss is 1 - similarity (we want high similarity)
        loss = (1 - similarity.mean())
        
        return loss
    
    def _compute_spatial_loss(self, current_patches: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial feature consistency loss.
        
        This encourages local semantic correspondence to be maintained.
        For each patch in the current render, we find its best matching
        patches in the reference bank and encourage similarity.
        """
        B, H, W, D = current_patches.shape
        
        # Reshape and normalize
        current_flat = current_patches.reshape(B * H * W, D)  # (B*H*W, D)
        current_flat = F.normalize(current_flat, dim=1)
        
        # Compute similarity to reference patch bank
        # (B*H*W, D) x (N*H*W, D)^T -> (B*H*W, N*H*W)
        similarity_matrix = torch.mm(current_flat, self.patch_feature_bank.T)
        
        if self.use_soft_matching:
            # Soft matching: weighted average similarity using softmax
            weights = F.softmax(similarity_matrix / self.temperature, dim=1)
            # Weighted similarity score per current patch
            weighted_sim = (weights * similarity_matrix).sum(dim=1)
            loss = (1 - weighted_sim.mean())
        else:
            # Hard matching: max similarity per current patch
            max_sim, _ = similarity_matrix.max(dim=1)
            loss = (1 - max_sim.mean())
        
        return loss
    
    def _compute_structural_consistency_loss(self, current_patches: torch.Tensor) -> torch.Tensor:
        """
        Compute structural consistency loss.
        
        This encourages the spatial relationship between patches to remain
        similar. Patches that were close in feature space should remain close.
        """
        B, H, W, D = current_patches.shape
        
        # Compute pairwise distances in current features
        current_flat = current_patches.reshape(B, H * W, D)
        current_flat = F.normalize(current_flat, dim=2)
        
        # Self-similarity matrix for current
        current_sim = torch.bmm(current_flat, current_flat.transpose(1, 2))  # (B, HW, HW)
        
        # Get reference self-similarity (averaged across views)
        N_views, H_ref, W_ref, D_ref = self.reference_patch_features.shape
        ref_flat = self.reference_patch_features.reshape(N_views, H_ref * W_ref, D_ref)
        ref_flat = F.normalize(ref_flat, dim=2)
        ref_sim = torch.bmm(ref_flat, ref_flat.transpose(1, 2))  # (N_views, HW, HW)
        ref_sim_mean = ref_sim.mean(dim=0)  # (HW, HW)
        
        # Resize if needed
        if current_sim.shape[1] != ref_sim_mean.shape[0]:
            # Interpolate reference similarity to match current size
            ref_sim_mean = F.interpolate(
                ref_sim_mean.unsqueeze(0).unsqueeze(0),
                size=(current_sim.shape[1], current_sim.shape[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        # Loss: difference in self-similarity structure
        loss = F.mse_loss(current_sim, ref_sim_mean.unsqueeze(0).expand_as(current_sim))
        
        return loss
    
    def forward(
        self,
        rendered_images: torch.Tensor,
        epoch: int = 0,
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute DINOv2 feature consistency loss.
        
        Args:
            rendered_images: Current rendered images (B, C, H, W) in [0, 1]
            epoch: Current epoch (for warmup)
            return_components: Whether to return loss components
        
        Returns:
            Total loss, or dict with 'total', 'global', 'spatial' if return_components
        """
        if not self.initialized:
            if return_components:
                return {'total': torch.tensor(0.0, device=self.device),
                        'global': torch.tensor(0.0, device=self.device),
                        'spatial': torch.tensor(0.0, device=self.device),
                        'warmup': 0.0}
            return torch.tensor(0.0, device=rendered_images.device)
        
        # Get warmup factor
        warmup = self.get_warmup_factor(epoch)
        if warmup == 0:
            if return_components:
                return {'total': torch.tensor(0.0, device=self.device),
                        'global': torch.tensor(0.0, device=self.device),
                        'spatial': torch.tensor(0.0, device=self.device),
                        'warmup': 0.0}
            return torch.tensor(0.0, device=rendered_images.device)
        
        # Ensure correct format
        if rendered_images.shape[-1] == 3:
            rendered_images = rendered_images.permute(0, 3, 1, 2)
        
        # Extract features from current renders
        features = self.dino(rendered_images, return_cls=True, return_patches=True)
        
        # Compute loss components
        global_loss = self._compute_global_loss(features['cls'])
        spatial_loss = self._compute_spatial_loss(features['patches'])
        
        # Combine losses
        total_loss = (
            self.global_weight * global_loss +
            self.spatial_weight * spatial_loss
        ) * self.weight * warmup
        
        if return_components:
            return {
                'total': total_loss,
                'global': global_loss * self.global_weight * self.weight * warmup,
                'spatial': spatial_loss * self.spatial_weight * self.weight * warmup,
                'warmup': warmup
            }
        
        return total_loss


def create_dino_correspondence_loss(
    device: str = 'cuda',
    model_name: str = 'dinov2_vits14',
    weight: float = 0.1,
    warmup_epochs: int = 100,
    **kwargs
) -> DINOCorrespondenceLoss:
    """
    Convenience function to create the DINOv2 correspondence loss.
    
    Args:
        device: Device to run on
        model_name: DINOv2 model variant
        weight: Overall loss weight
        warmup_epochs: Epochs before loss fully activates
        **kwargs: Additional arguments passed to DINOCorrespondenceLoss
    
    Returns:
        Configured DINOCorrespondenceLoss instance
    """
    return DINOCorrespondenceLoss(
        model_name=model_name,
        device=device,
        weight=weight,
        warmup_epochs=warmup_epochs,
        **kwargs
    )
