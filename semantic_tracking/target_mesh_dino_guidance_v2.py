"""
Target Mesh DINO Guidance V2 - Viewpoint-Aligned Version

This is a corrected version of the target mesh guidance that properly aligns
viewpoints between source and target mesh renders.

Key Fix:
--------
The original version compared source renders (from random training viewpoints)
to target features (from fixed canonical viewpoints). This caused misalignment
where a frontal source view might match to patches from a side target view.

This version:
1. Pre-computes target features for a DENSE grid of viewpoints (e.g., 36 views)
2. When computing loss, matches each source view to the CLOSEST target view
3. Compares spatial features only between aligned viewpoints

Author: MeshUp Enhanced Pipeline
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pymeshlab
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import math

try:
    import nvdiffrast.torch as dr
    from nvdiffmodeling.src import obj, mesh, render, texture
    from utilities.camera import get_camera_params
    from utilities.helpers import create_scene
    RENDERING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Rendering modules not available: {e}")
    RENDERING_AVAILABLE = False

try:
    from .dino_correspondence_loss import DINOv2Extractor
    DINO_EXTRACTOR_AVAILABLE = True
except ImportError:
    DINO_EXTRACTOR_AVAILABLE = False


class TargetMeshDINOGuidanceV2(nn.Module):
    """
    Viewpoint-Aligned Target Mesh DINO Guidance.
    
    This version properly aligns source and target viewpoints before
    comparing DINO features.
    """
    
    def __init__(
        self,
        target_mesh_path: str,
        device: str = 'cuda',
        weight: float = 0.5,
        model_name: str = 'dinov2_vits14_reg',
        n_azimuths: int = 12,  # Azimuth angles to pre-compute
        n_elevations: int = 3,  # Elevation angles to pre-compute
        warmup_epochs: int = 50,
        warmup_type: str = 'linear',
        global_weight: float = 0.3,
        spatial_weight: float = 0.7,
        render_weight: float = 0.0,
        use_soft_matching: bool = True,
        temperature: float = 0.1,
        image_resolution: int = 224,
        online_target_render: bool = False,
        online_cache_features: bool = True,
        online_cache_max_size: int = 4096,
        view_rounding_deg: float = 2.0,
        view_rounding_dist: float = 0.05,
        view_rounding_fov: float = 1.0,
    ):
        super().__init__()
        
        self.target_mesh_path = target_mesh_path
        self.device = device
        self.weight = weight
        self.n_azimuths = n_azimuths
        self.n_elevations = n_elevations
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.global_weight = global_weight
        self.spatial_weight = spatial_weight
        self.render_weight = render_weight
        self.use_soft_matching = use_soft_matching
        self.temperature = temperature
        self.image_resolution = image_resolution
        self.online_target_render = online_target_render
        self.online_cache_features = online_cache_features
        self.online_cache_max_size = online_cache_max_size
        self.view_rounding_deg = view_rounding_deg
        self.view_rounding_dist = view_rounding_dist
        self.view_rounding_fov = view_rounding_fov
        
        # Initialize DINO extractor
        self.dino = self._create_dino_extractor(model_name, device)
        
        # Target features indexed by viewpoint
        # Shape: (n_elevations, n_azimuths, ...) for easy lookup
        self.target_cls_by_view = None       # (E, A, D)
        self.target_patches_by_view = None   # (E, A, H, W, D)
        
        # Azimuth/elevation grids for lookup
        self.azimuth_grid = None    # (A,) angles in degrees
        self.elevation_grid = None  # (E,) angles in degrees

        # Keep target scene for optional exact camera-matched online rendering.
        self.target_scene = None
        self.glctx = None
        self.target_feature_cache = {}
        
        self.initialized = False
        
        print(f"TargetMeshDINOGuidanceV2 created (viewpoint-aligned):")
        print(f"  - Target mesh: {target_mesh_path}")
        print(f"  - Weight: {weight}")
        print(f"  - Viewpoints: {n_azimuths} azimuths x {n_elevations} elevations = {n_azimuths * n_elevations} views")
        print(f"  - Render weight: {render_weight}")
        print(f"  - Warmup: {warmup_epochs} epochs")
        print(f"  - Online target render: {online_target_render}")
    
    def _create_dino_extractor(self, model_name: str, device: str):
        """Create DINO feature extractor."""
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
        """Compute warmup factor."""
        if epoch < self.warmup_epochs:
            if self.warmup_type == 'linear':
                return epoch / self.warmup_epochs
            elif self.warmup_type == 'cosine':
                return 0.5 * (1 - math.cos(math.pi * epoch / self.warmup_epochs))
            elif self.warmup_type == 'step':
                return 0.0
        return 1.0
    
    def _get_viewpoint_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dense viewpoint grid for target feature extraction."""
        azimuths = torch.linspace(0, 360 - 360/self.n_azimuths, self.n_azimuths)
        elevations = torch.linspace(0, 60, self.n_elevations)  # 0 to 60 degrees
        return azimuths, elevations
    
    def _find_closest_viewpoint(
        self, 
        azimuth: float, 
        elevation: float
    ) -> Tuple[int, int]:
        """Find the closest pre-computed viewpoint index."""
        # Handle azimuth wrap-around (350° is close to 10°)
        azim_diff = torch.abs(self.azimuth_grid - azimuth)
        azim_diff = torch.minimum(azim_diff, 360 - azim_diff)
        azim_idx = azim_diff.argmin().item()
        
        # Simple closest for elevation
        elev_idx = torch.abs(self.elevation_grid - elevation).argmin().item()
        
        return elev_idx, azim_idx
    
    @torch.no_grad()
    def initialize_from_target_mesh(
        self,
        glctx=None,
    ):
        """
        Initialize target features by rendering from a dense grid of viewpoints.
        """
        if not RENDERING_AVAILABLE:
            raise RuntimeError("Rendering modules not available.")
        
        print(f"Initializing target mesh features from: {self.target_mesh_path}")
        print(f"  Pre-computing {self.n_elevations} x {self.n_azimuths} = {self.n_elevations * self.n_azimuths} viewpoints...")
        
        if glctx is None:
            glctx = dr.RasterizeGLContext()
        
        # Load target mesh
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(self.target_mesh_path))
        
        if not ms.current_mesh().has_wedge_tex_coord():
            ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_obj = os.path.join(tmpdir, "target.obj")
            ms.save_current_mesh(tmp_obj)
            target_mesh = obj.load_obj(tmp_obj)
        
        target_mesh = mesh.unit_size(target_mesh)
        
        # Create simple material
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
        
        scene = create_scene([target_mesh.eval()], sz=512)
        scene = mesh.compute_tangents(mesh.auto_normals(scene))
        self.target_scene = scene
        self.glctx = glctx

        if self.online_target_render:
            self.initialized = True
            print("Target features initialized in online camera-matched mode")
            return
        
        # Generate viewpoint grid
        self.azimuth_grid, self.elevation_grid = self._get_viewpoint_grid()
        
        # Extract features for each viewpoint
        n_elev = len(self.elevation_grid)
        n_azim = len(self.azimuth_grid)
        
        # First pass to determine feature dimensions
        test_params = get_camera_params(30.0, 0.0, 3.0, self.image_resolution, 60.0)
        for k, v in test_params.items():
            if isinstance(v, torch.Tensor):
                test_params[k] = v.to(self.device)
        
        test_render = render.render_mesh(
            glctx, scene.eval(test_params),
            test_params["mvp"], test_params["campos"], test_params["lightpos"],
            5.0, self.image_resolution, spp=1, num_layers=1, msaa=False,
            background=torch.ones(1, self.image_resolution, self.image_resolution, 3, device=self.device)
        )
        test_render = test_render.permute(0, 3, 1, 2)
        test_feat = self.dino(test_render, return_cls=True, return_patches=True)
        
        D = test_feat['cls'].shape[-1]
        H_patch, W_patch = test_feat['patches'].shape[1:3]
        
        # Initialize storage
        self.target_cls_by_view = torch.zeros(n_elev, n_azim, D, device=self.device)
        self.target_patches_by_view = torch.zeros(n_elev, n_azim, H_patch, W_patch, D, device=self.device)
        
        # Extract features for all viewpoints
        for e_idx, elev in enumerate(self.elevation_grid):
            for a_idx, azim in enumerate(self.azimuth_grid):
                cam_params = get_camera_params(elev.item(), azim.item(), 3.0, self.image_resolution, 60.0)
                for k, v in cam_params.items():
                    if isinstance(v, torch.Tensor):
                        cam_params[k] = v.to(self.device)
                
                final_mesh = scene.eval(cam_params)
                rendered = render.render_mesh(
                    glctx, final_mesh,
                    cam_params["mvp"], cam_params["campos"], cam_params["lightpos"],
                    5.0, self.image_resolution, spp=1, num_layers=1, msaa=False,
                    background=torch.ones(1, self.image_resolution, self.image_resolution, 3, device=self.device)
                )
                rendered = rendered.permute(0, 3, 1, 2)
                
                features = self.dino(rendered, return_cls=True, return_patches=True)
                
                self.target_cls_by_view[e_idx, a_idx] = F.normalize(features['cls'][0], dim=0)
                self.target_patches_by_view[e_idx, a_idx] = F.normalize(features['patches'][0], dim=-1)
        
        self.initialized = True
        
        print(f"Target features initialized:")
        print(f"  - CLS per view: {self.target_cls_by_view.shape}")
        print(f"  - Patches per view: {self.target_patches_by_view.shape}")

    def _quantize_value(self, value: float, step: float) -> float:
        if step <= 0:
            return float(value)
        return round(float(value) / step) * step

    def _build_cache_key(self, azimuth: float, elevation: float, distance: float, fov: float) -> Tuple[float, float, float, float]:
        return (
            self._quantize_value(azimuth, self.view_rounding_deg),
            self._quantize_value(elevation, self.view_rounding_deg),
            self._quantize_value(distance, self.view_rounding_dist),
            self._quantize_value(fov, self.view_rounding_fov),
        )

    @torch.no_grad()
    def _get_online_target_features(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        fov: float,
        light_position: Optional[torch.Tensor] = None,
        light_power: float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.target_scene is None or self.glctx is None:
            raise RuntimeError("Target scene is not initialized for online rendering")

        cache_key = self._build_cache_key(azimuth, elevation, distance, fov)
        if self.online_cache_features and cache_key in self.target_feature_cache:
            return self.target_feature_cache[cache_key]

        cam_params = get_camera_params(elevation, azimuth, distance, self.image_resolution, fov)
        for k, v in cam_params.items():
            if isinstance(v, torch.Tensor):
                cam_params[k] = v.to(self.device)

        if light_position is not None:
            cam_params["lightpos"] = light_position.reshape(1, 3).to(self.device)

        final_mesh = self.target_scene.eval(cam_params)
        rendered = render.render_mesh(
            self.glctx,
            final_mesh,
            cam_params["mvp"],
            cam_params["campos"],
            cam_params["lightpos"],
            light_power,
            self.image_resolution,
            spp=1,
            num_layers=1,
            msaa=False,
            background=torch.ones(1, self.image_resolution, self.image_resolution, 3, device=self.device),
        )
        rendered = rendered.permute(0, 3, 1, 2)

        features = self.dino(rendered, return_cls=True, return_patches=True)
        target_cls = F.normalize(features['cls'][0], dim=0)
        target_patches = F.normalize(features['patches'][0], dim=-1)
        target_rgb = rendered[0].detach()

        if self.online_cache_features:
            if len(self.target_feature_cache) >= self.online_cache_max_size:
                oldest_key = next(iter(self.target_feature_cache))
                del self.target_feature_cache[oldest_key]
            self.target_feature_cache[cache_key] = (target_cls, target_patches, target_rgb)

        return target_cls, target_patches, target_rgb
    
    def _compute_view_aligned_loss(
        self,
        source_cls: torch.Tensor,       # (B, D)
        source_patches: torch.Tensor,   # (B, H, W, D)
        azimuths: torch.Tensor,         # (B,) in degrees
        elevations: torch.Tensor,       # (B,) in degrees
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute view-aligned loss by matching each source view to closest target view.
        """
        B = source_cls.shape[0]
        global_losses = []
        spatial_losses = []

        def _bidirectional_patch_loss(src_flat: torch.Tensor, tgt_flat: torch.Tensor) -> torch.Tensor:
            """
            Bidirectional patch matching prevents many-to-one collapse where only
            part of the source shape is strongly constrained by target features.
            """
            sim_matrix = torch.mm(src_flat, tgt_flat.T)  # (Ns, Nt)

            if self.use_soft_matching:
                # Source -> target
                w_s2t = F.softmax(sim_matrix / self.temperature, dim=1)
                s2t = (w_s2t * sim_matrix).sum(dim=1).mean()

                # Target -> source
                sim_t = sim_matrix.T
                w_t2s = F.softmax(sim_t / self.temperature, dim=1)
                t2s = (w_t2s * sim_t).sum(dim=1).mean()
            else:
                s2t = sim_matrix.max(dim=1)[0].mean()
                t2s = sim_matrix.max(dim=0)[0].mean()

            return 1 - 0.5 * (s2t + t2s)
        
        for b in range(B):
            # Find closest target viewpoint
            e_idx, a_idx = self._find_closest_viewpoint(
                azimuths[b].item(), 
                elevations[b].item()
            )
            
            # Get target features for this viewpoint
            target_cls = self.target_cls_by_view[e_idx, a_idx]        # (D,)
            target_patches = self.target_patches_by_view[e_idx, a_idx] # (H, W, D)
            
            # Global loss (CLS similarity)
            src_cls_norm = F.normalize(source_cls[b], dim=0)
            global_sim = torch.dot(src_cls_norm, target_cls)
            global_losses.append(1 - global_sim)
            
            # Spatial loss (patch similarity)
            src_patches = F.normalize(source_patches[b].reshape(-1, source_patches.shape[-1]), dim=1)  # (H*W, D)
            tgt_patches = target_patches.reshape(-1, target_patches.shape[-1])  # (H*W, D)

            spatial_losses.append(_bidirectional_patch_loss(src_patches, tgt_patches))
        
        global_loss = torch.stack(global_losses).mean()
        spatial_loss = torch.stack(spatial_losses).mean()
        
        return global_loss, spatial_loss
    
    def forward(
        self,
        rendered_images: torch.Tensor,
        azimuths: torch.Tensor,
        elevations: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
        fovs: Optional[torch.Tensor] = None,
        light_positions: Optional[torch.Tensor] = None,
        light_power: float = 5.0,
        epoch: int = 0,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute view-aligned target mesh guidance loss.
        
        Args:
            rendered_images: Current source renders (B, C, H, W) or (B, H, W, C)
            azimuths: Azimuth angles for each render (B,) in degrees
            elevations: Elevation angles for each render (B,) in degrees
            epoch: Current epoch for warmup
            return_components: Return loss breakdown
        
        Returns:
            Loss or dict with components
        """
        if not self.initialized:
            zero = torch.tensor(0.0, device=self.device)
            if return_components:
                return {'total': zero, 'global': zero, 'spatial': zero, 'warmup': 0.0}
            return zero
        
        warmup = self.get_warmup_factor(epoch)
        if warmup == 0:
            zero = torch.tensor(0.0, device=self.device)
            if return_components:
                return {'total': zero, 'global': zero, 'spatial': zero, 'warmup': 0.0}
            return zero
        
        # Ensure correct format
        if rendered_images.shape[-1] == 3:
            rendered_images = rendered_images.permute(0, 3, 1, 2)
        
        # Extract source features
        features = self.dino(rendered_images, return_cls=True, return_patches=True)
        
        if self.online_target_render:
            B = features['cls'].shape[0]
            global_losses = []
            spatial_losses = []
            render_losses = []

            def _bidirectional_patch_loss(src_flat: torch.Tensor, tgt_flat: torch.Tensor) -> torch.Tensor:
                sim_matrix = torch.mm(src_flat, tgt_flat.T)  # (Ns, Nt)

                if self.use_soft_matching:
                    w_s2t = F.softmax(sim_matrix / self.temperature, dim=1)
                    s2t = (w_s2t * sim_matrix).sum(dim=1).mean()

                    sim_t = sim_matrix.T
                    w_t2s = F.softmax(sim_t / self.temperature, dim=1)
                    t2s = (w_t2s * sim_t).sum(dim=1).mean()
                else:
                    s2t = sim_matrix.max(dim=1)[0].mean()
                    t2s = sim_matrix.max(dim=0)[0].mean()

                return 1 - 0.5 * (s2t + t2s)

            for b in range(B):
                distance = float(distances[b].item()) if distances is not None else 3.0
                fov = float(fovs[b].item()) if fovs is not None else 60.0
                light_pos = light_positions[b] if light_positions is not None else None

                target_cls, target_patches, target_rgb = self._get_online_target_features(
                    azimuth=float(azimuths[b].item()),
                    elevation=float(elevations[b].item()),
                    distance=distance,
                    fov=fov,
                    light_position=light_pos,
                    light_power=light_power,
                )

                src_cls_norm = F.normalize(features['cls'][b], dim=0)
                global_sim = torch.dot(src_cls_norm, target_cls)
                global_losses.append(1 - global_sim)

                src_patches = F.normalize(features['patches'][b].reshape(-1, features['patches'].shape[-1]), dim=1)
                tgt_patches = target_patches.reshape(-1, target_patches.shape[-1])
                spatial_losses.append(_bidirectional_patch_loss(src_patches, tgt_patches))

                if self.render_weight > 0:
                    current_rgb = rendered_images[b]
                    if target_rgb.shape[-2:] != current_rgb.shape[-2:]:
                        target_rgb = F.interpolate(
                            target_rgb.unsqueeze(0),
                            size=current_rgb.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze(0)
                    render_losses.append(F.l1_loss(current_rgb.clamp(0, 1), target_rgb.clamp(0, 1)))

            global_loss = torch.stack(global_losses).mean()
            spatial_loss = torch.stack(spatial_losses).mean()
            render_loss = torch.stack(render_losses).mean() if render_losses else torch.tensor(0.0, device=self.device)
        else:
            # Compute view-aligned losses from precomputed grid
            global_loss, spatial_loss = self._compute_view_aligned_loss(
                features['cls'],
                features['patches'],
                azimuths,
                elevations,
            )
            render_loss = torch.tensor(0.0, device=self.device)
        
        # Combine
        total_loss = (
            self.global_weight * global_loss +
            self.spatial_weight * spatial_loss +
            self.render_weight * render_loss
        ) * self.weight * warmup
        
        if return_components:
            return {
                'total': total_loss,
                'global': global_loss * self.global_weight * self.weight * warmup,
                'spatial': spatial_loss * self.spatial_weight * self.weight * warmup,
                'render': render_loss * self.render_weight * self.weight * warmup,
                'warmup': warmup
            }
        
        return total_loss


def create_target_mesh_guidance_v2(
    target_mesh_path: str,
    device: str = 'cuda',
    weight: float = 0.5,
    warmup_epochs: int = 100,
    model_name: str = 'dinov2_vits14_reg',
    n_azimuths: int = 12,
    n_elevations: int = 3,
    global_weight: float = 0.3,
    spatial_weight: float = 0.7,
    render_weight: float = 0.0,
    online_target_render: bool = False,
    online_cache_features: bool = True,
    online_cache_max_size: int = 4096,
    view_rounding_deg: float = 2.0,
    view_rounding_dist: float = 0.05,
    view_rounding_fov: float = 1.0,
) -> TargetMeshDINOGuidanceV2:
    """Factory function for creating viewpoint-aligned target mesh guidance."""
    return TargetMeshDINOGuidanceV2(
        target_mesh_path=target_mesh_path,
        device=device,
        weight=weight,
        warmup_epochs=warmup_epochs,
        model_name=model_name,
        n_azimuths=n_azimuths,
        n_elevations=n_elevations,
        global_weight=global_weight,
        spatial_weight=spatial_weight,
        render_weight=render_weight,
        online_target_render=online_target_render,
        online_cache_features=online_cache_features,
        online_cache_max_size=online_cache_max_size,
        view_rounding_deg=view_rounding_deg,
        view_rounding_dist=view_rounding_dist,
        view_rounding_fov=view_rounding_fov,
    )
