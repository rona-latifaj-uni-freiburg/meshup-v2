"""
Modified MeshUp Loop with Semantic Tracking

This is a modified version of the main loop.py that adds:
1. Vertex color tracking for semantic correspondence
2. Optional DINOv2 feature consistency loss
3. Export of colored meshes at each logging interval

Usage:
    python main_with_tracking.py --config ./configs/base_config.yml \
        --mesh ./meshes/hound.obj \
        --output_path ./outputs/hippo_tracked \
        --text_prompt "a hippo" \
        --track_correspondence \
        --color_method position
"""

import kornia
import os
import pathlib
import pymeshlab
import shutil
import torch
import torch.nn.functional as F
import torchvision
import logging
import yaml
import numpy as np
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt

from easydict import EasyDict

from NeuralJacobianFields import SourceMesh

from nvdiffmodeling.src import obj, util, mesh, render, texture, regularizer

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from utilities.uv import get_uvmap
except ModuleNotFoundError:
    print("No module named 'utilities.uv'")

from utilities.video import Video
from utilities.helpers import cosine_avg, create_scene, get_vp_map, get_vp_map_, occlude_vp_map
from utilities.camera import CameraBatch, get_camera_params
from utilities.resize_right import resize, cubic, linear, lanczos2, lanczos3
from deepfloyd import DeepFloydGuidance
from utilities.io import load_ply

# Import semantic tracking modules
from semantic_tracking.vertex_color_tracking import (
    VertexColorTracker,
    initialize_semantic_colors,
    export_ply
)
from semantic_tracking.correspondence_export import (
    export_mesh_with_colors,
    export_correspondence_map,
    visualize_correspondence_displacement
)


def should_log_epoch(cfg, epoch: int) -> bool:
    epoch_num = epoch + 1
    extra_epochs = set(int(e) for e in getattr(cfg, "extra_log_epochs", []) or [])
    return epoch == 0 or epoch_num % cfg.log_interval_im == 0 or epoch_num in extra_epochs

# Import improved DINOv2 correspondence loss
try:
    from semantic_tracking.dino_correspondence_loss import (
        DINOCorrespondenceLoss,
        create_dino_correspondence_loss
    )
    DINO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DINOv2 correspondence loss not available: {e}")
    DINO_AVAILABLE = False

# Import Cross-Attention Semantic Guidance
try:
    from semantic_tracking.cross_attention_guidance import (
        CrossAttentionGuidance,
        create_cross_attention_guidance
    )
    CROSS_ATTN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cross-attention guidance not available: {e}")
    CROSS_ATTN_AVAILABLE = False

# Import DINO PCA Visualization
try:
    from semantic_tracking.dino_pca_visualization import (
        DINOPCAColorizer,
        create_dino_pca_visualization
    )
    DINO_PCA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DINO PCA visualization not available: {e}")
    DINO_PCA_AVAILABLE = False

# Import Target Mesh DINO Guidance (for mesh-to-mesh transformation)
try:
    from semantic_tracking.target_mesh_dino_guidance import (
        TargetMeshDINOGuidance,
        create_target_mesh_guidance
    )
    TARGET_MESH_GUIDANCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Target mesh DINO guidance not available: {e}")
    TARGET_MESH_GUIDANCE_AVAILABLE = False

# Import Target Mesh DINO Guidance V2 (viewpoint-aligned version)
try:
    from semantic_tracking.target_mesh_dino_guidance_v2 import (
        TargetMeshDINOGuidanceV2,
        create_target_mesh_guidance_v2
    )
    TARGET_MESH_GUIDANCE_V2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Target mesh DINO guidance V2 not available: {e}")
    TARGET_MESH_GUIDANCE_V2_AVAILABLE = False

# Import target-guidance experiment losses kept outside the original pipeline.
try:
    from jobs_with_target_guidance.part_aware_chamfer import create_part_aware_chamfer_loss
    PART_AWARE_CHAMFER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Part-aware Chamfer guidance not available: {e}")
    PART_AWARE_CHAMFER_AVAILABLE = False

# Import rendered-DINO semantic bucket guidance experiments.
try:
    from jobs_with_semantic_buckets.rendered_dino_buckets import create_rendered_dino_semantic_chamfer_loss
    SEMANTIC_BUCKET_CHAMFER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Rendered-DINO semantic bucket guidance not available: {e}")
    SEMANTIC_BUCKET_CHAMFER_AVAILABLE = False

# Import SAMPart3D-label guidance experiments.
try:
    from jobs_with_sampart3d_guidance.sampart3d_label_chamfer import create_sampart3d_label_chamfer_loss
    SAMPART3D_LABEL_CHAMFER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAMPart3D label guidance not available: {e}")
    SAMPART3D_LABEL_CHAMFER_AVAILABLE = False

# Import PartField feature guidance experiments.
try:
    from jobs_with_target_guidance.partfield_chamfer import create_partfield_chamfer_loss
    PARTFIELD_CHAMFER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PartField Chamfer guidance not available: {e}")
    PARTFIELD_CHAMFER_AVAILABLE = False

# Import semantic dense vertex correspondence guidance.
try:
    from jobs_with_target_guidance.semantic_vertex_correspondence import (
        create_semantic_vertex_correspondence_loss,
    )
    SEMANTIC_VERTEX_CORRESPONDENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Semantic vertex correspondence guidance not available: {e}")
    SEMANTIC_VERTEX_CORRESPONDENCE_AVAILABLE = False

################################################################################

class TrackedVisualizer:
    """Extended visualizer with semantic tracking support."""
    
    def __init__(
        self,
        out_path: pathlib.Path,
        cfg: EasyDict,
        tb: SummaryWriter,
        tracker: VertexColorTracker = None,
        dino_pca_colorizer: DINOPCAColorizer = None,
        color_method: str = 'position'
    ):
        self.out_path = out_path
        self.cfg = cfg
        self.tb = tb
        self.tracker = tracker
        self.dino_pca_colorizer = dino_pca_colorizer
        self.color_method = color_method
        
        # Create only essential directories by default.
        for d in ["images", "colored_meshes", "correspondence"]:
            os.makedirs(self.out_path / d, exist_ok=True)

    @torch.no_grad()
    def log_mesh(self, step: int, m: mesh.Mesh, cam_params: dict, video: Video, glctx, device):
        m = mesh.unit_size(m.eval(cam_params))
        img = render.render_mesh(
            glctx,
            m,
            cam_params["mvp"],
            cam_params["campos"],
            cam_params["lightpos"],
            self.cfg.log_light_power,
            self.cfg.log_res,
            1,
            background=torch.ones(1, self.cfg.log_res, self.cfg.log_res, 3, device=device),
        )
        video.ready_image(img)
        self.tb.add_mesh("predicted_mesh", vertices=m.v_pos.unsqueeze(0), faces=m.t_pos_idx.unsqueeze(0), global_step=step)
    
    @torch.no_grad()
    def save_epoch_render(
        self,
        epoch: int,
        mesh_obj,
        glctx,
        device,
        n_views: int = 4
    ):
        """
        Save rendered images from multiple viewpoints at this epoch.
        
        This provides visual tracking of how the mesh changes during optimization.
        Images are saved with fixed camera angles for consistency.
        
        Args:
            epoch: Current epoch number
            mesh_obj: The current mesh object
            glctx: nvdiffrast GL context
            device: Torch device
            n_views: Number of viewpoints to render (default 4: front, side, back, other side)
        """
        # Check if we should save at this epoch
        save_interval = getattr(self.cfg, 'save_renders_interval', self.cfg.log_interval_im)
        extra_epochs = set(int(e) for e in getattr(self.cfg, "extra_log_epochs", []) or [])
        epoch_num = epoch + 1
        if epoch != 0 and epoch_num % save_interval != 0 and epoch_num not in extra_epochs:
            return
        
        # Check if rendering is enabled
        if not getattr(self.cfg, 'save_epoch_renders', True):
            return
        
        try:
            render_res = getattr(self.cfg, 'epoch_render_res', 512)
            
            # Create epoch directory
            epoch_dir = self.out_path / "epoch_renders" / f"epoch_{epoch + 1:05d}"
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Fixed viewpoints for consistent comparison across epochs
            azimuths = [0, 90, 180, 270][:n_views]  # Front, right, back, left
            elevations = [15, 15, 15, 15][:n_views]  # Slightly elevated view
            
            for view_idx, (azim, elev) in enumerate(zip(azimuths, elevations)):
                # Get camera parameters for this viewpoint
                cam_params = get_camera_params(
                    elev,  # elevation
                    azim,  # azimuth in degrees
                    self.cfg.log_dist,
                    render_res,
                    self.cfg.log_fov
                )
                
                # Ensure camera params are on correct device
                for k in cam_params:
                    if isinstance(cam_params[k], torch.Tensor):
                        cam_params[k] = cam_params[k].to(device)
                
                # Prepare mesh for rendering
                m_eval = mesh.unit_size(mesh_obj.eval(cam_params))
                
                # Render with white background
                rendered = render.render_mesh(
                    glctx,
                    m_eval,
                    cam_params["mvp"],
                    cam_params["campos"],
                    cam_params["lightpos"],
                    self.cfg.log_light_power,
                    render_res,
                    1,
                    background=torch.ones(1, render_res, render_res, 3, device=device),
                )
                
                # Save image
                img_np = rendered[0].mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                img_pil = Image.fromarray(img_np)
                img_path = epoch_dir / f"view_{view_idx}_azim{azim}_elev{elev}.png"
                img_pil.save(str(img_path))
            
            # Also create a grid of all views
            if n_views > 1:
                # Create 2x2 grid (or 1xN if fewer views)
                grid_imgs = []
                for view_idx, (azim, elev) in enumerate(zip(azimuths, elevations)):
                    img_path = epoch_dir / f"view_{view_idx}_azim{azim}_elev{elev}.png"
                    grid_imgs.append(Image.open(str(img_path)))
                
                # Create grid
                if len(grid_imgs) == 4:
                    grid_size = (render_res * 2, render_res * 2)
                    grid = Image.new('RGB', grid_size)
                    grid.paste(grid_imgs[0], (0, 0))
                    grid.paste(grid_imgs[1], (render_res, 0))
                    grid.paste(grid_imgs[2], (0, render_res))
                    grid.paste(grid_imgs[3], (render_res, render_res))
                else:
                    grid_size = (render_res * len(grid_imgs), render_res)
                    grid = Image.new('RGB', grid_size)
                    for i, img in enumerate(grid_imgs):
                        grid.paste(img, (render_res * i, 0))
                
                grid.save(str(epoch_dir / "grid_all_views.png"))
            
            # Log to tensorboard
            if n_views > 0:
                img_path = epoch_dir / f"view_0_azim{azimuths[0]}_elev{elevations[0]}.png"
                img = Image.open(str(img_path))
                self.tb.add_image("epoch_render/front_view", np.array(img).transpose(2, 0, 1), global_step=epoch)
                
        except Exception as e:
            print(f"Warning: Failed to save epoch render at epoch {epoch + 1}: {e}")

    @torch.no_grad()
    def save_combined_pca_visualization(
        self,
        epoch: int,
        mesh_obj,
        glctx,
        device,
        dino_model=None,
        n_views: int = 4
    ):
        """
        Save combined 2-step PCA visualization across multiple viewpoints.
        
        This uses the same approach as the purnasai/Dino_V2 paper:
        - Render from multiple viewpoints
        - Extract DINOv2 features from all views
        - Apply COMBINED PCA across all views (not per-view!)
        - This produces semantically consistent coloring across views
        
        Args:
            epoch: Current epoch number
            mesh_obj: The current mesh object
            glctx: nvdiffrast GL context
            device: Torch device
            dino_model: DINOv2 model (loaded once externally for efficiency)
            n_views: Number of viewpoints to render
        """
        # Check if we should save at this epoch
        pca_interval = getattr(self.cfg, 'pca_interval', self.cfg.log_interval_im)
        if bool(getattr(self.cfg, "pca_use_extra_log_epochs", False)):
            extra_epochs = set(int(e) for e in getattr(self.cfg, "extra_log_epochs", []) or [])
        else:
            extra_epochs = set()
        epoch_num = epoch + 1
        if epoch != 0 and epoch_num % pca_interval != 0 and epoch_num not in extra_epochs:
            return
        
        # Check if PCA visualization is enabled
        if not getattr(self.cfg, 'save_pca_visualization', True):
            return
        
        if dino_model is None:
            print(f"  Skipping PCA visualization: no DINO model provided")
            return
        
        try:
            from sklearn.decomposition import PCA
            
            print(f"\n=== Generating Combined PCA Visualization (epoch {epoch + 1}) ===")
            
            render_res = getattr(self.cfg, 'pca_render_res', 518)  # Must be divisible by 14
            
            # Create directory for PCA visualizations
            pca_dir = self.out_path / "pca_visualization" / f"epoch_{epoch + 1:05d}"
            os.makedirs(pca_dir, exist_ok=True)
            
            # Fixed viewpoints
            n_views = max(1, int(n_views))
            azimuths = np.linspace(0, 360, n_views, endpoint=False).tolist()
            elevations = [15.0] * n_views
            
            # Step 1: Render all views and collect images
            rendered_images = []
            original_images = []
            fg_prior_masks = []
            
            for view_idx, (azim, elev) in enumerate(zip(azimuths, elevations)):
                cam_params = get_camera_params(
                    elev,
                    azim,
                    self.cfg.log_dist,
                    render_res,
                    self.cfg.log_fov
                )
                
                for k in cam_params:
                    if isinstance(cam_params[k], torch.Tensor):
                        cam_params[k] = cam_params[k].to(device)
                
                m_eval = mesh.unit_size(mesh_obj.eval(cam_params))
                
                rendered = render.render_mesh(
                    glctx,
                    m_eval,
                    cam_params["mvp"],
                    cam_params["campos"],
                    cam_params["lightpos"],
                    self.cfg.log_light_power,
                    render_res,
                    1,
                    background=torch.ones(1, render_res, render_res, 3, device=device),
                )
                
                # Convert to (B, C, H, W) format for DINO
                img_tensor = rendered.permute(0, 3, 1, 2)  # (1, 3, H, W)
                rendered_images.append(img_tensor)
                
                # Save original render
                img_np = rendered[0].mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                original_images.append(img_np)

                # Foreground prior from white background compositing (used only for polarity check)
                fg_prior = (rendered[0].mean(dim=-1) < 0.99).cpu().numpy().astype(bool)
                fg_prior_masks.append(fg_prior)
            
            # Stack all images
            all_images = torch.cat(rendered_images, dim=0)  # (n_views, 3, H, W)
            
            # Step 2: Extract DINO features from all views
            # Normalize as per original paper: mean=0.5, std=0.2
            mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.2, 0.2, 0.2], device=device).view(1, 3, 1, 1)
            normalized_images = (all_images - mean) / std
            
            # Get patch features
            with torch.no_grad():
                outputs = dino_model.forward_features(normalized_images)
                patch_tokens = outputs['x_norm_patchtokens']  # (n_views, N, D)
            
            # Reshape to spatial grid
            B, N, D = patch_tokens.shape
            patch_size = dino_model.patch_size
            H_patches = render_res // patch_size
            W_patches = render_res // patch_size
            
            features = patch_tokens.reshape(B, H_patches, W_patches, D)  # (n_views, H, W, D)
            
            # Step 3: Apply COMBINED PCA across all views
            features_np = features.cpu().numpy()
            n_patches_per_image = H_patches * W_patches
            
            # Combine all features
            all_features = features_np.reshape(-1, D)  # (n_views * H * W, D)
            
            # First PCA on combined features (exact style from test_image_pca.py)
            pca1 = PCA(n_components=3)
            pca_features1 = pca1.fit_transform(all_features)
            
            # Min-max scale PC1
            pc1 = pca_features1[:, 0]
            pc1_scaled = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
            
            # Fixed threshold from the known-good implementation:
            # bg = pc1_scaled > 0.35, fg = ~bg
            threshold = 0.35
            bg_mask = pc1_scaled > threshold
            fg_mask = ~bg_mask

            # Polarity guard for mesh renders: if fg/bg assignment is flipped,
            # invert masks so the mesh region is treated as foreground. DINO
            # PCA masks live on the patch grid, so downsample the render-space
            # foreground prior to the same H_patches x W_patches layout.
            fg_prior_patch_masks = []
            for prior in fg_prior_masks:
                prior_crop = prior[: H_patches * patch_size, : W_patches * patch_size]
                prior_patch = prior_crop.reshape(
                    H_patches,
                    patch_size,
                    W_patches,
                    patch_size,
                ).mean(axis=(1, 3)) > 0.05
                fg_prior_patch_masks.append(prior_patch)
            all_fg_prior = np.concatenate([m.reshape(-1) for m in fg_prior_patch_masks], axis=0)

            def _iou(a, b):
                inter = np.logical_and(a, b).sum()
                union = np.logical_or(a, b).sum()
                return float(inter) / float(union + 1e-8)

            normal_iou = _iou(fg_mask, all_fg_prior)
            inverted_iou = _iou(bg_mask, all_fg_prior)
            if inverted_iou > normal_iou + 0.02:
                bg_mask, fg_mask = fg_mask, bg_mask
            
            n_fg = fg_mask.sum()
            n_bg = bg_mask.sum()
            print(f"  Foreground: {n_fg} pixels, Background: {n_bg} pixels")
            
            # Second PCA on foreground only (exact style from test_image_pca.py)
            if n_fg > 100:
                pca2 = PCA(n_components=3)
                fg_features = all_features[fg_mask]
                pca_features_fg = pca2.fit_transform(fg_features)
                
                # Min-max scale each component
                for i in range(3):
                    col = pca_features_fg[:, i]
                    pca_features_fg[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)
                
                # Build output colors
                colors = np.zeros((len(all_features), 3), dtype=np.float32)
                colors[bg_mask] = 0.0  # Black background
                colors[fg_mask] = pca_features_fg
            else:
                # Fall back to simple scaling
                colors = np.zeros((len(all_features), 3), dtype=np.float32)
                for i in range(3):
                    col = pca_features1[:, i]
                    colors[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)
            
            # Step 4: Split and save visualizations
            for view_idx in range(n_views):
                start_idx = view_idx * n_patches_per_image
                end_idx = (view_idx + 1) * n_patches_per_image
                img_colors = colors[start_idx:end_idx].reshape(H_patches, W_patches, 3)
                
                # Upsample to render resolution
                pca_tensor = torch.from_numpy(img_colors).permute(2, 0, 1).unsqueeze(0)
                pca_up = F.interpolate(
                    pca_tensor, size=(render_res, render_res), mode='bilinear', align_corners=False
                )
                pca_np = pca_up[0].permute(1, 2, 0).numpy()
                pca_np = (pca_np * 255).clip(0, 255).astype(np.uint8)
                
                # Save PCA only
                pca_img = Image.fromarray(pca_np)
                pca_img.save(str(pca_dir / f"view_{view_idx}_azim{azimuths[view_idx]}_pca.png"))
                
                # Save original + PCA side by side
                orig_img = Image.fromarray(original_images[view_idx])
                combined = Image.new('RGB', (render_res * 2, render_res))
                combined.paste(orig_img, (0, 0))
                combined.paste(pca_img, (render_res, 0))
                combined.save(str(pca_dir / f"view_{view_idx}_combined.png"))
            
            # Create grid of all PCA views (supports arbitrary n_views, including 8)
            cols = min(n_views, 4)
            rows = (n_views + cols - 1) // cols
            grid_size = (render_res * cols, render_res * rows)
            grid = Image.new('RGB', grid_size)
            for i in range(n_views):
                pca_path = pca_dir / f"view_{i}_azim{azimuths[i]}_pca.png"
                pca_img = Image.open(str(pca_path))
                x = (i % cols) * render_res
                y = (i // cols) * render_res
                grid.paste(pca_img, (x, y))
            grid.save(str(pca_dir / "grid_pca_all_views.png"))

            # Log to tensorboard
            self.tb.add_image("pca_visualization/grid", np.array(grid).transpose(2, 0, 1), global_step=epoch)
            
            print(f"  Saved PCA visualizations to {pca_dir}")
            
        except Exception as e:
            print(f"Warning: Failed to save PCA visualization at epoch {epoch + 1}: {e}")
            import traceback
            traceback.print_exc()

    def save_epoch(self, epoch: int, rt: dict, train_render: torch.Tensor):
        if not should_log_epoch(self.cfg, epoch):
            return

        # Get actual batch size from the tensor to avoid index out of bounds
        actual_batch_size = train_render.shape[0]
        
        if self.cfg.log:
            # Clamp to actual batch size to prevent index out of bounds
            max_images = min(15, actual_batch_size)
            idx_list = torch.arange(max_images)
        else:
            max_images = min(5, actual_batch_size)
            idx_list = torch.randperm(actual_batch_size)[:max_images]

        grid = torchvision.utils.make_grid(train_render[idx_list])
        ndarr = (
            grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        )
        Image.fromarray(ndarr).save(self.out_path / "images" / f"epoch_{epoch + 1}.png")

    def save_colored_mesh(
        self,
        epoch: int,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        colors: np.ndarray,
        original_vertices: np.ndarray = None,
        current_mesh: 'mesh.Mesh' = None,
        glctx = None,
        device = None
    ):
        """Save mesh with vertex colors and optional correspondence info."""
        if not should_log_epoch(self.cfg, epoch):
            return
        
        # Get vertex colors based on method
        final_colors = colors  # Default to passed colors (position-based)
        
        if self.color_method == 'dino_pca' and self.dino_pca_colorizer is not None:
            if vertices is not None and faces is not None:
                try:
                    print(f"Generating DINO PCA colors for epoch {epoch + 1}...")
                    # Convert tensors to numpy if needed
                    verts_np = vertices.detach().cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
                    faces_np = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else faces
                    final_colors = self.dino_pca_colorizer.colorize_mesh(
                        verts_np, faces_np, normalize_mesh=True
                    )
                except Exception as e:
                    print(f"Warning: Failed to generate DINO PCA colors, using position colors: {e}")
                    final_colors = colors
            else:
                print("Warning: DINO PCA requires vertices and faces - using position colors")
                final_colors = colors
        
        # Export PLY with colors
        ply_path = self.out_path / "colored_meshes" / f"mesh_epoch_{epoch + 1}.ply"
        export_mesh_with_colors(
            str(ply_path),
            vertices,
            faces,
            final_colors,
            format='ply'
        )
        
        # Export correspondence map
        if original_vertices is not None:
            corr_path = self.out_path / "correspondence" / f"correspondence_epoch_{epoch + 1}.json"
            export_correspondence_map(
                str(corr_path),
                original_vertices,
                vertices,
                faces,
                colors=final_colors,
                metadata={'epoch': epoch + 1}
            )


def loop_with_tracking(cfg):
    """
    Main optimization loop with semantic tracking.
    
    This extends the original MeshUp loop with:
    - Vertex color initialization and tracking
    - Optional DINOv2 feature consistency loss
    - Export of colored meshes throughout optimization
    """
    out_path = pathlib.Path(cfg["output_path"])
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "config.yml", "w") as f:
        yaml.dump(cfg, f)
    cfg = EasyDict(cfg)
    print("Output directory", cfg.output_path)

    device = torch.device(f"cuda:{cfg.gpu}")
    torch.cuda.set_device(device)

    video = Video(cfg.output_path)
    glctx = dr.RasterizeGLContext()

    resize_method = {
        "cubic": cubic,
        "linear": linear,
        "lanczos2": lanczos2,
        "lanczos3": lanczos3,
    }[cfg.resize_method]

    use_sds = bool(getattr(cfg, 'use_sds', True)) and float(getattr(cfg, 'image_weight', 1.0)) > 0.0
    if not use_sds:
        cfg.image_weight = 0.0
        if bool(getattr(cfg, 'use_cross_attn_loss', False)):
            print("Warning: cross-attention guidance requires SDS; disabling cross-attention for no-SDS run.")
            cfg.use_cross_attn_loss = False
        if getattr(cfg, 'score', 'SDS') == "ActvnReplace":
            print("Warning: ActvnReplace score requires diffusion; using zero SDS loss for no-SDS run.")

    def _linear_ramp(epoch: int, ramp_epochs: int, start_factor: float = 0.0) -> float:
        if ramp_epochs <= 0:
            return 1.0
        t = min(max(epoch / float(ramp_epochs), 0.0), 1.0)
        return float(start_factor) + (1.0 - float(start_factor)) * t

    def _cosine_lerp(start: float, end: float, progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        w = 0.5 * (1.0 - np.cos(np.pi * progress))
        return float(start) * (1.0 - w) + float(end) * w

    def _loss_schedule_factors(epoch: int) -> tuple[float, float]:
        schedule = getattr(cfg, 'loss_schedule', 'independent')
        sds_start = float(getattr(cfg, 'loss_schedule_sds_start', 1.0))
        sds_mid = float(getattr(cfg, 'loss_schedule_sds_mid', 0.5))
        target_floor = float(getattr(cfg, 'loss_schedule_target_floor', 1.0))

        if schedule == 'independent':
            return 1.0, 1.0

        if schedule == 'cosine':
            schedule_epochs = int(getattr(cfg, 'loss_schedule_epochs', cfg.epochs))
            progress = epoch / float(max(schedule_epochs, 1))
            sds_factor = _cosine_lerp(sds_start, 1.0, progress)
            target_factor = _cosine_lerp(1.0, target_floor, progress)
            return sds_factor, target_factor

        stage1_epochs = int(getattr(cfg, 'loss_schedule_stage1_epochs', max(cfg.epochs // 3, 1)))

        if schedule == 'two_stage':
            if epoch < stage1_epochs:
                return sds_start, 1.0
            return 1.0, target_floor

        if schedule == 'three_stage':
            stage2_epochs = int(getattr(cfg, 'loss_schedule_stage2_epochs', max(2 * cfg.epochs // 3, stage1_epochs + 1)))
            if epoch < stage1_epochs:
                return sds_start, 1.0
            if epoch < stage2_epochs:
                progress = (epoch - stage1_epochs) / float(max(stage2_epochs - stage1_epochs, 1))
                sds_factor = _cosine_lerp(sds_start, sds_mid, progress)
                target_factor = _cosine_lerp(1.0, target_floor, progress)
                return sds_factor, target_factor
            return 1.0, target_floor

        return 1.0, 1.0

    def _cfg_list(value, cast=None) -> list:
        if value is None:
            return []
        if isinstance(value, str):
            items = [item for item in value.replace(",", " ").split() if item]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            items = [value]
        if cast is None:
            return items
        return [cast(item) for item in items]

    def _partfield_multiscale_stage_starts(n_scales: int) -> list[int]:
        configured = _cfg_list(getattr(cfg, 'partfield_multiscale_stage_epochs', None), int)
        if configured:
            if len(configured) != n_scales:
                raise ValueError(
                    "partfield_multiscale_stage_epochs must have one start epoch per "
                    f"PartField scale ({n_scales}), got {len(configured)}."
                )
            if configured[0] != 0:
                raise ValueError("partfield_multiscale_stage_epochs must start with 0.")
            if any(b <= a for a, b in zip(configured, configured[1:])):
                raise ValueError("partfield_multiscale_stage_epochs must be strictly increasing.")
            return configured
        return [int(round(i * cfg.epochs / float(n_scales))) for i in range(n_scales)]

    def _partfield_multiscale_weights(epoch: int, stage_starts: list[int], blend_epochs: int) -> list[float]:
        n_scales = len(stage_starts)
        if n_scales == 1:
            return [1.0]

        active_idx = 0
        for idx, start_epoch in enumerate(stage_starts):
            if epoch >= start_epoch:
                active_idx = idx

        weights = [0.0 for _ in range(n_scales)]
        if active_idx > 0 and blend_epochs > 0:
            transition_start = stage_starts[active_idx]
            transition_end = transition_start + blend_epochs
            if epoch < transition_end:
                progress = (epoch - transition_start) / float(max(blend_epochs, 1))
                progress = min(max(progress, 0.0), 1.0)
                weights[active_idx - 1] = 1.0 - progress
                weights[active_idx] = progress
                return weights

        weights[active_idx] = 1.0
        return weights

    def _indexed_cfg_list(values: list, idx: int, expected: int, name: str):
        if not values:
            return None
        if len(values) != expected:
            raise ValueError(f"{name} must have {expected} entries, got {len(values)}.")
        return values[idx]

    def _label_path_from_dir(label_dir: str, template_path: str | None) -> str | None:
        if not label_dir or not template_path:
            return None
        base_name = pathlib.Path(template_path).name
        path = pathlib.Path(label_dir)
        if path.name != "labels":
            path = path / "labels"
        return str(path / base_name)

    def _unit_vertices(vertices: torch.Tensor) -> torch.Tensor:
        vmin = vertices.amin(dim=0)
        vmax = vertices.amax(dim=0)
        scale = 2.0 / torch.clamp((vmax - vmin).max(), min=1e-8)
        return (vertices - (vmax + vmin) * 0.5) * scale

    def _load_target_vertices(path: str, dev: torch.device) -> torch.Tensor:
        target_ms = pymeshlab.MeshSet()
        target_ms.load_new_mesh(path)
        target_vertices = torch.tensor(
            target_ms.current_mesh().vertex_matrix(),
            dtype=torch.float32,
            device=dev,
        )
        return _unit_vertices(target_vertices)

    def _load_target_mesh_arrays(path: str, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        target_ms = pymeshlab.MeshSet()
        target_ms.load_new_mesh(path)
        target_mesh = target_ms.current_mesh()
        target_vertices = torch.tensor(
            target_mesh.vertex_matrix(),
            dtype=torch.float32,
            device=dev,
        )
        target_faces = torch.tensor(
            target_mesh.face_matrix(),
            dtype=torch.long,
            device=dev,
        )
        return _unit_vertices(target_vertices), target_faces

    def _sample_vertices(vertices: torch.Tensor, n_points: int) -> torch.Tensor:
        if n_points <= 0 or vertices.shape[0] <= n_points:
            return vertices
        idx = torch.linspace(
            0,
            vertices.shape[0] - 1,
            steps=n_points,
            device=vertices.device,
        ).long()
        return vertices[idx]

    def _symmetric_chamfer(src_vertices: torch.Tensor, tgt_vertices: torch.Tensor, n_points: int) -> torch.Tensor:
        src = _sample_vertices(_unit_vertices(src_vertices), n_points)
        tgt = _sample_vertices(tgt_vertices, n_points)
        dists = torch.cdist(src, tgt, p=2).pow(2)
        return dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()

    def _face_adjacency_pairs(faces: torch.Tensor, dev: torch.device) -> torch.Tensor:
        edge_to_face = {}
        pairs = []
        for face_idx, face in enumerate(faces.detach().cpu().tolist()):
            a, b, c = (int(face[0]), int(face[1]), int(face[2]))
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                prev_face = edge_to_face.get(key)
                if prev_face is None:
                    edge_to_face[key] = face_idx
                else:
                    pairs.append((prev_face, face_idx))
        if not pairs:
            return torch.empty((0, 2), dtype=torch.long, device=dev)
        return torch.tensor(pairs, dtype=torch.long, device=dev)

    def _unique_edge_pairs(faces: torch.Tensor, dev: torch.device) -> torch.Tensor:
        edges = torch.cat(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ],
            dim=0,
        )
        edges = torch.sort(edges, dim=1).values
        return torch.unique(edges, dim=0).to(dev)

    # misc dirs
    os.makedirs(out_path / "tmp", exist_ok=True)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(cfg.mesh)

    # prompts
    txt = cfg.text_prompt
    print("Target text prompt:", txt)
    df = None
    text_embeds = None
    prompt_num = 1
    if use_sds:
        df = DeepFloydGuidance(cfg, device)
        if isinstance(txt, list):
            prompts = [p + ", a 3d rendering" if i != len(txt) - 1 else p for i, p in enumerate(txt)]
        else:
            prompts = [txt + ", a 3d rendering"]
        text_embeds = df.encode_text_2(prompts, batch_size=cfg.batch_size).to(device)
        prompt_num = len(prompts)
    else:
        print("SDS/diffusion disabled: skipping DeepFloyd pipeline load and text embeddings.")

    # mesh prep
    if cfg.retriangulate:
        ms.meshing_isotropic_explicit_remeshing()
    if not ms.current_mesh().has_wedge_tex_coord():
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
    ms.save_current_mesh(str(out_path / "tmp" / "mesh.obj"))
    load_mesh = obj.load_obj(str(out_path / "tmp" / "mesh.obj"))
    load_mesh = mesh.unit_size(load_mesh)
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=load_mesh.v_pos.cpu().numpy(), face_matrix=load_mesh.t_pos_idx.cpu().numpy()))
    ms.save_current_mesh(str(out_path / "tmp" / "mesh.obj"), save_vertex_color=False)

    # =========================================================================
    # MESH STATISTICS
    # =========================================================================
    n_vertices = load_mesh.v_pos.shape[0]
    n_faces = load_mesh.t_pos_idx.shape[0]
    print(f"\n{'='*60}")
    print(f"MESH STATISTICS:")
    print(f"  - Number of vertices: {n_vertices}")
    print(f"  - Number of faces: {n_faces}")
    print(f"  - Vertex position range:")
    print(f"      X: [{load_mesh.v_pos[:, 0].min().item():.4f}, {load_mesh.v_pos[:, 0].max().item():.4f}]")
    print(f"      Y: [{load_mesh.v_pos[:, 1].min().item():.4f}, {load_mesh.v_pos[:, 1].max().item():.4f}]")
    print(f"      Z: [{load_mesh.v_pos[:, 2].min().item():.4f}, {load_mesh.v_pos[:, 2].max().item():.4f}]")
    print(f"{'='*60}\n")

    # =========================================================================
    # SEMANTIC TRACKING INITIALIZATION
    # =========================================================================
    track_correspondence = getattr(cfg, 'track_correspondence', True)
    color_method = getattr(cfg, 'color_method', 'position')
    
    if track_correspondence:
        print(f"Initializing semantic tracking with method: {color_method}")
        
        # For DINO PCA, we don't use VertexColorTracker (requires rendering)
        # Instead, we'll use position-based colors as fallback and generate
        # DINO PCA colors during training when we can render
        if color_method == 'dino_pca':
            print("DINO PCA coloring will be generated during training (requires rendering)")
            print("Using position-based colors as fallback for initial mesh")
            tracker = initialize_semantic_colors(
                load_mesh.v_pos.cpu().numpy(),
                load_mesh.t_pos_idx.cpu().numpy(),
                method='position',
                n_parts=getattr(cfg, 'n_parts', 8)
            )
        else:
            tracker = initialize_semantic_colors(
                load_mesh.v_pos.cpu().numpy(),
                load_mesh.t_pos_idx.cpu().numpy(),
                method=color_method,
                n_parts=getattr(cfg, 'n_parts', 8)
            )
        
        original_vertices = load_mesh.v_pos.cpu().numpy().copy()
        vertex_colors = tracker.vertex_colors
        
        # Save initial colored mesh
        initial_ply_path = out_path / "colored_meshes" / "mesh_initial.ply"
        os.makedirs(out_path / "colored_meshes", exist_ok=True)
        export_mesh_with_colors(
            str(initial_ply_path),
            original_vertices,
            load_mesh.t_pos_idx.cpu().numpy(),
            vertex_colors,
            format='ply'
        )
        print(f"Saved initial colored mesh to {initial_ply_path}")
    else:
        tracker = None
        original_vertices = None
        vertex_colors = None
    
    # =========================================================================
    # Optional DINOv2 Feature Consistency Loss (Improved Version)
    # =========================================================================
    use_dino_loss = getattr(cfg, 'use_dino_loss', False)
    dino_loss_fn = None
    
    if use_dino_loss and DINO_AVAILABLE:
        try:
            dino_loss_fn = create_dino_correspondence_loss(
                device=str(device),
                model_name=getattr(cfg, 'dino_model', 'dinov2_vits14_reg'),
                weight=getattr(cfg, 'dino_weight', 0.1),
                warmup_epochs=getattr(cfg, 'dino_warmup_epochs', 100),
                global_weight=getattr(cfg, 'dino_global_weight', 0.3),
                spatial_weight=getattr(cfg, 'dino_spatial_weight', 0.7),
                n_reference_views=getattr(cfg, 'dino_n_views', 8),
                use_soft_matching=getattr(cfg, 'dino_soft_matching', True),
                temperature=getattr(cfg, 'dino_temperature', 0.1),
            )
            print(f"DINOv2 correspondence loss enabled:")
            print(f"  - Weight: {cfg.dino_weight}")
            print(f"  - Warmup epochs: {getattr(cfg, 'dino_warmup_epochs', 100)}")
            print(f"  - Global/Spatial weights: {getattr(cfg, 'dino_global_weight', 0.3)}/{getattr(cfg, 'dino_spatial_weight', 0.7)}")
        except Exception as e:
            print(f"Warning: Could not initialize DINOv2 loss: {e}")
            import traceback
            traceback.print_exc()
            use_dino_loss = False
    elif use_dino_loss and not DINO_AVAILABLE:
        print("Warning: DINOv2 loss requested but module not available")
        use_dino_loss = False
    
    # =========================================================================
    # Optional Cross-Attention Semantic Guidance (Standalone from DINO)
    # =========================================================================
    use_cross_attn_loss = getattr(cfg, 'use_cross_attn_loss', False)
    cross_attn_fn = None
    
    if use_cross_attn_loss and CROSS_ATTN_AVAILABLE:
        try:
            cross_attn_fn = create_cross_attention_guidance(
                device=str(device),
                weight=getattr(cfg, 'cross_attn_weight', 0.1),
                warmup_epochs=getattr(cfg, 'cross_attn_warmup_epochs', 50),
                consistency_weight=getattr(cfg, 'cross_attn_consistency_weight', 0.5),
                entropy_weight=getattr(cfg, 'cross_attn_entropy_weight', 0.3),
                coverage_weight=getattr(cfg, 'cross_attn_coverage_weight', 0.2),
                use_consistency_guidance=getattr(cfg, 'cross_attn_use_consistency', True),
                use_entropy_guidance=getattr(cfg, 'cross_attn_use_entropy', True),
                use_coverage_guidance=getattr(cfg, 'cross_attn_use_coverage', True),
            )
            # Set prompt info for token analysis
            target_prompt = cfg.prompts[-1] if hasattr(cfg, 'prompts') else cfg.prompt
            cross_attn_fn.set_prompt_info(df.pipe.tokenizer, target_prompt)
            
            print(f"Cross-Attention guidance enabled:")
            print(f"  - Weight: {getattr(cfg, 'cross_attn_weight', 0.1)}")
            print(f"  - Warmup epochs: {getattr(cfg, 'cross_attn_warmup_epochs', 50)}")
            print(f"  - Target prompt: '{target_prompt}'")
        except Exception as e:
            print(f"Warning: Could not initialize Cross-Attention guidance: {e}")
            import traceback
            traceback.print_exc()
            use_cross_attn_loss = False
    elif use_cross_attn_loss and not CROSS_ATTN_AVAILABLE:
        print("Warning: Cross-Attention guidance requested but module not available")
        use_cross_attn_loss = False
    
    # =========================================================================
    # DINO PCA Visualization (for coloring meshes with semantic features)
    # =========================================================================
    dino_pca_colorizer = None
    
    if color_method == 'dino_pca' and DINO_PCA_AVAILABLE:
        try:
            dino_pca_colorizer = DINOPCAColorizer(
                dino_model_name=getattr(cfg, 'dino_pca_model', 'dinov2_vits14'),
                n_views=getattr(cfg, 'dino_pca_n_views', 8),
                image_size=getattr(cfg, 'dino_pca_image_size', 512),
                device=str(device)
            )
            print(f"DINO PCA visualization enabled:")
            print(f"  - Model: {getattr(cfg, 'dino_pca_model', 'dinov2_vits14')}")
            print(f"  - Views: {getattr(cfg, 'dino_pca_n_views', 8)}")
            print(f"  - Image size: {getattr(cfg, 'dino_pca_image_size', 512)}")
        except Exception as e:
            print(f"Warning: Could not initialize DINO PCA colorizer: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to position-based coloring
            color_method = 'position'
    elif color_method == 'dino_pca' and not DINO_PCA_AVAILABLE:
        print("Warning: DINO PCA requested but module not available, falling back to position coloring")
        color_method = 'position'
    
    # =========================================================================
    # TARGET MESH DINO GUIDANCE V2 (viewpoint-aligned)
    # =========================================================================
    use_target_mesh_guidance = getattr(cfg, 'use_target_mesh_guidance', False)
    target_mesh_guidance_fn = None
    use_v2_guidance = True  # Always use V2 (viewpoint-aligned) version
    
    if use_target_mesh_guidance:
        target_mesh_path = getattr(cfg, 'target_mesh', None)
        if target_mesh_path is not None:
            # Try V2 first (viewpoint-aligned), fall back to V1
            if use_v2_guidance and TARGET_MESH_GUIDANCE_V2_AVAILABLE:
                try:
                    target_mesh_guidance_fn = create_target_mesh_guidance_v2(
                        target_mesh_path=target_mesh_path,
                        device=str(device),
                        weight=getattr(cfg, 'target_mesh_weight', 0.5),
                        warmup_epochs=getattr(cfg, 'target_mesh_warmup_epochs', 100),
                        model_name=getattr(cfg, 'target_mesh_dino_model', 'dinov2_vits14_reg'),
                        n_azimuths=getattr(cfg, 'target_mesh_n_azimuths', 12),
                        n_elevations=getattr(cfg, 'target_mesh_n_elevations', 3),
                        global_weight=getattr(cfg, 'target_mesh_global_weight', 0.3),
                        spatial_weight=getattr(cfg, 'target_mesh_spatial_weight', 0.7),
                        render_weight=getattr(cfg, 'target_mesh_render_weight', 0.0),
                        online_target_render=getattr(cfg, 'target_mesh_online_render', False),
                        online_cache_features=getattr(cfg, 'target_mesh_online_cache', True),
                        online_cache_max_size=getattr(cfg, 'target_mesh_online_cache_max', 4096),
                        view_rounding_deg=getattr(cfg, 'target_mesh_view_rounding_deg', 2.0),
                        view_rounding_dist=getattr(cfg, 'target_mesh_view_rounding_dist', 0.05),
                        view_rounding_fov=getattr(cfg, 'target_mesh_view_rounding_fov', 1.0),
                    )
                    print(f"Target Mesh DINO Guidance V2 (viewpoint-aligned) enabled:")
                    print(f"  - Target mesh: {target_mesh_path}")
                    print(f"  - Weight: {getattr(cfg, 'target_mesh_weight', 0.5)}")
                    print(f"  - Viewpoints: {getattr(cfg, 'target_mesh_n_azimuths', 12)} x {getattr(cfg, 'target_mesh_n_elevations', 3)}")
                    print(f"  - Warmup epochs: {getattr(cfg, 'target_mesh_warmup_epochs', 100)}")
                except Exception as e:
                    print(f"Warning: Could not initialize Target Mesh guidance V2: {e}")
                    import traceback
                    traceback.print_exc()
                    use_target_mesh_guidance = False
            elif TARGET_MESH_GUIDANCE_AVAILABLE:
                # Fall back to V1
                try:
                    target_mesh_guidance_fn = create_target_mesh_guidance(
                        target_mesh_path=target_mesh_path,
                        device=str(device),
                        weight=getattr(cfg, 'target_mesh_weight', 0.5),
                        warmup_epochs=getattr(cfg, 'target_mesh_warmup_epochs', 100),
                        model_name=getattr(cfg, 'target_mesh_dino_model', 'dinov2_vits14_reg'),
                        n_views=getattr(cfg, 'target_mesh_n_views', 8),
                        global_weight=getattr(cfg, 'target_mesh_global_weight', 0.3),
                        spatial_weight=getattr(cfg, 'target_mesh_spatial_weight', 0.7),
                    )
                    print(f"Target Mesh DINO Guidance V1 enabled (warning: not viewpoint-aligned)")
                except Exception as e:
                    print(f"Warning: Could not initialize Target Mesh guidance: {e}")
                    use_target_mesh_guidance = False
            else:
                print("Warning: Target Mesh guidance requested but no module available")
                use_target_mesh_guidance = False
        else:
            print("Warning: use_target_mesh_guidance=True but no target_mesh path provided")
            use_target_mesh_guidance = False

    # =========================================================================
    # TARGET MESH GEOMETRY GUIDANCE
    # =========================================================================
    target_mesh_vertices = None
    target_mesh_faces = None
    target_chamfer_weight = float(getattr(cfg, 'target_mesh_chamfer_weight', 0.0))
    target_chamfer_points = int(getattr(cfg, 'target_mesh_chamfer_points', 2048))
    target_chamfer_warmup_epochs = int(
        getattr(cfg, 'target_mesh_chamfer_warmup_epochs', getattr(cfg, 'target_mesh_warmup_epochs', 0))
    )
    target_vertex_corr_weight = float(getattr(cfg, 'target_mesh_vertex_correspondence_weight', 0.0))
    target_vertex_corr_warmup_epochs = int(
        getattr(cfg, 'target_mesh_vertex_correspondence_warmup_epochs', target_chamfer_warmup_epochs)
    )
    target_vertex_corr_available = False
    target_semantic_vertex_corr_weight = float(
        getattr(cfg, 'target_mesh_semantic_vertex_correspondence_weight', 0.0)
    )
    target_semantic_vertex_corr_warmup_epochs = int(
        getattr(
            cfg,
            'target_mesh_semantic_vertex_correspondence_warmup_epochs',
            target_chamfer_warmup_epochs,
        )
    )
    target_semantic_vertex_corr_fn = None
    target_semantic_vertex_corr_metadata = None
    target_part_chamfer_weight = float(getattr(cfg, 'target_mesh_part_chamfer_weight', 0.0))
    target_part_chamfer_points = int(getattr(cfg, 'target_mesh_part_chamfer_points', 512))
    target_part_chamfer_warmup_epochs = int(
        getattr(cfg, 'target_mesh_part_chamfer_warmup_epochs', target_chamfer_warmup_epochs)
    )
    target_part_chamfer_fn = None
    target_semantic_bucket_weight = float(getattr(cfg, 'target_mesh_semantic_bucket_chamfer_weight', 0.0))
    target_semantic_bucket_points = int(getattr(cfg, 'target_mesh_semantic_bucket_chamfer_points', 512))
    target_semantic_bucket_warmup_epochs = int(
        getattr(cfg, 'target_mesh_semantic_bucket_chamfer_warmup_epochs', target_chamfer_warmup_epochs)
    )
    target_semantic_bucket_fn = None
    target_sampart3d_weight = float(getattr(cfg, 'target_mesh_sampart3d_chamfer_weight', 0.0))
    target_sampart3d_points = int(getattr(cfg, 'target_mesh_sampart3d_chamfer_points', 512))
    target_sampart3d_warmup_epochs = int(
        getattr(cfg, 'target_mesh_sampart3d_chamfer_warmup_epochs', target_chamfer_warmup_epochs)
    )
    target_sampart3d_fn = None
    target_partfield_weight = float(getattr(cfg, 'target_mesh_partfield_chamfer_weight', 0.0))
    target_partfield_points = int(getattr(cfg, 'target_mesh_partfield_chamfer_points', 512))
    target_partfield_warmup_epochs = int(
        getattr(cfg, 'target_mesh_partfield_chamfer_warmup_epochs', target_chamfer_warmup_epochs)
    )
    target_partfield_fn = None
    target_partfield_scales = []
    target_partfield_stage_starts = []
    target_partfield_blend_epochs = int(getattr(cfg, 'partfield_multiscale_blend_epochs', 0))

    if (
        target_chamfer_weight > 0
        or target_vertex_corr_weight > 0
        or target_semantic_vertex_corr_weight > 0
        or target_part_chamfer_weight > 0
        or target_semantic_bucket_weight > 0
        or target_sampart3d_weight > 0
        or target_partfield_weight > 0
    ):
        target_mesh_path = getattr(cfg, 'target_mesh', None)
        if target_mesh_path is None:
            print("Warning: target mesh geometry guidance requested but no target_mesh path provided")
            target_chamfer_weight = 0.0
            target_vertex_corr_weight = 0.0
            target_semantic_vertex_corr_weight = 0.0
            target_part_chamfer_weight = 0.0
            target_semantic_bucket_weight = 0.0
            target_sampart3d_weight = 0.0
            target_partfield_weight = 0.0
        else:
            try:
                target_mesh_vertices, target_mesh_faces = _load_target_mesh_arrays(target_mesh_path, device)
                if target_chamfer_weight > 0:
                    print("Target mesh Chamfer guidance enabled:")
                    print(f"  - Target vertices: {target_mesh_vertices.shape[0]}")
                    print(f"  - Sampled points: {target_chamfer_points}")
                    print(f"  - Weight: {target_chamfer_weight}")
                    print(f"  - Warmup epochs: {target_chamfer_warmup_epochs}")
            except Exception as e:
                print(f"Warning: Could not initialize target mesh geometry guidance: {e}")
                import traceback
                traceback.print_exc()
                target_chamfer_weight = 0.0
                target_vertex_corr_weight = 0.0
                target_semantic_vertex_corr_weight = 0.0
                target_part_chamfer_weight = 0.0
                target_semantic_bucket_weight = 0.0
                target_sampart3d_weight = 0.0
                target_partfield_weight = 0.0
                target_mesh_vertices = None
                target_mesh_faces = None

    if target_vertex_corr_weight > 0:
        if target_mesh_vertices is None or target_mesh_faces is None:
            print("Warning: vertex-correspondence target loss requested but target mesh arrays were not loaded")
            target_vertex_corr_weight = 0.0
        else:
            source_faces_for_match = load_mesh.t_pos_idx.to(device)
            same_vertices = target_mesh_vertices.shape[0] == load_mesh.v_pos.shape[0]
            same_faces = target_mesh_faces.shape == source_faces_for_match.shape and torch.equal(
                target_mesh_faces,
                source_faces_for_match,
            )
            if same_vertices and same_faces:
                target_vertex_corr_available = True
                print("Topology-matched target vertex correspondence enabled:")
                print(f"  - Vertices: {target_mesh_vertices.shape[0]}")
                print(f"  - Weight: {target_vertex_corr_weight}")
                print(f"  - Warmup epochs: {target_vertex_corr_warmup_epochs}")
            else:
                print("Warning: vertex-correspondence target loss disabled because source/target topology differs")
                print(f"  - Source vertices/faces: {load_mesh.v_pos.shape[0]} / {source_faces_for_match.shape[0]}")
                print(f"  - Target vertices/faces: {target_mesh_vertices.shape[0]} / {target_mesh_faces.shape[0]}")
                target_vertex_corr_weight = 0.0

    if target_semantic_vertex_corr_weight > 0:
        if not SEMANTIC_VERTEX_CORRESPONDENCE_AVAILABLE:
            print("Warning: semantic vertex correspondence requested but module is not available")
            target_semantic_vertex_corr_weight = 0.0
        elif target_mesh_vertices is None or target_mesh_faces is None:
            print("Warning: semantic vertex correspondence requested but target mesh arrays were not loaded")
            target_semantic_vertex_corr_weight = 0.0
        else:
            source_features_path = getattr(cfg, 'partfield_source_features', None)
            target_features_path = getattr(cfg, 'partfield_target_features', None)
            if source_features_path is None or target_features_path is None:
                print("Warning: semantic vertex correspondence needs partfield_source_features and partfield_target_features")
                target_semantic_vertex_corr_weight = 0.0
            else:
                try:
                    semantic_vcorr_cache = getattr(cfg, 'semantic_vertex_correspondence_cache', None)
                    if semantic_vcorr_cache is None:
                        semantic_vcorr_cache = str(out_path / "semantic_vertex_correspondence" / "source_to_target.npz")
                    target_semantic_vertex_corr_fn, semantic_vcorr = create_semantic_vertex_correspondence_loss(
                        source_vertices=load_mesh.v_pos.to(device),
                        source_faces=load_mesh.t_pos_idx.to(device),
                        target_vertices=target_mesh_vertices,
                        target_faces=target_mesh_faces,
                        source_features_path=source_features_path,
                        target_features_path=target_features_path,
                        source_labels_path=getattr(cfg, 'partfield_source_labels', None),
                        target_labels_path=getattr(cfg, 'partfield_target_labels', None),
                        feature_mode=getattr(cfg, 'partfield_feature_mode', 'auto'),
                        label_mode=getattr(cfg, 'partfield_label_mode', 'auto'),
                        label_filter=getattr(cfg, 'semantic_vertex_correspondence_label_filter', 'soft'),
                        semantic_weight=float(getattr(cfg, 'semantic_vertex_correspondence_semantic_weight', 1.0)),
                        position_weight=float(getattr(cfg, 'semantic_vertex_correspondence_position_weight', 0.20)),
                        normal_weight=float(getattr(cfg, 'semantic_vertex_correspondence_normal_weight', 0.05)),
                        label_mismatch_penalty=float(
                            getattr(cfg, 'semantic_vertex_correspondence_label_mismatch_penalty', 0.25)
                        ),
                        topk=int(getattr(cfg, 'semantic_vertex_correspondence_topk', 32)),
                        min_similarity=float(getattr(cfg, 'semantic_vertex_correspondence_min_similarity', 0.05)),
                        confidence_margin=float(
                            getattr(cfg, 'semantic_vertex_correspondence_confidence_margin', 0.04)
                        ),
                        confidence_floor=float(
                            getattr(cfg, 'semantic_vertex_correspondence_confidence_floor', 0.25)
                        ),
                        nonmutual_weight=float(
                            getattr(cfg, 'semantic_vertex_correspondence_nonmutual_weight', 0.60)
                        ),
                        topology_prior_weight=float(
                            getattr(cfg, 'semantic_vertex_correspondence_topology_prior_weight', 0.0)
                        ),
                        cache_path=semantic_vcorr_cache,
                        rebuild_cache=bool(getattr(cfg, 'semantic_vertex_correspondence_rebuild_cache', False)),
                    )
                    target_semantic_vertex_corr_metadata = semantic_vcorr.metadata
                    print("Semantic target vertex correspondence enabled:")
                    print(f"  - Source features: {source_features_path}")
                    print(f"  - Target features: {target_features_path}")
                    print(f"  - Cache: {semantic_vcorr_cache}")
                    print(f"  - Weight: {target_semantic_vertex_corr_weight}")
                    print(f"  - Warmup epochs: {target_semantic_vertex_corr_warmup_epochs}")
                    for key in (
                        "same_topology",
                        "mean_similarity",
                        "p05_similarity",
                        "mean_weight",
                        "p05_weight",
                        "mutual_fraction",
                        "identity_fraction",
                        "identity_prior_kept_fraction",
                        "unique_target_fraction",
                        "label_agreement_fraction",
                    ):
                        if key in target_semantic_vertex_corr_metadata:
                            print(f"  - {key}: {target_semantic_vertex_corr_metadata[key]}")
                except Exception as e:
                    print(f"Warning: Could not initialize semantic vertex correspondence guidance: {e}")
                    import traceback
                    traceback.print_exc()
                    target_semantic_vertex_corr_weight = 0.0
                    target_semantic_vertex_corr_fn = None
                    target_semantic_vertex_corr_metadata = None

    if target_part_chamfer_weight > 0:
        if not PART_AWARE_CHAMFER_AVAILABLE:
            print("Warning: part-aware Chamfer requested but module is not available")
            target_part_chamfer_weight = 0.0
        elif target_mesh_vertices is None:
            print("Warning: part-aware Chamfer requested but target vertices were not loaded")
            target_part_chamfer_weight = 0.0
        else:
            try:
                target_part_chamfer_fn = create_part_aware_chamfer_loss(
                    source_reference_vertices=load_mesh.v_pos.to(device),
                    target_vertices=target_mesh_vertices,
                    schema=getattr(cfg, 'target_mesh_part_chamfer_schema', 'car'),
                    longitudinal_axis=getattr(cfg, 'target_mesh_part_chamfer_long_axis', 'z'),
                    lateral_axis=getattr(cfg, 'target_mesh_part_chamfer_lateral_axis', 'x'),
                    vertical_axis=getattr(cfg, 'target_mesh_part_chamfer_vertical_axis', 'y'),
                    points_per_part=target_part_chamfer_points,
                    min_points_per_part=int(getattr(cfg, 'target_mesh_part_chamfer_min_points', 12)),
                    global_weight=float(getattr(cfg, 'target_mesh_part_chamfer_global_weight', 0.0)),
                    flip_target_longitudinal=bool(getattr(cfg, 'target_mesh_part_chamfer_flip_target_longitudinal', False)),
                )
                print("Target mesh part-aware Chamfer guidance enabled:")
                print(f"  - Schema: {getattr(cfg, 'target_mesh_part_chamfer_schema', 'car')}")
                print(f"  - Weight: {target_part_chamfer_weight}")
                print(f"  - Points per part: {target_part_chamfer_points}")
                print(f"  - Active parts: {len(target_part_chamfer_fn.active_part_ids)} / {len(target_part_chamfer_fn.part_names)}")
                print(f"  - Warmup epochs: {target_part_chamfer_warmup_epochs}")
                for part_name, counts in target_part_chamfer_fn.part_counts().items():
                    status = "active" if counts["active"] else "skip"
                    print(f"    {part_name}: source={counts['source']} target={counts['target']} {status}")
            except Exception as e:
                print(f"Warning: Could not initialize part-aware Chamfer guidance: {e}")
                import traceback
                traceback.print_exc()
                target_part_chamfer_weight = 0.0
                target_part_chamfer_fn = None

    if target_semantic_bucket_weight > 0:
        if not SEMANTIC_BUCKET_CHAMFER_AVAILABLE:
            print("Warning: rendered-DINO semantic bucket Chamfer requested but module is not available")
            target_semantic_bucket_weight = 0.0
        elif target_mesh_vertices is None or target_mesh_faces is None:
            print("Warning: rendered-DINO semantic bucket Chamfer requested but target mesh arrays were not loaded")
            target_semantic_bucket_weight = 0.0
        else:
            try:
                semantic_cache = getattr(cfg, 'semantic_bucket_cache', None)
                if semantic_cache is None:
                    semantic_cache = str(out_path / "semantic_buckets" / "rendered_dino_bucket_labels.npz")
                target_semantic_bucket_fn = create_rendered_dino_semantic_chamfer_loss(
                    source_vertices=load_mesh.v_pos.to(device),
                    source_faces=load_mesh.t_pos_idx.to(device),
                    target_vertices=target_mesh_vertices,
                    target_faces=target_mesh_faces,
                    glctx=glctx,
                    device=str(device),
                    model_name=getattr(cfg, 'semantic_bucket_dino_model', 'dinov2_vits14_reg'),
                    n_buckets=int(getattr(cfg, 'semantic_bucket_n_buckets', 10)),
                    n_views=int(getattr(cfg, 'semantic_bucket_n_views', 8)),
                    image_size=int(getattr(cfg, 'semantic_bucket_image_size', 224)),
                    points_per_bucket=target_semantic_bucket_points,
                    min_points_per_bucket=int(getattr(cfg, 'semantic_bucket_min_points', 12)),
                    position_weight=float(getattr(cfg, 'semantic_bucket_position_weight', 0.25)),
                    normal_weight=float(getattr(cfg, 'semantic_bucket_normal_weight', 0.10)),
                    global_weight=float(getattr(cfg, 'target_mesh_semantic_bucket_chamfer_global_weight', 0.0)),
                    cache_path=semantic_cache,
                    random_state=int(getattr(cfg, 'seed', 42)),
                )
                print("Rendered-DINO semantic bucket Chamfer guidance enabled:")
                print(f"  - Weight: {target_semantic_bucket_weight}")
                print(f"  - Buckets: {target_semantic_bucket_fn.n_buckets}")
                print(f"  - Active buckets: {len(target_semantic_bucket_fn.active_bucket_ids)} / {target_semantic_bucket_fn.n_buckets}")
                print(f"  - Points per bucket: {target_semantic_bucket_points}")
                print(f"  - Warmup epochs: {target_semantic_bucket_warmup_epochs}")
                for bucket_name, counts in target_semantic_bucket_fn.bucket_counts().items():
                    status = "active" if counts["active"] else "skip"
                    print(f"    {bucket_name}: source={counts['source']} target={counts['target']} {status}")
            except Exception as e:
                print(f"Warning: Could not initialize rendered-DINO semantic bucket Chamfer guidance: {e}")
                import traceback
                traceback.print_exc()
                target_semantic_bucket_weight = 0.0
                target_semantic_bucket_fn = None

    if target_sampart3d_weight > 0:
        if not SAMPART3D_LABEL_CHAMFER_AVAILABLE:
            print("Warning: SAMPart3D label Chamfer requested but module is not available")
            target_sampart3d_weight = 0.0
        elif target_mesh_vertices is None or target_mesh_faces is None:
            print("Warning: SAMPart3D label Chamfer requested but target mesh arrays were not loaded")
            target_sampart3d_weight = 0.0
        else:
            source_labels_path = getattr(cfg, 'sampart3d_source_labels', None)
            target_labels_path = getattr(cfg, 'sampart3d_target_labels', None)
            if source_labels_path is None or target_labels_path is None:
                print("Warning: SAMPart3D label Chamfer requested but source/target label paths are missing")
                target_sampart3d_weight = 0.0
            else:
                try:
                    debug_path = getattr(cfg, 'sampart3d_debug_path', None)
                    if debug_path is None:
                        debug_path = str(out_path / "sampart3d_labels" / "bucket_labels.npz")
                    target_sampart3d_fn = create_sampart3d_label_chamfer_loss(
                        source_vertices=load_mesh.v_pos.to(device),
                        source_faces=load_mesh.t_pos_idx.to(device),
                        target_vertices=target_mesh_vertices,
                        target_faces=target_mesh_faces,
                        source_labels_path=source_labels_path,
                        target_labels_path=target_labels_path,
                        label_mode=getattr(cfg, 'sampart3d_label_mode', 'auto'),
                        points_per_bucket=target_sampart3d_points,
                        min_points_per_bucket=int(getattr(cfg, 'sampart3d_min_points', 12)),
                        global_weight=float(getattr(cfg, 'target_mesh_sampart3d_chamfer_global_weight', 0.0)),
                        centroid_weight=float(getattr(cfg, 'sampart3d_match_centroid_weight', 1.0)),
                        extent_weight=float(getattr(cfg, 'sampart3d_match_extent_weight', 0.35)),
                        size_weight=float(getattr(cfg, 'sampart3d_match_size_weight', 0.10)),
                        debug_path=debug_path,
                    )
                    print("SAMPart3D-label Chamfer guidance enabled:")
                    print(f"  - Source labels: {source_labels_path}")
                    print(f"  - Target labels: {target_labels_path}")
                    print(f"  - Weight: {target_sampart3d_weight}")
                    print(f"  - Buckets: {target_sampart3d_fn.n_buckets}")
                    print(f"  - Active buckets: {len(target_sampart3d_fn.active_bucket_ids)} / {target_sampart3d_fn.n_buckets}")
                    print(f"  - Points per bucket: {target_sampart3d_points}")
                    print(f"  - Warmup epochs: {target_sampart3d_warmup_epochs}")
                    for bucket_name, counts in target_sampart3d_fn.bucket_counts().items():
                        status = "active" if counts["active"] else "skip"
                        print(f"    {bucket_name}: source={counts['source']} target={counts['target']} {status}")
                except Exception as e:
                    print(f"Warning: Could not initialize SAMPart3D-label Chamfer guidance: {e}")
                    import traceback
                    traceback.print_exc()
                    target_sampart3d_weight = 0.0
                    target_sampart3d_fn = None

    if target_partfield_weight > 0:
        if not PARTFIELD_CHAMFER_AVAILABLE:
            print("Warning: PartField Chamfer requested but module is not available")
            target_partfield_weight = 0.0
        elif target_mesh_vertices is None or target_mesh_faces is None:
            print("Warning: PartField Chamfer requested but target mesh arrays were not loaded")
            target_partfield_weight = 0.0
        else:
            try:
                multiscale_buckets = _cfg_list(getattr(cfg, 'partfield_multiscale_buckets', None), int)
                multiscale_enabled_value = getattr(cfg, 'partfield_multiscale_enabled', None)
                if multiscale_enabled_value is None:
                    multiscale_requested = len(multiscale_buckets) > 1
                else:
                    multiscale_requested = bool(multiscale_enabled_value)
                if not multiscale_buckets:
                    multiscale_buckets = [int(getattr(cfg, 'partfield_n_buckets', 12))]
                if not multiscale_requested:
                    multiscale_buckets = [int(getattr(cfg, 'partfield_n_buckets', 12))]
                if len(multiscale_buckets) == 1:
                    multiscale_requested = False

                n_scales = len(multiscale_buckets)
                source_label_paths = _cfg_list(getattr(cfg, 'partfield_multiscale_source_labels', None), str)
                target_label_paths = _cfg_list(getattr(cfg, 'partfield_multiscale_target_labels', None), str)
                label_dirs = _cfg_list(getattr(cfg, 'partfield_multiscale_label_dirs', None), str)
                default_source_labels = getattr(cfg, 'partfield_source_labels', None)
                default_target_labels = getattr(cfg, 'partfield_target_labels', None)
                default_n_buckets = int(getattr(cfg, 'partfield_n_buckets', 12))
                cache_dir = pathlib.Path(
                    getattr(cfg, 'partfield_multiscale_cache_dir', None)
                    or out_path / "partfield_labels"
                )

                target_partfield_stage_starts = _partfield_multiscale_stage_starts(n_scales)

                for scale_idx, bucket_count in enumerate(multiscale_buckets):
                    source_labels_path = _indexed_cfg_list(
                        source_label_paths,
                        scale_idx,
                        n_scales,
                        'partfield_multiscale_source_labels',
                    )
                    target_labels_path = _indexed_cfg_list(
                        target_label_paths,
                        scale_idx,
                        n_scales,
                        'partfield_multiscale_target_labels',
                    )
                    label_dir = _indexed_cfg_list(
                        label_dirs,
                        scale_idx,
                        n_scales,
                        'partfield_multiscale_label_dirs',
                    )

                    if label_dir:
                        source_labels_path = source_labels_path or _label_path_from_dir(label_dir, default_source_labels)
                        target_labels_path = target_labels_path or _label_path_from_dir(label_dir, default_target_labels)
                    elif not multiscale_requested or int(bucket_count) == default_n_buckets:
                        source_labels_path = source_labels_path or default_source_labels
                        target_labels_path = target_labels_path or default_target_labels

                    if multiscale_requested:
                        partfield_cache = str(cache_dir / f"bucket_labels_{int(bucket_count):02d}.npz")
                        partfield_debug_path = str(cache_dir / f"bucket_labels_{int(bucket_count):02d}.npz")
                    else:
                        partfield_cache = getattr(cfg, 'partfield_cache', None)
                        if partfield_cache is None:
                            partfield_cache = str(out_path / "partfield_labels" / "bucket_labels.npz")
                        partfield_debug_path = getattr(cfg, 'partfield_debug_path', None)

                    partfield_fn = create_partfield_chamfer_loss(
                        source_vertices=load_mesh.v_pos.to(device),
                        source_faces=load_mesh.t_pos_idx.to(device),
                        target_vertices=target_mesh_vertices,
                        target_faces=target_mesh_faces,
                        source_features_path=getattr(cfg, 'partfield_source_features', None),
                        target_features_path=getattr(cfg, 'partfield_target_features', None),
                        source_labels_path=source_labels_path,
                        target_labels_path=target_labels_path,
                        feature_mode=getattr(cfg, 'partfield_feature_mode', 'auto'),
                        label_mode=getattr(cfg, 'partfield_label_mode', 'auto'),
                        labels_aligned=bool(getattr(cfg, 'partfield_labels_aligned', False)),
                        n_buckets=int(bucket_count),
                        points_per_bucket=target_partfield_points,
                        min_points_per_bucket=int(getattr(cfg, 'partfield_min_points', 12)),
                        global_weight=float(getattr(cfg, 'target_mesh_partfield_chamfer_global_weight', 0.0)),
                        position_weight=float(getattr(cfg, 'partfield_position_weight', 0.05)),
                        normal_weight=float(getattr(cfg, 'partfield_normal_weight', 0.0)),
                        cache_path=partfield_cache,
                        debug_path=partfield_debug_path,
                        random_state=int(getattr(cfg, 'seed', 42)) + scale_idx,
                        guidance_mode=getattr(cfg, 'partfield_guidance_mode', 'hard'),
                        hard_weight=float(getattr(cfg, 'partfield_hard_weight', 1.0)),
                        source_to_target_weight=float(getattr(cfg, 'partfield_source_to_target_weight', 1.0)),
                        target_to_source_weight=float(getattr(cfg, 'partfield_target_to_source_weight', 1.0)),
                        tgt_to_src_robust_scale=float(getattr(cfg, 'partfield_tgt_to_src_robust_scale', 0.0)),
                        src_to_tgt_unmatched_weight=float(getattr(cfg, 'partfield_src_to_tgt_unmatched_weight', 1.0)),
                        hard_semantic_weight=float(getattr(cfg, 'partfield_hard_semantic_weight', 0.0)),
                        hard_geometry_sigma=float(getattr(cfg, 'partfield_hard_geometry_sigma', 1.0)),
                        semantic_confidence_min_similarity=float(
                            getattr(cfg, 'partfield_semantic_confidence_min_similarity', -1.0)
                        ),
                        semantic_confidence_margin=float(getattr(cfg, 'partfield_semantic_confidence_margin', 0.0)),
                        semantic_confidence_floor=float(getattr(cfg, 'partfield_semantic_confidence_floor', 1.0)),
                        semantic_confidence_power=float(getattr(cfg, 'partfield_semantic_confidence_power', 1.0)),
                        unbalanced_transport_weight=float(getattr(cfg, 'partfield_unbalanced_transport_weight', 0.0)),
                        unbalanced_transport_rho=float(getattr(cfg, 'partfield_unbalanced_transport_rho', 0.30)),
                        soft_weight=float(getattr(cfg, 'partfield_soft_weight', 1.0)),
                        soft_points=int(getattr(cfg, 'partfield_soft_points', target_partfield_points)),
                        soft_semantic_weight=float(getattr(cfg, 'partfield_soft_semantic_weight', 0.10)),
                        soft_temperature=float(getattr(cfg, 'partfield_soft_temperature', 0.03)),
                        soft_match_space=getattr(cfg, 'partfield_soft_match_space', 'hybrid'),
                        soft_geometry_sigma=float(getattr(cfg, 'partfield_soft_geometry_sigma', 1.0)),
                        containment_weight=float(getattr(cfg, 'partfield_containment_weight', 0.0)),
                        containment_margin=float(getattr(cfg, 'partfield_containment_margin', 0.02)),
                        containment_max_weight=float(getattr(cfg, 'partfield_containment_max_weight', 1.0)),
                        moment_weight=float(getattr(cfg, 'partfield_moment_weight', 0.0)),
                        moment_extent_weight=float(getattr(cfg, 'partfield_moment_extent_weight', 0.5)),
                        profile_weight=float(getattr(cfg, 'partfield_profile_weight', 0.0)),
                        profile_bins=int(getattr(cfg, 'partfield_profile_bins', 9)),
                        profile_trim=float(getattr(cfg, 'partfield_profile_trim', 0.08)),
                        anchor_weight=float(getattr(cfg, 'partfield_anchor_weight', 0.0)),
                        anchor_geometry_sigma=float(getattr(cfg, 'partfield_anchor_geometry_sigma', 0.35)),
                        anchor_semantic_weight=float(getattr(cfg, 'partfield_anchor_semantic_weight', 0.0)),
                        balanced_sinkhorn_iters=int(getattr(cfg, 'partfield_balanced_sinkhorn_iters', 30)),
                    )
                    target_partfield_scales.append({
                        "name": f"scale_{int(bucket_count):02d}",
                        "bucket_count": int(bucket_count),
                        "fn": partfield_fn,
                        "source_labels": source_labels_path,
                        "target_labels": target_labels_path,
                    })

                target_partfield_fn = target_partfield_scales[0]["fn"] if target_partfield_scales else None

                if multiscale_requested:
                    print("Multi-scale PartField Chamfer guidance enabled:")
                    print(f"  - Buckets: {multiscale_buckets}")
                    print(f"  - Stage starts: {target_partfield_stage_starts}")
                    print(f"  - Blend epochs: {target_partfield_blend_epochs}")
                else:
                    print("PartField Chamfer guidance enabled:")
                print(f"  - Source features: {getattr(cfg, 'partfield_source_features', None)}")
                print(f"  - Target features: {getattr(cfg, 'partfield_target_features', None)}")
                print(f"  - Guidance mode: {getattr(cfg, 'partfield_guidance_mode', 'hard')}")
                print(f"  - Hard/soft weights: {getattr(cfg, 'partfield_hard_weight', 1.0)} / {getattr(cfg, 'partfield_soft_weight', 1.0)}")
                print(f"  - Hard Chamfer source->target weight: {getattr(cfg, 'partfield_source_to_target_weight', 1.0)}")
                print(f"  - Hard Chamfer target->source weight: {getattr(cfg, 'partfield_target_to_source_weight', 1.0)}")
                print(f"  - Hard Chamfer target->source robust scale: {getattr(cfg, 'partfield_tgt_to_src_robust_scale', 0.0)}")
                print(f"  - Hard Chamfer source->target unmatched weight: {getattr(cfg, 'partfield_src_to_tgt_unmatched_weight', 1.0)}")
                print(f"  - Hard Chamfer semantic match weight: {getattr(cfg, 'partfield_hard_semantic_weight', 0.0)}")
                print(f"  - Hard Chamfer geometry sigma: {getattr(cfg, 'partfield_hard_geometry_sigma', 1.0)}")
                print(f"  - Semantic confidence min similarity: {getattr(cfg, 'partfield_semantic_confidence_min_similarity', -1.0)}")
                print(f"  - Semantic confidence margin: {getattr(cfg, 'partfield_semantic_confidence_margin', 0.0)}")
                print(f"  - Semantic confidence floor: {getattr(cfg, 'partfield_semantic_confidence_floor', 1.0)}")
                print(f"  - Semantic confidence power: {getattr(cfg, 'partfield_semantic_confidence_power', 1.0)}")
                print(f"  - Unbalanced transport weight: {getattr(cfg, 'partfield_unbalanced_transport_weight', 0.0)}")
                print(f"  - Unbalanced transport rho: {getattr(cfg, 'partfield_unbalanced_transport_rho', 0.30)}")
                print(f"  - Soft points: {getattr(cfg, 'partfield_soft_points', target_partfield_points)}")
                print(f"  - Soft semantic weight: {getattr(cfg, 'partfield_soft_semantic_weight', 0.10)}")
                print(f"  - Soft temperature: {getattr(cfg, 'partfield_soft_temperature', 0.03)}")
                print(f"  - Soft match space: {getattr(cfg, 'partfield_soft_match_space', 'hybrid')}")
                print(f"  - Soft geometry sigma: {getattr(cfg, 'partfield_soft_geometry_sigma', 1.0)}")
                print(f"  - Balanced Sinkhorn iters: {getattr(cfg, 'partfield_balanced_sinkhorn_iters', 30)}")
                print(f"  - Containment weight: {getattr(cfg, 'partfield_containment_weight', 0.0)}")
                print(f"  - Containment margin: {getattr(cfg, 'partfield_containment_margin', 0.02)}")
                print(f"  - Containment max weight: {getattr(cfg, 'partfield_containment_max_weight', 1.0)}")
                print(f"  - Moment weight: {getattr(cfg, 'partfield_moment_weight', 0.0)}")
                print(f"  - Moment extent weight: {getattr(cfg, 'partfield_moment_extent_weight', 0.5)}")
                print(f"  - Profile weight: {getattr(cfg, 'partfield_profile_weight', 0.0)}")
                print(f"  - Profile bins: {getattr(cfg, 'partfield_profile_bins', 9)}")
                print(f"  - Profile trim: {getattr(cfg, 'partfield_profile_trim', 0.08)}")
                print(f"  - Anchor weight: {getattr(cfg, 'partfield_anchor_weight', 0.0)}")
                print(f"  - Anchor geometry sigma: {getattr(cfg, 'partfield_anchor_geometry_sigma', 0.35)}")
                print(f"  - Anchor semantic weight: {getattr(cfg, 'partfield_anchor_semantic_weight', 0.0)}")
                print(f"  - Weight: {target_partfield_weight}")
                print(f"  - Points per bucket: {target_partfield_points}")
                print(f"  - Warmup epochs: {target_partfield_warmup_epochs}")
                for scale_idx, scale in enumerate(target_partfield_scales):
                    fn = scale["fn"]
                    start_epoch = target_partfield_stage_starts[scale_idx]
                    print(
                        f"  - {scale['name']}: starts at epoch {start_epoch}, "
                        f"active buckets {len(fn.active_bucket_ids)} / {fn.n_buckets}"
                    )
                    print(f"    Source labels: {scale['source_labels']}")
                    print(f"    Target labels: {scale['target_labels']}")
                    for bucket_name, counts in fn.bucket_counts().items():
                        status = "active" if counts["active"] else "skip"
                        print(f"    {bucket_name}: source={counts['source']} target={counts['target']} {status}")
            except Exception as e:
                print(f"Warning: Could not initialize PartField Chamfer guidance: {e}")
                import traceback
                traceback.print_exc()
                target_partfield_weight = 0.0
                target_partfield_fn = None
                target_partfield_scales = []
    
    # =========================================================================
    # 2D PCA VISUALIZATION MODEL (for semantic tracking across views)
    # Uses dinov2_vitl14 (Large) as in purnasai/Dino_V2 paper
    # =========================================================================
    pca_dino_model = None
    save_pca_visualization = getattr(cfg, 'save_pca_visualization', False)
    
    if save_pca_visualization:
        try:
            pca_model_name = getattr(cfg, 'pca_dino_model', 'dinov2_vitl14')
            print(f"\nLoading DINO model for PCA visualization: {pca_model_name}")
            pca_dino_model = torch.hub.load('facebookresearch/dinov2', pca_model_name)
            pca_dino_model = pca_dino_model.to(device)
            pca_dino_model.eval()
            print(f"  PCA DINO model loaded successfully")
            print(f"  - Patch size: {pca_dino_model.patch_size}")
            print(f"  - Embed dim: {pca_dino_model.embed_dim}")
            print(f"  - PCA interval: every {getattr(cfg, 'pca_interval', 150)} epochs")
        except Exception as e:
            print(f"Warning: Could not load PCA DINO model: {e}")
            save_pca_visualization = False
    
    # =========================================================================
    # Continue with standard MeshUp setup
    # =========================================================================
    tex_map  = texture.create_trainable(np.random.uniform(size=[512, 512, 3], low=0.0, high=1.0), [512, 512], True)
    norm_map = texture.create_trainable(np.array([0, 0, 1]), [512, 512], True)
    spec_map = texture.create_trainable(np.array([0, 0, 0]), [512, 512], True)

    load_mesh = mesh.Mesh(
        material={"bsdf": cfg.bsdf, "kd": tex_map, "ks": spec_map, "normal": norm_map},
        base=load_mesh,
    )

    # local deformation stub
    v_indicator_local = None
    f_indicator_local = None
    if cfg.local_def:
        data = load_ply(cfg.local_sel)
        v_weight_local = data["vertex_selection"]
        f_mat = load_mesh.t_pos_idx.cpu().numpy()
        f_weight_local = (v_weight_local[f_mat].mean(axis=1) == 1).astype(np.float32)
        f_indicator_local = np.squeeze(f_weight_local > 0, 1)
        v_indicator_local = torch.as_tensor(
            np.asarray(v_weight_local).reshape(-1) > 0,
            dtype=torch.bool,
            device=device,
        )

    jac_src = SourceMesh.SourceMesh(0, str(out_path / "tmp" / "mesh.obj"), {}, 1, ttype=torch.float)
    jac_src.load(); jac_src.to(device)
    with torch.no_grad():
        gt_jac = jac_src.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0))
    if cfg.local_def:
        gt_jac[:, f_indicator_local] = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
    reference_vertices = jac_src.vertices_from_jacobians(gt_jac).squeeze().detach()
    deformation_parameterization = str(getattr(cfg, 'deformation_parameterization', 'jacobian')).lower()
    if deformation_parameterization not in {'jacobian', 'vertex'}:
        raise ValueError(
            "deformation_parameterization must be either 'jacobian' or 'vertex', "
            f"got {deformation_parameterization!r}."
        )

    vertex_offsets = None
    if deformation_parameterization == 'jacobian':
        gt_jac.requires_grad_(True)
        deformation_params = [gt_jac]
    else:
        gt_jac = gt_jac.detach()
        vertex_offsets = torch.zeros_like(reference_vertices, requires_grad=True)
        deformation_params = [vertex_offsets]

    print("Deformation parameterization:")
    print(f"  - Mode: {deformation_parameterization}")
    if deformation_parameterization == 'jacobian':
        print(f"  - Trainable tensor: per-face Jacobians {tuple(gt_jac.shape)}")
    else:
        print(f"  - Trainable tensor: per-vertex offsets {tuple(vertex_offsets.shape)}")

    def _vertices_from_deformation() -> torch.Tensor:
        if deformation_parameterization == 'jacobian':
            return jac_src.vertices_from_jacobians(gt_jac).squeeze()
        return reference_vertices + vertex_offsets

    def _jacobians_from_deformation(vertices: torch.Tensor) -> torch.Tensor:
        if deformation_parameterization == 'jacobian':
            return gt_jac
        return jac_src.jacobians_from_vertices(vertices.unsqueeze(0))

    jacobian_neighbor_smooth_weight = float(getattr(cfg, 'jacobian_neighbor_smooth_weight', 0.0))
    jacobian_outlier_weight = float(getattr(cfg, 'jacobian_outlier_weight', 0.0))
    jacobian_outlier_power = float(getattr(cfg, 'jacobian_outlier_power', 4.0))
    deformation_grad_clip_norm = float(getattr(cfg, 'deformation_grad_clip_norm', 0.0))
    edge_stretch_weight = float(getattr(cfg, 'edge_stretch_weight', 0.0))
    edge_stretch_threshold = max(float(getattr(cfg, 'edge_stretch_threshold', 1.5)), 1.0)
    edge_stretch_max_weight = max(float(getattr(cfg, 'edge_stretch_max_weight', 1.0)), 0.0)
    edge_displacement_jump_weight = float(getattr(cfg, 'edge_displacement_jump_weight', 0.0))
    edge_displacement_jump_threshold = max(float(getattr(cfg, 'edge_displacement_jump_threshold', 1.25)), 0.0)
    edge_displacement_jump_max_weight = max(float(getattr(cfg, 'edge_displacement_jump_max_weight', 1.0)), 0.0)
    jacobian_neighbor_pairs = None
    mesh_edge_pairs = None
    reference_edge_lengths = None
    if jacobian_neighbor_smooth_weight > 0.0:
        jacobian_neighbor_pairs = _face_adjacency_pairs(load_mesh.t_pos_idx, device)
        print("Neighboring-face Jacobian smoothness enabled:")
        print(f"  - Weight: {jacobian_neighbor_smooth_weight}")
        print(f"  - Face-neighbor pairs: {jacobian_neighbor_pairs.shape[0]}")
    if jacobian_outlier_weight > 0.0:
        print("Outlier-sensitive Jacobian norm regularization enabled:")
        print(f"  - Weight: {jacobian_outlier_weight}")
        print(f"  - Power: {jacobian_outlier_power}")
    if deformation_grad_clip_norm > 0.0:
        print("Deformation gradient clipping enabled:")
        print(f"  - Max norm: {deformation_grad_clip_norm}")
    if edge_stretch_weight > 0.0 or edge_displacement_jump_weight > 0.0:
        mesh_edge_pairs = _unique_edge_pairs(load_mesh.t_pos_idx.to(device), device)
        with torch.no_grad():
            reference_edge_lengths = (
                reference_vertices[mesh_edge_pairs[:, 0]]
                - reference_vertices[mesh_edge_pairs[:, 1]]
            ).norm(dim=1).clamp_min(1e-8)
        if edge_stretch_weight > 0.0:
            print("Edge-stretch spike guard enabled:")
            print(f"  - Weight: {edge_stretch_weight}")
            print(f"  - Threshold ratio: {edge_stretch_threshold}")
            print(f"  - Max violation weight: {edge_stretch_max_weight}")
            print(f"  - Mesh edges: {mesh_edge_pairs.shape[0]}")
    if edge_displacement_jump_weight > 0.0:
        print("Edge-displacement jump spike guard enabled:")
        print(f"  - Weight: {edge_displacement_jump_weight}")
        print(f"  - Threshold displacement/edge ratio: {edge_displacement_jump_threshold}")
        print(f"  - Max violation weight: {edge_displacement_jump_max_weight}")
        if mesh_edge_pairs is not None:
            print(f"  - Mesh edges: {mesh_edge_pairs.shape[0]}")

    opt = torch.optim.Adam(deformation_params, lr=cfg.lr)
    background = torch.tensor(cfg.background, device=device)

    cams = torch.utils.data.DataLoader(
        CameraBatch(
            cfg.train_res,
            [cfg.dist_min, cfg.dist_max],
            [cfg.azim_min, cfg.azim_max],
            [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
            [cfg.fov_min, cfg.fov_max],
            cfg.aug_loc,
            cfg.aug_light,
            cfg.aug_bkg,
            cfg.batch_size,
            rand_solid=True,
        ),
        cfg.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    for t in ["final"]:
        os.makedirs(out_path / f"mesh_{t}", exist_ok=True)
    for d in ["images", "jacobians", "vertex_offsets"]:
        os.makedirs(out_path / d, exist_ok=True)

    tb = SummaryWriter(str(out_path / "logs"))
    vis = TrackedVisualizer(out_path, cfg, tb, tracker, dino_pca_colorizer, color_method)

    # =========================================================================
    # Initialize DINOv2 reference features from multiple canonical viewpoints
    # =========================================================================
    if use_dino_loss and dino_loss_fn is not None:
        print("Initializing DINOv2 reference features from canonical viewpoints...")
        
        # Create a render function for the DINO initialization
        def render_for_dino(mesh_obj, cam_params, glctx_ctx, dev):
            """Render mesh for DINO feature extraction."""
            # Get initial vertices
            init_verts = _vertices_from_deformation()
            m_init = mesh.Mesh(
                init_verts, 
                load_mesh.t_pos_idx, 
                material={"bsdf": cfg.bsdf, "kd": texture.Texture2D(torch.full((1, 512, 512, 3), 0.5, device=dev)), 
                         "ks": texture.Texture2D(torch.zeros(1, 512, 512, 3, device=dev)), 
                         "normal": texture.Texture2D(torch.tensor([[[0., 0., 1.]]]).expand(1, 512, 512, 3).to(dev))}, 
                base=load_mesh
            )
            scene_init = create_scene([m_init.eval()], sz=512)
            scene_init = mesh.compute_tangents(mesh.auto_normals(scene_init))
            
            # Ensure cam_params are on the right device
            for k in cam_params:
                if isinstance(cam_params[k], torch.Tensor):
                    cam_params[k] = cam_params[k].to(dev)
            
            final_m_init = scene_init.eval(cam_params)
            rendered = render.render_mesh(
                glctx_ctx, final_m_init, 
                cam_params["mvp"], cam_params["campos"], cam_params["lightpos"], 
                cfg.light_power, 224, spp=1, num_layers=1, msaa=False,
                background=torch.ones(1, 224, 224, 3, device=dev)
            )
            return rendered  # (1, H, W, 3)
        
        try:
            dino_loss_fn.initialize_reference(
                render_function=render_for_dino,
                mesh=load_mesh,
                glctx=glctx,
                device=device,
            )
            print("DINOv2 reference features initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize DINOv2 reference: {e}")
            import traceback
            traceback.print_exc()
            use_dino_loss = False
            dino_loss_fn = None

    # =========================================================================
    # Initialize Target Mesh DINO Guidance (if enabled)
    # =========================================================================
    if use_target_mesh_guidance and target_mesh_guidance_fn is not None:
        print("Initializing Target Mesh DINO Guidance features...")
        try:
            target_mesh_guidance_fn.initialize_from_target_mesh(glctx=glctx)
            print("Target Mesh DINO Guidance initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Target Mesh guidance: {e}")
            import traceback
            traceback.print_exc()
            use_target_mesh_guidance = False
            target_mesh_guidance_fn = None

    needs_train_render = (
        use_sds
        or (use_dino_loss and dino_loss_fn is not None)
        or (use_cross_attn_loss and cross_attn_fn is not None)
        or (use_target_mesh_guidance and target_mesh_guidance_fn is not None)
    )
    if not needs_train_render:
        print("No image-space losses enabled: skipping stochastic training renders during optimization.")

    rot = 0.0
    for epoch in tqdm(range(cfg.epochs), leave=False):
        n_vert = _vertices_from_deformation()

        # blur textures
        def _blur(tex):
            return kornia.filters.gaussian_blur2d(tex.permute(0, 3, 1, 2), (7, 7), (3, 3)).permute(0, 2, 3, 1).contiguous()
        ready_kd = texture.Texture2D(_blur(load_mesh.material["kd"].data))
        ready_ks = texture.Texture2D(_blur(load_mesh.material["ks"].data))
        ready_nm = texture.Texture2D(_blur(load_mesh.material["normal"].data))
        kd_flat  = texture.Texture2D(torch.full_like(ready_kd.data, 0.5))

        m = mesh.Mesh(n_vert, load_mesh.t_pos_idx, material={"bsdf": cfg.bsdf, "kd": kd_flat, "ks": ready_ks, "normal": ready_nm}, base=load_mesh)
        scene = create_scene([m.eval()], sz=512)
        scene = mesh.compute_tangents(mesh.auto_normals(scene))

        # tb mesh
        if getattr(cfg, 'log_mesh', True) and should_log_epoch(cfg, epoch):
            cam_p = get_camera_params(cfg.log_elev, rot, cfg.log_dist, cfg.log_res, cfg.log_fov)
            rot += 1
            vis.log_mesh(epoch, scene, cam_p, video, glctx, device)

        # adapt dist
        if cfg.adapt_dist and epoch > 0:
            with torch.no_grad():
                vp = m.v_pos
                vp -= (vp.amin(0) + vp.amax(0)) / 2
                mult = torch.cat([vp.amin(0), vp.amax(0)]).abs().amax().cpu()
                cams.dataset.dist_min = cfg.dist_min * mult
                cams.dataset.dist_max = cfg.dist_max * mult

        zero_img_loss = n_vert.sum() * 0.0
        rt = {
            "loss_sds": zero_img_loss,
            "grad_norm": torch.zeros((), device=device),
            "grad": torch.zeros((), device=device),
        }
        cam_batch = None
        train_r = None
        if needs_train_render:
            cam_batch = next(iter(cams))
            for k in cam_batch:
                cam_batch[k] = cam_batch[k].to(device)
            final_m = scene.eval(cam_batch)
            train_r = render.render_mesh(
                glctx, final_m, cam_batch["mvp"], cam_batch["campos"], cam_batch["lightpos"], cfg.light_power, cfg.train_res, spp=1, num_layers=1, msaa=False, background=torch.broadcast_to(background, [1, cfg.log_res, cfg.log_res, 3])
            ).permute(0, 3, 1, 2)
            train_r = resize(train_r, out_shape=(224, 224), interp_method=resize_method)

        opt.zero_grad()
        for _ in range(cfg.accum_iter):
            # Determine if we need attention maps for cross-attention guidance
            extract_attention = use_sds and use_cross_attn_loss and cross_attn_fn is not None

            if use_sds:
                if cfg.score == "ActvnReplace":
                    rt = df.ActvnReplace(torch.cat([train_r] * (prompt_num + 1)), text_embeds, cfg.modified_cfg, prompt_num=prompt_num, controller=None, attn_ctrl_alphas=cfg.attn_ctrl_alphas)
                else:
                    # Use SDS_with_attention if we need attention maps
                    if extract_attention:
                        rt = df.SDS_with_attention(train_r, text_embeds, extract_attention=True)
                    else:
                        rt = df.SDS(train_r, text_embeds, controller=None)
                img_loss = rt["loss_sds"]
            else:
                img_loss = zero_img_loss

            current_jac = _jacobians_from_deformation(n_vert)
            identity_jac = torch.eye(3, device=device, dtype=current_jac.dtype).view(1, 1, 3, 3)
            jac_delta = current_jac - identity_jac
            jac_reg = jac_delta.pow(2).mean()
            jac_face_norm = jac_delta.squeeze(0).pow(2).sum(dim=(1, 2)).clamp_min(1e-12).sqrt()
            if jacobian_outlier_weight > 0.0:
                jac_outlier_reg = jac_face_norm.clamp_min(1e-12).pow(jacobian_outlier_power).mean()
            else:
                jac_outlier_reg = current_jac.sum() * 0.0
            if jacobian_neighbor_smooth_weight > 0.0 and jacobian_neighbor_pairs is not None and jacobian_neighbor_pairs.numel() > 0:
                jac_neighbor_delta = current_jac[:, jacobian_neighbor_pairs[:, 0]] - current_jac[:, jacobian_neighbor_pairs[:, 1]]
                jac_neighbor_reg = jac_neighbor_delta.pow(2).mean()
            else:
                jac_neighbor_reg = current_jac.sum() * 0.0
            if edge_stretch_weight > 0.0 and mesh_edge_pairs is not None and reference_edge_lengths is not None:
                current_edge_lengths = (
                    n_vert[mesh_edge_pairs[:, 0]]
                    - n_vert[mesh_edge_pairs[:, 1]]
                ).norm(dim=1)
                stretch_violation = torch.relu(current_edge_lengths / reference_edge_lengths - edge_stretch_threshold)
                stretch_sq = stretch_violation.pow(2)
                edge_stretch_reg = stretch_sq.mean() + edge_stretch_max_weight * stretch_sq.max()
            else:
                edge_stretch_reg = current_jac.sum() * 0.0
            if edge_displacement_jump_weight > 0.0 and mesh_edge_pairs is not None and reference_edge_lengths is not None:
                vertex_displacement = n_vert - reference_vertices
                edge_displacement_delta = (
                    vertex_displacement[mesh_edge_pairs[:, 0]]
                    - vertex_displacement[mesh_edge_pairs[:, 1]]
                ).norm(dim=1)
                jump_violation = torch.relu(
                    edge_displacement_delta / reference_edge_lengths - edge_displacement_jump_threshold
                )
                jump_sq = jump_violation.pow(2)
                edge_displacement_jump_reg = jump_sq.mean() + edge_displacement_jump_max_weight * jump_sq.max()
            else:
                edge_displacement_jump_reg = current_jac.sum() * 0.0
            
            image_weight_factor = _linear_ramp(
                epoch,
                int(getattr(cfg, 'image_weight_ramp_epochs', 0)),
                float(getattr(cfg, 'image_weight_start_factor', 1.0)),
            )
            sds_schedule_factor, target_schedule_factor = _loss_schedule_factors(epoch)
            current_image_weight = cfg.image_weight * image_weight_factor * sds_schedule_factor

            # Add DINOv2 loss if enabled
            total_loss = jac_reg * cfg.regularize_jacobians_weight + img_loss * current_image_weight
            total_loss = total_loss + jac_neighbor_reg * jacobian_neighbor_smooth_weight
            total_loss = total_loss + jac_outlier_reg * jacobian_outlier_weight
            total_loss = total_loss + edge_stretch_reg * edge_stretch_weight
            total_loss = total_loss + edge_displacement_jump_reg * edge_displacement_jump_weight
            
            if use_dino_loss and dino_loss_fn is not None:
                # Use the new API with epoch for warmup support
                dino_result = dino_loss_fn(train_r, epoch=epoch, return_components=True)
                dino_loss = dino_result['total']
                total_loss = total_loss + dino_loss
                
                # Log all DINO loss components
                tb.add_scalar("dino_loss/total", dino_loss.item(), global_step=epoch)
                tb.add_scalar("dino_loss/global", dino_result['global'].item(), global_step=epoch)
                tb.add_scalar("dino_loss/spatial", dino_result['spatial'].item(), global_step=epoch)
                tb.add_scalar("dino_loss/warmup_factor", dino_result['warmup'], global_step=epoch)
            
            # Add Cross-Attention guidance loss if enabled
            if use_cross_attn_loss and cross_attn_fn is not None:
                attention_maps = rt.get("attention_maps", None)
                
                # Set reference on first epoch (or first few epochs with valid attention)
                if epoch < 5 and attention_maps is not None and not cross_attn_fn.initialized:
                    cross_attn_fn.set_reference(attention_maps)
                
                cross_attn_result = cross_attn_fn(
                    attention_maps, 
                    epoch=epoch, 
                    return_components=True
                )
                cross_attn_loss = cross_attn_result['total']
                total_loss = total_loss + cross_attn_loss
                
                # Log all cross-attention loss components
                tb.add_scalar("cross_attn_loss/total", cross_attn_loss.item(), global_step=epoch)
                tb.add_scalar("cross_attn_loss/consistency", cross_attn_result['consistency'].item(), global_step=epoch)
                tb.add_scalar("cross_attn_loss/entropy", cross_attn_result['entropy'].item(), global_step=epoch)
                tb.add_scalar("cross_attn_loss/coverage", cross_attn_result['coverage'].item(), global_step=epoch)
                tb.add_scalar("cross_attn_loss/warmup_factor", cross_attn_result['warmup'], global_step=epoch)
            
            # Add Target Mesh DINO Guidance loss if enabled
            if use_target_mesh_guidance and target_mesh_guidance_fn is not None:
                # Check if using V2 (viewpoint-aligned) which needs camera angles
                if hasattr(target_mesh_guidance_fn, 'azimuth_grid'):
                    # V2: Pass camera angles (convert from radians to degrees)
                    azimuths_deg = torch.rad2deg(cam_batch['azim'])
                    elevations_deg = torch.rad2deg(cam_batch['elev'])
                    target_result = target_mesh_guidance_fn(
                        train_r, 
                        azimuths=azimuths_deg,
                        elevations=elevations_deg,
                        distances=cam_batch.get('dist', None),
                        fovs=cam_batch.get('fov', None),
                        light_positions=cam_batch.get('lightpos', None),
                        light_power=cfg.light_power,
                        epoch=epoch, 
                        return_components=True
                    )
                else:
                    # V1: Original call without camera angles
                    target_result = target_mesh_guidance_fn(train_r, epoch=epoch, return_components=True)
                
                target_loss_raw = target_result['total']
                target_loss = target_loss_raw * target_schedule_factor
                total_loss = total_loss + target_loss
                
                # Log all target mesh guidance components
                tb.add_scalar("target_mesh_loss/total", target_loss.item(), global_step=epoch)
                tb.add_scalar("target_mesh_loss/raw_total", target_loss_raw.item(), global_step=epoch)
                tb.add_scalar("target_mesh_loss/global", target_result['global'].item(), global_step=epoch)
                tb.add_scalar("target_mesh_loss/spatial", target_result['spatial'].item(), global_step=epoch)
                if 'render' in target_result:
                    tb.add_scalar("target_mesh_loss/render", target_result['render'].item(), global_step=epoch)
                if 'view_match' in target_result:
                    tb.add_scalar("target_mesh_loss/view_match", target_result['view_match'].item(), global_step=epoch)
                tb.add_scalar("target_mesh_loss/warmup_factor", target_result['warmup'], global_step=epoch)

            if target_chamfer_weight > 0 and target_mesh_vertices is not None:
                chamfer_warmup = _linear_ramp(epoch, target_chamfer_warmup_epochs, 0.0)
                target_chamfer_raw = _symmetric_chamfer(n_vert, target_mesh_vertices, target_chamfer_points)
                target_chamfer_loss = target_chamfer_raw * target_chamfer_weight * chamfer_warmup * target_schedule_factor
                total_loss = total_loss + target_chamfer_loss

                tb.add_scalar("target_mesh_chamfer/raw", target_chamfer_raw.item(), global_step=epoch)
                tb.add_scalar("target_mesh_chamfer/total", target_chamfer_loss.item(), global_step=epoch)
                tb.add_scalar("target_mesh_chamfer/warmup_factor", chamfer_warmup, global_step=epoch)

            if target_vertex_corr_weight > 0 and target_vertex_corr_available and target_mesh_vertices is not None:
                vertex_corr_warmup = _linear_ramp(epoch, target_vertex_corr_warmup_epochs, 0.0)
                target_vertex_corr_raw = (n_vert - target_mesh_vertices).pow(2).sum(dim=1).mean()
                target_vertex_corr_loss = (
                    target_vertex_corr_raw
                    * target_vertex_corr_weight
                    * vertex_corr_warmup
                    * target_schedule_factor
                )
                total_loss = total_loss + target_vertex_corr_loss

                tb.add_scalar("target_mesh_vertex_correspondence/raw", target_vertex_corr_raw.item(), global_step=epoch)
                tb.add_scalar("target_mesh_vertex_correspondence/total", target_vertex_corr_loss.item(), global_step=epoch)
                tb.add_scalar("target_mesh_vertex_correspondence/warmup_factor", vertex_corr_warmup, global_step=epoch)

            if target_semantic_vertex_corr_weight > 0 and target_semantic_vertex_corr_fn is not None:
                semantic_vertex_corr_warmup = _linear_ramp(
                    epoch,
                    target_semantic_vertex_corr_warmup_epochs,
                    0.0,
                )
                semantic_vertex_corr_result = target_semantic_vertex_corr_fn(n_vert, return_components=True)
                target_semantic_vertex_corr_raw = semantic_vertex_corr_result["raw"]
                target_semantic_vertex_corr_loss = (
                    target_semantic_vertex_corr_raw
                    * target_semantic_vertex_corr_weight
                    * semantic_vertex_corr_warmup
                    * target_schedule_factor
                )
                total_loss = total_loss + target_semantic_vertex_corr_loss

                tb.add_scalar(
                    "target_mesh_semantic_vertex_correspondence/raw",
                    target_semantic_vertex_corr_raw.item(),
                    global_step=epoch,
                )
                tb.add_scalar(
                    "target_mesh_semantic_vertex_correspondence/total",
                    target_semantic_vertex_corr_loss.item(),
                    global_step=epoch,
                )
                tb.add_scalar(
                    "target_mesh_semantic_vertex_correspondence/warmup_factor",
                    semantic_vertex_corr_warmup,
                    global_step=epoch,
                )
                tb.add_scalar(
                    "target_mesh_semantic_vertex_correspondence/mean_weight",
                    semantic_vertex_corr_result["mean_weight"].item(),
                    global_step=epoch,
                )
                tb.add_scalar(
                    "target_mesh_semantic_vertex_correspondence/min_weight",
                    semantic_vertex_corr_result["min_weight"].item(),
                    global_step=epoch,
                )
                tb.add_scalar(
                    "target_mesh_semantic_vertex_correspondence/max_weight",
                    semantic_vertex_corr_result["max_weight"].item(),
                    global_step=epoch,
                )
                tb.add_scalar(
                    "target_mesh_semantic_vertex_correspondence/unique_targets",
                    semantic_vertex_corr_result["unique_targets"],
                    global_step=epoch,
                )

            if target_part_chamfer_weight > 0 and target_part_chamfer_fn is not None:
                part_chamfer_warmup = _linear_ramp(epoch, target_part_chamfer_warmup_epochs, 0.0)
                part_chamfer_result = target_part_chamfer_fn(n_vert, return_components=True)
                target_part_chamfer_raw = part_chamfer_result["raw"]
                target_part_chamfer_loss = (
                    target_part_chamfer_raw
                    * target_part_chamfer_weight
                    * part_chamfer_warmup
                    * target_schedule_factor
                )
                total_loss = total_loss + target_part_chamfer_loss

                tb.add_scalar("target_mesh_part_chamfer/raw", target_part_chamfer_raw.item(), global_step=epoch)
                tb.add_scalar("target_mesh_part_chamfer/part_raw", part_chamfer_result["part_raw"].item(), global_step=epoch)
                tb.add_scalar("target_mesh_part_chamfer/global_raw", part_chamfer_result["global_raw"].item(), global_step=epoch)
                tb.add_scalar("target_mesh_part_chamfer/total", target_part_chamfer_loss.item(), global_step=epoch)
                tb.add_scalar("target_mesh_part_chamfer/warmup_factor", part_chamfer_warmup, global_step=epoch)
                tb.add_scalar("target_mesh_part_chamfer/active_parts", part_chamfer_result["active_parts"], global_step=epoch)
                for part_name, part_loss in part_chamfer_result["parts"].items():
                    tb.add_scalar(f"target_mesh_part_chamfer/parts/{part_name}", part_loss.item(), global_step=epoch)

            if target_semantic_bucket_weight > 0 and target_semantic_bucket_fn is not None:
                semantic_bucket_warmup = _linear_ramp(epoch, target_semantic_bucket_warmup_epochs, 0.0)
                semantic_bucket_result = target_semantic_bucket_fn(n_vert, return_components=True)
                target_semantic_bucket_raw = semantic_bucket_result["raw"]
                target_semantic_bucket_loss = (
                    target_semantic_bucket_raw
                    * target_semantic_bucket_weight
                    * semantic_bucket_warmup
                    * target_schedule_factor
                )
                total_loss = total_loss + target_semantic_bucket_loss

                tb.add_scalar("target_mesh_semantic_bucket_chamfer/raw", target_semantic_bucket_raw.item(), global_step=epoch)
                tb.add_scalar("target_mesh_semantic_bucket_chamfer/bucket_raw", semantic_bucket_result["bucket_raw"].item(), global_step=epoch)
                tb.add_scalar("target_mesh_semantic_bucket_chamfer/global_raw", semantic_bucket_result["global_raw"].item(), global_step=epoch)
                tb.add_scalar("target_mesh_semantic_bucket_chamfer/total", target_semantic_bucket_loss.item(), global_step=epoch)
                tb.add_scalar("target_mesh_semantic_bucket_chamfer/warmup_factor", semantic_bucket_warmup, global_step=epoch)
                tb.add_scalar("target_mesh_semantic_bucket_chamfer/active_buckets", semantic_bucket_result["active_buckets"], global_step=epoch)
                for bucket_name, bucket_loss in semantic_bucket_result["buckets"].items():
                    tb.add_scalar(f"target_mesh_semantic_bucket_chamfer/buckets/{bucket_name}", bucket_loss.item(), global_step=epoch)

            if target_sampart3d_weight > 0 and target_sampart3d_fn is not None:
                sampart3d_warmup = _linear_ramp(epoch, target_sampart3d_warmup_epochs, 0.0)
                sampart3d_result = target_sampart3d_fn(n_vert, return_components=True)
                target_sampart3d_raw = sampart3d_result["raw"]
                target_sampart3d_loss = (
                    target_sampart3d_raw
                    * target_sampart3d_weight
                    * sampart3d_warmup
                    * target_schedule_factor
                )
                total_loss = total_loss + target_sampart3d_loss

                tb.add_scalar("target_mesh_sampart3d_chamfer/raw", target_sampart3d_raw.item(), global_step=epoch)
                tb.add_scalar("target_mesh_sampart3d_chamfer/bucket_raw", sampart3d_result["bucket_raw"].item(), global_step=epoch)
                tb.add_scalar("target_mesh_sampart3d_chamfer/global_raw", sampart3d_result["global_raw"].item(), global_step=epoch)
                tb.add_scalar("target_mesh_sampart3d_chamfer/total", target_sampart3d_loss.item(), global_step=epoch)
                tb.add_scalar("target_mesh_sampart3d_chamfer/warmup_factor", sampart3d_warmup, global_step=epoch)
                tb.add_scalar("target_mesh_sampart3d_chamfer/active_buckets", sampart3d_result["active_buckets"], global_step=epoch)
                for bucket_name, bucket_loss in sampart3d_result["buckets"].items():
                    tb.add_scalar(f"target_mesh_sampart3d_chamfer/buckets/{bucket_name}", bucket_loss.item(), global_step=epoch)

            if target_partfield_weight > 0 and target_partfield_fn is not None:
                partfield_warmup = _linear_ramp(epoch, target_partfield_warmup_epochs, 0.0)
                if len(target_partfield_scales) > 1:
                    scale_weights = _partfield_multiscale_weights(
                        epoch,
                        target_partfield_stage_starts,
                        target_partfield_blend_epochs,
                    )
                    target_partfield_raw = n_vert.sum() * 0.0
                    active_buckets_weighted = 0.0
                    partfield_result = {
                        "bucket_raw": n_vert.sum() * 0.0,
                        "soft_raw": n_vert.sum() * 0.0,
                        "global_raw": n_vert.sum() * 0.0,
                        "containment_raw": n_vert.sum() * 0.0,
                        "moment_raw": n_vert.sum() * 0.0,
                        "profile_raw": n_vert.sum() * 0.0,
                        "anchor_raw": n_vert.sum() * 0.0,
                        "active_buckets": 0.0,
                        "buckets": {},
                    }
                    for scale, scale_weight in zip(target_partfield_scales, scale_weights):
                        tb.add_scalar(
                            f"target_mesh_partfield_chamfer/scales/{scale['name']}/weight",
                            scale_weight,
                            global_step=epoch,
                        )
                        if scale_weight <= 0:
                            continue

                        scale_result = scale["fn"](n_vert, return_components=True)
                        target_partfield_raw = target_partfield_raw + scale_result["raw"] * scale_weight
                        partfield_result["bucket_raw"] = (
                            partfield_result["bucket_raw"] + scale_result["bucket_raw"] * scale_weight
                        )
                        partfield_result["soft_raw"] = (
                            partfield_result["soft_raw"]
                            + scale_result.get("soft_raw", n_vert.sum() * 0.0) * scale_weight
                        )
                        partfield_result["global_raw"] = (
                            partfield_result["global_raw"] + scale_result["global_raw"] * scale_weight
                        )
                        partfield_result["containment_raw"] = (
                            partfield_result.get("containment_raw", n_vert.sum() * 0.0)
                            + scale_result.get("containment_raw", n_vert.sum() * 0.0) * scale_weight
                        )
                        partfield_result["moment_raw"] = (
                            partfield_result.get("moment_raw", n_vert.sum() * 0.0)
                            + scale_result.get("moment_raw", n_vert.sum() * 0.0) * scale_weight
                        )
                        partfield_result["profile_raw"] = (
                            partfield_result.get("profile_raw", n_vert.sum() * 0.0)
                            + scale_result.get("profile_raw", n_vert.sum() * 0.0) * scale_weight
                        )
                        partfield_result["anchor_raw"] = (
                            partfield_result.get("anchor_raw", n_vert.sum() * 0.0)
                            + scale_result.get("anchor_raw", n_vert.sum() * 0.0) * scale_weight
                        )
                        active_buckets_weighted += float(scale_result["active_buckets"]) * scale_weight

                        tb.add_scalar(
                            f"target_mesh_partfield_chamfer/scales/{scale['name']}/raw",
                            scale_result["raw"].item(),
                            global_step=epoch,
                        )
                        tb.add_scalar(
                            f"target_mesh_partfield_chamfer/scales/{scale['name']}/bucket_raw",
                            scale_result["bucket_raw"].item(),
                            global_step=epoch,
                        )
                        tb.add_scalar(
                            f"target_mesh_partfield_chamfer/scales/{scale['name']}/soft_raw",
                            scale_result.get("soft_raw", n_vert.sum() * 0.0).item(),
                            global_step=epoch,
                        )
                        tb.add_scalar(
                            f"target_mesh_partfield_chamfer/scales/{scale['name']}/containment_raw",
                            scale_result.get("containment_raw", n_vert.sum() * 0.0).item(),
                            global_step=epoch,
                        )
                        tb.add_scalar(
                            f"target_mesh_partfield_chamfer/scales/{scale['name']}/moment_raw",
                            scale_result.get("moment_raw", n_vert.sum() * 0.0).item(),
                            global_step=epoch,
                        )
                        tb.add_scalar(
                            f"target_mesh_partfield_chamfer/scales/{scale['name']}/profile_raw",
                            scale_result.get("profile_raw", n_vert.sum() * 0.0).item(),
                            global_step=epoch,
                        )
                        tb.add_scalar(
                            f"target_mesh_partfield_chamfer/scales/{scale['name']}/anchor_raw",
                            scale_result.get("anchor_raw", n_vert.sum() * 0.0).item(),
                            global_step=epoch,
                        )
                        tb.add_scalar(
                            f"target_mesh_partfield_chamfer/scales/{scale['name']}/active_buckets",
                            scale_result["active_buckets"],
                            global_step=epoch,
                        )
                        for bucket_name, bucket_loss in scale_result["buckets"].items():
                            tb.add_scalar(
                                f"target_mesh_partfield_chamfer/scales/{scale['name']}/buckets/{bucket_name}",
                                bucket_loss.item(),
                                global_step=epoch,
                            )
                    partfield_result["raw"] = target_partfield_raw
                    partfield_result["active_buckets"] = active_buckets_weighted
                else:
                    partfield_result = target_partfield_fn(n_vert, return_components=True)
                    target_partfield_raw = partfield_result["raw"]
                target_partfield_loss = (
                    target_partfield_raw
                    * target_partfield_weight
                    * partfield_warmup
                    * target_schedule_factor
                )
                total_loss = total_loss + target_partfield_loss

                tb.add_scalar("target_mesh_partfield_chamfer/raw", target_partfield_raw.item(), global_step=epoch)
                tb.add_scalar("target_mesh_partfield_chamfer/bucket_raw", partfield_result["bucket_raw"].item(), global_step=epoch)
                tb.add_scalar("target_mesh_partfield_chamfer/soft_raw", partfield_result.get("soft_raw", n_vert.sum() * 0.0).item(), global_step=epoch)
                tb.add_scalar("target_mesh_partfield_chamfer/global_raw", partfield_result["global_raw"].item(), global_step=epoch)
                tb.add_scalar("target_mesh_partfield_chamfer/containment_raw", partfield_result.get("containment_raw", n_vert.sum() * 0.0).item(), global_step=epoch)
                tb.add_scalar("target_mesh_partfield_chamfer/moment_raw", partfield_result.get("moment_raw", n_vert.sum() * 0.0).item(), global_step=epoch)
                tb.add_scalar("target_mesh_partfield_chamfer/profile_raw", partfield_result.get("profile_raw", n_vert.sum() * 0.0).item(), global_step=epoch)
                tb.add_scalar("target_mesh_partfield_chamfer/anchor_raw", partfield_result.get("anchor_raw", n_vert.sum() * 0.0).item(), global_step=epoch)
                tb.add_scalar("target_mesh_partfield_chamfer/total", target_partfield_loss.item(), global_step=epoch)
                tb.add_scalar("target_mesh_partfield_chamfer/warmup_factor", partfield_warmup, global_step=epoch)
                tb.add_scalar("target_mesh_partfield_chamfer/active_buckets", partfield_result["active_buckets"], global_step=epoch)
                for bucket_name, bucket_loss in partfield_result["buckets"].items():
                    tb.add_scalar(f"target_mesh_partfield_chamfer/buckets/{bucket_name}", bucket_loss.item(), global_step=epoch)
            
            tb.add_scalar("jacobian_regularization", jac_reg, global_step=epoch)
            tb.add_scalar("jacobian_regularization/weight", cfg.regularize_jacobians_weight, global_step=epoch)
            tb.add_scalar("jacobian_neighbor_smoothness", jac_neighbor_reg, global_step=epoch)
            tb.add_scalar("jacobian_neighbor_smoothness/weight", jacobian_neighbor_smooth_weight, global_step=epoch)
            tb.add_scalar("jacobian_outlier_regularization", jac_outlier_reg, global_step=epoch)
            tb.add_scalar("jacobian_outlier_regularization/weight", jacobian_outlier_weight, global_step=epoch)
            tb.add_scalar("edge_stretch_regularization", edge_stretch_reg, global_step=epoch)
            tb.add_scalar("edge_stretch_regularization/weight", edge_stretch_weight, global_step=epoch)
            tb.add_scalar("edge_displacement_jump_regularization", edge_displacement_jump_reg, global_step=epoch)
            tb.add_scalar("edge_displacement_jump_regularization/weight", edge_displacement_jump_weight, global_step=epoch)
            with torch.no_grad():
                tb.add_scalar("jacobian_norm/mean", jac_face_norm.mean().item(), global_step=epoch)
                tb.add_scalar("jacobian_norm/p95", torch.quantile(jac_face_norm, 0.95).item(), global_step=epoch)
                tb.add_scalar("jacobian_norm/max", jac_face_norm.max().item(), global_step=epoch)
            tb.add_scalar("image_loss", img_loss, global_step=epoch)
            tb.add_scalar("image_loss/current_weight", current_image_weight, global_step=epoch)
            tb.add_scalar("loss_schedule/sds_factor", sds_schedule_factor, global_step=epoch)
            tb.add_scalar("loss_schedule/target_factor", target_schedule_factor, global_step=epoch)
            
            total_loss.backward(retain_graph=True)
            
            if cfg.local_def and deformation_parameterization == 'jacobian':
                gt_jac.grad[:, f_indicator_local] = 0
            elif cfg.local_def and vertex_offsets is not None and v_indicator_local is not None:
                vertex_offsets.grad[v_indicator_local] = 0
            if deformation_grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(deformation_params, deformation_grad_clip_norm)
            opt.step()

        if train_r is not None:
            vis.save_epoch(epoch, rt, train_r)
        
        # =====================================================================
        # SAVE RENDERED IMAGES AT THIS EPOCH (for visual tracking)
        # =====================================================================
        if getattr(cfg, 'save_epoch_renders', True):
            vis.save_epoch_render(epoch, scene, glctx, device, n_views=4)
        
        # =====================================================================
        # SAVE COMBINED PCA VISUALIZATION (semantic feature tracking)
        # Uses 2-step PCA across multiple viewpoints as in purnasai/Dino_V2
        # =====================================================================
        if save_pca_visualization and pca_dino_model is not None:
            vis.save_combined_pca_visualization(
                epoch, scene, glctx, device, 
                dino_model=pca_dino_model, 
                n_views=getattr(cfg, 'pca_n_views', 8)
            )
        
        # =====================================================================
        # SAVE COLORED MESH WITH TRACKING
        # =====================================================================
        if track_correspondence and vertex_colors is not None:
            vis.save_colored_mesh(
                epoch,
                n_vert.detach().cpu(),
                load_mesh.t_pos_idx.detach().cpu(),
                vertex_colors,
                original_vertices=original_vertices,
                current_mesh=m,
                glctx=glctx,
                device=device
            )

        if should_log_epoch(cfg, epoch):
            obj.write_obj(str(out_path / "mesh_final"), m.eval())
            save_jac = _jacobians_from_deformation(n_vert).detach().cpu().numpy()
            np.save(out_path / "jacobians" / f"jacobians_epoch_{epoch + 1}.npy", save_jac)
            if vertex_offsets is not None:
                np.save(
                    out_path / "vertex_offsets" / f"vertex_offsets_epoch_{epoch + 1}.npy",
                    vertex_offsets.detach().cpu().numpy(),
                )

        if "grad" in rt:
            del rt["grad"]

    # =========================================================================
    # FINAL EXPORTS
    # =========================================================================
    video.close()
    obj.write_obj(str(out_path / "mesh_final"), m.eval())
    
    # Export final colored mesh and displacement visualization
    if track_correspondence:
        final_vertices = n_vert.detach().cpu().numpy()
        final_faces = load_mesh.t_pos_idx.cpu().numpy()
        
        # Print final mesh statistics
        print(f"\n{'='*60}")
        print(f"FINAL MESH STATISTICS:")
        print(f"  - Number of vertices: {final_vertices.shape[0]}")
        print(f"  - Number of faces: {final_faces.shape[0]}")
        print(f"  - Vertex position range:")
        print(f"      X: [{final_vertices[:, 0].min():.4f}, {final_vertices[:, 0].max():.4f}]")
        print(f"      Y: [{final_vertices[:, 1].min():.4f}, {final_vertices[:, 1].max():.4f}]")
        print(f"      Z: [{final_vertices[:, 2].min():.4f}, {final_vertices[:, 2].max():.4f}]")
        print(f"{'='*60}\n")
        
        # =====================================================================
        # ALWAYS SAVE POSITION-COLORED MESH (semantic tracking via position)
        # =====================================================================
        position_ply_path = out_path / "colored_meshes" / "mesh_final_position.ply"
        export_mesh_with_colors(
            str(position_ply_path),
            final_vertices,
            final_faces,
            vertex_colors,
            format='ply'
        )
        print(f"Saved position-colored mesh to {position_ply_path}")
        
        # Keep final artifacts minimal: skip final DINO-PCA/correspondence/
        # displacement PLY exports for cleaner outputs.
        final_colors = vertex_colors
        
        # Final correspondence map
        corr_path = out_path / "correspondence" / "final_correspondence.json"
        os.makedirs(out_path / "correspondence", exist_ok=True)
        export_correspondence_map(
            str(corr_path),
            original_vertices,
            final_vertices,
            final_faces,
            colors=final_colors,
            metadata={
                'text_prompt': cfg.text_prompt,
                'epochs': cfg.epochs,
                'color_method': color_method,
            }
        )
        print(f"Saved correspondence map to {corr_path}")


# For compatibility with main.py
loop = loop_with_tracking
