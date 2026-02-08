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

################################################################################

class TrackedVisualizer:
    """Extended visualizer with semantic tracking support."""
    
    def __init__(
        self,
        out_path: pathlib.Path,
        cfg: EasyDict,
        tb: SummaryWriter,
        tracker: VertexColorTracker = None
    ):
        self.out_path = out_path
        self.cfg = cfg
        self.tb = tb
        self.tracker = tracker
        
        for d in ["figure", "images", "colored_meshes", "correspondence"]:
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

    def save_epoch(self, epoch: int, rt: dict, train_render: torch.Tensor):
        if (epoch + 1) % self.cfg.log_interval_im != 0 and epoch != 0:
            return

        # Get actual batch size from the tensor to avoid index out of bounds
        actual_batch_size = train_render.shape[0]
        
        if self.cfg.log:
            # Clamp to actual batch size to prevent index out of bounds
            max_images = min(15, actual_batch_size)
            idx_list = torch.arange(max_images)
            fig_dir = self.out_path / "figure" / f"epoch_{epoch + 1}"
            os.makedirs(fig_dir, exist_ok=True)
        else:
            max_images = min(5, actual_batch_size)
            idx_list = torch.randperm(actual_batch_size)[:max_images]
            fig_dir = None

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
        original_vertices: np.ndarray = None
    ):
        """Save mesh with vertex colors and optional correspondence info."""
        if (epoch + 1) % self.cfg.log_interval_im != 0 and epoch != 0:
            return
        
        # Export PLY with colors
        ply_path = self.out_path / "colored_meshes" / f"mesh_epoch_{epoch + 1}.ply"
        export_mesh_with_colors(
            str(ply_path),
            vertices,
            faces,
            colors,
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
                colors=colors,
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

    # misc dirs
    os.makedirs(out_path / "tmp", exist_ok=True)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(cfg.mesh)
    df = DeepFloydGuidance(cfg, device)

    # prompts
    txt = cfg.text_prompt
    if isinstance(txt, list):
        prompts = [p + ", a 3d rendering" if i != len(txt) - 1 else p for i, p in enumerate(txt)]
    else:
        prompts = [txt + ", a 3d rendering"]
    print("Target text prompt:", txt)
    text_embeds = df.encode_text_2(prompts, batch_size=cfg.batch_size).to(device)
    prompt_num = len(prompts)

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
    # SEMANTIC TRACKING INITIALIZATION
    # =========================================================================
    track_correspondence = getattr(cfg, 'track_correspondence', True)
    color_method = getattr(cfg, 'color_method', 'position')
    
    if track_correspondence:
        print(f"Initializing semantic tracking with method: {color_method}")
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
                model_name=getattr(cfg, 'dino_model', 'dinov2_vits14'),
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
    if cfg.local_def:
        data = load_ply(cfg.local_sel)
        v_weight_local = data["vertex_selection"]
        f_mat = load_mesh.t_pos_idx.cpu().numpy()
        f_weight_local = (v_weight_local[f_mat].mean(axis=1) == 1).astype(np.float32)
        f_indicator_local = np.squeeze(f_weight_local > 0, 1)

    jac_src = SourceMesh.SourceMesh(0, str(out_path / "tmp" / "mesh.obj"), {}, 1, ttype=torch.float)
    jac_src.load(); jac_src.to(device)
    with torch.no_grad():
        gt_jac = jac_src.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0))
    if cfg.local_def:
        gt_jac[:, f_indicator_local] = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
    gt_jac.requires_grad_(True)

    opt = torch.optim.Adam([gt_jac], lr=cfg.lr)
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

    for t in ["final", "final_texture", "mesh_log", "best_clip", "best_total"]:
        os.makedirs(out_path / f"mesh_{t}", exist_ok=True)
    for d in ["n_vert", "images", "grads", "jacobians"]:
        os.makedirs(out_path / d, exist_ok=True)

    tb = SummaryWriter(str(out_path / "logs"))
    vis = TrackedVisualizer(out_path, cfg, tb, tracker)

    # =========================================================================
    # Initialize DINOv2 reference features from multiple canonical viewpoints
    # =========================================================================
    if use_dino_loss and dino_loss_fn is not None:
        print("Initializing DINOv2 reference features from canonical viewpoints...")
        
        # Create a render function for the DINO initialization
        def render_for_dino(mesh_obj, cam_params, glctx_ctx, dev):
            """Render mesh for DINO feature extraction."""
            # Get initial vertices
            init_verts = jac_src.vertices_from_jacobians(gt_jac).squeeze()
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

    rot = 0.0
    for epoch in tqdm(range(cfg.epochs), leave=False):
        # verts from jacobians
        n_vert = jac_src.vertices_from_jacobians(gt_jac).squeeze()

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
        if (epoch + 1) % cfg.log_interval_im == 0 or epoch == 0:
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
            extract_attention = use_cross_attn_loss and cross_attn_fn is not None
            
            if cfg.score == "ActvnReplace":
                rt = df.ActvnReplace(torch.cat([train_r] * (prompt_num + 1)), text_embeds, cfg.modified_cfg, prompt_num=prompt_num, controller=None, attn_ctrl_alphas=cfg.attn_ctrl_alphas)
            else:
                # Use SDS_with_attention if we need attention maps
                if extract_attention:
                    rt = df.SDS_with_attention(train_r, text_embeds, extract_attention=True)
                else:
                    rt = df.SDS(train_r, text_embeds, controller=None)

            img_loss = rt["loss_sds"]
            jac_reg = ((gt_jac - torch.eye(3, device=device)) ** 2).mean()
            
            # Add DINOv2 loss if enabled
            total_loss = jac_reg * cfg.regularize_jacobians_weight + img_loss * cfg.image_weight
            
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
            
            tb.add_scalar("jacobian_regularization", jac_reg, global_step=epoch)
            tb.add_scalar("image_loss", img_loss, global_step=epoch)
            
            total_loss.backward(retain_graph=True)
            
            if cfg.local_def:
                gt_jac.grad[:, f_indicator_local] = 0
            opt.step()

        vis.save_epoch(epoch, rt, train_r)
        
        # =====================================================================
        # SAVE COLORED MESH WITH TRACKING
        # =====================================================================
        if track_correspondence and vertex_colors is not None:
            vis.save_colored_mesh(
                epoch,
                n_vert.detach().cpu(),
                load_mesh.t_pos_idx.detach().cpu(),
                vertex_colors,
                original_vertices=original_vertices
            )

        if (epoch + 1) % cfg.log_interval_im == 0 or epoch == 0:
            obj.write_obj(str(out_path / "mesh_final"), m.eval())
            np.save(out_path / "jacobians" / f"jacobians_epoch_{epoch + 1}.npy", gt_jac.detach().cpu().numpy())

        del rt["grad"]

    # =========================================================================
    # FINAL EXPORTS
    # =========================================================================
    video.close()
    obj.write_obj(str(out_path / "mesh_final"), m.eval())
    
    # Export final colored mesh and displacement visualization
    if track_correspondence:
        final_vertices = n_vert.detach().cpu().numpy()
        
        # Final mesh with original colors (showing correspondence)
        final_ply_path = out_path / "colored_meshes" / "mesh_final_correspondence.ply"
        export_mesh_with_colors(
            str(final_ply_path),
            final_vertices,
            load_mesh.t_pos_idx.cpu().numpy(),
            vertex_colors,
            format='ply'
        )
        print(f"Saved final colored mesh to {final_ply_path}")
        
        # Displacement visualization
        disp_ply_path = out_path / "colored_meshes" / "mesh_final_displacement.ply"
        visualize_correspondence_displacement(
            original_vertices,
            final_vertices,
            load_mesh.t_pos_idx.cpu().numpy(),
            str(disp_ply_path)
        )
        
        # Final correspondence map
        corr_path = out_path / "correspondence" / "final_correspondence.json"
        os.makedirs(out_path / "correspondence", exist_ok=True)
        export_correspondence_map(
            str(corr_path),
            original_vertices,
            final_vertices,
            load_mesh.t_pos_idx.cpu().numpy(),
            colors=vertex_colors,
            metadata={
                'text_prompt': cfg.text_prompt,
                'epochs': cfg.epochs,
                'color_method': color_method,
            }
        )
        print(f"Saved correspondence map to {corr_path}")


# For compatibility with main.py
loop = loop_with_tracking
