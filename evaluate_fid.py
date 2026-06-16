#!/usr/bin/env python3
"""
Post-training evaluation script that:
1. Renders the final mesh from 8 fixed angles
2. Generates reference images using the diffusion model
3. Computes FID between mesh renders and reference images
4. Saves all results

Usage:
    python evaluate_fid.py --mesh_path outputs/hound_to_hippo/mesh_final/mesh.obj \
                          --text_prompt "a hippo" \
                          --output_dir outputs/hound_to_hippo/evaluation
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
import importlib.util

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

def _load_fid_evaluator_class():
    """Load FIDEvaluator directly from file to avoid package side-effects."""
    module_path = Path(__file__).parent / "semantic_tracking" / "fid_evaluation.py"
    spec = importlib.util.spec_from_file_location("fid_evaluation_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FIDEvaluator


FIDEvaluator = _load_fid_evaluator_class()


def generate_reference_images(
    text_prompt: str,
    n_images: int = 16,
    device: str = 'cuda',
    if_model_size: str = 'XL',
    if_num_inference_steps: int = 50,
    if_guidance_scale: float = 7.5,
    if_height: int = 512,
    if_width: int = 512,
    if_cpu_offload: bool = False,
) -> torch.Tensor:
    """Generate reference images using DeepFloyd IF."""
    from diffusers import IFPipeline
    
    print(f"Generating {n_images} reference images for: '{text_prompt}'")
    
    model_map = {
        'M': 'DeepFloyd/IF-I-M-v1.0',
        'L': 'DeepFloyd/IF-I-L-v1.0',
        'XL': 'DeepFloyd/IF-I-XL-v1.0',
    }
    if_model_size = if_model_size.upper()
    if if_model_size not in model_map:
        raise ValueError(f"Invalid --if_model_size '{if_model_size}'. Use one of: M, L, XL")

    # Initialize DeepFloyd pipeline directly
    model_name = model_map[if_model_size]
    print(f"Loading DeepFloyd {if_model_size} model: {model_name}")
    pipeline = IFPipeline.from_pretrained(
        model_name,
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        watermarker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    if if_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    pipeline.enable_attention_slicing()
    
    images = []
    for i in range(n_images):
        print(f"  Generating image {i+1}/{n_images}...")
        with torch.no_grad():
            result = pipeline(
                prompt=text_prompt,
                height=if_height,
                width=if_width,
                num_inference_steps=if_num_inference_steps,
                guidance_scale=if_guidance_scale,
                generator=torch.Generator(device=device).manual_seed(42 + i)
            )
            img = result.images[0]
            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
            images.append(img_tensor)
    
    return torch.stack(images)


def render_mesh_multiview(
    mesh_path: str,
    vertex_colors_path: str = None,
    n_views: int = 8,
    resolution: int = 512,
    device: str = 'cuda'
) -> torch.Tensor:
    """Render mesh from multiple fixed viewpoints."""
    import nvdiffrast.torch as dr
    from nvdiffmodeling.src import obj, mesh
    from utilities.camera import get_camera_params
    from utilities.helpers import create_scene
    from nvdiffmodeling.src import render, texture
    
    print(f"Rendering mesh from {n_views} viewpoints...")
    
    # Load mesh
    m = obj.load_obj(mesh_path)
    m = mesh.unit_size(m)
    
    # Create simple material
    kd = texture.Texture2D(torch.full((1, 512, 512, 3), 0.5, device=device))
    ks = texture.Texture2D(torch.zeros(1, 512, 512, 3, device=device))
    nm = texture.Texture2D(torch.tensor([[[0., 0., 1.]]]).expand(1, 512, 512, 3).to(device))
    
    m = mesh.Mesh(
        v_pos=m.v_pos.to(device),
        t_pos_idx=m.t_pos_idx.to(device),
        v_tex=m.v_tex.to(device) if m.v_tex is not None else None,
        t_tex_idx=m.t_tex_idx.to(device) if m.t_tex_idx is not None else None,
        material={"bsdf": "diffuse", "kd": kd, "ks": ks, "normal": nm},
    )
    
    scene = create_scene([m.eval()], sz=512)
    scene = mesh.compute_tangents(mesh.auto_normals(scene))
    
    glctx = dr.RasterizeGLContext()
    
    images = []
    azimuths = [i * (360.0 / n_views) for i in range(n_views)]
    
    for azim in azimuths:
        cam_params = get_camera_params(30.0, azim, 3.0, resolution, 60.0)
        for k, v in cam_params.items():
            if isinstance(v, torch.Tensor):
                cam_params[k] = v.to(device)
        
        final_m = scene.eval(cam_params)
        rendered = render.render_mesh(
            glctx, final_m,
            cam_params["mvp"], cam_params["campos"], cam_params["lightpos"],
            5.0, resolution, spp=1, num_layers=1, msaa=False,
            background=torch.ones(1, resolution, resolution, 3, device=device)
        )  # (1, H, W, 3)
        
        rendered = rendered.permute(0, 3, 1, 2)  # (1, 3, H, W)
        images.append(rendered[0])
    
    return torch.stack(images)


def main():
    parser = argparse.ArgumentParser(description='Evaluate mesh with FID')
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to mesh .obj file')
    parser.add_argument('--text_prompt', type=str, required=True, help='Text prompt for reference generation')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--n_views', type=int, default=8, help='Number of viewpoints')
    parser.add_argument('--n_references', type=int, default=16, help='Number of reference images')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--if_model_size', type=str, default='XL', choices=['M', 'L', 'XL'],
                        help='DeepFloyd IF model size for reference generation')
    parser.add_argument('--if_num_inference_steps', type=int, default=50,
                        help='Diffusion inference steps for reference generation')
    parser.add_argument('--if_guidance_scale', type=float, default=7.5,
                        help='Guidance scale for reference generation')
    parser.add_argument('--if_height', type=int, default=512,
                        help='Reference image generation height')
    parser.add_argument('--if_width', type=int, default=512,
                        help='Reference image generation width')
    parser.add_argument('--if_cpu_offload', action='store_true',
                        help='Enable model CPU offload for lower GPU memory usage')
    args = parser.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but not available in this shell. "
            "Run this on a GPU allocation (sbatch/srun) or pass --device cpu."
        )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FID EVALUATION")
    print("=" * 60)
    
    # Render mesh
    mesh_renders = render_mesh_multiview(
        args.mesh_path,
        n_views=args.n_views,
        device=args.device
    )
    
    # Save mesh renders
    render_dir = output_dir / "mesh_renders"
    render_dir.mkdir(exist_ok=True)
    for i, img in enumerate(mesh_renders):
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(render_dir / f"view_{i:02d}.png")
    print(f"Saved {len(mesh_renders)} mesh renders to {render_dir}")
    
    # Generate reference images
    ref_images = generate_reference_images(
        args.text_prompt,
        n_images=args.n_references,
        device=args.device,
        if_model_size=args.if_model_size,
        if_num_inference_steps=args.if_num_inference_steps,
        if_guidance_scale=args.if_guidance_scale,
        if_height=args.if_height,
        if_width=args.if_width,
        if_cpu_offload=args.if_cpu_offload,
    )
    
    # Save reference images
    ref_dir = output_dir / "reference_images"
    ref_dir.mkdir(exist_ok=True)
    for i, img in enumerate(ref_images):
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(ref_dir / f"ref_{i:02d}.png")
    print(f"Saved {len(ref_images)} reference images to {ref_dir}")
    
    # Compute FID
    print("\nComputing FID...")
    fid_evaluator = FIDEvaluator(args.device)
    fid_score = fid_evaluator.compute_fid(mesh_renders, ref_images)
    
    print(f"\n{'=' * 60}")
    print(f"FID SCORE: {fid_score:.2f}")
    print(f"{'=' * 60}")
    
    # Save results
    results = {
        'fid_score': fid_score,
        'mesh_path': args.mesh_path,
        'text_prompt': args.text_prompt,
        'n_views': args.n_views,
        'n_references': args.n_references,
        'if_model_size': args.if_model_size,
        'if_num_inference_steps': args.if_num_inference_steps,
        'if_guidance_scale': args.if_guidance_scale,
        'if_height': args.if_height,
        'if_width': args.if_width,
        'if_cpu_offload': args.if_cpu_offload,
    }
    
    with open(output_dir / "fid_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'fid_results.json'}")
    
    return fid_score


if __name__ == '__main__':
    main()
