#!/usr/bin/env python3
"""
Post-training evaluation script that computes:
1. FID score between mesh renders and diffusion-generated references.
2. CLIP similarity between mesh renders and the text prompt.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
import importlib.util
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def _load_fid_evaluator_class():
    """Load FIDEvaluator directly from file."""
    module_path = Path(__file__).parent / "semantic_tracking" / "fid_evaluation.py"
    spec = importlib.util.spec_from_file_location("fid_evaluation_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FIDEvaluator

FIDEvaluator = _load_fid_evaluator_class()

def _load_clip_evaluator_class():
    """Load CLIPEvaluator with openai-clip or fallback to HF CLIP."""
    try:
        import clip as openai_clip

        class CLIPEvaluator:
            def __init__(self, device='cuda'):
                self.device = device
                self.model, self.preprocess = openai_clip.load("ViT-B/32", device=self.device)

            def compute_clip_similarity(self, images, text_prompt):
                images_preprocessed = torch.stack([
                    self.preprocess(
                        Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                    )
                    for img in images
                ]).to(self.device)
                text = openai_clip.tokenize([text_prompt]).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(images_preprocessed)
                    text_features = self.model.encode_text(text)

                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    similarity = (100.0 * image_features @ text_features.T).mean()
                return similarity.item()

        return CLIPEvaluator
    except ImportError:
        from transformers import CLIPModel, CLIPProcessor

        class CLIPEvaluator:
            def __init__(self, device='cuda'):
                self.device = device
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            def compute_clip_similarity(self, images, text_prompt):
                pil_images = [
                    Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                    for img in images
                ]
                inputs = self.processor(
                    text=[text_prompt], images=pil_images, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                    text_features = self.model.get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).mean()
                return similarity.item()

        return CLIPEvaluator

CLIPEvaluator = _load_clip_evaluator_class()


def generate_reference_images(
    text_prompt: str, n_images: int, device: str, if_model_size: str,
    if_num_inference_steps: int, if_guidance_scale: float, if_height: int,
    if_width: int, if_cpu_offload: bool
) -> torch.Tensor:
    from diffusers import IFPipeline
    model_map = {'M': 'DeepFloyd/IF-I-M-v1.0', 'L': 'DeepFloyd/IF-I-L-v1.0', 'XL': 'DeepFloyd/IF-I-XL-v1.0'}
    pipeline = IFPipeline.from_pretrained(
        model_map[if_model_size.upper()], variant="fp16", torch_dtype=torch.float16,
        safety_checker=None, watermarker=None, feature_extractor=None, requires_safety_checker=False
    )
    if if_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    images = []
    for i in range(n_images):
        with torch.no_grad():
            result = pipeline(
                prompt=text_prompt, height=if_height, width=if_width,
                num_inference_steps=if_num_inference_steps, guidance_scale=if_guidance_scale,
                generator=torch.Generator(device=device).manual_seed(42 + i)
            )
            img_tensor = torch.from_numpy(np.array(result.images[0])).float() / 255.0
            images.append(img_tensor.permute(2, 0, 1))
    return torch.stack(images)

def render_mesh_multiview(mesh_path: str, n_views: int, resolution: int, device: str) -> torch.Tensor:
    import nvdiffrast.torch as dr
    from nvdiffmodeling.src import obj, mesh, render, texture
    from utilities.camera import get_camera_params
    from utilities.helpers import create_scene
    
    m = obj.load_obj(mesh_path)
    m = mesh.unit_size(m)
    kd = texture.Texture2D(torch.full((1, 512, 512, 3), 0.5, device=device))
    ks = texture.Texture2D(torch.zeros((1, 512, 512, 3), device=device))
    normal = texture.Texture2D(torch.tensor([[[0.0, 0.0, 1.0]]], device=device).expand(1, 512, 512, 3))
    m = mesh.Mesh(
        v_pos=m.v_pos.to(device),
        t_pos_idx=m.t_pos_idx.to(device),
        v_tex=m.v_tex.to(device) if m.v_tex is not None else None,
        t_tex_idx=m.t_tex_idx.to(device) if m.t_tex_idx is not None else None,
        material={"bsdf": "diffuse", "kd": kd, "ks": ks, "normal": normal},
    )
    scene = create_scene([m.eval()], sz=512)
    scene = mesh.compute_tangents(mesh.auto_normals(scene))
    glctx = dr.RasterizeGLContext()
    images = []
    azimuths = [i * (360.0 / n_views) for i in range(n_views)]
    for azim in azimuths:
        cam_params = get_camera_params(30.0, azim, 3.0, resolution, 60.0)
        for k, v in cam_params.items():
            if isinstance(v, torch.Tensor): cam_params[k] = v.to(device)
        final_m = scene.eval(cam_params)
        rendered = render.render_mesh(
            glctx, final_m, cam_params["mvp"], cam_params["campos"], cam_params["lightpos"],
            5.0, resolution, spp=1, num_layers=1, msaa=False,
            background=torch.ones(1, resolution, resolution, 3, device=device)
        )
        images.append(rendered.permute(0, 3, 1, 2)[0])
    return torch.stack(images)

def main():
    parser = argparse.ArgumentParser(description='Evaluate mesh with FID and CLIP similarity')
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--text_prompt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_views', type=int, default=8)
    parser.add_argument('--n_references', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--if_model_size', type=str, default='XL', choices=['M', 'L', 'XL'])
    parser.add_argument('--if_num_inference_steps', type=int, default=50)
    parser.add_argument('--if_guidance_scale', type=float, default=7.5)
    parser.add_argument('--if_height', type=int, default=512)
    parser.add_argument('--if_width', type=int, default=512)
    parser.add_argument('--if_cpu_offload', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_renders = render_mesh_multiview(args.mesh_path, args.n_views, args.if_height, args.device)
    ref_images = generate_reference_images(
        args.text_prompt, args.n_references, args.device, args.if_model_size,
        args.if_num_inference_steps, args.if_guidance_scale, args.if_height,
        args.if_width, args.if_cpu_offload
    )

    fid_evaluator = FIDEvaluator(args.device)
    fid_score = fid_evaluator.compute_fid(mesh_renders, ref_images)
    
    clip_evaluator = CLIPEvaluator(args.device)
    clip_similarity = clip_evaluator.compute_clip_similarity(mesh_renders, args.text_prompt)

    results = {
        'fid_score': fid_score,
        'clip_similarity': clip_similarity,
        'mesh_path': args.mesh_path,
        'text_prompt': args.text_prompt,
    }
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"FID Score: {fid_score:.2f}")
    print(f"CLIP Similarity: {clip_similarity:.4f}")
    print(f"Results saved to {output_dir / 'evaluation_results.json'}")

if __name__ == '__main__':
    main()
