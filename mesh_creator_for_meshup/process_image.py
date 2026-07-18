import argparse
import logging
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_sample_name(image_path: Path) -> str:
    """Extract sample name from image path (filename without extension)."""
    return image_path.stem


def process_single_image(
    base_path: Path,
    image_name: str,
    inference: Any,
    load_image_fn: Callable[[str], Any],
    load_mask_fn: Callable[[str], Any],
    seed: int = 42,
    rotate_x_deg: float = 0.0,
    rotate_y_deg: float = 0.0,
    rotate_z_deg: float = 0.0,
    require_mesh: bool = False,
) -> dict:
    """Process a single image and return results."""
    image_path = base_path / "image" / image_name
    mask_path = base_path / "mask" / image_name
    sample_name = get_sample_name(image_path)

    try:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Load image
        image = load_image_fn(str(image_path))

        # Load mask
        mask = load_mask_fn(str(mask_path))

        # Run inference
        output = inference(image, mask, seed=seed)

        # Prepare output paths
        meta_dir = base_path / "sam3D" / "meta"
        mesh_splat_dir = base_path / "sam3D" / "splat"
        mesh_mesh_dir = base_path / "sam3D" / "mesh"

        # Create directories
        meta_dir.mkdir(parents=True, exist_ok=True)
        mesh_splat_dir.mkdir(parents=True, exist_ok=True)
        mesh_mesh_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        # Note: output['translation'], output['scale'] are already decoded to
        # camera space by the pose_decoder (they overwrite the raw SSI values).
        # output['rotation'] is the decoded camera-space quaternion (w,x,y,z).
        meta_file = meta_dir / f"{sample_name}.yaml"

        def _to_list(v):
            return v.cpu().numpy().tolist() if hasattr(v, 'cpu') else v

        metadata = {
            # Raw model outputs (SSI space)
            '6drotation_normalized': _to_list(output['6drotation_normalized']),
            'translation_scale': _to_list(output['translation_scale']),
            'shape': _to_list(output['shape']),
            # Decoded camera-space pose (after pose_decoder + downsample correction)
            'rotation_cam': _to_list(output['rotation']),       # quaternion (w,x,y,z), shape (1, 4)
            'translation_cam': _to_list(output['translation']), # camera-space translation, shape (1, 3)
            'scale_cam': _to_list(output['scale']),             # camera-space scale, shape (1, 3)
            # Optional post-export Euler rotation in degrees, for MeshUp frame alignment
            'export_rotation_deg_xyz': [rotate_x_deg, rotate_y_deg, rotate_z_deg],
        }

        with open(meta_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        # Save gaussian splat if available
        splat_file = mesh_splat_dir / f"{sample_name}.ply"
        if "gs" in output and output["gs"] is not None:
            gs = output["gs"]
            if isinstance(gs, list):
                gs = gs[0] if len(gs) > 0 else None
            if gs is not None:
                gs.save_ply(str(splat_file))

        # Save mesh (if available)
        mesh_file = mesh_mesh_dir / f"{sample_name}.ply"
        if "mesh" in output and output["mesh"] is not None:
            mesh = output["mesh"]
            if isinstance(mesh, list):
                mesh = mesh[0] if len(mesh) > 0 else None

            if mesh is not None:
                # Convert MeshExtractResult to trimesh
                try:
                    vertices = mesh.vertices.float().cpu().numpy()
                    faces = mesh.faces.cpu().numpy()

                    # Create trimesh object with vertex colors if available
                    if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None:
                        vert_colors = mesh.vertex_attrs[:, :3].cpu().numpy()
                        # Convert to [0, 255] range if needed
                        if vert_colors.max() <= 1.0:
                            vert_colors = (vert_colors * 255).astype(np.uint8)
                        mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                        mesh_obj.visual.vertex_colors = vert_colors
                    else:
                        mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

                    if any(v != 0.0 for v in (rotate_x_deg, rotate_y_deg, rotate_z_deg)):
                        transform = trimesh.transformations.euler_matrix(
                            np.deg2rad(rotate_x_deg),
                            np.deg2rad(rotate_y_deg),
                            np.deg2rad(rotate_z_deg),
                            axes="sxyz",
                        )
                        mesh_obj.apply_transform(transform)

                    mesh_obj.export(str(mesh_file))
                except Exception as e:
                    logger.warning(f"Failed to save mesh for {sample_name}: {e}")
            else:
                logger.warning(f"Mesh is None for {sample_name}")
        elif require_mesh:
            raise RuntimeError(
                f"Mesh output missing for {sample_name}. "
                "Set a mesh-enabled config or more GPU memory."
            )
        else:
            logger.warning(f"No mesh or gaussian output available for {sample_name}")

        return {
            'status': 'success',
            'sample': sample_name
        }

    except Exception as e:
        error_msg = f"Error processing {sample_name}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {
            'status': 'failed',
            'sample': sample_name,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Process one image with SAM-3D-Objects model")
    parser.add_argument(
        '--image_name',
        type=str,
        default='',
        help='Image filename in BASE_PATH/image, for example 0000.jpg'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all images that have matching masks in BASE_PATH/image and BASE_PATH/mask'
    )
    parser.add_argument(
        '--base_path',
        type=str,
        default=str(SCRIPT_DIR),
        help='Base directory containing image/, mask/, sam3D/ folders'
    )
    parser.add_argument(
        '--sam3d_repo',
        type=str,
        default='',
        help='Path to cloned sam-3d-objects repository. If omitted, auto-detects.'
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Compile model for faster inference'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='checkpoints/hf/pipeline.yaml',
        help='Config path, relative to sam-3d-objects repo unless absolute'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for inference'
    )
    parser.add_argument(
        '--rotate_x_deg',
        type=float,
        default=0.0,
        help='Optional export rotation around X axis in degrees (for frame alignment)'
    )
    parser.add_argument(
        '--rotate_y_deg',
        type=float,
        default=0.0,
        help='Optional export rotation around Y axis in degrees (for frame alignment)'
    )
    parser.add_argument(
        '--rotate_z_deg',
        type=float,
        default=0.0,
        help='Optional export rotation around Z axis in degrees (for frame alignment)'
    )
    parser.add_argument(
        '--require_mesh',
        action='store_true',
        help='Fail the run if a triangle mesh with faces is not produced'
    )

    args = parser.parse_args()

    base_path = Path(args.base_path).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    repo_candidates = []
    if args.sam3d_repo:
        repo_candidates.append(Path(args.sam3d_repo).expanduser().resolve())
    repo_candidates.append(SCRIPT_DIR)
    repo_candidates.append((SCRIPT_DIR / "sam-3d-objects").resolve())

    sam3d_repo = None
    for candidate in repo_candidates:
        if (candidate / "notebook" / "inference.py").exists():
            sam3d_repo = candidate
            break

    if sam3d_repo is None:
        raise FileNotFoundError(
            "Could not locate sam-3d-objects repository. "
            "Pass --sam3d_repo /path/to/sam-3d-objects"
        )

    import sys

    sys.path.insert(0, str(sam3d_repo / "notebook"))
    from inference import Inference, load_image, load_mask

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = sam3d_repo / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    logger.info("Loading SAM-3D model from %s", config_path)
    inference = Inference(str(config_path), compile=args.compile)

    image_dir = base_path / "image"
    mask_dir = base_path / "mask"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    if args.all:
        image_names = sorted([
            p.name
            for p in image_dir.iterdir()
            if p.is_file() and (mask_dir / p.name).exists()
        ])
        if not image_names:
            raise FileNotFoundError(
                "No image/mask pairs found. Expected matching filenames in image/ and mask/"
            )
    else:
        if not args.image_name:
            raise ValueError("Provide --image_name <file> or use --all")
        image_names = [args.image_name]

    succeeded = 0
    failed = 0

    for image_name in image_names:
        result = process_single_image(
            base_path=base_path,
            image_name=image_name,
            inference=inference,
            load_image_fn=load_image,
            load_mask_fn=load_mask,
            seed=args.seed,
            rotate_x_deg=args.rotate_x_deg,
            rotate_y_deg=args.rotate_y_deg,
            rotate_z_deg=args.rotate_z_deg,
            require_mesh=args.require_mesh,
        )
        if result['status'] == 'success':
            succeeded += 1
        else:
            failed += 1

    logger.info("Processing completed: %d succeeded, %d failed", succeeded, failed)

    if failed > 0:
        raise RuntimeError(f"{failed} image(s) failed. Check logs above.")


if __name__ == "__main__":
    main()
