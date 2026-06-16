#!/usr/bin/env python3
"""
Quick PCA Visualization Test Script

Tests the 2-step DINO PCA visualization on various meshes:
- hound, dinosaur, shrimp, guitar, helmet

Outputs colored PLY files to visualize the semantic correspondence.

Usage:
    python test_pca_visualization.py [--meshes mesh1,mesh2,...] [--output_dir dir]
    
Example:
    python test_pca_visualization.py --meshes hound,dinosaur,shrimp
"""

import os
import sys
import argparse
import numpy as np
import trimesh
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_tracking.dino_pca_visualization import (
    DINOPCAColorizer,
    create_dino_pca_visualization
)
from semantic_tracking.vertex_color_tracking import export_ply


# Mesh paths configuration
MESH_PATHS = {
    'hound': 'meshes/hound.obj',
    'dinosaur_014': 'meshes/decimated/dinosaur_014_decimated.obj',
    'dinosaur_032': 'meshes/decimated/dinosaur_032_decimated.obj',
    'shrimp': 'data/Omni6DPose/PAM/object_meshes/omniobject3d-shrimp_001/Aligned.obj',
    'guitar': 'data/Omni6DPose/PAM/object_meshes/omniobject3d-guitar_001/Aligned.obj',
    'helmet': 'data/Omni6DPose/PAM/object_meshes/omniobject3d-helmet_002/Aligned.obj',
    # Additional interesting meshes
    'teddy': 'data/Omni6DPose/PAM/object_meshes/omniobject3d-teddy_bear_001/Aligned.obj',
    'doll': 'data/Omni6DPose/PAM/object_meshes/omniobject3d-doll_001/Aligned.obj',
    'dinosaur_omni': 'data/Omni6DPose/PAM/object_meshes/omniobject3d-dinosaur_014/Aligned.obj',
    'toy_plane': 'data/Omni6DPose/PAM/object_meshes/omniobject3d-toy_plane_001/Aligned.obj',
}


def load_mesh(mesh_path: str):
    """Load mesh using trimesh."""
    print(f"Loading mesh from: {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh')
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int64)
    print(f"  Loaded: {len(vertices)} vertices, {len(faces)} faces")
    return vertices, faces


def visualize_mesh_pca(
    mesh_name: str,
    mesh_path: str,
    output_dir: str,
    device: str = 'cuda',
    n_views: int = 8
):
    """
    Generate DINO PCA visualization for a single mesh.
    
    Args:
        mesh_name: Name for the output file
        mesh_path: Path to input mesh
        output_dir: Directory to save output PLY
        device: CUDA device
        n_views: Number of viewpoints for feature extraction
    """
    print(f"\n{'='*60}")
    print(f"Processing: {mesh_name}")
    print(f"{'='*60}")
    
    # Check mesh exists
    if not os.path.exists(mesh_path):
        print(f"ERROR: Mesh not found: {mesh_path}")
        return None
    
    # Load mesh
    vertices, faces = load_mesh(mesh_path)
    
    # Output path
    output_path = os.path.join(output_dir, f"{mesh_name}_dino_pca.ply")
    
    # Create DINO PCA visualization
    colors = create_dino_pca_visualization(
        vertices=vertices,
        faces=faces,
        output_path=output_path,
        dino_model='dinov2_vits14_reg',  # With register tokens
        n_views=n_views,
        normalize_mesh=True,
        device=device
    )
    
    print(f"Saved: {output_path}")
    print(f"Color stats: min={colors.min():.3f}, max={colors.max():.3f}, mean={colors.mean():.3f}")
    
    return colors


def main():
    parser = argparse.ArgumentParser(
        description='Test DINO PCA visualization on various meshes'
    )
    parser.add_argument(
        '--meshes', 
        type=str, 
        default='hound,dinosaur_014,shrimp,guitar,helmet',
        help='Comma-separated list of mesh names to visualize'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/pca_test',
        help='Output directory for PLY files'
    )
    parser.add_argument(
        '--n_views',
        type=int,
        default=8,
        help='Number of viewpoints for feature extraction'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--list_meshes',
        action='store_true',
        help='List available mesh names and exit'
    )
    
    args = parser.parse_args()
    
    # List available meshes
    if args.list_meshes:
        print("Available meshes:")
        for name, path in MESH_PATHS.items():
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {name}: {path}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Parse mesh names
    mesh_names = [m.strip() for m in args.meshes.split(',')]
    
    # Validate mesh names
    invalid = [m for m in mesh_names if m not in MESH_PATHS]
    if invalid:
        print(f"WARNING: Unknown meshes: {invalid}")
        print(f"Available: {list(MESH_PATHS.keys())}")
        mesh_names = [m for m in mesh_names if m in MESH_PATHS]
    
    if not mesh_names:
        print("No valid meshes to process!")
        return
    
    print(f"\nWill process {len(mesh_names)} meshes: {mesh_names}")
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Process each mesh
    results = {}
    for mesh_name in mesh_names:
        mesh_path = MESH_PATHS[mesh_name]
        try:
            colors = visualize_mesh_pca(
                mesh_name=mesh_name,
                mesh_path=mesh_path,
                output_dir=args.output_dir,
                device=args.device,
                n_views=args.n_views
            )
            results[mesh_name] = 'success' if colors is not None else 'failed'
        except Exception as e:
            print(f"ERROR processing {mesh_name}: {e}")
            import traceback
            traceback.print_exc()
            results[mesh_name] = 'error'
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "✓" if status == 'success' else "✗"
        print(f"  {icon} {name}: {status}")
    
    print(f"\nOutput files saved to: {args.output_dir}/")
    print("View PLY files with MeshLab, Blender, or any 3D viewer supporting vertex colors.")


if __name__ == '__main__':
    main()
