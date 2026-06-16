"""
Mesh Decimation Script

Creates simplified versions of dinosaur meshes by reducing vertex/face count.
This helps with optimization stability and faster experimentation.

Usage:
    python decimate_meshes.py
"""

import pymeshlab
import os
from pathlib import Path


def decimate_mesh(input_path: str, output_path: str, target_ratio: float = 0.33):
    """
    Decimate a mesh to reduce its vertex/face count.
    
    Args:
        input_path: Path to input mesh
        output_path: Path for output mesh
        target_ratio: Target ratio of faces to keep (0.33 = ~1/3 of original)
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)
    
    original_mesh = ms.current_mesh()
    original_verts = original_mesh.vertex_number()
    original_faces = original_mesh.face_number()
    
    print(f"\nProcessing: {input_path}")
    print(f"  Original: {original_verts} vertices, {original_faces} faces")
    
    # Calculate target faces
    target_faces = int(original_faces * target_ratio)
    
    # Apply quadric edge collapse decimation
    # This is a high-quality decimation that preserves shape well
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        qualitythr=0.5,  # Quality threshold
        planarquadric=True  # Better handling of planar regions
    )
    
    decimated_mesh = ms.current_mesh()
    new_verts = decimated_mesh.vertex_number()
    new_faces = decimated_mesh.face_number()
    
    print(f"  Decimated: {new_verts} vertices, {new_faces} faces")
    print(f"  Reduction: {100*(1-new_verts/original_verts):.1f}% vertices, {100*(1-new_faces/original_faces):.1f}% faces")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ms.save_current_mesh(output_path)
    print(f"  Saved to: {output_path}")
    
    return new_verts, new_faces


def main():
    # Base paths
    data_dir = Path("./data/Omni6DPose/PAM/object_meshes")
    output_dir = Path("./meshes/decimated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Meshes to decimate
    meshes = [
        ("omniobject3d-dinosaur_014", "dinosaur_014_decimated.obj"),
        ("omniobject3d-dinosaur_032", "dinosaur_032_decimated.obj"),
    ]
    
    print("=" * 60)
    print("MESH DECIMATION SCRIPT")
    print("Reducing mesh complexity by ~3x for faster experimentation")
    print("=" * 60)
    
    results = []
    for mesh_folder, output_name in meshes:
        input_path = data_dir / mesh_folder / "Aligned.obj"
        output_path = output_dir / output_name
        
        if input_path.exists():
            verts, faces = decimate_mesh(str(input_path), str(output_path), target_ratio=0.33)
            results.append((output_name, verts, faces))
        else:
            print(f"\nWarning: {input_path} not found!")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, verts, faces in results:
        print(f"  {name}: {verts} vertices, {faces} faces")
    
    print("\nDecimated meshes saved to ./meshes/decimated/")
    print("These can be used as source or target meshes for faster experiments.")


if __name__ == "__main__":
    main()
