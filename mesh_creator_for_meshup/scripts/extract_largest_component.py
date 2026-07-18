#!/usr/bin/env python3
"""
Extract largest connected component from a PLY mesh.
Removes disconnected parts (e.g., wheels, separate geometry).
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import trimesh

def extract_largest_component_ply(ply_path, output_path=None):
    """
    Load PLY mesh, extract largest connected component, save result.
    
    Args:
        ply_path: Path to input PLY file
        output_path: Path to save result (default: overwrite input)
    
    Returns:
        dict with stats: num_components, kept_vertices, removed_vertices, etc.
    """
    
    if output_path is None:
        output_path = ply_path
    
    # Load mesh
    mesh = trimesh.load(ply_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometry = list(mesh.geometry.values())
        if not geometry:
            print(f"Empty scene in {ply_path}, skipping")
            return None
        mesh = trimesh.util.concatenate(geometry)
    
    if mesh.vertices.shape[0] == 0:
        print(f"Empty mesh in {ply_path}, skipping")
        return None
    
    n_verts = mesh.vertices.shape[0]
    if mesh.faces.shape[0] == 0:
        print(f"Mesh has no faces, keeping as-is")
        return {"num_components": 1, "kept_vertices": n_verts, "removed_vertices": 0}

    # Build adjacency from triangle edges.
    edges = np.vstack(
        [
            mesh.faces[:, [0, 1]],
            mesh.faces[:, [1, 2]],
            mesh.faces[:, [2, 0]],
        ]
    )
    
    # Create adjacency matrix (undirected)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row))
    adj = sparse.coo_matrix((data, (row, col)), shape=(n_verts, n_verts))
    
    # Find connected components
    n_components, labels = connected_components(adj, directed=False)
    
    if n_components == 1:
        print(f"  Single component: {n_verts} vertices, keeping as-is")
        return {"num_components": 1, "kept_vertices": n_verts, "removed_vertices": 0}
    
    # Find largest component by vertex count.
    component_sizes = np.bincount(labels)
    largest_component_id = np.argmax(component_sizes)
    largest_size = component_sizes[largest_component_id]
    
    # Keep only vertices in largest component
    keep_mask = labels == largest_component_id
    kept_vert_indices = np.where(keep_mask)[0]
    
    # Map old vertex indices to new
    old_to_new = np.full(n_verts, -1, dtype=np.int32)
    old_to_new[keep_mask] = np.arange(kept_vert_indices.shape[0])
    
    # Filter vertices
    new_verts = mesh.vertices[keep_mask]
    
    face_keep = np.all(keep_mask[mesh.faces], axis=1)
    new_faces = old_to_new[mesh.faces[face_keep]].astype(np.int32, copy=False)
    
    # Create new mesh
    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    
    # Save
    new_mesh.export(output_path)
    
    removed = n_verts - largest_size
    print(f"  Components: {n_components}, Kept: {largest_size}/{n_verts} verts, Removed: {removed} verts, Faces: {len(new_faces)}")
    
    return {
        "num_components": n_components,
        "kept_vertices": largest_size,
        "removed_vertices": removed,
        "kept_faces": len(new_faces),
        "largest_component_id": largest_component_id
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract largest connected component from PLY mesh")
    parser.add_argument("input_ply", help="Input PLY file")
    parser.add_argument("--output", "-o", default=None, help="Output PLY (default: overwrite input)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_ply)
    output_path = Path(args.output) if args.output else input_path
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        exit(1)
    
    result = extract_largest_component_ply(input_path, output_path)
    if result:
        print(f"Saved to {output_path}")
