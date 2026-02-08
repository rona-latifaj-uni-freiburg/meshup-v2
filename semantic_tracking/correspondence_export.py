"""
Correspondence Export Utilities for MeshUp

This module provides functions to export meshes with correspondence information
and create visualizations of semantic tracking.
"""

import torch
import numpy as np
import os
from typing import Optional, Union, Dict, List, Tuple
import json


def export_mesh_with_colors(
    filepath: str,
    vertices: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
    colors: Union[torch.Tensor, np.ndarray],
    format: str = 'ply'
):
    """
    Export a mesh with vertex colors to file.
    
    Args:
        filepath: Output file path
        vertices: Vertex positions, shape (V, 3)
        faces: Face indices, shape (F, 3)
        colors: RGB colors per vertex, shape (V, 3) in [0, 1]
        format: Output format ('ply' or 'obj')
    """
    # Convert to numpy
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    
    # Ensure correct shape
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    colors = np.asarray(colors, dtype=np.float32)
    
    # Clip colors to valid range
    colors = np.clip(colors, 0, 1)
    
    if format.lower() == 'ply':
        _export_ply(filepath, vertices, faces, colors)
    elif format.lower() == 'obj':
        _export_obj_colored(filepath, vertices, faces, colors)
    else:
        raise ValueError(f"Unknown format: {format}")


def _export_ply(filepath: str, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    """Export to PLY format with vertex colors."""
    if not filepath.endswith('.ply'):
        filepath += '.ply'
    
    n_vertices = len(vertices)
    n_faces = len(faces)
    
    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Vertices with colors
        colors_uint8 = (colors * 255).astype(np.uint8)
        for i in range(n_vertices):
            v = vertices[i]
            c = colors_uint8[i]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        
        # Faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Exported colored mesh to {filepath}")


def _export_obj_colored(filepath: str, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    """Export to OBJ format with vertex colors as extended vertex format."""
    if not filepath.endswith('.obj'):
        filepath += '.obj'
    
    # OBJ doesn't officially support vertex colors, but some software accepts
    # the extended format: v x y z r g b
    with open(filepath, 'w') as f:
        f.write("# Mesh with vertex colors\n")
        f.write("# Format: v x y z r g b\n")
        
        # Vertices with colors
        for i, (v, c) in enumerate(zip(vertices, colors)):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        
        # Faces
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Exported colored mesh to {filepath}")


def export_correspondence_map(
    filepath: str,
    original_vertices: Union[torch.Tensor, np.ndarray],
    deformed_vertices: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    part_labels: Optional[np.ndarray] = None,
    part_names: Optional[List[str]] = None,
    metadata: Optional[Dict] = None
):
    """
    Export a comprehensive correspondence map file.
    
    This creates a JSON file containing all correspondence information
    that can be used for analysis or visualization in external tools.
    
    Args:
        filepath: Output file path (.json)
        original_vertices: Original vertex positions, shape (V, 3)
        deformed_vertices: Deformed vertex positions, shape (V, 3)
        faces: Face indices, shape (F, 3)
        colors: Optional vertex colors, shape (V, 3)
        part_labels: Optional part label per vertex
        part_names: Optional names for each part label
        metadata: Optional additional metadata to include
    """
    # Convert to numpy
    def to_list(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
        elif isinstance(x, np.ndarray):
            return x.tolist()
        return x
    
    data = {
        'num_vertices': len(original_vertices) if hasattr(original_vertices, '__len__') else original_vertices.shape[0],
        'num_faces': len(faces) if hasattr(faces, '__len__') else faces.shape[0],
        'original_vertices': to_list(original_vertices),
        'deformed_vertices': to_list(deformed_vertices),
        'faces': to_list(faces),
    }
    
    if colors is not None:
        data['vertex_colors'] = to_list(colors)
    
    if part_labels is not None:
        data['part_labels'] = to_list(part_labels)
        
    if part_names is not None:
        data['part_names'] = part_names
    
    # Compute displacement statistics
    orig = np.asarray(original_vertices)
    deformed = np.asarray(deformed_vertices)
    displacement = deformed - orig
    
    data['statistics'] = {
        'mean_displacement': float(np.linalg.norm(displacement, axis=1).mean()),
        'max_displacement': float(np.linalg.norm(displacement, axis=1).max()),
        'displacement_std': float(np.linalg.norm(displacement, axis=1).std()),
    }
    
    if metadata is not None:
        data['metadata'] = metadata
    
    if not filepath.endswith('.json'):
        filepath += '.json'
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported correspondence map to {filepath}")


def export_deformation_sequence(
    output_dir: str,
    vertices_sequence: List[Union[torch.Tensor, np.ndarray]],
    faces: Union[torch.Tensor, np.ndarray],
    colors: Union[torch.Tensor, np.ndarray],
    prefix: str = 'frame'
):
    """
    Export a sequence of deformed meshes with consistent vertex colors.
    
    This is useful for creating animations showing the deformation process.
    
    Args:
        output_dir: Output directory
        vertices_sequence: List of vertex positions at each frame
        faces: Face indices (constant across frames)
        colors: Vertex colors (constant across frames)
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, vertices in enumerate(vertices_sequence):
        filepath = os.path.join(output_dir, f"{prefix}_{i:04d}.ply")
        export_mesh_with_colors(filepath, vertices, faces, colors, format='ply')
    
    print(f"Exported {len(vertices_sequence)} frames to {output_dir}")


def load_correspondence_map(filepath: str) -> Dict:
    """
    Load a correspondence map from JSON file.
    
    Args:
        filepath: Input file path (.json)
    
    Returns:
        Dictionary containing correspondence information
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays
    data['original_vertices'] = np.array(data['original_vertices'], dtype=np.float32)
    data['deformed_vertices'] = np.array(data['deformed_vertices'], dtype=np.float32)
    data['faces'] = np.array(data['faces'], dtype=np.int32)
    
    if 'vertex_colors' in data:
        data['vertex_colors'] = np.array(data['vertex_colors'], dtype=np.float32)
    
    if 'part_labels' in data:
        data['part_labels'] = np.array(data['part_labels'], dtype=np.int32)
    
    return data


def visualize_correspondence_displacement(
    original_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    colormap: str = 'viridis'
):
    """
    Create a mesh colored by displacement magnitude.
    
    Args:
        original_vertices: Original positions, shape (V, 3)
        deformed_vertices: Deformed positions, shape (V, 3)
        faces: Face indices, shape (F, 3)
        output_path: Output file path
        colormap: Color scheme for displacement visualization
    """
    displacement = np.linalg.norm(deformed_vertices - original_vertices, axis=1)
    
    # Normalize displacement to [0, 1]
    if displacement.max() > 0:
        normalized = displacement / displacement.max()
    else:
        normalized = displacement
    
    # Apply colormap (simple viridis-like)
    if colormap == 'viridis':
        colors = np.column_stack([
            0.267 + 0.329 * normalized,
            0.004 + 0.873 * normalized - 0.277 * normalized**2,
            0.329 + 0.387 * normalized - 0.716 * normalized**2
        ]).astype(np.float32)
        colors = np.clip(colors, 0, 1)
    elif colormap == 'hot':
        # Black -> Red -> Yellow -> White
        colors = np.zeros((len(normalized), 3), dtype=np.float32)
        colors[:, 0] = np.clip(normalized * 3, 0, 1)
        colors[:, 1] = np.clip(normalized * 3 - 1, 0, 1)
        colors[:, 2] = np.clip(normalized * 3 - 2, 0, 1)
    else:
        # Default grayscale
        colors = np.column_stack([normalized, normalized, normalized]).astype(np.float32)
    
    export_mesh_with_colors(output_path, deformed_vertices, faces, colors, format='ply')
    print(f"Exported displacement visualization to {output_path}")
