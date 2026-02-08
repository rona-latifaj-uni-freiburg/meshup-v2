"""
Vertex Color Tracking for Semantic Correspondence in MeshUp

This module provides utilities to track semantic correspondence during mesh
deformation by assigning and maintaining vertex colors. Each vertex's color
encodes its semantic part identity, allowing you to visualize how parts
(e.g., "right paw of cat" -> "right paw of dog") correspond after deformation.

Usage:
    tracker = VertexColorTracker(mesh_vertices, mesh_faces)
    tracker.initialize_colors_by_position()  # or by clustering, or manual
    # During/after deformation:
    tracker.export_colored_mesh(deformed_vertices, "output.ply")
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from sklearn.cluster import KMeans
import colorsys


def generate_distinct_colors(n_colors: int, saturation: float = 0.8, value: float = 0.9) -> np.ndarray:
    """
    Generate n visually distinct colors using HSV color space.
    
    Args:
        n_colors: Number of distinct colors to generate
        saturation: Color saturation (0-1)
        value: Color brightness (0-1)
    
    Returns:
        Array of shape (n_colors, 3) with RGB values in [0, 1]
    """
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return np.array(colors, dtype=np.float32)


def position_to_color(positions: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Map 3D positions to RGB colors for visualization.
    
    This creates a smooth color gradient based on position, useful for
    tracking how vertices move during deformation.
    
    Args:
        positions: Vertex positions, shape (N, 3)
        normalize: Whether to normalize positions to [0, 1] range
    
    Returns:
        RGB colors, shape (N, 3) in [0, 1]
    """
    if normalize:
        pos_min = positions.min(axis=0, keepdims=True)
        pos_max = positions.max(axis=0, keepdims=True)
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1  # Avoid division by zero
        colors = (positions - pos_min) / pos_range
    else:
        colors = np.clip(positions, 0, 1)
    return colors.astype(np.float32)


class VertexColorTracker:
    """
    Tracks semantic correspondence through vertex colors during mesh deformation.
    
    The key insight is that vertex indices remain constant during Neural Jacobian
    Fields deformation - only positions change. By assigning each vertex a unique
    or part-based color before deformation, we can track which original vertices
    (and thus semantic parts) end up where.
    """
    
    def __init__(
        self,
        vertices: Union[torch.Tensor, np.ndarray],
        faces: Union[torch.Tensor, np.ndarray],
        device: str = 'cuda'
    ):
        """
        Initialize the vertex color tracker.
        
        Args:
            vertices: Original mesh vertices, shape (V, 3)
            faces: Mesh faces (triangles), shape (F, 3)
            device: Torch device for computations
        """
        if isinstance(vertices, torch.Tensor):
            self.original_vertices = vertices.detach().cpu().numpy()
        else:
            self.original_vertices = np.array(vertices)
            
        if isinstance(faces, torch.Tensor):
            self.faces = faces.detach().cpu().numpy()
        else:
            self.faces = np.array(faces)
            
        self.device = device
        self.num_vertices = len(self.original_vertices)
        self.num_faces = len(self.faces)
        
        # Initialize colors (will be set by one of the initialization methods)
        self.vertex_colors = None
        self.part_labels = None
        self.part_names = None
        
    def initialize_colors_by_position(self) -> np.ndarray:
        """
        Assign colors based on original vertex positions.
        
        This creates a smooth color gradient where spatially close vertices
        have similar colors. Useful for general correspondence visualization.
        
        Returns:
            Vertex colors, shape (V, 3)
        """
        self.vertex_colors = position_to_color(self.original_vertices)
        return self.vertex_colors
    
    def initialize_colors_by_axis(
        self,
        axis: str = 'y',
        colormap: str = 'rainbow'
    ) -> np.ndarray:
        """
        Assign colors based on position along a single axis.
        
        Useful for visualizing vertical (y) or horizontal (x, z) correspondence.
        
        Args:
            axis: Which axis to use ('x', 'y', or 'z')
            colormap: Color scheme ('rainbow', 'viridis', 'coolwarm')
        
        Returns:
            Vertex colors, shape (V, 3)
        """
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        values = self.original_vertices[:, axis_idx]
        
        # Normalize to [0, 1]
        v_min, v_max = values.min(), values.max()
        if v_max - v_min > 0:
            normalized = (values - v_min) / (v_max - v_min)
        else:
            normalized = np.zeros_like(values)
        
        # Apply colormap
        if colormap == 'rainbow':
            # HSV rainbow
            self.vertex_colors = np.zeros((self.num_vertices, 3), dtype=np.float32)
            for i, v in enumerate(normalized):
                self.vertex_colors[i] = colorsys.hsv_to_rgb(v * 0.8, 0.9, 0.9)
        elif colormap == 'viridis':
            # Simple viridis-like gradient
            self.vertex_colors = np.column_stack([
                0.267 + 0.329 * normalized,
                0.004 + 0.873 * normalized - 0.277 * normalized**2,
                0.329 + 0.387 * normalized - 0.716 * normalized**2
            ]).astype(np.float32)
            self.vertex_colors = np.clip(self.vertex_colors, 0, 1)
        elif colormap == 'coolwarm':
            # Blue to red
            self.vertex_colors = np.column_stack([
                normalized,
                0.3 * np.ones(self.num_vertices),
                1 - normalized
            ]).astype(np.float32)
        else:
            raise ValueError(f"Unknown colormap: {colormap}")
            
        return self.vertex_colors
    
    def initialize_colors_by_clustering(
        self,
        n_parts: int = 8,
        features: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign colors by clustering vertices into semantic parts.
        
        Uses K-means clustering on vertex positions (or custom features) to
        group vertices into parts, then assigns each part a distinct color.
        
        Args:
            n_parts: Number of parts to segment the mesh into
            features: Optional custom features for clustering, shape (V, D).
                      If None, uses vertex positions.
        
        Returns:
            Tuple of (vertex_colors, part_labels)
        """
        if features is None:
            features = self.original_vertices
            
        # Cluster vertices
        kmeans = KMeans(n_clusters=n_parts, random_state=42, n_init=10)
        self.part_labels = kmeans.fit_predict(features)
        
        # Assign distinct colors to each cluster
        part_colors = generate_distinct_colors(n_parts)
        self.vertex_colors = part_colors[self.part_labels]
        
        return self.vertex_colors, self.part_labels
    
    def initialize_colors_manual(
        self,
        part_masks: Dict[str, np.ndarray],
        part_colors: Optional[Dict[str, Tuple[float, float, float]]] = None
    ) -> np.ndarray:
        """
        Manually assign colors based on part masks.
        
        This is useful when you have ground-truth part segmentation or
        want to define specific semantic regions (e.g., "head", "body", "legs").
        
        Args:
            part_masks: Dictionary mapping part names to boolean masks or vertex indices
            part_colors: Optional dictionary mapping part names to RGB colors.
                        If None, colors are auto-generated.
        
        Returns:
            Vertex colors, shape (V, 3)
        """
        self.part_names = list(part_masks.keys())
        n_parts = len(self.part_names)
        
        # Generate colors if not provided
        if part_colors is None:
            auto_colors = generate_distinct_colors(n_parts)
            part_colors = {name: tuple(auto_colors[i]) for i, name in enumerate(self.part_names)}
        
        # Initialize with default color (gray for unassigned)
        self.vertex_colors = np.full((self.num_vertices, 3), 0.5, dtype=np.float32)
        self.part_labels = np.full(self.num_vertices, -1, dtype=np.int32)
        
        for part_idx, (part_name, mask) in enumerate(part_masks.items()):
            if isinstance(mask, (list, np.ndarray)):
                if len(mask) == self.num_vertices and mask.dtype == bool:
                    # Boolean mask
                    indices = np.where(mask)[0]
                else:
                    # Index array
                    indices = np.array(mask)
            else:
                continue
                
            self.vertex_colors[indices] = part_colors[part_name]
            self.part_labels[indices] = part_idx
            
        return self.vertex_colors
    
    def initialize_from_texture_uv(
        self,
        uv_coords: np.ndarray,
        texture_image: np.ndarray
    ) -> np.ndarray:
        """
        Initialize colors by sampling from a texture image using UV coordinates.
        
        This is useful if you have a pre-painted texture that defines parts.
        
        Args:
            uv_coords: UV coordinates per vertex, shape (V, 2)
            texture_image: RGB texture image, shape (H, W, 3)
        
        Returns:
            Vertex colors, shape (V, 3)
        """
        H, W = texture_image.shape[:2]
        
        # Convert UV to pixel coordinates
        u = np.clip(uv_coords[:, 0] * (W - 1), 0, W - 1).astype(int)
        v = np.clip((1 - uv_coords[:, 1]) * (H - 1), 0, H - 1).astype(int)
        
        # Sample texture
        self.vertex_colors = texture_image[v, u].astype(np.float32)
        if self.vertex_colors.max() > 1:
            self.vertex_colors /= 255.0
            
        return self.vertex_colors
    
    def get_face_colors(self) -> np.ndarray:
        """
        Compute per-face colors by averaging vertex colors.
        
        Returns:
            Face colors, shape (F, 3)
        """
        if self.vertex_colors is None:
            raise ValueError("Vertex colors not initialized. Call one of the initialize_* methods first.")
        
        return self.vertex_colors[self.faces].mean(axis=1)
    
    def get_correspondence_map(
        self,
        deformed_vertices: Union[torch.Tensor, np.ndarray]
    ) -> Dict:
        """
        Create a correspondence map between original and deformed mesh.
        
        Args:
            deformed_vertices: Deformed vertex positions, shape (V, 3)
        
        Returns:
            Dictionary containing correspondence information
        """
        if isinstance(deformed_vertices, torch.Tensor):
            deformed = deformed_vertices.detach().cpu().numpy()
        else:
            deformed = np.array(deformed_vertices)
            
        return {
            'original_positions': self.original_vertices.copy(),
            'deformed_positions': deformed.copy(),
            'vertex_colors': self.vertex_colors.copy() if self.vertex_colors is not None else None,
            'part_labels': self.part_labels.copy() if self.part_labels is not None else None,
            'part_names': self.part_names.copy() if self.part_names is not None else None,
            'displacement': deformed - self.original_vertices,
            'faces': self.faces.copy(),
        }
    
    def export_ply(
        self,
        filepath: str,
        vertices: Optional[Union[torch.Tensor, np.ndarray]] = None,
        colors: Optional[np.ndarray] = None
    ):
        """
        Export mesh with vertex colors to PLY format.
        
        Args:
            filepath: Output file path (.ply)
            vertices: Vertex positions to export. If None, uses original vertices.
            colors: Vertex colors to use. If None, uses tracked colors.
        """
        if vertices is None:
            verts = self.original_vertices
        elif isinstance(vertices, torch.Tensor):
            verts = vertices.detach().cpu().numpy()
        else:
            verts = np.array(vertices)
            
        if colors is None:
            if self.vertex_colors is None:
                self.initialize_colors_by_position()
            colors = self.vertex_colors
            
        export_ply(filepath, verts, self.faces, colors)
    
    def export_obj_with_colors(
        self,
        filepath: str,
        vertices: Optional[Union[torch.Tensor, np.ndarray]] = None
    ):
        """
        Export mesh with vertex colors to OBJ format (as a texture).
        
        Note: OBJ doesn't natively support vertex colors, so this creates
        a simple color texture and UV mapping.
        
        Args:
            filepath: Output file path (.obj)
            vertices: Vertex positions to export. If None, uses original vertices.
        """
        if vertices is None:
            verts = self.original_vertices
        elif isinstance(vertices, torch.Tensor):
            verts = vertices.detach().cpu().numpy()
        else:
            verts = np.array(vertices)
            
        if self.vertex_colors is None:
            self.initialize_colors_by_position()
            
        export_obj_with_vertex_colors(filepath, verts, self.faces, self.vertex_colors)


def initialize_semantic_colors(
    vertices: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
    method: str = 'position',
    **kwargs
) -> VertexColorTracker:
    """
    Convenience function to create and initialize a VertexColorTracker.
    
    Args:
        vertices: Mesh vertices, shape (V, 3)
        faces: Mesh faces, shape (F, 3)
        method: Initialization method:
            - 'position': Color by 3D position (XYZ -> RGB)
            - 'axis_y': Color by height (rainbow gradient)
            - 'axis_x': Color by X position
            - 'axis_z': Color by depth
            - 'cluster': K-means clustering
        **kwargs: Additional arguments passed to initialization method
    
    Returns:
        Initialized VertexColorTracker
    """
    tracker = VertexColorTracker(vertices, faces)
    
    if method == 'position':
        tracker.initialize_colors_by_position()
    elif method.startswith('axis_'):
        axis = method.split('_')[1]
        tracker.initialize_colors_by_axis(axis=axis, **kwargs)
    elif method == 'cluster':
        n_parts = kwargs.get('n_parts', 8)
        tracker.initialize_colors_by_clustering(n_parts=n_parts)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return tracker


def assign_part_colors(
    vertices: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
    part_vertices: Dict[str, List[int]],
    part_colors: Optional[Dict[str, Tuple[float, float, float]]] = None
) -> VertexColorTracker:
    """
    Assign colors to specific mesh parts by vertex indices.
    
    Args:
        vertices: Mesh vertices, shape (V, 3)
        faces: Mesh faces, shape (F, 3)
        part_vertices: Dictionary mapping part names to vertex index lists
        part_colors: Optional dictionary mapping part names to RGB colors
    
    Returns:
        Initialized VertexColorTracker with part colors
    
    Example:
        tracker = assign_part_colors(
            vertices, faces,
            part_vertices={
                'head': [0, 1, 2, 3, ...],
                'body': [100, 101, 102, ...],
                'left_paw': [200, 201, ...],
                'right_paw': [300, 301, ...],
            }
        )
    """
    tracker = VertexColorTracker(vertices, faces)
    
    # Convert index lists to boolean masks
    part_masks = {}
    n_verts = len(vertices) if isinstance(vertices, (list, np.ndarray)) else vertices.shape[0]
    
    for name, indices in part_vertices.items():
        mask = np.zeros(n_verts, dtype=bool)
        mask[indices] = True
        part_masks[name] = mask
    
    tracker.initialize_colors_manual(part_masks, part_colors)
    return tracker


def export_ply(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: Optional[np.ndarray] = None
):
    """
    Export mesh to PLY format with optional vertex colors.
    
    Args:
        filepath: Output file path
        vertices: Vertex positions, shape (V, 3)
        faces: Face indices, shape (F, 3)
        vertex_colors: RGB colors per vertex, shape (V, 3) in [0, 1]
    """
    n_vertices = len(vertices)
    n_faces = len(faces)
    has_colors = vertex_colors is not None
    
    with open(filepath, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Vertices
        for i in range(n_vertices):
            v = vertices[i]
            if has_colors:
                c = (vertex_colors[i] * 255).astype(int)
                f.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
            else:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        # Faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Exported PLY mesh to {filepath}")


def export_obj_with_vertex_colors(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray
):
    """
    Export mesh to OBJ format with vertex colors encoded in a per-vertex texture.
    
    Creates both .obj and .mtl files, plus a simple texture image.
    
    Args:
        filepath: Output file path (.obj)
        vertices: Vertex positions, shape (V, 3)
        faces: Face indices, shape (F, 3)
        vertex_colors: RGB colors per vertex, shape (V, 3) in [0, 1]
    """
    import os
    from PIL import Image
    
    base_path = os.path.splitext(filepath)[0]
    obj_path = base_path + '.obj'
    mtl_path = base_path + '.mtl'
    tex_path = base_path + '_colors.png'
    
    n_vertices = len(vertices)
    
    # Create a simple 1D texture where each pixel is a vertex's color
    # We use a 2D texture with width = num_vertices, height = 1
    tex_width = min(n_vertices, 4096)
    tex_height = (n_vertices + tex_width - 1) // tex_width
    
    texture = np.zeros((tex_height, tex_width, 3), dtype=np.uint8)
    for i, color in enumerate(vertex_colors):
        row = i // tex_width
        col = i % tex_width
        texture[row, col] = (color * 255).astype(np.uint8)
    
    Image.fromarray(texture).save(tex_path)
    
    # Write MTL file
    with open(mtl_path, 'w') as f:
        f.write("newmtl vertex_color_material\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd {os.path.basename(tex_path)}\n")
    
    # Write OBJ file
    with open(obj_path, 'w') as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        
        # Vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # UV coordinates (one per vertex, mapping to texture)
        for i in range(n_vertices):
            row = i // tex_width
            col = i % tex_width
            u = (col + 0.5) / tex_width
            v = 1.0 - (row + 0.5) / tex_height
            f.write(f"vt {u} {v}\n")
        
        f.write("usemtl vertex_color_material\n")
        
        # Faces with UV indices
        for face in faces:
            f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
    
    print(f"Exported OBJ mesh with vertex colors to {obj_path}")
