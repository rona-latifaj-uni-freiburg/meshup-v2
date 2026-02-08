"""
Example script demonstrating semantic tracking in MeshUp.

This script shows how to:
1. Initialize vertex colors for correspondence tracking
2. Run deformation with tracking enabled
3. Visualize and analyze correspondence results
"""

import os
import sys
import torch
import numpy as np

# Add meshup to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_tracking.vertex_color_tracking import (
    VertexColorTracker,
    initialize_semantic_colors,
    assign_part_colors,
)
from semantic_tracking.correspondence_export import (
    export_mesh_with_colors,
    export_correspondence_map,
    load_correspondence_map,
    visualize_correspondence_displacement,
)


def demo_basic_tracking():
    """
    Demonstrate basic vertex color tracking on a simple mesh.
    """
    print("=" * 60)
    print("Demo: Basic Vertex Color Tracking")
    print("=" * 60)
    
    # Create a simple cube mesh for demonstration
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],  # Front face
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Back
        [4, 5, 6], [4, 6, 7],  # Front
        [0, 1, 5], [0, 5, 4],  # Bottom
        [2, 3, 7], [2, 7, 6],  # Top
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 2, 6], [1, 6, 5],  # Right
    ], dtype=np.int32)
    
    # Initialize tracker with position-based colors
    print("\n1. Creating tracker with position-based colors...")
    tracker = initialize_semantic_colors(vertices, faces, method='position')
    print(f"   - Vertex colors shape: {tracker.vertex_colors.shape}")
    print(f"   - Sample colors (first 3 vertices):")
    for i in range(3):
        c = tracker.vertex_colors[i]
        print(f"     Vertex {i}: RGB({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
    
    # Simulate deformation (just shift vertices)
    print("\n2. Simulating deformation...")
    deformed_vertices = vertices.copy()
    deformed_vertices[:, 1] += 0.5  # Shift up
    deformed_vertices[4:, 0] *= 1.2  # Stretch front face
    
    # Export both meshes
    output_dir = "./outputs/tracking_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n3. Exporting meshes to {output_dir}...")
    
    # Original mesh with colors
    tracker.export_ply(
        f"{output_dir}/original_colored.ply",
        vertices=vertices
    )
    
    # Deformed mesh with SAME colors (shows correspondence)
    tracker.export_ply(
        f"{output_dir}/deformed_colored.ply",
        vertices=deformed_vertices
    )
    
    # Displacement visualization
    visualize_correspondence_displacement(
        vertices,
        deformed_vertices,
        faces,
        f"{output_dir}/displacement_viz.ply"
    )
    
    # Export correspondence map
    export_correspondence_map(
        f"{output_dir}/correspondence.json",
        vertices,
        deformed_vertices,
        faces,
        colors=tracker.vertex_colors
    )
    
    print("\n4. Loading and analyzing correspondence map...")
    corr = load_correspondence_map(f"{output_dir}/correspondence.json")
    print(f"   - Mean displacement: {corr['statistics']['mean_displacement']:.4f}")
    print(f"   - Max displacement: {corr['statistics']['max_displacement']:.4f}")
    
    print("\nDemo complete! Check the outputs in:", output_dir)


def demo_part_based_tracking():
    """
    Demonstrate part-based vertex color tracking.
    """
    print("\n" + "=" * 60)
    print("Demo: Part-Based Color Tracking")
    print("=" * 60)
    
    # Create a simple animal-like shape (simplified)
    # This is just for demonstration - in practice you'd load a real mesh
    
    # Let's create a simple mesh with identifiable parts
    np.random.seed(42)
    n_vertices = 100
    
    # Generate vertices in clusters representing body parts
    head_verts = np.random.randn(20, 3) * 0.3 + [0, 1, 0]  # Head at top
    body_verts = np.random.randn(40, 3) * 0.5 + [0, 0, 0]  # Body in center
    left_leg_verts = np.random.randn(20, 3) * 0.2 + [-0.5, -0.8, 0]
    right_leg_verts = np.random.randn(20, 3) * 0.2 + [0.5, -0.8, 0]
    
    vertices = np.vstack([head_verts, body_verts, left_leg_verts, right_leg_verts]).astype(np.float32)
    
    # Simple triangulation (in practice, use proper mesh)
    from scipy.spatial import Delaunay
    tri = Delaunay(vertices[:, :2])  # 2D triangulation projected
    faces = tri.simplices.astype(np.int32)
    
    # Define part indices
    part_vertices = {
        'head': list(range(0, 20)),
        'body': list(range(20, 60)),
        'left_leg': list(range(60, 80)),
        'right_leg': list(range(80, 100)),
    }
    
    # Define colors for each part
    part_colors = {
        'head': (1.0, 0.8, 0.0),      # Yellow
        'body': (0.0, 0.8, 0.2),      # Green
        'left_leg': (0.2, 0.4, 1.0),  # Blue
        'right_leg': (1.0, 0.2, 0.4), # Red
    }
    
    print("\n1. Creating tracker with manual part labels...")
    tracker = assign_part_colors(vertices, faces, part_vertices, part_colors)
    
    print("   Part colors assigned:")
    for name, color in part_colors.items():
        print(f"     {name}: RGB{color}")
    
    # Simulate transformation (like deforming a cat into a dog)
    print("\n2. Simulating transformation...")
    deformed_vertices = vertices.copy()
    
    # Elongate the head (like changing species)
    deformed_vertices[part_vertices['head'], 1] *= 1.3
    
    # Change leg proportions
    deformed_vertices[part_vertices['left_leg'], 1] *= 0.8
    deformed_vertices[part_vertices['right_leg'], 1] *= 0.8
    
    # Export
    output_dir = "./outputs/part_tracking_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n3. Exporting to {output_dir}...")
    
    tracker.export_ply(f"{output_dir}/original_parts.ply", vertices=vertices)
    tracker.export_ply(f"{output_dir}/deformed_parts.ply", vertices=deformed_vertices)
    
    # The key insight: same colors on deformed mesh show correspondence
    print("\n4. Correspondence interpretation:")
    print("   - Yellow vertices in deformed mesh = original head")
    print("   - Green vertices = original body")
    print("   - Blue vertices = original left leg")
    print("   - Red vertices = original right leg")
    print("\n   By viewing both meshes, you can see exactly how each")
    print("   semantic part transformed during deformation!")
    
    print("\nDemo complete! Check outputs in:", output_dir)


def demo_clustering_based_tracking():
    """
    Demonstrate automatic clustering for part discovery.
    """
    print("\n" + "=" * 60)
    print("Demo: Clustering-Based Part Discovery")
    print("=" * 60)
    
    # Create a mesh with natural clusters
    np.random.seed(42)
    
    # Generate clustered vertices
    cluster1 = np.random.randn(30, 3) * 0.3 + [0, 1, 0]
    cluster2 = np.random.randn(30, 3) * 0.4 + [0, 0, 0]
    cluster3 = np.random.randn(30, 3) * 0.2 + [-0.7, -0.5, 0]
    cluster4 = np.random.randn(30, 3) * 0.2 + [0.7, -0.5, 0]
    
    vertices = np.vstack([cluster1, cluster2, cluster3, cluster4]).astype(np.float32)
    
    from scipy.spatial import Delaunay
    tri = Delaunay(vertices[:, :2])
    faces = tri.simplices.astype(np.int32)
    
    # Create tracker with automatic clustering
    print("\n1. Running K-means clustering on vertices...")
    tracker = VertexColorTracker(vertices, faces)
    colors, labels = tracker.initialize_colors_by_clustering(n_parts=4)
    
    print(f"   Found {len(np.unique(labels))} clusters")
    for i in range(4):
        count = np.sum(labels == i)
        print(f"   Cluster {i}: {count} vertices")
    
    output_dir = "./outputs/clustering_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    tracker.export_ply(f"{output_dir}/clustered_mesh.ply")
    
    print(f"\n2. Exported clustered mesh to {output_dir}")
    print("   Each cluster has a distinct color for tracking.")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("MeshUp Semantic Tracking Demos")
    print("=" * 60)
    
    try:
        demo_basic_tracking()
    except Exception as e:
        print(f"Basic tracking demo failed: {e}")
    
    try:
        demo_part_based_tracking()
    except ImportError as e:
        print(f"Part tracking demo requires scipy: {e}")
    except Exception as e:
        print(f"Part tracking demo failed: {e}")
    
    try:
        demo_clustering_based_tracking()
    except ImportError as e:
        print(f"Clustering demo requires scipy and sklearn: {e}")
    except Exception as e:
        print(f"Clustering demo failed: {e}")
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
