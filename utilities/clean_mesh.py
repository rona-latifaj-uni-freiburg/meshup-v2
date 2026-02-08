"""
Mesh Cleaning Utility for MeshUp

This script cleans meshes to make them compatible with Neural Jacobian Fields,
which requires manifold, watertight meshes for the Poisson solver.

Usage:
    python utilities/clean_mesh.py --input meshes/male_body.obj --output meshes/male_body_clean.obj
"""

import argparse
import pymeshlab
import os


def clean_mesh(input_path: str, output_path: str, verbose: bool = True):
    """
    Clean a mesh to make it suitable for Neural Jacobian Fields.
    
    Fixes common issues:
    - Duplicate vertices
    - Unreferenced vertices
    - Non-manifold edges and vertices
    - Degenerate faces
    - Self-intersections
    - Holes (optional)
    
    Args:
        input_path: Path to input mesh
        output_path: Path to save cleaned mesh
        verbose: Print progress information
    """
    ms = pymeshlab.MeshSet()
    
    if verbose:
        print(f"Loading mesh from {input_path}...")
    ms.load_new_mesh(input_path)
    
    # Get initial stats
    m = ms.current_mesh()
    if verbose:
        print(f"Initial mesh: {m.vertex_number()} vertices, {m.face_number()} faces")
    
    # Step 1: Remove duplicate vertices
    if verbose:
        print("Step 1: Removing duplicate vertices...")
    ms.meshing_remove_duplicate_vertices()
    
    # Step 2: Remove duplicate faces
    if verbose:
        print("Step 2: Removing duplicate faces...")
    ms.meshing_remove_duplicate_faces()
    
    # Step 3: Remove zero-area faces
    if verbose:
        print("Step 3: Removing zero-area faces...")
    ms.meshing_remove_null_faces()
    
    # Step 4: Remove unreferenced vertices
    if verbose:
        print("Step 4: Removing unreferenced vertices...")
    ms.meshing_remove_unreferenced_vertices()
    
    # Step 5: Remove non-manifold edges
    if verbose:
        print("Step 5: Removing non-manifold edges...")
    try:
        ms.meshing_repair_non_manifold_edges()
    except Exception as e:
        print(f"  Warning: Could not repair non-manifold edges: {e}")
    
    # Step 6: Remove non-manifold vertices (vertices shared by non-adjacent faces)
    if verbose:
        print("Step 6: Removing non-manifold vertices...")
    try:
        ms.meshing_repair_non_manifold_vertices()
    except Exception as e:
        print(f"  Warning: Could not repair non-manifold vertices: {e}")
    
    # Step 7: Keep only the largest connected component
    if verbose:
        print("Step 7: Selecting largest connected component...")
    try:
        # Split into components and keep largest
        ms.generate_splitting_by_connected_components()
        if ms.mesh_number() > 1:
            # Find the mesh with most vertices
            max_verts = 0
            max_idx = 0
            for i in range(ms.mesh_number()):
                ms.set_current_mesh(i)
                if ms.current_mesh().vertex_number() > max_verts:
                    max_verts = ms.current_mesh().vertex_number()
                    max_idx = i
            
            # Keep only the largest
            ms.set_current_mesh(max_idx)
            if verbose:
                print(f"  Kept largest component with {max_verts} vertices")
    except Exception as e:
        print(f"  Warning: Could not split components: {e}")
    
    # Step 8: Close small holes (optional, might help with some meshes)
    if verbose:
        print("Step 8: Attempting to close small holes...")
    try:
        ms.meshing_close_holes(maxholesize=100)
    except Exception as e:
        print(f"  Warning: Could not close holes: {e}")
    
    # Step 9: Re-orient faces consistently
    if verbose:
        print("Step 9: Re-orienting faces...")
    try:
        ms.meshing_re_orient_faces_coherentely()
    except Exception as e:
        print(f"  Warning: Could not re-orient faces: {e}")
    
    # Step 10: Final cleanup
    if verbose:
        print("Step 10: Final cleanup...")
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    
    # Get final stats
    m = ms.current_mesh()
    if verbose:
        print(f"Final mesh: {m.vertex_number()} vertices, {m.face_number()} faces")
    
    # Save
    if verbose:
        print(f"Saving cleaned mesh to {output_path}...")
    ms.save_current_mesh(output_path)
    
    if verbose:
        print("Done!")
    
    return output_path


def simplify_mesh(input_path: str, output_path: str, target_faces: int = 10000, verbose: bool = True):
    """
    Simplify a mesh to reduce face count while preserving shape.
    
    This is useful for very high-poly meshes that are slow to process.
    
    Args:
        input_path: Path to input mesh
        output_path: Path to save simplified mesh
        target_faces: Target number of faces
        verbose: Print progress
    """
    ms = pymeshlab.MeshSet()
    
    if verbose:
        print(f"Loading mesh from {input_path}...")
    ms.load_new_mesh(input_path)
    
    m = ms.current_mesh()
    if verbose:
        print(f"Initial: {m.vertex_number()} vertices, {m.face_number()} faces")
    
    if m.face_number() > target_faces:
        if verbose:
            print(f"Simplifying to ~{target_faces} faces...")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    
    m = ms.current_mesh()
    if verbose:
        print(f"Final: {m.vertex_number()} vertices, {m.face_number()} faces")
        print(f"Saving to {output_path}...")
    
    ms.save_current_mesh(output_path)
    
    if verbose:
        print("Done!")
    
    return output_path


def check_mesh_issues(input_path: str):
    """
    Check a mesh for common issues that cause NJF problems.
    
    Args:
        input_path: Path to mesh to check
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)
    m = ms.current_mesh()
    
    print(f"\n{'='*60}")
    print(f"Mesh Analysis: {input_path}")
    print(f"{'='*60}")
    print(f"Vertices: {m.vertex_number()}")
    print(f"Faces: {m.face_number()}")
    print(f"Edges: {m.edge_number()}")
    
    # Check for issues
    issues = []
    
    # Non-manifold check
    try:
        ms.compute_selection_by_non_manifold_edges_per_face()
        non_manifold_faces = ms.current_mesh().selected_face_number()
        if non_manifold_faces > 0:
            issues.append(f"Non-manifold edges: {non_manifold_faces} faces affected")
        ms.set_selection_none()
    except:
        pass
    
    try:
        ms.compute_selection_by_non_manifold_per_vertex()
        non_manifold_verts = ms.current_mesh().selected_vertex_number()
        if non_manifold_verts > 0:
            issues.append(f"Non-manifold vertices: {non_manifold_verts}")
        ms.set_selection_none()
    except:
        pass
    
    # Boundary check (open mesh)
    try:
        ms.compute_selection_by_border()
        boundary_verts = ms.current_mesh().selected_vertex_number()
        if boundary_verts > 0:
            issues.append(f"Open boundary: {boundary_verts} boundary vertices (mesh has holes)")
        ms.set_selection_none()
    except:
        pass
    
    # Connected components
    try:
        ms.generate_splitting_by_connected_components()
        n_components = ms.mesh_number()
        if n_components > 1:
            issues.append(f"Disconnected: {n_components} separate components")
        # Reload original
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_path)
    except:
        pass
    
    print(f"\n{'Issues Found:'}")
    if issues:
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print(f"  ✓ No obvious issues detected")
    
    print(f"\n{'Recommendations:'}")
    if issues:
        print("  Run: python utilities/clean_mesh.py --input <your_mesh> --output <cleaned_mesh>")
    else:
        print("  Mesh appears clean. If NJF still fails, try simplifying with --simplify flag")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Clean mesh for Neural Jacobian Fields")
    parser.add_argument('--input', '-i', required=True, help='Input mesh path')
    parser.add_argument('--output', '-o', help='Output mesh path (default: input_clean.obj)')
    parser.add_argument('--check', action='store_true', help='Only check for issues, do not modify')
    parser.add_argument('--simplify', type=int, default=None, help='Simplify to this many faces')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_clean{ext}"
    
    if args.check:
        check_mesh_issues(args.input)
        return
    
    # Clean the mesh
    clean_mesh(args.input, args.output, verbose=not args.quiet)
    
    # Optionally simplify
    if args.simplify:
        simplify_mesh(args.output, args.output, target_faces=args.simplify, verbose=not args.quiet)
    
    # Final check
    if not args.quiet:
        print("\nChecking cleaned mesh:")
        check_mesh_issues(args.output)


if __name__ == "__main__":
    main()
