#!/usr/bin/env python3
"""
Clean up, decimate, and reorient SAM3D animal mesh outputs.

Pipeline per mesh:
  1. Repair: merge close vertices, drop degenerate/duplicate/unreferenced
     geometry, fix non-manifold edges/vertices.
  2. Drop small disconnected floating fragments (scan noise / floaters like
     the extra blob in a tiger's mouth or the fragmented bits in a horse
     tail) while keeping every component that is a meaningful part of the
     animal.
  3. Close remaining holes (scan gaps, e.g. holes in a face).
  4. Decimate to a 5000-6000 vertex target via iterative quadric edge
     collapse.
  5. Reorient: raw SAM3D output for these meshes comes out with X=width,
     Y=nose-to-tail length, Z=height. Common viewers/this repo's own
     camera convention (utilities/camera.py) expect Y=up. Rotating -90
     degrees about X (a proper rotation, determinant +1, so winding/normals
     stay correct) maps height->Y and length->Z, which fixes the
     "see the back/left ear from above" default-view problem.

Usage:
    python mesh_creator_for_meshup/scripts/process_sam3d_animal_meshes.py
"""

import argparse
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "mesh_creator_for_meshup/sam3D/mesh"
OUTPUT_DIR = REPO_ROOT / "mesh_creator_for_meshup/sam3D/processed_meshes"

TARGET_VERTS = 5500
MIN_COMPONENT_FRACTION = 0.03  # drop components smaller than 3% of the largest one

# Rotate -90 deg about X: (x, y, z) -> (x, z, -y)
REORIENT_R = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)

MESH_NAMES = [
    "elephant3.ply",
    "bear.ply",
    "bear2.ply",
    "cat2.ply",
    "Chihuahua2.ply",
    "fox.ply",
    "fox1.ply",
    "fox3.ply",
    "giraffe1.ply",
    "giraffe4.ply",
    "golden_retriever_img.ply",
    "goldie.ply",
    "horse2.ply",
    "horse3.ply",
    "panda2.ply",
    "panda3.ply",
    "pig.ply",
    "pug.ply",
    "tiger1.ply",
    "tiger2.ply",
]


def drop_small_components(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    n_verts = mesh.vertices.shape[0]
    edges = np.vstack([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row))
    adj = sparse.coo_matrix((data, (row, col)), shape=(n_verts, n_verts))

    n_components, labels = connected_components(adj, directed=False)
    if n_components == 1:
        return mesh, {"num_components": 1, "dropped_components": 0, "dropped_vertices": 0}

    sizes = np.bincount(labels)
    largest = sizes.max()
    keep_labels = np.where(sizes >= largest * MIN_COMPONENT_FRACTION)[0]
    keep_mask = np.isin(labels, keep_labels)

    kept_vert_idx = np.where(keep_mask)[0]
    old_to_new = np.full(n_verts, -1, dtype=np.int64)
    old_to_new[kept_vert_idx] = np.arange(kept_vert_idx.shape[0])

    face_keep = np.all(keep_mask[mesh.faces], axis=1)
    new_faces = old_to_new[mesh.faces[face_keep]]
    new_verts = mesh.vertices[keep_mask]

    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    stats = {
        "num_components": int(n_components),
        "dropped_components": int(n_components - len(keep_labels)),
        "dropped_vertices": int(n_verts - kept_vert_idx.shape[0]),
    }
    return new_mesh, stats


def repair_and_close_holes(ms: pymeshlab.MeshSet) -> None:
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.3))

    try:
        ms.meshing_repair_non_manifold_edges(method=0)
    except Exception as e:
        print(f"    warning: non-manifold edge repair failed: {e}")
    try:
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0.0)
    except Exception as e:
        print(f"    warning: non-manifold vertex repair failed: {e}")

    try:
        ms.meshing_close_holes(maxholesize=1000, newfaceselected=False)
    except Exception as e:
        print(f"    warning: hole closing failed: {e}")

    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()


def decimate_to_target(ms: pymeshlab.MeshSet, target_verts: int = TARGET_VERTS):
    target_faces = 11000
    best = None
    for _ in range(7):
        ms_local = pymeshlab.MeshSet()
        ms_local.add_mesh(ms.current_mesh())
        ms_local.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces),
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            qualitythr=0.5,
            planarquadric=True,
        )
        m = ms_local.current_mesh()
        v = int(m.vertex_number())
        f = int(m.face_number())
        err = abs(v - target_verts)
        if best is None or err < best[0]:
            best = (err, target_faces, v, f, ms_local)
        if err <= 150:
            break
        ratio = target_verts / max(v, 1)
        next_tf = int(target_faces * ratio)
        target_faces = max(4000, min(40000, next_tf if next_tf != target_faces else target_faces + (500 if v < target_verts else -500)))
    return best


def process_one(src_path: Path, out_dir: Path) -> dict:
    name = src_path.name
    print(f"Processing {name} ...")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src_path))
    v0 = ms.current_mesh().vertex_number()

    repair_and_close_holes(ms)

    tmp_repaired = out_dir / f"_tmp_{name}"
    ms.save_current_mesh(str(tmp_repaired))

    mesh = trimesh.load(str(tmp_repaired), process=False)
    mesh, comp_stats = drop_small_components(mesh)

    cleaned_path = out_dir / f"_tmp_clean_{name}"
    mesh.export(cleaned_path)

    ms2 = pymeshlab.MeshSet()
    ms2.load_new_mesh(str(cleaned_path))
    try:
        ms2.meshing_close_holes(maxholesize=1000, newfaceselected=False)
    except Exception as e:
        print(f"    warning: post-component hole closing failed: {e}")

    err, target_faces, v, f, decimated_ms = decimate_to_target(ms2)

    verts = decimated_ms.current_mesh().vertex_number()
    faces_arr = decimated_ms.current_mesh().face_matrix()
    verts_arr = decimated_ms.current_mesh().vertex_matrix()

    reoriented = verts_arr @ REORIENT_R.T

    out_path = out_dir / name
    out_mesh = trimesh.Trimesh(vertices=reoriented, faces=faces_arr, process=False)
    out_mesh.export(out_path)

    tmp_repaired.unlink(missing_ok=True)
    cleaned_path.unlink(missing_ok=True)

    stats = {
        "source": name,
        "verts_raw": v0,
        "verts_final": int(verts),
        "faces_final": int(f),
        "vert_target_err": int(err),
        **comp_stats,
    }
    print(
        f"  raw_verts={v0} -> final_verts={verts} faces={f} "
        f"(components dropped={comp_stats['dropped_components']}, "
        f"dropped_verts={comp_stats['dropped_vertices']})"
    )
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*", default=None, help="subset of filenames to process")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    names = args.names if args.names else MESH_NAMES
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for name in names:
        src = INPUT_DIR / name
        if not src.exists():
            print(f"SKIP missing: {src}")
            continue
        all_stats.append(process_one(src, out_dir))

    print("\nSummary:")
    for s in all_stats:
        print(s)


if __name__ == "__main__":
    main()
