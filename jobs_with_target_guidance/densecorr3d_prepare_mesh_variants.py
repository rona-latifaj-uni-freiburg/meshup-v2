"""Prepare DenseCorr3D semantic-bucket mesh variants for MeshUp runs.

DenseCorr3D provides supervised semantic correspondence groups on each
object's ``simple_mesh.obj``. This script transfers those groups to the
higher-resolution ``color_mesh.obj`` and, optionally, to a decimated mesh with
approximately a requested vertex count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jobs_with_target_guidance.densecorr3d_segment import (  # noqa: E402
    IndexBase,
    load_densecorr3d_groups,
    resolve_object,
    select_objects,
)
from jobs_with_target_guidance.partfield_segment import (  # noqa: E402
    labels_to_face_labels,
    load_obj_mesh,
    write_face_colored_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create full-resolution and decimated DenseCorr3D-labeled mesh variants.",
    )
    parser.add_argument("--densecorr3d-root", type=Path, required=True)
    parser.add_argument("--category", type=str, default="animals")
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("jobs_with_target_guidance/densematcher_runs/prepared"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("full", "5k"),
        default=("full", "5k"),
        help="Which mesh variants to write.",
    )
    parser.add_argument("--target-vertices", type=int, default=5000)
    parser.add_argument(
        "--index-base",
        choices=("auto", "zero", "one"),
        default="auto",
        help="Whether groups.txt vertex ids are zero- or one-indexed.",
    )
    parser.add_argument(
        "--raw-coordinates",
        action="store_true",
        help="Transfer labels in raw coordinates instead of per-object bbox-normalized coordinates.",
    )
    return parser.parse_args()


def bbox_coordinates(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    return (vertices - vmin) / np.maximum(vmax - vmin, 1e-8)


def maybe_normalize(vertices: np.ndarray, normalize: bool) -> np.ndarray:
    if not normalize:
        return vertices.astype(np.float32)
    return bbox_coordinates(vertices).astype(np.float32)


def nearest_indices(source_points: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(source_points)
        return nn.kneighbors(query_points, return_distance=False).reshape(-1)
    except Exception:
        nearest = np.empty(query_points.shape[0], dtype=np.int64)
        chunk_size = 2048
        for start in range(0, query_points.shape[0], chunk_size):
            chunk = query_points[start : start + chunk_size]
            delta = chunk[:, None, :] - source_points[None, :, :]
            dist2 = np.einsum("ijk,ijk->ij", delta, delta)
            nearest[start : start + chunk.shape[0]] = np.argmin(dist2, axis=1)
        return nearest


def transfer_vertex_labels(
    source_vertices: np.ndarray,
    source_labels: np.ndarray,
    target_vertices: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    source_points = maybe_normalize(source_vertices, normalize)
    target_points = maybe_normalize(target_vertices, normalize)
    return source_labels[nearest_indices(source_points, target_points)].astype(np.int64)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# DenseCorr3D prepared geometry\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
            f.write(f"f {a} {b} {c}\n")


def save_labels(
    labels_path: Path,
    vertex_labels: np.ndarray,
    faces: np.ndarray,
    n_buckets: int,
) -> np.ndarray:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    face_labels = labels_to_face_labels(vertex_labels, faces)
    np.savez(
        labels_path,
        labels=vertex_labels,
        vertex_labels=vertex_labels,
        face_labels=face_labels,
        n_buckets=np.asarray(n_buckets, dtype=np.int64),
    )
    return face_labels


def label_stats(labels: np.ndarray, n_buckets: int) -> Dict[str, int]:
    return {
        f"group_{bucket_id:02d}": int((labels == bucket_id).sum())
        for bucket_id in range(n_buckets)
    }


def decimate_to_target_vertices(
    input_obj: Path,
    output_obj: Path,
    target_vertices: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    vertices, faces = load_obj_mesh(input_obj)
    original_vertices = int(vertices.shape[0])
    original_faces = int(faces.shape[0])

    if original_vertices <= target_vertices:
        write_obj(output_obj, vertices, faces)
        return vertices, faces, {
            "original_vertices": original_vertices,
            "original_faces": original_faces,
            "target_vertices": int(target_vertices),
            "actual_vertices": original_vertices,
            "actual_faces": original_faces,
            "decimated": 0,
        }

    try:
        import pymeshlab
    except ImportError as exc:
        raise RuntimeError(
            "pymeshlab is required for --variants 5k. Run this inside the meshup_new environment."
        ) from exc

    target_faces = max(4, int(round(original_faces * (float(target_vertices) / float(original_vertices)))))
    tmp_obj = output_obj.with_suffix(".pymeshlab_tmp.obj")
    if tmp_obj.exists():
        tmp_obj.unlink()

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_obj))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        qualitythr=0.5,
        planarquadric=True,
    )
    ms.save_current_mesh(str(tmp_obj))

    dec_vertices, dec_faces = load_obj_mesh(tmp_obj)
    write_obj(output_obj, dec_vertices, dec_faces)
    tmp_obj.unlink(missing_ok=True)
    return dec_vertices, dec_faces, {
        "original_vertices": original_vertices,
        "original_faces": original_faces,
        "target_vertices": int(target_vertices),
        "target_faces": int(target_faces),
        "actual_vertices": int(dec_vertices.shape[0]),
        "actual_faces": int(dec_faces.shape[0]),
        "decimated": 1,
    }


def process_object(
    object_dir: Path,
    output_dir: Path,
    variants: Sequence[str],
    target_vertices: int,
    index_base: IndexBase,
    normalize_transfer: bool,
) -> Dict[str, object]:
    name = object_dir.name
    simple_mesh = object_dir / "simple_mesh.obj"
    color_mesh = object_dir / "color_mesh.obj"
    groups_path = object_dir / "groups.txt"
    if not color_mesh.is_file():
        raise FileNotFoundError(f"Missing DenseCorr3D color mesh: {color_mesh}")

    simple_vertices, _simple_faces = load_obj_mesh(simple_mesh)
    simple_labels, group_summary = load_densecorr3d_groups(
        groups_path=groups_path,
        n_vertices=simple_vertices.shape[0],
        vertices=simple_vertices,
        index_base=index_base,
        fill_mode="nearest",
    )
    n_buckets = int(group_summary["n_groups"])

    mesh_dir = output_dir / "meshes"
    labels_dir = output_dir / "labels"
    colored_dir = output_dir / "colored"
    entries: List[Dict[str, object]] = []

    full_obj = mesh_dir / f"{name}_densecorr3d_full.obj"
    full_labels_path = labels_dir / f"{name}_densecorr3d_full_labels.npz"
    color_vertices, color_faces = load_obj_mesh(color_mesh)
    full_vertex_labels = transfer_vertex_labels(
        source_vertices=simple_vertices,
        source_labels=simple_labels,
        target_vertices=color_vertices,
        normalize=normalize_transfer,
    )

    if "full" in variants or "5k" in variants:
        write_obj(full_obj, color_vertices, color_faces)
        full_face_labels = save_labels(
            full_labels_path,
            full_vertex_labels,
            color_faces,
            n_buckets=n_buckets,
        )
        full_colored = colored_dir / f"{name}_densecorr3d_full_groups.ply"
        write_face_colored_ply(full_colored, color_vertices, color_faces, full_face_labels)
        entries.append(
            {
                "variant": "full",
                "mesh": str(full_obj),
                "labels": str(full_labels_path),
                "colored_mesh": str(full_colored),
                "vertices": int(color_vertices.shape[0]),
                "faces": int(color_faces.shape[0]),
                "n_buckets": n_buckets,
                "bucket_vertex_counts": label_stats(full_vertex_labels, n_buckets),
            }
        )

    if "5k" in variants:
        mesh_5k = mesh_dir / f"{name}_densecorr3d_5k.obj"
        labels_5k_path = labels_dir / f"{name}_densecorr3d_5k_labels.npz"
        vertices_5k, faces_5k, decimation_summary = decimate_to_target_vertices(
            full_obj,
            mesh_5k,
            target_vertices=target_vertices,
        )
        labels_5k = transfer_vertex_labels(
            source_vertices=color_vertices,
            source_labels=full_vertex_labels,
            target_vertices=vertices_5k,
            normalize=normalize_transfer,
        )
        face_labels_5k = save_labels(labels_5k_path, labels_5k, faces_5k, n_buckets=n_buckets)
        colored_5k = colored_dir / f"{name}_densecorr3d_5k_groups.ply"
        write_face_colored_ply(colored_5k, vertices_5k, faces_5k, face_labels_5k)
        entries.append(
            {
                "variant": "5k",
                "mesh": str(mesh_5k),
                "labels": str(labels_5k_path),
                "colored_mesh": str(colored_5k),
                "vertices": int(vertices_5k.shape[0]),
                "faces": int(faces_5k.shape[0]),
                "n_buckets": n_buckets,
                "decimation": decimation_summary,
                "bucket_vertex_counts": label_stats(labels_5k, n_buckets),
            }
        )

    return {
        "name": name,
        "object_dir": str(object_dir),
        "simple_mesh": str(simple_mesh),
        "color_mesh": str(color_mesh),
        "groups": str(groups_path),
        "simple_vertices": int(simple_vertices.shape[0]),
        "color_vertices": int(color_vertices.shape[0]),
        "n_buckets": n_buckets,
        "group_summary": group_summary,
        "variants": entries,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_objects = select_objects(args.densecorr3d_root, args.category, None, 0)
    selected = [
        resolve_object(args.densecorr3d_root, args.category, token, all_objects)
        for token in args.objects
    ]

    entries = [
        process_object(
            object_dir=object_dir,
            output_dir=args.output_dir,
            variants=args.variants,
            target_vertices=args.target_vertices,
            index_base=args.index_base,
            normalize_transfer=not args.raw_coordinates,
        )
        for object_dir in selected
    ]
    summary = {
        "source": "DenseCorr3D",
        "category": args.category,
        "target_vertices": int(args.target_vertices),
        "normalize_transfer": not args.raw_coordinates,
        "objects": entries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote prepared DenseCorr3D mesh variants for {len(entries)} objects to {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
