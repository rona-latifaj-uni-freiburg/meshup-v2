#!/usr/bin/env python3
"""Evaluate target-guided mesh deformation outputs.

This script is intentionally independent from the training loop. It compares a
deformed MeshUp output against its target mesh, and optionally checks source
distortion and PartField part-wise geometry when aligned labels are available.
All geometry is normalized with MeshUp's unit-box convention before metrics are
computed, so values are comparable across shapes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial import cKDTree

try:
    import pymeshlab
except ImportError as exc:  # pragma: no cover - exercised in real envs.
    raise SystemExit("pymeshlab is required. Activate the MeshUp environment first.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs_with_target_guidance.partfield_chamfer import load_label_array


ArrayPair = Tuple[np.ndarray, np.ndarray]


def load_mesh(path: str | Path) -> ArrayPair:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    mesh = ms.current_mesh()
    vertices = np.asarray(mesh.vertex_matrix(), dtype=np.float64)
    faces = np.asarray(mesh.face_matrix(), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{path} did not load as an Nx3 vertex array.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path} must contain triangular faces, got {faces.shape}.")
    return vertices, faces


def unit_vertices(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    scale = 2.0 / max(float((vmax - vmin).max()), 1e-12)
    return (vertices - (vmax + vmin) * 0.5) * scale


def face_areas_and_normals(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    areas = 0.5 * lengths
    normals = cross / np.maximum(lengths[:, None], 1e-12)
    return areas, normals


def sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
    face_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if face_mask is not None:
        faces = faces[np.asarray(face_mask, dtype=bool)]
    if faces.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    areas, face_normals = face_areas_and_normals(vertices, faces)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Mesh has no positive-area faces to sample.")

    probs = areas / areas.sum()
    face_idx = rng.choice(faces.shape[0], size=int(n_points), replace=True, p=probs)
    tri = vertices[faces[face_idx]]
    u = rng.random(int(n_points))
    v = rng.random(int(n_points))
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
    normals = face_normals[face_idx]
    return points, normals


def nearest_distances(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(tgt)
    dists, indices = tree.query(src, workers=-1)
    return dists.astype(np.float64), indices.astype(np.int64)


def geometry_metrics(
    src_points: np.ndarray,
    src_normals: np.ndarray,
    tgt_points: np.ndarray,
    tgt_normals: np.ndarray,
    thresholds: Iterable[float],
) -> Dict[str, float]:
    if src_points.size == 0 or tgt_points.size == 0:
        return {}

    src_to_tgt, src_nn = nearest_distances(src_points, tgt_points)
    tgt_to_src, tgt_nn = nearest_distances(tgt_points, src_points)

    metrics: Dict[str, float] = {
        "chamfer_l2": float(src_to_tgt.mean() + tgt_to_src.mean()),
        "chamfer_l2_sq": float(np.square(src_to_tgt).mean() + np.square(tgt_to_src).mean()),
        "hausdorff_l2": float(max(src_to_tgt.max(initial=0.0), tgt_to_src.max(initial=0.0))),
        "src_to_tgt_l2": float(src_to_tgt.mean()),
        "tgt_to_src_l2": float(tgt_to_src.mean()),
    }

    if src_normals.size and tgt_normals.size:
        src_dot = np.abs((src_normals * tgt_normals[src_nn]).sum(axis=1))
        tgt_dot = np.abs((tgt_normals * src_normals[tgt_nn]).sum(axis=1))
        metrics["normal_consistency"] = float(0.5 * (src_dot.mean() + tgt_dot.mean()))

    for threshold in thresholds:
        precision = float((src_to_tgt <= threshold).mean())
        recall = float((tgt_to_src <= threshold).mean())
        denom = precision + recall
        fscore = 0.0 if denom <= 0 else 2.0 * precision * recall / denom
        key = f"{threshold:g}".replace(".", "p")
        metrics[f"precision_tau_{key}"] = precision
        metrics[f"recall_tau_{key}"] = recall
        metrics[f"fscore_tau_{key}"] = float(fscore)

    return metrics


def unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def source_distortion_metrics(
    source_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
    faces: np.ndarray,
) -> Dict[str, float]:
    if source_vertices.shape != deformed_vertices.shape:
        return {
            "source_distortion_available": 0.0,
            "source_distortion_reason": "source/deformed vertex counts differ",
        }

    edges = unique_edges(faces)
    src_len = np.linalg.norm(source_vertices[edges[:, 0]] - source_vertices[edges[:, 1]], axis=1)
    def_len = np.linalg.norm(deformed_vertices[edges[:, 0]] - deformed_vertices[edges[:, 1]], axis=1)
    valid = src_len > 1e-12
    rel = np.abs(def_len[valid] / src_len[valid] - 1.0)

    src_lap = uniform_laplacian_coordinates(source_vertices, faces)
    def_lap = uniform_laplacian_coordinates(deformed_vertices, faces)
    lap_delta = np.linalg.norm(def_lap - src_lap, axis=1)

    return {
        "source_distortion_available": 1.0,
        "edge_length_relative_mean": float(rel.mean()) if rel.size else 0.0,
        "edge_length_relative_p95": float(np.percentile(rel, 95)) if rel.size else 0.0,
        "edge_length_relative_max": float(rel.max(initial=0.0)) if rel.size else 0.0,
        "laplacian_delta_mean": float(lap_delta.mean()),
        "laplacian_delta_p95": float(np.percentile(lap_delta, 95)),
    }


def uniform_laplacian_coordinates(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    neighbors: List[set[int]] = [set() for _ in range(vertices.shape[0])]
    for a, b, c in faces:
        neighbors[int(a)].update((int(b), int(c)))
        neighbors[int(b)].update((int(a), int(c)))
        neighbors[int(c)].update((int(a), int(b)))

    lap = np.zeros_like(vertices)
    for idx, nbrs in enumerate(neighbors):
        if nbrs:
            nbr_idx = np.fromiter(nbrs, dtype=np.int64)
            lap[idx] = vertices[idx] - vertices[nbr_idx].mean(axis=0)
    return lap


def labels_to_vertex_labels(labels_path: str | Path, n_vertices: int, faces: np.ndarray, mode: str) -> np.ndarray:
    labels = load_label_array(str(labels_path))
    n_faces = faces.shape[0]
    if mode == "vertex" or (mode == "auto" and labels.shape[0] == n_vertices):
        if labels.shape[0] != n_vertices:
            raise ValueError(f"{labels_path}: expected {n_vertices} vertex labels, got {labels.shape[0]}.")
        return labels.astype(np.int64)
    if mode == "face" or (mode == "auto" and labels.shape[0] == n_faces):
        if labels.shape[0] != n_faces:
            raise ValueError(f"{labels_path}: expected {n_faces} face labels, got {labels.shape[0]}.")
        return face_labels_to_vertex_labels(labels, faces, n_vertices)
    raise ValueError(
        f"{labels_path}: cannot infer label mode from {labels.shape[0]} labels; "
        f"mesh has {n_vertices} vertices and {n_faces} faces."
    )


def face_labels_to_vertex_labels(face_labels: np.ndarray, faces: np.ndarray, n_vertices: int) -> np.ndarray:
    votes: List[Dict[int, int]] = [dict() for _ in range(n_vertices)]
    for face_idx, face in enumerate(faces):
        label = int(face_labels[face_idx])
        for vertex_id in face:
            bucket = votes[int(vertex_id)]
            bucket[label] = bucket.get(label, 0) + 1
    labels = np.zeros(n_vertices, dtype=np.int64)
    for vertex_id, bucket in enumerate(votes):
        if bucket:
            labels[vertex_id] = max(bucket.items(), key=lambda item: item[1])[0]
    return labels


def vertex_labels_to_face_labels(vertex_labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_labels = np.empty(faces.shape[0], dtype=np.int64)
    for face_idx, face in enumerate(faces):
        values, counts = np.unique(vertex_labels[face], return_counts=True)
        face_labels[face_idx] = int(values[np.argmax(counts)])
    return face_labels


def partfield_metrics(
    deformed_vertices: np.ndarray,
    deformed_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    source_labels_path: str,
    target_labels_path: str,
    label_mode: str,
    points_per_part: int,
    min_faces_per_part: int,
    thresholds: Iterable[float],
    seed: int,
) -> Dict[str, object]:
    src_vertex_labels = labels_to_vertex_labels(
        source_labels_path,
        deformed_vertices.shape[0],
        deformed_faces,
        label_mode,
    )
    tgt_vertex_labels = labels_to_vertex_labels(
        target_labels_path,
        target_vertices.shape[0],
        target_faces,
        label_mode,
    )
    src_face_labels = vertex_labels_to_face_labels(src_vertex_labels, deformed_faces)
    tgt_face_labels = vertex_labels_to_face_labels(tgt_vertex_labels, target_faces)

    rng = np.random.default_rng(seed)
    part_rows: List[Dict[str, float | int | str]] = []
    active_chamfers = []
    active_fscores = []
    labels = sorted(set(src_face_labels.tolist()) | set(tgt_face_labels.tolist()))
    primary_threshold = list(thresholds)[0]
    primary_key = f"fscore_tau_{str(primary_threshold).replace('.', 'p')}"

    for label in labels:
        src_mask = src_face_labels == label
        tgt_mask = tgt_face_labels == label
        row: Dict[str, float | int | str] = {
            "bucket": int(label),
            "source_faces": int(src_mask.sum()),
            "target_faces": int(tgt_mask.sum()),
            "active": int(src_mask.sum() >= min_faces_per_part and tgt_mask.sum() >= min_faces_per_part),
        }
        if row["active"]:
            src_points, src_normals = sample_surface(
                deformed_vertices,
                deformed_faces,
                points_per_part,
                rng,
                face_mask=src_mask,
            )
            tgt_points, tgt_normals = sample_surface(
                target_vertices,
                target_faces,
                points_per_part,
                rng,
                face_mask=tgt_mask,
            )
            metrics = geometry_metrics(src_points, src_normals, tgt_points, tgt_normals, thresholds)
            row.update(metrics)
            active_chamfers.append(float(metrics["chamfer_l2_sq"]))
            active_fscores.append(float(metrics.get(primary_key, 0.0)))
        part_rows.append(row)

    summary = {
        "active_parts": int(sum(int(row["active"]) for row in part_rows)),
        "total_parts": int(len(part_rows)),
        "mean_part_chamfer_l2_sq": float(np.mean(active_chamfers)) if active_chamfers else None,
        f"mean_part_{primary_key}": float(np.mean(active_fscores)) if active_fscores else None,
        "parts": part_rows,
    }
    return summary


def resolve_from_output_dir(output_dir: Optional[str]) -> Dict[str, Optional[str]]:
    if output_dir is None:
        return {}
    out = Path(output_dir)
    config_path = out / "config.yml"
    values: Dict[str, Optional[str]] = {
        "deformed_mesh": str(out / "mesh_final" / "mesh.obj"),
        "source_mesh": None,
        "target_mesh": None,
        "partfield_source_labels": None,
        "partfield_target_labels": None,
    }
    if config_path.exists():
        with config_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        source_labels = cfg.get("partfield_source_labels")
        target_labels = cfg.get("partfield_target_labels")
        if cfg.get("partfield_multiscale_enabled"):
            source_label_list = cfg_list(cfg.get("partfield_multiscale_source_labels"))
            target_label_list = cfg_list(cfg.get("partfield_multiscale_target_labels"))
            label_dirs = cfg_list(cfg.get("partfield_multiscale_label_dirs"))
            if source_label_list and target_label_list:
                source_labels = source_label_list[-1]
                target_labels = target_label_list[-1]
            elif label_dirs:
                source_labels = label_path_from_dir(label_dirs[-1], source_labels)
                target_labels = label_path_from_dir(label_dirs[-1], target_labels)
        values.update(
            {
                "source_mesh": cfg.get("mesh"),
                "target_mesh": cfg.get("target_mesh"),
                "partfield_source_labels": source_labels,
                "partfield_target_labels": target_labels,
            }
        )
    return values


def cfg_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.replace(",", " ").split() if item]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def label_path_from_dir(label_dir: str, template_path: Optional[str]) -> Optional[str]:
    if not label_dir or not template_path:
        return template_path
    path = Path(label_dir)
    if path.name != "labels":
        path = path / "labels"
    return str(path / Path(template_path).name)


def write_part_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="MeshUp output directory containing config.yml and mesh_final/mesh.obj.")
    parser.add_argument("--deformed-mesh", help="Deformed/final mesh path. Overrides --output-dir inference.")
    parser.add_argument("--source-mesh", help="Initial source mesh path. Used for distortion metrics.")
    parser.add_argument("--target-mesh", help="Target mesh path.")
    parser.add_argument("--partfield-source-labels", help="Aligned source PartField labels for the source/deformed topology.")
    parser.add_argument("--partfield-target-labels", help="Aligned target PartField labels.")
    parser.add_argument("--partfield-label-mode", choices=["auto", "vertex", "face"], default="auto")
    parser.add_argument("--samples", type=int, default=20000, help="Surface samples for global target metrics.")
    parser.add_argument("--part-samples", type=int, default=4000, help="Surface samples per active part.")
    parser.add_argument("--min-part-faces", type=int, default=12, help="Minimum faces required on both meshes for a part metric.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.02, 0.05], help="F-score thresholds in unit-box coordinates.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-json", help="Output JSON path. Defaults to <output-dir>/evaluation/target_metrics.json.")
    parser.add_argument("--save-part-csv", help="Optional CSV path for per-part metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inferred = resolve_from_output_dir(args.output_dir)

    deformed_mesh = args.deformed_mesh or inferred.get("deformed_mesh")
    source_mesh = args.source_mesh or inferred.get("source_mesh")
    target_mesh = args.target_mesh or inferred.get("target_mesh")
    source_labels = args.partfield_source_labels or inferred.get("partfield_source_labels")
    target_labels = args.partfield_target_labels or inferred.get("partfield_target_labels")

    if not deformed_mesh or not target_mesh:
        raise SystemExit("Need --deformed-mesh and --target-mesh, or --output-dir with config.yml.")

    rng = np.random.default_rng(args.seed)
    deformed_vertices, deformed_faces = load_mesh(deformed_mesh)
    target_vertices, target_faces = load_mesh(target_mesh)
    deformed_vertices = unit_vertices(deformed_vertices)
    target_vertices = unit_vertices(target_vertices)

    deformed_points, deformed_normals = sample_surface(deformed_vertices, deformed_faces, args.samples, rng)
    target_points, target_normals = sample_surface(target_vertices, target_faces, args.samples, rng)
    metrics: Dict[str, object] = {
        "deformed_mesh": str(deformed_mesh),
        "target_mesh": str(target_mesh),
        "source_mesh": str(source_mesh) if source_mesh else None,
        "samples": int(args.samples),
        "thresholds": [float(x) for x in args.thresholds],
        "global": geometry_metrics(
            deformed_points,
            deformed_normals,
            target_points,
            target_normals,
            args.thresholds,
        ),
    }

    if source_mesh:
        source_vertices, source_faces = load_mesh(source_mesh)
        source_vertices = unit_vertices(source_vertices)
        if source_faces.shape == deformed_faces.shape and np.array_equal(source_faces, deformed_faces):
            distortion_faces = deformed_faces
        else:
            distortion_faces = source_faces
        metrics["source_distortion"] = source_distortion_metrics(source_vertices, deformed_vertices, distortion_faces)

    if source_labels and target_labels:
        metrics["partfield"] = partfield_metrics(
            deformed_vertices=deformed_vertices,
            deformed_faces=deformed_faces,
            target_vertices=target_vertices,
            target_faces=target_faces,
            source_labels_path=source_labels,
            target_labels_path=target_labels,
            label_mode=args.partfield_label_mode,
            points_per_part=args.part_samples,
            min_faces_per_part=args.min_part_faces,
            thresholds=args.thresholds,
            seed=args.seed + 1,
        )

    if args.save_json:
        json_path = Path(args.save_json)
    elif args.output_dir:
        json_path = Path(args.output_dir) / "evaluation" / "target_metrics.json"
    else:
        json_path = Path("target_metrics.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    if "partfield" in metrics:
        csv_path = Path(args.save_part_csv) if args.save_part_csv else json_path.with_name("partfield_part_metrics.csv")
        write_part_csv(csv_path, metrics["partfield"]["parts"])  # type: ignore[index]

    print(json.dumps(metrics["global"], indent=2))
    print(f"Saved metrics to {json_path}")


if __name__ == "__main__":
    main()
