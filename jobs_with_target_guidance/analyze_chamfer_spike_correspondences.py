#!/usr/bin/env python3
"""Analyze hard-Chamfer correspondences for local spike vertices.

This script is intentionally lightweight: it only needs numpy and pyyaml.  It
mirrors the deterministic vertex sampling used by loop_tracked.py and
partfield_chamfer.py, then reports which sampled target vertices the final
deformed source vertices are closest to globally and inside their PartField
buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--vertices",
        default="",
        help="Optional comma-separated source vertex ids to analyze instead of top local outliers.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return root / p


def unit_vertices(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    scale = 2.0 / max(float((vmax - vmin).max()), 1e-12)
    return (vertices - (vmax + vmin) * 0.5) * scale


def sample_indices(n_items: int, n_points: int) -> np.ndarray:
    if n_points <= 0 or n_items <= n_points:
        return np.arange(n_items, dtype=np.int64)
    return np.linspace(0, n_items - 1, num=n_points).astype(np.int64)


def load_correspondence(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r") as f:
        data = json.load(f)
    return (
        np.asarray(data["original_vertices"], dtype=np.float64),
        np.asarray(data["deformed_vertices"], dtype=np.float64),
        np.asarray(data["faces"], dtype=np.int64),
    )


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                for part in parts[:3]:
                    # OBJ supports v, v/vt, v//vn, v/vt/vn. Indices are 1-based.
                    face.append(int(part.split("/")[0]) - 1)
                if len(face) == 3:
                    faces.append(face)
    if not vertices or not faces:
        raise ValueError(f"Could not read vertices/faces from OBJ: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj(path)
    raise ValueError(f"Only OBJ is supported by this lightweight analyzer, got: {path}")


def face_edges(faces: np.ndarray) -> Iterable[tuple[int, int]]:
    for a, b, c in faces:
        yield tuple(sorted((int(a), int(b))))
        yield tuple(sorted((int(b), int(c))))
        yield tuple(sorted((int(c), int(a))))


def build_neighbors(n_vertices: int, faces: np.ndarray) -> list[np.ndarray]:
    neighbor_sets = [set() for _ in range(n_vertices)]
    for a, b in face_edges(faces):
        neighbor_sets[a].add(b)
        neighbor_sets[b].add(a)
    return [np.asarray(sorted(n), dtype=np.int64) for n in neighbor_sets]


def local_means(values: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(values)
    for idx, nbr in enumerate(neighbors):
        out[idx] = values[nbr].mean(axis=0) if len(nbr) else values[idx]
    return out


def local_edge_mean(vertices: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(len(vertices), dtype=np.float64)
    for idx, nbr in enumerate(neighbors):
        if len(nbr):
            out[idx] = np.linalg.norm(vertices[nbr] - vertices[idx], axis=1).mean()
    nonzero = out[out > 0]
    out[out <= 0] = float(np.median(nonzero)) if len(nonzero) else 1.0
    return out


def robust_z(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-12:
        return np.zeros_like(values)
    return 0.6745 * (values - med) / mad


def spike_scores(base: np.ndarray, deformed: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    neighbors = build_neighbors(len(base), faces)
    local_scale = local_edge_mean(base, neighbors)
    disp = deformed - base
    mean_disp = local_means(disp, neighbors)
    mean_def = local_means(deformed, neighbors)
    mean_edge = local_edge_mean(deformed, neighbors)
    base_lap = np.linalg.norm(base - local_means(base, neighbors), axis=1)
    lap = np.linalg.norm(deformed - mean_def, axis=1)
    disp_jump = np.linalg.norm(disp - mean_disp, axis=1) / local_scale
    lap_change = np.maximum(lap - base_lap, 0.0) / local_scale
    edge_ratio = mean_edge / local_scale
    metrics = {
        "disp_mag": np.linalg.norm(disp, axis=1),
        "disp_jump": disp_jump,
        "lap_change": lap_change,
        "edge_ratio": edge_ratio,
        "robust_jump_z": robust_z(disp_jump),
        "local_scale": local_scale,
        "valence": np.asarray([len(n) for n in neighbors], dtype=np.int64),
    }
    score = disp_jump + lap_change + np.maximum(edge_ratio - 1.0, 0.0)
    return score, metrics


def nearest(query: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = candidates[None, :, :] - query[:, None, :]
    dist2 = np.einsum("qci,qci->qc", diff, diff)
    idx = dist2.argmin(axis=1)
    return np.sqrt(dist2[np.arange(len(query)), idx]), idx.astype(np.int64)


def nearest_single(point: np.ndarray, candidates: np.ndarray) -> tuple[float, int]:
    diff = candidates - point[None, :]
    dist2 = np.einsum("ij,ij->i", diff, diff)
    idx = int(dist2.argmin())
    return float(math.sqrt(float(dist2[idx]))), idx


def quantile_position(values: np.ndarray, value: float) -> float:
    if len(values) <= 1:
        return 0.0
    return float((values <= value).mean())


def descriptor(unit_point: np.ndarray, bucket_points: np.ndarray) -> str:
    axes = ["x", "y", "z"]
    pieces = []
    for axis_id, axis in enumerate(axes):
        q = quantile_position(bucket_points[:, axis_id], float(unit_point[axis_id]))
        if q <= 0.15:
            pieces.append(f"low {axis}")
        elif q >= 0.85:
            pieces.append(f"high {axis}")
        else:
            pieces.append(f"mid {axis}")
    return ", ".join(pieces)


def write_colored_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    highlighted: set[int],
    highlight_color: tuple[int, int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = colors.copy()
    for idx in highlighted:
        if 0 <= idx < len(rgb):
            rgb[idx] = np.asarray(highlight_color, dtype=np.uint8)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(vertices, rgb):
            f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def write_match_lines_ply(
    path: Path,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    source_labels: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    target_labels: np.ndarray,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = source_vertices.copy()
    target = target_vertices.copy()
    gap = 0.35
    source[:, 0] -= 1.0 + gap
    target[:, 0] += 1.0 + gap

    source_colors = label_colors(source_labels)
    target_colors = label_colors(target_labels)
    source_highlights = {int(row["source_vertex"]) for row in rows}
    target_highlights = {int(row["partfield_nearest_target_vertex"]) for row in rows}
    for idx in source_highlights:
        source_colors[idx] = np.asarray([255, 255, 255], dtype=np.uint8)
    for idx in target_highlights:
        target_colors[idx] = np.asarray([255, 255, 255], dtype=np.uint8)

    vertices = np.concatenate([source, target], axis=0)
    colors = np.concatenate([source_colors, target_colors], axis=0)
    target_offset = len(source)
    faces = np.concatenate([source_faces, target_faces + target_offset], axis=0)
    edges = [
        (int(row["source_vertex"]), target_offset + int(row["partfield_nearest_target_vertex"]))
        for row in rows
    ]

    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v, c in zip(vertices, colors):
            f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
        for a, b in edges:
            f.write(f"{a} {b} 255 220 0\n")


def label_colors(labels: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
        ],
        dtype=np.uint8,
    )
    return palette[np.asarray(labels, dtype=np.int64) % len(palette)]


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    run_dir = args.run_dir
    out_dir = args.output_dir or run_dir / "displacement_viz" / "chamfer_correspondence_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((run_dir / "config.yml").read_text())
    base, deformed, faces = load_correspondence(run_dir / "correspondence" / "final_correspondence.json")
    target_path = resolve_path(cfg["target_mesh"], root)
    target_vertices, target_faces = load_mesh(target_path)

    labels_npz = np.load(run_dir / "partfield_labels" / "bucket_labels.npz")
    source_labels = labels_npz["source_labels"].astype(np.int64)
    target_labels = labels_npz["target_labels"].astype(np.int64)
    n_buckets = int(labels_npz["n_buckets"])

    source_unit = unit_vertices(deformed)
    target_unit = unit_vertices(target_vertices)

    score, metrics = spike_scores(base, deformed, faces)
    if args.vertices.strip():
        vertices = [int(v) for v in args.vertices.replace(" ", "").split(",") if v]
    else:
        vertices = np.argsort(score)[::-1][: args.top_k].astype(int).tolist()

    global_points = int(cfg.get("target_mesh_chamfer_points", 3072))
    global_src_sample = sample_indices(len(source_unit), global_points)
    global_tgt_sample = sample_indices(len(target_unit), global_points)
    global_src_pos = {int(v): i for i, v in enumerate(global_src_sample)}

    rows: list[dict[str, object]] = []
    matched_target_vertices: set[int] = set()
    highlighted_source_vertices = set(vertices)

    for rank, vertex in enumerate(vertices, start=1):
        bucket = int(source_labels[vertex])
        bucket_src_all = np.flatnonzero(source_labels == bucket)
        bucket_tgt_all = np.flatnonzero(target_labels == bucket)
        bucket_src_sample = bucket_src_all[sample_indices(len(bucket_src_all), int(cfg.get("target_mesh_partfield_chamfer_points", 512)))]
        bucket_tgt_sample = bucket_tgt_all[sample_indices(len(bucket_tgt_all), int(cfg.get("target_mesh_partfield_chamfer_points", 512)))]
        bucket_src_pos = {int(v): i for i, v in enumerate(bucket_src_sample)}

        pf_dist, pf_local_idx = nearest_single(source_unit[vertex], target_unit[bucket_tgt_sample])
        pf_target_vertex = int(bucket_tgt_sample[pf_local_idx])
        matched_target_vertices.add(pf_target_vertex)

        global_dist, global_local_idx = nearest_single(source_unit[vertex], target_unit[global_tgt_sample])
        global_target_vertex = int(global_tgt_sample[global_local_idx])

        pf_candidate_is_sampled = int(vertex) in bucket_src_pos
        global_candidate_is_sampled = int(vertex) in global_src_pos

        pf_tgt_to_src_count = 0
        pf_tgt_to_src_mean_dist = math.nan
        pf_tgt_to_src_targets: list[int] = []
        if pf_candidate_is_sampled:
            pf_tgt_dist, pf_tgt_nn = nearest(target_unit[bucket_tgt_sample], source_unit[bucket_src_sample])
            local_src_pos = bucket_src_pos[int(vertex)]
            selected = np.flatnonzero(pf_tgt_nn == local_src_pos)
            pf_tgt_to_src_count = int(len(selected))
            if len(selected):
                pf_tgt_to_src_mean_dist = float(pf_tgt_dist[selected].mean())
                closest = selected[np.argsort(pf_tgt_dist[selected])[:8]]
                pf_tgt_to_src_targets = [int(bucket_tgt_sample[i]) for i in closest]
                matched_target_vertices.update(pf_tgt_to_src_targets)

        global_tgt_to_src_count = 0
        global_tgt_to_src_mean_dist = math.nan
        global_tgt_to_src_targets: list[int] = []
        if global_candidate_is_sampled:
            g_tgt_dist, g_tgt_nn = nearest(target_unit[global_tgt_sample], source_unit[global_src_sample])
            local_src_pos = global_src_pos[int(vertex)]
            selected = np.flatnonzero(g_tgt_nn == local_src_pos)
            global_tgt_to_src_count = int(len(selected))
            if len(selected):
                global_tgt_to_src_mean_dist = float(g_tgt_dist[selected].mean())
                closest = selected[np.argsort(g_tgt_dist[selected])[:8]]
                global_tgt_to_src_targets = [int(global_tgt_sample[i]) for i in closest]
                matched_target_vertices.update(global_tgt_to_src_targets)

        bucket_points = target_unit[bucket_tgt_all]
        rows.append(
            {
                "rank": rank,
                "source_vertex": int(vertex),
                "source_bucket": bucket,
                "source_bucket_size": int(len(bucket_src_all)),
                "target_bucket_size": int(len(bucket_tgt_all)),
                "source_unit_x": float(source_unit[vertex, 0]),
                "source_unit_y": float(source_unit[vertex, 1]),
                "source_unit_z": float(source_unit[vertex, 2]),
                "deformed_x": float(deformed[vertex, 0]),
                "deformed_y": float(deformed[vertex, 1]),
                "deformed_z": float(deformed[vertex, 2]),
                "disp_mag": float(metrics["disp_mag"][vertex]),
                "disp_jump": float(metrics["disp_jump"][vertex]),
                "lap_change": float(metrics["lap_change"][vertex]),
                "edge_ratio": float(metrics["edge_ratio"][vertex]),
                "robust_jump_z": float(metrics["robust_jump_z"][vertex]),
                "score": float(score[vertex]),
                "partfield_source_sampled": bool(pf_candidate_is_sampled),
                "partfield_nearest_target_vertex": pf_target_vertex,
                "partfield_nearest_dist_unit": pf_dist,
                "partfield_target_unit_x": float(target_unit[pf_target_vertex, 0]),
                "partfield_target_unit_y": float(target_unit[pf_target_vertex, 1]),
                "partfield_target_unit_z": float(target_unit[pf_target_vertex, 2]),
                "partfield_target_raw_x": float(target_vertices[pf_target_vertex, 0]),
                "partfield_target_raw_y": float(target_vertices[pf_target_vertex, 1]),
                "partfield_target_raw_z": float(target_vertices[pf_target_vertex, 2]),
                "partfield_target_region": descriptor(target_unit[pf_target_vertex], bucket_points),
                "partfield_tgt_to_src_count": pf_tgt_to_src_count,
                "partfield_tgt_to_src_mean_dist_unit": pf_tgt_to_src_mean_dist,
                "partfield_tgt_to_src_target_vertices": " ".join(map(str, pf_tgt_to_src_targets)),
                "global_source_sampled": bool(global_candidate_is_sampled),
                "global_nearest_target_vertex": global_target_vertex,
                "global_nearest_dist_unit": global_dist,
                "global_target_unit_x": float(target_unit[global_target_vertex, 0]),
                "global_target_unit_y": float(target_unit[global_target_vertex, 1]),
                "global_target_unit_z": float(target_unit[global_target_vertex, 2]),
                "global_target_bucket": int(target_labels[global_target_vertex]),
                "global_tgt_to_src_count": global_tgt_to_src_count,
                "global_tgt_to_src_mean_dist_unit": global_tgt_to_src_mean_dist,
                "global_tgt_to_src_target_vertices": " ".join(map(str, global_tgt_to_src_targets)),
            }
        )

    csv_path = out_dir / "chamfer_spike_correspondences.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_bucket = Counter(int(r["source_bucket"]) for r in rows)
    summary = {
        "run_dir": str(run_dir),
        "target_mesh": str(target_path),
        "global_chamfer_weight": float(cfg.get("target_mesh_chamfer_weight", 0.0)),
        "partfield_chamfer_weight": float(cfg.get("target_mesh_partfield_chamfer_weight", 0.0)),
        "partfield_points_per_bucket": int(cfg.get("target_mesh_partfield_chamfer_points", 512)),
        "global_chamfer_points": global_points,
        "n_buckets": n_buckets,
        "analyzed_vertices": vertices,
        "analyzed_vertices_by_bucket": dict(sorted(by_bucket.items())),
        "rows": rows,
    }
    summary_path = out_dir / "chamfer_spike_correspondences.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    source_colors = label_colors(source_labels)
    target_colors = label_colors(target_labels)
    write_colored_ply(
        out_dir / "source_spikes_marked.ply",
        deformed,
        faces,
        source_colors,
        highlighted_source_vertices,
        (255, 255, 255),
    )
    write_colored_ply(
        out_dir / "target_matches_marked.ply",
        target_vertices,
        target_faces,
        target_colors,
        matched_target_vertices,
        (255, 255, 255),
    )
    write_match_lines_ply(
        out_dir / "source_target_partfield_match_lines.ply",
        source_unit,
        faces,
        source_labels,
        target_unit,
        target_faces,
        target_labels,
        rows,
    )

    md_path = out_dir / "chamfer_spike_correspondences.md"
    with md_path.open("w") as f:
        f.write("# Chamfer Spike Correspondence Probe\n\n")
        f.write(f"- run: `{run_dir}`\n")
        f.write(f"- target: `{target_path}`\n")
        f.write(f"- global Chamfer: weight `{cfg.get('target_mesh_chamfer_weight', 0.0)}`, sampled vertices `{global_points}`\n")
        f.write(
            f"- PartField Chamfer: weight `{cfg.get('target_mesh_partfield_chamfer_weight', 0.0)}`, "
            f"sampled vertices per bucket `{cfg.get('target_mesh_partfield_chamfer_points', 512)}`\n\n"
        )
        f.write(
            "| rank | src v | bucket | score | sampled PF/global | PF nearest target | PF dist | PF region | PF tgt->src count | global nearest target | global dist | global bucket |\n"
        )
        f.write("|---:|---:|---:|---:|:---:|---:|---:|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(row["rank"]),
                        str(row["source_vertex"]),
                        str(row["source_bucket"]),
                        f"{row['score']:.4f}",
                        f"{int(row['partfield_source_sampled'])}/{int(row['global_source_sampled'])}",
                        str(row["partfield_nearest_target_vertex"]),
                        f"{row['partfield_nearest_dist_unit']:.4f}",
                        str(row["partfield_target_region"]),
                        str(row["partfield_tgt_to_src_count"]),
                        str(row["global_nearest_target_vertex"]),
                        f"{row['global_nearest_dist_unit']:.4f}",
                        str(row["global_target_bucket"]),
                    ]
                )
                + " |\n"
            )
    print(md_path)
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
