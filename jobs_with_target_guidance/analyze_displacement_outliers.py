#!/usr/bin/env python3
"""Analyze when and how local mesh spikes emerge in saved MeshUp runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

try:
    import trimesh
except Exception:  # pragma: no cover - only used in cluster env
    trimesh = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional
    cKDTree = None


EPOCH_RE = re.compile(r"correspondence_epoch_(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output-name", default="outlier_analysis")
    return parser.parse_args()


def epoch_from_path(path: Path) -> int:
    match = EPOCH_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse epoch from {path}")
    return int(match.group(1))


def load_correspondence(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r") as f:
        data = json.load(f)
    base = np.asarray(data["original_vertices"], dtype=np.float64)
    deformed = np.asarray(data["deformed_vertices"], dtype=np.float64)
    faces = np.asarray(data["faces"], dtype=np.int64)
    return base, deformed, faces


def face_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for a, b, c in faces:
        edges.append(tuple(sorted((int(a), int(b)))))
        edges.append(tuple(sorted((int(b), int(c)))))
        edges.append(tuple(sorted((int(c), int(a)))))
    return edges


def build_topology(n_vertices: int, faces: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    edge_counts = Counter(face_edges(faces))
    neighbor_sets = [set() for _ in range(n_vertices)]
    boundary = np.zeros(n_vertices, dtype=bool)
    for (a, b), count in edge_counts.items():
        neighbor_sets[a].add(b)
        neighbor_sets[b].add(a)
        if count == 1:
            boundary[a] = True
            boundary[b] = True
    neighbors = [np.asarray(sorted(s), dtype=np.int64) for s in neighbor_sets]
    unique_edges = np.asarray(list(edge_counts.keys()), dtype=np.int64)
    return neighbors, unique_edges, boundary


def local_means(values: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    result = np.zeros_like(values)
    for i, nbr in enumerate(neighbors):
        if len(nbr):
            result[i] = values[nbr].mean(axis=0)
        else:
            result[i] = values[i]
    return result


def local_edge_mean(vertices: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    result = np.zeros(len(vertices), dtype=np.float64)
    for i, nbr in enumerate(neighbors):
        if len(nbr):
            result[i] = np.linalg.norm(vertices[nbr] - vertices[i], axis=1).mean()
    nonzero = result[result > 0]
    fallback = float(np.median(nonzero)) if len(nonzero) else 1.0
    result[result <= 0] = fallback
    return result


def robust_z(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-12:
        return np.zeros_like(values)
    return 0.6745 * (values - med) / mad


def load_target_tree(config_path: Path):
    if trimesh is None or cKDTree is None:
        return None
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    target_path = Path(cfg.get("target_mesh", ""))
    if not target_path.is_absolute():
        target_path = Path.cwd() / target_path
    if not target_path.exists():
        return None
    mesh = trimesh.load(str(target_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return cKDTree(np.asarray(mesh.vertices, dtype=np.float64))


def metric_pack(
    base: np.ndarray,
    deformed: np.ndarray,
    neighbors: list[np.ndarray],
    local_scale: np.ndarray,
    base_lap: np.ndarray,
) -> dict[str, np.ndarray]:
    displacement = deformed - base
    mean_deformed_neighbors = local_means(deformed, neighbors)
    mean_neighbor_displacement = local_means(displacement, neighbors)
    mean_deformed_edge = local_edge_mean(deformed, neighbors)

    disp_mag = np.linalg.norm(displacement, axis=1)
    disp_jump = np.linalg.norm(displacement - mean_neighbor_displacement, axis=1) / local_scale
    lap = np.linalg.norm(deformed - mean_deformed_neighbors, axis=1)
    lap_change = np.maximum(lap - base_lap, 0.0) / local_scale
    edge_ratio = mean_deformed_edge / local_scale
    return {
        "disp_mag": disp_mag,
        "disp_jump": disp_jump,
        "lap_change": lap_change,
        "edge_ratio": edge_ratio,
        "robust_jump_z": robust_z(disp_jump),
    }


def first_epoch(rows: list[dict], vertex: int, key: str, threshold: float) -> int | None:
    for row in rows:
        if row[key][vertex] >= threshold:
            return int(row["epoch"])
    return None


def fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if math.isfinite(value):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    corr_files = sorted((run_dir / "correspondence").glob("correspondence_epoch_*.json"), key=epoch_from_path)
    if not corr_files:
        raise FileNotFoundError(f"No correspondence files found under {run_dir}")

    base0, deformed0, faces = load_correspondence(corr_files[0])
    neighbors, unique_edges, boundary = build_topology(len(base0), faces)
    valence = np.asarray([len(n) for n in neighbors], dtype=np.int64)
    local_scale = local_edge_mean(base0, neighbors)
    base_lap = np.linalg.norm(base0 - local_means(base0, neighbors), axis=1)

    target_tree = load_target_tree(run_dir / "config.yml")
    rows = []
    for path in corr_files:
        epoch = epoch_from_path(path)
        base, deformed, _ = load_correspondence(path)
        metrics = metric_pack(base, deformed, neighbors, local_scale, base_lap)
        row = {"epoch": epoch, "base": base, "deformed": deformed}
        row.update(metrics)
        if target_tree is not None:
            target_dist, _ = target_tree.query(deformed)
            row["target_dist"] = target_dist
        rows.append(row)

    final = rows[-1]
    spike_score = final["disp_jump"] + final["lap_change"] + np.maximum(final["edge_ratio"] - 1.0, 0.0)
    candidate_idx = np.argsort(spike_score)[::-1][: args.top_k]

    out_dir = run_dir / "displacement_viz" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "candidate_metrics_by_epoch.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "vertex",
                "epoch",
                "disp_mag",
                "disp_jump",
                "lap_change",
                "edge_ratio",
                "robust_jump_z",
                "target_dist",
            ]
        )
        for vertex in candidate_idx:
            for row in rows:
                writer.writerow(
                    [
                        int(vertex),
                        int(row["epoch"]),
                        float(row["disp_mag"][vertex]),
                        float(row["disp_jump"][vertex]),
                        float(row["lap_change"][vertex]),
                        float(row["edge_ratio"][vertex]),
                        float(row["robust_jump_z"][vertex]),
                        float(row.get("target_dist", np.full(len(base0), np.nan))[vertex]),
                    ]
                )

    top_path = out_dir / "final_top_outliers.csv"
    with top_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "vertex",
                "score",
                "disp_mag",
                "disp_jump",
                "lap_change",
                "edge_ratio",
                "robust_jump_z",
                "valence",
                "boundary",
                "local_scale",
                "first_epoch_jump_gt_2",
                "first_epoch_jump_top_outlier",
                "first_epoch_half_final_jump",
                "target_dist_epoch_1",
                "target_dist_final",
            ]
        )
        for rank, vertex in enumerate(candidate_idx, start=1):
            final_jump = float(final["disp_jump"][vertex])
            top_epoch = first_epoch(rows, int(vertex), "robust_jump_z", 8.0)
            half_epoch = first_epoch(rows, int(vertex), "disp_jump", max(2.0, 0.5 * final_jump))
            target_epoch_1 = rows[0].get("target_dist", np.full(len(base0), np.nan))[vertex]
            target_final = final.get("target_dist", np.full(len(base0), np.nan))[vertex]
            writer.writerow(
                [
                    rank,
                    int(vertex),
                    float(spike_score[vertex]),
                    float(final["disp_mag"][vertex]),
                    final_jump,
                    float(final["lap_change"][vertex]),
                    float(final["edge_ratio"][vertex]),
                    float(final["robust_jump_z"][vertex]),
                    int(valence[vertex]),
                    bool(boundary[vertex]),
                    float(local_scale[vertex]),
                    first_epoch(rows, int(vertex), "disp_jump", 2.0),
                    top_epoch,
                    half_epoch,
                    float(target_epoch_1),
                    float(target_final),
                ]
            )

    config = yaml.safe_load((run_dir / "config.yml").read_text())
    summary_path = out_dir / "summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Outlier Analysis\n\n")
        f.write(f"Run: `{run_dir}`\n\n")
        f.write("## Loss/Config Signals\n\n")
        for key in [
            "use_sds",
            "use_dino_loss",
            "image_weight",
            "target_mesh_render_weight",
            "target_mesh_chamfer_weight",
            "target_mesh_partfield_chamfer_weight",
            "deformation_parameterization",
            "regularize_jacobians_weight",
            "jacobian_neighbor_smooth_weight",
            "jacobian_outlier_weight",
            "jacobian_outlier_power",
            "target_mesh_chamfer_points",
            "target_mesh_n_azimuths",
            "target_mesh_n_elevations",
        ]:
            f.write(f"- `{key}`: `{config.get(key)}`\n")
        f.write("\n## Mesh Quality Signals\n\n")
        f.write(f"- vertices: `{len(base0)}`\n")
        f.write(f"- faces: `{len(faces)}`\n")
        f.write(f"- unique edges: `{len(unique_edges)}`\n")
        f.write(f"- boundary vertices: `{int(boundary.sum())}`\n")
        f.write(f"- median original local edge scale: `{float(np.median(local_scale)):.6f}`\n")
        f.write("\n## Final Top Local Outliers\n\n")
        f.write(
            "| rank | vertex | score | disp | jump | lap_change | edge_ratio | z | valence | boundary | first jump>2 | first z>8 | final target dist |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|\n")
        for rank, vertex in enumerate(candidate_idx[:8], start=1):
            target_final = final.get("target_dist", np.full(len(base0), np.nan))[vertex]
            f.write(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        str(int(vertex)),
                        fmt(float(spike_score[vertex])),
                        fmt(float(final["disp_mag"][vertex])),
                        fmt(float(final["disp_jump"][vertex])),
                        fmt(float(final["lap_change"][vertex])),
                        fmt(float(final["edge_ratio"][vertex])),
                        fmt(float(final["robust_jump_z"][vertex])),
                        str(int(valence[vertex])),
                        str(bool(boundary[vertex])),
                        fmt(first_epoch(rows, int(vertex), "disp_jump", 2.0)),
                        fmt(first_epoch(rows, int(vertex), "robust_jump_z", 8.0)),
                        fmt(float(target_final)),
                    ]
                )
                + " |\n"
            )
        f.write("\n## Interpretation Hints\n\n")
        f.write("- `jump` is the vertex displacement minus the average displacement of its 1-ring neighbors, normalized by original local edge scale.\n")
        f.write("- `edge_ratio` is current 1-ring edge length divided by original 1-ring edge length; high values mean local stretching/spike geometry.\n")
        f.write("- If `use_sds`, `use_dino_loss`, `image_weight`, and `target_mesh_render_weight` are zero/false, camera/view angle losses are not driving geometry.\n")
        f.write("- A late rise in `jump`/`edge_ratio` points to optimization concentrating Chamfer error into a local vertex/patch rather than an initial mesh loading bug.\n")

    print(summary_path)
    print(top_path)
    print(csv_path)


if __name__ == "__main__":
    main()
