#!/usr/bin/env python3
"""Per-bucket deformation discrepancy and DenseCorr-style AUC analysis.

This script complements evaluate_deformation_quality.py.  It focuses on
localized failure analysis:

* per-bucket source/deformed-to-target success rates;
* per-bucket target coverage rates;
* source-vertex semantic AUC curves in the DenseCorr3D style;
* colored PLY heatmaps for per-vertex global and semantic distance;
* optional PNG plots for bucket success, bucket Chamfer, vertex error
  histograms/CDFs, and rendered vertex-error heatmaps.

The DenseCorr3D paper uses normalized geodesic error and AUC of a
threshold-accuracy curve for predicted correspondences.  The local MeshUp
outputs are deformed source meshes, not point-to-point maps, and the local
DenseCorr3D copy does not include the paper's geodesic matrices.  Therefore
the AUC here is an analogous Euclidean, unit-box, semantic-group AUC.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs_with_target_guidance.evaluate_deformation_quality import (  # noqa: E402
    ResolvedRun,
    nearest_distances,
    percentile_metrics,
    resolve_path,
    resolve_run,
    sample_surface_with_labels,
    threshold_key,
)
from jobs_with_target_guidance.evaluate_target_pipeline import (  # noqa: E402
    labels_to_vertex_labels,
    load_mesh,
    unit_vertices,
    vertex_labels_to_face_labels,
)

KNOWN_ANIMAL_NAMES = [
    "dense_giraffe",
    "giraffe4",
    "horse2",
    "panda3",
    "panda2",
    "cheetah",
    "panther",
    "elephant",
    "giraffe",
    "moose",
    "bear",
    "cat2",
    "fox",
]

DENSECORR_ID_TO_ANIMAL = {
    "2d6b3": "elephant",
    "1d6d1": "moose",
    "34fb4": "giraffe",
    "071b8": "panther",
    "96615": "bear",
    "bdfd0": "cheetah",
}


def sanitize_name(name: str, max_len: int = 120) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return clean[:max_len].strip("_") or "run"


def display_path(path: Path) -> str:
    path = path.resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return str(path.relative_to(PROJECT_ROOT))
    return str(path)


def pretty_name(name: str) -> str:
    return name.replace("_", " ")


def animal_from_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    stem = path.stem.lower()
    for densecorr_id, animal in DENSECORR_ID_TO_ANIMAL.items():
        if stem.startswith(densecorr_id):
            return animal
    for animal in sorted(KNOWN_ANIMAL_NAMES, key=len, reverse=True):
        if re.search(rf"(^|_){re.escape(animal)}($|_)", stem):
            return animal
    return None


def infer_pair_names(run: ResolvedRun) -> Tuple[str, str]:
    run_name = run.output_dir.name.lower()
    for source in sorted(KNOWN_ANIMAL_NAMES, key=len, reverse=True):
        prefix = f"{source}_to_"
        if not run_name.startswith(prefix):
            continue
        rest = run_name[len(prefix) :]
        for target in sorted(KNOWN_ANIMAL_NAMES, key=len, reverse=True):
            if rest == target or rest.startswith(target + "_"):
                return source, target

    source = animal_from_path(run.source_mesh)
    target = animal_from_path(run.target_mesh)
    if source and target:
        return source, target

    match = re.match(r"(.+?)_to_(.+?)(?:_(?:sam3d|chain|densecorr3d|hard|pf\d|bucketavg|srcstable|full|5k)|$)", run_name)
    if match:
        return match.group(1), match.group(2)
    return "unknown_source", "unknown_target"


def method_label_from_family(family: str) -> str:
    lower = family.lower()
    if "sam3d" in lower:
        return "SAM3D selected"
    if "chain" in lower:
        return "chain"
    if "bucketbalanced" in lower:
        return "DenseCorr3D bucket-balanced 5k"
    if "srcstable" in lower:
        return "DenseCorr3D source-stable 5k"
    if "requested_animals" in lower:
        return "DenseCorr3D requested animals"
    return family


def context_line(context: Dict[str, object]) -> str:
    return (
        f"Source: {context['source_label']} | Target: {context['target_label']} | "
        f"Method: {context['method_label']}"
    )


def finite_mean(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def finite_percentile(values: np.ndarray, percentile: float) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, percentile))


def harmonic(precision: float, recall: float) -> float:
    denom = precision + recall
    return 0.0 if denom <= 0 else float(2.0 * precision * recall / denom)


def safe_float(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def auc_from_errors(errors: np.ndarray, max_threshold: float, n_steps: int) -> Tuple[float, np.ndarray, np.ndarray]:
    finite = errors[np.isfinite(errors)]
    thresholds = np.linspace(0.0, max_threshold, n_steps)
    if finite.size == 0 or max_threshold <= 0:
        return 0.0, thresholds, np.zeros_like(thresholds)
    accuracies = np.array([(finite <= t).mean() for t in thresholds], dtype=np.float64)
    return float(np.trapz(accuracies, thresholds) / max_threshold), thresholds, accuracies


def label_kdtrees(points: np.ndarray, labels: np.ndarray) -> Dict[int, cKDTree]:
    trees: Dict[int, cKDTree] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if mask.any():
            trees[int(label)] = cKDTree(points[mask])
    return trees


def same_label_distances_from_trees(
    src_points: np.ndarray,
    src_labels: np.ndarray,
    target_trees: Dict[int, cKDTree],
) -> np.ndarray:
    dists = np.full(src_points.shape[0], np.inf, dtype=np.float64)
    for label in sorted(set(src_labels.tolist())):
        tree = target_trees.get(int(label))
        if tree is None:
            continue
        mask = src_labels == label
        dists[mask] = tree.query(src_points[mask], workers=-1)[0]
    return dists


def nearest_label_agreement_from_samples(
    src_points: np.ndarray,
    src_labels: np.ndarray,
    tgt_points: np.ndarray,
    tgt_labels: np.ndarray,
) -> float:
    if src_points.size == 0 or tgt_points.size == 0:
        return 0.0
    tree = cKDTree(tgt_points)
    _, nn = tree.query(src_points, workers=-1)
    return float((tgt_labels[nn] == src_labels).mean())


def color_ramp(values: np.ndarray, max_value: float) -> np.ndarray:
    """Blue-cyan-green-yellow-red ramp for scalar errors."""
    colors = np.array(
        [
            [35, 70, 185],
            [0, 180, 255],
            [30, 190, 80],
            [250, 220, 40],
            [215, 25, 32],
        ],
        dtype=np.float64,
    )
    finite = np.isfinite(values)
    t = np.zeros_like(values, dtype=np.float64)
    if max_value > 0:
        t[finite] = np.clip(values[finite] / max_value, 0.0, 1.0)
    scaled = t * (len(colors) - 1)
    lo = np.floor(scaled).astype(int)
    hi = np.clip(lo + 1, 0, len(colors) - 1)
    frac = scaled - lo
    rgb = colors[lo] * (1.0 - frac[:, None]) + colors[hi] * frac[:, None]
    rgb[~finite] = np.array([255, 0, 255], dtype=np.float64)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def bucket_highlight_colors(labels: np.ndarray, buckets: Sequence[int]) -> np.ndarray:
    colors = np.full((labels.shape[0], 3), 190, dtype=np.uint8)
    palette = np.array(
        [
            [220, 38, 38],
            [245, 158, 11],
            [124, 58, 237],
            [5, 150, 105],
            [8, 145, 178],
        ],
        dtype=np.uint8,
    )
    for idx, bucket in enumerate(buckets):
        colors[labels == int(bucket)] = palette[idx % len(palette)]
    return colors


def write_colored_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex, color in zip(vertices, colors):
            f.write(
                f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def maybe_import_pyplot():
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/meshup_matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - optional plotting path.
        print(f"Warning: matplotlib unavailable, skipping PNG plots: {exc}")
        return None


def write_bucket_plot(
    path: Path,
    rows: List[Dict[str, object]],
    success_threshold: float,
    context: Dict[str, object],
    plt,
) -> None:
    if plt is None or not rows:
        return
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    labels = [str(row["bucket"]) for row in rows]
    fscores = [float(row[f"surface_fscore_tau_{threshold_key(success_threshold)}"]) for row in rows]
    p95s = [safe_float(row.get("vertex_semantic_error_p95")) for row in rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(max(10.5, len(labels) * 0.8), 6.2))
    ax1.bar(x, fscores, color="#3b82f6", label=f"F@{success_threshold:g}")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Per-bucket semantic F-score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Bucket / semantic region")
    ax2 = ax1.twinx()
    ax2.plot(x, p95s, color="#dc2626", marker="o", label="vertex p95 error")
    ax2.set_ylabel("Vertex semantic error p95")
    ax1.grid(axis="y", alpha=0.25)
    handles = [
        Patch(facecolor="#3b82f6", label=f"blue bars: bucket F-score at tau={success_threshold:g}; higher is better"),
        Line2D([0], [0], color="#dc2626", marker="o", label="red line: p95 vertex semantic distance; lower is better"),
    ]
    ax1.set_title(f"Per-bucket target success\n{context_line(context)}", fontsize=12)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.08), ncol=1, fontsize=9)
    fig.text(
        0.5,
        0.02,
        "Distances use unit-box normalization. Semantic distance only compares points inside the same source/target bucket.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.34, top=0.82)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_hist_plot(
    path: Path,
    global_errors: np.ndarray,
    semantic_errors: np.ndarray,
    thresholds: Sequence[float],
    max_error: float,
    context: Dict[str, object],
    plt,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    finite_global = global_errors[np.isfinite(global_errors)]
    finite_semantic = semantic_errors[np.isfinite(semantic_errors)]
    bins = np.linspace(0.0, max_error, 50)
    ax.hist(finite_global, bins=bins, alpha=0.55, color="#2563eb", label="nearest target surface")
    ax.hist(finite_semantic, bins=bins, alpha=0.55, color="#dc2626", label="matching bucket only")
    for idx, threshold in enumerate(thresholds):
        if threshold <= max_error:
            ax.axvline(
                threshold,
                color="#111827",
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                label="success thresholds" if idx == 0 else None,
            )
    ax.set_xlabel("Per-vertex nearest-distance contribution to Chamfer")
    ax.set_ylabel("Vertex count")
    ax.set_title(f"Per-vertex Chamfer-distance histogram\n{context_line(context)}", fontsize=12)
    ax.legend(fontsize=9)
    fig.text(
        0.5,
        0.02,
        f"Dashed lines are tested thresholds: {', '.join(f'{t:g}' for t in thresholds)}. Distances are clipped to {max_error:g} on this plot.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_cdf_plot(
    path: Path,
    global_errors: np.ndarray,
    semantic_errors: np.ndarray,
    thresholds: Sequence[float],
    max_error: float,
    context: Dict[str, object],
    plt,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.linspace(0.0, max_error, 250)
    for values, color, label in [
        (global_errors, "#2563eb", "nearest target surface"),
        (semantic_errors, "#dc2626", "matching bucket only"),
    ]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        y = np.array([(finite <= threshold).mean() for threshold in x], dtype=np.float64)
        ax.plot(x, y, color=color, linewidth=2.0, label=label)
    for idx, threshold in enumerate(thresholds):
        if threshold <= max_error:
            ax.axvline(
                threshold,
                color="#111827",
                linestyle="--",
                linewidth=1.0,
                alpha=0.55,
                label="success thresholds" if idx == 0 else None,
            )
    ax.set_xlim(0.0, max_error)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Per-vertex Chamfer-like distance")
    ax.set_ylabel("Fraction of vertices below distance")
    ax.set_title(f"Per-vertex Chamfer-distance CDF\n{context_line(context)}", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.text(
        0.5,
        0.02,
        "Higher curve means more vertices are close to the target. Dashed lines show the thresholds used for success/F-score.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_vertex_error_surface_image(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_errors: np.ndarray,
    title: str,
    colorbar_label: str,
    max_error: float,
    context: Dict[str, object],
    plt,
    face_limit: int,
) -> None:
    if plt is None:
        return
    from matplotlib import cm, colors as mpl_colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if face_limit > 0 and faces.shape[0] > face_limit:
        face_idx = np.linspace(0, faces.shape[0] - 1, face_limit, dtype=np.int64)
        plot_faces = faces[face_idx]
    else:
        plot_faces = faces

    cleaned_errors = vertex_errors.astype(np.float64, copy=True)
    cleaned_errors[~np.isfinite(cleaned_errors)] = np.nan
    face_errors = np.nanmean(cleaned_errors[plot_faces], axis=1)
    try:
        cmap = plt.get_cmap("turbo")
    except ValueError:
        cmap = plt.get_cmap("viridis")
    norm = mpl_colors.Normalize(vmin=0.0, vmax=max_error)
    clipped = np.clip(face_errors, 0.0, max_error)
    face_colors = cmap(norm(clipped))
    face_colors[~np.isfinite(face_errors)] = np.array([1.0, 0.0, 1.0, 1.0])
    polys = vertices[plot_faces]

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    if radius <= 0:
        radius = 0.5

    fig = plt.figure(figsize=(12.5, 5.1))
    axes = []
    views = [(15, -60, "view A"), (15, 30, "view B"), (80, -90, "top")]
    for idx, (elev, azim, view_title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        axes.append(ax)
        mesh = Poly3DCollection(polys, facecolors=face_colors, linewidths=0.0, edgecolors="none")
        ax.add_collection3d(mesh)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_title, fontsize=10)
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.06, pad=0.08, aspect=45)
    cbar.set_label(colorbar_label)
    fig.suptitle(f"{title}\n{context_line(context)}", fontsize=12)
    fig.text(
        0.5,
        0.03,
        f"Color scale is clipped at {max_error:g}. Blue/green = low distance; yellow/red = high distance. Mesh shown is the deformed source.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.22, top=0.82, wspace=0.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_bucket_chamfer_plot(
    path: Path,
    rows: List[Dict[str, object]],
    success_threshold: float,
    worst_chamfer_buckets: Sequence[int],
    context: Dict[str, object],
    plt,
) -> None:
    if plt is None or not rows:
        return
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    labels = [str(row["bucket"]) for row in rows]
    chamfers = [safe_float(row.get("surface_semantic_chamfer_l1_sum")) for row in rows]
    fscores = [safe_float(row.get(f"surface_fscore_tau_{threshold_key(success_threshold)}")) for row in rows]
    worst = {int(bucket) for bucket in worst_chamfer_buckets}
    bar_colors = ["#dc2626" if int(row["bucket"]) in worst else "#64748b" for row in rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(max(10.5, len(labels) * 0.8), 6.2))
    ax1.bar(x, chamfers, color=bar_colors, label="semantic Chamfer")
    ax1.set_ylabel("Per-bucket semantic Chamfer")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Bucket / semantic region")
    ax2 = ax1.twinx()
    ax2.plot(x, fscores, color="#2563eb", marker="o", linewidth=1.8, label=f"F@{success_threshold:g}")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Per-bucket F-score")
    ax1.grid(axis="y", alpha=0.25)
    handles = [
        Patch(facecolor="#64748b", label="gray bars: semantic Chamfer per bucket; lower is better"),
        Patch(facecolor="#dc2626", label=f"red bars: top {len(worst)} worst Chamfer buckets"),
        Line2D([0], [0], color="#2563eb", marker="o", label=f"blue line: F-score at tau={success_threshold:g}; higher is better"),
    ]
    ax1.set_title(f"Per-bucket Chamfer distance\n{context_line(context)}", fontsize=12)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.08), ncol=1, fontsize=9)
    fig.text(
        0.5,
        0.02,
        "Chamfer here is same-bucket bidirectional mean distance: deformed bucket -> target bucket plus target bucket -> deformed bucket.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.34, top=0.82)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_worst_bucket_image(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    worst_rows: List[Dict[str, object]],
    context: Dict[str, object],
    success_threshold: float,
    plt,
    face_limit: int,
) -> None:
    if plt is None or not worst_rows:
        return
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if face_limit > 0 and faces.shape[0] > face_limit:
        face_idx = np.linspace(0, faces.shape[0] - 1, face_limit, dtype=np.int64)
        plot_faces = faces[face_idx]
        plot_labels = face_labels[face_idx]
    else:
        plot_faces = faces
        plot_labels = face_labels

    palette = ["#dc2626", "#f59e0b", "#7c3aed", "#059669", "#0891b2"]
    worst_buckets = [int(row["bucket"]) for row in worst_rows]
    bucket_colors = {bucket: palette[idx % len(palette)] for idx, bucket in enumerate(worst_buckets)}
    face_colors = [bucket_colors.get(int(label), "#c7c7c7") for label in plot_labels]
    polys = vertices[plot_faces]

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    if radius <= 0:
        radius = 0.5

    fig = plt.figure(figsize=(12.5, 5.4))
    views = [(15, -60, "view A"), (15, 30, "view B"), (80, -90, "top")]
    for idx, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        mesh = Poly3DCollection(polys, facecolors=face_colors, linewidths=0.0, edgecolors="none")
        ax.add_collection3d(mesh)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    legend_handles = [
        Patch(
            facecolor=bucket_colors[int(row["bucket"])],
            label=(
                f"rank {rank + 1}: bucket {int(row['bucket'])} "
                f"CD={safe_float(row.get('surface_semantic_chamfer_l1_sum')):.4f}, "
                f"F={safe_float(row.get(f'surface_fscore_tau_{threshold_key(success_threshold)}')):.3f}"
            ),
        )
        for rank, row in enumerate(worst_rows)
    ]
    legend_handles.append(Patch(facecolor="#c7c7c7", label="other buckets"))
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.09), ncol=2, fontsize=9)
    fig.suptitle(f"Worst buckets by semantic Chamfer\n{context_line(context)}", fontsize=12)
    fig.text(
        0.5,
        0.03,
        f"Colored regions are the top Chamfer buckets. CD = bidirectional same-bucket Chamfer. F uses tau={success_threshold:g}.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.22, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_run_geometry(run: ResolvedRun, label_mode: str):
    deformed_vertices_raw, deformed_faces = load_mesh(run.deformed_mesh)
    target_vertices_raw, target_faces = load_mesh(run.target_mesh)
    if run.source_labels is None or run.target_labels is None:
        raise ValueError(f"{run.output_dir}: missing source or target labels")
    source_vertex_labels = labels_to_vertex_labels(
        run.source_labels,
        deformed_vertices_raw.shape[0],
        deformed_faces,
        label_mode,
    )
    target_vertex_labels = labels_to_vertex_labels(
        run.target_labels,
        target_vertices_raw.shape[0],
        target_faces,
        label_mode,
    )
    deformed_face_labels = vertex_labels_to_face_labels(source_vertex_labels, deformed_faces)
    target_face_labels = vertex_labels_to_face_labels(target_vertex_labels, target_faces)
    return {
        "deformed_vertices_raw": deformed_vertices_raw,
        "deformed_faces": deformed_faces,
        "target_vertices_raw": target_vertices_raw,
        "target_faces": target_faces,
        "deformed_vertices": unit_vertices(deformed_vertices_raw),
        "target_vertices": unit_vertices(target_vertices_raw),
        "source_vertex_labels": source_vertex_labels.astype(np.int64),
        "target_vertex_labels": target_vertex_labels.astype(np.int64),
        "deformed_face_labels": deformed_face_labels.astype(np.int64),
        "target_face_labels": target_face_labels.astype(np.int64),
    }


def evaluate_run(
    run: ResolvedRun,
    out_dir: Path,
    target_samples: int,
    deformed_samples: int,
    thresholds: Sequence[float],
    success_threshold: float,
    auc_max_threshold: float,
    auc_steps: int,
    color_max_error: float,
    seed: int,
    label_mode: str,
    worst_bucket_count: int,
    render_face_limit: int,
    plt,
) -> Tuple[List[Dict[str, object]], Dict[str, object], List[Dict[str, object]], Dict[str, object]]:
    geom = load_run_geometry(run, label_mode)
    rng = np.random.default_rng(seed)

    deformed_points, _, deformed_sample_labels = sample_surface_with_labels(
        geom["deformed_vertices"],
        geom["deformed_faces"],
        deformed_samples,
        rng,
        geom["deformed_face_labels"],
    )
    target_points, _, target_sample_labels = sample_surface_with_labels(
        geom["target_vertices"],
        geom["target_faces"],
        target_samples,
        rng,
        geom["target_face_labels"],
    )
    if deformed_sample_labels is None or target_sample_labels is None:
        raise RuntimeError("sample labels unexpectedly missing")

    global_vertex_errors, global_nn = nearest_distances(geom["deformed_vertices"], target_points)
    global_vertex_nn_labels = target_sample_labels[global_nn]
    target_sample_trees = label_kdtrees(target_points, target_sample_labels)
    deformed_sample_trees = label_kdtrees(deformed_points, deformed_sample_labels)
    target_vertex_trees = label_kdtrees(geom["target_vertices"], geom["target_vertex_labels"])

    vertex_semantic_errors = same_label_distances_from_trees(
        geom["deformed_vertices"],
        geom["source_vertex_labels"],
        target_sample_trees,
    )
    deformed_sample_semantic_errors = same_label_distances_from_trees(
        deformed_points,
        deformed_sample_labels,
        target_sample_trees,
    )
    target_sample_semantic_errors = same_label_distances_from_trees(
        target_points,
        target_sample_labels,
        deformed_sample_trees,
    )

    # DenseCorr-style source-vertex error: predicted deformed source vertex to
    # nearest target vertex in the matching semantic group.
    vertex_auc_errors = same_label_distances_from_trees(
        geom["deformed_vertices"],
        geom["source_vertex_labels"],
        target_vertex_trees,
    )
    run_auc, curve_thresholds, curve_acc = auc_from_errors(vertex_auc_errors, auc_max_threshold, auc_steps)

    run_name = run.output_dir.name
    family = run.output_dir.parent.name
    safe_name = sanitize_name(run_name)
    source_name, target_name = infer_pair_names(run)
    pair_name = f"{source_name}_to_{target_name}"
    pair_dir_name = sanitize_name(pair_name, max_len=80)
    context = {
        "source_name": source_name,
        "target_name": target_name,
        "source_label": pretty_name(source_name),
        "target_label": pretty_name(target_name),
        "pair_name": pair_name,
        "pair_label": f"{pretty_name(source_name)} -> {pretty_name(target_name)}",
        "method_label": method_label_from_family(family),
    }
    heatmap_dir = out_dir / "colored_meshes"
    plot_dir = out_dir / "plots"
    run_plot_dir = plot_dir / pair_dir_name

    write_colored_ply(
        heatmap_dir / f"{safe_name}_global_vertex_error.ply",
        geom["deformed_vertices_raw"],
        geom["deformed_faces"],
        color_ramp(global_vertex_errors, color_max_error),
    )
    write_colored_ply(
        heatmap_dir / f"{safe_name}_semantic_vertex_error.ply",
        geom["deformed_vertices_raw"],
        geom["deformed_faces"],
        color_ramp(vertex_semantic_errors, color_max_error),
    )
    write_hist_plot(
        run_plot_dir / f"{safe_name}_vertex_error_hist.png",
        global_vertex_errors,
        vertex_semantic_errors,
        thresholds,
        color_max_error,
        context,
        plt,
    )
    write_cdf_plot(
        run_plot_dir / f"{safe_name}_vertex_chamfer_cdf.png",
        global_vertex_errors,
        vertex_semantic_errors,
        thresholds,
        color_max_error,
        context,
        plt,
    )
    write_vertex_error_surface_image(
        run_plot_dir / f"{safe_name}_global_vertex_chamfer_heatmap.png",
        geom["deformed_vertices"],
        geom["deformed_faces"],
        global_vertex_errors,
        "Per-vertex Chamfer distance",
        "nearest target surface distance",
        color_max_error,
        context,
        plt,
        render_face_limit,
    )
    write_vertex_error_surface_image(
        run_plot_dir / f"{safe_name}_semantic_vertex_chamfer_heatmap.png",
        geom["deformed_vertices"],
        geom["deformed_faces"],
        vertex_semantic_errors,
        "Per-vertex semantic Chamfer distance",
        "matching bucket distance",
        color_max_error,
        context,
        plt,
        render_face_limit,
    )

    primary_key = threshold_key(success_threshold)
    shared_labels = sorted(set(geom["source_vertex_labels"].tolist()) & set(geom["target_vertex_labels"].tolist()))
    bucket_rows: List[Dict[str, object]] = []
    for bucket in shared_labels:
        src_v_mask = geom["source_vertex_labels"] == bucket
        tgt_v_mask = geom["target_vertex_labels"] == bucket
        def_s_mask = deformed_sample_labels == bucket
        tgt_s_mask = target_sample_labels == bucket

        src_vertex_errors = vertex_semantic_errors[src_v_mask]
        def_s_errors = deformed_sample_semantic_errors[def_s_mask]
        tgt_s_errors = target_sample_semantic_errors[tgt_s_mask]
        precision_error_mean = finite_mean(def_s_errors)
        recall_error_mean = finite_mean(tgt_s_errors)
        chamfer_l1_sum = (
            None
            if precision_error_mean is None or recall_error_mean is None
            else precision_error_mean + recall_error_mean
        )

        row: Dict[str, object] = {
            "family": family,
            "run_name": run_name,
            "source_animal": source_name,
            "target_animal": target_name,
            "pair": pair_name,
            "bucket": int(bucket),
            "source_vertices": int(src_v_mask.sum()),
            "target_vertices": int(tgt_v_mask.sum()),
            "deformed_surface_samples": int(def_s_mask.sum()),
            "target_surface_samples": int(tgt_s_mask.sum()),
            "vertex_semantic_error_mean": finite_mean(src_vertex_errors),
            "vertex_semantic_error_p50": finite_percentile(src_vertex_errors, 50),
            "vertex_semantic_error_p90": finite_percentile(src_vertex_errors, 90),
            "vertex_semantic_error_p95": finite_percentile(src_vertex_errors, 95),
            "surface_precision_error_mean": precision_error_mean,
            "surface_recall_error_mean": recall_error_mean,
            "surface_semantic_chamfer_l1_sum": chamfer_l1_sum,
            "surface_semantic_chamfer_l1_mean": None if chamfer_l1_sum is None else chamfer_l1_sum * 0.5,
        }
        for threshold in thresholds:
            key = threshold_key(threshold)
            vertex_success = float((src_vertex_errors[np.isfinite(src_vertex_errors)] <= threshold).mean()) if np.isfinite(src_vertex_errors).any() else 0.0
            precision = float((def_s_errors[np.isfinite(def_s_errors)] <= threshold).mean()) if np.isfinite(def_s_errors).any() else 0.0
            recall = float((tgt_s_errors[np.isfinite(tgt_s_errors)] <= threshold).mean()) if np.isfinite(tgt_s_errors).any() else 0.0
            row[f"vertex_success_tau_{key}"] = vertex_success
            row[f"surface_precision_tau_{key}"] = precision
            row[f"surface_recall_tau_{key}"] = recall
            row[f"surface_fscore_tau_{key}"] = harmonic(precision, recall)

        bucket_auc, _, _ = auc_from_errors(src_vertex_errors, auc_max_threshold, auc_steps)
        row[f"vertex_semantic_auc_0_to_{threshold_key(auc_max_threshold)}"] = bucket_auc
        row["vertex_semantic_err_x100"] = None if finite_mean(src_vertex_errors) is None else finite_mean(src_vertex_errors) * 100.0
        row["global_nearest_label_agreement"] = (
            float((global_vertex_nn_labels[src_v_mask] == bucket).mean()) if src_v_mask.any() else 0.0
        )
        bucket_rows.append(row)

    write_bucket_plot(run_plot_dir / f"{safe_name}_bucket_success.png", bucket_rows, success_threshold, context, plt)

    chamfer_values = np.array(
        [
            safe_float(row.get("surface_semantic_chamfer_l1_sum"), np.nan)
            for row in bucket_rows
        ],
        dtype=np.float64,
    )
    finite_chamfer_values = chamfer_values[np.isfinite(chamfer_values)]
    worst_chamfer_sorted = sorted(
        bucket_rows,
        key=lambda row: safe_float(row.get("surface_semantic_chamfer_l1_sum"), -1.0),
        reverse=True,
    )
    worst_chamfer_rows = worst_chamfer_sorted[: max(0, worst_bucket_count)]
    worst_chamfer_buckets = [int(row["bucket"]) for row in worst_chamfer_rows]
    write_colored_ply(
        heatmap_dir / f"{safe_name}_worst_chamfer_buckets.ply",
        geom["deformed_vertices_raw"],
        geom["deformed_faces"],
        bucket_highlight_colors(geom["source_vertex_labels"], worst_chamfer_buckets),
    )
    write_worst_bucket_image(
        run_plot_dir / f"{safe_name}_worst_chamfer_buckets.png",
        geom["deformed_vertices"],
        geom["deformed_faces"],
        geom["deformed_face_labels"],
        worst_chamfer_rows,
        context,
        success_threshold,
        plt,
        render_face_limit,
    )
    write_bucket_chamfer_plot(
        run_plot_dir / f"{safe_name}_bucket_chamfer.png",
        bucket_rows,
        success_threshold,
        worst_chamfer_buckets,
        context,
        plt,
    )

    success_values = np.array([float(row[f"surface_fscore_tau_{primary_key}"]) for row in bucket_rows], dtype=np.float64)
    worst_sorted = sorted(bucket_rows, key=lambda row: float(row[f"surface_fscore_tau_{primary_key}"]))
    if success_values.size:
        spread = float(success_values.max() - success_values.min())
        std = float(success_values.std())
        min_success = float(success_values.min())
        max_success = float(success_values.max())
    else:
        spread = std = min_success = max_success = 0.0

    bad_count = int((success_values < 0.70).sum()) if success_values.size else 0
    if bad_count >= 1 and spread >= 0.25:
        discrepancy_pattern = "localized_bad_buckets"
    elif min_success >= 0.80 and spread <= 0.20:
        discrepancy_pattern = "uniformly_good"
    elif spread <= 0.20:
        discrepancy_pattern = "uniform_moderate"
    else:
        discrepancy_pattern = "mixed"

    run_summary = {
        "family": family,
        "run_name": run_name,
        "source_animal": source_name,
        "target_animal": target_name,
        "pair": pair_name,
        "plot_group_dir": display_path(run_plot_dir),
        "num_buckets": len(bucket_rows),
        f"bucket_surface_fscore_tau_{primary_key}_mean": float(success_values.mean()) if success_values.size else 0.0,
        f"bucket_surface_fscore_tau_{primary_key}_min": min_success,
        f"bucket_surface_fscore_tau_{primary_key}_max": max_success,
        f"bucket_surface_fscore_tau_{primary_key}_std": std,
        f"bucket_surface_fscore_tau_{primary_key}_spread": spread,
        f"bad_bucket_count_fscore_lt_0p70_tau_{primary_key}": bad_count,
        "worst_buckets": ",".join(str(row["bucket"]) for row in worst_sorted[:3]),
        "bucket_surface_semantic_chamfer_l1_sum_mean": float(finite_chamfer_values.mean()) if finite_chamfer_values.size else None,
        "bucket_surface_semantic_chamfer_l1_sum_min": float(finite_chamfer_values.min()) if finite_chamfer_values.size else None,
        "bucket_surface_semantic_chamfer_l1_sum_max": float(finite_chamfer_values.max()) if finite_chamfer_values.size else None,
        "bucket_surface_semantic_chamfer_l1_sum_std": float(finite_chamfer_values.std()) if finite_chamfer_values.size else None,
        "bucket_surface_semantic_chamfer_l1_sum_spread": float(finite_chamfer_values.max() - finite_chamfer_values.min()) if finite_chamfer_values.size else None,
        "worst_chamfer_buckets": ",".join(str(bucket) for bucket in worst_chamfer_buckets),
        "discrepancy_pattern": discrepancy_pattern,
        "global_vertex_error_mean": finite_mean(global_vertex_errors),
        "global_vertex_error_p95": finite_percentile(global_vertex_errors, 95),
        "semantic_vertex_error_mean": finite_mean(vertex_semantic_errors),
        "semantic_vertex_error_p95": finite_percentile(vertex_semantic_errors, 95),
        f"vertex_semantic_auc_0_to_{threshold_key(auc_max_threshold)}": run_auc,
        "vertex_semantic_err_x100": None if finite_mean(vertex_auc_errors) is None else finite_mean(vertex_auc_errors) * 100.0,
        "vertex_nearest_label_agreement": float((global_vertex_nn_labels == geom["source_vertex_labels"]).mean()),
        "global_error_ply": display_path(heatmap_dir / f"{safe_name}_global_vertex_error.ply"),
        "semantic_error_ply": display_path(heatmap_dir / f"{safe_name}_semantic_vertex_error.ply"),
        "bucket_plot": display_path(run_plot_dir / f"{safe_name}_bucket_success.png"),
        "bucket_chamfer_plot": display_path(run_plot_dir / f"{safe_name}_bucket_chamfer.png"),
        "hist_plot": display_path(run_plot_dir / f"{safe_name}_vertex_error_hist.png"),
        "vertex_chamfer_cdf_plot": display_path(run_plot_dir / f"{safe_name}_vertex_chamfer_cdf.png"),
        "global_chamfer_heatmap_image": display_path(run_plot_dir / f"{safe_name}_global_vertex_chamfer_heatmap.png"),
        "semantic_chamfer_heatmap_image": display_path(run_plot_dir / f"{safe_name}_semantic_vertex_chamfer_heatmap.png"),
        "worst_bucket_highlight_ply": display_path(heatmap_dir / f"{safe_name}_worst_chamfer_buckets.ply"),
        "worst_bucket_highlight_image": display_path(run_plot_dir / f"{safe_name}_worst_chamfer_buckets.png"),
    }

    curve_rows = [
        {
            "family": family,
            "run_name": run_name,
            "threshold": float(t),
            "accuracy": float(a),
        }
        for t, a in zip(curve_thresholds, curve_acc)
    ]

    comparability = comparability_row(run, geom)
    return bucket_rows, run_summary, curve_rows, comparability


def comparability_row(run: ResolvedRun, geom: Dict[str, object]) -> Dict[str, object]:
    source_mesh = str(run.source_mesh or "")
    target_mesh = str(run.target_mesh)
    source_labels = str(run.source_labels or "")
    target_labels = str(run.target_labels or "")
    family = run.output_dir.parent.name
    is_densecorr = "densecorr3d" in family.lower() or "densecorr3d" in source_mesh.lower() or "densecorr3d" in target_mesh.lower()
    uses_densecorr_labels = "densecorr3d" in source_labels.lower() and "densecorr3d" in target_labels.lower()
    uses_paper_simple_mesh = (
        "external/DenseCorr3D" in source_mesh
        and "external/DenseCorr3D" in target_mesh
        and source_mesh.endswith("simple_mesh.obj")
        and target_mesh.endswith("simple_mesh.obj")
    )
    source_groups = len(set(np.asarray(geom["source_vertex_labels"]).tolist()))
    target_groups = len(set(np.asarray(geom["target_vertex_labels"]).tolist()))
    geodesic_files = list(Path("external/DenseCorr3D").glob("**/*geodesic*")) + list(Path("external/DenseCorr3D").glob("**/*dist*.npy")) + list(Path("external/DenseCorr3D").glob("**/*dist*.npz"))
    geodesics_available = bool(geodesic_files)
    exact_paper_comparable = bool(is_densecorr and uses_densecorr_labels and uses_paper_simple_mesh and geodesics_available)
    project_proxy_comparable = bool(is_densecorr and uses_densecorr_labels and source_groups == target_groups)
    reason = []
    if not is_densecorr:
        reason.append("not DenseCorr3D")
    if not uses_densecorr_labels:
        reason.append("labels are not DenseCorr3D groups")
    if not uses_paper_simple_mesh:
        reason.append("uses MeshUp prepared/full/5k meshes rather than paper simple_mesh inputs")
    if not geodesics_available:
        reason.append("local DenseCorr3D copy has no geodesic distance matrices")
    reason.append("pipeline output is a deformed source mesh, not a direct point-to-point matcher output")
    return {
        "family": family,
        "run_name": run.output_dir.name,
        "is_densecorr3d_run": is_densecorr,
        "uses_densecorr_group_labels": uses_densecorr_labels,
        "source_group_count": source_groups,
        "target_group_count": target_groups,
        "uses_paper_simple_mesh": uses_paper_simple_mesh,
        "geodesic_matrices_available": geodesics_available,
        "exactly_comparable_to_paper_auc": exact_paper_comparable,
        "comparable_for_project_euclidean_semantic_auc": project_proxy_comparable,
        "reason": "; ".join(reason),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, run_rows: List[Dict[str, object]], comparability_rows: List[Dict[str, object]], auc_key: str, success_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dense = [row for row in run_rows if "densecorr3d" in str(row["family"]).lower()]
    with path.open("w") as f:
        f.write("# Bucket Discrepancy And AUC Report\n\n")
        f.write("Distances are computed after MeshUp unit-box normalization. Colored PLY meshes use blue for low error and red/magenta for high or unmatched error. Worst-bucket highlight images color the highest semantic-Chamfer buckets and leave the rest of the mesh gray.\n\n")
        f.write("## Paper Comparability Check\n\n")
        f.write("DenseMatcher/DenseCorr3D reports normalized geodesic error and AUC for point-to-point mesh correspondence. Our outputs are deformed source meshes. The local dataset copy has semantic `groups.txt` labels, but no geodesic distance matrices, so the AUC here is a Euclidean semantic deformation proxy. It is useful for comparing our runs to each other, but should not be compared numerically to the paper's Table 1 AUC values.\n\n")
        exact = sum(1 for row in comparability_rows if row["exactly_comparable_to_paper_auc"])
        proxy = sum(1 for row in comparability_rows if row["comparable_for_project_euclidean_semantic_auc"])
        f.write(f"- Exact paper-comparable rows: `{exact}`\n")
        f.write(f"- Project-comparable Euclidean semantic AUC rows: `{proxy}`\n\n")
        f.write("## DenseCorr3D Run Summary\n\n")
        write_summary_table(f, dense, auc_key, success_key)
        f.write("\n## All Run Summary\n\n")
        write_summary_table(f, run_rows, auc_key, success_key)
        f.write("\n## How To Read The Discrepancy Pattern\n\n")
        f.write("- `uniformly_good`: all buckets are high and similar.\n")
        f.write("- `uniform_moderate`: buckets are similarly imperfect, so the issue is spread across the whole shape.\n")
        f.write("- `localized_bad_buckets`: a few buckets are much worse than the others.\n")
        f.write("- `mixed`: neither cleanly uniform nor clearly localized.\n")


def write_summary_table(handle, rows: List[Dict[str, object]], auc_key: str, success_key: str) -> None:
    columns = [
        "family",
        "pair",
        "plot_group_dir",
        "run_name",
        success_key + "_mean",
        success_key + "_min",
        success_key + "_spread",
        "worst_buckets",
        "bucket_surface_semantic_chamfer_l1_sum_mean",
        "bucket_surface_semantic_chamfer_l1_sum_max",
        "worst_chamfer_buckets",
        "discrepancy_pattern",
        "semantic_vertex_error_mean",
        "semantic_vertex_error_p95",
        auc_key,
    ]
    if not rows:
        handle.write("No rows.\n")
        return
    handle.write("| " + " | ".join(columns) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        handle.write("| " + " | ".join(format_cell(row.get(col)) for col in columns) + " |\n")


def format_cell(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Run output directories or final mesh paths.")
    parser.add_argument("--paths-file", help="Newline-delimited run paths.")
    parser.add_argument("--output-dir", default="jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy")
    parser.add_argument("--target-samples", type=int, default=60000)
    parser.add_argument("--deformed-samples", type=int, default=60000)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--success-threshold", type=float, default=0.02)
    parser.add_argument("--auc-max-threshold", type=float, default=0.05)
    parser.add_argument("--auc-steps", type=int, default=101)
    parser.add_argument("--color-max-error", type=float, default=0.05)
    parser.add_argument("--worst-bucket-count", type=int, default=3)
    parser.add_argument("--render-face-limit", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--partfield-label-mode", choices=["auto", "vertex", "face"], default="auto")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def collect_paths(args: argparse.Namespace) -> List[str]:
    paths = list(args.paths)
    if args.paths_file:
        with resolve_path(args.paths_file).open("r") as f:
            paths.extend(line.strip() for line in f if line.strip() and not line.lstrip().startswith("#"))
    if not paths:
        raise SystemExit("Provide at least one path or --paths-file.")
    return paths


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_paths(args)
    plt = None if args.skip_plots else maybe_import_pyplot()

    all_bucket_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []
    all_curve_rows: List[Dict[str, object]] = []
    comparability_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for idx, path in enumerate(paths):
        try:
            run = resolve_run(path)
            print(f"[{idx + 1}/{len(paths)}] Evaluating buckets for {run.output_dir.name}")
            bucket_rows, run_summary, curve_rows, comparability = evaluate_run(
                run=run,
                out_dir=out_dir,
                target_samples=args.target_samples,
                deformed_samples=args.deformed_samples,
                thresholds=args.thresholds,
                success_threshold=args.success_threshold,
                auc_max_threshold=args.auc_max_threshold,
                auc_steps=args.auc_steps,
                color_max_error=args.color_max_error,
                seed=args.seed + idx * 1009,
                label_mode=args.partfield_label_mode,
                worst_bucket_count=args.worst_bucket_count,
                render_face_limit=args.render_face_limit,
                plt=plt,
            )
            all_bucket_rows.extend(bucket_rows)
            run_rows.append(run_summary)
            all_curve_rows.extend(curve_rows)
            comparability_rows.append(comparability)
        except Exception as exc:
            print(f"Warning: failed {path}: {exc}")
            failures.append({"path": path, "error": str(exc)})

    success_prefix = f"bucket_surface_fscore_tau_{threshold_key(args.success_threshold)}"
    auc_key = f"vertex_semantic_auc_0_to_{threshold_key(args.auc_max_threshold)}"

    write_csv(out_dir / "per_bucket_success.csv", all_bucket_rows)
    write_csv(out_dir / "run_bucket_summary.csv", run_rows)
    write_csv(out_dir / "auc_curves.csv", all_curve_rows)
    write_csv(out_dir / "paper_comparability.csv", comparability_rows)
    write_csv(out_dir / "failures.csv", failures)
    write_report(out_dir / "bucket_discrepancy_report.md", run_rows, comparability_rows, auc_key, success_prefix)
    with (out_dir / "bucket_discrepancy_metrics.json").open("w") as f:
        json.dump(
            {
                "per_bucket_success": all_bucket_rows,
                "run_bucket_summary": run_rows,
                "auc_curves": all_curve_rows,
                "paper_comparability": comparability_rows,
                "failures": failures,
            },
            f,
            indent=2,
        )

    print(f"Saved report: {out_dir / 'bucket_discrepancy_report.md'}")
    print(f"Saved per-bucket CSV: {out_dir / 'per_bucket_success.csv'}")
    print(f"Saved run summary CSV: {out_dir / 'run_bucket_summary.csv'}")
    print(f"Saved colored PLYs under: {out_dir / 'colored_meshes'}")
    print(f"Saved plots under: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
