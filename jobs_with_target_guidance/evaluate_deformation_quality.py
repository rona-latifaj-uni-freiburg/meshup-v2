#!/usr/bin/env python3
"""Batch quantitative evaluation for target-guided animal deformations.

The script compares a final deformed mesh against the real target mesh stored
in each run's config.yml.  It reports:

* global surface geometry: Chamfer, robust Hausdorff percentiles, normals,
  and F-scores at unit-box thresholds;
* DenseCorr3D/PartField semantic-region geometry: distance from each deformed
  source semantic group to the matching target semantic group;
* nearest-neighbor semantic agreement: whether nearest target surface points
  have the same annotated group;
* source distortion: edge stretch, Laplacian change, area change, and flips;
* before/after improvement relative to the undeformed source mesh.

All positions are normalized with MeshUp's unit-box convention before scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs_with_target_guidance.evaluate_target_pipeline import (  # noqa: E402
    face_areas_and_normals,
    labels_to_vertex_labels,
    load_mesh,
    source_distortion_metrics,
    unique_edges,
    unit_vertices,
    vertex_labels_to_face_labels,
)


ArrayPair = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ResolvedRun:
    output_dir: Path
    deformed_mesh: Path
    source_mesh: Optional[Path]
    target_mesh: Path
    source_labels: Optional[Path]
    target_labels: Optional[Path]
    config: Dict[str, object]


def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def infer_output_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    if p.is_dir():
        return p
    if p.parent.name in {"colored_meshes", "mesh_final"}:
        return p.parent.parent
    return p.parent


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


def resolve_run(path: str | Path) -> ResolvedRun:
    output_dir = infer_output_dir(path)
    config_path = output_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yml for {output_dir}")

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
            source_labels = label_path_from_dir(label_dirs[-1], str(source_labels) if source_labels else None)
            target_labels = label_path_from_dir(label_dirs[-1], str(target_labels) if target_labels else None)

    deformed_default = output_dir / "mesh_final" / "mesh.obj"
    if Path(path).is_file():
        deformed_default = resolve_path(path)

    target_mesh = cfg.get("target_mesh")
    if not target_mesh:
        raise ValueError(f"{output_dir}: config.yml does not contain target_mesh")

    source_mesh = cfg.get("mesh")
    return ResolvedRun(
        output_dir=output_dir,
        deformed_mesh=resolve_path(deformed_default),
        source_mesh=resolve_path(str(source_mesh)) if source_mesh else None,
        target_mesh=resolve_path(str(target_mesh)),
        source_labels=resolve_path(str(source_labels)) if source_labels else None,
        target_labels=resolve_path(str(target_labels)) if target_labels else None,
        config=cfg,
    )


def sample_surface_with_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
    face_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if faces.size == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty, np.empty((0,), dtype=np.int64) if face_labels is not None else None

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
    sampled_labels = face_labels[face_idx].astype(np.int64) if face_labels is not None else None
    return points, normals, sampled_labels


def nearest_distances(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(tgt)
    dists, indices = tree.query(src, workers=-1)
    return dists.astype(np.float64), indices.astype(np.int64)


def percentile_metrics(prefix: str, values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_p50": float(np.percentile(finite, 50)),
        f"{prefix}_p90": float(np.percentile(finite, 90)),
        f"{prefix}_p95": float(np.percentile(finite, 95)),
        f"{prefix}_p99": float(np.percentile(finite, 99)),
        f"{prefix}_max": float(finite.max(initial=0.0)),
    }


def geometry_metrics(
    pred_points: np.ndarray,
    pred_normals: np.ndarray,
    tgt_points: np.ndarray,
    tgt_normals: np.ndarray,
    thresholds: Sequence[float],
) -> Dict[str, float]:
    pred_to_tgt, pred_nn = nearest_distances(pred_points, tgt_points)
    tgt_to_pred, tgt_nn = nearest_distances(tgt_points, pred_points)
    combined = np.concatenate([pred_to_tgt, tgt_to_pred])

    metrics: Dict[str, float] = {
        "chamfer_l1_sum": float(pred_to_tgt.mean() + tgt_to_pred.mean()),
        "chamfer_l1_mean": float(0.5 * (pred_to_tgt.mean() + tgt_to_pred.mean())),
        "chamfer_l2_sq_sum": float(np.square(pred_to_tgt).mean() + np.square(tgt_to_pred).mean()),
        "pred_to_tgt_l1": float(pred_to_tgt.mean()),
        "tgt_to_pred_l1": float(tgt_to_pred.mean()),
        "hausdorff_l1": float(max(pred_to_tgt.max(initial=0.0), tgt_to_pred.max(initial=0.0))),
        "robust_hausdorff_p95": float(max(np.percentile(pred_to_tgt, 95), np.percentile(tgt_to_pred, 95))),
        "robust_hausdorff_p99": float(max(np.percentile(pred_to_tgt, 99), np.percentile(tgt_to_pred, 99))),
    }
    metrics.update(percentile_metrics("surface_l1", combined))

    if pred_normals.size and tgt_normals.size:
        pred_dot = np.abs((pred_normals * tgt_normals[pred_nn]).sum(axis=1))
        tgt_dot = np.abs((tgt_normals * pred_normals[tgt_nn]).sum(axis=1))
        metrics["normal_consistency"] = float(0.5 * (pred_dot.mean() + tgt_dot.mean()))

    for threshold in thresholds:
        precision = float((pred_to_tgt <= threshold).mean())
        recall = float((tgt_to_pred <= threshold).mean())
        denom = precision + recall
        fscore = 0.0 if denom <= 0 else 2.0 * precision * recall / denom
        key = threshold_key(threshold)
        metrics[f"precision_tau_{key}"] = precision
        metrics[f"recall_tau_{key}"] = recall
        metrics[f"fscore_tau_{key}"] = float(fscore)

    return metrics


def threshold_key(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p")


def same_label_distances(
    src_points: np.ndarray,
    src_labels: np.ndarray,
    tgt_points: np.ndarray,
    tgt_labels: np.ndarray,
) -> np.ndarray:
    dists = np.full(src_points.shape[0], np.inf, dtype=np.float64)
    for label in sorted(set(src_labels.tolist())):
        src_mask = src_labels == label
        tgt_mask = tgt_labels == label
        if not src_mask.any() or not tgt_mask.any():
            continue
        tree = cKDTree(tgt_points[tgt_mask])
        dists[src_mask] = tree.query(src_points[src_mask], workers=-1)[0]
    return dists


def nearest_label_agreement(
    src_points: np.ndarray,
    src_labels: np.ndarray,
    tgt_points: np.ndarray,
    tgt_labels: np.ndarray,
) -> Tuple[float, float]:
    if src_points.size == 0 or tgt_points.size == 0:
        return 0.0, 0.0
    tree = cKDTree(tgt_points)
    _, nn = tree.query(src_points, workers=-1)
    agreement = tgt_labels[nn] == src_labels
    per_label = []
    for label in sorted(set(src_labels.tolist())):
        mask = src_labels == label
        if mask.any():
            per_label.append(float(agreement[mask].mean()))
    balanced = float(np.mean(per_label)) if per_label else 0.0
    return float(agreement.mean()), balanced


def semantic_group_metrics(
    pred_points: np.ndarray,
    pred_labels: Optional[np.ndarray],
    tgt_points: np.ndarray,
    tgt_labels: Optional[np.ndarray],
    thresholds: Sequence[float],
) -> Dict[str, object]:
    if pred_labels is None or tgt_labels is None:
        return {"available": 0.0, "reason": "missing source or target labels"}
    if pred_labels.shape[0] != pred_points.shape[0] or tgt_labels.shape[0] != tgt_points.shape[0]:
        return {"available": 0.0, "reason": "sample label counts do not match point counts"}

    pred_to_tgt = same_label_distances(pred_points, pred_labels, tgt_points, tgt_labels)
    tgt_to_pred = same_label_distances(tgt_points, tgt_labels, pred_points, pred_labels)
    combined = np.concatenate([pred_to_tgt[np.isfinite(pred_to_tgt)], tgt_to_pred[np.isfinite(tgt_to_pred)]])
    if combined.size == 0:
        return {"available": 0.0, "reason": "no shared semantic labels"}

    pred_agree, pred_balanced = nearest_label_agreement(pred_points, pred_labels, tgt_points, tgt_labels)
    tgt_agree, tgt_balanced = nearest_label_agreement(tgt_points, tgt_labels, pred_points, pred_labels)
    shared_labels = sorted(set(pred_labels.tolist()) & set(tgt_labels.tolist()))

    metrics: Dict[str, object] = {
        "available": 1.0,
        "shared_labels": int(len(shared_labels)),
        "pred_to_tgt_same_label_l1": float(np.mean(pred_to_tgt[np.isfinite(pred_to_tgt)])),
        "tgt_to_pred_same_label_l1": float(np.mean(tgt_to_pred[np.isfinite(tgt_to_pred)])),
        "semantic_chamfer_l1_sum": float(
            np.mean(pred_to_tgt[np.isfinite(pred_to_tgt)]) + np.mean(tgt_to_pred[np.isfinite(tgt_to_pred)])
        ),
        "semantic_chamfer_l1_mean": float(
            0.5 * (np.mean(pred_to_tgt[np.isfinite(pred_to_tgt)]) + np.mean(tgt_to_pred[np.isfinite(tgt_to_pred)]))
        ),
        "semantic_chamfer_l2_sq_sum": float(
            np.square(pred_to_tgt[np.isfinite(pred_to_tgt)]).mean()
            + np.square(tgt_to_pred[np.isfinite(tgt_to_pred)]).mean()
        ),
        "nearest_label_agreement": float(0.5 * (pred_agree + tgt_agree)),
        "nearest_label_agreement_pred_to_tgt": pred_agree,
        "nearest_label_agreement_tgt_to_pred": tgt_agree,
        "balanced_nearest_label_agreement": float(0.5 * (pred_balanced + tgt_balanced)),
    }
    metrics.update(percentile_metrics("same_label_l1", combined))

    for threshold in thresholds:
        precision = float((pred_to_tgt[np.isfinite(pred_to_tgt)] <= threshold).mean())
        recall = float((tgt_to_pred[np.isfinite(tgt_to_pred)] <= threshold).mean())
        denom = precision + recall
        fscore = 0.0 if denom <= 0 else 2.0 * precision * recall / denom
        key = threshold_key(threshold)
        metrics[f"semantic_precision_tau_{key}"] = precision
        metrics[f"semantic_recall_tau_{key}"] = recall
        metrics[f"semantic_fscore_tau_{key}"] = float(fscore)

    auc_limit = max(thresholds) if thresholds else 0.05
    auc_thresholds = np.linspace(0.0, auc_limit, num=101)
    accuracies = np.array([(combined <= t).mean() for t in auc_thresholds], dtype=np.float64)
    metrics[f"semantic_pck_auc_0_to_{threshold_key(auc_limit)}"] = float(np.trapz(accuracies, auc_thresholds) / auc_limit)
    return metrics


def extended_source_distortion_metrics(
    source_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
    faces: np.ndarray,
) -> Dict[str, object]:
    metrics: Dict[str, object] = source_distortion_metrics(source_vertices, deformed_vertices, faces)
    if source_vertices.shape != deformed_vertices.shape:
        return metrics

    displacement = np.linalg.norm(deformed_vertices - source_vertices, axis=1)
    metrics.update(percentile_metrics("vertex_displacement_l1", displacement))

    edges = unique_edges(faces)
    src_len = np.linalg.norm(source_vertices[edges[:, 0]] - source_vertices[edges[:, 1]], axis=1)
    def_len = np.linalg.norm(deformed_vertices[edges[:, 0]] - deformed_vertices[edges[:, 1]], axis=1)
    valid_edges = src_len > 1e-12
    if valid_edges.any():
        rel = def_len[valid_edges] / src_len[valid_edges]
        metrics["edge_length_ratio_p50"] = float(np.percentile(rel, 50))
        metrics["edge_length_ratio_p95"] = float(np.percentile(rel, 95))
        metrics["edge_length_ratio_p99"] = float(np.percentile(rel, 99))

    src_area, src_normals = face_areas_and_normals(source_vertices, faces)
    def_area, def_normals = face_areas_and_normals(deformed_vertices, faces)
    valid_faces = src_area > 1e-12
    if valid_faces.any():
        ratio = def_area[valid_faces] / src_area[valid_faces]
        log_ratio_abs = np.abs(np.log(np.maximum(ratio, 1e-12)))
        metrics.update(percentile_metrics("face_area_abs_log_ratio", log_ratio_abs))
        metrics["severe_area_change_fraction_4x"] = float((log_ratio_abs > math.log(4.0)).mean())
        normal_dot = (src_normals[valid_faces] * def_normals[valid_faces]).sum(axis=1)
        metrics["face_normal_flip_fraction"] = float((normal_dot < 0.0).mean())
    metrics["deformed_degenerate_face_fraction"] = float((def_area <= 1e-12).mean())
    return metrics


def load_face_labels(
    labels_path: Optional[Path],
    n_vertices: int,
    faces: np.ndarray,
    label_mode: str,
) -> Optional[np.ndarray]:
    if labels_path is None:
        return None
    vertex_labels = labels_to_vertex_labels(labels_path, n_vertices, faces, label_mode)
    return vertex_labels_to_face_labels(vertex_labels, faces)


def evaluate_run(run: ResolvedRun, samples: int, thresholds: Sequence[float], seed: int, label_mode: str) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    deformed_vertices_raw, deformed_faces = load_mesh(run.deformed_mesh)
    target_vertices_raw, target_faces = load_mesh(run.target_mesh)
    source_vertices_raw: Optional[np.ndarray] = None
    source_faces: Optional[np.ndarray] = None
    if run.source_mesh is not None and run.source_mesh.exists():
        source_vertices_raw, source_faces = load_mesh(run.source_mesh)

    deformed_vertices = unit_vertices(deformed_vertices_raw)
    target_vertices = unit_vertices(target_vertices_raw)
    source_vertices = unit_vertices(source_vertices_raw) if source_vertices_raw is not None else None

    source_face_labels = load_face_labels(run.source_labels, deformed_vertices.shape[0], deformed_faces, label_mode)
    target_face_labels = load_face_labels(run.target_labels, target_vertices.shape[0], target_faces, label_mode)

    deformed_points, deformed_normals, deformed_sample_labels = sample_surface_with_labels(
        deformed_vertices,
        deformed_faces,
        samples,
        rng,
        source_face_labels,
    )
    target_points, target_normals, target_sample_labels = sample_surface_with_labels(
        target_vertices,
        target_faces,
        samples,
        rng,
        target_face_labels,
    )

    result: Dict[str, object] = {
        "run_name": run.output_dir.name,
        "family": run.output_dir.parent.name,
        "output_dir": str(run.output_dir.relative_to(PROJECT_ROOT) if run.output_dir.is_relative_to(PROJECT_ROOT) else run.output_dir),
        "deformed_mesh": str(run.deformed_mesh),
        "source_mesh": str(run.source_mesh) if run.source_mesh else None,
        "target_mesh": str(run.target_mesh),
        "source_labels": str(run.source_labels) if run.source_labels else None,
        "target_labels": str(run.target_labels) if run.target_labels else None,
        "samples": int(samples),
        "thresholds": [float(t) for t in thresholds],
        "global": geometry_metrics(deformed_points, deformed_normals, target_points, target_normals, thresholds),
        "semantic": semantic_group_metrics(
            deformed_points,
            deformed_sample_labels,
            target_points,
            target_sample_labels,
            thresholds,
        ),
    }

    if source_vertices is not None and source_faces is not None:
        if source_vertices.shape == deformed_vertices.shape:
            distortion_faces = deformed_faces if np.array_equal(source_faces, deformed_faces) else source_faces
            result["source_distortion"] = extended_source_distortion_metrics(
                source_vertices,
                deformed_vertices,
                distortion_faces,
            )
        else:
            result["source_distortion"] = {
                "source_distortion_available": 0.0,
                "source_distortion_reason": "source/deformed vertex counts differ",
            }

        source_labels_for_source_mesh = load_face_labels(run.source_labels, source_vertices.shape[0], source_faces, label_mode)
        source_points, source_normals, source_sample_labels = sample_surface_with_labels(
            source_vertices,
            source_faces,
            samples,
            rng,
            source_labels_for_source_mesh,
        )
        source_global = geometry_metrics(source_points, source_normals, target_points, target_normals, thresholds)
        source_semantic = semantic_group_metrics(
            source_points,
            source_sample_labels,
            target_points,
            target_sample_labels,
            thresholds,
        )
        result["source_to_target_baseline"] = {
            "global": source_global,
            "semantic": source_semantic,
        }
        result["improvement"] = improvement_metrics(result["global"], result["semantic"], source_global, source_semantic)

    return result


def pct_reduction(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None or abs(before) < 1e-12:
        return None
    return float(100.0 * (before - after) / before)


def scalar(mapping: object, key: str) -> Optional[float]:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def improvement_metrics(
    final_global: object,
    final_semantic: object,
    source_global: object,
    source_semantic: object,
) -> Dict[str, Optional[float]]:
    return {
        "global_chamfer_l1_sum_reduction_pct": pct_reduction(
            scalar(source_global, "chamfer_l1_sum"),
            scalar(final_global, "chamfer_l1_sum"),
        ),
        "global_robust_hausdorff_p95_reduction_pct": pct_reduction(
            scalar(source_global, "robust_hausdorff_p95"),
            scalar(final_global, "robust_hausdorff_p95"),
        ),
        "semantic_chamfer_l1_sum_reduction_pct": pct_reduction(
            scalar(source_semantic, "semantic_chamfer_l1_sum"),
            scalar(final_semantic, "semantic_chamfer_l1_sum"),
        ),
    }


def row_for_result(result: Dict[str, object], primary_threshold: float) -> Dict[str, object]:
    global_metrics = result.get("global", {})
    semantic_metrics = result.get("semantic", {})
    distortion = result.get("source_distortion", {})
    improvement = result.get("improvement", {})
    key = threshold_key(primary_threshold)
    return {
        "family": result.get("family"),
        "run_name": result.get("run_name"),
        "global_chamfer_l1_sum": scalar(global_metrics, "chamfer_l1_sum"),
        "global_chamfer_l2_sq_sum": scalar(global_metrics, "chamfer_l2_sq_sum"),
        "global_fscore": scalar(global_metrics, f"fscore_tau_{key}"),
        "global_normal_consistency": scalar(global_metrics, "normal_consistency"),
        "global_robust_hausdorff_p95": scalar(global_metrics, "robust_hausdorff_p95"),
        "semantic_chamfer_l1_sum": scalar(semantic_metrics, "semantic_chamfer_l1_sum"),
        "semantic_fscore": scalar(semantic_metrics, f"semantic_fscore_tau_{key}"),
        "semantic_nn_agreement": scalar(semantic_metrics, "nearest_label_agreement"),
        "semantic_p95": scalar(semantic_metrics, "same_label_l1_p95"),
        "edge_relative_p95": scalar(distortion, "edge_length_relative_p95"),
        "area_abs_log_ratio_p95": scalar(distortion, "face_area_abs_log_ratio_p95"),
        "normal_flip_fraction": scalar(distortion, "face_normal_flip_fraction"),
        "global_chamfer_reduction_pct": scalar(improvement, "global_chamfer_l1_sum_reduction_pct"),
        "semantic_chamfer_reduction_pct": scalar(improvement, "semantic_chamfer_l1_sum_reduction_pct"),
        "output_dir": result.get("output_dir"),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, rows: List[Dict[str, object]], primary_threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dense_rows = [row for row in rows if "densecorr3d" in str(row.get("family", "")).lower()]
    with path.open("w") as f:
        f.write("# Deformation Quality Metrics\n\n")
        f.write("All meshes are normalized to MeshUp unit-box coordinates before scoring.\n\n")
        f.write("See `metric_methodology.md` for the metric definitions and literature basis.\n\n")
        f.write("## Metric Reading\n\n")
        f.write("- Lower is better: Chamfer, robust Hausdorff p95, semantic p95, edge/area distortion, normal flips.\n")
        f.write("- Higher is better: F-score, normal consistency, semantic nearest-neighbor agreement, reduction percentages.\n")
        f.write("- DenseCorr3D semantic metrics use matching annotated groups from `groups.txt`/prepared label arrays.\n\n")
        f.write(f"Primary F-score threshold: `{primary_threshold:g}` in unit-box coordinates.\n\n")
        f.write("## DenseCorr3D Runs\n\n")
        write_markdown_table(f, dense_rows)
        f.write("\n## All Requested Runs\n\n")
        write_markdown_table(f, rows)


def write_markdown_table(handle, rows: List[Dict[str, object]]) -> None:
    columns = [
        "family",
        "run_name",
        "global_chamfer_l1_sum",
        "global_fscore",
        "global_normal_consistency",
        "semantic_chamfer_l1_sum",
        "semantic_fscore",
        "semantic_nn_agreement",
        "edge_relative_p95",
        "global_chamfer_reduction_pct",
        "semantic_chamfer_reduction_pct",
    ]
    if not rows:
        handle.write("No rows.\n")
        return
    handle.write("| " + " | ".join(columns) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        values = [fmt(row.get(col)) for col in columns]
        handle.write("| " + " | ".join(values) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Output directories or final mesh paths. Final meshes under colored_meshes/ or mesh_final/ infer the run directory.",
    )
    parser.add_argument("--paths-file", help="Optional newline-delimited list of output directories or final mesh paths.")
    parser.add_argument("--samples", type=int, default=20000, help="Surface samples per mesh.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--primary-threshold", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--partfield-label-mode", choices=["auto", "vertex", "face"], default="auto")
    parser.add_argument(
        "--output-dir",
        default="jobs_with_target_guidance/cross_animal_spike_runs/evaluation/deformation_quality",
        help="Directory for metrics.json, summary.csv, and summary.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = list(args.paths)
    if args.paths_file:
        with resolve_path(args.paths_file).open("r") as f:
            paths.extend(line.strip() for line in f if line.strip() and not line.lstrip().startswith("#"))
    if not paths:
        raise SystemExit("Provide at least one path or --paths-file.")

    results: List[Dict[str, object]] = []
    for idx, path in enumerate(paths):
        run = resolve_run(path)
        print(f"[{idx + 1}/{len(paths)}] Evaluating {run.output_dir}")
        result = evaluate_run(
            run,
            samples=args.samples,
            thresholds=args.thresholds,
            seed=args.seed + idx * 1009,
            label_mode=args.partfield_label_mode,
        )
        results.append(result)

    rows = [row_for_result(result, args.primary_threshold) for result in results]
    rows.sort(key=lambda row: (str(row.get("family")), str(row.get("run_name"))))

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2)
    write_csv(out_dir / "summary.csv", rows)
    write_markdown(out_dir / "summary.md", rows, args.primary_threshold)

    print(f"Saved JSON: {out_dir / 'metrics.json'}")
    print(f"Saved CSV:  {out_dir / 'summary.csv'}")
    print(f"Saved MD:   {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
