"""Post-process a clean topomatch deformation with semantic vertex indexing.

This script does not optimize or move vertices. It builds a one-to-one semantic
assignment from source vertices to target vertices, then rewrites the already
deformed topomatch mesh under that vertex permutation.

The output OBJ has the same geometry/connectivity as the input deformed OBJ,
up to a pure vertex renumbering. Vertex index j in the reindexed OBJ is the
source vertex assigned to semantic target vertex j.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs_with_target_guidance.partfield_chamfer import (
    FeatureMode,
    LabelMode,
    coerce_features_to_vertices,
    coerce_labels_to_vertices,
    normalize_rows,
    vertex_normals_np,
)
from jobs_with_target_guidance.semantic_vertex_correspondence import load_obj, write_obj


@dataclass
class SemanticPermutation:
    source_to_target: np.ndarray
    target_to_source: np.ndarray
    costs: np.ndarray
    similarities: np.ndarray
    source_labels: Optional[np.ndarray]
    target_labels: Optional[np.ndarray]
    metadata: Dict[str, object]


def _unit_vertices_np(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    scale = 2.0 / max(float((vmax - vmin).max()), 1e-12)
    return (vertices - (vmax + vmin) * 0.5) * scale


def _bbox_coords_np(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    return (vertices - vmin) / np.maximum(vmax - vmin, 1e-8)


def _descriptor_matrix(
    semantic_features: np.ndarray,
    vertices: np.ndarray,
    normals: np.ndarray,
    semantic_weight: float,
    position_weight: float,
    normal_weight: float,
) -> np.ndarray:
    pieces = []
    if semantic_weight > 0:
        pieces.append(normalize_rows(semantic_features) * float(semantic_weight))
    if position_weight > 0:
        pieces.append(_bbox_coords_np(vertices).astype(np.float32) * float(position_weight))
    if normal_weight > 0:
        pieces.append(normalize_rows(normals) * float(normal_weight))
    if not pieces:
        raise ValueError("At least one descriptor weight must be positive.")
    return np.concatenate(pieces, axis=1).astype(np.float32)


def _candidate_costs(
    source_idx: int,
    candidates: np.ndarray,
    base_costs: np.ndarray,
    source_labels: Optional[np.ndarray],
    target_labels: Optional[np.ndarray],
    label_mismatch_penalty: float,
    topology_prior_weight: float,
) -> np.ndarray:
    costs = base_costs.astype(np.float32, copy=True)
    if source_labels is not None and target_labels is not None and label_mismatch_penalty > 0:
        mismatch = target_labels[candidates] != int(source_labels[source_idx])
        costs[mismatch] += float(label_mismatch_penalty)
    if topology_prior_weight > 0:
        costs[candidates != int(source_idx)] += float(topology_prior_weight)
    return costs


def _assign_leftovers(
    source_to_target: np.ndarray,
    costs: np.ndarray,
    source_desc: np.ndarray,
    target_desc: np.ndarray,
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: Optional[np.ndarray],
    target_labels: Optional[np.ndarray],
    label_mismatch_penalty: float,
    topology_prior_weight: float,
) -> None:
    unused_targets = set(np.where(np.bincount(source_to_target[source_to_target >= 0], minlength=target_desc.shape[0]) == 0)[0].tolist())
    leftover_sources = np.where(source_to_target < 0)[0]
    for src_idx in leftover_sources:
        if not unused_targets:
            break
        tgt_arr = np.fromiter(unused_targets, dtype=np.int64)
        delta = target_desc[tgt_arr] - source_desc[src_idx]
        base = np.einsum("ij,ij->i", delta, delta).astype(np.float32)
        adjusted = _candidate_costs(
            int(src_idx),
            tgt_arr,
            base,
            source_labels,
            target_labels,
            label_mismatch_penalty,
            topology_prior_weight,
        )
        best_pos = int(np.argmin(adjusted))
        best_tgt = int(tgt_arr[best_pos])
        source_to_target[src_idx] = best_tgt
        costs[src_idx] = float(adjusted[best_pos])
        unused_targets.remove(best_tgt)


def build_semantic_permutation(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    source_features_path: str,
    target_features_path: str,
    source_labels_path: Optional[str],
    target_labels_path: Optional[str],
    feature_mode: FeatureMode = "auto",
    label_mode: LabelMode = "auto",
    semantic_weight: float = 1.0,
    position_weight: float = 0.15,
    normal_weight: float = 0.05,
    label_mismatch_penalty: float = 0.35,
    topology_prior_weight: float = 0.02,
    topk: int = 256,
) -> SemanticPermutation:
    n_src = int(source_vertices.shape[0])
    n_tgt = int(target_vertices.shape[0])
    if n_src != n_tgt:
        raise ValueError(
            "Semantic reindexing needs equal vertex counts for a bijective OBJ permutation. "
            f"Got source={n_src}, target={n_tgt}."
        )

    src_f_t = torch.tensor(source_faces, dtype=torch.long)
    tgt_f_t = torch.tensor(target_faces, dtype=torch.long)
    source_features = normalize_rows(coerce_features_to_vertices(source_features_path, n_src, src_f_t, feature_mode))
    target_features = normalize_rows(coerce_features_to_vertices(target_features_path, n_tgt, tgt_f_t, feature_mode))

    source_labels = None
    target_labels = None
    if source_labels_path and target_labels_path:
        source_labels = coerce_labels_to_vertices(source_labels_path, n_src, src_f_t, label_mode)
        target_labels = coerce_labels_to_vertices(target_labels_path, n_tgt, tgt_f_t, label_mode)

    src_unit = _unit_vertices_np(source_vertices.astype(np.float32))
    tgt_unit = _unit_vertices_np(target_vertices.astype(np.float32))
    src_normals = vertex_normals_np(torch.tensor(source_vertices, dtype=torch.float32), src_f_t)
    tgt_normals = vertex_normals_np(torch.tensor(target_vertices, dtype=torch.float32), tgt_f_t)

    source_desc = _descriptor_matrix(
        source_features,
        src_unit,
        src_normals,
        semantic_weight=semantic_weight,
        position_weight=position_weight,
        normal_weight=normal_weight,
    )
    target_desc = _descriptor_matrix(
        target_features,
        tgt_unit,
        tgt_normals,
        semantic_weight=semantic_weight,
        position_weight=position_weight,
        normal_weight=normal_weight,
    )

    k = min(max(int(topk), 8), n_tgt)
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(target_desc)
    dists, candidates = nn.kneighbors(source_desc, return_distance=True)
    base_costs = np.square(dists).astype(np.float32)

    best_costs = np.empty(n_src, dtype=np.float32)
    margins = np.zeros(n_src, dtype=np.float32)
    ordered_candidates = []
    ordered_costs = []
    for src_idx in range(n_src):
        adjusted = _candidate_costs(
            src_idx,
            candidates[src_idx],
            base_costs[src_idx],
            source_labels,
            target_labels,
            label_mismatch_penalty,
            topology_prior_weight,
        )
        order = np.argsort(adjusted)
        cand = candidates[src_idx][order]
        cost = adjusted[order]
        ordered_candidates.append(cand)
        ordered_costs.append(cost)
        best_costs[src_idx] = float(cost[0])
        margins[src_idx] = float(cost[1] - cost[0]) if cost.shape[0] > 1 else 0.0

    source_order = np.lexsort((-margins, best_costs))
    source_to_target = np.full(n_src, -1, dtype=np.int64)
    costs = np.full(n_src, np.inf, dtype=np.float32)
    used_targets = np.zeros(n_tgt, dtype=bool)

    for src_idx in source_order:
        cand = ordered_candidates[int(src_idx)]
        cost = ordered_costs[int(src_idx)]
        available = ~used_targets[cand]
        if not available.any():
            continue
        first = int(np.argmax(available))
        tgt_idx = int(cand[first])
        source_to_target[int(src_idx)] = tgt_idx
        costs[int(src_idx)] = float(cost[first])
        used_targets[tgt_idx] = True

    _assign_leftovers(
        source_to_target,
        costs,
        source_desc,
        target_desc,
        source_features,
        target_features,
        source_labels,
        target_labels,
        label_mismatch_penalty,
        topology_prior_weight,
    )

    if np.any(source_to_target < 0):
        raise RuntimeError("Failed to assign every source vertex to a unique target vertex.")
    if np.unique(source_to_target).shape[0] != n_tgt:
        raise RuntimeError("Semantic assignment is not bijective.")

    target_to_source = np.empty(n_tgt, dtype=np.int64)
    target_to_source[source_to_target] = np.arange(n_src, dtype=np.int64)
    similarities = np.einsum("ij,ij->i", source_features, target_features[source_to_target]).astype(np.float32)

    metadata: Dict[str, object] = {
        "n_vertices": n_src,
        "source_features_path": str(source_features_path),
        "target_features_path": str(target_features_path),
        "source_labels_path": str(source_labels_path or ""),
        "target_labels_path": str(target_labels_path or ""),
        "semantic_weight": float(semantic_weight),
        "position_weight": float(position_weight),
        "normal_weight": float(normal_weight),
        "label_mismatch_penalty": float(label_mismatch_penalty),
        "topology_prior_weight": float(topology_prior_weight),
        "topk": int(k),
        "identity_fraction": float((source_to_target == np.arange(n_src)).mean()),
        "mean_similarity": float(similarities.mean()),
        "p05_similarity": float(np.percentile(similarities, 5)),
        "mean_cost": float(np.mean(costs)),
        "p95_cost": float(np.percentile(costs, 95)),
    }
    if source_labels is not None and target_labels is not None:
        metadata["label_agreement_fraction"] = float((source_labels == target_labels[source_to_target]).mean())

    return SemanticPermutation(
        source_to_target=source_to_target,
        target_to_source=target_to_source,
        costs=costs,
        similarities=similarities,
        source_labels=source_labels,
        target_labels=target_labels,
        metadata=metadata,
    )


def write_mapping_csv(path: Path, perm: SemanticPermutation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_index", "semantic_target_index", "cost", "similarity", "source_label", "target_label"])
        for src_idx, tgt_idx in enumerate(perm.source_to_target):
            src_label = "" if perm.source_labels is None else int(perm.source_labels[src_idx])
            tgt_label = "" if perm.target_labels is None else int(perm.target_labels[int(tgt_idx)])
            writer.writerow([src_idx, int(tgt_idx), float(perm.costs[src_idx]), float(perm.similarities[src_idx]), src_label, tgt_label])


def save_outputs(
    output_dir: Path,
    deformed_vertices: np.ndarray,
    deformed_faces: np.ndarray,
    perm: SemanticPermutation,
    args: argparse.Namespace,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    reindexed_vertices = deformed_vertices[perm.target_to_source]
    reindexed_faces = perm.source_to_target[deformed_faces]

    recovered_vertices = np.empty_like(deformed_vertices)
    recovered_vertices[perm.target_to_source] = reindexed_vertices
    recovered_faces = perm.target_to_source[reindexed_faces]
    vertex_error = float(np.max(np.abs(recovered_vertices - deformed_vertices)))
    face_equal = bool(np.array_equal(recovered_faces, deformed_faces))

    obj_path = output_dir / "mesh_semantic_reindexed.obj"
    npz_path = output_dir / "semantic_permutation.npz"
    csv_path = output_dir / "semantic_index_map.csv"
    json_path = output_dir / "summary.json"

    write_obj(obj_path, reindexed_vertices, reindexed_faces)
    np.savez_compressed(
        npz_path,
        source_to_target=perm.source_to_target.astype(np.int64),
        target_to_source=perm.target_to_source.astype(np.int64),
        costs=perm.costs.astype(np.float32),
        similarities=perm.similarities.astype(np.float32),
        metadata=np.asarray(json.dumps(perm.metadata, sort_keys=True)),
    )
    write_mapping_csv(csv_path, perm)

    summary = {
        **perm.metadata,
        "source_mesh": str(args.source_mesh),
        "target_mesh": str(args.target_mesh),
        "deformed_mesh": str(args.deformed_mesh),
        "output_obj": str(obj_path),
        "output_npz": str(npz_path),
        "output_csv": str(csv_path),
        "pure_reindex_vertex_recovery_max_abs_error": vertex_error,
        "pure_reindex_faces_recover_exactly": face_equal,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", required=True)
    parser.add_argument("--target-mesh", required=True)
    parser.add_argument("--deformed-mesh", required=True)
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--source-labels", default=None)
    parser.add_argument("--target-labels", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-mode", default="auto", choices=["auto", "face", "vertex"])
    parser.add_argument("--label-mode", default="auto", choices=["auto", "face", "vertex"])
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--position-weight", type=float, default=0.15)
    parser.add_argument("--normal-weight", type=float, default=0.05)
    parser.add_argument("--label-mismatch-penalty", type=float, default=0.35)
    parser.add_argument("--topology-prior-weight", type=float, default=0.02)
    parser.add_argument("--topk", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_vertices, source_faces = load_obj(args.source_mesh)
    target_vertices, target_faces = load_obj(args.target_mesh)
    deformed_vertices, deformed_faces = load_obj(args.deformed_mesh)
    if deformed_vertices.shape[0] != source_vertices.shape[0]:
        raise ValueError("Deformed mesh vertex count must match source mesh vertex count.")
    if not np.array_equal(deformed_faces, source_faces):
        print("Warning: deformed faces differ from source faces; reindexing will preserve deformed connectivity.")

    perm = build_semantic_permutation(
        source_vertices=source_vertices,
        source_faces=source_faces,
        target_vertices=target_vertices,
        target_faces=target_faces,
        source_features_path=args.source_features,
        target_features_path=args.target_features,
        source_labels_path=args.source_labels,
        target_labels_path=args.target_labels,
        feature_mode=args.feature_mode,
        label_mode=args.label_mode,
        semantic_weight=args.semantic_weight,
        position_weight=args.position_weight,
        normal_weight=args.normal_weight,
        label_mismatch_penalty=args.label_mismatch_penalty,
        topology_prior_weight=args.topology_prior_weight,
        topk=args.topk,
    )
    summary = save_outputs(Path(args.output_dir), deformed_vertices, deformed_faces, perm, args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
