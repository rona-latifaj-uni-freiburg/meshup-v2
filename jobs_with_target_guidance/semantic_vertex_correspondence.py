"""Dense semantic vertex correspondences for target mesh deformation.

This module turns frozen semantic descriptors (PartField by default, but any
per-vertex/per-face feature array with the same shape conventions works) into a
source-vertex -> target-vertex map.  The resulting map can drive a direct MSE
target loss, which behaves like the successful topology-matched vertex loss
without requiring identical mesh topology.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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
    unit_vertices,
    vertex_normals_np,
)


LabelFilter = Literal["none", "soft", "hard"]


@dataclass
class SemanticVertexCorrespondence:
    source_to_target: np.ndarray
    weights: np.ndarray
    costs: np.ndarray
    similarities: np.ndarray
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


def _as_torch_faces(faces: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(faces, torch.Tensor):
        return faces.detach().cpu().long()
    return torch.tensor(np.asarray(faces), dtype=torch.long)


def _as_torch_vertices(vertices: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(vertices, torch.Tensor):
        return vertices.detach().cpu().float()
    return torch.tensor(np.asarray(vertices), dtype=torch.float32)


def _same_topology(
    source_faces: torch.Tensor,
    target_faces: torch.Tensor,
    n_source_vertices: int,
    n_target_vertices: int,
) -> bool:
    return (
        n_source_vertices == n_target_vertices
        and source_faces.shape == target_faces.shape
        and torch.equal(source_faces.cpu(), target_faces.cpu())
    )


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


def _label_penalties(
    source_label: int,
    candidate_labels: Optional[np.ndarray],
    label_filter: LabelFilter,
    label_mismatch_penalty: float,
) -> np.ndarray:
    if candidate_labels is None or label_filter == "none" or label_mismatch_penalty <= 0:
        return np.zeros(0 if candidate_labels is None else candidate_labels.shape[0], dtype=np.float32)
    penalties = np.zeros(candidate_labels.shape[0], dtype=np.float32)
    penalties[candidate_labels != int(source_label)] = float(label_mismatch_penalty)
    return penalties


def _topk_query(
    source_desc: np.ndarray,
    target_desc: np.ndarray,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    k = min(max(int(topk), 2), target_desc.shape[0])
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(target_desc)
    dists, idx = nn.kneighbors(source_desc, return_distance=True)
    return dists.astype(np.float32), idx.astype(np.int64)


def _target_to_source_nn(source_desc: np.ndarray, target_desc: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    nn.fit(source_desc)
    _, idx = nn.kneighbors(target_desc, return_distance=True)
    return idx[:, 0].astype(np.int64)


def _identity_cost(
    source_idx: int,
    source_desc: np.ndarray,
    target_desc: np.ndarray,
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: Optional[np.ndarray],
    target_labels: Optional[np.ndarray],
    label_filter: LabelFilter,
    label_mismatch_penalty: float,
) -> Tuple[float, float]:
    desc_delta = source_desc[source_idx] - target_desc[source_idx]
    cost = float(desc_delta @ desc_delta)
    if source_labels is not None and target_labels is not None and label_filter == "soft":
        if int(source_labels[source_idx]) != int(target_labels[source_idx]):
            cost += float(label_mismatch_penalty)
    sim = float(np.dot(source_features[source_idx], target_features[source_idx]))
    return cost, sim


def build_semantic_vertex_correspondence(
    source_vertices: torch.Tensor | np.ndarray,
    source_faces: torch.Tensor | np.ndarray,
    target_vertices: torch.Tensor | np.ndarray,
    target_faces: torch.Tensor | np.ndarray,
    source_features_path: str,
    target_features_path: str,
    source_labels_path: Optional[str] = None,
    target_labels_path: Optional[str] = None,
    feature_mode: FeatureMode = "auto",
    label_mode: LabelMode = "auto",
    label_filter: LabelFilter = "soft",
    semantic_weight: float = 1.0,
    position_weight: float = 0.20,
    normal_weight: float = 0.05,
    label_mismatch_penalty: float = 0.25,
    topk: int = 32,
    min_similarity: float = 0.05,
    confidence_margin: float = 0.04,
    confidence_floor: float = 0.25,
    nonmutual_weight: float = 0.60,
    topology_prior_weight: float = 0.0,
) -> SemanticVertexCorrespondence:
    src_v_t = _as_torch_vertices(source_vertices)
    tgt_v_t = _as_torch_vertices(target_vertices)
    src_f_t = _as_torch_faces(source_faces)
    tgt_f_t = _as_torch_faces(target_faces)

    n_src = int(src_v_t.shape[0])
    n_tgt = int(tgt_v_t.shape[0])
    if n_src == 0 or n_tgt == 0:
        raise ValueError("Source and target meshes must both contain vertices.")

    source_features = coerce_features_to_vertices(source_features_path, n_src, src_f_t, feature_mode)
    target_features = coerce_features_to_vertices(target_features_path, n_tgt, tgt_f_t, feature_mode)
    source_features = normalize_rows(source_features)
    target_features = normalize_rows(target_features)

    source_labels = None
    target_labels = None
    if source_labels_path and target_labels_path and label_filter != "none":
        source_labels = coerce_labels_to_vertices(source_labels_path, n_src, src_f_t, label_mode)
        target_labels = coerce_labels_to_vertices(target_labels_path, n_tgt, tgt_f_t, label_mode)

    src_v = _unit_vertices_np(src_v_t.numpy().astype(np.float32))
    tgt_v = _unit_vertices_np(tgt_v_t.numpy().astype(np.float32))
    src_normals = vertex_normals_np(src_v_t, src_f_t)
    tgt_normals = vertex_normals_np(tgt_v_t, tgt_f_t)

    source_desc = _descriptor_matrix(
        source_features,
        src_v,
        src_normals,
        semantic_weight=semantic_weight,
        position_weight=position_weight,
        normal_weight=normal_weight,
    )
    target_desc = _descriptor_matrix(
        target_features,
        tgt_v,
        tgt_normals,
        semantic_weight=semantic_weight,
        position_weight=position_weight,
        normal_weight=normal_weight,
    )

    query_dists, query_idx = _topk_query(source_desc, target_desc, topk=topk)
    target_to_source = _target_to_source_nn(source_desc, target_desc)
    same_topology = _same_topology(src_f_t, tgt_f_t, n_src, n_tgt)

    source_to_target = np.zeros(n_src, dtype=np.int64)
    costs = np.zeros(n_src, dtype=np.float32)
    similarities = np.zeros(n_src, dtype=np.float32)
    margin_scores = np.ones(n_src, dtype=np.float32)
    identity_kept = np.zeros(n_src, dtype=bool)

    for src_idx in range(n_src):
        candidates = query_idx[src_idx]
        candidate_costs = np.square(query_dists[src_idx]).astype(np.float32)

        if source_labels is not None and target_labels is not None:
            if label_filter == "hard":
                label_mask = target_labels[candidates] == int(source_labels[src_idx])
                if label_mask.any():
                    candidates = candidates[label_mask]
                    candidate_costs = candidate_costs[label_mask]
            elif label_filter == "soft":
                candidate_costs = candidate_costs + _label_penalties(
                    int(source_labels[src_idx]),
                    target_labels[candidates],
                    label_filter,
                    label_mismatch_penalty,
                )

        order = np.argsort(candidate_costs)
        candidates = candidates[order]
        candidate_costs = candidate_costs[order]
        best_target = int(candidates[0])
        best_cost = float(candidate_costs[0])
        best_sim = float(np.dot(source_features[src_idx], target_features[best_target]))
        second_cost = float(candidate_costs[1]) if candidate_costs.shape[0] > 1 else best_cost + confidence_margin

        if same_topology and topology_prior_weight > 0:
            ident_cost, ident_sim = _identity_cost(
                src_idx,
                source_desc,
                target_desc,
                source_features,
                target_features,
                source_labels,
                target_labels,
                label_filter,
                label_mismatch_penalty,
            )
            if ident_cost <= best_cost + float(topology_prior_weight):
                best_target = int(src_idx)
                best_cost = float(ident_cost)
                best_sim = float(ident_sim)
                identity_kept[src_idx] = True

        source_to_target[src_idx] = best_target
        costs[src_idx] = best_cost
        similarities[src_idx] = best_sim
        if confidence_margin > 0:
            margin_scores[src_idx] = np.clip((second_cost - best_cost) / float(confidence_margin), 0.0, 1.0)

    sim_scores = np.ones(n_src, dtype=np.float32)
    if min_similarity > -1.0:
        denom = max(1.0 - float(min_similarity), 1e-6)
        sim_scores = np.clip((similarities - float(min_similarity)) / denom, 0.0, 1.0).astype(np.float32)

    mutual = target_to_source[source_to_target] == np.arange(n_src, dtype=np.int64)
    mutual_scores = np.where(mutual, 1.0, max(float(nonmutual_weight), 0.0)).astype(np.float32)
    confidence = np.minimum(sim_scores, margin_scores) * mutual_scores
    confidence_floor = min(max(float(confidence_floor), 0.0), 1.0)
    weights = confidence_floor + (1.0 - confidence_floor) * confidence
    weights = weights.astype(np.float32)

    metadata: Dict[str, object] = {
        "source_features_path": str(source_features_path),
        "target_features_path": str(target_features_path),
        "source_labels_path": str(source_labels_path or ""),
        "target_labels_path": str(target_labels_path or ""),
        "feature_mode": feature_mode,
        "label_mode": label_mode,
        "n_source_vertices": n_src,
        "n_target_vertices": n_tgt,
        "same_topology": bool(same_topology),
        "label_filter": label_filter,
        "topk": int(topk),
        "semantic_weight": float(semantic_weight),
        "position_weight": float(position_weight),
        "normal_weight": float(normal_weight),
        "label_mismatch_penalty": float(label_mismatch_penalty),
        "min_similarity": float(min_similarity),
        "confidence_margin": float(confidence_margin),
        "confidence_floor": float(confidence_floor),
        "nonmutual_weight": float(nonmutual_weight),
        "topology_prior_weight": float(topology_prior_weight),
        "mean_similarity": float(similarities.mean()),
        "p05_similarity": float(np.percentile(similarities, 5)),
        "mean_weight": float(weights.mean()),
        "p05_weight": float(np.percentile(weights, 5)),
        "mutual_fraction": float(mutual.mean()),
        "identity_fraction": float((source_to_target == np.arange(n_src)).mean()) if n_src == n_tgt else 0.0,
        "identity_prior_kept_fraction": float(identity_kept.mean()),
        "unique_target_fraction": float(np.unique(source_to_target).shape[0] / max(n_tgt, 1)),
    }
    if source_labels is not None and target_labels is not None:
        metadata["label_agreement_fraction"] = float((source_labels == target_labels[source_to_target]).mean())

    return SemanticVertexCorrespondence(
        source_to_target=source_to_target,
        weights=weights,
        costs=costs,
        similarities=similarities,
        metadata=metadata,
    )


def save_correspondence(path: str | Path, corr: SemanticVertexCorrespondence) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        source_to_target=corr.source_to_target.astype(np.int64),
        weights=corr.weights.astype(np.float32),
        costs=corr.costs.astype(np.float32),
        similarities=corr.similarities.astype(np.float32),
        metadata=np.asarray(json.dumps(corr.metadata, sort_keys=True)),
    )


def load_correspondence(path: str | Path) -> SemanticVertexCorrespondence:
    data = np.load(path, allow_pickle=False)
    metadata: Dict[str, object] = {}
    if "metadata" in data:
        metadata = json.loads(str(data["metadata"].item()))
    return SemanticVertexCorrespondence(
        source_to_target=np.asarray(data["source_to_target"], dtype=np.int64),
        weights=np.asarray(data["weights"], dtype=np.float32),
        costs=np.asarray(data["costs"], dtype=np.float32) if "costs" in data else np.zeros(0, dtype=np.float32),
        similarities=np.asarray(data["similarities"], dtype=np.float32)
        if "similarities" in data
        else np.zeros(0, dtype=np.float32),
        metadata=metadata,
    )


def _metadata_matches(metadata: Dict[str, object], expected: Dict[str, object]) -> bool:
    for key, expected_value in expected.items():
        if key not in metadata:
            return False
        actual_value = metadata[key]
        if isinstance(expected_value, float):
            try:
                if abs(float(actual_value) - expected_value) > 1e-8:
                    return False
            except (TypeError, ValueError):
                return False
        elif actual_value != expected_value:
            return False
    return True


class SemanticVertexCorrespondenceLoss(nn.Module):
    """Direct target MSE using a precomputed semantic source->target map."""

    def __init__(
        self,
        target_vertices: torch.Tensor,
        source_to_target: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        super().__init__()
        if source_to_target.ndim != 1:
            raise ValueError("source_to_target must be a 1D array.")
        if weights.shape[0] != source_to_target.shape[0]:
            raise ValueError("weights and source_to_target must have the same length.")
        target_vertices_unit = unit_vertices(target_vertices.detach())
        max_idx = int(source_to_target.max(initial=-1))
        if max_idx >= target_vertices_unit.shape[0] or int(source_to_target.min(initial=0)) < 0:
            raise ValueError("source_to_target contains indices outside target vertex range.")

        self.register_buffer("source_to_target", torch.tensor(source_to_target, dtype=torch.long, device=target_vertices.device))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32, device=target_vertices.device))
        self.register_buffer("target_vertices", target_vertices_unit)

    def forward(self, current_vertices: torch.Tensor, return_components: bool = False):
        source_vertices = unit_vertices(current_vertices)
        target_points = self.target_vertices.index_select(0, self.source_to_target)
        sq = (source_vertices - target_points).pow(2).sum(dim=1)
        raw = (sq * self.weights).sum() / self.weights.sum().clamp(min=1e-8)
        if not return_components:
            return raw
        return {
            "raw": raw,
            "mean_weight": self.weights.mean().detach(),
            "min_weight": self.weights.min().detach(),
            "max_weight": self.weights.max().detach(),
            "unique_targets": float(torch.unique(self.source_to_target).numel()),
        }


def create_semantic_vertex_correspondence_loss(
    source_vertices: torch.Tensor,
    source_faces: torch.Tensor,
    target_vertices: torch.Tensor,
    target_faces: torch.Tensor,
    source_features_path: str,
    target_features_path: str,
    source_labels_path: Optional[str] = None,
    target_labels_path: Optional[str] = None,
    feature_mode: FeatureMode = "auto",
    label_mode: LabelMode = "auto",
    label_filter: LabelFilter = "soft",
    semantic_weight: float = 1.0,
    position_weight: float = 0.20,
    normal_weight: float = 0.05,
    label_mismatch_penalty: float = 0.25,
    topk: int = 32,
    min_similarity: float = 0.05,
    confidence_margin: float = 0.04,
    confidence_floor: float = 0.25,
    nonmutual_weight: float = 0.60,
    topology_prior_weight: float = 0.0,
    cache_path: Optional[str] = None,
    rebuild_cache: bool = False,
) -> Tuple[SemanticVertexCorrespondenceLoss, SemanticVertexCorrespondence]:
    expected_metadata = {
        "source_features_path": str(source_features_path),
        "target_features_path": str(target_features_path),
        "source_labels_path": str(source_labels_path or ""),
        "target_labels_path": str(target_labels_path or ""),
        "feature_mode": feature_mode,
        "label_mode": label_mode,
        "label_filter": label_filter,
        "semantic_weight": float(semantic_weight),
        "position_weight": float(position_weight),
        "normal_weight": float(normal_weight),
        "label_mismatch_penalty": float(label_mismatch_penalty),
        "topk": int(topk),
        "min_similarity": float(min_similarity),
        "confidence_margin": float(confidence_margin),
        "confidence_floor": float(confidence_floor),
        "nonmutual_weight": float(nonmutual_weight),
        "topology_prior_weight": float(topology_prior_weight),
    }
    if cache_path and Path(cache_path).is_file() and not rebuild_cache:
        corr = load_correspondence(cache_path)
        if not _metadata_matches(corr.metadata, expected_metadata):
            corr = build_semantic_vertex_correspondence(
                source_vertices=source_vertices,
                source_faces=source_faces,
                target_vertices=target_vertices,
                target_faces=target_faces,
                source_features_path=source_features_path,
                target_features_path=target_features_path,
                source_labels_path=source_labels_path,
                target_labels_path=target_labels_path,
                feature_mode=feature_mode,
                label_mode=label_mode,
                label_filter=label_filter,
                semantic_weight=semantic_weight,
                position_weight=position_weight,
                normal_weight=normal_weight,
                label_mismatch_penalty=label_mismatch_penalty,
                topk=topk,
                min_similarity=min_similarity,
                confidence_margin=confidence_margin,
                confidence_floor=confidence_floor,
                nonmutual_weight=nonmutual_weight,
                topology_prior_weight=topology_prior_weight,
            )
            save_correspondence(cache_path, corr)
    else:
        corr = build_semantic_vertex_correspondence(
            source_vertices=source_vertices,
            source_faces=source_faces,
            target_vertices=target_vertices,
            target_faces=target_faces,
            source_features_path=source_features_path,
            target_features_path=target_features_path,
            source_labels_path=source_labels_path,
            target_labels_path=target_labels_path,
            feature_mode=feature_mode,
            label_mode=label_mode,
            label_filter=label_filter,
            semantic_weight=semantic_weight,
            position_weight=position_weight,
            normal_weight=normal_weight,
            label_mismatch_penalty=label_mismatch_penalty,
            topk=topk,
            min_similarity=min_similarity,
            confidence_margin=confidence_margin,
            confidence_floor=confidence_floor,
            nonmutual_weight=nonmutual_weight,
            topology_prior_weight=topology_prior_weight,
        )
        if cache_path:
            save_correspondence(cache_path, corr)

    loss = SemanticVertexCorrespondenceLoss(
        target_vertices=target_vertices,
        source_to_target=corr.source_to_target,
        weights=corr.weights,
    )
    return loss, corr


def load_obj(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with Path(path).open("r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                raw = [token.split("/")[0] for token in line.split()[1:]]
                idx = [int(token) - 1 for token in raw]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"Could not load OBJ vertices/faces from {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# semantic vertex-correspondence retargeted mesh\n")
        for v in vertices:
            f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for face in faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", required=True)
    parser.add_argument("--target-mesh", required=True)
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--source-labels", default=None)
    parser.add_argument("--target-labels", default=None)
    parser.add_argument("--feature-mode", choices=["auto", "vertex", "face"], default="auto")
    parser.add_argument("--label-mode", choices=["auto", "vertex", "face"], default="auto")
    parser.add_argument("--label-filter", choices=["none", "soft", "hard"], default="soft")
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--position-weight", type=float, default=0.20)
    parser.add_argument("--normal-weight", type=float, default=0.05)
    parser.add_argument("--label-mismatch-penalty", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--min-similarity", type=float, default=0.05)
    parser.add_argument("--confidence-margin", type=float, default=0.04)
    parser.add_argument("--confidence-floor", type=float, default=0.25)
    parser.add_argument("--nonmutual-weight", type=float, default=0.60)
    parser.add_argument("--topology-prior-weight", type=float, default=0.0)
    parser.add_argument("--output-map", required=True)
    parser.add_argument("--output-retargeted-mesh", default=None)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_vertices, source_faces = load_obj(args.source_mesh)
    target_vertices, target_faces = load_obj(args.target_mesh)
    corr = build_semantic_vertex_correspondence(
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
        label_filter=args.label_filter,
        semantic_weight=args.semantic_weight,
        position_weight=args.position_weight,
        normal_weight=args.normal_weight,
        label_mismatch_penalty=args.label_mismatch_penalty,
        topk=args.topk,
        min_similarity=args.min_similarity,
        confidence_margin=args.confidence_margin,
        confidence_floor=args.confidence_floor,
        nonmutual_weight=args.nonmutual_weight,
        topology_prior_weight=args.topology_prior_weight,
    )
    save_correspondence(args.output_map, corr)

    if args.output_retargeted_mesh:
        retargeted_vertices = target_vertices[corr.source_to_target]
        write_obj(args.output_retargeted_mesh, retargeted_vertices, source_faces)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(corr.metadata, indent=2, sort_keys=True) + "\n")

    print(json.dumps(corr.metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
