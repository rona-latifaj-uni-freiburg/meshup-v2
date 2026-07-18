"""PartField feature buckets for target Chamfer guidance.

PartField predicts per-face or per-vertex part features for a mesh. This module
uses those features as semantic buckets for MeshUp target guidance:

1. Load source/target PartField features or clustered labels.
2. Convert face-level outputs to vertex-level buckets when needed.
3. Cluster source features and assign target vertices in the same feature space,
   or match provided target labels to source buckets.
4. Compute Chamfer only between matching source/target buckets, or use
   PartField latent distances as soft source-target correspondences.

The bucket labels are frozen at initialization. During optimization, source
vertices move while keeping their initial PartField bucket identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


FeatureMode = Literal["auto", "vertex", "face"]
LabelMode = Literal["auto", "vertex", "face"]
GuidanceMode = Literal["hard", "soft", "hybrid", "balanced", "hard_unbalanced"]
SoftMatchSpace = Literal["latent", "geometry", "hybrid"]


def unit_vertices(vertices: torch.Tensor) -> torch.Tensor:
    vmin = vertices.amin(dim=0)
    vmax = vertices.amax(dim=0)
    scale = 2.0 / torch.clamp((vmax - vmin).max(), min=1e-8)
    return (vertices - (vmax + vmin) * 0.5) * scale


def sample_vertices(vertices: torch.Tensor, n_points: int) -> torch.Tensor:
    if n_points <= 0 or vertices.shape[0] <= n_points:
        return vertices
    idx = sample_indices(vertices.shape[0], n_points, vertices.device)
    return vertices.index_select(0, idx)


def sample_indices(n_items: int, n_points: int, device: torch.device) -> torch.Tensor:
    if n_points <= 0 or n_items <= n_points:
        return torch.arange(n_items, device=device, dtype=torch.long)
    return torch.linspace(0, n_items - 1, steps=n_points, device=device).long()


def symmetric_chamfer(
    src_vertices: torch.Tensor,
    tgt_vertices: torch.Tensor,
    n_points: int,
    source_to_target_weight: float = 1.0,
    target_to_source_weight: float = 1.0,
    tgt_to_src_robust_scale: float = 0.0,
    src_to_tgt_unmatched_weight: float = 1.0,
) -> torch.Tensor:
    src = sample_vertices(src_vertices, n_points)
    tgt = sample_vertices(tgt_vertices, n_points)
    dists = torch.cdist(src, tgt, p=2).pow(2)
    src_d2 = dists.min(dim=1).values
    if src_to_tgt_unmatched_weight < 1:
        tgt_nn_src = dists.detach().argmin(dim=0)
        reverse_counts = torch.bincount(tgt_nn_src, minlength=src.shape[0])
        unmatched_weight = max(float(src_to_tgt_unmatched_weight), 0.0)
        src_weights = torch.where(
            reverse_counts > 0,
            torch.ones_like(src_d2),
            src_d2.new_full(src_d2.shape, unmatched_weight),
        )
        src_to_tgt = (src_weights * src_d2).sum() / src_weights.sum().clamp(min=1e-8)
    else:
        src_to_tgt = src_d2.mean()
    tgt_d2 = dists.min(dim=0).values
    if tgt_to_src_robust_scale != 0:
        if tgt_to_src_robust_scale > 0:
            scale = dists.new_tensor(float(tgt_to_src_robust_scale))
        else:
            scale = torch.sqrt(torch.median(tgt_d2.detach()).clamp(min=1e-12))
        c2 = scale.pow(2).clamp(min=1e-12)
        tgt_to_src = (c2 * tgt_d2 / (tgt_d2 + c2)).mean()
    else:
        tgt_to_src = tgt_d2.mean()
    return source_to_target_weight * src_to_tgt + target_to_source_weight * tgt_to_src


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp(min=1e-8)


def normalize_torch_rows(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return features / torch.clamp(features.norm(dim=1, keepdim=True), min=eps)


def semantic_confidence_from_similarity(
    similarities: torch.Tensor,
    dim: int,
    min_similarity: float,
    margin: float,
    floor: float,
    power: float,
) -> torch.Tensor:
    """Confidence from feature match quality; values are detached geometry weights."""
    floor = min(max(float(floor), 0.0), 1.0)
    if floor >= 1.0 and min_similarity <= -1.0 and margin <= 0.0:
        size = similarities.shape[1 - dim]
        return similarities.new_ones(size)

    best = similarities.max(dim=dim).values
    confidence = torch.ones_like(best)

    if min_similarity > -1.0:
        denom = max(1.0 - float(min_similarity), 1e-6)
        sim_gate = torch.clamp((best - float(min_similarity)) / denom, min=0.0, max=1.0)
        confidence = confidence * sim_gate

    if margin > 0.0 and similarities.shape[dim] > 1:
        top2 = torch.topk(similarities, k=2, dim=dim).values
        gap = top2.select(dim, 0) - top2.select(dim, 1)
        margin_gate = torch.clamp(gap / float(margin), min=0.0, max=1.0)
        confidence = confidence * margin_gate

    if power != 1.0:
        confidence = confidence.clamp(min=0.0).pow(float(power))
    return (floor + (1.0 - floor) * confidence).detach()


def feature_guided_symmetric_chamfer(
    src_vertices: torch.Tensor,
    tgt_vertices: torch.Tensor,
    n_points: int,
    source_to_target_weight: float = 1.0,
    target_to_source_weight: float = 1.0,
    tgt_to_src_robust_scale: float = 0.0,
    src_to_tgt_unmatched_weight: float = 1.0,
    src_features: Optional[torch.Tensor] = None,
    tgt_features: Optional[torch.Tensor] = None,
    semantic_weight: float = 0.0,
    geometry_sigma: float = 1.0,
    confidence_min_similarity: float = -1.0,
    confidence_margin: float = 0.0,
    confidence_floor: float = 1.0,
    confidence_power: float = 1.0,
) -> torch.Tensor:
    src_idx = sample_indices(src_vertices.shape[0], n_points, src_vertices.device)
    tgt_idx = sample_indices(tgt_vertices.shape[0], n_points, tgt_vertices.device)
    src = src_vertices.index_select(0, src_idx)
    tgt = tgt_vertices.index_select(0, tgt_idx)
    dists = torch.cdist(src, tgt, p=2).pow(2)

    similarities = None
    match_cost = dists.detach() / (max(float(geometry_sigma), 1e-6) ** 2)
    has_features = src_features is not None and tgt_features is not None and src_features.numel() and tgt_features.numel()
    if has_features:
        src_feat = src_features.index_select(0, src_idx)
        tgt_feat = tgt_features.index_select(0, tgt_idx)
        similarities = src_feat @ tgt_feat.t()
        if semantic_weight > 0.0:
            semantic_dists = torch.clamp(1.0 - similarities, min=0.0)
            match_cost = match_cost + float(semantic_weight) * semantic_dists

    src_nn = match_cost.argmin(dim=1)
    src_d2 = dists.gather(1, src_nn.unsqueeze(1)).squeeze(1)
    src_weights = torch.ones_like(src_d2)
    if similarities is not None:
        src_weights = src_weights * semantic_confidence_from_similarity(
            similarities,
            dim=1,
            min_similarity=confidence_min_similarity,
            margin=confidence_margin,
            floor=confidence_floor,
            power=confidence_power,
        )
    if src_to_tgt_unmatched_weight < 1:
        tgt_nn_src = match_cost.detach().argmin(dim=0)
        reverse_counts = torch.bincount(tgt_nn_src, minlength=src.shape[0])
        unmatched_weight = max(float(src_to_tgt_unmatched_weight), 0.0)
        src_weights = src_weights * torch.where(
            reverse_counts > 0,
            torch.ones_like(src_d2),
            src_d2.new_full(src_d2.shape, unmatched_weight),
        )
    src_to_tgt = weighted_mean(src_d2, src_weights)

    tgt_nn = match_cost.argmin(dim=0)
    tgt_d2 = dists.gather(0, tgt_nn.unsqueeze(0)).squeeze(0)
    if tgt_to_src_robust_scale != 0:
        if tgt_to_src_robust_scale > 0:
            scale = dists.new_tensor(float(tgt_to_src_robust_scale))
        else:
            scale = torch.sqrt(torch.median(tgt_d2.detach()).clamp(min=1e-12))
        c2 = scale.pow(2).clamp(min=1e-12)
        tgt_values = c2 * tgt_d2 / (tgt_d2 + c2)
    else:
        tgt_values = tgt_d2

    tgt_weights = torch.ones_like(tgt_values)
    if similarities is not None:
        tgt_weights = tgt_weights * semantic_confidence_from_similarity(
            similarities,
            dim=0,
            min_similarity=confidence_min_similarity,
            margin=confidence_margin,
            floor=confidence_floor,
            power=confidence_power,
        )
    tgt_to_src = weighted_mean(tgt_values, tgt_weights)
    return source_to_target_weight * src_to_tgt + target_to_source_weight * tgt_to_src


def load_numpy_array(path: str, preferred_keys: Tuple[str, ...]) -> np.ndarray:
    if not path:
        raise ValueError("Empty PartField array path.")

    suffix = Path(path).suffix.lower()
    if suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, dict):
            for key in preferred_keys:
                if key in data:
                    value = data[key]
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    return np.asarray(value)
            first_key = next(iter(data))
            value = data[first_key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value)
        return np.asarray(data)

    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        for key in preferred_keys:
            if key in data:
                return np.asarray(data[key])
        first_key = data.files[0]
        return np.asarray(data[first_key])
    return np.asarray(data)


def load_feature_array(path: str) -> np.ndarray:
    features = load_numpy_array(
        path,
        preferred_keys=("features", "part_features", "part_feat", "point_feat", "vertex_features", "face_features"),
    )
    features = np.asarray(features)
    if features.ndim == 1:
        raise ValueError(f"PartField features must be 2D, got shape {features.shape} from {path}")
    return features.reshape(features.shape[0], -1).astype(np.float32)


def load_label_array(path: str) -> np.ndarray:
    labels = load_numpy_array(
        path,
        preferred_keys=("labels", "part_labels", "pred", "source_labels", "target_labels", "face_labels", "vertex_labels"),
    )
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        raise ValueError(f"Empty PartField label array: {path}")
    return labels.astype(np.int64)


def face_features_to_vertex_features(face_features: np.ndarray, faces: torch.Tensor, n_vertices: int) -> np.ndarray:
    faces_np = faces.detach().cpu().numpy().astype(np.int64)
    vertex_features = np.zeros((n_vertices, face_features.shape[1]), dtype=np.float32)
    counts = np.zeros(n_vertices, dtype=np.float32)

    for corner in range(3):
        np.add.at(vertex_features, faces_np[:, corner], face_features)
        np.add.at(counts, faces_np[:, corner], 1.0)

    seen = counts > 0
    vertex_features[seen] /= counts[seen, None]
    if (~seen).any() and seen.any():
        vertices_seen = np.where(seen)[0]
        fallback = vertex_features[vertices_seen[0]]
        vertex_features[~seen] = fallback
    return vertex_features


def face_labels_to_vertex_labels(face_labels: np.ndarray, faces: torch.Tensor, n_vertices: int) -> np.ndarray:
    faces_np = faces.detach().cpu().numpy().astype(np.int64)
    votes: List[Dict[int, int]] = [dict() for _ in range(n_vertices)]
    for face_idx, face in enumerate(faces_np):
        label = int(face_labels[face_idx])
        for vertex_id in face:
            bucket = votes[int(vertex_id)]
            bucket[label] = bucket.get(label, 0) + 1

    vertex_labels = np.full(n_vertices, -1, dtype=np.int64)
    for vertex_id, bucket in enumerate(votes):
        if bucket:
            vertex_labels[vertex_id] = max(bucket.items(), key=lambda item: item[1])[0]

    missing = vertex_labels < 0
    if missing.any():
        vertex_labels[missing] = 0
    return vertex_labels


def coerce_features_to_vertices(
    features_path: str,
    n_vertices: int,
    faces: torch.Tensor,
    mode: FeatureMode = "auto",
) -> np.ndarray:
    features = load_feature_array(features_path)
    n_faces = faces.shape[0]

    if mode == "vertex" or (mode == "auto" and features.shape[0] == n_vertices):
        if features.shape[0] != n_vertices:
            raise ValueError(f"Expected {n_vertices} vertex features in {features_path}, got {features.shape[0]}")
        return features

    if mode == "face" or (mode == "auto" and features.shape[0] == n_faces):
        if features.shape[0] != n_faces:
            raise ValueError(f"Expected {n_faces} face features in {features_path}, got {features.shape[0]}")
        return face_features_to_vertex_features(features, faces, n_vertices)

    raise ValueError(
        f"Cannot infer PartField feature mode for {features_path}: got {features.shape[0]} rows, "
        f"but mesh has {n_vertices} vertices and {n_faces} faces."
    )


def coerce_labels_to_vertices(
    labels_path: str,
    n_vertices: int,
    faces: torch.Tensor,
    mode: LabelMode = "auto",
) -> np.ndarray:
    labels = load_label_array(labels_path)
    n_faces = faces.shape[0]

    if mode == "vertex" or (mode == "auto" and labels.shape[0] == n_vertices):
        if labels.shape[0] != n_vertices:
            raise ValueError(f"Expected {n_vertices} vertex labels in {labels_path}, got {labels.shape[0]}")
        return labels

    if mode == "face" or (mode == "auto" and labels.shape[0] == n_faces):
        if labels.shape[0] != n_faces:
            raise ValueError(f"Expected {n_faces} face labels in {labels_path}, got {labels.shape[0]}")
        return face_labels_to_vertex_labels(labels, faces, n_vertices)

    raise ValueError(
        f"Cannot infer PartField label mode for {labels_path}: got {labels.shape[0]} labels, "
        f"but mesh has {n_vertices} vertices and {n_faces} faces."
    )


def normalize_rows(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    features = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def bbox_coordinates_np(vertices: torch.Tensor) -> np.ndarray:
    verts = vertices.detach().cpu().numpy().astype(np.float32)
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    return (verts - vmin) / np.maximum(vmax - vmin, 1e-8)


def vertex_normals_np(vertices: torch.Tensor, faces: torch.Tensor) -> np.ndarray:
    verts = vertices.detach().cpu().numpy().astype(np.float32)
    faces_np = faces.detach().cpu().numpy().astype(np.int64)
    normals = np.zeros_like(verts)
    tri = verts[faces_np]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-8)
    for corner in range(3):
        np.add.at(normals, faces_np[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return normals.astype(np.float32)


def clustering_features(
    partfield_features: np.ndarray,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    position_weight: float,
    normal_weight: float,
) -> np.ndarray:
    features = [normalize_rows(partfield_features)]
    if position_weight > 0:
        features.append(bbox_coordinates_np(vertices) * float(position_weight))
    if normal_weight > 0:
        features.append(vertex_normals_np(vertices, faces) * float(normal_weight))
    return np.concatenate(features, axis=1).astype(np.float32)


def remap_to_contiguous(labels: np.ndarray, min_points: int = 1) -> Tuple[np.ndarray, Dict[int, int]]:
    labels = labels.astype(np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    kept = [int(label) for label, count in zip(unique, counts) if int(count) >= min_points]
    if not kept:
        raise ValueError("No PartField labels remain after min_points filtering.")

    largest_label = int(unique[np.argmax(counts)])
    mapping = {old_label: new_label for new_label, old_label in enumerate(kept)}
    fallback = mapping.get(largest_label, 0)
    remapped = np.array([mapping.get(int(label), fallback) for label in labels], dtype=np.int64)
    return remapped, mapping


def remap_aligned_label_pair(source_labels: np.ndarray, target_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    labels = np.concatenate([source_labels.reshape(-1), target_labels.reshape(-1)]).astype(np.int64)
    unique = np.unique(labels)
    mapping = {int(old_label): new_label for new_label, old_label in enumerate(unique)}
    source_mapped = np.array([mapping[int(label)] for label in source_labels], dtype=np.int64)
    target_mapped = np.array([mapping[int(label)] for label in target_labels], dtype=np.int64)
    return source_mapped, target_mapped, mapping


def assign_target_by_feature_centroids(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_bucket_labels: np.ndarray,
    target_raw_labels: Optional[np.ndarray],
    n_source_buckets: int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    source_features = normalize_rows(source_features)
    target_features = normalize_rows(target_features)

    centroids = []
    for bucket_id in range(n_source_buckets):
        mask = source_bucket_labels == bucket_id
        if mask.any():
            centroids.append(source_features[mask].mean(axis=0))
        else:
            centroids.append(np.zeros(source_features.shape[1], dtype=np.float32))
    centroids = normalize_rows(np.stack(centroids, axis=0))

    if target_raw_labels is None:
        dists = ((target_features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        return dists.argmin(axis=1).astype(np.int64), {}

    target_to_source: Dict[int, int] = {}
    mapped = np.zeros(target_raw_labels.shape[0], dtype=np.int64)
    for target_label in np.unique(target_raw_labels):
        mask = target_raw_labels == target_label
        feature = normalize_rows(target_features[mask].mean(axis=0, keepdims=True))[0]
        dists = ((centroids - feature[None, :]) ** 2).sum(axis=-1)
        source_bucket = int(dists.argmin())
        target_to_source[int(target_label)] = source_bucket
        mapped[mask] = source_bucket
    return mapped, target_to_source


def part_geometry_feature(
    vertices_unit: torch.Tensor,
    labels: np.ndarray,
    label_id: int,
    centroid_weight: float = 1.0,
    extent_weight: float = 0.35,
    size_weight: float = 0.10,
) -> np.ndarray:
    mask_np = labels == label_id
    mask = torch.tensor(mask_np, device=vertices_unit.device, dtype=torch.bool)
    part_vertices = vertices_unit[mask]
    if part_vertices.numel() == 0:
        centroid = torch.zeros(3, device=vertices_unit.device)
        extent = torch.zeros(3, device=vertices_unit.device)
        size = torch.zeros(1, device=vertices_unit.device)
    else:
        centroid = part_vertices.mean(dim=0)
        extent = part_vertices.amax(dim=0) - part_vertices.amin(dim=0)
        size = torch.tensor([float(part_vertices.shape[0]) / float(vertices_unit.shape[0])], device=vertices_unit.device)
    feature = torch.cat([centroid * centroid_weight, extent * extent_weight, size * size_weight], dim=0)
    return feature.detach().cpu().numpy().astype(np.float32)


def match_target_labels_by_geometry(
    source_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    source_bucket_labels: np.ndarray,
    target_raw_labels: np.ndarray,
    n_source_buckets: int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    source_unit = unit_vertices(source_vertices.detach())
    target_unit = unit_vertices(target_vertices.detach())

    source_features = np.stack([
        part_geometry_feature(source_unit, source_bucket_labels, bucket_id)
        for bucket_id in range(n_source_buckets)
    ])

    target_unique = np.unique(target_raw_labels)
    target_features = np.stack([
        part_geometry_feature(target_unit, target_raw_labels, int(label))
        for label in target_unique
    ])

    dists = ((target_features[:, None, :] - source_features[None, :, :]) ** 2).sum(axis=-1)
    nearest_source = dists.argmin(axis=1)
    target_to_source = {
        int(target_label): int(source_bucket)
        for target_label, source_bucket in zip(target_unique, nearest_source)
    }
    mapped_target = np.array([target_to_source[int(label)] for label in target_raw_labels], dtype=np.int64)
    return mapped_target, target_to_source


class PartFieldChamferLoss(nn.Module):
    """PartField-guided Chamfer using hard labels and/or latent soft matches."""

    def __init__(
        self,
        source_vertices: torch.Tensor,
        target_vertices: torch.Tensor,
        source_bucket_labels: np.ndarray,
        target_bucket_labels: np.ndarray,
        n_buckets: int,
        points_per_bucket: int = 512,
        min_points_per_bucket: int = 12,
        global_weight: float = 0.0,
        guidance_mode: GuidanceMode = "hard",
        hard_weight: float = 1.0,
        source_to_target_weight: float = 1.0,
        target_to_source_weight: float = 1.0,
        tgt_to_src_robust_scale: float = 0.0,
        src_to_tgt_unmatched_weight: float = 1.0,
        hard_semantic_weight: float = 0.0,
        hard_geometry_sigma: float = 1.0,
        semantic_confidence_min_similarity: float = -1.0,
        semantic_confidence_margin: float = 0.0,
        semantic_confidence_floor: float = 1.0,
        semantic_confidence_power: float = 1.0,
        unbalanced_transport_weight: float = 0.0,
        unbalanced_transport_rho: float = 0.30,
        soft_weight: float = 1.0,
        soft_points: Optional[int] = None,
        soft_semantic_weight: float = 0.10,
        soft_temperature: float = 0.03,
        soft_match_space: SoftMatchSpace = "hybrid",
        soft_geometry_sigma: float = 1.0,
        containment_weight: float = 0.0,
        containment_margin: float = 0.02,
        containment_max_weight: float = 1.0,
        moment_weight: float = 0.0,
        moment_extent_weight: float = 0.5,
        profile_weight: float = 0.0,
        profile_bins: int = 9,
        profile_trim: float = 0.08,
        anchor_weight: float = 0.0,
        anchor_geometry_sigma: float = 0.35,
        anchor_semantic_weight: float = 0.0,
        balanced_sinkhorn_iters: int = 30,
        source_semantic_features: Optional[np.ndarray] = None,
        target_semantic_features: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        if guidance_mode not in {"hard", "soft", "hybrid", "balanced", "hard_unbalanced"}:
            raise ValueError(f"Unknown PartField guidance_mode: {guidance_mode}")
        if soft_match_space not in {"latent", "geometry", "hybrid"}:
            raise ValueError(f"Unknown PartField soft_match_space: {soft_match_space}")

        self.n_buckets = int(n_buckets)
        self.points_per_bucket = int(points_per_bucket)
        self.min_points_per_bucket = int(min_points_per_bucket)
        self.global_weight = float(global_weight)
        self.guidance_mode = guidance_mode
        self.hard_weight = float(hard_weight)
        self.source_to_target_weight = max(float(source_to_target_weight), 0.0)
        self.target_to_source_weight = max(float(target_to_source_weight), 0.0)
        self.tgt_to_src_robust_scale = float(tgt_to_src_robust_scale)
        self.src_to_tgt_unmatched_weight = max(float(src_to_tgt_unmatched_weight), 0.0)
        self.hard_semantic_weight = max(float(hard_semantic_weight), 0.0)
        self.hard_geometry_sigma = max(float(hard_geometry_sigma), 1e-6)
        self.semantic_confidence_min_similarity = float(semantic_confidence_min_similarity)
        self.semantic_confidence_margin = max(float(semantic_confidence_margin), 0.0)
        self.semantic_confidence_floor = min(max(float(semantic_confidence_floor), 0.0), 1.0)
        self.semantic_confidence_power = max(float(semantic_confidence_power), 1e-6)
        self.unbalanced_transport_weight = max(float(unbalanced_transport_weight), 0.0)
        self.unbalanced_transport_rho = max(float(unbalanced_transport_rho), 1e-6)
        self.soft_weight = float(soft_weight)
        self.soft_points = int(soft_points or points_per_bucket)
        self.soft_semantic_weight = float(soft_semantic_weight)
        self.soft_temperature = max(float(soft_temperature), 1e-6)
        self.soft_match_space = soft_match_space
        self.soft_geometry_sigma = max(float(soft_geometry_sigma), 1e-6)
        self.containment_weight = float(containment_weight)
        self.containment_margin = max(float(containment_margin), 0.0)
        self.containment_max_weight = max(float(containment_max_weight), 0.0)
        self.moment_weight = max(float(moment_weight), 0.0)
        self.moment_extent_weight = max(float(moment_extent_weight), 0.0)
        self.profile_weight = max(float(profile_weight), 0.0)
        self.profile_bins = max(int(profile_bins), 2)
        self.profile_trim = min(max(float(profile_trim), 0.0), 0.45)
        self.anchor_weight = max(float(anchor_weight), 0.0)
        self.anchor_geometry_sigma = max(float(anchor_geometry_sigma), 1e-6)
        self.anchor_semantic_weight = max(float(anchor_semantic_weight), 0.0)
        self.balanced_sinkhorn_iters = max(int(balanced_sinkhorn_iters), 1)

        self.register_buffer("source_labels", torch.tensor(source_bucket_labels, dtype=torch.long, device=source_vertices.device))
        self.register_buffer("target_labels", torch.tensor(target_bucket_labels, dtype=torch.long, device=target_vertices.device))
        self.register_buffer("target_vertices", unit_vertices(target_vertices.detach()))
        target_lower, target_upper = self._compute_target_bucket_bounds()
        self.register_buffer("target_bucket_lower", target_lower)
        self.register_buffer("target_bucket_upper", target_upper)
        target_centroid, target_spread = self._compute_target_bucket_moments()
        self.register_buffer("target_bucket_centroid", target_centroid)
        self.register_buffer("target_bucket_spread", target_spread)
        target_profiles = self._compute_target_bucket_profiles()
        self.register_buffer("target_bucket_profiles", target_profiles)
        if source_semantic_features is not None and target_semantic_features is not None:
            self.register_buffer(
                "source_semantic_features",
                normalize_torch_rows(torch.tensor(source_semantic_features, dtype=torch.float32, device=source_vertices.device)),
            )
            self.register_buffer(
                "target_semantic_features",
                normalize_torch_rows(torch.tensor(target_semantic_features, dtype=torch.float32, device=target_vertices.device)),
            )
            self.has_soft_features = True
        else:
            self.register_buffer("source_semantic_features", torch.empty(0, 0, device=source_vertices.device))
            self.register_buffer("target_semantic_features", torch.empty(0, 0, device=target_vertices.device))
            self.has_soft_features = False

        if self.guidance_mode in {"soft", "hybrid"} and self.soft_weight > 0 and not self.has_soft_features:
            raise ValueError("Soft/Hybrid PartField guidance needs source and target PartField feature arrays.")

        self.active_bucket_ids = self._compute_active_bucket_ids()
        anchor_targets, anchor_mask = self._compute_source_anchor_targets(source_vertices.detach())
        self.register_buffer("source_anchor_targets", anchor_targets)
        self.register_buffer("source_anchor_mask", anchor_mask)

    def _compute_active_bucket_ids(self) -> List[int]:
        active = []
        for bucket_id in range(self.n_buckets):
            src_count = int((self.source_labels == bucket_id).sum().item())
            tgt_count = int((self.target_labels == bucket_id).sum().item())
            if src_count >= self.min_points_per_bucket and tgt_count >= self.min_points_per_bucket:
                active.append(bucket_id)
        return active

    def bucket_counts(self) -> Dict[str, Dict[str, int]]:
        return {
            f"bucket_{bucket_id:02d}": {
                "source": int((self.source_labels == bucket_id).sum().item()),
                "target": int((self.target_labels == bucket_id).sum().item()),
                "active": bucket_id in self.active_bucket_ids,
            }
            for bucket_id in range(self.n_buckets)
        }

    def forward(self, current_vertices: torch.Tensor, return_components: bool = False):
        source_vertices = unit_vertices(current_vertices)
        bucket_losses = []
        bucket_components = {}

        use_hard = self.guidance_mode in {"hard", "hybrid", "hard_unbalanced"} and self.hard_weight > 0
        use_balanced = self.guidance_mode == "balanced" and self.hard_weight > 0
        use_unbalanced = self.guidance_mode == "hard_unbalanced" and self.unbalanced_transport_weight > 0
        use_soft = self.guidance_mode in {"soft", "hybrid"} and self.soft_weight > 0

        if use_hard or use_balanced:
            for bucket_id in self.active_bucket_ids:
                src_mask = self.source_labels == bucket_id
                tgt_mask = self.target_labels == bucket_id
                if use_balanced:
                    loss = self._balanced_bucket_transport(source_vertices, bucket_id)
                else:
                    loss = self._hard_bucket_chamfer(source_vertices, bucket_id, src_mask, tgt_mask)
                    if use_unbalanced:
                        loss = loss + self.unbalanced_transport_weight * self._unbalanced_bucket_transport(
                            source_vertices, bucket_id
                        )
                bucket_losses.append(loss)
                bucket_components[f"bucket_{bucket_id:02d}"] = loss.detach()

        if bucket_losses:
            bucket_raw = torch.stack(bucket_losses).mean()
        else:
            bucket_raw = current_vertices.sum() * 0.0

        soft_raw = current_vertices.sum() * 0.0
        if use_soft:
            soft_raw = self._soft_feature_chamfer(source_vertices)

        global_raw = current_vertices.sum() * 0.0
        if self.global_weight > 0:
            global_raw = symmetric_chamfer(
                source_vertices,
                self.target_vertices,
                self.points_per_bucket,
                source_to_target_weight=self.source_to_target_weight,
                target_to_source_weight=self.target_to_source_weight,
                tgt_to_src_robust_scale=self.tgt_to_src_robust_scale,
                src_to_tgt_unmatched_weight=self.src_to_tgt_unmatched_weight,
            )

        containment_raw = current_vertices.sum() * 0.0
        if self.containment_weight > 0:
            containment_raw = self._bucket_containment_loss(source_vertices)

        moment_raw = current_vertices.sum() * 0.0
        if self.moment_weight > 0:
            moment_raw = self._bucket_moment_loss(source_vertices)

        profile_raw = current_vertices.sum() * 0.0
        if self.profile_weight > 0:
            profile_raw = self._bucket_profile_loss(source_vertices)

        anchor_raw = current_vertices.sum() * 0.0
        if self.anchor_weight > 0:
            anchor_raw = self._bucket_anchor_loss(source_vertices)

        raw = (
            self.hard_weight * bucket_raw
            + self.soft_weight * soft_raw
            + self.global_weight * global_raw
            + self.containment_weight * containment_raw
            + self.moment_weight * moment_raw
            + self.profile_weight * profile_raw
            + self.anchor_weight * anchor_raw
        )
        if not return_components:
            return raw

        return {
            "raw": raw,
            "bucket_raw": bucket_raw.detach(),
            "soft_raw": soft_raw.detach(),
            "global_raw": global_raw.detach(),
            "containment_raw": containment_raw.detach(),
            "moment_raw": moment_raw.detach(),
            "profile_raw": profile_raw.detach(),
            "anchor_raw": anchor_raw.detach(),
            "buckets": bucket_components,
            "active_buckets": len(self.active_bucket_ids),
        }

    def _hard_bucket_chamfer(
        self,
        source_vertices: torch.Tensor,
        bucket_id: int,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_features = None
        tgt_features = None
        if self.has_soft_features:
            src_features = self.source_semantic_features[src_mask]
            tgt_features = self.target_semantic_features[tgt_mask]
        return feature_guided_symmetric_chamfer(
            source_vertices[src_mask],
            self.target_vertices[tgt_mask],
            self.points_per_bucket,
            source_to_target_weight=self.source_to_target_weight,
            target_to_source_weight=self.target_to_source_weight,
            tgt_to_src_robust_scale=self.tgt_to_src_robust_scale,
            src_to_tgt_unmatched_weight=self.src_to_tgt_unmatched_weight,
            src_features=src_features,
            tgt_features=tgt_features,
            semantic_weight=self.hard_semantic_weight,
            geometry_sigma=self.hard_geometry_sigma,
            confidence_min_similarity=self.semantic_confidence_min_similarity,
            confidence_margin=self.semantic_confidence_margin,
            confidence_floor=self.semantic_confidence_floor,
            confidence_power=self.semantic_confidence_power,
        )

    def _compute_target_bucket_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lowers = []
        uppers = []
        fallback_lower = self.target_vertices.amin(dim=0)
        fallback_upper = self.target_vertices.amax(dim=0)
        for bucket_id in range(self.n_buckets):
            tgt_mask = self.target_labels == bucket_id
            if tgt_mask.any():
                target_bucket = self.target_vertices[tgt_mask]
                lowers.append(target_bucket.amin(dim=0))
                uppers.append(target_bucket.amax(dim=0))
            else:
                lowers.append(fallback_lower)
                uppers.append(fallback_upper)
        return torch.stack(lowers, dim=0), torch.stack(uppers, dim=0)

    def _compute_target_bucket_moments(self) -> Tuple[torch.Tensor, torch.Tensor]:
        centroids = []
        spreads = []
        fallback_centroid = self.target_vertices.mean(dim=0)
        fallback_spread = (self.target_vertices - fallback_centroid).pow(2).mean(dim=0).clamp(min=1e-12).sqrt()
        for bucket_id in range(self.n_buckets):
            tgt_mask = self.target_labels == bucket_id
            if tgt_mask.any():
                target_bucket = self.target_vertices[tgt_mask]
                centroid = target_bucket.mean(dim=0)
                centroids.append(centroid)
                spreads.append((target_bucket - centroid).pow(2).mean(dim=0).clamp(min=1e-12).sqrt())
            else:
                centroids.append(fallback_centroid)
                spreads.append(fallback_spread)
        return torch.stack(centroids, dim=0), torch.stack(spreads, dim=0)

    def _compute_target_bucket_profiles(self) -> torch.Tensor:
        profiles = []
        fallback_profile = self._axis_quantile_profile(self.target_vertices)
        for bucket_id in range(self.n_buckets):
            tgt_mask = self.target_labels == bucket_id
            if tgt_mask.any():
                profiles.append(self._axis_quantile_profile(self.target_vertices[tgt_mask]))
            else:
                profiles.append(fallback_profile)
        return torch.stack(profiles, dim=0)

    def _axis_quantile_profile(self, vertices: torch.Tensor) -> torch.Tensor:
        sorted_values = torch.sort(vertices, dim=0).values
        n_values = sorted_values.shape[0]
        if n_values == 1:
            return sorted_values[0].view(3, 1).expand(3, self.profile_bins)

        q = torch.linspace(
            self.profile_trim,
            1.0 - self.profile_trim,
            steps=self.profile_bins,
            device=vertices.device,
            dtype=vertices.dtype,
        )
        positions = q * float(n_values - 1)
        lower_idx = torch.floor(positions).long().clamp(0, n_values - 1)
        upper_idx = torch.ceil(positions).long().clamp(0, n_values - 1)
        blend = (positions - lower_idx.to(vertices.dtype)).view(-1, 1)
        lower = sorted_values.index_select(0, lower_idx)
        upper = sorted_values.index_select(0, upper_idx)
        return ((1.0 - blend) * lower + blend * upper).transpose(0, 1)

    def _bucket_containment_loss(self, source_vertices: torch.Tensor) -> torch.Tensor:
        losses = []
        for bucket_id in self.active_bucket_ids:
            src_mask = self.source_labels == bucket_id
            src = source_vertices[src_mask]
            if src.numel() == 0:
                continue
            lower = self.target_bucket_lower[bucket_id] - self.containment_margin
            upper = self.target_bucket_upper[bucket_id] + self.containment_margin
            outside = torch.relu(lower - src) + torch.relu(src - upper)
            per_vertex = outside.pow(2).sum(dim=1)
            losses.append(per_vertex.mean() + self.containment_max_weight * per_vertex.max())
        if not losses:
            return source_vertices.sum() * 0.0
        return torch.stack(losses).mean()

    def _bucket_moment_loss(self, source_vertices: torch.Tensor) -> torch.Tensor:
        losses = []
        for bucket_id in self.active_bucket_ids:
            src_mask = self.source_labels == bucket_id
            src = source_vertices[src_mask]
            if src.numel() == 0:
                continue
            src_centroid = src.mean(dim=0)
            centroid_loss = (src_centroid - self.target_bucket_centroid[bucket_id]).pow(2).sum()
            src_spread = (src - src_centroid).pow(2).mean(dim=0).clamp(min=1e-12).sqrt()
            spread_loss = (src_spread - self.target_bucket_spread[bucket_id]).pow(2).sum()
            losses.append(centroid_loss + self.moment_extent_weight * spread_loss)
        if not losses:
            return source_vertices.sum() * 0.0
        return torch.stack(losses).mean()

    def _bucket_profile_loss(self, source_vertices: torch.Tensor) -> torch.Tensor:
        losses = []
        for bucket_id in self.active_bucket_ids:
            src_mask = self.source_labels == bucket_id
            src = source_vertices[src_mask]
            if src.numel() == 0:
                continue
            src_profile = self._axis_quantile_profile(src)
            target_profile = self.target_bucket_profiles[bucket_id]
            losses.append((src_profile - target_profile).pow(2).mean())
        if not losses:
            return source_vertices.sum() * 0.0
        return torch.stack(losses).mean()

    def _compute_source_anchor_targets(self, source_vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        source_unit = unit_vertices(source_vertices)
        anchor_targets = torch.zeros_like(source_unit)
        anchor_mask = torch.zeros(source_unit.shape[0], dtype=torch.bool, device=source_unit.device)
        for bucket_id in self.active_bucket_ids:
            src_idx = torch.nonzero(self.source_labels == bucket_id, as_tuple=False).flatten()
            tgt_idx = torch.nonzero(self.target_labels == bucket_id, as_tuple=False).flatten()
            if src_idx.numel() == 0 or tgt_idx.numel() == 0:
                continue

            src = source_unit.index_select(0, src_idx)
            tgt = self.target_vertices.index_select(0, tgt_idx)
            match_cost = torch.cdist(src, tgt, p=2).pow(2) / (self.anchor_geometry_sigma ** 2)
            if self.has_soft_features and self.anchor_semantic_weight > 0:
                src_features = self.source_semantic_features.index_select(0, src_idx)
                tgt_features = self.target_semantic_features.index_select(0, tgt_idx)
                semantic_dists = torch.clamp(1.0 - src_features @ tgt_features.t(), min=0.0)
                match_cost = match_cost + self.anchor_semantic_weight * semantic_dists

            nearest = match_cost.argmin(dim=1)
            anchor_targets[src_idx] = tgt.index_select(0, nearest)
            anchor_mask[src_idx] = True
        return anchor_targets, anchor_mask

    def _bucket_anchor_loss(self, source_vertices: torch.Tensor) -> torch.Tensor:
        if not bool(self.source_anchor_mask.any()):
            return source_vertices.sum() * 0.0
        src = source_vertices[self.source_anchor_mask]
        tgt = self.source_anchor_targets[self.source_anchor_mask]
        return (src - tgt).pow(2).sum(dim=1).mean()

    def _balanced_bucket_transport(self, source_vertices: torch.Tensor, bucket_id: int) -> torch.Tensor:
        src_full_idx = torch.nonzero(self.source_labels == bucket_id, as_tuple=False).flatten()
        tgt_full_idx = torch.nonzero(self.target_labels == bucket_id, as_tuple=False).flatten()
        if src_full_idx.numel() == 0 or tgt_full_idx.numel() == 0:
            return source_vertices.sum() * 0.0

        src_sample = sample_indices(src_full_idx.numel(), self.points_per_bucket, source_vertices.device)
        tgt_sample = sample_indices(tgt_full_idx.numel(), self.points_per_bucket, source_vertices.device)
        src_idx = src_full_idx.index_select(0, src_sample)
        tgt_idx = tgt_full_idx.index_select(0, tgt_sample)

        src = source_vertices.index_select(0, src_idx)
        tgt = self.target_vertices.index_select(0, tgt_idx)
        geo_dists = torch.cdist(src, tgt, p=2).pow(2)

        matching_cost = geo_dists.detach() / (self.soft_geometry_sigma ** 2)
        if self.has_soft_features and self.soft_match_space in {"latent", "hybrid"}:
            src_features = self.source_semantic_features.index_select(0, src_idx)
            tgt_features = self.target_semantic_features.index_select(0, tgt_idx)
            semantic_dists = torch.clamp(1.0 - src_features @ tgt_features.t(), min=0.0)
            if self.soft_match_space == "latent":
                matching_cost = self.soft_semantic_weight * semantic_dists
            else:
                matching_cost = matching_cost + self.soft_semantic_weight * semantic_dists

        transport = self._sinkhorn_transport(matching_cost, self.soft_temperature, self.balanced_sinkhorn_iters)
        return 2.0 * (transport * geo_dists).sum()

    def _unbalanced_bucket_transport(self, source_vertices: torch.Tensor, bucket_id: int) -> torch.Tensor:
        src_idx, tgt_idx, geo_dists, matching_cost = self._bucket_transport_costs(source_vertices, bucket_id)
        if src_idx is None or tgt_idx is None:
            return source_vertices.sum() * 0.0

        transport = self._unbalanced_sinkhorn_transport(
            matching_cost,
            self.soft_temperature,
            self.unbalanced_transport_rho,
            self.balanced_sinkhorn_iters,
        )
        mass = transport.sum().clamp(min=1e-8)
        return 2.0 * (transport * geo_dists).sum() / mass

    def _bucket_transport_costs(
        self,
        source_vertices: torch.Tensor,
        bucket_id: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        src_full_idx = torch.nonzero(self.source_labels == bucket_id, as_tuple=False).flatten()
        tgt_full_idx = torch.nonzero(self.target_labels == bucket_id, as_tuple=False).flatten()
        if src_full_idx.numel() == 0 or tgt_full_idx.numel() == 0:
            empty = source_vertices.new_empty(0, 0)
            return None, None, empty, empty

        src_sample = sample_indices(src_full_idx.numel(), self.points_per_bucket, source_vertices.device)
        tgt_sample = sample_indices(tgt_full_idx.numel(), self.points_per_bucket, source_vertices.device)
        src_idx = src_full_idx.index_select(0, src_sample)
        tgt_idx = tgt_full_idx.index_select(0, tgt_sample)

        src = source_vertices.index_select(0, src_idx)
        tgt = self.target_vertices.index_select(0, tgt_idx)
        geo_dists = torch.cdist(src, tgt, p=2).pow(2)

        matching_cost = geo_dists.detach() / (self.soft_geometry_sigma ** 2)
        if self.has_soft_features and self.soft_match_space in {"latent", "hybrid"}:
            src_features = self.source_semantic_features.index_select(0, src_idx)
            tgt_features = self.target_semantic_features.index_select(0, tgt_idx)
            semantic_dists = torch.clamp(1.0 - src_features @ tgt_features.t(), min=0.0)
            if self.soft_match_space == "latent":
                matching_cost = self.soft_semantic_weight * semantic_dists
            else:
                matching_cost = matching_cost + self.soft_semantic_weight * semantic_dists

        return src_idx, tgt_idx, geo_dists, matching_cost

    @staticmethod
    def _sinkhorn_transport(cost: torch.Tensor, temperature: float, n_iters: int) -> torch.Tensor:
        eps = max(float(temperature), 1e-6)
        n_src, n_tgt = cost.shape
        log_kernel = -cost / eps
        log_mu = cost.new_full((n_src,), -np.log(float(n_src)))
        log_nu = cost.new_full((n_tgt,), -np.log(float(n_tgt)))
        log_u = torch.zeros_like(log_mu)
        log_v = torch.zeros_like(log_nu)
        for _ in range(n_iters):
            log_u = log_mu - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
            log_v = log_nu - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
        return torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))

    @staticmethod
    def _unbalanced_sinkhorn_transport(
        cost: torch.Tensor,
        temperature: float,
        marginal_relaxation: float,
        n_iters: int,
    ) -> torch.Tensor:
        eps = max(float(temperature), 1e-6)
        rho = max(float(marginal_relaxation), 1e-6)
        tau = rho / (rho + eps)
        n_src, n_tgt = cost.shape
        log_kernel = -cost / eps
        log_mu = cost.new_full((n_src,), -np.log(float(n_src)))
        log_nu = cost.new_full((n_tgt,), -np.log(float(n_tgt)))
        log_u = torch.zeros_like(log_mu)
        log_v = torch.zeros_like(log_nu)
        for _ in range(n_iters):
            log_u = tau * (log_mu - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1))
            log_v = tau * (log_nu - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0))
        return torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))

    def _soft_feature_chamfer(self, source_vertices: torch.Tensor) -> torch.Tensor:
        src_idx = sample_indices(source_vertices.shape[0], self.soft_points, source_vertices.device)
        tgt_idx = sample_indices(self.target_vertices.shape[0], self.soft_points, self.target_vertices.device)
        src_vertices = source_vertices.index_select(0, src_idx)
        tgt_vertices = self.target_vertices.index_select(0, tgt_idx)
        src_features = self.source_semantic_features.index_select(0, src_idx)
        tgt_features = self.target_semantic_features.index_select(0, tgt_idx)

        geo_dists = torch.cdist(src_vertices, tgt_vertices, p=2).pow(2)
        semantic_dists = torch.clamp(1.0 - src_features @ tgt_features.t(), min=0.0)
        scaled_geo_dists = geo_dists.detach() / (self.soft_geometry_sigma ** 2)
        if self.soft_match_space == "latent":
            matching_cost = self.soft_semantic_weight * semantic_dists
        elif self.soft_match_space == "geometry":
            matching_cost = scaled_geo_dists
        else:
            matching_cost = scaled_geo_dists + self.soft_semantic_weight * semantic_dists

        src_to_tgt = torch.softmax(-matching_cost / self.soft_temperature, dim=1)
        tgt_to_src = torch.softmax(-matching_cost / self.soft_temperature, dim=0)
        src_values = (src_to_tgt * geo_dists).sum(dim=1)
        tgt_values = (tgt_to_src * geo_dists).sum(dim=0)
        src_weights = semantic_confidence_from_similarity(
            1.0 - semantic_dists,
            dim=1,
            min_similarity=self.semantic_confidence_min_similarity,
            margin=self.semantic_confidence_margin,
            floor=self.semantic_confidence_floor,
            power=self.semantic_confidence_power,
        )
        tgt_weights = semantic_confidence_from_similarity(
            1.0 - semantic_dists,
            dim=0,
            min_similarity=self.semantic_confidence_min_similarity,
            margin=self.semantic_confidence_margin,
            floor=self.semantic_confidence_floor,
            power=self.semantic_confidence_power,
        )
        return (
            self.source_to_target_weight * weighted_mean(src_values, src_weights)
            + self.target_to_source_weight * weighted_mean(tgt_values, tgt_weights)
        )


def _load_cached_labels(cache_path: Path, n_source: int, n_target: int) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    if not cache_path.exists():
        return None
    data = np.load(cache_path)
    source_labels = data["source_labels"]
    target_labels = data["target_labels"]
    if source_labels.shape[0] != n_source or target_labels.shape[0] != n_target:
        return None
    n_buckets = int(data["n_buckets"])
    print(f"Loaded PartField bucket cache: {cache_path}")
    return source_labels.astype(np.int64), target_labels.astype(np.int64), n_buckets


def save_debug(
    debug_path: Optional[str],
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    n_buckets: int,
    counts: Dict[str, Dict[str, int]],
    target_to_source: Dict[int, int],
    metadata: Dict[str, object],
) -> None:
    if not debug_path:
        return
    path = Path(debug_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        source_labels=source_labels.astype(np.int64),
        target_labels=target_labels.astype(np.int64),
        n_buckets=np.array(n_buckets, dtype=np.int64),
    )
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(
            {
                "n_buckets": n_buckets,
                "target_to_source_bucket": {str(k): int(v) for k, v in target_to_source.items()},
                "counts": counts,
                "metadata": metadata,
            },
            f,
            indent=2,
        )


def create_partfield_chamfer_loss(
    source_vertices: torch.Tensor,
    source_faces: torch.Tensor,
    target_vertices: torch.Tensor,
    target_faces: torch.Tensor,
    source_features_path: Optional[str] = None,
    target_features_path: Optional[str] = None,
    source_labels_path: Optional[str] = None,
    target_labels_path: Optional[str] = None,
    feature_mode: FeatureMode = "auto",
    label_mode: LabelMode = "auto",
    n_buckets: int = 12,
    points_per_bucket: int = 512,
    min_points_per_bucket: int = 12,
    global_weight: float = 0.0,
    position_weight: float = 0.05,
    normal_weight: float = 0.0,
    labels_aligned: bool = False,
    cache_path: Optional[str] = None,
    debug_path: Optional[str] = None,
    random_state: int = 42,
    guidance_mode: GuidanceMode = "hard",
    hard_weight: float = 1.0,
    source_to_target_weight: float = 1.0,
    target_to_source_weight: float = 1.0,
    tgt_to_src_robust_scale: float = 0.0,
    src_to_tgt_unmatched_weight: float = 1.0,
    hard_semantic_weight: float = 0.0,
    hard_geometry_sigma: float = 1.0,
    semantic_confidence_min_similarity: float = -1.0,
    semantic_confidence_margin: float = 0.0,
    semantic_confidence_floor: float = 1.0,
    semantic_confidence_power: float = 1.0,
    unbalanced_transport_weight: float = 0.0,
    unbalanced_transport_rho: float = 0.30,
    soft_weight: float = 1.0,
    soft_points: Optional[int] = None,
    soft_semantic_weight: float = 0.10,
    soft_temperature: float = 0.03,
    soft_match_space: SoftMatchSpace = "hybrid",
    soft_geometry_sigma: float = 1.0,
    containment_weight: float = 0.0,
    containment_margin: float = 0.02,
    containment_max_weight: float = 1.0,
    moment_weight: float = 0.0,
    moment_extent_weight: float = 0.5,
    profile_weight: float = 0.0,
    profile_bins: int = 9,
    profile_trim: float = 0.08,
    anchor_weight: float = 0.0,
    anchor_geometry_sigma: float = 0.35,
    anchor_semantic_weight: float = 0.0,
    balanced_sinkhorn_iters: int = 30,
) -> PartFieldChamferLoss:
    cache = Path(cache_path) if cache_path else None
    cached = None
    if cache is not None:
        cached = _load_cached_labels(cache, source_vertices.shape[0], target_vertices.shape[0])

    target_to_source: Dict[int, int] = {}
    source_features = None
    target_features = None
    has_features = bool(source_features_path) and bool(target_features_path)
    needs_soft_features = guidance_mode in {"soft", "hybrid"} and float(soft_weight) > 0
    if needs_soft_features and not has_features:
        raise ValueError("Soft/Hybrid PartField guidance requires partfield_source_features and partfield_target_features.")

    if has_features:
        source_features = coerce_features_to_vertices(
            source_features_path,
            n_vertices=source_vertices.shape[0],
            faces=source_faces,
            mode=feature_mode,
        )
        target_features = coerce_features_to_vertices(
            target_features_path,
            n_vertices=target_vertices.shape[0],
            faces=target_faces,
            mode=feature_mode,
        )

    metadata: Dict[str, object] = {
        "source_features_path": source_features_path,
        "target_features_path": target_features_path,
        "source_labels_path": source_labels_path,
        "target_labels_path": target_labels_path,
        "feature_mode": feature_mode,
        "label_mode": label_mode,
        "requested_n_buckets": int(n_buckets),
        "position_weight": float(position_weight),
        "normal_weight": float(normal_weight),
        "labels_aligned": bool(labels_aligned),
        "guidance_mode": guidance_mode,
        "hard_weight": float(hard_weight),
        "source_to_target_weight": float(source_to_target_weight),
        "target_to_source_weight": float(target_to_source_weight),
        "tgt_to_src_robust_scale": float(tgt_to_src_robust_scale),
        "src_to_tgt_unmatched_weight": float(src_to_tgt_unmatched_weight),
        "hard_semantic_weight": float(hard_semantic_weight),
        "hard_geometry_sigma": float(hard_geometry_sigma),
        "semantic_confidence_min_similarity": float(semantic_confidence_min_similarity),
        "semantic_confidence_margin": float(semantic_confidence_margin),
        "semantic_confidence_floor": float(semantic_confidence_floor),
        "semantic_confidence_power": float(semantic_confidence_power),
        "unbalanced_transport_weight": float(unbalanced_transport_weight),
        "unbalanced_transport_rho": float(unbalanced_transport_rho),
        "soft_weight": float(soft_weight),
        "soft_points": int(soft_points or points_per_bucket),
        "soft_semantic_weight": float(soft_semantic_weight),
        "soft_temperature": float(soft_temperature),
        "soft_match_space": soft_match_space,
        "soft_geometry_sigma": float(soft_geometry_sigma),
        "containment_weight": float(containment_weight),
        "containment_margin": float(containment_margin),
        "containment_max_weight": float(containment_max_weight),
        "moment_weight": float(moment_weight),
        "moment_extent_weight": float(moment_extent_weight),
        "profile_weight": float(profile_weight),
        "profile_bins": int(profile_bins),
        "profile_trim": float(profile_trim),
        "anchor_weight": float(anchor_weight),
        "anchor_geometry_sigma": float(anchor_geometry_sigma),
        "anchor_semantic_weight": float(anchor_semantic_weight),
        "balanced_sinkhorn_iters": int(balanced_sinkhorn_iters),
    }

    if cached is None:
        has_source_labels = bool(source_labels_path)
        has_target_labels = bool(target_labels_path)
        used_aligned_labels = False

        if labels_aligned and has_source_labels and has_target_labels:
            source_raw_labels = coerce_labels_to_vertices(
                source_labels_path,
                n_vertices=source_vertices.shape[0],
                faces=source_faces,
                mode=label_mode,
            )
            target_raw_labels = coerce_labels_to_vertices(
                target_labels_path,
                n_vertices=target_vertices.shape[0],
                faces=target_faces,
                mode=label_mode,
            )
            source_labels, target_labels, aligned_mapping = remap_aligned_label_pair(source_raw_labels, target_raw_labels)
            actual_n_buckets = len(aligned_mapping)
            target_to_source = {old_label: new_label for old_label, new_label in aligned_mapping.items()}
            used_aligned_labels = True
        elif has_source_labels:
            source_raw_labels = coerce_labels_to_vertices(
                source_labels_path,
                n_vertices=source_vertices.shape[0],
                faces=source_faces,
                mode=label_mode,
            )
            source_labels, source_mapping = remap_to_contiguous(source_raw_labels, min_points=min_points_per_bucket)
            actual_n_buckets = len(set(source_mapping.values()))
        elif has_features:
            source_cluster_features = clustering_features(
                source_features,
                source_vertices,
                source_faces,
                position_weight=position_weight,
                normal_weight=normal_weight,
            )
            actual_n_buckets = min(int(n_buckets), source_cluster_features.shape[0])
            if actual_n_buckets < 1:
                raise ValueError("PartField clustering needs at least one source vertex.")
            kmeans = KMeans(n_clusters=actual_n_buckets, random_state=random_state, n_init=10)
            source_labels = kmeans.fit_predict(source_cluster_features).astype(np.int64)
        else:
            raise ValueError(
                "PartField Chamfer needs either source/target feature paths or at least source/target label paths."
            )

        if not used_aligned_labels:
            if has_features and not has_source_labels:
                target_cluster_features = clustering_features(
                    target_features,
                    target_vertices,
                    target_faces,
                    position_weight=position_weight,
                    normal_weight=normal_weight,
                )
                target_labels = kmeans.predict(target_cluster_features).astype(np.int64)
            elif has_features:
                target_raw_labels = None
                if target_labels_path:
                    target_raw_labels = coerce_labels_to_vertices(
                        target_labels_path,
                        n_vertices=target_vertices.shape[0],
                        faces=target_faces,
                        mode=label_mode,
                    )
                target_labels, target_to_source = assign_target_by_feature_centroids(
                    source_features=source_features,
                    target_features=target_features,
                    source_bucket_labels=source_labels,
                    target_raw_labels=target_raw_labels,
                    n_source_buckets=actual_n_buckets,
                )
            else:
                if not target_labels_path:
                    raise ValueError("PartField label-only mode also needs target_labels_path.")
                target_raw_labels = coerce_labels_to_vertices(
                    target_labels_path,
                    n_vertices=target_vertices.shape[0],
                    faces=target_faces,
                    mode=label_mode,
                )
                target_labels, target_to_source = match_target_labels_by_geometry(
                    source_vertices=source_vertices,
                    target_vertices=target_vertices,
                    source_bucket_labels=source_labels,
                    target_raw_labels=target_raw_labels,
                    n_source_buckets=actual_n_buckets,
                )
    else:
        source_labels, target_labels, actual_n_buckets = cached

    loss_fn = PartFieldChamferLoss(
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        source_bucket_labels=source_labels,
        target_bucket_labels=target_labels,
        n_buckets=actual_n_buckets,
        points_per_bucket=points_per_bucket,
        min_points_per_bucket=min_points_per_bucket,
        global_weight=global_weight,
        guidance_mode=guidance_mode,
        hard_weight=hard_weight,
        source_to_target_weight=source_to_target_weight,
        target_to_source_weight=target_to_source_weight,
        tgt_to_src_robust_scale=tgt_to_src_robust_scale,
        src_to_tgt_unmatched_weight=src_to_tgt_unmatched_weight,
        hard_semantic_weight=hard_semantic_weight,
        hard_geometry_sigma=hard_geometry_sigma,
        semantic_confidence_min_similarity=semantic_confidence_min_similarity,
        semantic_confidence_margin=semantic_confidence_margin,
        semantic_confidence_floor=semantic_confidence_floor,
        semantic_confidence_power=semantic_confidence_power,
        unbalanced_transport_weight=unbalanced_transport_weight,
        unbalanced_transport_rho=unbalanced_transport_rho,
        soft_weight=soft_weight,
        soft_points=soft_points,
        soft_semantic_weight=soft_semantic_weight,
        soft_temperature=soft_temperature,
        soft_match_space=soft_match_space,
        soft_geometry_sigma=soft_geometry_sigma,
        containment_weight=containment_weight,
        containment_margin=containment_margin,
        containment_max_weight=containment_max_weight,
        moment_weight=moment_weight,
        moment_extent_weight=moment_extent_weight,
        profile_weight=profile_weight,
        profile_bins=profile_bins,
        profile_trim=profile_trim,
        anchor_weight=anchor_weight,
        anchor_geometry_sigma=anchor_geometry_sigma,
        anchor_semantic_weight=anchor_semantic_weight,
        balanced_sinkhorn_iters=balanced_sinkhorn_iters,
        source_semantic_features=source_features,
        target_semantic_features=target_features,
    )

    debug_target = debug_path or cache_path
    if debug_target and cached is None:
        save_debug(
            debug_target,
            source_labels=source_labels,
            target_labels=target_labels,
            n_buckets=actual_n_buckets,
            counts=loss_fn.bucket_counts(),
            target_to_source=target_to_source,
            metadata=metadata,
        )

    return loss_fn
