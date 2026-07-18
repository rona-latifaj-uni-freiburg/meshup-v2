"""Refine aligned DenseCorr3D labels without crossing parent semantic groups.

The DenseCorr3D animal labels used by the cross-animal jobs have only a few
large groups.  This helper subdivides each parent group using source+target
unit-box coordinates, so labels remain semantically parent-aligned while giving
the Chamfer loss smaller local regions to match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans


def load_obj(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with path.open("r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                idx = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def load_labels(path: Path) -> np.ndarray:
    data = np.load(path)
    for key in ("labels", "vertex_labels", "part_labels", "source_labels", "target_labels"):
        if key in data:
            return np.asarray(data[key], dtype=np.int64).reshape(-1)
    return np.asarray(data[data.files[0]], dtype=np.int64).reshape(-1)


def unit_vertices(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    scale = 2.0 / max(float((vmax - vmin).max()), 1e-12)
    return (vertices - (vmax + vmin) * 0.5) * scale


def one_hot(labels: np.ndarray, n_labels: int) -> np.ndarray:
    features = np.zeros((labels.shape[0], n_labels), dtype=np.float32)
    features[np.arange(labels.shape[0]), labels] = 1.0
    return features


def stable_cluster_order(centers: np.ndarray) -> Dict[int, int]:
    # Sort primarily along the animal long axis, then vertical, then lateral.
    order = sorted(range(centers.shape[0]), key=lambda i: (centers[i, 0], centers[i, 1], centers[i, 2]))
    return {old: new for new, old in enumerate(order)}


def quantile_labels(values: np.ndarray, n_parts: int) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    labels = np.empty(values.shape[0], dtype=np.int64)
    ranks = np.arange(values.shape[0], dtype=np.int64)
    labels[order] = np.minimum((ranks * n_parts) // max(values.shape[0], 1), n_parts - 1)
    return labels


def split_parent_group_by_quantile(
    source_points: np.ndarray,
    target_points: np.ndarray,
    requested_parts: int,
    min_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_parts = min(requested_parts, source_points.shape[0] // min_count, target_points.shape[0] // min_count)
    n_parts = max(1, int(n_parts))
    if n_parts == 1:
        return (
            np.zeros(source_points.shape[0], dtype=np.int64),
            np.zeros(target_points.shape[0], dtype=np.int64),
            np.mean(np.vstack([source_points, target_points]), axis=0, keepdims=True),
        )

    axis = int(np.ptp(target_points, axis=0).argmax())
    src_labels = quantile_labels(source_points[:, axis], n_parts)
    tgt_labels = quantile_labels(target_points[:, axis], n_parts)
    centers = []
    for local_id in range(n_parts):
        pieces = []
        if (src_labels == local_id).any():
            pieces.append(source_points[src_labels == local_id])
        if (tgt_labels == local_id).any():
            pieces.append(target_points[tgt_labels == local_id])
        centers.append(np.vstack(pieces).mean(axis=0))
    return src_labels, tgt_labels, np.asarray(centers, dtype=np.float32)


def split_parent_group(
    source_points: np.ndarray,
    target_points: np.ndarray,
    requested_parts: int,
    min_count: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_parts = min(requested_parts, source_points.shape[0] // min_count, target_points.shape[0] // min_count)
    max_parts = max(1, int(max_parts))

    for n_parts in range(max_parts, 0, -1):
        if n_parts == 1:
            return (
                np.zeros(source_points.shape[0], dtype=np.int64),
                np.zeros(target_points.shape[0], dtype=np.int64),
                np.mean(np.vstack([source_points, target_points]), axis=0, keepdims=True),
            )

        combined = np.vstack([source_points, target_points])
        kmeans = KMeans(n_clusters=n_parts, random_state=random_state, n_init=20)
        combined_labels = kmeans.fit_predict(combined).astype(np.int64)
        src_labels = combined_labels[: source_points.shape[0]]
        tgt_labels = combined_labels[source_points.shape[0] :]

        src_counts = np.bincount(src_labels, minlength=n_parts)
        tgt_counts = np.bincount(tgt_labels, minlength=n_parts)
        if (src_counts >= min_count).all() and (tgt_counts >= min_count).all():
            remap = stable_cluster_order(kmeans.cluster_centers_)
            src_labels = np.asarray([remap[int(label)] for label in src_labels], dtype=np.int64)
            tgt_labels = np.asarray([remap[int(label)] for label in tgt_labels], dtype=np.int64)
            centers = np.zeros_like(kmeans.cluster_centers_)
            for old, new in remap.items():
                centers[new] = kmeans.cluster_centers_[old]
            return src_labels, tgt_labels, centers

    return (
        np.zeros(source_points.shape[0], dtype=np.int64),
        np.zeros(target_points.shape[0], dtype=np.int64),
        np.mean(np.vstack([source_points, target_points]), axis=0, keepdims=True),
    )


def refine_pair(args: argparse.Namespace) -> None:
    source_vertices, _ = load_obj(Path(args.source_mesh))
    target_vertices, _ = load_obj(Path(args.target_mesh))
    source_labels = load_labels(Path(args.source_labels))
    target_labels = load_labels(Path(args.target_labels))

    if source_vertices.shape[0] != source_labels.shape[0]:
        raise ValueError("source label count does not match source vertex count")
    if target_vertices.shape[0] != target_labels.shape[0]:
        raise ValueError("target label count does not match target vertex count")

    source_unit = unit_vertices(source_vertices)
    target_unit = unit_vertices(target_vertices)

    refined_source = np.full(source_labels.shape[0], -1, dtype=np.int64)
    refined_target = np.full(target_labels.shape[0], -1, dtype=np.int64)
    parent_for_refined = []
    summaries = []
    next_label = 0

    parent_ids = sorted(set(source_labels.tolist()) | set(target_labels.tolist()))
    for parent_id in parent_ids:
        source_idx = np.nonzero(source_labels == parent_id)[0]
        target_idx = np.nonzero(target_labels == parent_id)[0]
        if source_idx.size == 0 or target_idx.size == 0:
            continue

        if args.method == "quantile":
            src_local, tgt_local, centers = split_parent_group_by_quantile(
                source_unit[source_idx],
                target_unit[target_idx],
                requested_parts=args.subparts_per_parent,
                min_count=args.min_count,
            )
        else:
            src_local, tgt_local, centers = split_parent_group(
                source_unit[source_idx],
                target_unit[target_idx],
                requested_parts=args.subparts_per_parent,
                min_count=args.min_count,
                random_state=args.random_state + int(parent_id),
            )
        n_parts = int(max(src_local.max(initial=0), tgt_local.max(initial=0)) + 1)

        for local_id in range(n_parts):
            global_id = next_label + local_id
            refined_source[source_idx[src_local == local_id]] = global_id
            refined_target[target_idx[tgt_local == local_id]] = global_id
            parent_for_refined.append(int(parent_id))
            summaries.append(
                {
                    "refined_bucket": int(global_id),
                    "parent_bucket": int(parent_id),
                    "local_bucket": int(local_id),
                    "source_count": int((src_local == local_id).sum()),
                    "target_count": int((tgt_local == local_id).sum()),
                    "center": [float(x) for x in centers[local_id]],
                }
            )
        next_label += n_parts

    if (refined_source < 0).any() or (refined_target < 0).any():
        raise RuntimeError("Some vertices were not assigned refined labels")

    output_dir = Path(args.output_dir)
    labels_dir = output_dir / "labels"
    features_dir = output_dir / "features"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.write_one_hot_features:
        features_dir.mkdir(parents=True, exist_ok=True)

    source_out = labels_dir / f"{args.prefix}_source_labels.npz"
    target_out = labels_dir / f"{args.prefix}_target_labels.npz"
    n_buckets = int(next_label)
    parent_for_refined_arr = np.asarray(parent_for_refined, dtype=np.int64)

    np.savez_compressed(
        source_out,
        labels=refined_source,
        parent_labels=source_labels,
        parent_for_refined=parent_for_refined_arr,
        n_buckets=np.asarray(n_buckets, dtype=np.int64),
    )
    np.savez_compressed(
        target_out,
        labels=refined_target,
        parent_labels=target_labels,
        parent_for_refined=parent_for_refined_arr,
        n_buckets=np.asarray(n_buckets, dtype=np.int64),
    )

    summary = {
        "source_mesh": str(args.source_mesh),
        "target_mesh": str(args.target_mesh),
        "source_labels": str(args.source_labels),
        "target_labels": str(args.target_labels),
        "subparts_per_parent": int(args.subparts_per_parent),
        "min_count": int(args.min_count),
        "method": args.method,
        "n_buckets": n_buckets,
        "buckets": summaries,
    }

    if args.write_one_hot_features:
        source_features_out = features_dir / f"{args.prefix}_source_onehot.npy"
        target_features_out = features_dir / f"{args.prefix}_target_onehot.npy"
        np.save(source_features_out, one_hot(refined_source, n_buckets))
        np.save(target_features_out, one_hot(refined_target, n_buckets))
        summary["source_features"] = str(source_features_out)
        summary["target_features"] = str(target_features_out)

    summary_path = output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {source_out}")
    print(f"Wrote {target_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", required=True)
    parser.add_argument("--target-mesh", required=True)
    parser.add_argument("--source-labels", required=True)
    parser.add_argument("--target-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--subparts-per-parent", type=int, default=4)
    parser.add_argument("--min-count", type=int, default=12)
    parser.add_argument("--method", choices=["quantile", "kmeans"], default="quantile")
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--write-one-hot-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    refine_pair(parse_args())


if __name__ == "__main__":
    main()
