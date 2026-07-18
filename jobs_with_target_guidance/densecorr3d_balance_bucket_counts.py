"""Create DenseCorr3D mesh variants with matched semantic bucket sizes.

The normal DenseCorr3D preparation step writes ``full`` and ``5k`` variants
without changing the relative vertex density of each semantic group.  For
cross-animal deformation that can leave, for example, a giraffe neck bucket
with far fewer vertices than the matching elephant bucket.  This helper creates
an opt-in pair-specific variant where both meshes have the same vertex count in
each aligned DenseCorr3D bucket.

The remeshing is deliberately conservative:

* overfull buckets are reduced by collapsing the shortest same-bucket edges;
* underfull buckets are expanded by splitting the longest same-bucket edges;
* no existing prepared meshes are modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jobs_with_target_guidance.densecorr3d_prepare_mesh_variants import write_obj  # noqa: E402
from jobs_with_target_guidance.partfield_segment import (  # noqa: E402
    labels_to_face_labels,
    load_obj_mesh,
    write_face_colored_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--target-object", required=True)
    parser.add_argument("--base-variant", default="5k")
    parser.add_argument(
        "--output-variant",
        default="",
        help="Variant suffix to write. Defaults to bucket<mode><total>.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("average", "max", "source", "target"),
        default="average",
        help=(
            "How to choose the shared per-bucket counts. 'average' uses the "
            "rounded source/target mean; 'max' avoids removing vertices when "
            "the pair still fits under --max-vertices."
        ),
    )
    parser.add_argument("--max-vertices", type=int, default=6000)
    parser.add_argument("--min-bucket-vertices", type=int, default=12)
    parser.add_argument("--smooth-steps", type=int, default=1)
    parser.add_argument("--smooth-alpha", type=float, default=0.08)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_vertex_labels(path: Path) -> Tuple[np.ndarray, int]:
    data = np.load(path)
    for key in ("vertex_labels", "labels", "part_labels"):
        if key in data:
            labels = np.asarray(data[key], dtype=np.int64).reshape(-1)
            break
    else:
        labels = np.asarray(data[data.files[0]], dtype=np.int64).reshape(-1)

    if "n_buckets" in data:
        n_buckets = int(np.asarray(data["n_buckets"]).reshape(()))
    else:
        n_buckets = int(labels.max(initial=-1) + 1)
    return labels, n_buckets


def bucket_counts(labels: np.ndarray, n_buckets: int) -> np.ndarray:
    return np.bincount(labels, minlength=n_buckets).astype(np.int64)


def counts_dict(counts: Sequence[int]) -> Dict[str, int]:
    return {f"group_{idx:02d}": int(count) for idx, count in enumerate(counts)}


def round_counts(values: np.ndarray) -> np.ndarray:
    floors = np.floor(values).astype(np.int64)
    target_total = int(round(float(values.sum())))
    remainder = target_total - int(floors.sum())
    if remainder > 0:
        order = np.argsort(-(values - floors), kind="mergesort")
        floors[order[:remainder]] += 1
    return floors


def fit_counts_to_budget(counts: np.ndarray, max_vertices: int, min_bucket_vertices: int) -> np.ndarray:
    counts = counts.astype(np.int64)
    if int(counts.sum()) <= max_vertices:
        return counts
    if max_vertices < min_bucket_vertices * counts.shape[0]:
        raise ValueError(
            f"max_vertices={max_vertices} cannot satisfy {counts.shape[0]} buckets "
            f"with min_bucket_vertices={min_bucket_vertices}."
        )

    scaled = counts.astype(np.float64) * (float(max_vertices) / float(counts.sum()))
    floors = np.maximum(np.floor(scaled).astype(np.int64), int(min_bucket_vertices))
    overflow = int(floors.sum()) - int(max_vertices)
    if overflow > 0:
        order = np.argsort(floors - min_bucket_vertices, kind="mergesort")[::-1]
        for idx in order:
            take = min(overflow, int(floors[idx] - min_bucket_vertices))
            floors[idx] -= take
            overflow -= take
            if overflow == 0:
                break
    under = int(max_vertices) - int(floors.sum())
    if under > 0:
        frac = scaled - np.floor(scaled)
        order = np.argsort(-frac, kind="mergesort")
        for idx in order[:under]:
            floors[idx] += 1
    return floors


def choose_target_counts(
    source_counts: np.ndarray,
    target_counts: np.ndarray,
    mode: str,
    max_vertices: int,
    min_bucket_vertices: int,
) -> np.ndarray:
    if mode == "average":
        counts = round_counts((source_counts.astype(np.float64) + target_counts.astype(np.float64)) * 0.5)
    elif mode == "max":
        counts = np.maximum(source_counts, target_counts)
    elif mode == "source":
        counts = source_counts.copy()
    elif mode == "target":
        counts = target_counts.copy()
    else:
        raise ValueError(f"Unknown target count mode: {mode}")

    counts = np.maximum(counts.astype(np.int64), int(min_bucket_vertices))
    return fit_counts_to_budget(counts, max_vertices=max_vertices, min_bucket_vertices=min_bucket_vertices)


def variant_path(prepared_dir: Path, kind: str, object_id: str, variant: str, suffix: str) -> Path:
    return prepared_dir / kind / f"{object_id}_densecorr3d_{variant}{suffix}"


def unique_faces(faces: np.ndarray) -> np.ndarray:
    seen = set()
    kept: List[np.ndarray] = []
    for face in faces:
        key = tuple(sorted(int(v) for v in face))
        if len(set(key)) != 3 or key in seen:
            continue
        seen.add(key)
        kept.append(face)
    if not kept:
        raise RuntimeError("All faces vanished during bucket balancing.")
    return np.asarray(kept, dtype=np.int64)


def edge_faces(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    edges: Dict[Tuple[int, int], List[int]] = {}
    for face_id, face in enumerate(faces):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edges.setdefault(key, []).append(face_id)
    return edges


def edge_is_interior_to_bucket(
    faces: np.ndarray,
    labels: np.ndarray,
    face_ids: Iterable[int],
    bucket_id: int,
) -> bool:
    return all(np.all(labels[faces[face_id]] == bucket_id) for face_id in face_ids)


def vertex_face_degree(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    return np.bincount(faces.reshape(-1), minlength=n_vertices).astype(np.int64)


def collapse_is_safe(
    faces: np.ndarray,
    edge: Tuple[int, int],
    adjacent_faces: Sequence[int],
    degree: np.ndarray,
) -> bool:
    u, v = edge
    removed_face_set = set(int(face_id) for face_id in adjacent_faces)
    if not removed_face_set:
        return False

    third_vertices: List[int] = []
    for face_id in adjacent_faces:
        face = faces[int(face_id)]
        others = [int(x) for x in face if int(x) != u and int(x) != v]
        third_vertices.extend(others)

    for vertex_id in set(third_vertices):
        incident_removed = sum(1 for face_id in adjacent_faces if vertex_id in faces[int(face_id)])
        if degree[vertex_id] <= incident_removed:
            return False

    if degree[u] + degree[v] <= 2 * len(adjacent_faces):
        return False
    return True


MIN_TRIANGLE_AREA = 1e-9  # matches remesh_to_counts final validation


def _collapse_result_min_area(
    vertices: np.ndarray,
    faces: np.ndarray,
    edge: Tuple[int, int],
    adjacent_faces: Sequence[int],
) -> float:
    """Min area among faces surviving a collapse (excluding the removed ones)."""
    keep, remove = min(edge), max(edge)
    new_pos = (vertices[edge[0]] + vertices[edge[1]]) * 0.5
    removed_set = set(int(f) for f in adjacent_faces)
    incident_mask = np.any(faces == keep, axis=1) | np.any(faces == remove, axis=1)
    min_area = float("inf")
    for face_id in np.where(incident_mask)[0]:
        if int(face_id) in removed_set:
            continue
        face = faces[face_id]
        tri = np.array(
            [new_pos if int(vid) in (keep, remove) else vertices[int(vid)] for vid in face]
        )
        area = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
        min_area = min(min_area, area)
    return min_area


def select_collapse_edge(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    bucket_id: int,
) -> Tuple[Tuple[int, int], List[int]]:
    edges = edge_faces(faces)
    degree = vertex_face_degree(faces, vertices.shape[0])

    candidates: List[Tuple[bool, float, Tuple[int, int], List[int]]] = []
    for edge, face_ids in edges.items():
        u, v = edge
        if labels[u] != bucket_id or labels[v] != bucket_id:
            continue
        if not collapse_is_safe(faces, edge, face_ids, degree):
            continue
        length = float(np.linalg.norm(vertices[u] - vertices[v]))
        interior = edge_is_interior_to_bucket(faces, labels, face_ids, bucket_id)
        candidates.append((interior, length, edge, face_ids))

    # Prefer interior edges, then the shortest edge among each group.
    candidates.sort(key=lambda c: (not c[0], c[1]))

    for interior, length, edge, face_ids in candidates:
        # Moving the kept vertex to the collapsed midpoint can turn a
        # surviving neighboring face into a near-degenerate sliver even
        # when collapse_is_safe's topology checks pass. Reject those.
        if _collapse_result_min_area(vertices, faces, edge, face_ids) < MIN_TRIANGLE_AREA:
            continue
        return edge, face_ids

    if not candidates:
        raise RuntimeError(f"No safe same-bucket collapse edge found for bucket {bucket_id}.")
    raise RuntimeError(
        f"No non-degenerate same-bucket collapse edge found for bucket {bucket_id}: "
        "every safe candidate would create a sliver face."
    )


def collapse_edge(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    edge: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = min(edge)
    remove = max(edge)
    vertices = vertices.copy()
    vertices[keep] = (vertices[edge[0]] + vertices[edge[1]]) * 0.5

    remapped_faces = faces.copy()
    remapped_faces[remapped_faces == remove] = keep
    nondegenerate = np.array([len(set(map(int, face))) == 3 for face in remapped_faces], dtype=bool)
    remapped_faces = unique_faces(remapped_faces[nondegenerate])

    vertices = np.delete(vertices, remove, axis=0)
    labels = np.delete(labels, remove, axis=0)
    remapped_faces[remapped_faces > remove] -= 1
    return vertices, remapped_faces.astype(np.int64), labels


def _triangle_area(vertices: np.ndarray, face: Sequence[int]) -> float:
    a, b, c = vertices[int(face[0])], vertices[int(face[1])], vertices[int(face[2])]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a)))


def select_split_edge(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    bucket_id: int,
) -> Tuple[Tuple[int, int], List[int]]:
    edges = edge_faces(faces)

    # Repeated bisection can make an edge's midpoint land exactly on a
    # vertex created by an earlier split in the same bucket (observed on
    # DenseCorr3D bear/cheetah 5k meshes). Splitting such an edge creates a
    # zero-area face and a coincident-vertex pair, which later makes the
    # Poisson solve for jacobian-based deformation singular/NaN. Track
    # existing positions so candidates whose midpoint would collide are
    # skipped in favor of the next-best edge.
    # write_obj serializes vertices with only 8 decimal places, so two points
    # closer than that will become byte-identical on disk even if they still
    # differ in float64 memory; round with a safety margin below that.
    existing_positions = {tuple(np.round(p, 7)) for p in vertices}

    candidates: List[Tuple[bool, float, Tuple[int, int], List[int]]] = []
    for edge, face_ids in edges.items():
        u, v = edge
        if labels[u] != bucket_id or labels[v] != bucket_id:
            continue
        length = float(np.linalg.norm(vertices[u] - vertices[v]))
        interior = edge_is_interior_to_bucket(faces, labels, face_ids, bucket_id)
        candidates.append((interior, length, edge, face_ids))

    # Prefer interior edges, then the longest edge among each group.
    candidates.sort(key=lambda c: (not c[0], -c[1]))

    for interior, length, edge, face_ids in candidates:
        new_position = (vertices[edge[0]] + vertices[edge[1]]) * 0.5
        midpoint = tuple(np.round(new_position, 7))
        if midpoint in existing_positions:
            continue

        # Bisecting a thin/obtuse triangle can still create a near-degenerate
        # sub-triangle even when the new vertex doesn't collide with an
        # existing one. Simulate the split and reject candidates whose
        # resulting faces would be slivers.
        probe_vertices = np.vstack([vertices, new_position])
        new_vertex_probe = vertices.shape[0]
        min_new_area = min(
            _triangle_area(probe_vertices, sub_face)
            for face_id in face_ids
            for sub_face in split_face_on_edge(faces[int(face_id)], edge, new_vertex_probe)
        )
        if min_new_area < MIN_TRIANGLE_AREA:
            continue

        return edge, face_ids

    if not candidates:
        raise RuntimeError(f"No same-bucket split edge found for bucket {bucket_id}.")
    raise RuntimeError(
        f"No non-degenerate same-bucket split edge found for bucket {bucket_id}: "
        "every candidate would collide with an existing vertex or create a sliver face."
    )


def split_face_on_edge(face: np.ndarray, edge: Tuple[int, int], new_vertex: int) -> List[List[int]]:
    a, b = edge
    face_list = [int(x) for x in face]
    for first, second in ((a, b), (b, a)):
        for idx in range(3):
            if face_list[idx] == first and face_list[(idx + 1) % 3] == second:
                third = face_list[(idx + 2) % 3]
                return [[first, new_vertex, third], [new_vertex, second, third]]
    raise RuntimeError(f"Face {face_list} does not contain edge {edge}.")


def split_edge(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    edge: Tuple[int, int],
    adjacent_faces: Sequence[int],
    bucket_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_vertex = int(vertices.shape[0])
    new_position = (vertices[edge[0]] + vertices[edge[1]]) * 0.5
    vertices = np.vstack([vertices, new_position.astype(vertices.dtype)])
    labels = np.concatenate([labels, np.asarray([bucket_id], dtype=np.int64)])

    adjacent = set(int(face_id) for face_id in adjacent_faces)
    new_faces: List[List[int]] = []
    for face_id, face in enumerate(faces):
        if face_id in adjacent:
            new_faces.extend(split_face_on_edge(face, edge, new_vertex))
        else:
            new_faces.append([int(face[0]), int(face[1]), int(face[2])])
    return vertices, np.asarray(new_faces, dtype=np.int64), labels


def smooth_by_label(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    steps: int,
    alpha: float,
) -> np.ndarray:
    if steps <= 0 or alpha <= 0:
        return vertices
    alpha = float(np.clip(alpha, 0.0, 1.0))

    neighbors: List[set[int]] = [set() for _ in range(vertices.shape[0])]
    for face in faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    def face_areas(v: np.ndarray) -> np.ndarray:
        tri = v[faces]
        return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    smoothed = vertices.copy()
    for _ in range(steps):
        updated = smoothed.copy()
        for vertex_id, vertex_neighbors in enumerate(neighbors):
            same_label = [idx for idx in vertex_neighbors if labels[idx] == labels[vertex_id]]
            if len(same_label) < 2:
                continue
            target = smoothed[np.asarray(same_label, dtype=np.int64)].mean(axis=0)
            updated[vertex_id] = (1.0 - alpha) * smoothed[vertex_id] + alpha * target

        # A synchronous (Jacobi-style) update can create a degenerate face
        # through the *joint* movement of two or more of its vertices even
        # when each vertex's own displacement looks harmless checked in
        # isolation against its old neighbors. `smoothed` (this step's
        # starting point) is already known non-degenerate, so iteratively
        # rolling back any vertex touching a still-degenerate face to its
        # pre-step position is always safe and converges in a few passes.
        for _ in range(vertices.shape[0]):
            bad_faces = np.where(face_areas(updated) < MIN_TRIANGLE_AREA)[0]
            if bad_faces.size == 0:
                break
            bad_vertices = np.unique(faces[bad_faces].reshape(-1))
            updated[bad_vertices] = smoothed[bad_vertices]
        smoothed = updated
    return smoothed.astype(vertices.dtype)


def remesh_to_counts(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    target_counts: np.ndarray,
    smooth_steps: int,
    smooth_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    n_buckets = int(target_counts.shape[0])
    operations: List[Dict[str, int]] = []
    before_counts = bucket_counts(labels, n_buckets)

    for bucket_id in range(n_buckets):
        while int((labels == bucket_id).sum()) > int(target_counts[bucket_id]):
            edge, adjacent_faces = select_collapse_edge(vertices, faces, labels, bucket_id)
            vertices, faces, labels = collapse_edge(vertices, faces, labels, edge)
            operations.append({"bucket": bucket_id, "op": "collapse", "remaining": int((labels == bucket_id).sum())})

    for bucket_id in range(n_buckets):
        while int((labels == bucket_id).sum()) < int(target_counts[bucket_id]):
            edge, adjacent_faces = select_split_edge(vertices, faces, labels, bucket_id)
            vertices, faces, labels = split_edge(vertices, faces, labels, edge, adjacent_faces, bucket_id)
            operations.append({"bucket": bucket_id, "op": "split", "remaining": int((labels == bucket_id).sum())})

    vertices = smooth_by_label(vertices, faces, labels, steps=smooth_steps, alpha=smooth_alpha)
    after_counts = bucket_counts(labels, n_buckets)
    if not np.array_equal(after_counts, target_counts):
        raise RuntimeError(
            "Bucket balancing did not hit requested counts: "
            f"target={target_counts.tolist()}, actual={after_counts.tolist()}"
        )
    if faces.min(initial=0) < 0 or faces.max(initial=-1) >= vertices.shape[0]:
        raise RuntimeError("Balanced mesh has out-of-range face indices.")

    # Match write_obj's 8-decimal-place serialization precision (with margin)
    # so this check reflects what the saved mesh will actually look like.
    n_unique_positions = np.unique(np.round(vertices, 7), axis=0).shape[0]
    if n_unique_positions != vertices.shape[0]:
        raise RuntimeError(
            "Balanced mesh has "
            f"{vertices.shape[0] - n_unique_positions} coincident-vertex pair(s) "
            "after collapse/split/smoothing; this produces zero-area faces and a "
            "singular Poisson system downstream."
        )
    tri = vertices[faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    n_degenerate_faces = int(np.sum(face_areas < 1e-9))
    if n_degenerate_faces:
        raise RuntimeError(f"Balanced mesh has {n_degenerate_faces} zero-area/sliver face(s).")

    summary = {
        "vertices_before": int(before_counts.sum()),
        "vertices_after": int(vertices.shape[0]),
        "faces_after": int(faces.shape[0]),
        "bucket_counts_before": counts_dict(before_counts),
        "bucket_counts_after": counts_dict(after_counts),
        "operations_by_type": {
            "collapse": int(sum(1 for op in operations if op["op"] == "collapse")),
            "split": int(sum(1 for op in operations if op["op"] == "split")),
        },
        "operations_by_bucket": {
            f"group_{bucket_id:02d}": {
                "collapse": int(sum(1 for op in operations if op["bucket"] == bucket_id and op["op"] == "collapse")),
                "split": int(sum(1 for op in operations if op["bucket"] == bucket_id and op["op"] == "split")),
            }
            for bucket_id in range(n_buckets)
        },
    }
    return vertices, faces, labels, summary


def save_balanced_variant(
    prepared_dir: Path,
    object_id: str,
    output_variant: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    n_buckets: int,
    overwrite: bool,
) -> Dict[str, object]:
    mesh_path = variant_path(prepared_dir, "meshes", object_id, output_variant, ".obj")
    labels_path = variant_path(prepared_dir, "labels", object_id, output_variant, "_labels.npz")
    colored_path = variant_path(prepared_dir, "colored", object_id, output_variant, "_groups.ply")
    for path in (mesh_path, labels_path, colored_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")

    face_labels = labels_to_face_labels(labels, faces)
    write_obj(mesh_path, vertices, faces)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        labels_path,
        labels=labels.astype(np.int64),
        vertex_labels=labels.astype(np.int64),
        face_labels=face_labels.astype(np.int64),
        n_buckets=np.asarray(n_buckets, dtype=np.int64),
    )
    write_face_colored_ply(colored_path, vertices, faces, face_labels)

    return {
        "object": object_id,
        "mesh": str(mesh_path),
        "labels": str(labels_path),
        "colored_mesh": str(colored_path),
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "n_buckets": int(n_buckets),
        "bucket_vertex_counts": counts_dict(bucket_counts(labels, n_buckets)),
    }


def validate_variant_name(variant: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", variant):
        raise ValueError(f"Unsafe variant name {variant!r}; use letters, digits, '_', '.', or '-'.")


def load_prepared_object(prepared_dir: Path, object_id: str, variant: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mesh_path = variant_path(prepared_dir, "meshes", object_id, variant, ".obj")
    labels_path = variant_path(prepared_dir, "labels", object_id, variant, "_labels.npz")
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    vertices, faces = load_obj_mesh(mesh_path)
    labels, n_buckets = load_vertex_labels(labels_path)
    if vertices.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{object_id} {variant}: mesh has {vertices.shape[0]} vertices but labels have {labels.shape[0]} rows."
        )
    return vertices, faces, labels, n_buckets


def main() -> None:
    args = parse_args()
    validate_variant_name(args.base_variant)

    source_vertices, source_faces, source_labels, source_buckets = load_prepared_object(
        args.prepared_dir,
        args.source_object,
        args.base_variant,
    )
    target_vertices, target_faces, target_labels, target_buckets = load_prepared_object(
        args.prepared_dir,
        args.target_object,
        args.base_variant,
    )
    if source_buckets != target_buckets:
        raise ValueError(f"Bucket count mismatch: source={source_buckets}, target={target_buckets}.")
    n_buckets = source_buckets

    source_counts = bucket_counts(source_labels, n_buckets)
    target_counts = bucket_counts(target_labels, n_buckets)
    shared_counts = choose_target_counts(
        source_counts,
        target_counts,
        mode=args.target_mode,
        max_vertices=args.max_vertices,
        min_bucket_vertices=args.min_bucket_vertices,
    )
    if args.output_variant:
        output_variant = args.output_variant
    else:
        output_variant = f"bucket{args.target_mode}{int(shared_counts.sum())}"
    validate_variant_name(output_variant)

    balanced_source = remesh_to_counts(
        source_vertices,
        source_faces,
        source_labels,
        shared_counts,
        smooth_steps=args.smooth_steps,
        smooth_alpha=args.smooth_alpha,
    )
    balanced_target = remesh_to_counts(
        target_vertices,
        target_faces,
        target_labels,
        shared_counts,
        smooth_steps=args.smooth_steps,
        smooth_alpha=args.smooth_alpha,
    )

    source_entry = save_balanced_variant(
        args.prepared_dir,
        args.source_object,
        output_variant,
        balanced_source[0],
        balanced_source[1],
        balanced_source[2],
        n_buckets,
        overwrite=args.overwrite,
    )
    target_entry = save_balanced_variant(
        args.prepared_dir,
        args.target_object,
        output_variant,
        balanced_target[0],
        balanced_target[1],
        balanced_target[2],
        n_buckets,
        overwrite=args.overwrite,
    )

    summary = {
        "prepared_dir": str(args.prepared_dir),
        "source_object": args.source_object,
        "target_object": args.target_object,
        "base_variant": args.base_variant,
        "output_variant": output_variant,
        "target_mode": args.target_mode,
        "max_vertices": int(args.max_vertices),
        "smooth_steps": int(args.smooth_steps),
        "smooth_alpha": float(args.smooth_alpha),
        "n_buckets": int(n_buckets),
        "source_base_bucket_counts": counts_dict(source_counts),
        "target_base_bucket_counts": counts_dict(target_counts),
        "shared_bucket_counts": counts_dict(shared_counts),
        "source": {**source_entry, "balancing": balanced_source[3]},
        "target": {**target_entry, "balancing": balanced_target[3]},
    }
    summary_path = args.prepared_dir / f"{args.source_object}_to_{args.target_object}_{output_variant}_bucket_balance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
