"""Grow undersized DenseCorr3D label buckets by split-only local subdivision.

`densecorr3d_prepare_mesh_variants.py` decimates each animal's full mesh down
to ~5k vertices and re-derives labels afterward by nearest-neighbor transfer.
Decimation has no notion of label boundaries, so small/thin regions (an
elephant trunk, an animal's ears) can end up with very few vertices relative
to the same semantic bucket on a differently-shaped animal. When a hard-bucket
Chamfer loss later has to match a sparse source patch against a much larger
target patch (or vice versa), the few source points balloon outward to reach
full coverage instead of preserving the region's shape.

This script raises any bucket below `--min-bucket-vertices` up to that floor
by repeatedly splitting the longest mesh edge that is strictly interior to
that bucket (both adjacent triangles fully inside the bucket) -- i.e. a
Loop-style midpoint split, never an edge collapse, and never touching a
bucket-boundary edge. Splitting a positive-length edge of a non-degenerate
triangle can only ever produce two more non-degenerate triangles at a new,
distinct point, so this operation cannot introduce zero-area faces or
duplicate-coordinate vertices the way a collapse/re-smooth pipeline can.
The script also re-validates the output mesh geometry before writing
anything and aborts loudly if it finds duplicate vertices, degenerate faces,
non-manifold edges, or a vertex count above `--max-vertices`.

Runs on one animal's mesh+label pair at a time (not a source/target pair),
so it only needs to run once per animal and is reused across every pairing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jobs_with_target_guidance.partfield_segment import (  # noqa: E402
    labels_to_face_labels,
    load_obj_mesh,
    write_face_colored_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-mesh", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-colored", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--min-bucket-vertices", type=int, default=300)
    parser.add_argument("--max-vertices", type=int, default=7000)
    return parser.parse_args()


def load_labels(path: Path) -> np.ndarray:
    data = np.load(path)
    for key in ("labels", "vertex_labels", "part_labels"):
        if key in data:
            return np.asarray(data[key], dtype=np.int64).reshape(-1)
    return np.asarray(data[data.files[0]], dtype=np.int64).reshape(-1)


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def build_edge_to_faces(faces: List[Optional[List[int]]]) -> Dict[Tuple[int, int], List[int]]:
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fi, f in enumerate(faces):
        if f is None:
            continue
        a, b, c = f
        for u, v in ((a, b), (b, c), (c, a)):
            edge_to_faces[edge_key(u, v)].append(fi)
    return edge_to_faces


def protect_small_buckets(
    vertices_in: np.ndarray,
    faces_in: np.ndarray,
    labels_in: np.ndarray,
    min_bucket_vertices: int,
    max_total_vertices: int,
    max_passes: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Split-only densification, one label at a time, in fixed-snapshot passes.

    Each pass takes a snapshot of the currently-eligible interior edges
    (length-sorted, longest first) and splits from that fixed list only --
    it never re-splits an edge touching a vertex created earlier in the same
    pass. Without that restriction, a single high-valence "hub" vertex (a
    decimation artifact at a thin appendage tip, where many triangles fan
    around one point) can have its shrinking spoke edges repeatedly
    re-selected as "currently longest," geometrically halving toward the hub
    each time until multiple independent spokes numerically converge on the
    same point -- producing duplicate vertices and zero-area faces. Snapshot
    passes make that impossible: a freshly created midpoint can only compete
    for further splitting starting next pass, on equal footing with every
    other current edge, so short new spokes are naturally deprioritized
    behind genuinely long remaining edges.
    """
    vertices: List[np.ndarray] = [np.asarray(v, dtype=np.float64) for v in vertices_in]
    faces: List[Optional[List[int]]] = [list(map(int, f)) for f in faces_in]
    labels: List[int] = [int(x) for x in labels_in]

    edge_to_faces = build_edge_to_faces(faces)
    counts = Counter(labels)
    n_buckets = int(max(labels)) + 1

    def face_uniform_label(fi: int) -> Optional[int]:
        f = faces[fi]
        if f is None:
            return None
        la, lb, lc = labels[f[0]], labels[f[1]], labels[f[2]]
        if la == lb == lc:
            return la
        return None

    def live_faces_of(ek: Tuple[int, int]) -> List[int]:
        return [fi for fi in edge_to_faces.get(ek, []) if faces[fi] is not None]

    bucket_summary: Dict[str, object] = {}
    hit_vertex_cap = False

    for label_id in range(n_buckets):
        start_count = counts[label_id]
        if start_count >= min_bucket_vertices or start_count == 0:
            bucket_summary[str(label_id)] = {"start": start_count, "added": 0, "end": start_count, "passes": 0}
            continue

        added = 0
        passes_used = 0
        for _pass in range(max_passes):
            if counts[label_id] >= min_bucket_vertices or len(vertices) >= max_total_vertices:
                break
            passes_used += 1

            candidates: List[Tuple[float, Tuple[int, int]]] = []
            seen_edges = set()
            for ek, face_list in list(edge_to_faces.items()):
                if ek in seen_edges:
                    continue
                seen_edges.add(ek)
                live = [fi for fi in face_list if faces[fi] is not None]
                if live and all(face_uniform_label(fi) == label_id for fi in live):
                    u, v = ek
                    length = float(np.linalg.norm(vertices[u] - vertices[v]))
                    candidates.append((length, ek))
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0], reverse=True)

            for _length, ek in candidates:
                if counts[label_id] >= min_bucket_vertices:
                    break
                if len(vertices) >= max_total_vertices:
                    hit_vertex_cap = True
                    break

                live = live_faces_of(ek)
                if not live or not all(face_uniform_label(fi) == label_id for fi in live):
                    continue  # a face bordering this edge was already consumed earlier this pass

                u, v = ek
                m_idx = len(vertices)
                vertices.append((vertices[u] + vertices[v]) * 0.5)
                labels.append(label_id)
                counts[label_id] += 1
                added += 1

                for fi in live:
                    a, b, c = faces[fi]
                    for p0, p1, w in ((a, b, c), (b, c, a), (c, a, b)):
                        if edge_key(p0, p1) == ek:
                            break
                    else:
                        raise RuntimeError(f"Edge {ek} not found in face {fi}={faces[fi]}")

                    faces[fi] = None
                    nf1 = [p0, m_idx, w]
                    nf2 = [m_idx, p1, w]
                    fi1, fi2 = len(faces), len(faces) + 1
                    faces.append(nf1)
                    faces.append(nf2)
                    for nf_idx, nf in ((fi1, nf1), (fi2, nf2)):
                        for x, y in ((nf[0], nf[1]), (nf[1], nf[2]), (nf[2], nf[0])):
                            edge_to_faces[edge_key(x, y)].append(nf_idx)
                    # Deliberately not queued for this pass -- only a future
                    # pass's fresh snapshot may select edges touching m_idx.

        bucket_summary[str(label_id)] = {
            "start": start_count,
            "added": added,
            "end": counts[label_id],
            "passes": passes_used,
            "reached_floor": counts[label_id] >= min_bucket_vertices,
        }

    final_vertices = np.asarray(vertices, dtype=np.float64)
    final_faces = np.asarray([f for f in faces if f is not None], dtype=np.int64)
    final_labels = np.asarray(labels, dtype=np.int64)
    meta = {"buckets": bucket_summary, "hit_vertex_cap": hit_vertex_cap}
    return final_vertices, final_faces, final_labels, meta


def validate_mesh_geometry(vertices: np.ndarray, faces: np.ndarray) -> Dict[str, object]:
    _uniq, inverse, counts = np.unique(np.round(vertices, 8), axis=0, return_inverse=True, return_counts=True)
    duplicate_vertices = int((counts[inverse] > 1).sum())

    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    zero_area_faces = int((area2 < 1e-12).sum())

    edge_count: Counter = Counter()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge_count[edge_key(int(u), int(v))] += 1
    non_manifold_edges = sum(1 for n in edge_count.values() if n > 2)
    boundary_edges = sum(1 for n in edge_count.values() if n == 1)

    referenced = np.zeros(vertices.shape[0], dtype=bool)
    referenced[faces.reshape(-1)] = True
    unreferenced_vertices = int((~referenced).sum())

    return {
        "n_vertices": int(vertices.shape[0]),
        "n_faces": int(faces.shape[0]),
        "duplicate_vertices": duplicate_vertices,
        "zero_area_faces": zero_area_faces,
        "non_manifold_edges": non_manifold_edges,
        "boundary_edges": boundary_edges,
        "unreferenced_vertices": unreferenced_vertices,
    }


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# DenseCorr3D small-bucket-protected geometry\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
            f.write(f"f {a} {b} {c}\n")


def main() -> None:
    args = parse_args()

    vertices, faces = load_obj_mesh(args.mesh)
    labels = load_labels(args.labels)
    if labels.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"Label count {labels.shape[0]} does not match vertex count {vertices.shape[0]} for {args.mesh}"
        )

    input_issues = validate_mesh_geometry(vertices, faces)
    if input_issues["duplicate_vertices"] or input_issues["zero_area_faces"] or input_issues["non_manifold_edges"]:
        raise RuntimeError(f"Input mesh {args.mesh} already has degenerate geometry: {input_issues}")

    new_vertices, new_faces, new_labels, split_meta = protect_small_buckets(
        vertices,
        faces,
        labels,
        min_bucket_vertices=args.min_bucket_vertices,
        max_total_vertices=args.max_vertices,
    )

    output_issues = validate_mesh_geometry(new_vertices, new_faces)
    problems = []
    if output_issues["duplicate_vertices"] > 0:
        problems.append(f"{output_issues['duplicate_vertices']} duplicate-coordinate vertices")
    if output_issues["zero_area_faces"] > 0:
        problems.append(f"{output_issues['zero_area_faces']} zero-area faces")
    if output_issues["non_manifold_edges"] > 0:
        problems.append(f"{output_issues['non_manifold_edges']} non-manifold edges")
    if output_issues["unreferenced_vertices"] > 0:
        problems.append(f"{output_issues['unreferenced_vertices']} unreferenced vertices")
    if output_issues["n_vertices"] > args.max_vertices:
        problems.append(f"{output_issues['n_vertices']} vertices exceeds --max-vertices {args.max_vertices}")
    if problems:
        raise RuntimeError(
            f"Refusing to write {args.output_mesh}: output geometry is degenerate ({'; '.join(problems)}). "
            f"Full check: {output_issues}"
        )

    write_obj(args.output_mesh, new_vertices, new_faces)

    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    n_buckets = int(new_labels.max()) + 1
    face_labels = labels_to_face_labels(new_labels, new_faces)
    np.savez(
        args.output_labels,
        labels=new_labels,
        vertex_labels=new_labels,
        face_labels=face_labels,
        n_buckets=np.asarray(n_buckets, dtype=np.int64),
    )

    if args.output_colored:
        write_face_colored_ply(args.output_colored, new_vertices, new_faces, face_labels)

    summary = {
        "mesh": str(args.mesh),
        "labels": str(args.labels),
        "min_bucket_vertices": args.min_bucket_vertices,
        "max_vertices": args.max_vertices,
        "input_vertices": int(vertices.shape[0]),
        "output_vertices": int(new_vertices.shape[0]),
        "split_meta": split_meta,
        "output_geometry_check": output_issues,
    }
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_mesh} ({new_vertices.shape[0]} vertices, {new_faces.shape[0]} faces)")
    print(f"Wrote {args.output_labels}")


if __name__ == "__main__":
    main()
