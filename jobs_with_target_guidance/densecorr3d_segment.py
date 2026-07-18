"""Convert DenseCorr3D group annotations into MeshUp label files.

DenseCorr3D stores supervised dense correspondence groups as one ``groups.txt``
file per object. Each line contains the vertex ids for one semantic group, and
line ids are aligned across objects in the same category. This script converts
those groups into the same vertex-label NPZ format consumed by
``partfield_chamfer.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jobs_with_target_guidance.partfield_segment import (
    labels_to_face_labels,
    load_obj_mesh,
    write_face_colored_ply,
)


FillMode = Literal["nearest", "largest", "error"]
IndexBase = Literal["auto", "zero", "one"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DenseCorr3D groups.txt annotations to MeshUp label NPZ files.",
    )
    parser.add_argument("--densecorr3d-root", type=Path, required=True, help="Path to the unzipped DenseCorr3D root.")
    parser.add_argument("--category", type=str, default="animals", help="DenseCorr3D category to process.")
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Object ids, substrings, or direct object directories. Defaults to all objects in the category.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("jobs_with_target_guidance/densematcher_runs/densecorr3d/animals"),
        help="Output directory. Labels are written under output-dir/labels/.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of objects to process.")
    parser.add_argument(
        "--index-base",
        choices=("auto", "zero", "one"),
        default="auto",
        help="Whether groups.txt vertex ids are zero- or one-indexed.",
    )
    parser.add_argument(
        "--fill-unassigned",
        choices=("nearest", "largest", "error"),
        default="nearest",
        help="How to label vertices not present in groups.txt.",
    )
    parser.add_argument("--list", action="store_true", help="Only list matching object directories.")
    return parser.parse_args()


def discover_objects(root: Path, category: str) -> List[Path]:
    category_dir = root / category
    if not category_dir.is_dir():
        raise FileNotFoundError(f"DenseCorr3D category directory not found: {category_dir}")
    objects = [
        path
        for path in sorted(category_dir.iterdir())
        if (path / "simple_mesh.obj").is_file() and (path / "groups.txt").is_file()
    ]
    if not objects:
        raise FileNotFoundError(f"No DenseCorr3D object directories found under {category_dir}")
    return objects


def resolve_object(root: Path, category: str, token: str, all_objects: Sequence[Path]) -> Path:
    path = Path(token)
    if path.is_dir():
        return path

    candidate = root / category / token
    if candidate.is_dir():
        return candidate

    matches = [obj for obj in all_objects if token in obj.name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No DenseCorr3D {category!r} object matches {token!r}.")
    names = ", ".join(obj.name for obj in matches[:10])
    raise ValueError(f"Object token {token!r} is ambiguous. Matches include: {names}")


def select_objects(root: Path, category: str, tokens: Sequence[str] | None, limit: int) -> List[Path]:
    all_objects = discover_objects(root, category)
    if tokens:
        objects = [resolve_object(root, category, token, all_objects) for token in tokens]
    else:
        objects = all_objects
    if limit > 0:
        objects = objects[:limit]
    return objects


def parse_group_indices(line: str) -> List[int]:
    body = line.split("#", 1)[0]
    return [int(match.group(0)) for match in re.finditer(r"-?\d+", body)]


def infer_index_base(groups: Sequence[Sequence[int]], n_vertices: int, requested: IndexBase) -> int:
    if requested == "zero":
        return 0
    if requested == "one":
        return 1

    values = [idx for group in groups for idx in group]
    if not values:
        raise ValueError("groups.txt did not contain any vertex indices.")

    min_idx = min(values)
    max_idx = max(values)
    if min_idx >= 1 and max_idx == n_vertices:
        return 1
    if min_idx >= 0 and max_idx < n_vertices:
        return 0
    if min_idx >= 1 and max_idx <= n_vertices:
        return 1
    raise ValueError(
        f"Cannot infer groups.txt index base: min={min_idx}, max={max_idx}, n_vertices={n_vertices}."
    )


def fill_unassigned_vertices(vertices: np.ndarray, labels: np.ndarray, fill_mode: FillMode) -> Tuple[np.ndarray, int]:
    missing = np.flatnonzero(labels < 0)
    if missing.size == 0:
        return labels, 0
    if fill_mode == "error":
        raise ValueError(f"{missing.size} vertices are not covered by groups.txt.")

    assigned = np.flatnonzero(labels >= 0)
    if assigned.size == 0:
        raise ValueError("groups.txt did not assign any vertices.")

    labels = labels.copy()
    if fill_mode == "largest":
        unique, counts = np.unique(labels[assigned], return_counts=True)
        labels[missing] = int(unique[np.argmax(counts)])
        return labels, int(missing.size)

    assigned_vertices = vertices[assigned].astype(np.float32)
    for start in range(0, missing.size, 2048):
        chunk = missing[start : start + 2048]
        delta = vertices[chunk, None, :].astype(np.float32) - assigned_vertices[None, :, :]
        nearest = np.argmin(np.einsum("ijk,ijk->ij", delta, delta), axis=1)
        labels[chunk] = labels[assigned[nearest]]
    return labels, int(missing.size)


def load_densecorr3d_groups(
    groups_path: Path,
    n_vertices: int,
    vertices: np.ndarray,
    index_base: IndexBase,
    fill_mode: FillMode,
) -> Tuple[np.ndarray, Dict[str, object]]:
    raw_groups = []
    for line in groups_path.read_text().splitlines():
        indices = parse_group_indices(line)
        if indices:
            raw_groups.append(indices)
    base = infer_index_base(raw_groups, n_vertices=n_vertices, requested=index_base)
    labels = np.full(n_vertices, -1, dtype=np.int64)
    duplicate_assignments = 0

    for group_id, raw_indices in enumerate(raw_groups):
        indices = np.asarray(raw_indices, dtype=np.int64) - base
        if indices.size == 0:
            continue
        invalid = indices[(indices < 0) | (indices >= n_vertices)]
        if invalid.size:
            raise ValueError(
                f"{groups_path} group {group_id} has {invalid.size} out-of-range vertex ids "
                f"after {base}-based conversion."
            )
        already = labels[indices] >= 0
        duplicate_assignments += int(already.sum())
        labels[indices] = group_id

    labels, filled_count = fill_unassigned_vertices(vertices, labels, fill_mode)
    summary = {
        "groups_path": str(groups_path),
        "index_base": base,
        "n_groups": len(raw_groups),
        "n_vertices": int(n_vertices),
        "duplicate_assignments": int(duplicate_assignments),
        "unassigned_vertices_filled": int(filled_count),
        "fill_unassigned": fill_mode,
    }
    return labels, summary


def label_stats(labels: np.ndarray) -> Dict[str, Dict[str, int]]:
    unique, counts = np.unique(labels, return_counts=True)
    return {
        f"group_{int(label):02d}": {"vertices": int(count)}
        for label, count in zip(unique, counts)
    }


def convert_object(
    object_dir: Path,
    output_dir: Path,
    index_base: IndexBase,
    fill_mode: FillMode,
) -> Dict[str, object]:
    mesh_path = object_dir / "simple_mesh.obj"
    groups_path = object_dir / "groups.txt"
    vertices, faces = load_obj_mesh(mesh_path)
    vertex_labels, group_summary = load_densecorr3d_groups(
        groups_path,
        n_vertices=vertices.shape[0],
        vertices=vertices,
        index_base=index_base,
        fill_mode=fill_mode,
    )
    face_labels = labels_to_face_labels(vertex_labels, faces)

    labels_dir = output_dir / "labels"
    colored_dir = output_dir / "colored"
    labels_dir.mkdir(parents=True, exist_ok=True)
    colored_dir.mkdir(parents=True, exist_ok=True)

    name = object_dir.name
    label_npz = labels_dir / f"{name}_densecorr3d_labels.npz"
    vertex_label_path = labels_dir / f"{name}_vertex_labels.npy"
    face_label_path = labels_dir / f"{name}_face_labels.npy"
    colored_mesh = colored_dir / f"{name}_densecorr3d_groups.ply"

    np.save(vertex_label_path, vertex_labels)
    np.save(face_label_path, face_labels)
    np.savez(
        label_npz,
        labels=vertex_labels,
        vertex_labels=vertex_labels,
        face_labels=face_labels,
        n_buckets=np.asarray(group_summary["n_groups"], dtype=np.int64),
    )
    write_face_colored_ply(colored_mesh, vertices, faces, face_labels)

    return {
        "name": name,
        "object_dir": str(object_dir),
        "mesh": str(mesh_path),
        "groups": str(groups_path),
        "label_npz": str(label_npz),
        "vertex_labels": str(vertex_label_path),
        "face_labels": str(face_label_path),
        "colored_mesh": str(colored_mesh),
        "bucket_stats": label_stats(vertex_labels),
        **group_summary,
    }


def write_summary(output_dir: Path, category: str, entries: Iterable[Dict[str, object]]) -> None:
    entries = list(entries)
    summary = {
        "source": "DenseCorr3D",
        "category": category,
        "objects": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    objects = select_objects(args.densecorr3d_root, args.category, args.objects, args.limit)
    if args.list:
        for object_dir in objects:
            print(object_dir)
        return

    entries = [
        convert_object(
            object_dir=object_dir,
            output_dir=args.output_dir,
            index_base=args.index_base,
            fill_mode=args.fill_unassigned,
        )
        for object_dir in objects
    ]
    write_summary(args.output_dir, args.category, entries)
    print(f"Wrote DenseCorr3D labels for {len(entries)} objects to {args.output_dir}")


if __name__ == "__main__":
    main()
