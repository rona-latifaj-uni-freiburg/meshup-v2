#!/usr/bin/env python3
"""Merge selected DenseCorr3D label buckets without splitting any group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def load_labels(path: Path) -> np.ndarray:
    data = np.load(path)
    for key in ("labels", "vertex_labels", "source_labels", "target_labels"):
        if key in data:
            return np.asarray(data[key], dtype=np.int64)
    raise KeyError(f"No vertex labels found in {path}")


def parse_merge_specs(specs: Iterable[str]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    next_label = 0
    for spec in specs:
        group = [int(token) for token in spec.split(",") if token != ""]
        if not group:
            continue
        for label in group:
            if label in mapping:
                raise ValueError(f"Label {label} appears in more than one merge group")
            mapping[label] = next_label
        next_label += 1
    return mapping


def apply_mapping(labels: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    missing = sorted(set(labels.tolist()) - set(mapping))
    if missing:
        raise ValueError(f"Merge mapping is missing labels: {missing}")
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)


def face_labels_from_vertices(vertex_labels: np.ndarray, faces_path: Path) -> np.ndarray:
    faces: List[List[int]] = []
    with faces_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("f "):
                continue
            face = []
            for token in line.split()[1:]:
                face.append(int(token.split("/")[0]) - 1)
            faces.append(face)

    out = np.empty(len(faces), dtype=np.int64)
    for idx, face in enumerate(faces):
        values, counts = np.unique(vertex_labels[np.asarray(face, dtype=np.int64)], return_counts=True)
        out[idx] = int(values[counts.argmax()])
    return out


def write_labels(path: Path, labels: np.ndarray, mesh_path: Path, n_buckets: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        labels=labels,
        vertex_labels=labels,
        face_labels=face_labels_from_vertices(labels, mesh_path),
        n_buckets=np.asarray(n_buckets, dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-labels", required=True)
    parser.add_argument("--target-labels", required=True)
    parser.add_argument("--source-mesh", required=True)
    parser.add_argument("--target-mesh", required=True)
    parser.add_argument("--source-output", required=True)
    parser.add_argument("--target-output", required=True)
    parser.add_argument(
        "--merge",
        action="append",
        required=True,
        help="Comma-separated original labels to merge into the next output bucket. Repeat in output-bucket order.",
    )
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    mapping = parse_merge_specs(args.merge)
    n_buckets = max(mapping.values()) + 1

    source_labels = apply_mapping(load_labels(Path(args.source_labels)), mapping)
    target_labels = apply_mapping(load_labels(Path(args.target_labels)), mapping)
    write_labels(Path(args.source_output), source_labels, Path(args.source_mesh), n_buckets)
    write_labels(Path(args.target_output), target_labels, Path(args.target_mesh), n_buckets)

    summary = {
        "merge_mapping": {str(old): int(new) for old, new in sorted(mapping.items())},
        "n_buckets": int(n_buckets),
        "source_counts": {str(i): int((source_labels == i).sum()) for i in range(n_buckets)},
        "target_counts": {str(i): int((target_labels == i).sum()) for i in range(n_buckets)},
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
