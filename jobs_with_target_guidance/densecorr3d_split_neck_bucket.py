"""Split the DenseCorr3D torso bucket into torso + neck.

DenseCorr3D's 8-group `animals` category scheme has no dedicated neck group:
legs are groups {0,2,3,6}, head is group 4, snout/trunk is group 5,
ears/antlers are group 7, and the torso (group 1) is by far the largest group
on every animal (verified across elephant/giraffe/moose/bear/panther/cheetah).
For a giraffe, the neck is silently absorbed into the torso group, so a
per-bucket Chamfer loss has no dedicated target that says "this region should
be long and thin" -- it treats torso+neck as one blob.

This script splits group 1 into two groups by a vertical-axis (Y) median cut
on unit-box-normalized coordinates: the upper half (closer to the head)
becomes a new "neck" bucket (appended as the next free label id), the lower
half keeps the original torso label. Every other group passes through
unchanged. Because the same fixed rule (group 1, Y axis, median) is applied
independently per animal, the new neck bucket id stays semantically aligned
across animals, so partfield_labels_aligned=1 cross-animal bucket matching
still works without needing a paired source+target step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jobs_with_target_guidance.partfield_segment import (  # noqa: E402
    labels_to_face_labels,
    load_obj_mesh,
    write_face_colored_ply,
)

TORSO_LABEL = 1
VERTICAL_AXIS = 1  # y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-colored", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--torso-label", type=int, default=TORSO_LABEL)
    parser.add_argument("--vertical-axis", type=int, default=VERTICAL_AXIS, choices=(0, 1, 2))
    parser.add_argument("--min-split-count", type=int, default=24, help="Skip the split if torso has fewer vertices than this.")
    return parser.parse_args()


def load_labels(path: Path) -> np.ndarray:
    data = np.load(path)
    for key in ("labels", "vertex_labels", "part_labels"):
        if key in data:
            return np.asarray(data[key], dtype=np.int64).reshape(-1)
    return np.asarray(data[data.files[0]], dtype=np.int64).reshape(-1)


def unit_vertices(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    scale = 2.0 / max(float((vmax - vmin).max()), 1e-12)
    return (vertices - (vmax + vmin) * 0.5) * scale


def split_torso(
    vertices: np.ndarray,
    labels: np.ndarray,
    torso_label: int,
    vertical_axis: int,
    min_split_count: int,
) -> tuple[np.ndarray, dict]:
    unit = unit_vertices(vertices)
    torso_idx = np.nonzero(labels == torso_label)[0]
    new_labels = labels.copy()

    if torso_idx.size < min_split_count:
        return new_labels, {
            "torso_label": torso_label,
            "neck_label": None,
            "torso_count_before": int(torso_idx.size),
            "skipped": True,
            "reason": f"fewer than {min_split_count} torso vertices",
        }

    neck_label = int(labels.max()) + 1
    y = unit[torso_idx, vertical_axis]
    median = float(np.median(y))
    is_neck = y > median
    new_labels[torso_idx[is_neck]] = neck_label

    return new_labels, {
        "torso_label": torso_label,
        "neck_label": neck_label,
        "vertical_axis": vertical_axis,
        "median_y": median,
        "torso_count_before": int(torso_idx.size),
        "torso_count_after": int((~is_neck).sum()),
        "neck_count_after": int(is_neck.sum()),
        "skipped": False,
    }


def main() -> None:
    args = parse_args()

    vertices, faces = load_obj_mesh(args.mesh)
    labels = load_labels(args.labels)
    if labels.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"Label count {labels.shape[0]} does not match vertex count {vertices.shape[0]} for {args.mesh}"
        )

    new_labels, split_summary = split_torso(
        vertices,
        labels,
        torso_label=args.torso_label,
        vertical_axis=args.vertical_axis,
        min_split_count=args.min_split_count,
    )

    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    n_buckets = int(new_labels.max()) + 1
    face_labels = labels_to_face_labels(new_labels, faces)
    np.savez(
        args.output_labels,
        labels=new_labels,
        vertex_labels=new_labels,
        face_labels=face_labels,
        n_buckets=np.asarray(n_buckets, dtype=np.int64),
    )

    if args.output_colored:
        write_face_colored_ply(args.output_colored, vertices, faces, face_labels)

    summary = {
        "mesh": str(args.mesh),
        "labels": str(args.labels),
        "n_buckets_before": int(labels.max()) + 1,
        "n_buckets_after": n_buckets,
        "split": split_summary,
    }
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_labels} ({n_buckets} buckets)")


if __name__ == "__main__":
    main()
