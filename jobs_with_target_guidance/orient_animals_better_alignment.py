"""Create a DenseCorr3D animal category with a consistent side-profile frame.

The toy animal meshes are already mostly y-up, but a couple of instances use
z as the front/back body axis. DenseMatcher/PartField behaves better when the
category shares a stable object frame, so this script keeps feet/down on -Y,
puts the body side profile along X, and re-centers every OBJ variant.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from jobs_with_target_guidance.orient_densecorr_category_by_groups import (
    OBJ_VARIANTS,
    PASSTHROUGH_GLOBS,
    read_vertices,
    transform_obj_file,
)


# These two animals render as side profiles only from the z-axis view. Mapping
# old +Z to new +X makes their head/front direction match the rest of the set.
YAW_Z_TO_X = {
    "188f4_toy_animals_022",
    "d1889_dinosaur_002",
}


def rotation_for_instance(instance: str) -> np.ndarray:
    if instance in YAW_Z_TO_X:
        return np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    return np.eye(3, dtype=np.float64)


def bbox_center(vertices: np.ndarray) -> np.ndarray:
    return (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5


def copy_passthrough_files(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in PASSTHROUGH_GLOBS:
        for src in src_dir.glob(pattern):
            shutil.copy2(src, dst_dir / src.name)


def process_instance(src_dir: Path, dst_dir: Path) -> Dict[str, object]:
    vertices = read_vertices(src_dir / "simple_mesh.obj")
    rotation = rotation_for_instance(src_dir.name)
    centroid = bbox_center(vertices)
    rotated = (vertices - centroid) @ rotation.T
    shift = -bbox_center(rotated)

    processed: List[str] = []
    for name in OBJ_VARIANTS:
        src = src_dir / name
        if src.is_file():
            transform_obj_file(src, dst_dir / name, rotation, centroid, shift)
            processed.append(name)
    copy_passthrough_files(src_dir, dst_dir)

    final_vertices = rotated + shift
    return {
        "instance": src_dir.name,
        "operation": "yaw_z_to_x" if src_dir.name in YAW_Z_TO_X else "recenter_only",
        "processed_obj_variants": processed,
        "bbox_extent_before": np.round(vertices.max(axis=0) - vertices.min(axis=0), 8).tolist(),
        "bbox_extent_after": np.round(final_vertices.max(axis=0) - final_vertices.min(axis=0), 8).tolist(),
        "bbox_center_after": np.round(bbox_center(final_vertices), 10).tolist(),
        "rotation": np.round(rotation, 10).tolist(),
        "centroid": np.round(centroid, 10).tolist(),
        "shift": np.round(shift, 10).tolist(),
    }


def run(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    manifest_path = output_root / "animals_better_alignment_manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"{manifest_path} already exists; pass --force to overwrite.")

    rows: List[Dict[str, object]] = []
    for src_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if not (src_dir / "simple_mesh.obj").is_file():
            continue
        row = process_instance(src_dir, output_root / src_dir.name)
        rows.append(row)
        print(f"{src_dir.name}: {row['operation']}, extent={row['bbox_extent_after']}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "category": "animals",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "convention": "+Y up, -Y feet/down, +X head/front side-profile where identifiable",
        "num_instances": len(rows),
        "yaw_z_to_x_instances": sorted(YAW_Z_TO_X),
        "items": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {manifest_path}")
    print(f"{len(rows)} animal instances processed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="external/DenseCorr3D/animals")
    parser.add_argument("--output-root", default="external/DenseCorr3D/animals_better_alignment")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
