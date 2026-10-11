"""Orient a DenseCorr3D category using its ordered ``groups.txt`` annotations.

DenseCorr3D nonanimal categories often store evaluator groups as ordered
surface bands. For shape matching, that annotation frame is more important
than a human naming convention: if all meshes put group 0 on one side and the
last group on the other side, PartField/DenseMatcher sees the same coarse
body layout across the category.

This script maps:

* +Y: first non-empty DenseCorr group
* -Y: last non-empty DenseCorr group

For elongated curved objects such as bananas, it also uses the residual curve
of group centroids to choose a stable roll, so the bend points consistently
toward +X. For round/pole objects, roll around Y is mostly irrelevant and is
chosen from mesh/group PCA for determinism.

Vertex order, face order, UVs, material references, and ``groups.txt`` indices
are preserved. The same rigid transform is applied to ``simple_mesh.obj``,
``color_mesh.obj``, and ``groups_visualization.obj`` when present.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


OBJ_VARIANTS = ("simple_mesh.obj", "color_mesh.obj", "groups_visualization.obj")
PASSTHROUGH_GLOBS = ("*.mtl", "*.png", "groups.txt")


def read_vertices(path: Path) -> np.ndarray:
    vertices: List[List[float]] = []
    with path.open("r") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(vertices, dtype=np.float64)


def read_groups(path: Path, num_vertices: int) -> Tuple[List[np.ndarray], int]:
    groups: List[np.ndarray] = []
    all_indices: List[int] = []
    with path.open("r") as handle:
        for line in handle:
            values = [int(item) for item in line.split()]
            all_indices.extend(values)
            groups.append(np.asarray(values, dtype=np.int64))

    if not groups or not all_indices:
        raise ValueError(f"No DenseCorr groups found in {path}")

    min_idx = min(all_indices)
    max_idx = max(all_indices)
    index_base = 0
    if min_idx >= 1 and max_idx == num_vertices:
        index_base = 1
        groups = [group - 1 for group in groups]

    invalid = [int(item) for group in groups for item in group if item < 0 or item >= num_vertices]
    if invalid:
        raise ValueError(f"{path} has group indices outside [0, {num_vertices}): {invalid[:8]}")
    return groups, index_base


def pca_axes(points: np.ndarray, weights: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if weights is None:
        centroid = points.mean(axis=0)
        centered = points - centroid
        cov = centered.T @ centered / max(len(centered), 1)
    else:
        weight_sum = max(float(weights.sum()), 1.0)
        centroid = np.average(points, axis=0, weights=weights)
        centered = points - centroid
        cov = (centered * weights[:, None]).T @ centered / weight_sum
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    return eigval[order], eigvec[:, order], centroid


def nonempty_group_indices(groups: Sequence[np.ndarray]) -> List[int]:
    indices = [idx for idx, group in enumerate(groups) if len(group) > 0]
    if len(indices) < 2:
        raise ValueError("Need at least two non-empty DenseCorr groups to define an axis.")
    return indices


def densecorr_group_centers(vertices: np.ndarray, groups: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    indices = nonempty_group_indices(groups)
    centers = np.asarray([vertices[groups[idx]].mean(axis=0) for idx in indices], dtype=np.float64)
    weights = np.asarray([len(groups[idx]) for idx in indices], dtype=np.float64)
    return centers, weights, indices


def orient_axis_y(
    vertices: np.ndarray,
    groups: Sequence[np.ndarray],
    reverse_groups: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    centers, weights, indices = densecorr_group_centers(vertices, groups)
    eigval, eigvec, _centroid = pca_axes(centers, weights)
    axis_y = eigvec[:, 0].copy()
    first_center = centers[0]
    last_center = centers[-1]
    desired = last_center - first_center if reverse_groups else first_center - last_center
    if float(np.dot(desired, axis_y)) < 0:
        axis_y *= -1.0

    endpoint_axis = desired.copy()
    endpoint_axis /= max(float(np.linalg.norm(endpoint_axis)), 1e-12)
    return axis_y, {
        "densecorr_first_group": int(indices[0]),
        "densecorr_last_group": int(indices[-1]),
        "densecorr_num_nonempty_groups": int(len(indices)),
        "semantic_axis_source": "weighted_densecorr_group_centroid_pca_signed_by_endpoints",
        "reverse_groups": bool(reverse_groups),
        "semantic_axis_raw": np.round(axis_y, 8).tolist(),
        "semantic_endpoint_axis_raw": np.round(endpoint_axis, 8).tolist(),
        "group_centroid_eigval": np.round(eigval, 10).tolist(),
    }


def choose_axis_x(vertices: np.ndarray, groups: Sequence[np.ndarray], axis_y: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    centers, weights, _indices = densecorr_group_centers(vertices, groups)
    centered_centers = centers - np.average(centers, axis=0, weights=weights)
    along = centered_centers @ axis_y
    residual = centered_centers - np.outer(along, axis_y)
    residual_norm = np.linalg.norm(residual, axis=1)

    roll_source = "mesh_pca_secondary_axis"
    if float(residual_norm.max(initial=0.0)) > 1e-8:
        eigval, eigvec, _ = pca_axes(residual, weights)
        candidate = eigvec[:, 0] - axis_y * float(np.dot(eigvec[:, 0], axis_y))
        if float(np.linalg.norm(candidate)) > 1e-8 and eigval[0] > 1e-12:
            axis_x = candidate / float(np.linalg.norm(candidate))
            # The middle of a curved centroid path should bow toward +X.
            y_lo, y_hi = np.quantile(along, [0.35, 0.65])
            mid_mask = (along >= y_lo) & (along <= y_hi)
            bow = residual[mid_mask].mean(axis=0) if mid_mask.any() else residual.mean(axis=0)
            if float(np.dot(bow, axis_x)) < 0:
                axis_x *= -1.0
            return axis_x, {
                "roll_source": "densecorr_centroid_curve_residual_pca",
                "roll_residual_eigval": np.round(eigval, 10).tolist(),
            }

    eigval, eigvec, _ = pca_axes(vertices)
    best_axis = None
    best_norm = -1.0
    for col in range(eigvec.shape[1]):
        candidate = eigvec[:, col] - axis_y * float(np.dot(eigvec[:, col], axis_y))
        norm = float(np.linalg.norm(candidate))
        if norm > best_norm:
            best_axis = candidate
            best_norm = norm
    if best_axis is None or best_norm < 1e-8:
        candidates = np.eye(3)
        seed = min(candidates, key=lambda item: abs(float(np.dot(item, axis_y))))
        best_axis = seed - axis_y * float(np.dot(seed, axis_y))
        best_norm = float(np.linalg.norm(best_axis))

    axis_x = best_axis / best_norm
    dominant = int(np.argmax(np.abs(axis_x)))
    if axis_x[dominant] < 0:
        axis_x *= -1.0
    return axis_x, {"roll_source": roll_source, "mesh_eigval": np.round(eigval, 10).tolist()}


def group_y_means(vertices: np.ndarray, groups: Sequence[np.ndarray]) -> List[float]:
    y = vertices[:, 1]
    means: List[float] = []
    for group in groups:
        means.append(float(y[group].mean()) if len(group) else float("nan"))
    return means


def radius_profile(vertices: np.ndarray, bins: int = 10) -> List[Dict[str, float]]:
    y = vertices[:, 1]
    lo, hi = np.quantile(y, [0.02, 0.98])
    edges = np.linspace(lo, hi, bins + 1)
    profile: List[Dict[str, float]] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        mask = (y >= start) & (y < stop)
        if not mask.any():
            radius = 0.0
            count = 0
        else:
            xz = vertices[mask][:, [0, 2]]
            center = np.median(xz, axis=0)
            radius = float(np.linalg.norm(xz - center, axis=1).mean())
            count = int(mask.sum())
        profile.append({"center": float((start + stop) * 0.5), "radius": radius, "count": float(count)})
    return profile


def orient_frame(
    vertices: np.ndarray,
    groups: Sequence[np.ndarray],
    reverse_groups: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    mesh_eigval, _mesh_eigvec, centroid = pca_axes(vertices)
    axis_y, y_diag = orient_axis_y(vertices, groups, reverse_groups=reverse_groups)
    axis_y /= max(float(np.linalg.norm(axis_y)), 1e-12)
    axis_x, x_diag = choose_axis_x(vertices, groups, axis_y)

    axis_z = np.cross(axis_x, axis_y)
    axis_z /= max(float(np.linalg.norm(axis_z)), 1e-12)
    axis_x = np.cross(axis_y, axis_z)
    axis_x /= max(float(np.linalg.norm(axis_x)), 1e-12)

    r = np.stack([axis_x, axis_y, axis_z], axis=0)
    det = float(np.linalg.det(r))
    if det < 0:
        axis_z *= -1.0
        r = np.stack([axis_x, axis_y, axis_z], axis=0)
        det = float(np.linalg.det(r))
    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(f"Expected a proper rotation, got det={det:.8f}")

    rotated = (vertices - centroid) @ r.T
    bbox_center = (rotated.min(axis=0) + rotated.max(axis=0)) * 0.5
    shift = -bbox_center
    final = rotated + shift

    means = group_y_means(final, groups)
    valid = [value for value in means if np.isfinite(value)]
    if reverse_groups:
        ordered_pairs = sum(1 for a, b in zip(valid[:-1], valid[1:]) if a <= b)
    else:
        ordered_pairs = sum(1 for a, b in zip(valid[:-1], valid[1:]) if a >= b)
    ordered_fraction = ordered_pairs / max(len(valid) - 1, 1)

    diagnostics: Dict[str, object] = {
        **y_diag,
        **x_diag,
        "rotation": np.round(r, 10).tolist(),
        "centroid": np.round(centroid, 10).tolist(),
        "shift": np.round(shift, 10).tolist(),
        "mesh_eigval": np.round(mesh_eigval, 10).tolist(),
        "bbox_extent_before": np.round(vertices.max(axis=0) - vertices.min(axis=0), 8).tolist(),
        "bbox_extent_after": np.round(final.max(axis=0) - final.min(axis=0), 8).tolist(),
        "densecorr_group_y_ordered_fraction": round(float(ordered_fraction), 6),
        "densecorr_group_y_mean_first_to_last": [
            round(value, 7) if np.isfinite(value) else None for value in means
        ],
        "y_profile_bottom_to_top": [
            {"center": round(item["center"], 6), "radius": round(item["radius"], 6), "count": int(item["count"])}
            for item in radius_profile(final)
        ],
    }
    return r, centroid, shift, diagnostics


def transform_vertex(parts: Sequence[str], r: np.ndarray, centroid: np.ndarray, shift: np.ndarray) -> str:
    xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    transformed = r @ (xyz - centroid) + shift
    rest = list(parts[4:])
    return "v " + " ".join([f"{value:.9g}" for value in transformed] + rest) + "\n"


def transform_normal(parts: Sequence[str], r: np.ndarray) -> str:
    normal = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    transformed = r @ normal
    norm = float(np.linalg.norm(transformed))
    if norm > 1e-12:
        transformed /= norm
    return "vn " + " ".join(f"{value:.8g}" for value in transformed) + "\n"


def transform_obj_file(src: Path, dst: Path, r: np.ndarray, centroid: np.ndarray, shift: np.ndarray) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    same_path = src.resolve() == dst.resolve()
    if same_path:
        tmp_handle = tempfile.NamedTemporaryFile("w", dir=str(dst.parent), delete=False)
        tmp_path = Path(tmp_handle.name)
    else:
        tmp_path = dst
        tmp_handle = dst.open("w")

    try:
        with src.open("r") as in_handle, tmp_handle as out_handle:
            for line in in_handle:
                if line.startswith("v "):
                    out_handle.write(transform_vertex(line.split(), r, centroid, shift))
                elif line.startswith("vn "):
                    out_handle.write(transform_normal(line.split(), r))
                else:
                    out_handle.write(line)
        if same_path:
            shutil.copymode(src, tmp_path)
            os.replace(tmp_path, dst)
    except Exception:
        if same_path:
            tmp_path.unlink(missing_ok=True)
        raise


def copy_passthrough_files(src_dir: Path, dst_dir: Path) -> None:
    if src_dir.resolve() == dst_dir.resolve():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in PASSTHROUGH_GLOBS:
        for src in src_dir.glob(pattern):
            shutil.copy2(src, dst_dir / src.name)


def process_instance(src_dir: Path, dst_dir: Path, reverse_groups: bool = False) -> Dict[str, object]:
    ref_obj = src_dir / "simple_mesh.obj"
    vertices = read_vertices(ref_obj)
    groups, index_base = read_groups(src_dir / "groups.txt", len(vertices))
    r, centroid, shift, diagnostics = orient_frame(vertices, groups, reverse_groups=reverse_groups)

    processed: List[str] = []
    for name in OBJ_VARIANTS:
        src = src_dir / name
        if src.is_file():
            transform_obj_file(src, dst_dir / name, r, centroid, shift)
            processed.append(name)
    copy_passthrough_files(src_dir, dst_dir)

    return {
        "instance": src_dir.name,
        "num_vertices": int(len(vertices)),
        "group_index_base_detected": int(index_base),
        "num_densecorr_groups": int(len(groups)),
        "processed_obj_variants": processed,
        **diagnostics,
    }


def run(args: argparse.Namespace) -> None:
    category = args.category
    input_root = Path(args.input_root) if args.input_root else Path("external/DenseCorr3D") / category
    output_root = Path(args.output_root) if args.output_root else input_root
    manifest_path = output_root / f"{category}_densecorr_group_upright_orientation_manifest.json"
    if input_root.resolve() == output_root.resolve() and manifest_path.exists() and not args.force:
        raise SystemExit(
            f"{manifest_path} already exists. Refusing to run in-place again; "
            "pass --force only if you intentionally want to reprocess this directory."
        )

    instance_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    rows: List[Dict[str, object]] = []
    for src_dir in instance_dirs:
        if not (src_dir / "simple_mesh.obj").is_file():
            continue
        row = process_instance(src_dir, output_root / src_dir.name, reverse_groups=args.reverse_groups)
        rows.append(row)
        extent = row["bbox_extent_after"]
        order = row["densecorr_group_y_ordered_fraction"]
        sign_text = "+Y last, -Y group0" if args.reverse_groups else "+Y group0, -Y last"
        print(f"{src_dir.name}: {sign_text}, extent={extent}, group_order={order}")

    manifest = {
        "category": category,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "vertical_axis": "+Y",
        "top_axis": "+Y last non-empty DenseCorr group" if args.reverse_groups else "+Y first non-empty DenseCorr group",
        "bottom_axis": "-Y first non-empty DenseCorr group" if args.reverse_groups else "-Y last non-empty DenseCorr group",
        "reverse_groups": bool(args.reverse_groups),
        "num_instances": len(rows),
        "items": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {manifest_path}")
    print(f"{len(rows)} {category} instances processed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--reverse-groups", action="store_true", help="Put the last non-empty group at +Y and group 0 at -Y.")
    parser.add_argument("--force", action="store_true", help="Allow reprocessing when output-root already has a manifest.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
