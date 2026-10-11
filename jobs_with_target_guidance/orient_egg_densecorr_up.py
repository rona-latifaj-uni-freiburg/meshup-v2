"""Orient DenseCorr3D egg meshes in the evaluator's top-to-bottom frame.

DenseCorr3D egg annotations are ordered as latitude-like rings: group 0 is the
top pole, intermediate lines are body halos, and the last non-empty group is the
bottom pole. For DenseMatcher / PartField evaluation we want every egg in that
same frame:

* +Y: DenseCorr group 0 / evaluated top pole
* -Y: final DenseCorr group / evaluated bottom pole

The script estimates that semantic pole axis from ``groups.txt`` and
``simple_mesh.obj``, then applies the same rigid transform to
``simple_mesh.obj``, ``color_mesh.obj``, and ``groups_visualization.obj``.
Vertex order, face order, UVs, materials, and ``groups.txt`` indices are not
changed.
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
        sample = invalid[:8]
        raise ValueError(f"{path} has group indices outside [0, {num_vertices}): {sample}")

    return groups, index_base


def pca_axes(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    cov = centered.T @ centered / max(len(centered), 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    return eigval[order], eigvec[:, order], centroid


def nonempty_group_endpoints(groups: Sequence[np.ndarray]) -> Tuple[int, int]:
    nonempty = [idx for idx, group in enumerate(groups) if len(group) > 0]
    if len(nonempty) < 2:
        raise ValueError("Need at least two non-empty DenseCorr groups to define the egg axis.")
    return nonempty[0], nonempty[-1]


def binned_radius_profile(
    vertices: np.ndarray,
    axis: np.ndarray,
    centroid: np.ndarray,
    bins: int = 10,
) -> List[Dict[str, float]]:
    centered = vertices - centroid
    score = centered @ axis
    lo, hi = np.quantile(score, [0.02, 0.98])
    edges = np.linspace(lo, hi, bins + 1)
    profile: List[Dict[str, float]] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        mask = (score >= start) & (score < stop)
        count = int(mask.sum())
        if count == 0:
            radius = 0.0
        else:
            local_score = score[mask]
            radial_vec = centered[mask] - np.outer(local_score, axis)
            radial = np.linalg.norm(radial_vec, axis=1)
            radius = float(radial.mean())
        profile.append(
            {
                "center": float((start + stop) * 0.5),
                "radius": radius,
                "count": float(count),
            }
        )
    return profile


def tail_radius(vertices: np.ndarray, high_side: bool, quantile_width: float = 0.12) -> float:
    y = vertices[:, 1]
    if high_side:
        cutoff = np.quantile(y, 1.0 - quantile_width)
        mask = y >= cutoff
    else:
        cutoff = np.quantile(y, quantile_width)
        mask = y <= cutoff
    xz = vertices[mask][:, [0, 2]]
    center = np.median(xz, axis=0)
    return float(np.linalg.norm(xz - center, axis=1).mean())


def densecorr_group_y_means(vertices: np.ndarray, groups: Sequence[np.ndarray]) -> List[float]:
    means: List[float] = []
    y = vertices[:, 1]
    for group in groups:
        if len(group) == 0:
            means.append(float("nan"))
        else:
            means.append(float(y[group].mean()))
    return means


def densecorr_centroid_axis(
    vertices: np.ndarray,
    groups: Sequence[np.ndarray],
    top_group: int,
    bottom_group: int,
) -> np.ndarray:
    centers: List[np.ndarray] = []
    weights: List[float] = []
    for group in groups:
        if len(group) == 0:
            continue
        centers.append(vertices[group].mean(axis=0))
        weights.append(float(len(group)))

    if len(centers) < 2:
        raise ValueError("Need at least two group centroids to fit a DenseCorr axis.")

    center_arr = np.asarray(centers, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    weighted_mean = np.average(center_arr, axis=0, weights=weight_arr)
    centered = center_arr - weighted_mean
    cov = (centered * weight_arr[:, None]).T @ centered / max(float(weight_arr.sum()), 1.0)
    eigval, eigvec = np.linalg.eigh(cov)
    axis = eigvec[:, int(np.argmax(eigval))]

    top_center = vertices[groups[top_group]].mean(axis=0)
    bottom_center = vertices[groups[bottom_group]].mean(axis=0)
    if float(np.dot(top_center - bottom_center, axis)) < 0:
        axis *= -1.0
    return axis


def choose_secondary_axis(eigvec: np.ndarray, axis_y: np.ndarray) -> np.ndarray:
    best_axis = None
    best_norm = -1.0
    for col in range(eigvec.shape[1]):
        candidate = eigvec[:, col]
        candidate = candidate - axis_y * float(np.dot(candidate, axis_y))
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
    return axis_x


def orient_frame(
    vertices: np.ndarray,
    groups: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    eigval, eigvec, centroid = pca_axes(vertices)
    top_group, bottom_group = nonempty_group_endpoints(groups)
    top_center = vertices[groups[top_group]].mean(axis=0)
    bottom_center = vertices[groups[bottom_group]].mean(axis=0)

    endpoint_axis = top_center - bottom_center
    axis_y = densecorr_centroid_axis(vertices, groups, top_group, bottom_group)
    axis_norm = float(np.linalg.norm(axis_y))
    if axis_norm < 1e-8:
        axis_y = eigvec[:, 0].copy()
        axis_norm = float(np.linalg.norm(axis_y))
    axis_y = axis_y / axis_norm

    axis_x = choose_secondary_axis(eigvec, axis_y)
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

    final_group_means = densecorr_group_y_means(final, groups)
    valid_means = [value for value in final_group_means if np.isfinite(value)]
    descending_pairs = sum(1 for a, b in zip(valid_means[:-1], valid_means[1:]) if a >= b)
    monotonic_fraction = descending_pairs / max(len(valid_means) - 1, 1)
    profile = binned_radius_profile(final, np.array([0.0, 1.0, 0.0]), np.zeros(3))

    diagnostics: Dict[str, object] = {
        "eigval": np.round(eigval, 10).tolist(),
        "densecorr_top_group": int(top_group),
        "densecorr_bottom_group": int(bottom_group),
        "densecorr_top_center_raw": np.round(top_center, 10).tolist(),
        "densecorr_bottom_center_raw": np.round(bottom_center, 10).tolist(),
        "semantic_axis_source": "weighted_densecorr_group_centroid_pca_signed_by_endpoints",
        "semantic_axis_raw": np.round(axis_y, 8).tolist(),
        "semantic_endpoint_axis_raw": np.round(endpoint_axis / max(float(np.linalg.norm(endpoint_axis)), 1e-12), 8).tolist(),
        "rotation": np.round(r, 10).tolist(),
        "centroid": np.round(centroid, 10).tolist(),
        "shift": np.round(shift, 10).tolist(),
        "bbox_extent_before": np.round(vertices.max(axis=0) - vertices.min(axis=0), 8).tolist(),
        "bbox_extent_after": np.round(final.max(axis=0) - final.min(axis=0), 8).tolist(),
        "densecorr_group_y_mean_bottom_to_top_order_check": [
            round(value, 7) if np.isfinite(value) else None for value in final_group_means
        ],
        "densecorr_group_y_descending_fraction": round(float(monotonic_fraction), 6),
        "top_tail_radius": round(tail_radius(final, high_side=True), 7),
        "bottom_tail_radius": round(tail_radius(final, high_side=False), 7),
        "y_profile_bottom_to_top": [
            {"center": round(item["center"], 6), "radius": round(item["radius"], 6), "count": int(item["count"])}
            for item in profile
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


def process_instance(src_dir: Path, dst_dir: Path) -> Dict[str, object]:
    ref_obj = src_dir / "simple_mesh.obj"
    vertices = read_vertices(ref_obj)
    groups, index_base = read_groups(src_dir / "groups.txt", len(vertices))
    r, centroid, shift, diagnostics = orient_frame(vertices, groups)

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
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    manifest_path = output_root / "egg_densecorr_upright_orientation_manifest.json"
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
        row = process_instance(src_dir, output_root / src_dir.name)
        rows.append(row)
        extent = row["bbox_extent_after"]
        desc = row["densecorr_group_y_descending_fraction"]
        print(f"{src_dir.name}: +Y group0/top, -Y last/bottom, extent={extent}, group_order={desc}")

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "vertical_axis": "+Y",
        "top_axis": "+Y DenseCorr group 0",
        "bottom_axis": "-Y final DenseCorr group",
        "num_instances": len(rows),
        "items": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {manifest_path}")
    print(f"{len(rows)} egg instances processed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="external/DenseCorr3D/egg")
    parser.add_argument("--output-root", default="external/DenseCorr3D/egg")
    parser.add_argument("--force", action="store_true", help="Allow reprocessing when output-root already has a manifest.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
