"""Orient DenseCorr3D pear meshes with stem/taper upward.

The DenseCorr3D pear scans arrive in mixed axes. For PartField-style category
segmentation we want one consistent frame:

* +Y: stem / narrow tapered top
* -Y: wider pear body / bottom

For each instance directory this script estimates the pear growth axis from
``simple_mesh.obj`` via PCA, flips that axis so the wider radial mass is below
(-Y), and applies the same rigid transform to ``simple_mesh.obj``,
``color_mesh.obj``, and ``groups_visualization.obj``. Vertex order, face order,
UVs, materials, and ``groups.txt`` indices are not changed.
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


def pca_axes(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    cov = centered.T @ centered / max(len(centered), 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    return eigval[order], eigvec[:, order], centroid


def radial_moment_sign(vertices: np.ndarray, axis: np.ndarray, centroid: np.ndarray) -> float:
    """Positive means wider radial mass is on the +axis side."""
    centered = vertices - centroid
    score = centered @ axis
    radial_vec = centered - np.outer(score, axis)
    radial2 = np.einsum("ij,ij->i", radial_vec, radial_vec)
    denom = float(radial2.sum())
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(score, radial2) / denom)


def binned_radius_profile(
    vertices: np.ndarray,
    axis: np.ndarray,
    centroid: np.ndarray,
    bins: int = 9,
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


def orient_frame(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    eigval, eigvec, centroid = pca_axes(vertices)

    axis_y = eigvec[:, 0].copy()
    moment = radial_moment_sign(vertices, axis_y, centroid)
    # The pear's widest radial mass should be below the stem, i.e. at -Y.
    # If the PCA-positive side is wider, flip so that side becomes negative Y.
    flipped_growth_axis = bool(moment > 0.0)
    if flipped_growth_axis:
        axis_y *= -1.0
        moment = -moment

    axis_x = eigvec[:, 1] - axis_y * float(np.dot(eigvec[:, 1], axis_y))
    axis_x_norm = float(np.linalg.norm(axis_x))
    if axis_x_norm < 1e-8:
        candidates = np.eye(3)
        seed = min(candidates, key=lambda item: abs(float(np.dot(item, axis_y))))
        axis_x = seed - axis_y * float(np.dot(seed, axis_y))
        axis_x_norm = float(np.linalg.norm(axis_x))
    axis_x = axis_x / axis_x_norm

    dominant = int(np.argmax(np.abs(axis_x)))
    if axis_x[dominant] < 0:
        axis_x *= -1.0

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
    profile = binned_radius_profile(final, np.array([0.0, 1.0, 0.0]), np.zeros(3))

    diagnostics: Dict[str, object] = {
        "eigval": np.round(eigval, 10).tolist(),
        "growth_axis_raw": np.round(axis_y, 8).tolist(),
        "flipped_growth_axis": flipped_growth_axis,
        "wide_radial_moment_y": round(moment, 8),
        "rotation": np.round(r, 10).tolist(),
        "centroid": np.round(centroid, 10).tolist(),
        "shift": np.round(shift, 10).tolist(),
        "bbox_extent_before": np.round(vertices.max(axis=0) - vertices.min(axis=0), 8).tolist(),
        "bbox_extent_after": np.round(final.max(axis=0) - final.min(axis=0), 8).tolist(),
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
    r, centroid, shift, diagnostics = orient_frame(vertices)

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
        "processed_obj_variants": processed,
        **diagnostics,
    }


def run(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    manifest_path = output_root / "pear_stem_up_orientation_manifest.json"
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
        print(f"{src_dir.name}: +Y stem, -Y body/bottom, extent={extent}")

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "vertical_axis": "+Y",
        "bottom_axis": "-Y",
        "num_instances": len(rows),
        "items": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {manifest_path}")
    print(f"{len(rows)} pear instances processed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="external/DenseCorr3D/pear")
    parser.add_argument("--output-root", default="external/DenseCorr3D/pear")
    parser.add_argument("--force", action="store_true", help="Allow reprocessing when output-root already has a manifest.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
