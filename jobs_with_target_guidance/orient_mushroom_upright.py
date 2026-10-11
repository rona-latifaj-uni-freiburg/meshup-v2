"""Orient mushroom scan meshes so the stem is down and the cap is up.

For each instance under --input-root (one folder per scan, containing
simple_mesh.obj / color_mesh.obj / groups_visualization.obj sharing the same
vertex positions), this applies a rotation that maps the mushroom's growth
axis (stem-to-cap direction) onto +Z, then reproduces every OBJ variant plus
the associated mtl/texture/group files under --output-root.

The rotation decision per instance is human-verified, not purely automatic.
detect_growth_axis()/build_rotation() implement a geometric heuristic
(compare cross-sectional spread between the upper and lower half of the
mesh along each axis: a dome cap is wide across most of its half, a stem is
uniformly thin), but across this dataset that heuristic's confidence ratios
mostly fall in the noisy 1.0-1.4 range and it flagged many already-correct
meshes as needing rotation (false positives from natural shape asymmetry).
All 72 instances were rendered from the view_xz() camera and visually
reviewed; MANUAL_OVERRIDES below encodes that review:
  - 70 instances are already correct (identity, no rotation).
  - 2 instances (ab971de2..., c431b924...) are genuinely upside-down
    (wide bulb at the bottom, narrow column at top) and get a 180 degree
    flip about X (keeps X, negates Y and Z -- a proper rotation, no
    mirroring).
  - 7 instances are partial/ambiguous scans (no clear stem+cap silhouette,
    e.g. just a dome fragment or an off-axis blob) where "up" can't be
    determined from geometry alone; these are left untouched and flagged
    in the manifest for manual review rather than guessed at.

Vertex/normal lines are rewritten in place (rotation only, vertex order and
count unchanged); everything else (faces, uv coords, mtllib/usemtl,
comments) is copied through untouched, and groups.txt (vertex-index lists)
stays valid since no vertices are added, removed, or reordered.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def read_vertices(path: Path) -> np.ndarray:
    verts = []
    with path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=np.float64)


def cross_sectional_radius(points_2d: np.ndarray) -> float:
    if len(points_2d) == 0:
        return 0.0
    center = points_2d.mean(axis=0)
    return float(np.linalg.norm(points_2d - center, axis=1).mean())


def detect_growth_axis(vertices: np.ndarray, half_frac: float = 0.5) -> Tuple[int, int, float]:
    """Return (growth_axis, cap_sign, confidence).

    Splits vertices into lower/upper halves along each candidate axis and
    compares their cross-sectional radius. A dome-shaped cap is wide across
    most of its half (its equator dominates the vertex count, not the narrow
    apex tip), while a stem is uniformly thin along its whole half, so a
    half/half split is robust where comparing only the extreme tail slices
    is not (the tail slice near a dome's apex is itself narrow, and can
    misidentify the growth axis).

    cap_sign is +1 if the cap (wide end) is on the positive side of the
    growth axis, -1 otherwise. confidence is the ratio of wide-end to
    narrow-end cross-sectional radius (higher = more certain).
    """
    n = len(vertices)
    k = max(1, int(n * half_frac))
    best = (2, 1, 0.0)
    for axis in range(3):
        order = np.argsort(vertices[:, axis])
        other = [a for a in range(3) if a != axis]
        low_pts = vertices[order[:k]][:, other]
        high_pts = vertices[order[-k:]][:, other]
        low_r = cross_sectional_radius(low_pts)
        high_r = cross_sectional_radius(high_pts)
        wide, narrow = max(low_r, high_r), min(low_r, high_r)
        ratio = wide / narrow if narrow > 1e-9 else float("inf")
        if ratio > best[2]:
            cap_sign = 1 if high_r >= low_r else -1
            best = (axis, cap_sign, ratio)
    return best


def build_rotation(growth_axis: int, cap_sign: int) -> np.ndarray:
    e = np.eye(3)
    h = next(i for i in range(3) if i != growth_axis)
    new_z = cap_sign * e[growth_axis]
    new_x = e[h]
    new_y = np.cross(new_z, new_x)
    return np.stack([new_x, new_y, new_z], axis=0)


def rotate_obj_file(src: Path, dst: Path, R: np.ndarray) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_lines = []
    with src.open("r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                new_xyz = R @ xyz
                rest = parts[4:]
                fields = [f"{v:.7g}" for v in new_xyz] + rest
                out_lines.append("v " + " ".join(fields) + "\n")
            elif line.startswith("vn "):
                parts = line.split()
                n = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                new_n = R @ n
                out_lines.append("vn " + " ".join(f"{v:.6g}" for v in new_n) + "\n")
            else:
                out_lines.append(line)
    dst.write_text("".join(out_lines))


OBJ_VARIANTS = ["simple_mesh.obj", "color_mesh.obj", "groups_visualization.obj"]
PASSTHROUGH_GLOBS = ["*.mtl", "*.png", "groups.txt"]

# Human-verified from rendering every instance under the view_xz() camera
# (see module docstring). Instances not listed here default to "ok".
#   "ok"        -- already correct, no rotation.
#   "flip180"   -- genuinely upside-down (cap at -Z), rotate 180 deg about X.
#   "flip180z"  -- right-side up but the stem attaches off-center and points
#                  the wrong way in-plane; rotate 180 deg about Z (vertical)
#                  to swing it to the opposite side of the cap without
#                  touching which way is "up".
#   "unclear"   -- partial/ambiguous scan, left untouched, flagged for review.
MANUAL_REVIEW: Dict[str, str] = {
    "ab971de201d64bce81dbd51d4054eb74": "flip180",
    "c431b92427a94a7d86c1e2790d94a798": "flip180",
    "f965065b1e684e079f689287e5913003": "flip180z",
    "127dadb8391b40ad86903ddabefbeb72": "unclear",
    "4505b22d84e046a2813a33b8d8c2746f": "unclear",
    "60d65f9215d74b4dbf2efef4577403ef": "unclear",
    "bd15dd4edc6047e8adcfb51972b97e7a": "unclear",
    "da2c388cd499491b9a15a0dd58f10a00": "unclear",
    "ffda30903e744bd3b5bbe22abbfa453f": "unclear",
}

FLIP_180_X = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
FLIP_180_Z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
ROTATIONS = {"flip180": FLIP_180_X, "flip180z": FLIP_180_Z}


def process_instance(src_dir: Path, dst_dir: Path) -> Dict[str, object]:
    ref_obj = src_dir / "simple_mesh.obj"
    vertices = read_vertices(ref_obj)
    growth_axis, cap_sign, confidence = detect_growth_axis(vertices)

    verdict = MANUAL_REVIEW.get(src_dir.name, "ok")
    R = ROTATIONS.get(verdict, np.eye(3))  # "ok" and "unclear" both pass through unrotated.
    identity = np.allclose(R, np.eye(3))

    for name in OBJ_VARIANTS:
        src = src_dir / name
        if src.is_file():
            rotate_obj_file(src, dst_dir / name, R)

    for pattern in PASSTHROUGH_GLOBS:
        for src in src_dir.glob(pattern):
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)

    return {
        "instance": src_dir.name,
        "verdict": verdict,
        "identity_rotation": identity,
        "heuristic_growth_axis": "xyz"[growth_axis],
        "heuristic_cap_sign": cap_sign,
        "heuristic_confidence": round(confidence, 3),
        "num_vertices": int(len(vertices)),
    }


def run(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    instance_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())

    rows: List[Dict[str, object]] = []
    for inst in instance_dirs:
        if not (inst / "simple_mesh.obj").is_file():
            continue
        row = process_instance(inst, output_root / inst.name)
        rows.append(row)
        print(f"{row['instance']}: {row['verdict']}")

    flipped = [r for r in rows if r["verdict"] in ("flip180", "flip180z")]
    unclear = [r for r in rows if r["verdict"] == "unclear"]
    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_instances": len(rows),
        "flipped_instances": [r["instance"] for r in flipped],
        "unclear_instances": [r["instance"] for r in unclear],
        "items": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "orientation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {manifest_path}")
    print(
        f"{len(rows)} instances processed: {len(flipped)} flipped, "
        f"{len(unclear)} left as-is and flagged unclear, "
        f"{len(rows) - len(flipped) - len(unclear)} already correct."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="external/DenseCorr3D/mushroom")
    parser.add_argument("--output-root", default="external/DenseCorr3D/mushroom_upright")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
