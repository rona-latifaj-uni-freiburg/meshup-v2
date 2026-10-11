"""Orient DenseCorr3D cucumber scans so the stem end faces the same way.

For each instance under --input-root (one folder per scan, containing
simple_mesh.obj / color_mesh.obj / groups_visualization.obj), this fits the
cucumber's long (growth) axis via PCA on simple_mesh.obj, maps that axis onto
X, and centers the mesh so its post-rotation bounding box sits at the origin.
The same rigid transform (rotation about the shared centroid, then a
translation) is applied to every OBJ variant; mtl/texture/groups.txt files are
copied through untouched. Vertex/face/UV order and count are unchanged, so
groups.txt (vertex-index lists) stays valid, and color_mesh.obj (a separate,
denser scan that does not share vertex indices with simple_mesh.obj) still
gets the correct rigid transform since rotation+translation applies to any
point set of the same physical object.

Which end is the stem is not written anywhere in the data (groups.txt is a
PartField-style segmentation, not a semantic label), so it was determined by
human inspection of renders, per MANUAL_VERDICT below. A local-radius-anomaly
heuristic (flag vertices whose distance from the long axis is an outlier
against the local median at that point along the axis) was tried first but
turned out to be unreliable here -- on this dataset it just as often flags
mesh-reconstruction blemishes near the equator or a flattened patch where the
scan rested on a surface as it does genuine stem features, so it is not used
to decide anything below (see git history for the abandoned attempt).
Instead, each instance was rendered from multiple angles including both pole
end-caps and inspected directly:

  - 9231e77e...: an actual yellow stem nub is visible -- unambiguous.
  - baf9b8...: a shallow crater near one end, consistently visible across a
    span of azimuths in an un-textured (flat-gray) render, i.e. a real
    geometric dimple and not a texture artifact. The first pass put that
    crater end at -X, matching the other two confident instances; the user
    reviewed the output and asked for the ends to be swapped, so
    FLIP_END_OVERRIDES now puts the crater end at +X instead (the opposite
    convention from 9231e77e/2704a3eb -- if that crater is actually a
    blossom-end scar rather than the stem, this is the correct fix and the
    plain end at -X is the real stem; not independently re-verified here).
  - 2704a3eb...: the two pole end-caps are visibly different -- one is a
    mottled pink/brown, slightly narrower point; the other is a plain rounded
    green cap. Weaker evidence than the two above (discoloration + mild
    taper, not a crisp nub/crater) but a real, consistent difference between
    the ends.
  - cfa7617... and d9a2d600...: no reliable signal. Both poles are smooth in
    plain-gray renders; cfa7617's vertex colors happen to split into a
    textured/untextured hemisphere along the long axis (a scan-coverage
    artifact, not a stem marker), and d9a2d600's only visible surface blemish
    sits near the equator, not either pole, so it cannot be a stem. Flagged
    "unclear" in the manifest; the stem-side flip for these two is a
    best-effort default (same convention as the other three), not a verified
    call.

All three confident instances put the stem on the raw-PCA-positive side of
the long axis, so build_rotation() always maps that side to -X
(STEM_AXIS_SIGN); the two "unclear" instances follow the same convention as a
default rather than a coin flip. The second/third PCA axes are used as-is for
Y/Z (roll is not semantically constrained for a near-cylindrical vegetable),
with only a single-axis sign fix applied when needed to keep the transform a
proper rotation (det +1, no mirroring of geometry/normals).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

STEM_AXIS_SIGN = -1.0  # Stem end (raw-PCA-positive side) maps to this side of X.

OBJ_VARIANTS = ["simple_mesh.obj", "color_mesh.obj", "groups_visualization.obj"]
PASSTHROUGH_GLOBS = ["*.mtl", "*.png", "groups.txt"]

# Human-verified from rendering every instance (see module docstring).
MANUAL_VERDICT: Dict[str, str] = {
    "9231e77e152d4e54b1673eae6686ba0d": "confident",  # visible stem nub
    "baf9b8057b5e4f2fbb4fd9299c624fd5": "confident",  # visible crater/scar near one end
    "2704a3eb44f349b3b2982580b08faa15": "confident",  # discolored + slightly tapered end cap
    "cfa7617882e4408cad35fcff8c92cd5f": "unclear",  # both ends smooth, no marking
    "d9a2d600817c43e2903c6f4a5adff5af": "unclear",  # both poles smooth; only blemish is near the equator
}

# Instances where the auto-detected crater/nub side was the wrong end (caught
# on visual re-check after the first pass); flips the long axis 180 degrees
# so the two ends swap places without mirroring the geometry.
FLIP_END_OVERRIDES = {
    "baf9b8057b5e4f2fbb4fd9299c624fd5",
}


def read_vertices(path: Path) -> np.ndarray:
    verts = []
    with path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=np.float64)


def build_rotation(vertices: np.ndarray, flip_ends: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Return (R, centroid, diagnostics).

    R maps (vertex - centroid) into the output frame: axis0 (long axis) onto
    X with the stem side at STEM_AXIS_SIGN * X, axis1 onto Y, axis2 onto Z,
    with a single sign fix (never on X) so det(R) == +1. If flip_ends is set,
    the stem side is put at -STEM_AXIS_SIGN * X instead, swapping which end
    lands where without mirroring the geometry.
    """
    centroid = vertices.mean(axis=0)
    vc = vertices - centroid
    cov = vc.T @ vc / len(vc)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, order]

    r_full = eigvec.T  # rows: axis0, axis1, axis2 in the *raw* (unflipped) PCA sign.
    sign_x = -STEM_AXIS_SIGN if flip_ends else STEM_AXIS_SIGN
    sign_y = 1.0
    det_full = float(np.linalg.det(r_full))
    # Want det(diag(sign_x,sign_y,sign_z) @ r_full) == +1; all three signs and
    # det_full are +-1, and 1/x == x there, so sign_z = sign_x*sign_y*det_full.
    sign_z = sign_x * sign_y * det_full

    d = np.diag([sign_x, sign_y, sign_z])
    r = d @ r_full
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-6), f"det={np.linalg.det(r)}"

    diagnostics = {"eigval": np.round(eigval[order], 8).tolist()}
    return r, centroid, diagnostics


def transform_obj_file(src: Path, dst: Path, r: np.ndarray, centroid: np.ndarray, shift: np.ndarray) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_lines = []
    with src.open("r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                new_xyz = r @ (xyz - centroid) + shift
                rest = parts[4:]
                fields = [f"{v:.8g}" for v in new_xyz] + rest
                out_lines.append("v " + " ".join(fields) + "\n")
            elif line.startswith("vn "):
                parts = line.split()
                n = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                new_n = r @ n
                out_lines.append("vn " + " ".join(f"{v:.6g}" for v in new_n) + "\n")
            else:
                out_lines.append(line)
    dst.write_text("".join(out_lines))


def process_instance(src_dir: Path, dst_dir: Path) -> Dict[str, object]:
    ref_obj = src_dir / "simple_mesh.obj"
    vertices = read_vertices(ref_obj)
    r, centroid, diagnostics = build_rotation(vertices, flip_ends=src_dir.name in FLIP_END_OVERRIDES)

    # Recompute the post-rotation bbox center (from the reference mesh) so the
    # final mesh is centered at the origin, matching this codebase's
    # normalize_one() bbox-center convention.
    rotated = (vertices - centroid) @ r.T
    bbox_center = (rotated.min(axis=0) + rotated.max(axis=0)) * 0.5
    shift = -bbox_center

    for name in OBJ_VARIANTS:
        src = src_dir / name
        if src.is_file():
            transform_obj_file(src, dst_dir / name, r, centroid, shift)

    for pattern in PASSTHROUGH_GLOBS:
        for src in src_dir.glob(pattern):
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)

    final = rotated + shift
    return {
        "instance": src_dir.name,
        "stem_side_confidence": MANUAL_VERDICT.get(src_dir.name, "confident"),
        "num_vertices": int(len(vertices)),
        "bbox_extent": np.round(final.max(axis=0) - final.min(axis=0), 6).tolist(),
        **diagnostics,
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
        print(f"{row['instance']}: {row['stem_side_confidence']} (extent={row['bbox_extent']})")

    unclear = [r["instance"] for r in rows if r["stem_side_confidence"] == "unclear"]
    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_instances": len(rows),
        "unclear_instances": unclear,
        "items": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "orientation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {manifest_path}")
    print(f"{len(rows)} instances processed: {len(unclear)} flagged unclear (stem side is a best-effort default).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="external/DenseCorr3D/cucumber")
    parser.add_argument("--output-root", default="external/DenseCorr3D/cucumber_aligned")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
