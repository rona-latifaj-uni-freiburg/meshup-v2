"""Rebuild the banana bucket figure with two genuinely banana-shaped meshes
(m001 was some other object entirely -- not a banana) and with the mesh
BODIES actually rotated into a shared canonical frame (long axis horizontal,
stem on the same side), not just rendered from the same camera azimuth.
Same camera angle alone does not align two independently-oriented raw
meshes; the meshes themselves need to be rotated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import load_colored_ply, face_colors  # noqa: E402

SEG = ROOT / "jobs_with_target_guidance/partfield_segments/densecorr3d_categories/densecorr3d_nonanimals_test_20260806/banana/buckets_06/colored"
SOURCE = SEG / "banana_m000_693f7748_partfield_06_parts.ply"
TARGET = SEG / "banana_m004_1e23afca_partfield_06_parts.ply"


def align_long_axis(vertices: np.ndarray, colors: np.ndarray, flip: bool) -> np.ndarray:
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T  # rows: long, mid, short axis, right-handed by construction of eigh + ordering
    # Ensure a proper rotation (determinant +1), not a reflection.
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    aligned = centered @ axes.T  # now: x=long axis, y=mid axis, z=short axis

    # Put the stem end on the right (positive x) consistently. Rather than
    # guessing a fixed "stem red" RGB threshold (both a brown tip-cap and an
    # orange edge band are also red-dominant, so that misfires), directly
    # compare the two END caps of the long axis: whichever end is redder
    # (not browner/dimmer) is the stem, regardless of the exact palette.
    x = aligned[:, 0]
    lo_end = x <= np.quantile(x, 0.06)
    hi_end = x >= np.quantile(x, 0.94)
    redness = colors[:, 0] - 0.5 * (colors[:, 1] + colors[:, 2])
    lo_redness = redness[lo_end].mean()
    hi_redness = redness[hi_end].mean()
    print(f"  lo-end redness={lo_redness:.3f}  hi-end redness={hi_redness:.3f}")
    stem_is_hi = hi_redness > lo_redness
    if not stem_is_hi:
        aligned[:, 0] *= -1
        aligned[:, 2] *= -1  # keep it a rotation, not a mirror

    if flip:
        aligned[:, 1] *= -1
        aligned[:, 2] *= -1

    return aligned


def load_aligned(path: Path, flip: bool) -> Mesh:
    v, f, c = load_colored_ply(path)
    aligned = align_long_axis(v, c, flip)
    fc = face_colors(f, c)
    return Mesh(vertices=aligned, faces=f, colors=c, face_rgb=fc)


def main() -> None:
    mesh_source = load_aligned(SOURCE, flip=False)
    mesh_target = load_aligned(TARGET, flip=False)

    supersample = 3
    width, height = 1500, 620
    panel_w = width // 2
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)
    draw_text(image, "Banana buckets: bodies rotated into a shared frame (long axis horizontal, stem on the right)",
              (24 * supersample, 40 * supersample), scale=0.5 * supersample, color=INK, thickness=2 * supersample)

    top = 80 * supersample
    for i, (label, mesh) in enumerate((("Source PF6", mesh_source), ("Target PF6", mesh_target))):
        x0 = i * panel_w * supersample
        draw_text(image, label, (x0 + 18 * supersample, 70 * supersample), scale=0.46 * supersample, color=INK, thickness=max(1, supersample))
        render_panel(image, (x0, top, panel_w * supersample, h - top), mesh, "", azimuth=0.0, elevation=0.0, zoom=0.82)

    save_rgb(ROOT / "report_final" / "figures" / "banana_pf06.png", image, supersample, (width, height))
    print("done")


if __name__ == "__main__":
    main()
