"""Replacement for the single static banana-bucket figure: a grid of REAL banana-to-
banana deformation runs (PartField PF06), each shown as Source / Final / Target, so
both good and bad correspondence cases are visible together.

Only three banana pairs have a completed PF06 deformation run in this project:
  S0T1 = m000 -> m001  (m001 is not actually a banana-shaped mesh -- see note below)
  S2T4 = m002 -> m004
  S5T3 = m005 -> m003
All three are used; captions are written to match what each one actually shows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, load_mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK, MUTED  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import face_colors, load_colored_ply  # noqa: E402

OUT = ROOT / "report_final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# These runs' own colored_meshes/*.ply use continuous per-vertex position coloring, not
# discrete bucket colors (verified), so source/final are recolored here directly from
# the run's own saved bucket labels with a clean discrete palette.
BUCKET_PALETTE = np.array(
    [
        [0.122, 0.467, 0.706], [1.000, 0.498, 0.055], [0.173, 0.627, 0.173],
        [0.839, 0.153, 0.157], [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
        [0.890, 0.467, 0.761], [0.498, 0.498, 0.498], [0.737, 0.741, 0.133],
        [0.090, 0.745, 0.812],
    ],
    dtype=np.float32,
)


def load_source_final(run: Path) -> tuple[Mesh, Mesh]:
    labels = np.load(run / "partfield_labels/bucket_labels.npz")
    source_labels = labels["source_labels"]

    src_v, src_f, _ = load_colored_ply(run / "colored_meshes/mesh_initial.ply")
    src_colors = BUCKET_PALETTE[source_labels % len(BUCKET_PALETTE)]
    source_mesh = Mesh(vertices=src_v, faces=src_f, colors=src_colors, face_rgb=face_colors(src_f, src_colors))

    fin_v, fin_f, _ = load_colored_ply(run / "colored_meshes/mesh_final_position.ply")
    fin_colors = BUCKET_PALETTE[source_labels % len(BUCKET_PALETTE)]
    final_mesh = Mesh(vertices=fin_v, faces=fin_f, colors=fin_colors, face_rgb=face_colors(fin_f, fin_colors))
    return source_mesh, final_mesh

DIAG = ROOT / "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_banana_diagnostic_pf0110"
TARGET_SEG = ROOT / (
    "jobs_with_target_guidance/partfield_segments/densecorr3d_categories/"
    "densecorr3d_banana_20260810/banana/buckets_06/colored"
)

PAIRS = [
    (
        "banaS0T1_bananadiag_pf06_4000_hard_partfield_chamfer_only_banaS0T1_pf06_pfonly_best_asym035_"
        "jneighbor1000_jump500_4000ep_4000ep_6239424",
        "banana_m001_b9214be9_partfield_06_parts.ply",
        "m000 -> m001",
        60.0,
        False,  # left unflipped: horizontal and vertical pixel-flips were both tried and both
                # broke stem-top/tip-bottom consistency with Final, Target, and the other two
                # rows -- raw is the only version that matches on that axis.
    ),
    (
        "banaS2T4_bananadiag_pf06_4000_hard_partfield_chamfer_only_banaS2T4_pf06_pfonly_best_asym035_"
        "jneighbor1000_jump500_4000ep_4000ep_6239446",
        "banana_m004_1e23afca_partfield_06_parts.ply",
        "m002 -> m004",
        60.0,
        False,
    ),
    (
        "banaS5T3_bananadiag_pf06_4000_hard_partfield_chamfer_only_banaS5T3_pf06_pfonly_best_asym035_"
        "jneighbor1000_jump500_4000ep_4000ep_6239484",
        "banana_m003_5dd87a34_partfield_06_parts.ply",
        "m005 -> m003",
        60.0,
        False,
    ),
]


def main() -> None:
    supersample = 3
    width, height = 1500, 1500
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)

    row_h = h // len(PAIRS)
    panel_gap = 10 * supersample
    label_w = 210 * supersample
    panel_w = (w - label_w - 2 * panel_gap) // 3

    for row, (run_dir, target_file, title, az, flip_source_panel) in enumerate(PAIRS):
        run = DIAG / run_dir
        source, final = load_source_final(run)
        target = load_mesh(TARGET_SEG / target_file)

        y0 = row * row_h
        draw_text(image, title, (10 * supersample, y0 + 34 * supersample),
                  scale=0.6 * supersample, color=INK, thickness=max(1, int(1.6 * supersample)))

        panels = [("Source", source, flip_source_panel), ("Final", final, False), ("Target", target, False)]
        for i, (label, mesh, flip_panel) in enumerate(panels):
            x0 = label_w + i * (panel_w + panel_gap)
            draw_text(image, label, (x0 + 10 * supersample, y0 + 60 * supersample),
                      scale=0.42 * supersample, color=MUTED, thickness=max(1, supersample))
            px0, py0, pw, ph = x0, y0 + 70 * supersample, panel_w, row_h - 80 * supersample
            render_panel(image, (px0, py0, pw, ph), mesh, "", azimuth=az, elevation=6.0, zoom=0.85)
            if flip_panel:
                image[py0:py0 + ph, px0:px0 + pw] = image[py0:py0 + ph, px0:px0 + pw][:, ::-1, :]

    save_rgb(OUT / "banana_grid.png", image, supersample, (width, height))
    print("wrote banana_grid.png")


if __name__ == "__main__":
    main()
