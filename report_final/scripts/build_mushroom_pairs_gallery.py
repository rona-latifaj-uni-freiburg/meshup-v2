"""Browsing gallery of every directed mushroom pair (30 = 6 meshes x 5 targets each)
at PF02, which gives exactly 2 PartField buckets (cap + stem). Not used in the report
itself -- a selection aid so the report author can pick one pair by eye.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, render_panel, save_rgb, draw_text, INK  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import face_colors, load_colored_ply  # noqa: E402

OUT = ROOT / "report_final" / "data"
OUT.mkdir(parents=True, exist_ok=True)
WHITE = (255, 255, 255)

# bucket 0 = cap, bucket 1 = stem (or vice versa) -- just need two clearly distinct colors
BUCKET_PALETTE = np.array(
    [[0.122, 0.467, 0.706], [1.000, 0.498, 0.055]],
    dtype=np.float32,
)

RUN_DIR = ROOT / "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_mushroom_top4"

# (pair_prefix, AUC, Err), ranked by AUC, from fair_comparison_meshup_vs_densematcher_matched_rows.csv
PAIRS = [
    ("mushS2T4", 0.955, 1.110), ("mushS4T2", 0.943, 1.417), ("mushS5T4", 0.926, 1.833),
    ("mushS5T2", 0.926, 1.845), ("mushS2T5", 0.922, 1.926), ("mushS1T4", 0.893, 2.680),
    ("mushS4T5", 0.890, 2.730), ("mushS4T1", 0.859, 4.113), ("mushS1T2", 0.830, 4.256),
    ("mushS1T3", 0.830, 4.371), ("mushS0T4", 0.825, 4.351), ("mushS0T2", 0.823, 4.417),
    ("mushS5T0", 0.819, 4.693), ("mushS2T0", 0.818, 4.605), ("mushS4T3", 0.811, 4.779),
    ("mushS4T0", 0.811, 4.764), ("mushS3T4", 0.810, 4.934), ("mushS2T1", 0.807, 5.428),
    ("mushS0T5", 0.806, 4.841), ("mushS2T3", 0.772, 5.774), ("mushS1T5", 0.748, 6.288),
    ("mushS1T0", 0.738, 6.613), ("mushS3T2", 0.736, 6.919), ("mushS3T1", 0.723, 9.595),
    ("mushS5T3", 0.718, 7.116), ("mushS3T5", 0.682, 8.238), ("mushS3T0", 0.675, 8.424),
    ("mushS5T1", 0.670, 9.543), ("mushS0T1", 0.620, 10.38), ("mushS0T3", 0.556, 11.66),
]

AZ, EL, ZOOM = 30.0, 10.0, 0.85


def load_source_final(prefix: str) -> tuple[Mesh, Mesh]:
    run = next(RUN_DIR.glob(f"{prefix}_mushroom_pf02_*"))
    labels = np.load(run / "partfield_labels/bucket_labels.npz")["source_labels"]
    colors = BUCKET_PALETTE[labels % len(BUCKET_PALETTE)]
    sv, sf, _ = load_colored_ply(run / "colored_meshes/mesh_initial.ply")
    fv, ff, _ = load_colored_ply(run / "colored_meshes/mesh_final_position.ply")
    src = Mesh(sv, sf, colors, face_colors(sf, colors))
    fin = Mesh(fv, ff, colors, face_colors(ff, colors))
    return src, fin


def main() -> None:
    supersample = 2
    cols = 6
    rows = (len(PAIRS) + cols - 1) // cols
    cell_w, cell_h = 320, 300
    width, height = cell_w * cols, cell_h * rows
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), WHITE, dtype=np.uint8)

    for i, (prefix, auc, err) in enumerate(PAIRS):
        r, c = divmod(i, cols)
        src, fin = load_source_final(prefix)

        cx0 = c * cell_w * supersample
        cy0 = r * cell_h * supersample
        title_h = int(0.14 * cell_h) * supersample
        panel_h = cell_h * supersample - title_h

        label = f"{i + 1}. {prefix}  AUC {auc:.3f}"
        draw_text(image, label, (cx0 + 6 * supersample, cy0 + int(title_h * 0.6)),
                  scale=0.32 * supersample, color=INK, thickness=max(1, supersample))

        gap = int(0.07 * cell_w) * supersample
        panel_w = (cell_w * supersample - gap) // 2
        src_rect = (cx0, cy0 + title_h, panel_w, panel_h)
        fin_rect = (cx0 + panel_w + gap, cy0 + title_h, panel_w, panel_h)
        render_panel(image, src_rect, src, "", azimuth=AZ, elevation=EL, zoom=ZOOM)
        render_panel(image, fin_rect, fin, "", azimuth=AZ, elevation=EL, zoom=ZOOM)
        for px0, py0, pw, ph in (src_rect, fin_rect):
            cv2.rectangle(image, (px0, py0), (px0 + pw, py0 + ph), WHITE, 2)

        arrow_y = cy0 + title_h + panel_h // 2
        ax0 = cx0 + panel_w + int(0.15 * gap)
        ax1 = cx0 + panel_w + gap - int(0.15 * gap)
        cv2.arrowedLine(image, (ax0, arrow_y), (ax1, arrow_y), INK,
                         thickness=max(1, int(1.4 * supersample)), tipLength=0.4, line_type=cv2.LINE_AA)

    save_rgb(OUT / "mushroom_pairs_gallery.png", image, supersample, (width, height))
    print("wrote mushroom_pairs_gallery.png")


if __name__ == "__main__":
    main()
