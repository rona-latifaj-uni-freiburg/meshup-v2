"""Browsing gallery of chair Source->Final pairs so the report author can pick which
one to use for the title figure. Shows the 8 best-scoring chair pairs (by AUC, all 5
raw chair meshes covered) at a consistent angle. Not used in the report itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, render_panel, save_rgb, draw_text, INK, MUTED  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import face_colors, load_colored_ply  # noqa: E402

OUT = ROOT / "report_final" / "data"
OUT.mkdir(parents=True, exist_ok=True)
WHITE = (255, 255, 255)

BUCKET_PALETTE = np.array(
    [
        [0.122, 0.467, 0.706], [1.000, 0.498, 0.055], [0.173, 0.627, 0.173],
        [0.839, 0.153, 0.157], [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
        [0.890, 0.467, 0.761], [0.498, 0.498, 0.498], [0.737, 0.741, 0.133],
        [0.090, 0.745, 0.812],
    ],
    dtype=np.float32,
)

CHAIRS_DIR = ROOT / "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_chairs_fix_pf0408"

# (label, run subdir, title, AUC/Err)
PAIRS = [
    ("chair_010 -> chair_004",
     "chrS4T0_chairsfix_pf07_4000_hard_partfield_chamfer_only_chrS4T0_pf07_pfonly_best_asym035_"
     "jneighbor1000_jump500_4000ep_4000ep_6222486", "AUC 0.976 / Err 0.59"),
    ("chair_005 -> chair_001",
     "chrS1T2_chairsfix_pf08_4000_hard_partfield_chamfer_only_chrS1T2_pf08_pfonly_best_asym035_"
     "jneighbor1000_jump500_4000ep_4000ep_6222388", "AUC 0.975 / Err 0.63"),
    ("chair_010 -> chair_015",
     "chrS4T3_chairsfix_pf08_4000_hard_partfield_chamfer_only_chrS4T3_pf08_pfonly_best_asym035_"
     "jneighbor1000_jump500_4000ep_4000ep_6222346", "AUC 0.974 / Err 0.65"),
    ("chair_001 -> chair_005",
     "chrS2T1_chairsfix_pf08_4000_hard_partfield_chamfer_only_chrS2T1_pf08_pfonly_best_asym035_"
     "jneighbor1000_jump500_4000ep_4000ep_6222423", "AUC 0.973 / Err 0.68"),
    ("chair_004 -> chair_015",
     "chrS0T3_chairsfix_pf08_4000_hard_partfield_chamfer_only_chrS0T3_pf08_pfonly_best_asym035_"
     "jneighbor1000_jump500_4000ep_4000ep_6222361", "AUC 0.972 / Err 0.71"),
    ("chair_015 -> chair_004",
     "chrS3T0_chairsfix_pf08_4000_hard_partfield_chamfer_only_chrS3T0_pf08_pfonly_best_asym035_"
     "jneighbor1000_jump500_4000ep_4000ep_6222457", "AUC 0.968 / Err 0.83"),
    ("chair_004 -> chair_010",
     "chrS0T4_chairsfix_pf07_4000_hard_partfield_chamfer_only_chrS0T4_pf07_pfonly_best_asym035_"
     "jneighbor1000_jump500_4000ep_4000ep_6222365", "AUC 0.964 / Err 0.92"),
    ("chair_015 -> chair_010",
     "chrS3T4_chairsfix_pf08_4000_hard_partfield_chamfer_only_chrS3T4_pf08_pfonly_best_asym035_"
     "jneighbor1000_jump500_4000ep_4000ep_6222480", "AUC 0.958 / Err 1.04"),
]

AZ, EL, ZOOM = 300.0, 15.0, 0.78


def load_source_final(run: Path) -> tuple[Mesh, Mesh]:
    labels = np.load(run / "partfield_labels/bucket_labels.npz")["source_labels"]
    colors = BUCKET_PALETTE[labels % len(BUCKET_PALETTE)]
    sv, sf, _ = load_colored_ply(run / "colored_meshes/mesh_initial.ply")
    fv, ff, _ = load_colored_ply(run / "colored_meshes/mesh_final_position.ply")
    src = Mesh(sv, sf, colors, face_colors(sf, colors))
    fin = Mesh(fv, ff, colors, face_colors(ff, colors))
    return src, fin


def main() -> None:
    supersample = 2
    cols = 4
    rows = (len(PAIRS) + cols - 1) // cols
    cell_w, cell_h = 480, 420
    width, height = cell_w * cols, cell_h * rows
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), WHITE, dtype=np.uint8)

    for i, (label, run_dir, metric) in enumerate(PAIRS):
        r, c = divmod(i, cols)
        src, fin = load_source_final(CHAIRS_DIR / run_dir)

        cx0 = c * cell_w * supersample
        cy0 = r * cell_h * supersample
        title_h = int(0.11 * cell_h) * supersample
        label_h = int(0.07 * cell_h) * supersample
        panel_h = cell_h * supersample - title_h - label_h

        draw_text(image, f"{i + 1}. {label}  ({metric})", (cx0 + 8 * supersample, cy0 + int(title_h * 0.7)),
                  scale=0.4 * supersample, color=INK, thickness=max(1, int(1.3 * supersample)))

        gap = int(0.06 * cell_w) * supersample
        panel_w = (cell_w * supersample - gap) // 2
        src_rect = (cx0, cy0 + title_h, panel_w, panel_h)
        fin_rect = (cx0 + panel_w + gap, cy0 + title_h, panel_w, panel_h)
        render_panel(image, src_rect, src, "", azimuth=AZ, elevation=EL, zoom=ZOOM)
        render_panel(image, fin_rect, fin, "", azimuth=AZ, elevation=EL, zoom=ZOOM)
        # render_panel always draws a thin border rectangle; erase it since both the
        # panel fill and the canvas are already pure white.
        for px0, py0, pw, ph in (src_rect, fin_rect):
            cv2.rectangle(image, (px0, py0), (px0 + pw, py0 + ph), WHITE, 2)

        arrow_y = cy0 + title_h + panel_h // 2
        ax0 = cx0 + panel_w + int(0.15 * gap)
        ax1 = cx0 + panel_w + gap - int(0.15 * gap)
        cv2.arrowedLine(image, (ax0, arrow_y), (ax1, arrow_y), INK,
                         thickness=max(1, int(1.6 * supersample)), tipLength=0.4, line_type=cv2.LINE_AA)

    save_rgb(OUT / "chair_pairs_gallery.png", image, supersample, (width, height))
    print("wrote chair_pairs_gallery.png")


if __name__ == "__main__":
    main()
