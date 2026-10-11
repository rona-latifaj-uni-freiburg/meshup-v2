"""Browsing gallery of 10 candidate Source->Final pairs for the title-page third slot
(currently banana, which the report author rejected). Covers shampoo, knife, two TOSCA
animal poses (wolf, dog -- distinct from the horse->giraffe pair already in the row),
and several fruit/vegetable categories. Not used in the report itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, load_mesh, render_panel, save_rgb, draw_text, INK  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import face_colors, load_colored_ply  # noqa: E402

OUT = ROOT / "report_final" / "data"
OUT.mkdir(parents=True, exist_ok=True)
WHITE = (255, 255, 255)

BUCKET_PALETTE = np.array(
    [
        [0.122, 0.467, 0.706], [1.000, 0.498, 0.055], [0.173, 0.627, 0.173],
        [0.839, 0.153, 0.157], [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
        [0.890, 0.467, 0.761], [0.498, 0.498, 0.498], [0.737, 0.741, 0.133],
        [0.090, 0.745, 0.812], [0.700, 0.230, 0.500], [0.300, 0.680, 0.420],
    ],
    dtype=np.float32,
)

OUTPUTS = ROOT / "jobs_with_target_guidance/cross_animal_spike_runs/outputs"

# (label, run_dir, metric, az, el, zoom, mode) -- mode "buckets" for PartField-guided
# runs (recolor from bucket_labels.npz), "position" for TOSCA dense-correspondence
# runs (no PartField buckets involved at all, see Section 13; colored by per-axis
# normalized XYZ instead, matching the project's Limitations-figure convention).
CANDIDATES = [
    ("1. Shampoo",
     OUTPUTS / "densecorr3d_shampoo_pf0104/shpS3T0_shampoo_pf03_4000_hard_partfield_chamfer_only_shpS3T0_pf03_"
     "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6224622",
     "AUC 0.991 / Err 0.22", 30.0, 10.0, 0.85, "buckets"),
    ("2. Knife",
     OUTPUTS / "densecorr3d_knifeonly_pf0205/knfS1T2_knifeonly_pf04_4000_hard_partfield_chamfer_only_knfS1T2_pf04_"
     "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6223901",
     "AUC 0.990 / Err 0.26", 20.0, 10.0, 0.85, "buckets"),
    ("3. TOSCA Wolf",
     OUTPUTS / "tosca_correspondence_guided/wolf0_to_wolf2_tosca_vcorr_2000_hard_partfield_chamfer_only_wolf0_to_"
     "wolf2_tosca_identity_vcorr_w20000_jacns100_jump50_2000ep_2000ep_6175876",
     "AUC 1.000 / Err 0.00", 30.0, 10.0, 0.85, "position"),
    ("4. TOSCA Dog",
     OUTPUTS / "tosca_correspondence_guided/dog2_to_dog5_tosca_vcorr_2000_hard_partfield_chamfer_only_dog2_to_"
     "dog5_tosca_identity_vcorr_w20000_jacns100_jump50_2000ep_2000ep_6174464",
     "AUC 0.997 / Err 0.07", 30.0, 10.0, 0.85, "position"),
    ("5. Apple",
     OUTPUTS / "densecorr3d_apple_top4/applS4T5_apple_pf10_4000_hard_partfield_chamfer_only_applS4T5_pf10_pfonly_"
     "best_asym035_jneighbor1000_jump500_4000ep_4000ep_6246223",
     "AUC 0.933 / Err 1.67", 30.0, 10.0, 0.85, "buckets"),
    ("6. Pear",
     OUTPUTS / "densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_4000_a100/"
     "pear__pear_m003_a6193c4a_to_pear_m000_f1a0e1c3_densecorr3d_nonanimals_test_partfield_guided_fair_a100_pf04_"
     "4000_hard_partfield_chamfer_only_pear__pear_m003_a6193c4a_to_pear_m000_f1a0e1c3_pf04_pfonly_best_asym035_"
     "jneighbor1000_jump500_4000ep_4000ep_6183090",
     "AUC 0.942 / Err 1.46", 30.0, 10.0, 0.85, "buckets"),
    ("7. Pumpkin",
     OUTPUTS / "densecorr3d_pumpkin_top4/pumpS0T1_pumpkin_pf05_4000_hard_partfield_chamfer_only_pumpS0T1_pf05_"
     "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6244498",
     "AUC 0.935 / Err 1.61", 30.0, 10.0, 0.85, "buckets"),
    ("8. Onion",
     OUTPUTS / "densecorr3d_onion_top4/onioS2T1_onion_pf06_4000_hard_partfield_chamfer_only_onioS2T1_pf06_pfonly_"
     "best_asym035_jneighbor1000_jump500_4000ep_4000ep_6253436",
     "AUC 0.957 / Err 1.05", 30.0, 10.0, 0.85, "buckets"),
    ("9. Tomato",
     OUTPUTS / "densecorr3d_tomato_top4/tomaS4T2_tomato_pf02_4000_hard_partfield_chamfer_only_tomaS4T2_pf02_"
     "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6247108",
     "AUC 0.946 / Err 1.34", 30.0, 10.0, 0.85, "buckets"),
    ("10. Mushroom",
     OUTPUTS / "densecorr3d_mushroom_top4/mushS2T4_mushroom_pf02_4000_hard_partfield_chamfer_only_mushS2T4_pf02_"
     "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6253540",
     "AUC 0.955 / Err 1.11", 30.0, 10.0, 0.85, "buckets"),
]


def pos_color(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(0, keepdims=True), v.max(0, keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1
    return ((v - lo) / rng).astype(np.float32)


def load_source_final(run: Path, mode: str) -> tuple[Mesh, Mesh]:
    sv, sf, _ = load_colored_ply(run / "colored_meshes/mesh_initial.ply")
    fv, ff, _ = load_colored_ply(run / "colored_meshes/mesh_final_position.ply")
    if mode == "buckets":
        labels = np.load(run / "partfield_labels/bucket_labels.npz")["source_labels"]
        colors = BUCKET_PALETTE[labels % len(BUCKET_PALETTE)]
    else:
        colors = pos_color(sv)
    src = Mesh(sv, sf, colors, face_colors(sf, colors))
    fin = Mesh(fv, ff, colors, face_colors(ff, colors))
    return src, fin


def main() -> None:
    supersample = 2
    cols = 5
    rows = (len(CANDIDATES) + cols - 1) // cols
    cell_w, cell_h = 420, 380
    width, height = cell_w * cols, cell_h * rows
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), WHITE, dtype=np.uint8)

    for i, (label, run_dir, metric, az, el, zoom, mode) in enumerate(CANDIDATES):
        r, c = divmod(i, cols)
        src, fin = load_source_final(run_dir, mode)

        cx0 = c * cell_w * supersample
        cy0 = r * cell_h * supersample
        title_h = int(0.11 * cell_h) * supersample
        panel_h = cell_h * supersample - title_h

        draw_text(image, f"{label}  ({metric})", (cx0 + 8 * supersample, cy0 + int(title_h * 0.7)),
                  scale=0.38 * supersample, color=INK, thickness=max(1, int(1.2 * supersample)))

        gap = int(0.06 * cell_w) * supersample
        panel_w = (cell_w * supersample - gap) // 2
        src_rect = (cx0, cy0 + title_h, panel_w, panel_h)
        fin_rect = (cx0 + panel_w + gap, cy0 + title_h, panel_w, panel_h)
        render_panel(image, src_rect, src, "", azimuth=az, elevation=el, zoom=zoom)
        render_panel(image, fin_rect, fin, "", azimuth=az, elevation=el, zoom=zoom)
        for px0, py0, pw, ph in (src_rect, fin_rect):
            cv2.rectangle(image, (px0, py0), (px0 + pw, py0 + ph), WHITE, 2)

        arrow_y = cy0 + title_h + panel_h // 2
        ax0 = cx0 + panel_w + int(0.15 * gap)
        ax1 = cx0 + panel_w + gap - int(0.15 * gap)
        cv2.arrowedLine(image, (ax0, arrow_y), (ax1, arrow_y), INK,
                         thickness=max(1, int(1.6 * supersample)), tipLength=0.4, line_type=cv2.LINE_AA)

    save_rgb(OUT / "title_candidate_gallery.png", image, supersample, (width, height))
    print("wrote title_candidate_gallery.png")


if __name__ == "__main__":
    main()
