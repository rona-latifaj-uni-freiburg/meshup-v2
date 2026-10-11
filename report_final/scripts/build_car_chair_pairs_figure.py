"""User-picked pairs: car m000->m002 and chair m004->m000, rendered at an angled
(3/4) view so depth is visible -- a straight side-on view hides the chair's
backrest structure and the car's front/roof depth. Source / Final (deformed
source), white background, arrow between panels, labels underneath -- same style
as the title-page grid figure.
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

OUT = ROOT / "report_final" / "figures"
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

PAIRS = {
    "car": (
        "Car m000 -> m002",
        ROOT / (
            "jobs_with_target_guidance/cross_animal_spike_runs/outputs/"
            "densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_4000_a100/"
            "cars__cars_m000_35ebd_to_to_cars_m002_f756d_to_densecorr3d_nonanimals_test_partfield_guided_fair_"
            "a100_pf04_4000_hard_partfield_chamfer_only_cars__cars_m000_35ebd_to_to_cars_m002_f756d_to_pf04_"
            "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6179716"
        ),
        60.0, 18.0, 0.78,
    ),
    "chair": (
        "Chair m004 -> m000",
        ROOT / (
            "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_chairs_fix_pf0408/"
            "chrS4T0_chairsfix_pf07_4000_hard_partfield_chamfer_only_chrS4T0_pf07_pfonly_best_asym035_"
            "jneighbor1000_jump500_4000ep_4000ep_6222486"
        ),
        30.0, 18.0, 0.78,
    ),
}


def load_source_final(run: Path) -> tuple[Mesh, Mesh]:
    labels = np.load(run / "partfield_labels/bucket_labels.npz")["source_labels"]
    colors = BUCKET_PALETTE[labels % len(BUCKET_PALETTE)]
    sv, sf, _ = load_colored_ply(run / "colored_meshes/mesh_initial.ply")
    fv, ff, _ = load_colored_ply(run / "colored_meshes/mesh_final_position.ply")
    src = Mesh(sv, sf, colors, face_colors(sf, colors))
    fin = Mesh(fv, ff, colors, face_colors(ff, colors))
    return src, fin


def render_pair_cell(image: np.ndarray, rect: tuple[int, int, int, int], title: str,
                      src: Mesh, fin: Mesh, az: float, el: float, zoom: float,
                      supersample: int, title_scale: float) -> None:
    x0, y0, w, h = rect
    title_h = int(0.10 * h)
    label_h = int(0.08 * h)
    panel_h = h - title_h - label_h

    draw_text(image, title, (x0 + int(0.01 * w), y0 + int(title_h * 0.72)),
              scale=title_scale * supersample, color=INK, thickness=max(1, int(1.6 * supersample)))

    panel_y = y0 + title_h
    gap = int(0.07 * w)
    panel_w = (w - gap) // 2
    src_rect = (x0, panel_y, panel_w, panel_h)
    fin_rect = (x0 + panel_w + gap, panel_y, panel_w, panel_h)
    render_panel(image, src_rect, src, "", azimuth=az, elevation=el, zoom=zoom)
    render_panel(image, fin_rect, fin, "", azimuth=az, elevation=el, zoom=zoom)

    arrow_y = panel_y + panel_h // 2
    arrow_x0 = x0 + panel_w + int(0.012 * w)
    arrow_x1 = x0 + panel_w + gap - int(0.012 * w)
    cv2.arrowedLine(
        image, (arrow_x0, arrow_y), (arrow_x1, arrow_y), INK,
        thickness=max(1, int(1.8 * supersample)), tipLength=0.35, line_type=cv2.LINE_AA,
    )

    label_y = panel_y + panel_h + int(label_h * 0.68)
    for label, px0, pw in (("Source", x0, panel_w), ("Final (deformed source)", x0 + panel_w + gap, panel_w)):
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42 * supersample, max(1, supersample))
        text_x = px0 + max(0, (pw - text_w) // 2)
        draw_text(image, label, (text_x, label_y), scale=0.42 * supersample, color=MUTED,
                  thickness=max(1, supersample))


def main() -> None:
    supersample = 3
    width, height = 1500, 950
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), WHITE, dtype=np.uint8)

    row_h = h // 2
    for row, key in enumerate(["car", "chair"]):
        title, run, az, el, zoom = PAIRS[key]
        src, fin = load_source_final(run)
        rect = (0, row * row_h, w, row_h)
        render_pair_cell(image, rect, title, src, fin, az, el, zoom, supersample, title_scale=0.55)

    save_rgb(OUT / "user_picked_car_chair_pairs.png", image, supersample, (width, height))
    print("wrote user_picked_car_chair_pairs.png")


if __name__ == "__main__":
    main()
