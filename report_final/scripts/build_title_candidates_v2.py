"""Alternative title-row candidates: same rendering machinery as build_title_figure.py,
different pair combinations, so they can be compared side by side before picking one
for the real title page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, load_mesh, save_rgb  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import (  # noqa: E402
    face_colors, load_colored_ply,
)
from build_title_figure import (  # noqa: E402
    BUCKET_PALETTE, WHITE, render_pair_cell_shared_scale,
)

OUT = ROOT / "report_final" / "figures"


def load_source_final_from_labels(run: Path) -> tuple[Mesh, Mesh]:
    labels = np.load(run / "partfield_labels/bucket_labels.npz")
    source_labels = labels["source_labels"]
    colors = BUCKET_PALETTE[source_labels % len(BUCKET_PALETTE)]
    src_v, src_f, _ = load_colored_ply(run / "colored_meshes/mesh_initial.ply")
    source_mesh = Mesh(vertices=src_v, faces=src_f, colors=colors, face_rgb=face_colors(src_f, colors))
    fin_v, fin_f, _ = load_colored_ply(run / "colored_meshes/mesh_final_position.ply")
    final_mesh = Mesh(vertices=fin_v, faces=fin_f, colors=colors, face_rgb=face_colors(fin_f, colors))
    return source_mesh, final_mesh


HORSE_GIRAFFE_DIR = ROOT / (
    "jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_selected_best_asym035_jump500_4000/"
    "horse2_to_giraffe4_sam3d_selected_pf08_4000_hard_partfield_chamfer_only_horse2_to_giraffe4_pf08_"
    "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5816558/partfield_colored_meshes"
)

NEW_PAIRS = {
    "panda_bear": (
        "Panda to Bear", "labels",
        ROOT / (
            "jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_selected_best_asym035_jump500_4000/"
            "panda3_to_bear_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_panda3_to_bear_pf07_pfonly_"
            "best_asym035_jneighbor1000_jump500_4000ep_4000ep_5817242"
        ),
        20.0, 10.0, 0.88,
    ),
    "fox_cat": (
        "Fox to Cat", "labels",
        ROOT / (
            "jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_selected_best_asym035_jump500_4000/"
            "fox_to_cat2_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_fox_to_cat2_pf07_pfonly_"
            "best_asym035_jneighbor1000_jump500_4000ep_4000ep_5818235"
        ),
        20.0, 10.0, 0.88,
    ),
    "red_truck": (
        "Red Truck", "labels",
        ROOT / (
            "jobs_with_target_guidance/outputs/requested_car_pairs_clean_partfield_4000/"
            "red_truck1_to_red_truck3_car_clean_selected_pf09_4000_hard_partfield_chamfer_only_"
            "red_truck1_to_red_truck3_pf09_pfonly_car_clean_jneighbor1000_jump500_4000ep_5993980"
        ),
        30.0, 14.0, 0.82,
    ),
    "horse_giraffe": (
        "Horse to Giraffe", "prebaked", HORSE_GIRAFFE_DIR, -68.0, 12.0, 0.92,
    ),
}


def load_pair(key: str) -> tuple[str, Mesh, Mesh, float, float, float]:
    title, mode, folder, az, el, zoom = NEW_PAIRS[key]
    if mode == "prebaked":
        src = load_mesh(folder / "input_partfield_08.ply")
        fin = load_mesh(folder / "final_partfield_08.ply")
    else:
        src, fin = load_source_final_from_labels(folder)
    return title, src, fin, az, el, zoom


def build_row(name: str, keys: list[str]) -> None:
    supersample = 3
    width, height = 1800, 460
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), WHITE, dtype=np.uint8)
    cell_w = w // 3
    for i, key in enumerate(keys):
        title, src, fin, az, el, zoom = load_pair(key)
        rect = (i * cell_w, 0, cell_w, h)
        render_pair_cell_shared_scale(image, rect, title, src, fin, az, el, zoom, supersample)
    save_rgb(OUT / name, image, supersample, (width, height))
    print(f"wrote {name}")


def main() -> None:
    build_row("title_row_candidate2.png", ["panda_bear", "horse_giraffe", "red_truck"])
    build_row("title_row_candidate3.png", ["fox_cat", "horse_giraffe", "panda_bear"])


if __name__ == "__main__":
    main()
