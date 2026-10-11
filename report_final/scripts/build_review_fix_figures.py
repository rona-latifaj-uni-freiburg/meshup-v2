"""Regenerate figures requested during report review fixes.

Outputs:
- numeric-scale vertex-error heatmaps for Figures 10 and 11
- bell-pepper PF03 Source/Final/Target grid replacing the banana PF06 appendix figure
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import (  # noqa: E402
    INK,
    MUTED,
    PAGE_BG,
    Mesh,
    draw_text,
    heatmap_two_view,
    rel,
    render_panel,
    save_rgb,
)
from jobs_with_target_guidance.render_partfield_turntable_video import face_colors, load_colored_ply  # noqa: E402


OUT = ROOT / "report_final" / "figures"
RUN_BASE = ROOT / "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_bell_pepper_azfix_pf0104"
TARGET_SEG = ROOT / (
    "jobs_with_target_guidance/partfield_segments/densecorr3d_categories/"
    "densecorr3d_bell_pepper_azfix_20260810/bell_pepper/buckets_03/colored"
)

BUCKET_PALETTE = np.array(
    [
        [0.122, 0.467, 0.706],
        [1.000, 0.498, 0.055],
        [0.173, 0.627, 0.173],
        [0.839, 0.153, 0.157],
        [0.580, 0.404, 0.741],
        [0.549, 0.337, 0.294],
        [0.890, 0.467, 0.761],
        [0.498, 0.498, 0.498],
        [0.737, 0.741, 0.133],
        [0.090, 0.745, 0.812],
    ],
    dtype=np.float32,
)

PAIRS = [
    (
        "bpaz_m003_m005_bpaz_pf03_4000_hard_partfield_chamfer_only_bpaz_m003_m005_"
        "pf03_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6234200",
        "bell_pepper_m005_752dd9ec_partfield_03_parts.ply",
        "m003 -> m005",
        "AUC 0.866 / Err 3.70",
    ),
    (
        "bpaz_m005_m003_bpaz_pf03_4000_hard_partfield_chamfer_only_bpaz_m005_m003_"
        "pf03_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6234262",
        "bell_pepper_m003_4c978c49_partfield_03_parts.ply",
        "m005 -> m003",
        "AUC 0.874 / Err 3.35",
    ),
    (
        "bpaz_m005_m001_bpaz_pf03_4000_hard_partfield_chamfer_only_bpaz_m005_m001_"
        "pf03_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6234250",
        "bell_pepper_m001_8d56e833_partfield_03_parts.ply",
        "m005 -> m001",
        "AUC 0.848 / Err 4.10",
    ),
]


def mesh_with_labels(vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray) -> Mesh:
    colors = BUCKET_PALETTE[np.asarray(labels, dtype=np.int64) % len(BUCKET_PALETTE)]
    return Mesh(vertices=vertices, faces=faces, colors=colors, face_rgb=face_colors(faces, colors))


def load_bell_pepper_triplet(run_dir: str, target_file: str) -> tuple[Mesh, Mesh, Mesh]:
    run = RUN_BASE / run_dir
    labels = np.load(run / "partfield_labels" / "bucket_labels.npz")
    source_labels = labels["source_labels"]

    src_v, src_f, _ = load_colored_ply(run / "colored_meshes" / "mesh_initial.ply")
    fin_v, fin_f, _ = load_colored_ply(run / "colored_meshes" / "mesh_final_position.ply")
    tgt_v, tgt_f, tgt_colors = load_colored_ply(TARGET_SEG / target_file)

    source = mesh_with_labels(src_v, src_f, source_labels)
    final = mesh_with_labels(fin_v, fin_f, source_labels)
    if "target_labels" in labels.files and len(labels["target_labels"]) == len(tgt_v):
        target = mesh_with_labels(tgt_v, tgt_f, labels["target_labels"])
    else:
        target = Mesh(vertices=tgt_v, faces=tgt_f, colors=tgt_colors, face_rgb=face_colors(tgt_f, tgt_colors))
    return source, final, target


def build_bell_pepper_grid() -> None:
    supersample = 3
    width, height = 1500, 1500
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)

    row_h = h // len(PAIRS)
    panel_gap = 10 * supersample
    label_w = 235 * supersample
    panel_w = (w - label_w - 2 * panel_gap) // 3

    for row, (run_dir, target_file, title, metric) in enumerate(PAIRS):
        source, final, target = load_bell_pepper_triplet(run_dir, target_file)
        y0 = row * row_h
        draw_text(
            image,
            title,
            (10 * supersample, y0 + 34 * supersample),
            scale=0.58 * supersample,
            color=INK,
            thickness=max(1, int(1.6 * supersample)),
        )
        draw_text(
            image,
            metric,
            (10 * supersample, y0 + 70 * supersample),
            scale=0.36 * supersample,
            color=MUTED,
            thickness=max(1, supersample),
        )

        for i, (label, mesh) in enumerate((("Source", source), ("Final", final), ("Target", target))):
            x0 = label_w + i * (panel_w + panel_gap)
            draw_text(
                image,
                label,
                (x0 + 10 * supersample, y0 + 60 * supersample),
                scale=0.42 * supersample,
                color=MUTED,
                thickness=max(1, supersample),
            )
            render_panel(
                image,
                (x0, y0 + 70 * supersample, panel_w, row_h - 80 * supersample),
                mesh,
                "",
                azimuth=0.0,
                elevation=8.0,
                zoom=0.86,
            )

    save_rgb(OUT / "bell_pepper_pf03_grid.png", image, supersample, (width, height))


def build_heatmaps() -> None:
    hard = rel(
        "jobs_with_target_guidance/cross_animal_spike_runs/outputs/"
        "densecorr3d_requested_animals_best_asym035/"
        "giraffe_to_elephant_5k_densecorr3d_requested_animals_hard_partfield_chamfer_only_"
        "densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5546334"
    )
    horse = rel(
        "jobs_with_target_guidance/cross_animal_spike_runs/outputs/"
        "sam3d_selected_best_asym035_jump500_4000/"
        "horse2_to_giraffe4_sam3d_selected_pf08_4000_hard_partfield_chamfer_only_"
        "horse2_to_giraffe4_pf08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5816558"
    )

    heatmap_two_view(
        OUT / "failure_vertex_heatmap.png",
        title="Giraffe to Elephant vertex error: front and side views",
        final_path=hard / "partfield_colored_meshes/final_partfield_08.ply",
        target_path=hard / "partfield_colored_meshes/target_partfield_08.ply",
        front_azimuth=-90.0,
        side_azimuth=0.0,
    )
    heatmap_two_view(
        OUT / "good_vertex_heatmap_horse2_giraffe4.png",
        title="Horse2 to Giraffe4 vertex error: front and side views",
        final_path=horse / "partfield_colored_meshes/final_partfield_08.ply",
        target_path=horse / "partfield_colored_meshes/target_partfield_08.ply",
        front_azimuth=-68.0,
        side_azimuth=22.0,
    )


def main() -> None:
    build_heatmaps()
    build_bell_pepper_grid()


if __name__ == "__main__":
    main()
