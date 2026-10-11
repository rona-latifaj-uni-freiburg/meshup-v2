"""Qualitative complement to densematcher_category_comparison.png / auc_spread_histogram.png.

Those two figures show aggregate AUC/Err numbers. This script answers the
question a reader asks next: what does that gap actually look like on the
mesh? DenseMatcher only ever predicts a per-vertex point correspondence
(functional map over DiffusionNet features, see arXiv:2412.05268) -- it does
not itself output a deformed mesh. To make it directly comparable to this
pipeline's deformed output, we build the mesh DenseMatcher's own predicted
correspondence implies: keep the source topology (same faces/vertex count as
our "Final" mesh) and move each source vertex to the position of its
predicted target match. This is exactly the lookup
evaluate_densematcher_pair_predictions.py already performs to score AUC/Err;
here we render the resulting vertex positions instead of only scoring them.

Four pairs, all "clean" provenance (no groups.txt label leakage) from
fair_comparison_meshup_vs_densematcher_matched_rows.csv, chosen to span the
range of DenseMatcher failure severity actually measured:
  knife:   151e2_knife_025 -> 6b047_knife_028   (PF02) -- near-total DenseMatcher
           failure (AUC 0.219): blade/handle end-to-end symmetry confusion.
  potato:  1e66110d80f -> aa7e5d9e2da (PF01) -- rank 17/20 by DenseMatcher AUC
           for this category (AUC 0.357): still a severe failure.
  cars:    35ebd_toy_car_022 -> 94b39_toy_car_025 (PF07) -- a milder failure
           (AUC 0.469) from cross-instance part-layout variation.
  shampoo: f6485_facial_cream_019 -> 4161c_shampoo_007 (PF02) -- rank 1/20 by
           DenseMatcher AUC for this category (AUC 0.769): even shampoo's
           worst pair is DenseMatcher's mildest gap of the four.

Two figures:
  densematcher_deformation_coherence.png -- source vertices are colored by a
    smooth position-based rainbow; Final and the DenseMatcher warp inherit
    that same source coloring (same vertex indices/topology in both cases).
    A spatially coherent method keeps the rainbow smooth; a confused
    correspondence scrambles it. Standard texture-transfer diagnostic from
    the shape-correspondence literature, applied to both methods here. Each
    row's camera azimuth is picked automatically (see best_azimuth in
    densematcher_compare_lib.py) to show the source's widest visible side.
  densematcher_edge_distortion_bars.png -- median per-face edge-stretch
    distortion relative to the untouched source mesh (|log2(deformed edge
    length / source edge length)|, averaged over a face's three edges; a
    standard as-rigid-as-possible-style local distortion diagnostic: 0 means
    a face moved without local stretching or folding, large values mean it
    was locally crushed or torn). This is deliberately not "distance from
    the warp to the target surface": DenseMatcher's implied warp places
    every vertex exactly on an existing target vertex by construction (a
    gather from target_v), so that distance is trivially ~0 everywhere and
    would not show anything. A first pass tried coloring the mesh by this
    distortion, but DenseMatcher's values are so uniformly extreme (see the
    numbers below) that a shared color scale saturates solid red with no
    spatial pattern to show -- the honest way to report a uniform effect is
    a number, not a heatmap, so this is a bar chart instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_review_figures import INK, MUTED, PAGE_BG, LINE, draw_text, render_panel, save_rgb  # noqa: E402
from densematcher_compare_lib import best_azimuth, face_edge_distortion, load_pair_data  # noqa: E402

import cv2

OUT = ROOT / "report_final" / "figures"

INK_HEX = "#{:02x}{:02x}{:02x}".format(*INK)
MUTED_HEX = "#{:02x}{:02x}{:02x}".format(*MUTED)
PAGE_BG_HEX = "#{:02x}{:02x}{:02x}".format(*PAGE_BG)


ROWS = [
    dict(
        category="knife",
        source_id="151e2_knife_025",
        target_id="6b047_knife_028",
        run=ROOT
        / "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_knifeonly_pf0205"
        / "knfS0T3_knifeonly_pf02_4000_hard_partfield_chamfer_only_knfS0T3_pf02_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6223681",
        label="Knife",
        meshup_auc=0.9887, meshup_err=0.28,
        dm_auc=0.219, dm_err=57.80,
        zoom=0.92,
    ),
    dict(
        category="potato",
        source_id="1e66110d80f049efa4f76f010417b853",
        target_id="aa7e5d9e2dad49ba8124faebeb8b9e4d",
        run=ROOT
        / "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_potato_top4"
        / "potaS1T2_potato_pf01_4000_hard_partfield_chamfer_only_potaS1T2_pf01_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6253144",
        label="Potato",
        meshup_auc=0.7555, meshup_err=6.113,
        dm_auc=0.3573, dm_err=16.841,
        zoom=0.90,
    ),
    dict(
        category="cars",
        source_id="35ebd_toy_car_022",
        target_id="94b39_toy_car_025",
        run=ROOT
        / "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_4000_a100"
        / "cars__cars_m000_35ebd_to_to_cars_m004_94b39_to_densecorr3d_nonanimals_test_partfield_guided_fair_a100_pf07_4000_hard_partfield_chamfer_only_cars__cars_m000_35ebd_to_to_cars_m004_94b39_to_pf07_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6179736",
        label="Cars",
        meshup_auc=0.9565, meshup_err=1.10,
        dm_auc=0.469, dm_err=14.89,
        zoom=0.95,
    ),
    dict(
        category="shampoo",
        source_id="f6485_facial_cream_019",
        target_id="4161c_shampoo_007",
        run=ROOT
        / "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_shampoo_pf0104"
        / "shpS2T0_shampoo_pf02_4000_hard_partfield_chamfer_only_shpS2T0_pf02_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6224589",
        label="Shampoo",
        meshup_auc=0.9341, meshup_err=1.646,
        dm_auc=0.7690, dm_err=6.756,
        zoom=0.85,
    ),
]


def load_row_data(row: dict) -> dict:
    return load_pair_data(row["category"], row["source_id"], row["target_id"], row["run"])


def build_coherence_figure() -> None:
    supersample = 3
    width, height = 1900, 2140
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)

    top = 56 * supersample
    row_h = (h - top) // len(ROWS)
    panel_w = w // 4

    headers = ("Source mesh", "This pipeline: deformed mesh", "DenseMatcher: implied warp", "Target mesh")
    for col, header in enumerate(headers):
        draw_text(
            image,
            header,
            (col * panel_w + 16 * supersample, top - 14 * supersample),
            scale=0.6 * supersample,
            color=MUTED,
            thickness=max(1, int(1.6 * supersample)),
        )

    for row_idx, row in enumerate(ROWS):
        data = load_row_data(row)
        azimuth = best_azimuth(data["src"].vertices)
        y0 = top + row_idx * row_h
        caption = f"{row['label']}   AUC {row['meshup_auc']:.2f} vs {row['dm_auc']:.2f}   Err {row['meshup_err']:.1f} vs {row['dm_err']:.1f}"
        draw_text(
            image,
            caption,
            (16 * supersample, y0 + 26 * supersample),
            scale=0.58 * supersample,
            color=INK,
            thickness=max(1, int(1.8 * supersample)),
        )
        panel_top = y0 + 36 * supersample
        panel_h = row_h - 46 * supersample
        for col, mesh in enumerate((data["src"], data["final"], data["dm"], data["target"])):
            render_panel(
                image,
                (col * panel_w, panel_top, panel_w, panel_h),
                mesh,
                "",
                azimuth=azimuth,
                elevation=8.0,
                zoom=row["zoom"],
            )
        if row_idx < len(ROWS) - 1:
            line_y = y0 + row_h - 6 * supersample
            cv2.line(image, (12 * supersample, line_y), (w - 12 * supersample, line_y), LINE, max(1, supersample))

    save_rgb(OUT / "densematcher_deformation_coherence.png", image, supersample, (width, height))


def build_distortion_bar_chart() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MESHUP = "#1b7a72"
    DENSEMATCHER = "#b5651d"
    GRID = "#e4e7eb"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK_HEX,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_HEX,
            "xtick.color": MUTED_HEX,
            "ytick.color": MUTED_HEX,
            "figure.facecolor": PAGE_BG_HEX,
            "axes.facecolor": PAGE_BG_HEX,
            "savefig.facecolor": PAGE_BG_HEX,
        }
    )

    labels, meshup_vals, dm_vals = [], [], []
    for row in ROWS:
        data = load_row_data(row)
        source_v = data["src"].vertices
        final_dist = face_edge_distortion(data["final"].faces, source_v, data["final"].vertices)
        dm_dist = face_edge_distortion(data["dm"].faces, source_v, data["dm"].vertices)
        labels.append(row["category"])
        meshup_vals.append(float(np.median(final_dist)))
        dm_vals.append(float(np.median(dm_dist)))
        print(
            f"{row['category']}: median |log2 edge-stretch| final={meshup_vals[-1]:.3f} "
            f"dm={dm_vals[-1]:.3f} (ratio {2**dm_vals[-1] / 2**meshup_vals[-1]:.1f}x)"
        )

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    y = np.arange(len(labels))
    bar_h = 0.34
    ax.barh(y + bar_h / 2, meshup_vals, height=bar_h, color=MESHUP, zorder=3)
    ax.barh(y - bar_h / 2, dm_vals, height=bar_h, color=DENSEMATCHER, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, max(dm_vals) * 1.16)
    ax.set_xlabel("Median per-face |log2(deformed edge / source edge)|")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    offset = max(dm_vals) * 0.02
    for yi, mv, dv in zip(y, meshup_vals, dm_vals):
        ax.text(mv + offset, yi + bar_h / 2, f"{mv:.2f}", va="center", fontsize=9.5, color=INK_HEX)
        ax.text(dv + offset, yi - bar_h / 2, f"{dv:.2f}", va="center", fontsize=9.5, color=INK_HEX)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=MESHUP), plt.Rectangle((0, 0), 1, 1, color=DENSEMATCHER)]
    fig.legend(
        legend_handles,
        ["This pipeline: Final", "DenseMatcher: implied warp"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "densematcher_edge_distortion_bars.png", dpi=200)
    plt.close(fig)


def main() -> None:
    build_coherence_figure()
    build_distortion_bar_chart()


if __name__ == "__main__":
    main()
