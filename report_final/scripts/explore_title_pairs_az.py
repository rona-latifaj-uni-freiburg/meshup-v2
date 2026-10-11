"""Azimuth sweep for the new title-grid pairs (backpack, car->truck) to find clean
side-on views, matching the convention used for the other exploratory sweep scripts
in this directory."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))
from make_review_figures import Mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import load_colored_ply, face_colors  # noqa: E402

BUCKET_PALETTE = np.array(
    [
        [0.122, 0.467, 0.706], [1.000, 0.498, 0.055], [0.173, 0.627, 0.173],
        [0.839, 0.153, 0.157], [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
        [0.890, 0.467, 0.761], [0.498, 0.498, 0.498], [0.737, 0.741, 0.133],
        [0.090, 0.745, 0.812], [0.700, 0.230, 0.500], [0.300, 0.680, 0.420],
    ],
    dtype=np.float32,
)

RUNS = {
    "backpack_final": (
        ROOT / (
            "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_backpack_expanded_pf0408/"
            "bpkeS3T0_bpkexpanded_pf04_4000_hard_partfield_chamfer_only_bpkeS3T0_pf04_pfonly_best_asym035_"
            "jneighbor1000_jump500_4000ep_4000ep_6277223"
        ),
        "colored_meshes/mesh_final_position.ply",
    ),
    "car_truck_final": (
        ROOT / (
            "jobs_with_target_guidance/cross_animal_spike_runs/outputs/"
            "densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_4000_a100/"
            "cars__cars_m000_35ebd_to_to_cars_m003_88c92_to_densecorr3d_nonanimals_test_partfield_guided_fair_"
            "a100_pf07_4000_hard_partfield_chamfer_only_cars__cars_m000_35ebd_to_to_cars_m003_88c92_to_pf07_"
            "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6179731"
        ),
        "colored_meshes/mesh_final_position.ply",
    ),
}

azs = list(range(0, 360, 30))
supersample = 2
cell = 220
cols = 6
rows = (len(azs) + cols - 1) // cols
for name, (run, rel) in RUNS.items():
    v, f, _ = load_colored_ply(run / rel)
    labels = np.load(run / "partfield_labels/bucket_labels.npz")["source_labels"]
    c = BUCKET_PALETTE[labels % len(BUCKET_PALETTE)]
    mesh = Mesh(v, f, c, face_colors(f, c))
    width, height = cell * cols, cell * rows
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)
    for i, az in enumerate(azs):
        r, c2 = divmod(i, cols)
        rect = (c2 * cell * supersample, r * cell * supersample, cell * supersample, cell * supersample)
        draw_text(image, f"az={az}", (rect[0] + 6 * supersample, rect[1] + 14 * supersample), scale=0.35 * supersample, color=INK, thickness=supersample)
        render_panel(image, rect, mesh, "", azimuth=float(az), elevation=8.0, zoom=0.8)
    save_rgb(ROOT / "report_final" / "data" / f"{name}_az_sweep.png", image, supersample, (width, height))
print("done")
