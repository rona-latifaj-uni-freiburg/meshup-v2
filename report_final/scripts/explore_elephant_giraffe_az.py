"""Azimuth sweep for the elephant/giraffe pair to find a clear side view."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import load_mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK  # noqa: E402

RUN = ROOT / (
    "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_requested_animals_best_asym035/"
    "giraffe_to_elephant_5k_densecorr3d_requested_animals_hard_partfield_chamfer_only_densecorr3d_groups_"
    "pfonly_best_asym035_jneighbor1000_jump500_4000ep_5546334/partfield_colored_meshes"
)

elephant = load_mesh(RUN / "target_partfield_08.ply")
giraffe = load_mesh(RUN / "input_partfield_08.ply")

azs = list(range(-180, 181, 20))
supersample = 2
cell = 220
cols = 6
rows = (len(azs) + cols - 1) // cols
for name, mesh in (("elephant", elephant), ("giraffe", giraffe)):
    width, height = cell * cols, cell * rows
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)
    for i, az in enumerate(azs):
        r, c = divmod(i, cols)
        rect = (c * cell * supersample, r * cell * supersample, cell * supersample, cell * supersample)
        draw_text(image, f"az={az}", (rect[0] + 8 * supersample, rect[1] + 16 * supersample), scale=0.4 * supersample, color=INK, thickness=supersample)
        render_panel(image, rect, mesh, "", azimuth=float(az), elevation=6.0, zoom=0.85)
    save_rgb(ROOT / "report_final" / "data" / f"{name}_az_sweep.png", image, supersample, (width, height))
print("done")
