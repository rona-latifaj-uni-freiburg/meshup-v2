"""Exploratory render: goldie/pug PartField buckets at several azimuths, to
identify (by eye) which bucket color is the tail."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import load_mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK  # noqa: E402

RUN = ROOT / (
    "jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_more_selected_best_asym035_jump500_4000/"
    "goldie_to_pug_sam3d_more_selected_pf09_4000_hard_partfield_chamfer_only_goldie_to_pug_pf09_pfonly_best_asym035_"
    "jneighbor1000_jump500_4000ep_4000ep_5829938"
)
MESHES = [
    ("input (goldie)", RUN / "partfield_colored_meshes/input_partfield_09.ply"),
    ("final (deformed)", RUN / "partfield_colored_meshes/final_partfield_09.ply"),
    ("target (pug)", RUN / "partfield_colored_meshes/target_partfield_09.ply"),
]
AZIMUTHS = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 150.0]

supersample = 2
cell_w, cell_h = 260, 260
width = cell_w * len(AZIMUTHS)
height = cell_h * len(MESHES) + 40
w, h = width * supersample, height * supersample
image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)

for row, (name, path) in enumerate(MESHES):
    mesh = load_mesh(path)
    draw_text(image, name, (10 * supersample, (40 + row * cell_h + 20) * supersample), scale=0.45 * supersample, color=INK, thickness=supersample)
    for col, az in enumerate(AZIMUTHS):
        rect = (col * cell_w * supersample, (40 + row * cell_h) * supersample, cell_w * supersample, cell_h * supersample)
        render_panel(image, rect, mesh, f"az={az}", azimuth=az, elevation=10.0, zoom=0.85)

save_rgb(ROOT / "report_final" / "data" / "goldie_pug_explore.png", image, supersample, (width, height))
print("done")
