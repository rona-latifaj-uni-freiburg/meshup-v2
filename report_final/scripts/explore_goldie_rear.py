"""Full azimuth sweep of the (plain-shaded) Goldie mesh to find an angle
that shows the rear/butt, the tail, and the back legs clearly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import load_colored_ply  # noqa: E402

RUN = ROOT / (
    "jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_more_selected_best_asym035_jump500_4000/"
    "goldie_to_pug_sam3d_more_selected_pf09_4000_hard_partfield_chamfer_only_goldie_to_pug_pf09_pfonly_best_asym035_"
    "jneighbor1000_jump500_4000ep_4000ep_5829938"
)
v, f, _c = load_colored_ply(RUN / "colored_meshes/mesh_initial.ply")
BODY = np.array([0.80, 0.80, 0.78])
colors = np.tile(BODY, (v.shape[0], 1)).astype(np.float32)
face_rgb = colors[f].mean(axis=1)
mesh = Mesh(vertices=v, faces=f, colors=colors, face_rgb=face_rgb)

azs = list(range(-180, 181, 20))
supersample = 2
cell = 260
cols = 6
rows = (len(azs) + cols - 1) // cols
width, height = cell * cols, cell * rows
w, h = width * supersample, height * supersample
image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)
for i, az in enumerate(azs):
    r, c = divmod(i, cols)
    rect = (c * cell * supersample, r * cell * supersample, cell * supersample, cell * supersample)
    draw_text(image, f"az={az}", (rect[0] + 8 * supersample, rect[1] + 16 * supersample), scale=0.4 * supersample, color=INK, thickness=supersample)
    render_panel(image, rect, mesh, "", azimuth=float(az), elevation=12.0, zoom=0.8)
save_rgb(ROOT / "report_final" / "data" / "goldie_rear_sweep.png", image, supersample, (width, height))
print("done")
