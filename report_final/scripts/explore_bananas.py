"""Exploratory render: all 6 banana meshes, plain shading, several azimuths,
to find two that actually look like recognizable bananas."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import load_mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK  # noqa: E402

SEG = ROOT / "jobs_with_target_guidance/partfield_segments/densecorr3d_categories/densecorr3d_nonanimals_test_20260806/banana/buckets_06/colored"
MESHES = sorted(SEG.glob("banana_*_partfield_06_parts.ply"))
AZIMUTHS = [0.0, 45.0, 90.0, 135.0]

supersample = 2
cell_w, cell_h = 300, 260
width = cell_w * len(AZIMUTHS)
height = cell_h * len(MESHES) + 40
w, h = width * supersample, height * supersample
image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)

for row, path in enumerate(MESHES):
    mesh = load_mesh(path)
    draw_text(image, path.stem, (10 * supersample, (40 + row * cell_h + 20) * supersample), scale=0.42 * supersample, color=INK, thickness=supersample)
    for col, az in enumerate(AZIMUTHS):
        rect = (col * cell_w * supersample, (40 + row * cell_h) * supersample, cell_w * supersample, cell_h * supersample)
        render_panel(image, rect, mesh, f"az={az}", azimuth=az, elevation=10.0, zoom=0.8)

save_rgb(ROOT / "report_final" / "data" / "banana_explore.png", image, supersample, (width, height))
print("done")
