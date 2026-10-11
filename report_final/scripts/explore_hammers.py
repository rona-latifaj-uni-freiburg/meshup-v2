"""Exploratory render: all 5 hammer meshes at a few azimuths, to pick the
clearest pair and camera angle for the report figure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import load_mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK  # noqa: E402

SEG = ROOT / "jobs_with_target_guidance/partfield_segments/densecorr3d_categories/densecorr3d_hammer_only_20260808/hammeronly/buckets_04/colored"
MESHES = [
    ("m000_04083", SEG / "hammeronly_m000_04083_partfield_04_parts.ply"),
    ("m001_591e8", SEG / "hammeronly_m001_591e8_partfield_04_parts.ply"),
    ("m002_70ced", SEG / "hammeronly_m002_70ced_partfield_04_parts.ply"),
    ("m003_859f0", SEG / "hammeronly_m003_859f0_partfield_04_parts.ply"),
    ("m004_a206c", SEG / "hammeronly_m004_a206c_partfield_04_parts.ply"),
]
AZIMUTHS = [0.0, 45.0, 90.0]

supersample = 2
cell_w, cell_h = 340, 300
width = cell_w * len(AZIMUTHS)
height = cell_h * len(MESHES) + 40
w, h = width * supersample, height * supersample
image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)

for row, (name, path) in enumerate(MESHES):
    mesh = load_mesh(path)
    draw_text(image, name, (10 * supersample, (40 + row * cell_h + 20) * supersample), scale=0.5 * supersample, color=INK, thickness=supersample)
    for col, az in enumerate(AZIMUTHS):
        rect = (col * cell_w * supersample, (40 + row * cell_h) * supersample, cell_w * supersample, cell_h * supersample)
        render_panel(image, rect, mesh, f"az={az}", azimuth=az, elevation=10.0, zoom=0.85)

save_rgb(ROOT / "report_final" / "data" / "hammer_explore.png", image, supersample, (width, height))
print("done")
