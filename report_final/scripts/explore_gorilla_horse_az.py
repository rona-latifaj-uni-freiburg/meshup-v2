"""Azimuth sweep for gorilla1->gorilla5 and horse0->horse6 targets to find clear views."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))
from make_review_figures import Mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import load_colored_ply, face_colors  # noqa: E402


def parse_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:4]]
                faces.append(idx)
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def pos_color(v):
    lo, hi = v.min(0, keepdims=True), v.max(0, keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1
    return ((v - lo) / rng).astype(np.float32)


targets = {
    "horse15": ROOT / "external/TOSCA/meshup_prepared_zup_xfront/objs/horse15.obj",
}

azs = list(range(0, 360, 30))
supersample = 2
cell = 220
cols = 6
rows = (len(azs) + cols - 1) // cols
for name, path in targets.items():
    v, f = parse_obj(path)
    c = pos_color(v)
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
