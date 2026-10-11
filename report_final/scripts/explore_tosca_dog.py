"""Quick visual check of the TOSCA dense-correspondence dog0->dog1 result."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))
from make_review_figures import Mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK, MUTED  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import load_colored_ply, face_colors  # noqa: E402

RUN = ROOT / (
    "jobs_with_target_guidance/cross_animal_spike_runs/outputs/tosca_correspondence_guided/"
    "dog0_to_dog1_tosca_vcorr_2000_hard_partfield_chamfer_only_dog0_to_dog1_tosca_identity_vcorr_"
    "w20000_jacns100_jump50_2000ep_2000ep_6168492"
)


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


src_v, src_f, _ = load_colored_ply(RUN / "colored_meshes/mesh_initial.ply")
fin_v, fin_f, _ = load_colored_ply(RUN / "colored_meshes/mesh_final_position.ply")
tgt_v, tgt_f = parse_obj(ROOT / "external/TOSCA/meshup_prepared_zup_xfront/objs/dog1.obj")

src_c = pos_color(src_v)
fin_c = pos_color(src_v)  # carried through, not recomputed
tgt_c = pos_color(tgt_v)

meshes = {
    "Source (dog0)": Mesh(src_v, src_f, src_c, face_colors(src_f, src_c)),
    "Final (deformed)": Mesh(fin_v, fin_f, fin_c, face_colors(fin_f, fin_c)),
    "Target (dog1)": Mesh(tgt_v, tgt_f, tgt_c, face_colors(tgt_f, tgt_c)),
}

supersample = 2
azs = [0, 45, 90, 135, 180, -135, -90, -45]
cell = 260
cols = 4
rows = (len(azs) + cols - 1) // cols
width, height = cell * cols, cell * rows * len(meshes)
w, h = width * supersample, height * supersample
image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)
for mi, (name, mesh) in enumerate(meshes.items()):
    for i, az in enumerate(azs):
        r, c = divmod(i, cols)
        rect = (c * cell * supersample, (mi * rows + r) * cell * supersample, cell * supersample, cell * supersample)
        draw_text(image, f"{name} az={az}", (rect[0] + 6 * supersample, rect[1] + 14 * supersample), scale=0.32 * supersample, color=INK, thickness=supersample)
        render_panel(image, rect, mesh, "", azimuth=float(az), elevation=8.0, zoom=0.8)
save_rgb(ROOT / "report_final" / "data" / "tosca_dog_sweep.png", image, supersample, (width, height))
print("done")
