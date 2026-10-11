import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, render_panel, save_rgb, PAGE_BG  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import load_colored_ply  # noqa: E402

RUN = ROOT / (
    "jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_more_selected_best_asym035_jump500_4000/"
    "goldie_to_pug_sam3d_more_selected_pf09_4000_hard_partfield_chamfer_only_goldie_to_pug_pf09_pfonly_best_asym035_"
    "jneighbor1000_jump500_4000ep_4000ep_5829938"
)

npz = np.load(RUN / "partfield_labels/bucket_labels.npz")
labels = npz["source_labels"]
v, f, _c = load_colored_ply(RUN / "colored_meshes/mesh_initial.ply")

GREY = np.array([0.82, 0.82, 0.8])
HILITE = np.array([1.0, 0.0, 0.7])

colors = np.tile(GREY, (v.shape[0], 1))
colors[labels == 7] = HILITE
face_rgb = colors[f].mean(axis=1)
mesh = Mesh(vertices=v, faces=f, colors=colors, face_rgb=face_rgb)

supersample = 2
width, height = 1400, 500
w, h = width * supersample, height * supersample
image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)
for i, az in enumerate([-90, -60, -30, 0]):
    render_panel(image, (i * w // 4, 0, w // 4, h), mesh, f"az={az}", azimuth=az, elevation=10.0, zoom=0.85)
save_rgb(ROOT / "report_final" / "data" / "verify_tail_bucket.png", image, supersample, (width, height))
print("done")
