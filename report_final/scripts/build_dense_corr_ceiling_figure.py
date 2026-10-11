"""Limitations-section figure: what this pipeline achieves when given real dense
ground-truth correspondence instead of PartField buckets. Three real TOSCA pairs,
chosen for category variance (dog, gorilla, horse) and picked as strong real results
(by the project's own TOSCA benchmark numbers, not cherry-picked renders):
  dog0    -> dog1     AUC 0.964 / Err 0.90  (large standing -> sitting pose change)
  gorilla1 -> gorilla5 AUC 0.941 / Err 1.57  (knuckle-walking crouch pose change)
  horse6  -> horse10  AUC 0.989 / Err 0.27  (walk -> jump pose change)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))
from make_review_figures import Mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK, MUTED  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import load_colored_ply, face_colors  # noqa: E402

OUT = ROOT / "report_final" / "figures"
GUIDED = ROOT / "jobs_with_target_guidance/cross_animal_spike_runs/outputs/tosca_correspondence_guided"
OBJS = ROOT / "external/TOSCA/meshup_prepared_zup_xfront/objs"

ROWS = [
    (
        "dog0_to_dog1_tosca_vcorr_2000_hard_partfield_chamfer_only_dog0_to_dog1_tosca_identity_vcorr_"
        "w20000_jacns100_jump50_2000ep_2000ep_6168492",
        "dog1.obj", "Dog0 -> Dog1 (standing -> sitting)", "AUC 0.964 / Err 0.90", 0.0, 0.88,
    ),
    (
        "gorilla1_to_gorilla5_tosca_vcorr_2000_hard_partfield_chamfer_only_gorilla1_to_gorilla5_tosca_"
        "identity_vcorr_w20000_jacns100_jump50_2000ep_2000ep_6175928",
        "gorilla5.obj", "Gorilla1 -> Gorilla5 (knuckle-walk pose change)", "AUC 0.941 / Err 1.57", 0.0, 0.82,
    ),
    (
        "horse6_to_horse10_tosca_vcorr_2000_hard_partfield_chamfer_only_horse6_to_horse10_tosca_identity_"
        "vcorr_w20000_jacns100_jump50_2000ep_2000ep_6174633",
        "horse10.obj", "Horse6 -> Horse10 (walk to jump, challenging)", "AUC 0.989 / Err 0.27", 0.0, 0.8,
    ),
]


def parse_obj(path: Path):
    verts, faces = [], []
    with path.open() as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:4]]
                faces.append(idx)
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def pos_color(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(0, keepdims=True), v.max(0, keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1
    return ((v - lo) / rng).astype(np.float32)


def main() -> None:
    supersample = 3
    width, height = 1500, 1780
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)

    draw_text(image, "Dense ground-truth correspondence as the driving loss (TOSCA)",
              (24 * supersample, 40 * supersample), scale=0.56 * supersample, color=INK, thickness=2 * supersample)

    row_h = (h - 70 * supersample) // len(ROWS)
    panel_w = width // 3
    for row, (run_dir, target_obj, title, metric, az, zoom) in enumerate(ROWS):
        run = GUIDED / run_dir
        src_v, src_f, _ = load_colored_ply(run / "colored_meshes/mesh_initial.ply")
        fin_v, fin_f, _ = load_colored_ply(run / "colored_meshes/mesh_final_position.ply")
        tgt_v, tgt_f = parse_obj(OBJS / target_obj)

        src_c = pos_color(src_v)
        tgt_c = pos_color(tgt_v)
        meshes = [
            ("Source", Mesh(src_v, src_f, src_c, face_colors(src_f, src_c))),
            ("Final", Mesh(fin_v, fin_f, src_c, face_colors(fin_f, src_c))),
            ("Target", Mesh(tgt_v, tgt_f, tgt_c, face_colors(tgt_f, tgt_c))),
        ]

        y0 = 70 * supersample + row * row_h
        draw_text(image, f"{title}  --  {metric}", (18 * supersample, y0 + 26 * supersample),
                  scale=0.44 * supersample, color=INK, thickness=max(1, int(1.4 * supersample)))
        top = y0 + 40 * supersample
        for i, (label, mesh) in enumerate(meshes):
            x0 = i * panel_w * supersample
            draw_text(image, label, (x0 + 18 * supersample, top + 20 * supersample),
                      scale=0.4 * supersample, color=MUTED, thickness=max(1, supersample))
            rect = (x0, top + 30 * supersample, panel_w * supersample, row_h - 70 * supersample)
            render_panel(image, rect, mesh, "", azimuth=az, elevation=8.0, zoom=zoom)

    save_rgb(OUT / "tosca_dense_corr_ceiling.png", image, supersample, (width, height))
    print("wrote tosca_dense_corr_ceiling.png")


if __name__ == "__main__":
    main()
