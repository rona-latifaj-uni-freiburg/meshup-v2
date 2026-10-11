"""Tail-tip correspondence figure for Goldie -> Pug, v2.

Full-body smooth position-color gradient (the project's own `position_to_color`
scheme: RGB = per-axis normalized XYZ, semantic_tracking/vertex_color_tracking.py)
so the whole animal, including the tail, reads clearly -- not just a flat grey
body with a small highlighted patch. On top of that, the tail *tip* specifically
(farthest 30% of the tail's PartField bucket from the body) is marked in bright
yellow on all three panels, so the same-meaning marker can be compared directly:
yellow on the source = the tip; yellow on the final mesh = where those exact
vertices physically ended up; yellow on the target = where the tip should be.
Camera is a rear three-quarter view (azimuth 165) so the tail, the flank
where the tip lands, and the hind legs are all visible at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, render_panel, save_rgb, draw_text, PAGE_BG, INK, MUTED  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import load_colored_ply  # noqa: E402

RUN = ROOT / (
    "jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_more_selected_best_asym035_jump500_4000/"
    "goldie_to_pug_sam3d_more_selected_pf09_4000_hard_partfield_chamfer_only_goldie_to_pug_pf09_pfonly_best_asym035_"
    "jneighbor1000_jump500_4000ep_4000ep_5829938"
)
OUT = ROOT / "report_final" / "figures" / "goldie_pug_tail_points.png"

TIP_MARK = np.array([1.0, 0.85, 0.0])  # bright yellow: the tracked tip, same meaning in all 3 panels
TAIL_BUCKET = 7
TIP_FRACTION = 0.30


def position_to_color(positions: np.ndarray) -> np.ndarray:
    """Exact reimplementation of semantic_tracking.vertex_color_tracking.position_to_color."""
    pos_min = positions.min(axis=0, keepdims=True)
    pos_max = positions.max(axis=0, keepdims=True)
    pos_range = pos_max - pos_min
    pos_range[pos_range == 0] = 1
    return ((positions - pos_min) / pos_range).astype(np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    center = (v.min(0) + v.max(0)) / 2
    scale = (v.max(0) - v.min(0)).max()
    return (v - center) / scale


def tip_mask(vertices: np.ndarray, labels: np.ndarray, body_bucket: int = 3) -> np.ndarray:
    vn = normalize(vertices)
    tail_mask = labels == TAIL_BUCKET
    body_centroid = vn[labels == body_bucket].mean(axis=0)
    dist = np.linalg.norm(vn - body_centroid, axis=1)
    threshold = np.quantile(dist[tail_mask], 1.0 - TIP_FRACTION)
    return tail_mask & (dist >= threshold)


def gradient_mesh(vertices: np.ndarray, faces: np.ndarray, base_positions: np.ndarray, highlight: np.ndarray) -> Mesh:
    """base_positions drives the smooth color gradient (a mesh's own rest-pose
    positions, kept fixed even if `vertices` itself is a deformed copy, so the
    color is inherited rather than recomputed -- exactly how the project's own
    color-tracking works)."""
    colors = position_to_color(base_positions)
    colors[highlight] = TIP_MARK
    face_rgb = colors[faces].mean(axis=1)
    return Mesh(vertices=vertices, faces=faces, colors=colors, face_rgb=face_rgb)


def main() -> None:
    npz = np.load(RUN / "partfield_labels/bucket_labels.npz")
    source_labels = npz["source_labels"]
    target_labels = npz["target_labels"]

    src_v, src_f, _ = load_colored_ply(RUN / "colored_meshes/mesh_initial.ply")
    final_v, final_f, _ = load_colored_ply(RUN / "colored_meshes/mesh_final_position.ply")
    tgt_v, tgt_f, _ = load_colored_ply(RUN / "partfield_colored_meshes/target_partfield_09.ply")

    src_tip = tip_mask(src_v, source_labels)
    tgt_tip = tip_mask(tgt_v, target_labels)
    print(f"source tail-tip vertices: {src_tip.sum()}")
    print(f"target tail-tip vertices: {tgt_tip.sum()}")

    # Final mesh: identical topology/vertex order as source, so it gets the
    # SAME rest-pose gradient (computed from src_v, not final_v) -- the color
    # is carried by the vertex, not recomputed from the deformed position.
    mesh_source = gradient_mesh(src_v, src_f, src_v, src_tip)
    mesh_final = gradient_mesh(final_v, final_f, src_v, src_tip)
    mesh_target = gradient_mesh(tgt_v, tgt_f, tgt_v, tgt_tip)

    supersample = 3
    width, height = 2000, 820
    panel_w = width // 3
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)

    title = "Goldie to pug: full-body position gradient, tail tip marked bright yellow throughout"
    draw_text(image, title, (24 * supersample, 40 * supersample), scale=0.62 * supersample, color=INK, thickness=2 * supersample)

    panels = [
        ("Source (Goldie)", "color = position (own rest pose); yellow = tail tip", mesh_source, 165.0),
        ("Final (deformed)", "same per-vertex colors carried through; yellow tip lands on the flank", mesh_final, 165.0),
        ("Target (Pug)", "own position gradient; yellow = pug's real tail tip", mesh_target, 165.0),
    ]
    top = 150 * supersample
    for i, (label1, label2, mesh, az) in enumerate(panels):
        x0 = i * panel_w * supersample
        draw_text(image, label1, (x0 + 18 * supersample, 92 * supersample), scale=0.5 * supersample, color=INK, thickness=max(1, supersample) + 1)
        draw_text(image, label2, (x0 + 18 * supersample, 122 * supersample), scale=0.4 * supersample, color=MUTED, thickness=max(1, supersample))
        render_panel(image, (x0, top, panel_w * supersample, h - top), mesh, "", azimuth=az, elevation=8.0, zoom=0.86)

    save_rgb(OUT, image, supersample, (width, height))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
