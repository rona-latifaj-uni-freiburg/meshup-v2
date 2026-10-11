"""Re-render the zucchini PF01-vs-PF02 comparison at a legible size.

The figure previously reused two old, tiny, dimly-lit renders inherited from
an earlier report draft. This renders both cases fresh, large, from a clear
side angle, with a solid warm shading (not PartField bucket colors, since
this figure is about surface geometry quality, not part identity).
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

OUT = ROOT / "report_final" / "figures"

SHADE_COLOR = np.array([0.86, 0.70, 0.46])  # warm amber-tan, shows folds/lighting clearly


def load_uniform(path: Path) -> Mesh:
    vertices, faces, _colors = load_colored_ply(path)
    colors = np.tile(SHADE_COLOR, (vertices.shape[0], 1)).astype(np.float32)
    face_rgb = np.tile(SHADE_COLOR, (faces.shape[0], 1)).astype(np.float32)
    return Mesh(vertices=vertices, faces=faces, colors=colors, face_rgb=face_rgb)


def panel_row(image: np.ndarray, y0: int, height: int, width: int, mesh_path: Path, label: str, supersample: int) -> None:
    margin = 40 * supersample
    label_w = 300 * supersample
    panel_w = width - 2 * margin - label_w
    lines = label.split("\n")
    line_gap = 34 * supersample
    start_y = y0 + height // 2 - (len(lines) - 1) * line_gap // 2
    for idx, line in enumerate(lines):
        if not line:
            continue
        draw_text(
            image,
            line,
            (margin, start_y + idx * line_gap),
            scale=0.52 * supersample,
            color=INK if idx == 0 else MUTED,
            thickness=max(1, supersample) + (1 if idx == 0 else 0),
        )
    mesh = load_uniform(mesh_path)
    render_panel(
        image,
        (margin + label_w, y0, panel_w, height),
        mesh,
        "",
        azimuth=90.0,
        elevation=14.0,
        zoom=0.92,
    )


def main() -> None:
    base = ROOT / "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_zucchini_pf0106"
    pf01 = base / (
        "zucS0T2_zucchini_pf01_4000_hard_partfield_chamfer_only_zucS0T2_pf01_pfonly_best_asym035_"
        "jneighbor1000_jump500_4000ep_4000ep_6224795/colored_meshes/mesh_final_position.ply"
    )
    pf02 = base / (
        "zucS0T2_zucchini_pf02_4000_hard_partfield_chamfer_only_zucS0T2_pf02_pfonly_best_asym035_"
        "jneighbor1000_jump500_4000ep_4000ep_6224796/colored_meshes/mesh_final_position.ply"
    )

    supersample = 3
    width, height = 1700, 900
    row_h = height // 2
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)

    panel_row(
        image, 0, row_h * supersample, w,
        pf01, "PF01\nno segmentation\n\nsmooth, correct\nproportions",
        supersample,
    )
    panel_row(
        image, row_h * supersample, row_h * supersample, w,
        pf02, "PF02\nbuckets active\n\nvisibly torn,\nself-intersecting",
        supersample,
    )

    line_y = row_h * supersample
    import cv2

    cv2.line(image, (0, line_y), (w, line_y), (225, 229, 234), max(1, supersample))

    save_rgb(OUT / "zucchini_pf_collapse.png", image, supersample, (width, height))


if __name__ == "__main__":
    main()
