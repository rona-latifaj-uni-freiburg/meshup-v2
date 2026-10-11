"""Title-page figure: compact Source -> Final panels for real deformation pairs,
PartField-bucket colored, for use above the abstract.

Builds title_row_3pairs.png: a single horizontal row of three cherry-picked real pairs
(Bus->Car, Horse->Giraffe in the middle, Banana), each shown as Source / Final (the
actual deformed source, not the target) with an arrow between them. All text (pair
title only) sits below the meshes, never on top of them; no panel borders, no
Source/Final labels. Pure white canvas background. The row is deliberately short
(wide, low aspect ratio) so the image plus the title block above it still leaves room
for the abstract text on the same first page.

All meshes are real project outputs; no synthetic data. Bus/Car and Banana runs' own
colored_meshes/*.ply use continuous per-vertex position coloring, not discrete bucket
colors (verified: near-unique colors per vertex despite a handful of real buckets in
bucket_labels.npz) -- so both source and final are recolored here directly from the
run's own saved bucket labels with a clean discrete palette, applied identically to
both meshes (valid because the fixed-topology deformation preserves vertex
correspondence exactly, so source_labels indexes both meshes' vertices identically).
Horse->Giraffe already has pre-baked discrete PartField-bucket colored PLYs
(partfield_colored_meshes/input_partfield_08.ply, final_partfield_08.ply) and is used
as-is.

Horse->Giraffe is rendered with a SHARED scale between Source and Final instead of
make_review_figures.render_panel's normal per-panel independent normalization. A horse
(long, low) and a giraffe (short, very tall) each independently normalized to fill
their own panel end up looking like similarly-sized animals -- which is exactly what a
naive read of the two panels side by side gets wrong, since a giraffe is genuinely much
taller. render_pair_cell_shared_scale below normalizes both meshes from one shared
bounding box (Final's vertices are the deformed Source's, so they already share a
coordinate frame) and projects both with one shared on-screen scale, so the real size
relationship survives into the render.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, load_mesh, render_panel, save_rgb, draw_text, INK  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import (  # noqa: E402
    face_colors, load_colored_ply, transform_vertices,
)

OUT = ROOT / "report_final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

WHITE = (255, 255, 255)

# Clean, distinct categorical palette for buckets 0..11 (tab10 + 2 extra), used when a
# run's own colored PLY turns out to use continuous per-vertex coloring rather than
# discrete bucket colors (verified case: the bus/car and banana runs below).
BUCKET_PALETTE = np.array(
    [
        [0.122, 0.467, 0.706], [1.000, 0.498, 0.055], [0.173, 0.627, 0.173],
        [0.839, 0.153, 0.157], [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
        [0.890, 0.467, 0.761], [0.498, 0.498, 0.498], [0.737, 0.741, 0.133],
        [0.090, 0.745, 0.812], [0.700, 0.230, 0.500], [0.300, 0.680, 0.420],
    ],
    dtype=np.float32,
)


def load_source_final_from_labels(run: Path) -> tuple[Mesh, Mesh]:
    labels = np.load(run / "partfield_labels/bucket_labels.npz")
    source_labels = labels["source_labels"]
    colors = BUCKET_PALETTE[source_labels % len(BUCKET_PALETTE)]

    src_v, src_f, _ = load_colored_ply(run / "colored_meshes/mesh_initial.ply")
    source_mesh = Mesh(vertices=src_v, faces=src_f, colors=colors, face_rgb=face_colors(src_f, colors))

    fin_v, fin_f, _ = load_colored_ply(run / "colored_meshes/mesh_final_position.ply")
    final_mesh = Mesh(vertices=fin_v, faces=fin_f, colors=colors, face_rgb=face_colors(fin_f, colors))
    return source_mesh, final_mesh


def load_source_final_prebaked(folder: Path, src_name: str, fin_name: str) -> tuple[Mesh, Mesh]:
    return load_mesh(folder / src_name), load_mesh(folder / fin_name)


PAIRS = {
    "horse_giraffe": (
        "Horse to Giraffe",
        "prebaked",
        ROOT / (
            "jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_selected_best_asym035_jump500_4000/"
            "horse2_to_giraffe4_sam3d_selected_pf08_4000_hard_partfield_chamfer_only_horse2_to_giraffe4_pf08_"
            "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5816558/partfield_colored_meshes"
        ),
        "input_partfield_08.ply", "final_partfield_08.ply",
        -68.0, 12.0, 0.92,
    ),
    "bus_car": (
        "Bus to Car",
        "labels",
        ROOT / (
            "jobs_with_target_guidance/cross_animal_spike_runs/outputs/"
            "densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_4000_a100/"
            "cars__cars_m002_f756d_to_to_cars_m000_35ebd_to_densecorr3d_nonanimals_test_partfield_guided_fair_"
            "a100_pf08_4000_hard_partfield_chamfer_only_cars__cars_m002_f756d_to_to_cars_m000_35ebd_to_pf08_"
            "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_6179796"
        ),
        None, None,
        240.0, 18.0, 0.78,
    ),
    "mushroom": (
        "Mushroom",
        "labels",
        ROOT / (
            "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_mushroom_top4/"
            "mushS0T5_mushroom_pf02_4000_hard_partfield_chamfer_only_mushS0T5_pf02_pfonly_best_asym035_"
            "jneighbor1000_jump500_4000ep_4000ep_6253459"
        ),
        None, None,
        30.0, 10.0, 0.85,
    ),
}


def load_pair(key: str) -> tuple[str, Mesh, Mesh, float, float, float]:
    title, mode, folder, src_name, fin_name, az, el, zoom = PAIRS[key]
    if mode == "prebaked":
        src, fin = load_source_final_prebaked(folder, src_name, fin_name)
    else:
        src, fin = load_source_final_from_labels(folder)
    return title, src, fin, az, el, zoom


def draw_arrow(image: np.ndarray, x0: int, x1: int, y: int, supersample: int) -> None:
    thickness = max(1, int(2.2 * supersample))
    cv2.line(image, (x0, y), (x1 - int(6 * supersample), y), INK, thickness, cv2.LINE_AA)
    cv2.arrowedLine(
        image, (x1 - int(16 * supersample), y), (x1, y), INK,
        thickness=thickness, tipLength=1.1, line_type=cv2.LINE_AA,
    )


def rasterize_mesh(image: np.ndarray, rect: tuple[int, int, int, int], mesh: Mesh,
                    transformed: np.ndarray, xy_center: np.ndarray, xy_range: np.ndarray,
                    zoom: float) -> None:
    """Draw a mesh whose vertices have already been rotated (transform_vertices) and
    whose on-screen scale is driven by an externally supplied xy_center/xy_range,
    rather than render_panel's own per-mesh auto-fit. Mirrors render_panel's shading
    and face-sorting exactly."""
    x0, y0, width, height = rect
    pad_x = width * 0.07
    pad_top, pad_bottom = 18, 6
    content_h = max(height - pad_top - pad_bottom, 1)
    scale = min((width - 2 * pad_x) / xy_range[0], content_h / xy_range[1]) * float(zoom)

    projected = np.empty((transformed.shape[0], 2), dtype=np.float32)
    center_x = x0 + width * 0.5
    center_y = y0 + pad_top + content_h * 0.5
    projected[:, 0] = center_x + (transformed[:, 0] - xy_center[0]) * scale
    projected[:, 1] = center_y - (transformed[:, 1] - xy_center[1]) * scale

    tri_depth = transformed[mesh.faces, 2].mean(axis=1)
    order = np.argsort(tri_depth)
    tri_vertices = transformed[mesh.faces]
    normals = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    light = np.array([0.24, 0.56, 0.80], dtype=np.float32)
    light /= np.linalg.norm(light)
    brightness = 0.76 + 0.24 * np.maximum(np.abs(normals @ light), 0.0)

    for face_idx in order:
        pts = np.round(projected[mesh.faces[face_idx]]).astype(np.int32)
        if cv2.contourArea(pts) < 0.2:
            continue
        color = np.clip(mesh.face_rgb[face_idx] * brightness[face_idx] * 255.0, 0, 255).astype(np.uint8)
        cv2.fillConvexPoly(image, pts, (int(color[2]), int(color[1]), int(color[0])), lineType=cv2.LINE_AA)


def render_pair_cell_shared_scale(image: np.ndarray, rect: tuple[int, int, int, int], title: str,
                                   src: Mesh, fin: Mesh, az: float, el: float, zoom: float,
                                   supersample: int) -> None:
    """Like render_pair_cell, but Source and Final share one normalization and one
    on-screen scale so their real relative size survives (see module docstring)."""
    x0, y0, w, h = rect
    text_h = int(0.10 * h)
    panel_h = h - text_h

    gap = int(0.08 * w)
    panel_w = (w - gap) // 2
    src_rect = (x0, y0, panel_w, panel_h)
    fin_rect = (x0 + panel_w + gap, y0, panel_w, panel_h)

    all_v = np.concatenate([src.vertices, fin.vertices], axis=0)
    center = (all_v.min(0) + all_v.max(0)) * 0.5
    scale = float((all_v.max(0) - all_v.min(0)).max())
    src_norm = (src.vertices - center) / max(scale, 1e-8)
    fin_norm = (fin.vertices - center) / max(scale, 1e-8)

    src_t = transform_vertices(src_norm, az, el)
    fin_t = transform_vertices(fin_norm, az, el)
    all_t = np.concatenate([src_t, fin_t], axis=0)
    xy_min = all_t[:, :2].min(0)
    xy_max = all_t[:, :2].max(0)
    xy_center = (xy_min + xy_max) * 0.5
    xy_range = np.maximum(xy_max - xy_min, 1e-6)

    rasterize_mesh(image, src_rect, src, src_t, xy_center, xy_range, zoom)
    rasterize_mesh(image, fin_rect, fin, fin_t, xy_center, xy_range, zoom)

    arrow_y = y0 + panel_h // 2
    draw_arrow(image, x0 + panel_w + int(0.015 * w), x0 + panel_w + gap - int(0.015 * w), arrow_y, supersample)

    title_y = y0 + panel_h + int(text_h * 0.5)
    (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.56 * supersample, max(1, int(1.5 * supersample)))
    title_x = x0 + max(0, (w - title_w) // 2)
    draw_text(image, title, (title_x, title_y), scale=0.56 * supersample, color=INK,
              thickness=max(1, int(1.5 * supersample)))


def render_pair_cell(image: np.ndarray, rect: tuple[int, int, int, int], title: str,
                      src: Mesh, fin: Mesh, az: float, el: float, zoom: float,
                      supersample: int) -> None:
    x0, y0, w, h = rect
    text_h = int(0.10 * h)
    panel_h = h - text_h

    gap = int(0.08 * w)
    panel_w = (w - gap) // 2
    src_rect = (x0, y0, panel_w, panel_h)
    fin_rect = (x0 + panel_w + gap, y0, panel_w, panel_h)
    render_panel(image, src_rect, src, "", azimuth=az, elevation=el, zoom=zoom)
    render_panel(image, fin_rect, fin, "", azimuth=az, elevation=el, zoom=zoom)
    # render_panel always draws a thin border rectangle; erase it since both the panel
    # fill and the canvas are already pure white, so this leaves no visible seam.
    for px0, py0, pw, ph in (src_rect, fin_rect):
        cv2.rectangle(image, (px0, py0), (px0 + pw, py0 + ph), WHITE, 2)

    arrow_y = y0 + panel_h // 2
    draw_arrow(image, x0 + panel_w + int(0.015 * w), x0 + panel_w + gap - int(0.015 * w), arrow_y, supersample)

    title_y = y0 + panel_h + int(text_h * 0.5)
    (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.56 * supersample, max(1, int(1.5 * supersample)))
    title_x = x0 + max(0, (w - title_w) // 2)
    draw_text(image, title, (title_x, title_y), scale=0.56 * supersample, color=INK,
              thickness=max(1, int(1.5 * supersample)))


def build_row() -> None:
    supersample = 3
    width, height = 1800, 460
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), WHITE, dtype=np.uint8)

    cell_w = w // 3
    keys = ["bus_car", "horse_giraffe", "mushroom"]
    for i, key in enumerate(keys):
        title, src, fin, az, el, zoom = load_pair(key)
        rect = (i * cell_w, 0, cell_w, h)
        if key == "horse_giraffe":
            render_pair_cell_shared_scale(image, rect, title, src, fin, az, el, zoom, supersample)
        else:
            render_pair_cell(image, rect, title, src, fin, az, el, zoom, supersample)

    save_rgb(OUT / "title_row_3pairs.png", image, supersample, (width, height))
    print("wrote title_row_3pairs.png")


def main() -> None:
    build_row()


if __name__ == "__main__":
    main()
