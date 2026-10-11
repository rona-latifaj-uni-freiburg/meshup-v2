"""Render a static contact sheet for plain OBJ meshes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import imageio.v2 as imageio
import numpy as np

from jobs_with_target_guidance.orient_mesh_axes import load_obj
from jobs_with_target_guidance.render_partfield_turntable_video import (
    normalize_one,
    render_mesh_panel,
)


PALETTE = np.asarray(
    [
        [0.16, 0.45, 0.78],
        [0.86, 0.36, 0.20],
        [0.36, 0.62, 0.24],
        [0.62, 0.42, 0.75],
        [0.74, 0.54, 0.28],
        [0.38, 0.62, 0.68],
    ],
    dtype=np.float32,
)


def render_contact_sheet(args: argparse.Namespace) -> None:
    mesh_paths = [Path(item) for item in args.mesh]
    for mesh_path in mesh_paths:
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)

    titles = args.title_label or [path.stem for path in mesh_paths]
    if len(titles) != len(mesh_paths):
        raise ValueError("--title-label must be repeated once per --mesh.")

    supersample = int(args.supersample)
    width = int(args.width) * supersample
    height = int(args.height) * supersample
    margin = 24 * supersample
    gap = 14 * supersample
    header = 64 * supersample
    panel_w = (width - 2 * margin - gap * (len(mesh_paths) - 1)) // len(mesh_paths)
    panel_h = height - header - margin

    frame = np.full((height, width, 3), 246, dtype=np.uint8)
    cv2.putText(
        frame,
        args.heading,
        (margin, 40 * supersample),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78 * supersample,
        (32, 37, 45),
        max(1, 2 * supersample),
        cv2.LINE_AA,
    )

    for idx, (mesh_path, title) in enumerate(zip(mesh_paths, titles)):
        vertices, faces = load_obj(mesh_path)
        vertices = normalize_one(vertices)
        color = PALETTE[idx % len(PALETTE)]
        vertex_colors = np.repeat(color[None, :], vertices.shape[0], axis=0)
        face_rgb = np.repeat(color[None, :], faces.shape[0], axis=0)
        rect_x = margin + idx * (panel_w + gap)
        render_mesh_panel(
            frame,
            (rect_x, header, panel_w, panel_h),
            vertices,
            faces,
            vertex_colors,
            face_rgb,
            title,
            args.azimuth,
            args.elevation,
            args.zoom,
        )

    if supersample > 1:
        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", action="append", required=True, help="OBJ mesh to render.")
    parser.add_argument("--title-label", action="append", help="Panel label. Repeat once per mesh.")
    parser.add_argument("--output", required=True, help="Output PNG.")
    parser.add_argument("--heading", default="OBJ mesh orientation check")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--azimuth", type=float, default=-55.0)
    parser.add_argument("--elevation", type=float, default=18.0)
    parser.add_argument("--zoom", type=float, default=1.05)
    parser.add_argument("--supersample", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    render_contact_sheet(parse_args())


if __name__ == "__main__":
    main()
