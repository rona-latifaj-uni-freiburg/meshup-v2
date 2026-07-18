"""Render a static contact sheet comparing PartField bucket counts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import imageio.v2 as imageio
import numpy as np

from jobs_with_target_guidance.render_partfield_turntable_video import (
    face_colors,
    load_colored_ply,
    normalize_meshes,
    render_mesh_panel,
)


@dataclass(frozen=True)
class BucketCase:
    label: str
    left: Path
    right: Path


def parse_cases(raw_cases: List[List[str]]) -> List[BucketCase]:
    cases = []
    for raw in raw_cases:
        if len(raw) != 3:
            raise ValueError("--case expects: LABEL LEFT_PLY RIGHT_PLY")
        label, left, right = raw
        left_path = Path(left)
        right_path = Path(right)
        if not left_path.is_file():
            raise FileNotFoundError(f"Missing left colored mesh: {left_path}")
        if not right_path.is_file():
            raise FileNotFoundError(f"Missing right colored mesh: {right_path}")
        cases.append(BucketCase(label=label, left=left_path, right=right_path))
    return cases


def render_contact_sheet(args: argparse.Namespace) -> None:
    cases = parse_cases(args.case)
    supersample = int(args.supersample)
    width = int(args.width) * supersample
    row_height = int(args.row_height) * supersample
    header_height = 54 * supersample
    margin = 24 * supersample
    gap = 18 * supersample
    label_width = 132 * supersample
    height = header_height + row_height * len(cases) + margin

    frame = np.full((height, width, 3), 246, dtype=np.uint8)
    cv2.putText(
        frame,
        args.title,
        (margin, 35 * supersample),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78 * supersample,
        (32, 37, 45),
        max(1, 2 * supersample),
        cv2.LINE_AA,
    )

    panel_w = (width - (2 * margin) - label_width - gap) // 2
    panel_h = row_height - gap
    left_x = margin + label_width
    right_x = left_x + panel_w + gap

    for row_idx, bucket_case in enumerate(cases):
        y0 = header_height + row_idx * row_height
        label_y = y0 + 42 * supersample
        cv2.putText(
            frame,
            f"{bucket_case.label} buckets",
            (margin, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64 * supersample,
            (45, 51, 61),
            max(1, 2 * supersample),
            cv2.LINE_AA,
        )

        left_vertices, left_faces, left_colors = load_colored_ply(bucket_case.left)
        right_vertices, right_faces, right_colors = load_colored_ply(bucket_case.right)
        left_vertices, right_vertices = normalize_meshes(left_vertices, right_vertices, args.normalize)
        left_face_rgb = face_colors(left_faces, left_colors)
        right_face_rgb = face_colors(right_faces, right_colors)

        rect_y = y0 + 8 * supersample
        render_mesh_panel(
            frame,
            (left_x, rect_y, panel_w, panel_h),
            left_vertices,
            left_faces,
            left_colors,
            left_face_rgb,
            args.left_title,
            args.azimuth,
            args.elevation,
            args.zoom,
        )
        render_mesh_panel(
            frame,
            (right_x, rect_y, panel_w, panel_h),
            right_vertices,
            right_faces,
            right_colors,
            right_face_rgb,
            args.right_title,
            args.azimuth,
            args.elevation,
            args.zoom,
        )

    if supersample > 1:
        frame = cv2.resize(frame, (args.width, height // supersample), interpolation=cv2.INTER_AREA)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("LABEL", "LEFT_PLY", "RIGHT_PLY"),
        required=True,
        help="One bucket comparison row. Repeat for each bucket count.",
    )
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--title", default="PartField bucket comparison")
    parser.add_argument("--left-title", default="bear")
    parser.add_argument("--right-title", default="bear2")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--row-height", type=int, default=470)
    parser.add_argument("--azimuth", type=float, default=-35.0)
    parser.add_argument("--elevation", type=float, default=11.0)
    parser.add_argument("--zoom", type=float, default=0.90)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument(
        "--normalize",
        choices=["shared", "independent"],
        default="independent",
        help="Use shared scale or center/scale each mesh separately.",
    )
    return parser.parse_args()


def main() -> None:
    render_contact_sheet(parse_args())


if __name__ == "__main__":
    main()
