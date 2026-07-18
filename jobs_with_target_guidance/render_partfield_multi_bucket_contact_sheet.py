"""Render static contact sheets for multiple colored PartField meshes."""

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
    normalize_one,
    render_mesh_panel,
)


@dataclass(frozen=True)
class BucketCase:
    label: str
    meshes: List[Path]


def parse_cases(raw_cases: List[List[str]]) -> List[BucketCase]:
    cases: List[BucketCase] = []
    expected_count = None
    for raw in raw_cases:
        if len(raw) < 2:
            raise ValueError("--case expects: LABEL COLORED_PLY [COLORED_PLY ...]")
        label = raw[0]
        mesh_paths = [Path(item) for item in raw[1:]]
        for mesh_path in mesh_paths:
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Missing colored mesh: {mesh_path}")
        if expected_count is None:
            expected_count = len(mesh_paths)
        elif len(mesh_paths) != expected_count:
            raise ValueError("Every --case row must contain the same number of meshes.")
        cases.append(BucketCase(label=label, meshes=mesh_paths))
    return cases


def normalize_many(vertices_list: List[np.ndarray], mode: str) -> List[np.ndarray]:
    if mode == "independent":
        return [normalize_one(vertices) for vertices in vertices_list]

    combined = np.concatenate(vertices_list, axis=0)
    center = (combined.min(axis=0) + combined.max(axis=0)) * 0.5
    scale = max(float((combined.max(axis=0) - combined.min(axis=0)).max()), 1e-8)
    return [(vertices - center) / scale for vertices in vertices_list]


def render_contact_sheet(args: argparse.Namespace) -> None:
    cases = parse_cases(args.case)
    n_cols = len(cases[0].meshes)
    column_titles = args.column_title or [path.stem for path in cases[0].meshes]
    if len(column_titles) != n_cols:
        raise ValueError(f"Expected {n_cols} --column-title values, got {len(column_titles)}.")

    supersample = int(args.supersample)
    width = int(args.width) * supersample
    row_height = int(args.row_height) * supersample
    header_height = 64 * supersample
    margin = 24 * supersample
    gap = 14 * supersample
    label_width = 132 * supersample
    height = header_height + row_height * len(cases) + margin

    frame = np.full((height, width, 3), 246, dtype=np.uint8)
    cv2.putText(
        frame,
        args.title,
        (margin, 40 * supersample),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78 * supersample,
        (32, 37, 45),
        max(1, 2 * supersample),
        cv2.LINE_AA,
    )

    panel_w = (width - (2 * margin) - label_width - gap * (n_cols - 1)) // n_cols
    panel_h = row_height - gap
    if panel_w < 90 * supersample:
        raise ValueError(f"Panel width is too small ({panel_w / supersample:.1f}px). Increase --width.")

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

        loaded = [load_colored_ply(mesh_path) for mesh_path in bucket_case.meshes]
        vertices_list = normalize_many([item[0] for item in loaded], args.normalize)
        rect_y = y0 + 8 * supersample

        for col_idx, (vertices, (_, faces, colors), title) in enumerate(
            zip(vertices_list, loaded, column_titles)
        ):
            face_rgb = face_colors(faces, colors)
            rect_x = margin + label_width + col_idx * (panel_w + gap)
            render_mesh_panel(
                frame,
                (rect_x, rect_y, panel_w, panel_h),
                vertices,
                faces,
                colors,
                face_rgb,
                title,
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
        nargs="+",
        metavar=("LABEL", "COLORED_PLY"),
        required=True,
        help="One bucket row: LABEL followed by one colored PLY per column.",
    )
    parser.add_argument("--column-title", action="append", help="Column title. Repeat once per mesh.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--title", default="PartField bucket comparison")
    parser.add_argument("--width", type=int, default=2600)
    parser.add_argument("--row-height", type=int, default=360)
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
