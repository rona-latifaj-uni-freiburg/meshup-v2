"""Render synchronized side-by-side turntable videos for colored PartField meshes.

The SAM3D car meshes in this project use ``y`` as the vertical axis. This
renderer keeps that axis upright and rotates meshes around it, so the wheels
stay down while the car turns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import imageio.v2 as imageio
import cv2
import numpy as np
from plyfile import PlyData


def load_colored_ply(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex_data = ply["vertex"].data
    vertices = np.stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]],
        axis=1,
    ).astype(np.float32)
    faces = np.vstack(ply["face"].data["vertex_indices"]).astype(np.int64)
    if not {"red", "green", "blue"}.issubset(vertex_data.dtype.names or ()):
        raise ValueError(f"{path} has no red/green/blue vertex color fields.")
    colors = np.stack(
        [vertex_data["red"], vertex_data["green"], vertex_data["blue"]],
        axis=1,
    ).astype(np.float32) / 255.0
    return vertices, faces, colors


def face_colors(faces: np.ndarray, vertex_colors: np.ndarray) -> np.ndarray:
    tri_colors = vertex_colors[faces]
    out = np.empty((faces.shape[0], 3), dtype=np.float32)
    for idx, colors in enumerate(tri_colors):
        unique, counts = np.unique((colors * 255).round().astype(np.uint8), axis=0, return_counts=True)
        out[idx] = unique[np.argmax(counts)].astype(np.float32) / 255.0
    return out


def normalize_meshes(
    left_vertices: np.ndarray,
    right_vertices: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "independent":
        return normalize_one(left_vertices), normalize_one(right_vertices)

    combined = np.concatenate([left_vertices, right_vertices], axis=0)
    center = (combined.min(axis=0) + combined.max(axis=0)) * 0.5
    scale = max(float((combined.max(axis=0) - combined.min(axis=0)).max()), 1e-8)
    return (left_vertices - center) / scale, (right_vertices - center) / scale


def normalize_one(vertices: np.ndarray) -> np.ndarray:
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    scale = max(float((vertices.max(axis=0) - vertices.min(axis=0)).max()), 1e-8)
    return (vertices - center) / scale


def transform_vertices(
    vertices: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """Yaw around y-up, then pitch without roll so the mesh never flips."""
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    ce, se = np.cos(elevation), np.sin(elevation)

    x = ca * vertices[:, 0] + sa * vertices[:, 2]
    z = -sa * vertices[:, 0] + ca * vertices[:, 2]
    y = vertices[:, 1]

    y_pitched = ce * y - se * z
    z_pitched = se * y + ce * z
    return np.stack([x, y_pitched, z_pitched], axis=1)


def render_mesh_panel(
    image: np.ndarray,
    rect: Tuple[int, int, int, int],
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    face_rgb: np.ndarray,
    title: str,
    azimuth: float,
    elevation: float,
    zoom: float,
) -> None:
    x0, y0, width, height = rect
    cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height), (255, 255, 255), -1)
    cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height), (226, 230, 235), 1)

    title_pos = (x0 + 28, y0 + 38)
    cv2.putText(image, title, title_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.88, (34, 39, 47), 2, cv2.LINE_AA)

    center_x = x0 + width * 0.5
    center_y = y0 + height * 0.55
    scale = min(width, height) * float(zoom)
    transformed = transform_vertices(vertices, azimuth, elevation)
    projected = np.empty((vertices.shape[0], 2), dtype=np.float32)
    projected[:, 0] = center_x + transformed[:, 0] * scale
    projected[:, 1] = center_y - transformed[:, 1] * scale

    # Soft ground shadow, slightly below the mesh footprint.
    shadow_y = int(center_y + height * 0.27)
    shadow_w = int(width * 0.32)
    shadow_h = int(max(10, height * 0.035))
    cv2.ellipse(
        image,
        (int(center_x), shadow_y),
        (shadow_w, shadow_h),
        0,
        0,
        360,
        (224, 227, 231),
        -1,
        cv2.LINE_AA,
    )

    tri_depth = transformed[faces, 2].mean(axis=1)
    order = np.argsort(tri_depth)
    tri_vertices = transformed[faces]
    normals = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    light = np.array([0.22, 0.58, 0.78], dtype=np.float32)
    light /= np.linalg.norm(light)
    brightness = 0.72 + 0.28 * np.maximum(np.abs(normals @ light), 0.0)

    for face_idx in order:
        pts = np.round(projected[faces[face_idx]]).astype(np.int32)
        if cv2.contourArea(pts) < 0.2:
            continue
        color = np.clip(face_rgb[face_idx] * brightness[face_idx] * 255.0, 0, 255).astype(np.uint8)
        # OpenCV expects BGR.
        cv2.fillConvexPoly(image, pts, (int(color[2]), int(color[1]), int(color[0])), lineType=cv2.LINE_AA)


def render_frame(
    left_vertices: np.ndarray,
    left_faces: np.ndarray,
    left_colors: np.ndarray,
    left_face_rgb: np.ndarray,
    right_vertices: np.ndarray,
    right_faces: np.ndarray,
    right_colors: np.ndarray,
    right_face_rgb: np.ndarray,
    args: argparse.Namespace,
    frame_idx: int,
) -> np.ndarray:
    super_sample = int(args.supersample)
    width = int(args.width) * super_sample
    height = int(args.height) * super_sample
    frame = np.full((height, width, 3), 246, dtype=np.uint8)

    margin = 24 * super_sample
    gap = 20 * super_sample
    top = 24 * super_sample
    bottom = 24 * super_sample
    panel_w = (width - 2 * margin - gap) // 2
    panel_h = height - top - bottom
    left_rect = (margin, top, panel_w, panel_h)
    right_rect = (margin + panel_w + gap, top, panel_w, panel_h)

    azimuth = args.start_azimuth + 360.0 * frame_idx / args.frames
    render_mesh_panel(
        frame,
        left_rect,
        left_vertices,
        left_faces,
        left_colors,
        left_face_rgb,
        args.left_title,
        azimuth,
        args.elevation,
        args.zoom,
    )
    render_mesh_panel(
        frame,
        right_rect,
        right_vertices,
        right_faces,
        right_colors,
        right_face_rgb,
        args.right_title,
        azimuth,
        args.elevation,
        args.zoom,
    )

    if super_sample > 1:
        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_video(args: argparse.Namespace) -> None:
    left_path = Path(args.left)
    right_path = Path(args.right)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    left_vertices, left_faces, left_colors = load_colored_ply(left_path)
    right_vertices, right_faces, right_colors = load_colored_ply(right_path)
    left_vertices, right_vertices = normalize_meshes(left_vertices, right_vertices, args.normalize)
    left_face_rgb = face_colors(left_faces, left_colors)
    right_face_rgb = face_colors(right_faces, right_colors)

    writer = imageio.get_writer(str(output_path), fps=args.fps, codec="libx264", bitrate=args.bitrate)
    try:
        for frame_idx in range(args.frames):
            writer.append_data(
                render_frame(
                    left_vertices,
                    left_faces,
                    left_colors,
                    left_face_rgb,
                    right_vertices,
                    right_faces,
                    right_colors,
                    right_face_rgb,
                    args,
                    frame_idx,
                )
            )
    finally:
        writer.close()

    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="Left colored PLY mesh.")
    parser.add_argument("--right", required=True, help="Right colored PLY mesh.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--left-title", default="Final", help="Title above the left mesh.")
    parser.add_argument("--right-title", default="Target", help="Title above the right mesh.")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--elevation", type=float, default=11.0, help="Camera pitch in degrees; wheels stay down.")
    parser.add_argument("--start-azimuth", type=float, default=-35.0)
    parser.add_argument("--zoom", type=float, default=0.94, help="Larger values make the meshes fill more of each panel.")
    parser.add_argument("--supersample", type=int, default=2, help="Render scale for antialiasing.")
    parser.add_argument("--bitrate", default="10M")
    parser.add_argument(
        "--normalize",
        choices=["shared", "independent"],
        default="independent",
        help="Use shared scale or center/scale each mesh separately.",
    )
    return parser.parse_args()


def main() -> None:
    render_video(parse_args())


if __name__ == "__main__":
    main()
