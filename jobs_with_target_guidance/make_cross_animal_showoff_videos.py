"""Render PartField-coloured showreels for completed cross-animal runs."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml
from plyfile import PlyData

from jobs_with_target_guidance.partfield_segment import PALETTE
from jobs_with_target_guidance.render_partfield_turntable_video import (
    face_colors,
    normalize_one,
    normalize_meshes,
    render_mesh_panel,
)


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/meshup_mplconfig")


EPOCH_RE = re.compile(r"mesh_epoch_(\d+)\.ply$")


@dataclass(frozen=True)
class MeshFrame:
    label: str
    epoch: int | None
    path: Path
    vertices: np.ndarray


@dataclass(frozen=True)
class RunAssets:
    run_dir: Path
    title: str
    front_azimuth: float
    frames: list[MeshFrame]
    faces: np.ndarray
    face_rgb: np.ndarray
    color_source: str
    n_buckets: int | None


def load_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    ply = PlyData.read(str(path))
    vertex_data = ply["vertex"].data
    vertices = np.stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]],
        axis=1,
    ).astype(np.float32)
    faces = np.vstack(ply["face"].data["vertex_indices"]).astype(np.int64)

    names = vertex_data.dtype.names or ()
    colors = None
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack(
            [vertex_data["red"], vertex_data["green"], vertex_data["blue"]],
            axis=1,
        ).astype(np.float32) / 255.0
    return vertices, faces, colors


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64)
    colors = PALETTE[labels % len(PALETTE)].copy()
    cycle = (labels // len(PALETTE)).astype(np.uint8)
    if cycle.max(initial=0) > 0:
        colors = np.clip(colors.astype(np.int16) + (cycle[:, None] * 17), 0, 255).astype(np.uint8)
    return colors.astype(np.float32) / 255.0


def labels_to_face_rgb(labels: np.ndarray, faces: np.ndarray, vertex_count: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape == (len(faces),):
        return labels_to_colors(labels)
    if labels.shape != (vertex_count,):
        raise ValueError(
            f"Label array has shape {labels.shape}; expected {vertex_count} vertices "
            f"or {len(faces)} faces."
        )

    face_labels = np.empty(len(faces), dtype=np.int64)
    for idx, tri in enumerate(faces):
        vals, counts = np.unique(labels[tri], return_counts=True)
        face_labels[idx] = vals[np.argmax(counts)]
    return labels_to_colors(face_labels)


def resolve_path(raw: str | os.PathLike[str], run_dir: Path | None = None) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    if run_dir is not None:
        run_path = run_dir / path
        if run_path.exists():
            return run_path
    return cwd_path


def resolve_run_dir(mesh_path: Path) -> Path:
    path = resolve_path(mesh_path)
    candidates = [path if path.is_dir() else path.parent, *path.parents]
    for parent in candidates:
        if (parent / "config.yml").is_file() and (parent / "colored_meshes").is_dir():
            return parent
    raise FileNotFoundError(f"Could not resolve run directory from {mesh_path}")


def read_manifest(path: Path) -> list[Path]:
    out: list[Path] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(Path(line))
    return out


def title_from_run(run_dir: Path) -> str:
    name = run_dir.name
    prefix = name
    for marker in (
        "_sam3d_",
        "_topomatch_",
        "_chain_",
        "_densecorr3d_",
        "_hard_partfield",
    ):
        prefix = prefix.split(marker, 1)[0]
    if "_to_" in prefix:
        source, target = prefix.split("_to_", 1)
        for marker in (
            "_catdach",
            "_srcstable",
            "_requested",
            "_pfonly",
        ):
            target = target.split(marker, 1)[0]
        target = re.sub(r"_(5k|\d+)$", "", target)
        repeated_source = f"_{source}_"
        if repeated_source in target:
            target = target.split(repeated_source, 1)[0]
        return f"{source.replace('_', ' ')} -> {target.replace('_', ' ')}"
    return prefix.replace("_", " ")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return slug or "run"


def epoch_from_path(path: Path) -> int:
    match = EPOCH_RE.match(path.name)
    if match is None:
        raise ValueError(f"Cannot parse epoch from {path}")
    return int(match.group(1))


def label_candidates_from_npz(path: Path) -> tuple[list[tuple[str, np.ndarray]], int | None]:
    z = np.load(path)
    n_buckets = int(np.asarray(z["n_buckets"]).item()) if "n_buckets" in z.files else None
    candidates: list[tuple[str, np.ndarray]] = []
    for key in ("source_labels", "vertex_labels", "labels", "face_labels"):
        if key in z.files:
            candidates.append((key, np.asarray(z[key], dtype=np.int64)))
    return candidates, n_buckets


def source_label_paths(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    local = run_dir / "partfield_labels" / "bucket_labels.npz"
    if local.is_file():
        paths.append(local)

    cfg_path = run_dir / "config.yml"
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        raw_source = cfg.get("partfield_source_labels")
        if raw_source:
            resolved = resolve_path(raw_source, run_dir)
            if resolved.is_file() and resolved not in paths:
                paths.append(resolved)
    return paths


def load_partfield_face_rgb(
    run_dir: Path,
    faces: np.ndarray,
    vertex_count: int,
) -> tuple[np.ndarray | None, str | None, int | None]:
    errors: list[str] = []
    for label_path in source_label_paths(run_dir):
        candidates, n_buckets = label_candidates_from_npz(label_path)
        for key, labels in candidates:
            try:
                face_rgb = labels_to_face_rgb(labels, faces, vertex_count)
            except ValueError as exc:
                errors.append(f"{label_path}:{key}: {exc}")
                continue
            return face_rgb, f"{label_path}:{key}", n_buckets
    if errors:
        print(f"{run_dir}: no compatible PartField labels; " + " | ".join(errors))
    return None, None, None


def infer_front_azimuth(vertices: np.ndarray) -> float:
    """Estimate a front-facing camera azimuth from the higher end of the body axis."""
    verts = np.asarray(vertices, dtype=np.float64)
    xz = verts[:, [0, 2]]
    y = verts[:, 1]
    center_xz = np.median(xz, axis=0)
    centered_xz = xz - center_xz
    cov = np.cov(centered_xz.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    if axis[0] < 0:
        axis = -axis

    proj = centered_xz @ axis
    y_min = float(y.min())
    y_span = max(float(y.max() - y_min), 1e-8)
    y_norm = (y - y_min) / y_span
    scores: list[tuple[float, int]] = []
    for sign in (-1, 1):
        signed = sign * proj
        threshold = np.quantile(signed, 0.82)
        mask = signed >= threshold
        if mask.sum() < 8:
            mask = signed >= np.quantile(signed, 0.70)
        end_y = y_norm[mask]
        upper_fraction = float(np.mean(end_y >= 0.52))
        score = (
            0.42 * float(np.mean(end_y))
            + 0.38 * float(np.percentile(end_y, 85))
            + 0.20 * upper_fraction
        )
        scores.append((score, sign))
    front_sign = max(scores)[1]
    front_mask = front_sign * proj >= np.quantile(front_sign * proj, 0.90)
    if front_mask.sum() < 4:
        front_mask = front_sign * proj >= np.quantile(front_sign * proj, 0.82)
    front_point = xz[front_mask].mean(axis=0)
    front_vector = front_point - center_xz
    if np.linalg.norm(front_vector) < 1e-8:
        front_vector = front_sign * axis

    dx, dz = front_vector
    # transform_vertices() makes a direction face camera when its transformed z is positive.
    return float(np.degrees(np.arctan2(-dx, dz)))


def discover_frames(run_dir: Path) -> list[Path]:
    colored_dir = run_dir / "colored_meshes"
    epoch_paths = sorted(colored_dir.glob("mesh_epoch_*.ply"), key=epoch_from_path)
    final_path = colored_dir / "mesh_final_position.ply"
    if final_path.is_file():
        epoch_paths.append(final_path)
    if not epoch_paths:
        raise FileNotFoundError(f"No mesh_epoch_*.ply files in {colored_dir}")
    return epoch_paths


def load_run_assets(run_dir: Path) -> RunAssets:
    paths = discover_frames(run_dir)
    base_vertices, faces, base_colors = load_ply_mesh(paths[0])
    partfield_face_rgb, color_source, n_buckets = load_partfield_face_rgb(run_dir, faces, len(base_vertices))
    if partfield_face_rgb is None:
        if base_colors is None:
            raise ValueError(f"{paths[0]} has no vertex colours and no compatible labels were found.")
        partfield_face_rgb = face_colors(faces, base_colors)
        color_source = f"{paths[0]}:saved_vertex_colors"

    frames: list[MeshFrame] = []
    for path in paths:
        vertices, frame_faces, _colors = load_ply_mesh(path)
        if frame_faces.shape != faces.shape or not np.array_equal(frame_faces, faces):
            raise ValueError(f"{path}: face topology differs from {paths[0]}")
        if len(vertices) != len(base_vertices):
            raise ValueError(f"{path}: vertex count differs from {paths[0]}")
        if path.name == "mesh_final_position.ply":
            label = "final"
            epoch = None
        else:
            epoch = epoch_from_path(path)
            label = f"epoch {epoch}"
        frames.append(MeshFrame(label=label, epoch=epoch, path=path, vertices=vertices))

    return RunAssets(
        run_dir=run_dir,
        title=title_from_run(run_dir),
        front_azimuth=infer_front_azimuth(base_vertices),
        frames=frames,
        faces=faces,
        face_rgb=partfield_face_rgb,
        color_source=color_source or "unknown",
        n_buckets=n_buckets,
    )


def normalize_sequence(vertices: Iterable[np.ndarray]) -> list[np.ndarray]:
    arrays = [np.asarray(v, dtype=np.float32) for v in vertices]
    combined = np.concatenate(arrays, axis=0)
    center = (combined.min(axis=0) + combined.max(axis=0)) * 0.5
    scale = max(float((combined.max(axis=0) - combined.min(axis=0)).max()), 1e-8)
    return [(v - center) / scale for v in arrays]


def start_final_start_azimuth(assets: RunAssets, args: argparse.Namespace) -> float:
    if args.no_auto_front:
        return float(args.start_azimuth)
    return float(assets.front_azimuth + args.start_final_front_offset)


def deformation_start_azimuth(assets: RunAssets, args: argparse.Namespace) -> float:
    if args.no_auto_front:
        return float(args.deformation_start_azimuth)
    return float(assets.front_azimuth + args.deformation_front_offset)


def draw_header(frame: np.ndarray, title: str, subtitle: str, supersample: int) -> None:
    # Title (pair name) is deliberately not drawn into the frame: the showcase site already
    # shows it as an HTML caption under each video, so burning it into the pixels too was
    # showing the same name twice. Only the bucket-count/epoch subtitle is drawn, promoted up
    # to the title's old baseline.
    if subtitle:
        x = 28 * supersample
        cv2.putText(
            frame,
            subtitle,
            (x, 40 * supersample),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * supersample,
            (85, 91, 102),
            max(1, 1 * supersample),
            cv2.LINE_AA,
        )


def render_start_final_frame(
    assets: RunAssets,
    start_vertices: np.ndarray,
    final_vertices: np.ndarray,
    frame_idx: int,
    frames_per_run: int,
    width: int,
    height: int,
    supersample: int,
    start_azimuth: float,
    elevation: float,
    zoom: float,
) -> np.ndarray:
    ss = int(supersample)
    w = width * ss
    h = height * ss
    frame = np.full((h, w, 3), 246, dtype=np.uint8)
    subtitle = f"{assets.n_buckets} PartField buckets" if assets.n_buckets else "saved mesh colours"
    draw_header(frame, assets.title, subtitle, ss)

    margin = 24 * ss
    gap = 20 * ss
    top = 80 * ss
    bottom = 24 * ss
    panel_w = (w - 2 * margin - gap) // 2
    panel_h = h - top - bottom
    azimuth = start_azimuth + 360.0 * frame_idx / max(frames_per_run, 1)

    render_mesh_panel(
        frame,
        (margin, top, panel_w, panel_h),
        start_vertices,
        assets.faces,
        np.empty((len(start_vertices), 3), dtype=np.float32),
        assets.face_rgb,
        "epoch 1",
        azimuth,
        elevation,
        zoom,
    )
    render_mesh_panel(
        frame,
        (margin + panel_w + gap, top, panel_w, panel_h),
        final_vertices,
        assets.faces,
        np.empty((len(final_vertices), 3), dtype=np.float32),
        assets.face_rgb,
        "final",
        azimuth,
        elevation,
        zoom,
    )

    if ss > 1:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_deformation_frame(
    assets: RunAssets,
    vertices: np.ndarray,
    label: str,
    width: int,
    height: int,
    supersample: int,
    azimuth: float,
    elevation: float,
    zoom: float,
) -> np.ndarray:
    ss = int(supersample)
    w = width * ss
    h = height * ss
    frame = np.full((h, w, 3), 246, dtype=np.uint8)
    subtitle = f"{assets.n_buckets} PartField buckets | {label}" if assets.n_buckets else label
    draw_header(frame, assets.title, subtitle, ss)
    margin = 26 * ss
    top = 80 * ss
    bottom = 26 * ss
    render_mesh_panel(
        frame,
        (margin, top, w - 2 * margin, h - top - bottom),
        vertices,
        assets.faces,
        np.empty((len(vertices), 3), dtype=np.float32),
        assets.face_rgb,
        label,
        azimuth,
        elevation,
        zoom,
    )
    if ss > 1:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def segment_weights(frames: list[MeshFrame], early_epoch: int, early_weight: float) -> list[float]:
    weights: list[float] = []
    for left, right in zip(frames, frames[1:]):
        end_epoch = right.epoch if right.epoch is not None else (left.epoch or early_epoch)
        weights.append(float(early_weight if end_epoch <= early_epoch else 1.0))
    return weights


def allocate_segment_frames(weights: list[float], total_frames: int) -> list[int]:
    if not weights:
        return []
    raw = np.asarray(weights, dtype=np.float64)
    raw = raw / raw.sum() * total_frames
    counts = np.maximum(1, np.floor(raw).astype(int))
    while counts.sum() < total_frames:
        idx = int(np.argmax(raw - counts))
        counts[idx] += 1
    while counts.sum() > total_frames:
        candidates = np.where(counts > 1)[0]
        if not len(candidates):
            break
        idx = int(candidates[np.argmax(counts[candidates] - raw[candidates])])
        counts[idx] -= 1
    return counts.tolist()


def render_start_final_showreel(run_assets: list[RunAssets], args: argparse.Namespace) -> None:
    output = Path(args.start_final_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=args.fps, codec="libx264", bitrate=args.bitrate)
    try:
        for run_idx, assets in enumerate(run_assets, start=1):
            print(f"[start/final {run_idx}/{len(run_assets)}] {assets.title}")
            start = assets.frames[0].vertices
            final = assets.frames[-1].vertices
            start_norm, final_norm = normalize_meshes(start, final, args.start_final_normalize)
            start_azimuth = start_final_start_azimuth(assets, args)
            for frame_idx in range(args.start_final_frames_per_run):
                writer.append_data(
                    render_start_final_frame(
                        assets,
                        start_norm,
                        final_norm,
                        frame_idx,
                        args.start_final_frames_per_run,
                        args.width,
                        args.height,
                        args.supersample,
                        start_azimuth,
                        args.start_final_elevation,
                        args.start_final_zoom,
                    )
                )
    finally:
        writer.close()
    print(f"Wrote {output}")


def interpolated_label(left: MeshFrame, right: MeshFrame, alpha: float) -> str:
    if right.epoch is None:
        return "final"
    if left.epoch is None:
        return right.label
    epoch = int(round((1.0 - alpha) * left.epoch + alpha * right.epoch))
    return f"epoch {epoch}"


def render_deformation_showreel(run_assets: list[RunAssets], args: argparse.Namespace) -> None:
    output = Path(args.deformation_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=args.fps, codec="libx264", bitrate=args.bitrate)
    try:
        for run_idx, assets in enumerate(run_assets, start=1):
            print(f"[deformation {run_idx}/{len(run_assets)}] {assets.title}")
            if args.deformation_normalize == "shared":
                norm_vertices = normalize_sequence(frame.vertices for frame in assets.frames)
            else:
                norm_vertices = [frame.vertices for frame in assets.frames]
            norm_frames = [
                MeshFrame(frame.label, frame.epoch, frame.path, vertices)
                for frame, vertices in zip(assets.frames, norm_vertices)
            ]
            weights = segment_weights(norm_frames, args.early_epoch, args.early_weight)
            counts = allocate_segment_frames(weights, args.deformation_frames_per_run)
            total = max(sum(counts), 1)
            emitted = 0
            start_azimuth = deformation_start_azimuth(assets, args)
            for seg_idx, (left, right, count) in enumerate(zip(norm_frames, norm_frames[1:], counts)):
                for local_idx in range(count):
                    alpha = local_idx / max(count, 1)
                    vertices = (1.0 - alpha) * left.vertices + alpha * right.vertices
                    if args.deformation_normalize == "independent":
                        vertices = normalize_one(vertices)
                    label = interpolated_label(left, right, alpha)
                    progress = emitted / max(total - 1, 1)
                    azimuth = start_azimuth + args.deformation_orbit_degrees * progress
                    writer.append_data(
                        render_deformation_frame(
                            assets,
                            vertices,
                            label,
                            args.width,
                            args.height,
                            args.supersample,
                            azimuth,
                            args.deformation_elevation,
                            args.deformation_zoom,
                        )
                    )
                    emitted += 1
            final = norm_frames[-1]
            final_vertices = final.vertices
            if args.deformation_normalize == "independent":
                final_vertices = normalize_one(final_vertices)
            writer.append_data(
                render_deformation_frame(
                    assets,
                    final_vertices,
                    final.label,
                    args.width,
                    args.height,
                    args.supersample,
                    start_azimuth + args.deformation_orbit_degrees,
                    args.deformation_elevation,
                    args.deformation_zoom,
                )
            )
    finally:
        writer.close()
    print(f"Wrote {output}")


def render_single_start_final_video(
    assets: RunAssets,
    output: Path,
    args: argparse.Namespace,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    start_norm, final_norm = normalize_meshes(
        assets.frames[0].vertices,
        assets.frames[-1].vertices,
        args.start_final_normalize,
    )
    start_azimuth = start_final_start_azimuth(assets, args)
    writer = imageio.get_writer(str(output), fps=args.fps, codec="libx264", bitrate=args.bitrate)
    try:
        for frame_idx in range(args.start_final_frames_per_run):
            writer.append_data(
                render_start_final_frame(
                    assets,
                    start_norm,
                    final_norm,
                    frame_idx,
                    args.start_final_frames_per_run,
                    args.width,
                    args.height,
                    args.supersample,
                    start_azimuth,
                    args.start_final_elevation,
                    args.start_final_zoom,
                )
            )
    finally:
        writer.close()
    print(f"Wrote {output}")


def render_single_deformation_video(
    assets: RunAssets,
    output: Path,
    args: argparse.Namespace,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.deformation_normalize == "shared":
        norm_vertices = normalize_sequence(frame.vertices for frame in assets.frames)
    else:
        norm_vertices = [frame.vertices for frame in assets.frames]
    norm_frames = [
        MeshFrame(frame.label, frame.epoch, frame.path, vertices)
        for frame, vertices in zip(assets.frames, norm_vertices)
    ]
    weights = segment_weights(norm_frames, args.early_epoch, args.early_weight)
    counts = allocate_segment_frames(weights, args.deformation_frames_per_run)
    total = max(sum(counts), 1)
    emitted = 0
    start_azimuth = deformation_start_azimuth(assets, args)

    writer = imageio.get_writer(str(output), fps=args.fps, codec="libx264", bitrate=args.bitrate)
    try:
        for left, right, count in zip(norm_frames, norm_frames[1:], counts):
            for local_idx in range(count):
                alpha = local_idx / max(count, 1)
                vertices = (1.0 - alpha) * left.vertices + alpha * right.vertices
                if args.deformation_normalize == "independent":
                    vertices = normalize_one(vertices)
                label = interpolated_label(left, right, alpha)
                progress = emitted / max(total - 1, 1)
                azimuth = start_azimuth + args.deformation_orbit_degrees * progress
                writer.append_data(
                    render_deformation_frame(
                        assets,
                        vertices,
                        label,
                        args.width,
                        args.height,
                        args.supersample,
                        azimuth,
                        args.deformation_elevation,
                        args.deformation_zoom,
                    )
                )
                emitted += 1

        final = norm_frames[-1]
        final_vertices = final.vertices
        if args.deformation_normalize == "independent":
            final_vertices = normalize_one(final_vertices)
        writer.append_data(
            render_deformation_frame(
                assets,
                final_vertices,
                final.label,
                args.width,
                args.height,
                args.supersample,
                start_azimuth + args.deformation_orbit_degrees,
                args.deformation_elevation,
                args.deformation_zoom,
            )
        )
    finally:
        writer.close()
    print(f"Wrote {output}")


def render_per_run_videos(run_assets: list[RunAssets], args: argparse.Namespace) -> None:
    if args.per_run_output_dir is None:
        return
    output_root = Path(args.per_run_output_dir)
    start_dir = output_root / "start_final"
    deformation_dir = output_root / "deformation_slow"
    outputs = []
    for run_idx, assets in enumerate(run_assets, start=1):
        stem = f"{run_idx:02d}_{slugify(assets.title)}"
        print(f"[per-run {run_idx}/{len(run_assets)}] {assets.title}")
        start_output = start_dir / f"{stem}_start_final_partfield.mp4"
        deformation_output = deformation_dir / f"{stem}_deformation_slow_partfield.mp4"
        if not args.skip_start_final:
            render_single_start_final_video(assets, start_output, args)
        if not args.skip_deformation:
            render_single_deformation_video(assets, deformation_output, args)
        outputs.append(
            {
                "title": assets.title,
                "run_dir": str(assets.run_dir),
                "front_azimuth": assets.front_azimuth,
                "start_final": str(start_output) if not args.skip_start_final else None,
                "deformation_slow": str(deformation_output) if not args.skip_deformation else None,
            }
        )
    index_path = output_root / "per_run_video_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(outputs, indent=2) + "\n")
    print(f"Wrote {index_path}")


def write_summary(path: Path, run_assets: list[RunAssets], input_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_paths": input_count,
        "unique_runs": len(run_assets),
        "runs": [
            {
                "title": assets.title,
                "run_dir": str(assets.run_dir),
                "front_azimuth": assets.front_azimuth,
                "color_source": assets.color_source,
                "n_buckets": assets.n_buckets,
                "checkpoints": [
                    {"label": frame.label, "epoch": frame.epoch, "path": str(frame.path)}
                    for frame in assets.frames
                ],
            }
            for assets in run_assets
        ],
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Text file of pasted mesh paths.")
    parser.add_argument(
        "--start-final-output",
        default="jobs_with_target_guidance/cross_animal_spike_runs/reports/showoff_start_final_partfield_showreel.mp4",
        help="Combined epoch-1/final turntable MP4.",
    )
    parser.add_argument(
        "--deformation-output",
        default="jobs_with_target_guidance/cross_animal_spike_runs/reports/showoff_deformation_slow_partfield_showreel.mp4",
        help="Combined slow deformation MP4.",
    )
    parser.add_argument(
        "--summary-output",
        default="jobs_with_target_guidance/cross_animal_spike_runs/reports/showoff_video_summary.json",
        help="JSON summary of inputs and label sources.",
    )
    parser.add_argument("--per-run-output-dir", type=Path, default=None, help="Directory for separate per-run MP4s.")
    parser.add_argument("--only-per-run", action="store_true", help="Write only separate per-run MP4s.")
    parser.add_argument("--skip-start-final", action="store_true")
    parser.add_argument("--skip-deformation", action="store_true")
    parser.add_argument("--limit-runs", type=int, default=None, help="Debug limit.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--bitrate", default="10M")
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--start-final-frames-per-run", type=int, default=72)
    parser.add_argument("--deformation-frames-per-run", type=int, default=144)
    parser.add_argument("--start-final-normalize", choices=["shared", "independent"], default="independent")
    parser.add_argument("--deformation-normalize", choices=["shared", "independent"], default="independent")
    parser.add_argument("--early-epoch", type=int, default=500)
    parser.add_argument("--early-weight", type=float, default=10.0)
    parser.add_argument("--no-auto-front", action="store_true", help="Use fixed azimuths instead of inferred front views.")
    parser.add_argument("--start-azimuth", type=float, default=-55.0)
    parser.add_argument("--start-final-front-offset", type=float, default=-35.0)
    parser.add_argument("--start-final-elevation", type=float, default=17.0)
    parser.add_argument("--start-final-zoom", type=float, default=0.88)
    parser.add_argument("--deformation-start-azimuth", type=float, default=35.0)
    parser.add_argument("--deformation-front-offset", type=float, default=-35.0)
    parser.add_argument("--deformation-orbit-degrees", type=float, default=115.0)
    parser.add_argument("--deformation-elevation", type=float, default=23.0)
    parser.add_argument("--deformation-zoom", type=float, default=0.92)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = read_manifest(args.manifest)
    seen: set[Path] = set()
    run_dirs: list[Path] = []
    for input_path in input_paths:
        run_dir = resolve_run_dir(input_path).resolve()
        if run_dir not in seen:
            seen.add(run_dir)
            run_dirs.append(run_dir)
    if args.limit_runs is not None:
        run_dirs = run_dirs[: args.limit_runs]

    print(f"Resolved {len(input_paths)} mesh paths to {len(run_dirs)} unique runs.")
    run_assets = [load_run_assets(run_dir) for run_dir in run_dirs]
    write_summary(Path(args.summary_output), run_assets, len(input_paths))

    if args.per_run_output_dir is not None:
        render_per_run_videos(run_assets, args)
    if args.only_per_run:
        return
    if not args.skip_start_final:
        render_start_final_showreel(run_assets, args)
    if not args.skip_deformation:
        render_deformation_showreel(run_assets, args)


if __name__ == "__main__":
    main()
