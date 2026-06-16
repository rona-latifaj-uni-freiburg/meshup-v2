"""Color target-guidance output meshes with saved PartField bucket labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pymeshlab
import yaml
from plyfile import PlyData, PlyElement

from jobs_with_target_guidance.partfield_segment import PALETTE


def load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    mesh = ms.current_mesh()
    vertices = np.asarray(mesh.vertex_matrix(), dtype=np.float32)
    faces = np.asarray(mesh.face_matrix(), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{path} did not load as an Nx3 vertex array.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path} must contain triangular faces, got {faces.shape}.")
    return vertices, faces


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64)
    colors = PALETTE[labels % len(PALETTE)].copy()
    cycle = (labels // len(PALETTE)).astype(np.uint8)
    if cycle.max(initial=0) > 0:
        colors = np.clip(colors.astype(np.int16) + (cycle[:, None] * 17), 0, 255).astype(np.uint8)
    return colors


def write_vertex_colored_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray) -> None:
    if vertices.shape[0] != labels.shape[0]:
        raise ValueError(f"{path}: {vertices.shape[0]} vertices but {labels.shape[0]} labels.")

    path.parent.mkdir(parents=True, exist_ok=True)
    colors = labels_to_colors(labels)

    vertex_array = np.empty(
        vertices.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ],
    )
    vertex_array["x"] = vertices[:, 0]
    vertex_array["y"] = vertices[:, 1]
    vertex_array["z"] = vertices[:, 2]
    vertex_array["red"] = colors[:, 0]
    vertex_array["green"] = colors[:, 1]
    vertex_array["blue"] = colors[:, 2]
    vertex_array["alpha"] = 255

    face_array = np.empty(faces.shape[0], dtype=[("vertex_indices", "O")])
    face_array["vertex_indices"] = [row for row in faces.astype(np.int32)]
    PlyData(
        [
            PlyElement.describe(vertex_array, "vertex"),
            PlyElement.describe(face_array, "face"),
        ],
        text=False,
    ).write(str(path))


def scale_token(scale: int) -> str:
    return f"{int(scale):02d}"


def color_run(run_dir: Path, scales: Iterable[int], include_target: bool) -> dict:
    cfg_path = run_dir / "config.yml"
    final_mesh_path = run_dir / "mesh_final" / "mesh.obj"
    labels_dir = run_dir / "partfield_labels"
    output_dir = run_dir / "partfield_colored_meshes"

    if not cfg_path.is_file() or not final_mesh_path.is_file() or not labels_dir.is_dir():
        return {"run": str(run_dir), "skipped": True, "reason": "missing config, final mesh, or labels"}

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    source_mesh_path = Path(cfg["mesh"])
    target_mesh_path = Path(cfg["target_mesh"])
    source_vertices, source_faces = load_mesh(source_mesh_path)
    final_vertices, final_faces = load_mesh(final_mesh_path)
    target_vertices, target_faces = load_mesh(target_mesh_path)

    written = []
    for scale in scales:
        token = scale_token(scale)
        labels_path = labels_dir / f"bucket_labels_{token}.npz"
        if not labels_path.is_file():
            single_scale_path = labels_dir / "bucket_labels.npz"
            if single_scale_path.is_file():
                single_labels = np.load(single_scale_path)
                single_n_buckets = int(np.asarray(single_labels["n_buckets"]).item())
                if single_n_buckets == int(scale):
                    labels_path = single_scale_path
        if not labels_path.is_file():
            raise FileNotFoundError(f"Missing PartField label file: {labels_path}")

        labels = np.load(labels_path)
        source_labels = np.asarray(labels["source_labels"], dtype=np.int64)
        target_labels = np.asarray(labels["target_labels"], dtype=np.int64)

        input_out = output_dir / f"input_partfield_{token}.ply"
        final_out = output_dir / f"final_partfield_{token}.ply"
        write_vertex_colored_ply(input_out, source_vertices, source_faces, source_labels)
        write_vertex_colored_ply(final_out, final_vertices, final_faces, source_labels)
        written.extend([str(input_out), str(final_out)])

        if include_target:
            target_out = output_dir / f"target_partfield_{token}.ply"
            write_vertex_colored_ply(target_out, target_vertices, target_faces, target_labels)
            written.append(str(target_out))

    summary = {
        "run": str(run_dir),
        "source_mesh": str(source_mesh_path),
        "target_mesh": str(target_mesh_path),
        "final_mesh": str(final_mesh_path),
        "scales": [int(s) for s in scales],
        "include_target": bool(include_target),
        "written": written,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        default="jobs_with_target_guidance/outputs_new_cars",
        help="Directory containing target-guidance run output folders.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        help="Specific run directory to color. Can be repeated. Overrides --outputs-dir discovery.",
    )
    parser.add_argument("--scales", type=int, nargs="+", default=[8, 12, 20])
    parser.add_argument("--no-target", action="store_true", help="Do not also write target reference colored meshes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir:
        run_dirs = [Path(path) for path in args.run_dir]
    else:
        outputs_dir = Path(args.outputs_dir)
        run_dirs = sorted(path for path in outputs_dir.iterdir() if path.is_dir())
    results = [color_run(path, args.scales, include_target=not args.no_target) for path in run_dirs]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
