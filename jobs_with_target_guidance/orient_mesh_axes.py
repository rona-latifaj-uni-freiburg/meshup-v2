"""Create consistently oriented mesh copies.

This is meant for viewer-facing copies of benchmark/output meshes. It applies a
rigid axis permutation/sign flip to OBJ or vertex-colored PLY files without
changing topology or labels. For the TOSCA wolf meshes, the useful transform is
usually:

    --axis-map y,x,z

which moves the animal body axis from viewer-up Y into horizontal X, while
keeping Z as the visual up axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from plyfile import PlyData, PlyElement


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def parse_axis_map(raw: str) -> Tuple[int, int, int]:
    pieces = [piece.strip().lower() for piece in raw.split(",")]
    if len(pieces) != 3 or any(piece not in AXIS_INDEX for piece in pieces):
        raise ValueError("--axis-map must look like y,x,z or x,y,z")
    if len(set(pieces)) != 3:
        raise ValueError("--axis-map must use each axis exactly once.")
    return tuple(AXIS_INDEX[piece] for piece in pieces)  # type: ignore[return-value]


def transform_vertices(vertices: np.ndarray, axis_map: Sequence[int], flips: Sequence[float]) -> np.ndarray:
    out = vertices[:, list(axis_map)].astype(np.float32, copy=True)
    out *= np.asarray(flips, dtype=np.float32)[None, :]
    return out


def load_obj(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with path.open("r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                ids = [int(tok.split("/")[0]) - 1 for tok in parts[1:]]
                for i in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[i], ids[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"Could not load OBJ mesh from {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"# Oriented copy of {source}\n")
        f.write("g default\n")
        for x, y, z in vertices:
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        f.write("s 1\n")
        for a, b, c in faces:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def load_colored_ply(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex_data = ply["vertex"].data
    vertices = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float32)
    colors = np.stack(
        [
            vertex_data["red"],
            vertex_data["green"],
            vertex_data["blue"],
            vertex_data["alpha"] if "alpha" in vertex_data.dtype.names else np.full(len(vertices), 255),
        ],
        axis=1,
    ).astype(np.uint8)
    face_data = ply["face"].data
    faces = np.vstack(face_data["vertex_indices"]).astype(np.int64)
    return vertices, faces, colors


def write_colored_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    vertex_array["alpha"] = colors[:, 3] if colors.shape[1] > 3 else 255

    face_array = np.empty(faces.shape[0], dtype=[("vertex_indices", "O")])
    face_array["vertex_indices"] = [row.astype(np.int32) for row in faces]
    PlyData(
        [
            PlyElement.describe(vertex_array, "vertex"),
            PlyElement.describe(face_array, "face"),
        ],
        text=False,
    ).write(str(path))


def orient_one(input_path: Path, output_path: Path, axis_map: Sequence[int], flips: Sequence[float]) -> Dict[str, object]:
    suffix = input_path.suffix.lower()
    if suffix == ".obj":
        vertices, faces = load_obj(input_path)
        oriented = transform_vertices(vertices, axis_map, flips)
        write_obj(output_path, oriented, faces, input_path)
    elif suffix == ".ply":
        vertices, faces, colors = load_colored_ply(input_path)
        oriented = transform_vertices(vertices, axis_map, flips)
        write_colored_ply(output_path, oriented, faces, colors)
    else:
        raise ValueError(f"Unsupported mesh type: {input_path}")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "input_bbox_min": vertices.min(axis=0).round(6).tolist(),
        "input_bbox_max": vertices.max(axis=0).round(6).tolist(),
        "output_bbox_min": oriented.min(axis=0).round(6).tolist(),
        "output_bbox_max": oriented.max(axis=0).round(6).tolist(),
    }


def paired_inputs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    if args.input:
        if len(args.input) != len(args.output):
            raise ValueError("--input and --output must be repeated the same number of times.")
        pairs.extend((Path(src), Path(dst)) for src, dst in zip(args.input, args.output))

    if args.input_root:
        input_root = Path(args.input_root)
        output_root = Path(args.output_root)
        for pattern in args.glob:
            for src in sorted(input_root.glob(pattern)):
                if src.is_file():
                    pairs.append((src, output_root / src.relative_to(input_root)))

    if not pairs:
        raise ValueError("No inputs provided.")

    seen = set()
    unique_pairs: List[Tuple[Path, Path]] = []
    for src, dst in pairs:
        key = (src.resolve(), dst.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append((src, dst))
    return unique_pairs


def run(args: argparse.Namespace) -> None:
    axis_map = parse_axis_map(args.axis_map)
    flips = [
        -1.0 if args.flip_x else 1.0,
        -1.0 if args.flip_y else 1.0,
        -1.0 if args.flip_z else 1.0,
    ]
    rows = []
    for src, dst in paired_inputs(args):
        rows.append(orient_one(src, dst, axis_map, flips))
        print(f"{src} -> {dst}")

    if args.manifest:
        manifest = {
            "axis_map": args.axis_map,
            "flip_x": bool(args.flip_x),
            "flip_y": bool(args.flip_y),
            "flip_z": bool(args.flip_z),
            "items": rows,
        }
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", help="Input OBJ/PLY. Repeat with --output.")
    parser.add_argument("--output", action="append", default=[], help="Output OBJ/PLY. Repeat with --input.")
    parser.add_argument("--input-root", help="Root folder for bulk conversion.")
    parser.add_argument("--output-root", default="", help="Output root for bulk conversion.")
    parser.add_argument("--glob", action="append", default=["*.obj"], help="Glob under --input-root. Repeatable.")
    parser.add_argument("--axis-map", default="y,x,z", help="Output axes as input axes, e.g. y,x,z.")
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--flip-z", action="store_true")
    parser.add_argument("--manifest", help="Optional JSON manifest output.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
