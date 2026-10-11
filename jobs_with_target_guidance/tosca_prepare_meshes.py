"""Prepare TOSCA benchmark meshes for MeshUp runs.

TOSCA ships meshes as OFF files. MeshUp's optimizer path is OBJ-oriented, so
this script converts selected OFF files to simple OBJ files while preserving
the original vertex order. Preserving vertex order is important because TOSCA
same-class ground truth uses identical vertex ids.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


ArrayPair = Tuple[np.ndarray, np.ndarray]


def read_off(path: Path) -> ArrayPair:
    with path.open("r") as f:
        first = f.readline().strip()
        if first != "OFF":
            raise ValueError(f"{path} is not an OFF file.")

        line = f.readline().strip()
        while not line or line.startswith("#"):
            line = f.readline().strip()
        n_vertices, n_faces, *_ = [int(tok) for tok in line.split()]

        vertices: List[List[float]] = []
        while len(vertices) < n_vertices:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading vertices from {path}.")
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vertices.append([float(tok) for tok in line.split()[:3]])

        faces: List[List[int]] = []
        while len(faces) < n_faces:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading faces from {path}.")
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = [int(tok) for tok in line.split()]
            count, ids = raw[0], raw[1:]
            if count < 3:
                continue
            if len(ids) < count:
                raise ValueError(f"Malformed face in {path}: {line}")
            for i in range(1, count - 1):
                faces.append([ids[0], ids[i], ids[i + 1]])

        return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Converted from TOSCA OFF; vertex order preserved.\n")
        f.write("g default\n")
        for x, y, z in vertices:
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        f.write("s 1\n")
        for a, b, c in faces:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def parse_tosca_name(name: str) -> Tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse TOSCA mesh name {name!r}. Expected e.g. wolf0.")
    return match.group(1), int(match.group(2))


def iter_selected_meshes(mesh_dir: Path, names: Iterable[str] | None, classes: Iterable[str] | None) -> List[Path]:
    if names:
        return [mesh_dir / f"{name}.off" for name in names]

    class_set = set(classes or [])
    paths = []
    for path in sorted(mesh_dir.glob("*.off")):
        cls, _idx = parse_tosca_name(path.stem)
        if not class_set or cls in class_set:
            paths.append(path)
    return paths


def prepare(args: argparse.Namespace) -> None:
    benchmark_root = Path(args.benchmark_root)
    mesh_dir = benchmark_root / "Data" / "TOSCA" / "Meshes"
    output_root = Path(args.output_root)
    obj_dir = output_root / "objs"
    metadata_path = output_root / "metadata.json"

    selected = iter_selected_meshes(mesh_dir, args.names, args.classes)
    if not selected:
        raise ValueError("No TOSCA meshes selected.")

    metadata: Dict[str, object] = {
        "benchmark_root": str(benchmark_root),
        "source_mesh_dir": str(mesh_dir),
        "obj_dir": str(obj_dir),
        "items": {},
    }
    rows = ["name\tclass\tid\tvertices\tfaces\toff\tobj"]

    for off_path in selected:
        if not off_path.is_file():
            raise FileNotFoundError(off_path)
        name = off_path.stem
        cls, idx = parse_tosca_name(name)
        vertices, faces = read_off(off_path)
        obj_path = obj_dir / f"{name}.obj"
        write_obj(obj_path, vertices, faces)

        item = {
            "class": cls,
            "id": idx,
            "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]),
            "off": str(off_path),
            "obj": str(obj_path),
        }
        metadata["items"][name] = item
        rows.append(
            "\t".join(
                [
                    name,
                    cls,
                    str(idx),
                    str(vertices.shape[0]),
                    str(faces.shape[0]),
                    str(off_path),
                    str(obj_path),
                ]
            )
        )
        print(f"{name}: {vertices.shape[0]} vertices, {faces.shape[0]} faces -> {obj_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    (output_root / "inventory.tsv").write_text("\n".join(rows) + "\n")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote inventory: {output_root / 'inventory.tsv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        default="external/TOSCA/CorrsBenchmark",
        help="Path to the Surface Correspondence Benchmark root.",
    )
    parser.add_argument(
        "--output-root",
        default="external/TOSCA/meshup_prepared",
        help="Output directory for OBJ copies and metadata.",
    )
    parser.add_argument("--names", nargs="+", help="Exact TOSCA mesh names, e.g. wolf0 wolf1.")
    parser.add_argument(
        "--classes",
        nargs="+",
        help="TOSCA classes to prepare when --names is omitted, e.g. wolf horse.",
    )
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
