"""Analyze connected components in MeshUp car meshes.

Exports:
- component_summary.csv / .json with per-mesh component counts and largest part stats
- colored_components/*.ply with largest component green and all smaller components red
- largest_components/*.ply containing only the largest connected component
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from plyfile import PlyData, PlyElement


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[x] != x:
            parent = int(self.parent[x])
            self.parent[x] = root
            x = parent
        return root

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def load_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex_data = ply["vertex"].data
    vertices = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float32)
    faces = np.vstack(ply["face"].data["vertex_indices"]).astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path} must contain triangular faces, got {faces.shape}")
    return vertices, faces


def connected_components(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    uf = UnionFind(vertices.shape[0])
    for face in faces:
        a, b, c = [int(v) for v in face]
        uf.union(a, b)
        uf.union(b, c)

    roots = np.array([uf.find(i) for i in range(vertices.shape[0])], dtype=np.int64)
    unique_roots, vertex_labels = np.unique(roots, return_inverse=True)
    n_components = unique_roots.shape[0]

    face_labels = vertex_labels[faces[:, 0]]
    consistent = (
        (vertex_labels[faces[:, 0]] == vertex_labels[faces[:, 1]])
        & (vertex_labels[faces[:, 0]] == vertex_labels[faces[:, 2]])
    )
    if not np.all(consistent):
        raise RuntimeError("Found faces spanning multiple vertex components; this should not happen.")

    components: List[Dict[str, object]] = []
    for comp_id in range(n_components):
        v_mask = vertex_labels == comp_id
        f_mask = face_labels == comp_id
        comp_vertices = vertices[v_mask]
        if comp_vertices.size:
            bbox_min = comp_vertices.min(axis=0)
            bbox_max = comp_vertices.max(axis=0)
            centroid = comp_vertices.mean(axis=0)
        else:
            bbox_min = bbox_max = centroid = np.zeros(3, dtype=np.float32)
        components.append(
            {
                "component_id": comp_id,
                "root": int(unique_roots[comp_id]),
                "n_vertices": int(v_mask.sum()),
                "n_faces": int(f_mask.sum()),
                "bbox_min": bbox_min.round(6).tolist(),
                "bbox_max": bbox_max.round(6).tolist(),
                "bbox_extent": (bbox_max - bbox_min).round(6).tolist(),
                "centroid": centroid.round(6).tolist(),
            }
        )
    components.sort(key=lambda item: (int(item["n_faces"]), int(item["n_vertices"])), reverse=True)
    return vertex_labels, components


def write_colored_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, vertex_labels: np.ndarray, largest_label: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.zeros((vertices.shape[0], 3), dtype=np.uint8)
    colors[:] = np.array([210, 60, 50], dtype=np.uint8)
    colors[vertex_labels == largest_label] = np.array([30, 170, 80], dtype=np.uint8)

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
    PlyData([PlyElement.describe(vertex_array, "vertex"), PlyElement.describe(face_array, "face")], text=False).write(str(path))


def write_largest_component_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, vertex_labels: np.ndarray, largest_label: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep_vertices = vertex_labels == largest_label
    old_to_new = np.full(vertices.shape[0], -1, dtype=np.int64)
    old_to_new[np.where(keep_vertices)[0]] = np.arange(int(keep_vertices.sum()), dtype=np.int64)

    keep_faces = keep_vertices[faces].all(axis=1)
    new_vertices = vertices[keep_vertices]
    new_faces = old_to_new[faces[keep_faces]]

    vertex_array = np.empty(new_vertices.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_array["x"] = new_vertices[:, 0]
    vertex_array["y"] = new_vertices[:, 1]
    vertex_array["z"] = new_vertices[:, 2]
    face_array = np.empty(new_faces.shape[0], dtype=[("vertex_indices", "O")])
    face_array["vertex_indices"] = [row for row in new_faces.astype(np.int32)]
    PlyData([PlyElement.describe(vertex_array, "vertex"), PlyElement.describe(face_array, "face")], text=False).write(str(path))


def analyze_mesh(path: Path, output_dir: Path) -> Dict[str, object]:
    vertices, faces = load_ply(path)
    vertex_labels, components = connected_components(vertices, faces)
    largest = components[0]
    largest_label = int(largest["component_id"])

    stem = path.stem
    write_colored_ply(output_dir / "colored_components" / f"{stem}_largest_green.ply", vertices, faces, vertex_labels, largest_label)
    write_largest_component_ply(output_dir / "largest_components" / f"{stem}_largest_component.ply", vertices, faces, vertex_labels, largest_label)

    total_vertices = int(vertices.shape[0])
    total_faces = int(faces.shape[0])
    small_faces = total_faces - int(largest["n_faces"])
    return {
        "mesh": str(path),
        "name": stem,
        "n_components": len(components),
        "total_vertices": total_vertices,
        "total_faces": total_faces,
        "largest_vertices": int(largest["n_vertices"]),
        "largest_faces": int(largest["n_faces"]),
        "largest_vertex_pct": round(100.0 * int(largest["n_vertices"]) / max(total_vertices, 1), 3),
        "largest_face_pct": round(100.0 * int(largest["n_faces"]) / max(total_faces, 1), 3),
        "small_component_faces": small_faces,
        "small_component_face_pct": round(100.0 * small_faces / max(total_faces, 1), 3),
        "largest_bbox_extent": largest["bbox_extent"],
        "largest_centroid": largest["centroid"],
        "components": components,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", required=True, help="Directory containing PLY meshes.")
    parser.add_argument("--glob", default="*.ply", help="Mesh glob inside --mesh-dir.")
    parser.add_argument("--output-dir", required=True, help="Output directory for CSV/JSON and visual PLYs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_dir = Path(args.mesh_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [analyze_mesh(path, output_dir) for path in sorted(mesh_dir.glob(args.glob))]
    rows.sort(key=lambda row: row["name"])

    csv_path = output_dir / "component_summary.csv"
    fields = [
        "name",
        "n_components",
        "total_vertices",
        "total_faces",
        "largest_vertices",
        "largest_faces",
        "largest_vertex_pct",
        "largest_face_pct",
        "small_component_faces",
        "small_component_face_pct",
        "largest_bbox_extent",
        "largest_centroid",
        "mesh",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    json_path = output_dir / "component_summary.json"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote colored PLYs to {output_dir / 'colored_components'}")
    print(f"Wrote largest-component PLYs to {output_dir / 'largest_components'}")
    print()
    print("name,n_components,largest_face_pct,small_component_faces")
    for row in rows:
        print(f"{row['name']},{row['n_components']},{row['largest_face_pct']},{row['small_component_faces']}")


if __name__ == "__main__":
    main()
