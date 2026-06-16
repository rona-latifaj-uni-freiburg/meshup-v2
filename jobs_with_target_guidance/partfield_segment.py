"""Co-segment meshes from extracted PartField features.

This is the lightweight MeshUp-side step after running the official NVIDIA
PartField inference. PartField produces anonymous feature vectors; this script
clusters those vectors jointly across shapes so bucket ids are consistent enough
to use as source/target part labels for PartField Chamfer guidance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import KMeans


FeatureMode = Literal["auto", "vertex", "face"]


PALETTE = np.array(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [174, 199, 232],
        [255, 187, 120],
        [152, 223, 138],
        [255, 152, 150],
        [197, 176, 213],
        [196, 156, 148],
        [247, 182, 210],
        [199, 199, 199],
        [219, 219, 141],
        [158, 218, 229],
    ],
    dtype=np.uint8,
)


@dataclass
class MeshData:
    name: str
    mesh_path: Path
    feature_path: Path
    vertices: np.ndarray
    faces: np.ndarray
    raw_features: np.ndarray
    feature_mode: Literal["vertex", "face"]
    cluster_features: np.ndarray
    labels: Optional[np.ndarray] = None


def load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return load_ply_mesh(path)
    if suffix == ".obj":
        return load_obj_mesh(path)
    raise ValueError(f"Unsupported mesh format for {path}. Expected .ply or .obj.")


def load_ply_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex_data = ply["vertex"].data
    vertices = np.stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]],
        axis=1,
    ).astype(np.float32)

    face_data = ply["face"].data
    faces = np.vstack(face_data["vertex_indices"]).astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path} must contain triangular faces, got shape {faces.shape}.")
    return vertices, faces


def load_obj_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                raw = [token.split("/")[0] for token in line.split()[1:]]
                idx = [int(token) - 1 for token in raw]
                if len(idx) < 3:
                    continue
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"Could not load vertices/faces from {path}.")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def load_feature_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        for key in (
            "features",
            "part_features",
            "part_feat",
            "point_feat",
            "vertex_features",
            "face_features",
        ):
            if key in data:
                return np.asarray(data[key], dtype=np.float32).reshape(data[key].shape[0], -1)
        first_key = data.files[0]
        return np.asarray(data[first_key], dtype=np.float32).reshape(data[first_key].shape[0], -1)

    features = np.load(path)
    if features.ndim == 1:
        raise ValueError(f"PartField features must be 2D, got {features.shape} from {path}.")
    return np.asarray(features, dtype=np.float32).reshape(features.shape[0], -1)


def infer_feature_mode(
    features: np.ndarray,
    n_vertices: int,
    n_faces: int,
    requested: FeatureMode,
    feature_path: Path,
) -> Literal["vertex", "face"]:
    if requested == "vertex":
        expected = n_vertices
    elif requested == "face":
        expected = n_faces
    else:
        if features.shape[0] == n_faces:
            return "face"
        if features.shape[0] == n_vertices:
            return "vertex"
        raise ValueError(
            f"Cannot infer feature mode for {feature_path}: {features.shape[0]} feature rows, "
            f"but mesh has {n_vertices} vertices and {n_faces} faces."
        )

    if features.shape[0] != expected:
        raise ValueError(
            f"Feature mode {requested!r} for {feature_path} expects {expected} rows, "
            f"got {features.shape[0]}."
        )
    return requested


def normalize_rows(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    features = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def bbox_coordinates(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    return (vertices - vmin) / np.maximum(vmax - vmin, 1e-8)


def face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertices[faces].mean(axis=1)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return normalize_rows(normals)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    f_normals = face_normals(vertices, faces)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], f_normals)
    return normalize_rows(normals)


def build_cluster_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    partfield_features: np.ndarray,
    mode: Literal["vertex", "face"],
    position_weight: float,
    normal_weight: float,
) -> np.ndarray:
    pieces = [normalize_rows(partfield_features)]

    if position_weight > 0:
        if mode == "face":
            positions = bbox_coordinates(face_centroids(vertices, faces))
        else:
            positions = bbox_coordinates(vertices)
        pieces.append(positions.astype(np.float32) * float(position_weight))

    if normal_weight > 0:
        normals = face_normals(vertices, faces) if mode == "face" else vertex_normals(vertices, faces)
        pieces.append(normals.astype(np.float32) * float(normal_weight))

    return np.concatenate(pieces, axis=1).astype(np.float32)


def labels_to_vertex_labels(labels: np.ndarray, faces: np.ndarray, n_vertices: int) -> np.ndarray:
    votes: List[Dict[int, int]] = [dict() for _ in range(n_vertices)]
    for face_idx, face in enumerate(faces):
        label = int(labels[face_idx])
        for vertex_id in face:
            bucket = votes[int(vertex_id)]
            bucket[label] = bucket.get(label, 0) + 1

    vertex_labels = np.full(n_vertices, -1, dtype=np.int64)
    for vertex_id, bucket in enumerate(votes):
        if bucket:
            vertex_labels[vertex_id] = max(bucket.items(), key=lambda item: item[1])[0]
    vertex_labels[vertex_labels < 0] = 0
    return vertex_labels


def labels_to_face_labels(labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_labels = np.empty(faces.shape[0], dtype=np.int64)
    for face_idx, face in enumerate(faces):
        vals, counts = np.unique(labels[face], return_counts=True)
        face_labels[face_idx] = int(vals[np.argmax(counts)])
    return face_labels


def color_for_labels(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64)
    colors = PALETTE[labels % len(PALETTE)]
    cycle = (labels // len(PALETTE)).astype(np.uint8)
    if cycle.max(initial=0) > 0:
        colors = np.clip(colors.astype(np.int16) + (cycle[:, None] * 17), 0, 255).astype(np.uint8)
    return colors


def write_face_colored_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
) -> None:
    """Write a duplicate-vertex mesh so common viewers show hard face colors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dup_vertices = vertices[faces].reshape(-1, 3)
    dup_faces = np.arange(faces.shape[0] * 3, dtype=np.int32).reshape(-1, 3)
    dup_colors = np.repeat(color_for_labels(face_labels), 3, axis=0)

    vertex_array = np.empty(
        dup_vertices.shape[0],
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
    vertex_array["x"] = dup_vertices[:, 0]
    vertex_array["y"] = dup_vertices[:, 1]
    vertex_array["z"] = dup_vertices[:, 2]
    vertex_array["red"] = dup_colors[:, 0]
    vertex_array["green"] = dup_colors[:, 1]
    vertex_array["blue"] = dup_colors[:, 2]
    vertex_array["alpha"] = 255

    face_array = np.empty(dup_faces.shape[0], dtype=[("vertex_indices", "O")])
    face_array["vertex_indices"] = [row for row in dup_faces]

    PlyData(
        [
            PlyElement.describe(vertex_array, "vertex"),
            PlyElement.describe(face_array, "face"),
        ],
        text=False,
    ).write(str(path))


def bucket_stats(
    labels: np.ndarray,
    sample_positions: np.ndarray,
    n_buckets: int,
) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    coords = bbox_coordinates(sample_positions)
    for bucket_id in range(n_buckets):
        mask = labels == bucket_id
        if mask.any():
            selected = coords[mask]
            stats[f"bucket_{bucket_id:02d}"] = {
                "count": int(mask.sum()),
                "centroid_xyz01": selected.mean(axis=0).round(4).tolist(),
                "extent_xyz01": (selected.max(axis=0) - selected.min(axis=0)).round(4).tolist(),
            }
        else:
            stats[f"bucket_{bucket_id:02d}"] = {
                "count": 0,
                "centroid_xyz01": None,
                "extent_xyz01": None,
            }
    return stats


def default_name(mesh_path: Path) -> str:
    name = mesh_path.stem
    for suffix in ("_5k_upright_wheels_down", "_upright_wheels_down"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def load_items(args: argparse.Namespace) -> List[MeshData]:
    if len(args.mesh) != len(args.feature):
        raise ValueError("--mesh and --feature must be passed the same number of times.")
    if args.name and len(args.name) != len(args.mesh):
        raise ValueError("--name must be omitted or passed once per --mesh.")

    items = []
    for idx, (mesh_arg, feature_arg) in enumerate(zip(args.mesh, args.feature)):
        mesh_path = Path(mesh_arg)
        feature_path = Path(feature_arg)
        name = args.name[idx] if args.name else default_name(mesh_path)
        vertices, faces = load_mesh(mesh_path)
        raw_features = load_feature_array(feature_path)
        mode = infer_feature_mode(
            raw_features,
            n_vertices=vertices.shape[0],
            n_faces=faces.shape[0],
            requested=args.feature_mode,
            feature_path=feature_path,
        )
        cluster_features = build_cluster_features(
            vertices=vertices,
            faces=faces,
            partfield_features=raw_features,
            mode=mode,
            position_weight=args.position_weight,
            normal_weight=args.normal_weight,
        )
        items.append(
            MeshData(
                name=name,
                mesh_path=mesh_path,
                feature_path=feature_path,
                vertices=vertices,
                faces=faces,
                raw_features=raw_features,
                feature_mode=mode,
                cluster_features=cluster_features,
            )
        )
    return items


def split_labels(labels: np.ndarray, lengths: Iterable[int]) -> List[np.ndarray]:
    chunks = []
    start = 0
    for length in lengths:
        end = start + int(length)
        chunks.append(labels[start:end].astype(np.int64))
        start = end
    return chunks


def save_outputs(items: List[MeshData], output_dir: Path, n_buckets: int, args: argparse.Namespace) -> None:
    labels_dir = output_dir / "labels"
    colored_dir = output_dir / "colored"
    labels_dir.mkdir(parents=True, exist_ok=True)
    colored_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "n_buckets": int(n_buckets),
        "cluster_mode": "joint_kmeans",
        "feature_mode": args.feature_mode,
        "position_weight": float(args.position_weight),
        "normal_weight": float(args.normal_weight),
        "items": {},
    }

    for item in items:
        if item.labels is None:
            raise RuntimeError(f"Missing labels for {item.name}.")

        if item.feature_mode == "face":
            face_labels = item.labels
            vertex_labels = labels_to_vertex_labels(face_labels, item.faces, item.vertices.shape[0])
            stat_positions = face_centroids(item.vertices, item.faces)
        else:
            vertex_labels = item.labels
            face_labels = labels_to_face_labels(vertex_labels, item.faces)
            stat_positions = item.vertices

        np.save(labels_dir / f"{item.name}_face_labels.npy", face_labels.astype(np.int64))
        np.save(labels_dir / f"{item.name}_vertex_labels.npy", vertex_labels.astype(np.int64))
        np.savez_compressed(
            labels_dir / f"{item.name}_partfield_labels.npz",
            face_labels=face_labels.astype(np.int64),
            vertex_labels=vertex_labels.astype(np.int64),
            n_buckets=np.array(n_buckets, dtype=np.int64),
        )
        write_face_colored_ply(
            colored_dir / f"{item.name}_partfield_{n_buckets:02d}_parts.ply",
            vertices=item.vertices,
            faces=item.faces,
            face_labels=face_labels,
        )

        summary["items"][item.name] = {
            "mesh_path": str(item.mesh_path),
            "feature_path": str(item.feature_path),
            "feature_rows": int(item.raw_features.shape[0]),
            "feature_dim": int(item.raw_features.shape[1]),
            "resolved_feature_mode": item.feature_mode,
            "n_vertices": int(item.vertices.shape[0]),
            "n_faces": int(item.faces.shape[0]),
            "face_labels": str(labels_dir / f"{item.name}_face_labels.npy"),
            "vertex_labels": str(labels_dir / f"{item.name}_vertex_labels.npy"),
            "label_npz": str(labels_dir / f"{item.name}_partfield_labels.npz"),
            "colored_mesh": str(colored_dir / f"{item.name}_partfield_{n_buckets:02d}_parts.ply"),
            "bucket_stats": bucket_stats(item.labels, stat_positions, n_buckets),
        }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved PartField co-segmentation outputs to {output_dir}")
    print(f"Summary: {output_dir / 'summary.json'}")
    for item in items:
        print(f"  {item.name}: {colored_dir / f'{item.name}_partfield_{n_buckets:02d}_parts.ply'}")


def run(args: argparse.Namespace) -> None:
    items = load_items(args)
    all_features = np.concatenate([item.cluster_features for item in items], axis=0)
    n_buckets = min(int(args.n_buckets), all_features.shape[0])
    if n_buckets < 2:
        raise ValueError("PartField co-segmentation needs at least two buckets.")

    kmeans = KMeans(n_clusters=n_buckets, random_state=args.random_state, n_init=10)
    all_labels = kmeans.fit_predict(all_features).astype(np.int64)
    for item, labels in zip(items, split_labels(all_labels, [len(i.cluster_features) for i in items])):
        item.labels = labels

    save_outputs(items, Path(args.output_dir), n_buckets, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", action="append", required=True, help="Mesh path. Repeat once per shape.")
    parser.add_argument("--feature", action="append", required=True, help="PartField feature .npy/.npz path. Repeat once per shape.")
    parser.add_argument("--name", action="append", help="Optional output name. Repeat once per shape.")
    parser.add_argument("--output-dir", required=True, help="Directory for labels, colored meshes, and summary.json.")
    parser.add_argument("--n-buckets", type=int, default=12, help="Number of shared part buckets.")
    parser.add_argument("--feature-mode", choices=["auto", "vertex", "face"], default="auto")
    parser.add_argument(
        "--position-weight",
        type=float,
        default=0.05,
        help="Small normalized XYZ term. Helps split repeated parts such as four wheels.",
    )
    parser.add_argument("--normal-weight", type=float, default=0.0, help="Optional normal term for clustering.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
