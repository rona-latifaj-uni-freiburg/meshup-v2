"""Create specs for TOSCA vertex-correspondence-guided deformation jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs_with_target_guidance.orient_mesh_axes import load_obj


PAIR_PLAN: Dict[str, List[Tuple[str, str]]] = {
    "cat": [
        ("cat0", "cat1"),
        ("cat0", "cat2"),
        ("cat1", "cat4"),
        ("cat2", "cat5"),
        ("cat3", "cat9"),
    ],
    "dog": [
        ("dog0", "dog1"),
        ("dog0", "dog2"),
        ("dog1", "dog3"),
        ("dog2", "dog5"),
        ("dog6", "dog8"),
    ],
    "horse": [
        ("horse0", "horse5"),
        ("horse0", "horse6"),
        ("horse5", "horse7"),
        ("horse6", "horse10"),
        ("horse14", "horse15"),
    ],
    "wolf": [
        ("wolf0", "wolf1"),
        ("wolf0", "wolf2"),
        ("wolf1", "wolf2"),
    ],
    "gorilla": [
        ("gorilla1", "gorilla5"),
        ("gorilla1", "gorilla8"),
        ("gorilla1", "gorilla14"),
        ("gorilla5", "gorilla8"),
        ("gorilla8", "gorilla14"),
    ],
}


PROMPTS = {
    "cat": "a cat",
    "dog": "a dog",
    "horse": "a horse",
    "wolf": "a wolf",
    "gorilla": "a gorilla",
}


def rel(path: Path, root: Path) -> str:
    try:
        return "./" + str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def write_dummy_labels(mesh_path: Path, labels_path: Path) -> Dict[str, int]:
    vertices, faces = load_obj(mesh_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        labels_path,
        vertex_labels=np.zeros(vertices.shape[0], dtype=np.int64),
        face_labels=np.zeros(faces.shape[0], dtype=np.int64),
        n_buckets=np.asarray(1, dtype=np.int64),
    )
    return {"vertices": int(vertices.shape[0]), "faces": int(faces.shape[0])}


def validate_pair(source: Path, target: Path) -> Dict[str, object]:
    source_vertices, source_faces = load_obj(source)
    target_vertices, target_faces = load_obj(target)
    if source_vertices.shape[0] != target_vertices.shape[0]:
        raise ValueError(
            f"{source.stem}->{target.stem}: vertex counts differ "
            f"({source_vertices.shape[0]} vs {target_vertices.shape[0]})."
        )
    return {
        "source_vertices": int(source_vertices.shape[0]),
        "target_vertices": int(target_vertices.shape[0]),
        "source_faces": int(source_faces.shape[0]),
        "target_faces": int(target_faces.shape[0]),
        "same_face_array": bool(
            source_faces.shape == target_faces.shape and np.array_equal(source_faces, target_faces)
        ),
    }


def iter_pairs(classes: Sequence[str]) -> Iterable[Tuple[str, str, str]]:
    for class_name in classes:
        if class_name not in PAIR_PLAN:
            raise ValueError(f"Unknown class {class_name!r}. Choices: {sorted(PAIR_PLAN)}")
        for source_name, target_name in PAIR_PLAN[class_name]:
            yield class_name, source_name, target_name


def create_specs(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    mesh_root = Path(args.mesh_root)
    labels_root = Path(args.labels_root)
    spec_path = Path(args.spec)
    manifest_path = Path(args.manifest)
    classes = args.classes.split(",") if args.classes else list(PAIR_PLAN)

    rows: List[str] = [
        "# pair_slug\tsource\ttarget\tsource_name\ttarget_name\tsource_labels\ttarget_labels\tprompt"
    ]
    manifest: Dict[str, object] = {
        "mesh_root": rel(mesh_root, project_root),
        "labels_root": rel(labels_root, project_root),
        "pairs": [],
    }
    label_stats: Dict[str, Dict[str, int]] = {}

    for class_name, source_name, target_name in iter_pairs(classes):
        source = mesh_root / f"{source_name}.obj"
        target = mesh_root / f"{target_name}.obj"
        if not source.is_file():
            raise FileNotFoundError(source)
        if not target.is_file():
            raise FileNotFoundError(target)

        pair_stats = validate_pair(source, target)
        for mesh_name, mesh_path in ((source_name, source), (target_name, target)):
            label_path = labels_root / f"{mesh_name}_dummy_labels.npz"
            if mesh_name not in label_stats:
                label_stats[mesh_name] = write_dummy_labels(mesh_path, label_path)

        pair_slug = f"{source_name}_to_{target_name}"
        source_labels = labels_root / f"{source_name}_dummy_labels.npz"
        target_labels = labels_root / f"{target_name}_dummy_labels.npz"
        rows.append(
            "\t".join(
                [
                    pair_slug,
                    rel(source, project_root),
                    rel(target, project_root),
                    source_name,
                    target_name,
                    rel(source_labels, project_root),
                    rel(target_labels, project_root),
                    PROMPTS[class_name],
                ]
            )
        )
        manifest["pairs"].append(  # type: ignore[index]
            {
                "class": class_name,
                "pair_slug": pair_slug,
                **pair_stats,
                "source": rel(source, project_root),
                "target": rel(target, project_root),
            }
        )

    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("\n".join(rows) + "\n")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {spec_path}")
    print(f"Wrote {manifest_path}")
    print(f"Pairs: {len(rows) - 1}")
    print(f"Dummy labels: {len(label_stats)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mesh-root", default="external/TOSCA/meshup_prepared_zup_xfront/objs")
    parser.add_argument(
        "--labels-root",
        default="jobs_with_target_guidance/partfield_segments/tosca_correspondence_guided/dummy_labels",
    )
    parser.add_argument(
        "--spec",
        default="jobs_with_target_guidance/cross_animal_spike_runs/jobs/tosca_correspondence_guided_specs.tsv",
    )
    parser.add_argument(
        "--manifest",
        default="jobs_with_target_guidance/cross_animal_spike_runs/jobs/tosca_correspondence_guided_manifest.json",
    )
    parser.add_argument(
        "--classes",
        default="cat,dog,horse,wolf,gorilla",
        help="Comma-separated class list. Defaults to all planned classes.",
    )
    return parser.parse_args()


def main() -> None:
    create_specs(parse_args())


if __name__ == "__main__":
    main()
