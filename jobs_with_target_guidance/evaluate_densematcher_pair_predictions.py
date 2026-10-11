#!/usr/bin/env python3
"""Score DenseMatcher DenseCorr3D pair predictions with the local metric.

The DenseMatcher pair dump stores explicit point-to-point predictions under
``external/DenseCorr3D/densecorr_test/pairs/<category>/<src>__<tgt>.npz``.
By default this script scores ``surface_map``.  In the provided files that map
has shape ``(N_target,)`` and means: for each target vertex, the predicted
source vertex.

The metric mirrors ``evaluate_densecorr_paper_style_geodesic.py``: normalized
semantic geodesic error, with AUC integrated from 0 to 0.25 and Err = mean
error x100.  The default graph backend is not the official DenseCorr3D
exact-geodesic evaluator; it is the local metric used for the MeshUp/PartField
reports, applied to the supervisor-provided DenseMatcher predictions.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs_with_target_guidance.densecorr3d_segment import (  # noqa: E402
    infer_index_base,
    load_densecorr3d_groups,
    parse_group_indices,
)
from jobs_with_target_guidance.evaluate_densecorr_paper_style_geodesic import (  # noqa: E402
    auc_from_errors,
    distance_to_source_set,
    graph_from_mesh,
)
from jobs_with_target_guidance.evaluate_deformation_quality import resolve_path  # noqa: E402
from jobs_with_target_guidance.evaluate_target_pipeline import (  # noqa: E402
    face_areas_and_normals,
    load_mesh,
    unit_vertices,
)


@dataclass(frozen=True)
class DenseCorrObject:
    category: str
    object_id: str
    object_dir: Path
    mesh_path: Path
    vertices: np.ndarray
    faces: np.ndarray
    labels: np.ndarray
    annotated_mask: np.ndarray
    graph: object
    normalizer: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-root",
        default="external/DenseCorr3D/densecorr_test/pairs",
        help="Directory containing category subfolders of DenseMatcher .npz pair predictions.",
    )
    parser.add_argument(
        "--densecorr3d-root",
        default="external/DenseCorr3D",
        help="Raw DenseCorr3D root containing <category>/<object>/simple_mesh.obj and groups.txt.",
    )
    parser.add_argument(
        "--extra-densecorr3d-roots",
        nargs="*",
        default=[],
        help=(
            "Optional fallback roots with the same <category>/<object>/simple_mesh.obj layout. "
            "Use this only for documented rigidly transformed mirrors where vertex order and labels are preserved."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy",
    )
    parser.add_argument("--output-prefix", default="densematcher_surface_map_graph_rawgroups")
    parser.add_argument(
        "--prediction-key",
        default="surface_map",
        help="Prediction array key to score, e.g. surface_map or surface_map_inv.",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help=(
            "Also score the matching inverse map and append those rows. For example, "
            "surface_map is paired with surface_map_inv, and surface_map_icp with surface_map_inv_icp."
        ),
    )
    parser.add_argument(
        "--inverse-prediction-key",
        default=None,
        help="Optional explicit inverse key to use with --bidirectional.",
    )
    parser.add_argument(
        "--map-direction",
        choices=("auto", "target_to_source", "source_to_target"),
        default="auto",
        help=(
            "Direction of the prediction array. 'auto' treats keys containing '_inv' as source_to_target "
            "and all other keys as target_to_source."
        ),
    )
    parser.add_argument("--auc-max-threshold", type=float, default=0.25)
    parser.add_argument("--auc-steps", type=int, default=251)
    parser.add_argument(
        "--geodesic-method",
        choices=("exact", "heat", "graph"),
        default="graph",
        help="Use exact/heat surface geodesics, or graph shortest paths along mesh edges.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional category allow-list. Defaults to every category under --pairs-root.",
    )
    parser.add_argument(
        "--include-pair-table",
        action="store_true",
        help="Include the full per-pair table in the Markdown report.",
    )
    parser.add_argument(
        "--score-vertices",
        choices=("all", "annotated"),
        default="all",
        help=(
            "Vertices to include in the metric. 'all' preserves the original local proxy, where unannotated "
            "groups.txt vertices are nearest-filled before scoring. 'annotated' scores only vertices explicitly "
            "listed in groups.txt and uses only explicitly listed reference vertices as the correct semantic set."
        ),
    )
    return parser.parse_args()


def as_scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(()).item())


def parse_category_object(value: str, fallback_category: str) -> Tuple[str, str]:
    if "/" in value:
        category, object_id = value.split("/", 1)
        return category, object_id
    return fallback_category, value


def display_path(path: Path) -> str:
    path = path.resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return str(path.relative_to(PROJECT_ROOT))
    return str(path)


def load_annotated_mask(groups_path: Path, n_vertices: int) -> np.ndarray:
    raw_groups = []
    for line in groups_path.read_text().splitlines():
        indices = parse_group_indices(line)
        if indices:
            raw_groups.append(indices)
    base = infer_index_base(raw_groups, n_vertices=n_vertices, requested="auto")

    mask = np.zeros(n_vertices, dtype=bool)
    for raw_indices in raw_groups:
        indices = np.asarray(raw_indices, dtype=np.int64) - base
        invalid = indices[(indices < 0) | (indices >= n_vertices)]
        if invalid.size:
            raise ValueError(f"{groups_path} has {invalid.size} out-of-range vertex ids after base conversion.")
        mask[indices] = True
    return mask


def candidate_object_dir(root: Path, category: str, object_id: str) -> Path | None:
    category_dir = root / category
    exact = category_dir / object_id
    if exact.is_dir():
        return exact

    if not category_dir.is_dir():
        return None

    candidates = [
        path
        for path in category_dir.iterdir()
        if path.is_dir() and (path.name.startswith(object_id) or object_id.startswith(path.name))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    names = ", ".join(path.name for path in candidates[:8])
    raise RuntimeError(f"Ambiguous DenseCorr3D object directory for {category}/{object_id}: {names}")


def resolve_object_dir(roots: Sequence[Path], category: str, object_id: str) -> Path:
    for root in roots:
        candidate = candidate_object_dir(root, category, object_id)
        if candidate is not None:
            return candidate
    roots_text = ", ".join(display_path(root) for root in roots)
    raise FileNotFoundError(f"Missing DenseCorr3D object directory for {category}/{object_id} under: {roots_text}")


def load_object(
    roots: Sequence[Path],
    category: str,
    object_id: str,
    cache: Dict[Tuple[str, str], DenseCorrObject],
) -> DenseCorrObject:
    key = (category, object_id)
    if key in cache:
        return cache[key]

    object_dir = resolve_object_dir(roots, category, object_id)
    mesh_path = object_dir / "simple_mesh.obj"
    groups_path = object_dir / "groups.txt"
    if not mesh_path.is_file() or not groups_path.is_file():
        raise FileNotFoundError(f"Missing simple_mesh.obj or groups.txt under {object_dir}")

    raw_vertices, faces = load_mesh(mesh_path)
    vertices = unit_vertices(raw_vertices)
    labels, _summary = load_densecorr3d_groups(
        groups_path=groups_path,
        n_vertices=raw_vertices.shape[0],
        vertices=raw_vertices,
        index_base="auto",
        fill_mode="nearest",
    )
    annotated_mask = load_annotated_mask(groups_path, raw_vertices.shape[0])
    if labels.shape[0] != vertices.shape[0]:
        raise RuntimeError(f"Label/vertex mismatch for {category}/{object_id}: {labels.shape[0]} vs {vertices.shape[0]}")

    area = float(face_areas_and_normals(vertices, faces)[0].sum())
    obj = DenseCorrObject(
        category=category,
        object_id=object_id,
        object_dir=object_dir,
        mesh_path=mesh_path,
        vertices=vertices,
        faces=faces,
        labels=labels.astype(np.int64, copy=False),
        annotated_mask=annotated_mask,
        graph=graph_from_mesh(vertices, faces),
        normalizer=max(math.sqrt(area), 1e-12),
    )
    cache[key] = obj
    return obj


def infer_direction(prediction_key: str, map_direction: str) -> str:
    if map_direction != "auto":
        return map_direction
    return "source_to_target" if "_inv" in prediction_key else "target_to_source"


def infer_inverse_key(prediction_key: str) -> str:
    if "_inv" in prediction_key:
        return prediction_key.replace("_inv", "", 1)
    if prediction_key.startswith("surface_map_"):
        return prediction_key.replace("surface_map_", "surface_map_inv_", 1)
    if prediction_key == "surface_map":
        return "surface_map_inv"
    raise ValueError(f"Cannot infer inverse DenseMatcher key for {prediction_key!r}; pass --inverse-prediction-key")


def distance_tables(reference: DenseCorrObject, score_vertices: str, method: str = "graph") -> Dict[int, np.ndarray]:
    tables: Dict[int, np.ndarray] = {}
    for label in sorted(set(reference.labels.tolist())):
        label_mask = reference.labels == label
        if score_vertices == "annotated":
            label_mask = label_mask & reference.annotated_mask
        label_vertices = np.flatnonzero(label_mask)
        if label_vertices.size == 0:
            continue
        tables[int(label)] = (
            distance_to_source_set(
                reference.vertices,
                reference.faces,
                reference.graph,
                label_vertices,
                method,
            )
            / reference.normalizer
        )
    return tables


def evaluate_prediction(
    pair_path: Path,
    args: argparse.Namespace,
    object_cache: Dict[Tuple[str, str], DenseCorrObject],
    distance_cache: Dict[Tuple[str, str, str, str], Dict[int, np.ndarray]],
    prediction_key: str | None = None,
) -> Dict[str, object]:
    data = np.load(pair_path, allow_pickle=False)
    prediction_key = prediction_key or args.prediction_key
    if prediction_key not in data.files:
        raise KeyError(f"{prediction_key!r} not found")

    category = as_scalar_string(data["category"]) if "category" in data.files else pair_path.parent.name
    source_category, source_id = parse_category_object(as_scalar_string(data["source"]), category)
    target_category, target_id = parse_category_object(as_scalar_string(data["target"]), category)
    if source_category != target_category:
        raise RuntimeError(f"Cross-category pair is not supported: {source_category} -> {target_category}")
    category = source_category

    roots = [resolve_path(args.densecorr3d_root)] + [resolve_path(path) for path in args.extra_densecorr3d_roots]
    source = load_object(roots, category, source_id, object_cache)
    target = load_object(roots, category, target_id, object_cache)

    prediction = np.asarray(data[prediction_key], dtype=np.int64).reshape(-1)
    direction = infer_direction(prediction_key, args.map_direction)
    if direction == "target_to_source":
        domain = target
        reference = source
    else:
        domain = source
        reference = target

    if prediction.shape[0] != domain.vertices.shape[0]:
        raise RuntimeError(
            f"{prediction_key} length {prediction.shape[0]} does not match {direction} "
            f"domain vertex count {domain.vertices.shape[0]} for {pair_path}"
        )
    if prediction.size and (prediction.min() < 0 or prediction.max() >= reference.vertices.shape[0]):
        raise RuntimeError(
            f"{prediction_key} has indices outside reference vertex range 0..{reference.vertices.shape[0] - 1}"
        )

    ref_key = (reference.category, reference.object_id, args.score_vertices, args.geodesic_method)
    if ref_key not in distance_cache:
        distance_cache[ref_key] = distance_tables(reference, args.score_vertices, args.geodesic_method)

    errors = np.full(domain.labels.shape[0], np.inf, dtype=np.float64)
    for label in sorted(set(domain.labels.tolist())):
        distances = distance_cache[ref_key].get(int(label))
        if distances is None:
            continue
        mask = domain.labels == label
        errors[mask] = distances[prediction[mask]]

    eval_mask = np.ones(domain.labels.shape[0], dtype=bool)
    if args.score_vertices == "annotated":
        eval_mask = domain.annotated_mask
    eval_errors = errors[eval_mask]
    finite = eval_errors[np.isfinite(eval_errors)]
    auc, err_x100, pck = auc_from_errors(eval_errors, args.auc_max_threshold, args.auc_steps)
    same_bucket = math.nan
    if eval_mask.any():
        same_bucket = float((reference.labels[prediction[eval_mask]] == domain.labels[eval_mask]).mean())
    row: Dict[str, object] = {
        "method": f"DenseMatcher {prediction_key}",
        "category": category,
        "source": source_id,
        "target": target_id,
        "pair": f"{domain.object_id}_to_{reference.object_id}",
        "file_pair": f"{source_id}_to_{target_id}",
        "pair_file": display_path(pair_path),
        "prediction_key": prediction_key,
        "map_direction": direction,
        "domain_object": domain.object_id,
        "reference_object": reference.object_id,
        "domain_vertices": int(domain.vertices.shape[0]),
        "reference_vertices": int(reference.vertices.shape[0]),
        "domain_annotated_vertices": int(domain.annotated_mask.sum()),
        "reference_annotated_vertices": int(reference.annotated_mask.sum()),
        "num_vertices_evaluated": int(finite.size),
        "score_vertices": args.score_vertices,
        f"paper_style_auc_0_to_{str(args.auc_max_threshold).replace('.', 'p')}": auc,
        "paper_style_err_x100": err_x100,
        "paper_style_p50_x100": float(np.percentile(finite, 50) * 100.0) if finite.size else math.inf,
        "paper_style_p95_x100": float(np.percentile(finite, 95) * 100.0) if finite.size else math.inf,
        "nearest_reference_same_bucket_rate": same_bucket,
        "geodesic_method": args.geodesic_method,
        "ground_truth_label_source": "densecorr3d_raw",
        "source_mesh": display_path(source.mesh_path),
        "target_mesh": display_path(target.mesh_path),
    }
    row.update(pck)
    return row


def collect_pair_files(args: argparse.Namespace) -> List[Path]:
    pairs_root = resolve_path(args.pairs_root)
    categories = set(args.categories or [])
    files = sorted(path for path in pairs_root.glob("*/*.npz") if not categories or path.parent.name in categories)
    if not files:
        raise SystemExit(f"No .npz pair files found under {pairs_root}")
    return files


def mean(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else math.nan


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row.keys()})
    preferred = [
        "method",
        "category",
        "pair",
        "file_pair",
        "source",
        "target",
        "prediction_key",
        "map_direction",
        "score_vertices",
        "paper_style_auc_0_to_0p25",
        "paper_style_err_x100",
        "paper_style_p50_x100",
        "paper_style_p95_x100",
        "nearest_reference_same_bucket_rate",
        "domain_vertices",
        "reference_vertices",
        "domain_annotated_vertices",
        "reference_annotated_vertices",
        "num_vertices_evaluated",
        "pair_file",
    ]
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    auc_key = "paper_style_auc_0_to_0p25"
    overall = list(rows)
    output.append(
        {
            "category": "all",
            "n": len(overall),
            "mean_auc": mean(overall, auc_key),
            "mean_err": mean(overall, "paper_style_err_x100"),
            "mean_p50_err": mean(overall, "paper_style_p50_x100"),
            "mean_p95_err": mean(overall, "paper_style_p95_x100"),
            "mean_same_bucket_rate": mean(overall, "nearest_reference_same_bucket_rate"),
        }
    )
    for category in sorted({str(row["category"]) for row in rows}):
        group = [row for row in rows if row["category"] == category]
        output.append(
            {
                "category": category,
                "n": len(group),
                "mean_auc": mean(group, auc_key),
                "mean_err": mean(group, "paper_style_err_x100"),
                "mean_p50_err": mean(group, "paper_style_p50_x100"),
                "mean_p95_err": mean(group, "paper_style_p95_x100"),
                "mean_same_bucket_rate": mean(group, "nearest_reference_same_bucket_rate"),
            }
        )
    return output


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["category", "n", "mean_auc", "mean_err", "mean_p50_err", "mean_p95_err", "mean_same_bucket_rate"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}"
        return str(value)
    return str(value)


def write_markdown(path: Path, rows: Sequence[Dict[str, object]], failures: Sequence[Dict[str, str]], include_pair_table: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = summary_rows(rows)
    with path.open("w") as handle:
        handle.write("# DenseMatcher DenseCorr3D Pair-Prediction Evaluation\n\n")
        handle.write(
            "Scores supervisor-provided DenseMatcher `.npz` predictions with the local normalized "
            "semantic geodesic metric used by the MeshUp/PartField reports. The default graph backend is "
            "the fair local baseline for comparisons that also use graph geodesics and DenseCorr3D "
            "`groups.txt` labels.\n\n"
        )
        if rows:
            prediction_keys = ", ".join(sorted({f"`{row['prediction_key']}`" for row in rows}))
            directions = ", ".join(sorted({f"`{row['map_direction']}`" for row in rows}))
            handle.write(
                f"Prediction key(s): {prediction_keys}. Map direction(s): {directions}. "
                "AUC is integrated over normalized semantic geodesic error from 0 to 0.25; "
                "Err is mean normalized error x100.\n\n"
            )
        handle.write("## Summary\n\n")
        handle.write("| Category | n | mean AUC | mean Err | mean p50 Err | mean p95 Err | mean same-bucket rate |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries:
            handle.write(
                f"| {row['category']} | {row['n']} | {fmt(row['mean_auc'])} | {fmt(row['mean_err'])} | "
                f"{fmt(row['mean_p50_err'])} | {fmt(row['mean_p95_err'])} | {fmt(row['mean_same_bucket_rate'])} |\n"
            )

        if include_pair_table:
            handle.write("\n## Pairs\n\n")
            handle.write("| Category | Pair | AUC | Err | p95 Err | same-bucket | evaluated vertices |\n")
            handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
            for row in sorted(rows, key=lambda item: (str(item["category"]), str(item["pair"]))):
                handle.write(
                    f"| {row['category']} | `{row['pair']}` | "
                    f"{fmt(row['paper_style_auc_0_to_0p25'])} | {fmt(row['paper_style_err_x100'])} | "
                    f"{fmt(row['paper_style_p95_x100'])} | {fmt(row['nearest_reference_same_bucket_rate'])} | "
                    f"{row['num_vertices_evaluated']} |\n"
                )

        if failures:
            handle.write("\n## Failures\n\n")
            for failure in failures:
                key = failure.get("prediction_key")
                key_text = f" [{key}]" if key else ""
                handle.write(f"- `{failure['path']}`{key_text}: {failure['error']}\n")


def main() -> None:
    args = parse_args()
    pair_files = collect_pair_files(args)
    out_dir = resolve_path(args.output_dir)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    object_cache: Dict[Tuple[str, str], DenseCorrObject] = {}
    distance_cache: Dict[Tuple[str, str, str, str], Dict[int, np.ndarray]] = {}

    for idx, pair_path in enumerate(pair_files, start=1):
        print(f"[{idx}/{len(pair_files)}] {pair_path}")
        prediction_keys = [args.prediction_key]
        if args.bidirectional:
            prediction_keys.append(args.inverse_prediction_key or infer_inverse_key(args.prediction_key))
        for prediction_key in prediction_keys:
            try:
                rows.append(evaluate_prediction(pair_path, args, object_cache, distance_cache, prediction_key))
            except Exception as exc:
                print(f"Warning: failed {pair_path} [{prediction_key}]: {exc}")
                failures.append({"path": display_path(pair_path), "prediction_key": prediction_key, "error": str(exc)})

    if not rows:
        raise SystemExit("No DenseMatcher pair predictions evaluated.")

    csv_path = out_dir / f"{args.output_prefix}.csv"
    summary_csv_path = out_dir / f"{args.output_prefix}_summary.csv"
    markdown_path = out_dir / f"{args.output_prefix}.md"
    write_csv(csv_path, rows)
    write_summary_csv(summary_csv_path, summary_rows(rows))
    write_markdown(markdown_path, rows, failures, args.include_pair_table)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {markdown_path}")
    print(f"evaluated={len(rows)} failures={len(failures)} objects_loaded={len(object_cache)} distance_tables={len(distance_cache)}")


if __name__ == "__main__":
    main()
