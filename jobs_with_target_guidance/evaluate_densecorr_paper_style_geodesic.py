#!/usr/bin/env python3
"""DenseCorr3D paper-style semantic geodesic AUC for deformation outputs.

DenseMatcher/DenseCorr3D evaluates predicted correspondences by measuring the
geodesic distance on the target mesh from the predicted target point to the
nearest vertex in the correct semantic group.  MeshUp produces a deformed
source mesh rather than an explicit correspondence map, so this script induces
a map by nearest target vertex after normalizing the deformed and target meshes.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs_with_target_guidance.evaluate_bucket_discrepancies import (  # noqa: E402
    infer_pair_names,
    method_label_from_family,
)
from jobs_with_target_guidance.densecorr3d_prepare_mesh_variants import transfer_vertex_labels  # noqa: E402
from jobs_with_target_guidance.densecorr3d_segment import load_densecorr3d_groups  # noqa: E402
from jobs_with_target_guidance.evaluate_deformation_quality import (  # noqa: E402
    resolve_path,
    resolve_run,
)
from jobs_with_target_guidance.evaluate_target_pipeline import (  # noqa: E402
    face_areas_and_normals,
    labels_to_vertex_labels,
    load_mesh,
    unit_vertices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Run output dirs or final mesh paths.")
    parser.add_argument("--paths-file", help="Optional newline-separated paths file.")
    parser.add_argument(
        "--output-dir",
        default="jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy",
    )
    parser.add_argument("--family-regex", default="densecorr3d_.*(bucketbalanced|srcstable|requested).*")
    parser.add_argument("--run-regex", default="")
    parser.add_argument("--auc-max-threshold", type=float, default=0.25)
    parser.add_argument("--auc-steps", type=int, default=251)
    parser.add_argument("--label-mode", choices=("auto", "vertex", "face"), default="auto")
    parser.add_argument(
        "--ground-truth-label-source",
        choices=("run", "densecorr3d_raw"),
        default="run",
        help=(
            "'run' uses the labels that drove the deformation. 'densecorr3d_raw' ignores those labels and "
            "scores against DenseCorr3D groups.txt labels inferred from the source/target mesh object ids."
        ),
    )
    parser.add_argument("--densecorr3d-root", default="external/DenseCorr3D")
    parser.add_argument(
        "--densecorr3d-category",
        default="animals",
        help="DenseCorr3D category folder to score against, or 'auto' to infer it from each mesh path.",
    )
    parser.add_argument(
        "--output-prefix",
        default="densecorr_kim_princeton_geodesic_auc",
        help="Prefix for the CSV/Markdown files written under --output-dir.",
    )
    parser.add_argument(
        "--geodesic-method",
        choices=("exact", "heat", "graph"),
        default="graph",
        help="Use libigl exact/heat surface geodesics, or graph shortest paths along mesh edges.",
    )
    parser.add_argument("--max-target-vertices", type=int, default=20000)
    return parser.parse_args()


def collect_paths(args: argparse.Namespace) -> List[str]:
    paths = list(args.paths)
    if args.paths_file:
        with resolve_path(args.paths_file).open("r") as handle:
            paths.extend(line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#"))
    if not paths:
        raise SystemExit("Provide paths or --paths-file.")
    return paths


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).astype(np.int64)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def mesh_topology_counts(faces: np.ndarray) -> Tuple[int, int]:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).astype(np.int64)
    edges.sort(axis=1)
    _unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int((counts == 1).sum())
    nonmanifold_edges = int((counts > 2).sum())
    return boundary_edges, nonmanifold_edges


def graph_from_mesh(vertices: np.ndarray, faces: np.ndarray):
    edges = mesh_edges(faces)
    weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    data = np.concatenate((weights, weights))
    return coo_matrix((data, (rows, cols)), shape=(vertices.shape[0], vertices.shape[0])).tocsr()


def multi_source_distances(graph, sources: np.ndarray) -> np.ndarray:
    sources = np.asarray(sources, dtype=np.int64)
    if sources.size == 0:
        return np.full(graph.shape[0], np.inf, dtype=np.float64)
    try:
        return np.asarray(
            dijkstra(graph, directed=False, indices=sources, min_only=True, return_predecessors=False),
            dtype=np.float64,
        )
    except TypeError:
        dist = dijkstra(graph, directed=False, indices=sources, return_predecessors=False)
        return np.asarray(dist, dtype=np.float64).min(axis=0)


def exact_surface_distances(vertices: np.ndarray, faces: np.ndarray, sources: np.ndarray) -> np.ndarray:
    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    if sources.size == 0:
        return np.full(vertices.shape[0], np.inf, dtype=np.float64)

    boundary_edges, nonmanifold_edges = mesh_topology_counts(faces)
    if boundary_edges or nonmanifold_edges:
        raise RuntimeError(
            "exact geodesic backend requires a clean manifold target mesh; "
            f"found {boundary_edges} boundary edges and {nonmanifold_edges} non-manifold edges. "
            "Use the released DenseCorr3D geodesic matrices if available, or use --geodesic-method graph "
            "for the local non-manifold simple_mesh.obj files."
        )

    vendor_dir = PROJECT_ROOT / "jobs_with_target_guidance" / "vendor_pydeps"
    if vendor_dir.is_dir() and str(vendor_dir) not in sys.path:
        sys.path.append(str(vendor_dir))

    try:
        import pygeodesic.geodesic as geodesic

        geo = geodesic.PyGeodesicAlgorithmExact(
            np.ascontiguousarray(vertices, dtype=np.float64),
            np.ascontiguousarray(faces, dtype=np.int32),
        )
        targets = np.arange(vertices.shape[0], dtype=np.int32)
        distances, _paths = geo.geodesicDistances(sources.tolist(), targets.tolist())
        return np.asarray(distances, dtype=np.float64).reshape(-1)
    except ImportError:
        import igl

        targets = np.arange(vertices.shape[0], dtype=np.int32)
        return np.asarray(
            igl.exact_geodesic(
                np.ascontiguousarray(vertices, dtype=np.float64),
                np.ascontiguousarray(faces, dtype=np.int32),
                sources,
                targets,
            ),
            dtype=np.float64,
        ).reshape(-1)


def heat_surface_distances(vertices: np.ndarray, faces: np.ndarray, sources: np.ndarray) -> np.ndarray:
    import igl

    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    if sources.size == 0:
        return np.full(vertices.shape[0], np.inf, dtype=np.float64)
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    t = max((bbox_diag / max(vertices.shape[0], 1)) ** 2, 1e-8)
    distances = np.asarray(
        igl.heat_geodesic(
            vertices.astype(np.float64, copy=False),
            faces.astype(np.int32, copy=False),
            t,
            sources,
        ),
        dtype=np.float64,
    ).reshape(-1)
    finite = np.isfinite(distances)
    if finite.any():
        distances[finite] -= float(distances[finite].min())
        distances[finite] = np.maximum(distances[finite], 0.0)
    distances[sources] = 0.0
    return distances


def distance_to_source_set(
    vertices: np.ndarray,
    faces: np.ndarray,
    graph,
    sources: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "exact":
        return exact_surface_distances(vertices, faces, sources)
    if method == "heat":
        return heat_surface_distances(vertices, faces, sources)
    return multi_source_distances(graph, sources)


def densecorr3d_category_object_from_path(path: Path) -> Tuple[str, str]:
    parts = path.parts
    for idx, part in enumerate(parts[:-2]):
        if part == "DenseCorr3D":
            return parts[idx + 1], parts[idx + 2]
    for idx, part in enumerate(parts[:-1]):
        if part == "animals":
            return "animals", parts[idx + 1]
    text = str(path)
    match = re.search(r"DenseCorr3D/([^/]+)/([^/]+)/", text)
    if match:
        return match.group(1), match.group(2)
    match = re.search(r"([A-Za-z0-9]+_(?:toy_animals|dinosaur)_\d+)", text)
    if match:
        return "animals", match.group(1)
    match = re.search(r"densecorr3d_([a-z0-9_]+)_azimuth_fixed/([^/]+)/", text)
    if match:
        return match.group(1), match.group(2)
    raise RuntimeError(f"could not infer DenseCorr3D object id from {path}")


def load_densecorr3d_vertex_labels_for_mesh(mesh_path: Path, vertices: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    inferred_category, object_id = densecorr3d_category_object_from_path(mesh_path)
    category = inferred_category if args.densecorr3d_category == "auto" else args.densecorr3d_category
    object_dir = resolve_path(args.densecorr3d_root) / category / object_id
    simple_mesh = object_dir / "simple_mesh.obj"
    groups_path = object_dir / "groups.txt"
    if not simple_mesh.is_file() or not groups_path.is_file():
        raise RuntimeError(f"missing DenseCorr3D raw files for {object_id} under {object_dir}")
    simple_vertices, _simple_faces = load_mesh(simple_mesh)
    simple_labels, _summary = load_densecorr3d_groups(
        groups_path=groups_path,
        n_vertices=simple_vertices.shape[0],
        vertices=simple_vertices,
        index_base="auto",
        fill_mode="nearest",
    )
    # If the canonical ground-truth mesh is literally the same geometry as the
    # mesh being scored (the common case: vertices is target_raw/source_vertices
    # loaded straight from this same object's file), labels transfer by identity
    # index. Only fall back to spatial nearest-neighbor matching for genuinely
    # different point clouds. This avoids spurious mismatches from re-doing a
    # spatial match on tightly-packed geometry (e.g. thin cylindrical shapes)
    # where nearest-neighbor ties can pick an adjacent, not identical, vertex.
    if (
        simple_vertices.shape[0] == vertices.shape[0]
        and np.allclose(simple_vertices, vertices, atol=1e-5)
    ):
        return simple_labels
    return transfer_vertex_labels(
        source_vertices=simple_vertices,
        source_labels=simple_labels,
        target_vertices=vertices,
        normalize=True,
    )


def auc_from_errors(errors: np.ndarray, max_threshold: float, n_steps: int) -> Tuple[float, float, Dict[str, float]]:
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return 0.0, math.inf, {}
    thresholds = np.linspace(0.0, max_threshold, n_steps)
    acc = np.asarray([(finite <= t).mean() for t in thresholds], dtype=np.float64)
    pck = {f"pck_{key}": float((finite <= value).mean()) for key, value in {
        "0p01": 0.01,
        "0p02": 0.02,
        "0p05": 0.05,
        "0p10": 0.10,
        "0p25": 0.25,
    }.items()}
    return float(np.trapz(acc, thresholds) / max_threshold), float(finite.mean() * 100.0), pck


def evaluate_path(path: str, args: argparse.Namespace) -> Dict[str, object]:
    run = resolve_run(path)
    family = run.output_dir.parent.name
    if args.family_regex and not re.search(args.family_regex, family):
        raise RuntimeError("filtered_by_family")
    if args.run_regex and not re.search(args.run_regex, run.output_dir.name):
        raise RuntimeError("filtered_by_run")
    if args.ground_truth_label_source == "run" and (run.source_labels is None or run.target_labels is None):
        raise RuntimeError("missing labels")
    if args.ground_truth_label_source == "densecorr3d_raw" and run.source_mesh is None:
        raise RuntimeError("missing source mesh needed for DenseCorr3D raw ground-truth labels")

    deformed_raw, deformed_faces = load_mesh(run.deformed_mesh)
    target_raw, target_faces = load_mesh(run.target_mesh)
    if target_raw.shape[0] > args.max_target_vertices:
        raise RuntimeError(f"target has {target_raw.shape[0]} vertices; above --max-target-vertices")

    if args.ground_truth_label_source == "densecorr3d_raw":
        source_vertices_for_labels = load_mesh(run.source_mesh)[0]
        source_labels_on_source = load_densecorr3d_vertex_labels_for_mesh(run.source_mesh, source_vertices_for_labels, args)
        if source_labels_on_source.shape[0] == deformed_raw.shape[0]:
            source_labels = source_labels_on_source
        else:
            source_labels = transfer_vertex_labels(
                source_vertices=source_vertices_for_labels,
                source_labels=source_labels_on_source,
                target_vertices=deformed_raw,
                normalize=True,
            )
        target_labels = load_densecorr3d_vertex_labels_for_mesh(run.target_mesh, target_raw, args)
    else:
        source_labels = labels_to_vertex_labels(run.source_labels, deformed_raw.shape[0], deformed_faces, args.label_mode)
        target_labels = labels_to_vertex_labels(run.target_labels, target_raw.shape[0], target_faces, args.label_mode)
    deformed_vertices = unit_vertices(deformed_raw)
    target_vertices = unit_vertices(target_raw)

    predicted_target_vertex = cKDTree(target_vertices).query(deformed_vertices, workers=-1)[1].astype(np.int64)
    graph = graph_from_mesh(target_vertices, target_faces)
    target_area = float(face_areas_and_normals(target_vertices, target_faces)[0].sum())
    normalizer = max(math.sqrt(target_area), 1e-12)

    target_buckets = sorted(set(target_labels.tolist()))
    distance_to_bucket = {}
    for bucket in target_buckets:
        bucket_vertices = np.flatnonzero(target_labels == bucket)
        distance_to_bucket[bucket] = (
            distance_to_source_set(
                target_vertices,
                target_faces,
                graph,
                bucket_vertices,
                args.geodesic_method,
            )
            / normalizer
        )

    errors = np.full(source_labels.shape[0], np.inf, dtype=np.float64)
    for bucket in sorted(set(source_labels.tolist())):
        if bucket not in distance_to_bucket:
            continue
        mask = source_labels == bucket
        errors[mask] = distance_to_bucket[int(bucket)][predicted_target_vertex[mask]]

    auc, err_x100, pck = auc_from_errors(errors, args.auc_max_threshold, args.auc_steps)
    source_name, target_name = infer_pair_names(run)
    finite = errors[np.isfinite(errors)]
    row: Dict[str, object] = {
        "family": family,
        "method": method_label_from_family(family),
        "pair": f"{source_name}_to_{target_name}",
        "run_name": run.output_dir.name,
        "target_vertices": int(target_raw.shape[0]),
        "num_vertices_evaluated": int(finite.size),
        f"paper_style_auc_0_to_{str(args.auc_max_threshold).replace('.', 'p')}": auc,
        "paper_style_err_x100": err_x100,
        "paper_style_p50_x100": float(np.percentile(finite, 50) * 100.0) if finite.size else math.inf,
        "paper_style_p95_x100": float(np.percentile(finite, 95) * 100.0) if finite.size else math.inf,
        "nearest_target_same_bucket_rate": float((target_labels[predicted_target_vertex] == source_labels).mean()),
        "geodesic_method": args.geodesic_method,
        "ground_truth_label_source": args.ground_truth_label_source,
        "deformed_mesh": str(run.deformed_mesh.relative_to(PROJECT_ROOT)) if run.deformed_mesh.is_relative_to(PROJECT_ROOT) else str(run.deformed_mesh),
        "target_mesh": str(run.target_mesh.relative_to(PROJECT_ROOT)) if run.target_mesh.is_relative_to(PROJECT_ROOT) else str(run.target_mesh),
    }
    row.update(pck)
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "method",
        "pair",
        "run_name",
        "paper_style_auc_0_to_0p25",
        "paper_style_err_x100",
        "paper_style_p50_x100",
        "paper_style_p95_x100",
        "nearest_target_same_bucket_rate",
        "geodesic_method",
        "ground_truth_label_source",
        "target_vertices",
        "deformed_mesh",
    ]
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: Sequence[Dict[str, object]], failures: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["method"]), []).append(row)

    with path.open("w") as handle:
        handle.write("# DenseCorr3D Kim/Princeton Normalized Geodesic Semantic AUC\n\n")
        handle.write(
            "This table induces a correspondence from each deformed source vertex to its nearest target vertex, "
            "then measures normalized target-surface geodesic distance to the correct DenseCorr3D semantic group. "
            "AUC is integrated from 0 to 0.25 normalized geodesic error. `Err` is mean normalized geodesic error x100.\n\n"
        )
        handle.write(
            "The default `graph` geodesic uses Dijkstra shortest paths on target mesh edges, then divides by "
            "`sqrt(target surface area)`, matching the Kim/Princeton normalized geodesic-error convention used by "
            "DenseMatcher. This is metric-compatible with the DenseMatcher/DenseCorr3D evaluation idea, but the "
            "current rows are only your selected animal deformation runs, not the full DenseCorr3D benchmark split "
            "from the paper.\n\n"
        )
        if rows and str(rows[0].get("ground_truth_label_source")) == "densecorr3d_raw":
            handle.write(
                "These rows use DenseCorr3D `groups.txt` only at evaluation time. The deformation may have been "
                "driven by another label source, such as PartField, which avoids using the evaluation buckets as "
                "the deformation oracle.\n\n"
            )
        handle.write("## Method Summary\n\n")
        handle.write("| Method | n | mean AUC | mean Err | mean same-bucket rate |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for method, method_rows in sorted(groups.items()):
            aucs = np.asarray([float(row["paper_style_auc_0_to_0p25"]) for row in method_rows])
            errs = np.asarray([float(row["paper_style_err_x100"]) for row in method_rows])
            same = np.asarray([float(row["nearest_target_same_bucket_rate"]) for row in method_rows])
            handle.write(f"| {method} | {len(method_rows)} | {aucs.mean():.4f} | {errs.mean():.4f} | {same.mean():.4f} |\n")
        handle.write("\n## Run Summary\n\n")
        handle.write("| Pair | Method | Geodesic | AUC | Err | p95 Err | same-bucket rate | target verts | deformed mesh |\n")
        handle.write("|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        f"`{row['pair']}`",
                        str(row["method"]),
                        str(row["geodesic_method"]),
                        format_float(row["paper_style_auc_0_to_0p25"]),
                        format_float(row["paper_style_err_x100"]),
                        format_float(row["paper_style_p95_x100"]),
                        format_float(row["nearest_target_same_bucket_rate"]),
                        str(row["target_vertices"]),
                        f"`{row['deformed_mesh']}`",
                    ]
                )
                + " |\n"
            )
        if failures:
            handle.write("\n## Skipped\n\n")
            for failure in failures:
                handle.write(f"- `{failure['path']}`: {failure['error']}\n")


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.output_dir)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    paths = collect_paths(args)
    for idx, path in enumerate(paths, start=1):
        try:
            print(f"[{idx}/{len(paths)}] Kim/Princeton geodesic AUC: {path}")
            rows.append(evaluate_path(path, args))
        except RuntimeError as exc:
            if str(exc).startswith("filtered_by_"):
                continue
            print(f"Warning: skipped {path}: {exc}")
            failures.append({"path": path, "error": str(exc)})
        except Exception as exc:
            print(f"Warning: failed {path}: {exc}")
            failures.append({"path": path, "error": str(exc)})
    if not rows:
        raise SystemExit("No rows evaluated.")
    csv_path = out_dir / f"{args.output_prefix}.csv"
    markdown_path = out_dir / f"{args.output_prefix}.md"
    write_csv(csv_path, rows)
    write_markdown(markdown_path, rows, failures)
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
