"""Evaluate MeshUp deformations with the TOSCA benchmark protocol.

The script converts a final deformed mesh into a dense TOSCA `.map` by assigning
each deformed source vertex to its nearest target vertex in the optimizer's
unit-sized coordinate frame. TOSCA same-class ground truth is same vertex id, so
the official benchmark binary can score that map with the benchmark's geodesic
distance implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs_with_target_guidance.tosca_prepare_meshes import read_off


ArrayPair = Tuple[np.ndarray, np.ndarray]


@dataclass
class ToscaRun:
    run_dir: Path
    source_mesh: Path
    target_mesh: Path
    deformed_mesh: Path
    pair_slug: str
    bucket: Optional[int]


def parse_tosca_name(path: str | Path) -> Tuple[str, int]:
    stem = Path(path).stem
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", stem)
    if not match:
        raise ValueError(f"Cannot parse TOSCA name from {path}. Expected e.g. wolf0.obj.")
    return match.group(1), int(match.group(2))


def load_obj(path: Path) -> ArrayPair:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with path.open("r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
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
        raise ValueError(f"Could not read vertices/faces from {path}.")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def load_mesh(path: Path) -> ArrayPair:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj(path)
    if suffix == ".off":
        return read_off(path)
    raise ValueError(f"Unsupported mesh format {path}. Expected .obj or .off.")


def unit_vertices(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    scale = 2.0 / max(float((vmax - vmin).max()), 1e-8)
    return (vertices - (vmax + vmin) * 0.5) * scale


def nearest_vertices(query: np.ndarray, target: np.ndarray, chunk_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.empty(query.shape[0], dtype=np.int64)
    dists = np.empty(query.shape[0], dtype=np.float64)
    target64 = target.astype(np.float64, copy=False)
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        chunk = query[start:end].astype(np.float64, copy=False)
        diff = chunk[:, None, :] - target64[None, :, :]
        sq = np.einsum("ijk,ijk->ij", diff, diff)
        idx = np.argmin(sq, axis=1)
        indices[start:end] = idx
        dists[start:end] = np.sqrt(sq[np.arange(end - start), idx])
    return indices, dists


def auc_from_errors(errors: np.ndarray, threshold: float) -> float:
    clipped = np.clip(threshold - errors, 0.0, threshold)
    return float(clipped.mean() / threshold)


def parse_bracket_value(line: str) -> Optional[str]:
    start = line.find("[")
    end = line.rfind("]")
    if start < 0 or end <= start:
        return None
    return line[start + 1 : end].strip()


def parse_result_file(path: Path, auc_threshold: float) -> Dict[str, object]:
    sections: Dict[str, Dict[str, object]] = {}
    current: Optional[str] = None
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if line in {"Stats:", "StatsWithFlip:"}:
            current = line[:-1]
            sections[current] = {}
            continue
        if current is None or "[" not in line:
            continue
        key = line.split("[", 1)[0].strip()
        value = parse_bracket_value(line)
        if value is None:
            continue
        if key == "PerSampleErrors":
            vals = [float(tok) for tok in re.split(r"\s*,\s*", value) if tok]
            errors = np.asarray(vals, dtype=np.float64)
            sections[current][key] = vals
            sections[current]["AUC@0.25"] = auc_from_errors(errors, auc_threshold)
            sections[current]["p50"] = float(np.percentile(errors, 50))
            sections[current]["p95"] = float(np.percentile(errors, 95))
            sections[current]["p99"] = float(np.percentile(errors, 99))
            sections[current]["n_samples"] = int(errors.shape[0])
        else:
            try:
                sections[current][key] = float(value)
            except ValueError:
                sections[current][key] = value
    return sections


def find_run_dirs(run_roots: Iterable[Path]) -> List[Path]:
    runs: List[Path] = []
    for root in run_roots:
        if (root / "config.yml").is_file() and (root / "mesh_final" / "mesh.obj").is_file():
            runs.append(root)
            continue
        for cfg in sorted(root.rglob("config.yml")):
            run_dir = cfg.parent
            if (run_dir / "mesh_final" / "mesh.obj").is_file():
                runs.append(run_dir)
    return sorted(set(runs))


def infer_bucket(run_dir: Path) -> Optional[int]:
    match = re.search(r"_pf(\d+)_", run_dir.name)
    return int(match.group(1)) if match else None


def load_run(run_dir: Path) -> ToscaRun:
    cfg_path = run_dir / "config.yml"
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    source = Path(cfg["mesh"])
    target = Path(cfg["target_mesh"])
    if not source.is_absolute():
        source = Path.cwd() / source
    if not target.is_absolute():
        target = Path.cwd() / target
    return ToscaRun(
        run_dir=run_dir,
        source_mesh=source,
        target_mesh=target,
        deformed_mesh=run_dir / "mesh_final" / "mesh.obj",
        pair_slug=f"{source.stem}_to_{target.stem}",
        bucket=infer_bucket(run_dir),
    )


def write_map(path: Path, target_indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(i)) for i in target_indices) + "\n")


def run_official_tosca(
    benchmark_root: Path,
    source_class: str,
    source_id: int,
    target_class: str,
    target_id: int,
    map_path: Path,
    out_dir: Path,
    alg_id: str,
    timeout: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "./TestErrorAuto",
        "TOSCA",
        source_class,
        str(source_id),
        "TOSCA",
        target_class,
        str(target_id),
        str(map_path.resolve()),
        str(out_dir.resolve()),
        alg_id,
    ]
    subprocess.run(
        cmd,
        cwd=str(benchmark_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=True,
    )
    expected = out_dir / f"BenchResult_{alg_id}_{source_class}{source_id}_to_{target_class}{target_id}.txt"
    if expected.is_file():
        return expected
    candidates = sorted(out_dir.glob("BenchResult_*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No BenchResult file written in {out_dir}")
    return candidates[-1]


def score_mesh_map(
    run: ToscaRun,
    mesh_path: Path,
    label: str,
    args: argparse.Namespace,
    source_class: str,
    source_id: int,
    target_class: str,
    target_id: int,
) -> Dict[str, object]:
    mesh_vertices, _mesh_faces = load_mesh(mesh_path)
    target_vertices, target_faces = load_mesh(run.target_mesh)
    mesh_unit = unit_vertices(mesh_vertices)
    target_unit = unit_vertices(target_vertices)
    target_indices, nn_dists = nearest_vertices(mesh_unit, target_unit, chunk_size=args.nearest_chunk_size)

    map_path = Path(args.output_dir) / "maps" / f"{run.run_dir.name}_{label}.map"
    write_map(map_path, target_indices)
    official_dir = Path(args.output_dir) / "official" / f"{run.run_dir.name}_{label}"
    result_file = run_official_tosca(
        benchmark_root=Path(args.benchmark_root),
        source_class=source_class,
        source_id=source_id,
        target_class=target_class,
        target_id=target_id,
        map_path=map_path,
        out_dir=official_dir,
        alg_id=f"{label}_{run.run_dir.name[:80]}",
        timeout=args.official_timeout,
    )
    parsed = parse_result_file(result_file, auc_threshold=args.auc_threshold)
    stats = parsed.get("Stats", {})
    stats_flip = parsed.get("StatsWithFlip", {})
    identity_rate = float((target_indices == np.arange(target_indices.shape[0])).mean())

    return {
        "run_dir": str(run.run_dir),
        "mesh_label": label,
        "pair": run.pair_slug,
        "bucket": run.bucket,
        "source": f"{source_class}{source_id}",
        "target": f"{target_class}{target_id}",
        "vertices": int(mesh_vertices.shape[0]),
        "target_faces": int(target_faces.shape[0]),
        "map": str(map_path),
        "official_result": str(result_file),
        "nearest_identity_rate": identity_rate,
        "nearest_mean_euclidean": float(nn_dists.mean()),
        "nearest_p95_euclidean": float(np.percentile(nn_dists, 95)),
        "official_error": float(stats.get("Average", math.nan)),
        "official_err_x100": float(stats.get("Average", math.nan)) * 100.0,
        "official_auc": float(stats.get("AUC@0.25", math.nan)),
        "official_p50": float(stats.get("p50", math.nan)),
        "official_p95": float(stats.get("p95", math.nan)),
        "official_p99": float(stats.get("p99", math.nan)),
        "official_samples": int(stats.get("n_samples", 0) or 0),
        "official_flip_error": float(stats_flip.get("Average", math.nan)),
        "official_flip_err_x100": float(stats_flip.get("Average", math.nan)) * 100.0,
        "official_flip_auc": float(stats_flip.get("AUC@0.25", math.nan)),
        "official_flip_samples": int(stats_flip.get("n_samples", 0) or 0),
    }


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> List[Dict[str, object]]:
    run = load_run(run_dir)
    source_class, source_id = parse_tosca_name(run.source_mesh)
    target_class, target_id = parse_tosca_name(run.target_mesh)
    if source_class != target_class:
        raise ValueError(
            f"TOSCA ground truth is same-class only, got {source_class}{source_id} -> {target_class}{target_id}."
        )

    rows = [
        score_mesh_map(
            run=run,
            mesh_path=run.deformed_mesh,
            label="final",
            args=args,
            source_class=source_class,
            source_id=source_id,
            target_class=target_class,
            target_id=target_id,
        )
    ]
    if args.include_source_baseline:
        rows.append(
            score_mesh_map(
                run=run,
                mesh_path=run.source_mesh,
                label="source_baseline",
                args=args,
                source_class=source_class,
                source_id=source_id,
                target_class=target_class,
                target_id=target_id,
            )
        )
    return rows


def format_float(value: object, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(f):
        return ""
    return f"{f:.{digits}f}"


def write_tables(rows: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tosca_results.json"
    tsv_path = output_dir / "tosca_results.tsv"
    md_path = output_dir / "tosca_results.md"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    columns = [
        "mesh_label",
        "pair",
        "bucket",
        "vertices",
        "official_auc",
        "official_err_x100",
        "official_p95",
        "nearest_identity_rate",
        "nearest_mean_euclidean",
        "run_dir",
        "map",
        "official_result",
    ]
    with tsv_path.open("w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")

    md_columns = [
        ("mesh_label", "mesh"),
        ("pair", "pair"),
        ("bucket", "PF"),
        ("vertices", "verts"),
        ("official_auc", "AUC@0.25"),
        ("official_err_x100", "Err x100"),
        ("official_p95", "p95"),
        ("nearest_identity_rate", "same-ID NN"),
    ]
    lines = [
        "# TOSCA Correspondence Evaluation",
        "",
        "Official TOSCA scoring uses the benchmark binary on generated dense `.map` files. "
        "The map is produced by nearest target vertex after MeshUp's unit-size normalization.",
        "",
        "| " + " | ".join(label for _key, label in md_columns) + " |",
        "| " + " | ".join("---" for _ in md_columns) + " |",
    ]
    for row in rows:
        cells = []
        for key, _label in md_columns:
            if key in {"official_auc", "official_err_x100", "official_p95", "nearest_identity_rate"}:
                cells.append(format_float(row.get(key)))
            else:
                cells.append(str(row.get(key, "")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", f"JSON: `{json_path}`", f"TSV: `{tsv_path}`"])
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")
    print(f"Wrote {tsv_path}")
    print(f"Wrote {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", default=[], help="A MeshUp output run directory. Repeatable.")
    parser.add_argument("--run-root", action="append", default=[], help="Search recursively for completed runs.")
    parser.add_argument(
        "--benchmark-root",
        default="external/TOSCA/CorrsBenchmark",
        help="Surface Correspondence Benchmark root containing TestErrorAuto.",
    )
    parser.add_argument(
        "--output-dir",
        default="jobs_with_target_guidance/cross_animal_spike_runs/evaluation/tosca",
        help="Where maps and result tables are written.",
    )
    parser.add_argument("--auc-threshold", type=float, default=0.25)
    parser.add_argument("--nearest-chunk-size", type=int, default=512)
    parser.add_argument("--official-timeout", type=int, default=180)
    parser.add_argument("--include-source-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_roots = [Path(p) for p in args.run_dir] + [Path(p) for p in args.run_root]
    run_dirs = find_run_dirs(run_roots)
    if not run_dirs:
        raise SystemExit("No completed MeshUp runs found.")

    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        try:
            print(f"Evaluating {run_dir}")
            rows.extend(evaluate_run(run_dir, args))
        except Exception as exc:
            print(f"Skipping {run_dir}: {exc}")
    write_tables(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
