"""Shared helpers for comparing this pipeline's deformation against DenseMatcher's
implied warp (a source-topology mesh built by moving each source vertex to its
DenseMatcher-predicted match). Used by:
  build_densematcher_deformation_comparison.py -- the 3-pair report figures.
  build_densematcher_all_categories_pdf.py -- the full 23-category supervisor PDF.

See build_densematcher_deformation_comparison.py's module docstring for the
full rationale (DenseMatcher predicts correspondence only, not a deformed
mesh; this is the same lookup evaluate_densematcher_pair_predictions.py uses
to score AUC/Err, rendered instead of only scored).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import (  # noqa: E402
    face_colors,
    load_colored_ply,
)
from jobs_with_target_guidance.evaluate_target_pipeline import load_mesh as load_obj_mesh  # noqa: E402

DENSECORR3D = ROOT / "external" / "DenseCorr3D"
PAIRS_ROOT = DENSECORR3D / "densecorr_test" / "pairs"
OUTPUTS_ROOT = ROOT / "jobs_with_target_guidance" / "cross_animal_spike_runs" / "outputs"
MATCHED_ROWS_CSV = (
    ROOT
    / "jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy"
    / "fair_comparison_meshup_vs_densematcher_matched_rows.csv"
)


def unit_vertices(v: np.ndarray) -> np.ndarray:
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    scale = 2.0 / max(float((vmax - vmin).max()), 1e-12)
    return (v - (vmax + vmin) * 0.5) * scale


def pos_color(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(0, keepdims=True), v.max(0, keepdims=True)
    rng = np.maximum(hi - lo, 1e-8)
    return ((v - lo) / rng).astype(np.float32)


def best_azimuth(vertices: np.ndarray, elevation: float = 8.0, step_deg: float = 5.0) -> float:
    """Yaw (around y-up) that maximizes the mesh's visible on-screen width.

    A rotation by 180 degrees mirrors the same width, so sweeping [0, 180) is
    sufficient. This is a content-neutral way to pick "the wide side" of an
    elongated object (e.g. a knife blade face rather than its thin edge)
    without hand-tuning a camera angle per pair.
    """
    best_az, best_extent = 0.0, -1.0
    for az in np.arange(0.0, 180.0, step_deg):
        a = np.deg2rad(az)
        ca, sa = np.cos(a), np.sin(a)
        x = ca * vertices[:, 0] + sa * vertices[:, 2]
        extent = float(x.max() - x.min())
        if extent > best_extent:
            best_extent = extent
            best_az = float(az)
    return best_az


def find_pair_npz(category: str, source_id: str, target_id: str) -> tuple[Path | None, bool | None]:
    root = PAIRS_ROOT / category
    forward = root / f"{source_id}__{target_id}.npz"
    if forward.is_file():
        return forward, False
    backward = root / f"{target_id}__{source_id}.npz"
    if backward.is_file():
        return backward, True
    return None, None


def source_to_target_correspondence(category: str, source_id: str, target_id: str, n_source: int) -> np.ndarray:
    """For each of our source's vertices, the index of its DenseMatcher-predicted match in our target."""
    path, reversed_pair = find_pair_npz(category, source_id, target_id)
    if path is None:
        raise FileNotFoundError(f"No DenseMatcher pair file for {category}/{source_id} <-> {target_id}")
    data = np.load(path, allow_pickle=False)
    key = "surface_map" if reversed_pair else "surface_map_inv"
    corr = np.asarray(data[key], dtype=np.int64)
    if corr.shape[0] != n_source:
        raise RuntimeError(f"{path} [{key}] length {corr.shape[0]} != source vertex count {n_source}")
    return corr


def resolve_run_target_mesh(run_path: Path, category: str, target_id: str) -> Path:
    """The exact target mesh file the run actually trained against.

    Some categories ("assisted" preparation in the report, e.g. mushroom,
    onion, peach, zucchini) train against a re-oriented copy under
    ``<category>_better_alignment/`` instead of the raw DenseCorr3D folder,
    so the run's own mesh_initial.ply/mesh_final_position.ply are only
    spatially aligned with THAT copy, not with the raw one. Reading the path
    straight out of the run's config.yml is what keeps Target aligned with
    Source/Final; falling back to the raw path (with a warning) only if the
    config can't be read.
    """
    config_path = run_path / "config.yml"
    if config_path.is_file():
        try:
            config = yaml.safe_load(config_path.read_text())
            target_mesh = config.get("target_mesh") if isinstance(config, dict) else None
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not parse {config_path}: {exc}")
            target_mesh = None
        if target_mesh:
            resolved = (ROOT / target_mesh).resolve()
            if resolved.is_file():
                return resolved
            print(f"WARNING: {config_path} target_mesh {target_mesh!r} does not exist, falling back to raw path")
    else:
        print(f"WARNING: no config.yml under {run_path}, falling back to raw path")
    return DENSECORR3D / category / target_id / "simple_mesh.obj"


def load_pair_data(category: str, source_id: str, target_id: str, run_path: Path) -> dict:
    """Build Source / Final / DenseMatcher-warp / Target meshes for one pair."""
    src_v, src_f, src_c = load_colored_ply(run_path / "colored_meshes" / "mesh_initial.ply")
    fin_v_raw, fin_f, _ = load_colored_ply(run_path / "colored_meshes" / "mesh_final_position.ply")
    fin_v = unit_vertices(fin_v_raw)

    target_mesh_path = resolve_run_target_mesh(run_path, category, target_id)
    tgt_raw_v, tgt_raw_f = load_obj_mesh(target_mesh_path)

    raw_dataset_path = DENSECORR3D / category / target_id / "simple_mesh.obj"
    if target_mesh_path != raw_dataset_path and raw_dataset_path.is_file():
        raw_n_verts, _ = load_obj_mesh(raw_dataset_path)
        if raw_n_verts.shape[0] != tgt_raw_v.shape[0]:
            raise RuntimeError(
                f"{target_mesh_path} has {tgt_raw_v.shape[0]} vertices but the raw dataset copy "
                f"{raw_dataset_path} (which DenseMatcher's correspondence indices refer to) has "
                f"{raw_n_verts.shape[0]} -- these are not the same mesh under a rigid realignment, "
                "so the DenseMatcher correspondence indices would not apply."
            )

    tgt_v = unit_vertices(tgt_raw_v)

    corr = source_to_target_correspondence(category, source_id, target_id, src_v.shape[0])
    if corr.max() >= tgt_v.shape[0]:
        raise RuntimeError(
            f"DenseMatcher correspondence for {category}/{source_id}->{target_id} references vertex "
            f"{corr.max()} but the target mesh only has {tgt_v.shape[0]} vertices"
        )
    dm_v = tgt_v[corr]

    tgt_c = pos_color(tgt_raw_v)

    return dict(
        src=Mesh(src_v, src_f, src_c, face_colors(src_f, src_c)),
        final=Mesh(fin_v, fin_f, src_c, face_colors(fin_f, src_c)),
        dm=Mesh(dm_v, src_f, src_c, face_colors(src_f, src_c)),
        target=Mesh(tgt_v, tgt_raw_f, tgt_c, face_colors(tgt_raw_f, tgt_c)),
        target_v=tgt_v,
    )


def face_edge_distortion(faces: np.ndarray, source_v: np.ndarray, deformed_v: np.ndarray) -> np.ndarray:
    """Per-face mean |log2(deformed edge length / source edge length)|.

    0 = locally rigid (no stretch/compression); large values mean that face's
    edges were crushed or stretched relative to the untouched source mesh.
    """
    eps = 1e-9
    total = np.zeros(faces.shape[0], dtype=np.float64)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        src_len = np.linalg.norm(source_v[faces[:, b]] - source_v[faces[:, a]], axis=1)
        def_len = np.linalg.norm(deformed_v[faces[:, b]] - deformed_v[faces[:, a]], axis=1)
        total += np.abs(np.log2((def_len + eps) / (src_len + eps)))
    return total / 3.0


def build_run_index() -> dict[str, Path]:
    """Map run directory name -> full path, for every run under cross_animal_spike_runs/outputs.

    Run outputs are laid out as outputs/<experiment_family>/<run_name>/..., and
    run_name alone (as recorded in the fair-comparison CSVs) is unique enough
    in practice; this lets category selection resolve a CSV row to an actual
    directory on disk without one filesystem walk per category.
    """
    index: dict[str, Path] = {}
    if not OUTPUTS_ROOT.is_dir():
        return index
    for family in OUTPUTS_ROOT.iterdir():
        if not family.is_dir():
            continue
        for run in family.iterdir():
            if run.is_dir():
                index.setdefault(run.name, run)
    return index
