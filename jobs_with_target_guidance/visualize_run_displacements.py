#!/usr/bin/env python3
"""Create displacement-arrow diagnostics from saved MeshUp correspondence files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/meshup_mplconfig")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import trimesh


EPOCH_RE = re.compile(r"correspondence_epoch_(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save per-epoch displacement visualizations for completed MeshUp runs. "
            "Each run must contain correspondence/correspondence_epoch_*.json."
        )
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run output directories")
    parser.add_argument(
        "--output-name",
        default="displacement_viz",
        help="Output subdirectory created inside each run directory",
    )
    parser.add_argument(
        "--max-arrows",
        type=int,
        default=1500,
        help="Maximum number of vertices to draw as quiver arrows per PNG",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of largest-displacement vertices to save per epoch",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Optional exact epoch numbers to visualize; missing epochs are reported",
    )
    parser.add_argument(
        "--no-ply",
        action="store_true",
        help="Only save PNG/CSV diagnostics, not displacement-coloured PLY meshes",
    )
    return parser.parse_args()


def epoch_from_path(path: Path) -> int:
    match = EPOCH_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse epoch from {path}")
    return int(match.group(1))


def load_correspondence(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r") as f:
        data = json.load(f)
    original = np.asarray(data["original_vertices"], dtype=np.float32)
    deformed = np.asarray(data["deformed_vertices"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int64)
    if original.shape != deformed.shape:
        raise ValueError(f"Vertex shape mismatch in {path}: {original.shape} vs {deformed.shape}")
    return original, deformed, faces


def nocs_colors(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    return (vertices - vmin) / np.maximum(vmax - vmin, 1e-8)


def save_displacement_ply(
    out_path: Path,
    deformed: np.ndarray,
    faces: np.ndarray,
    magnitude: np.ndarray,
) -> None:
    vmax = float(magnitude.max()) if magnitude.size and magnitude.max() > 0 else 1.0
    colours_rgba = (cm.jet(magnitude / vmax) * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(
        vertices=deformed,
        faces=faces,
        vertex_colors=colours_rgba,
        process=False,
    )
    mesh.export(out_path)


def set_view_limits(ax: plt.Axes, base: np.ndarray, deformed: np.ndarray, ha: int, va: int) -> None:
    coords = np.concatenate([base[:, [ha, va]], deformed[:, [ha, va]]], axis=0)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.08 * span
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])


def save_quiver_png(
    out_path: Path,
    epoch: int,
    base: np.ndarray,
    deformed: np.ndarray,
    displacement: np.ndarray,
    magnitude: np.ndarray,
    max_arrows: int,
) -> None:
    if len(base) > max_arrows:
        idx = np.random.default_rng(42).choice(len(base), max_arrows, replace=False)
    else:
        idx = np.arange(len(base))

    nocs_sub = nocs_colors(base)[idx]
    views = [
        (0, 2, 0, 2, "XZ (top-down)", "X", "Z"),
        (0, 1, 0, 1, "XY (front)", "X", "Y"),
        (2, 1, 2, 1, "ZY (side)", "Z", "Y"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Displacement vectors - epoch {epoch} | "
        f"mean={magnitude.mean():.4e} max={magnitude.max():.4e} "
        f"median={np.median(magnitude):.4e}",
        fontsize=11,
    )

    for ax, (ha, va, dh, dv, title, xlabel, ylabel) in zip(axes, views):
        ax.scatter(base[idx, ha], base[idx, va], s=2, c=nocs_sub, alpha=0.25)
        ax.quiver(
            base[idx, ha],
            base[idx, va],
            displacement[idx, dh],
            displacement[idx, dv],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            alpha=0.8,
            color=nocs_sub,
        )
        set_view_limits(ax, base, deformed, ha, va)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary_plot(out_path: Path, rows: list[dict[str, float]]) -> None:
    epochs = np.asarray([row["epoch"] for row in rows], dtype=np.float32)
    mean = np.asarray([row["mean"] for row in rows], dtype=np.float32)
    median = np.asarray([row["median"] for row in rows], dtype=np.float32)
    p95 = np.asarray([row["p95"] for row in rows], dtype=np.float32)
    max_vals = np.asarray([row["max"] for row in rows], dtype=np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(epochs, mean, marker="o", label="mean")
    ax.plot(epochs, median, marker="o", label="median")
    ax.plot(epochs, p95, marker="o", label="p95")
    ax.plot(epochs, max_vals, marker="o", label="max")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Displacement magnitude")
    ax.set_title("Displacement magnitude over saved epochs")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_top_vertices(
    out_path: Path,
    epoch: int,
    base: np.ndarray,
    deformed: np.ndarray,
    displacement: np.ndarray,
    magnitude: np.ndarray,
    top_k: int,
) -> None:
    top_idx = np.argsort(magnitude)[::-1][:top_k]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "rank",
                "vertex",
                "magnitude",
                "base_x",
                "base_y",
                "base_z",
                "deformed_x",
                "deformed_y",
                "deformed_z",
                "dx",
                "dy",
                "dz",
            ]
        )
        for rank, vertex_id in enumerate(top_idx, start=1):
            writer.writerow(
                [
                    epoch,
                    rank,
                    int(vertex_id),
                    float(magnitude[vertex_id]),
                    *[float(v) for v in base[vertex_id]],
                    *[float(v) for v in deformed[vertex_id]],
                    *[float(v) for v in displacement[vertex_id]],
                ]
            )


def process_run(
    run_dir: Path,
    output_name: str,
    max_arrows: int,
    top_k: int,
    save_ply: bool,
    epochs: list[int] | None,
) -> None:
    corr_dir = run_dir / "correspondence"
    if not corr_dir.is_dir():
        raise FileNotFoundError(f"Missing correspondence directory: {corr_dir}")

    all_corr_files = sorted(corr_dir.glob("correspondence_epoch_*.json"), key=epoch_from_path)
    if not all_corr_files:
        raise FileNotFoundError(f"No correspondence_epoch_*.json files in {corr_dir}")
    if epochs is None:
        corr_files = all_corr_files
        missing_epochs: list[int] = []
    else:
        by_epoch = {epoch_from_path(path): path for path in all_corr_files}
        corr_files = [by_epoch[epoch] for epoch in epochs if epoch in by_epoch]
        missing_epochs = [epoch for epoch in epochs if epoch not in by_epoch]

    out_dir = run_dir / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if missing_epochs:
        missing_path = out_dir / "missing_requested_epochs.txt"
        missing_path.write_text(
            "Missing requested correspondence files for epochs:\n"
            + " ".join(str(epoch) for epoch in missing_epochs)
            + "\n"
        )
        print(f"{run_dir}: missing requested epochs {missing_epochs}")
    if not corr_files:
        print(f"{run_dir}: no requested epochs available to visualize")
        return

    rows: list[dict[str, float]] = []
    for corr_path in corr_files:
        epoch = epoch_from_path(corr_path)
        base, deformed, faces = load_correspondence(corr_path)
        displacement = deformed - base
        magnitude = np.linalg.norm(displacement, axis=1)
        rows.append(
            {
                "epoch": float(epoch),
                "mean": float(magnitude.mean()),
                "median": float(np.median(magnitude)),
                "p95": float(np.percentile(magnitude, 95)),
                "max": float(magnitude.max()),
            }
        )

        save_quiver_png(
            out_dir / f"disp_epoch_{epoch:04d}.png",
            epoch,
            base,
            deformed,
            displacement,
            magnitude,
            max_arrows=max_arrows,
        )
        if save_ply:
            save_displacement_ply(out_dir / f"disp_epoch_{epoch:04d}.ply", deformed, faces, magnitude)
        save_top_vertices(
            out_dir / f"top_displacements_epoch_{epoch:04d}.csv",
            epoch,
            base,
            deformed,
            displacement,
            magnitude,
            top_k=top_k,
        )

    stats_path = out_dir / ("displacement_stats_requested.csv" if epochs is not None else "displacement_stats.csv")
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "mean", "median", "p95", "max"])
        writer.writeheader()
        writer.writerows(rows)
    summary_name = "displacement_summary_requested.png" if epochs is not None else "displacement_summary.png"
    save_summary_plot(out_dir / summary_name, rows)
    print(f"{run_dir}: wrote {len(corr_files)} epochs to {out_dir}")


def main() -> None:
    args = parse_args()
    for run_dir in args.run_dirs:
        process_run(
            run_dir=run_dir,
            output_name=args.output_name,
            max_arrows=args.max_arrows,
            top_k=args.top_k,
            save_ply=not args.no_ply,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
