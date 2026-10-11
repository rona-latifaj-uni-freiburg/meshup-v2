"""Render a PartField bucket-color comparison image for one deformation run.

Reads a run's own config.yml (written by main.py at the start of every run) to
find the source mesh, target mesh, and the PartField label files that drove
the deformation, then renders a three-panel PNG: source (colored by bucket),
deformed/final (colored by the same frozen bucket ids as the source), and
target (colored by its own bucket ids). Bucket colors use the same palette as
jobs_with_target_guidance/partfield_segment.py, so colors are consistent with
any other PartField visualization in this repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from jobs_with_target_guidance.partfield_segment import PALETTE, load_mesh


def load_labels(path: Path) -> np.ndarray:
    data = np.load(path)
    return data["vertex_labels"].astype(np.int64)


def face_colors_from_vertex_labels(labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_label_votes = labels[faces]
    face_labels = np.array(
        [np.bincount(row).argmax() for row in face_label_votes],
        dtype=np.int64,
    )
    colors = PALETTE[face_labels % len(PALETTE)].astype(np.float32) / 255.0
    return colors


def plot_panel(ax, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, title: str) -> None:
    pc = Poly3DCollection(vertices[faces], facecolor=colors, edgecolor="none", alpha=1.0)
    ax.add_collection3d(pc)
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def render(run_dir: Path, out_path: Path, elev: float, azim: float, vertical_axis: str = "y") -> None:
    config_path = run_dir / "config.yml"
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    source_mesh_path = Path(cfg["mesh"])
    target_mesh_path = Path(cfg["target_mesh"])
    source_labels_path = Path(cfg["partfield_source_labels"])
    target_labels_path = Path(cfg["partfield_target_labels"])
    n_buckets = cfg.get("partfield_n_buckets", "?")
    deformed_mesh_path = run_dir / "mesh_final" / "mesh.obj"

    source_v, source_f = load_mesh(source_mesh_path)
    target_v, target_f = load_mesh(target_mesh_path)
    deformed_v, deformed_f = load_mesh(deformed_mesh_path)

    source_labels = load_labels(source_labels_path)
    target_labels = load_labels(target_labels_path)

    source_colors = face_colors_from_vertex_labels(source_labels, source_f)
    target_colors = face_colors_from_vertex_labels(target_labels, target_f)
    # deformation preserves source vertex order/topology, so the source's
    # frozen bucket ids are still valid for coloring the deformed result
    deformed_colors = face_colors_from_vertex_labels(source_labels, deformed_f)

    fig, axarr = plt.subplots(1, 3, figsize=(15, 5.5), subplot_kw={"projection": "3d"})
    plot_panel(axarr[0], source_v, source_f, source_colors, f"Source (PF{n_buckets})")
    plot_panel(axarr[1], deformed_v, deformed_f, deformed_colors, "Deformed (source buckets)")
    plot_panel(axarr[2], target_v, target_f, target_colors, f"Target (PF{n_buckets})")
    for ax in axarr:
        # DenseCorr3D meshes realigned this session use +Y as the canonical "up"
        # axis, but matplotlib's view_init treats Z as vertical by default --
        # without this, an upright Y-up mesh (e.g. a chair) renders lying on its
        # side. vertical_axis (added in matplotlib 3.6) fixes that directly.
        ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)

    fig.suptitle(run_dir.name, fontsize=8, wrap=True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="A deformation run's output directory (contains config.yml and mesh_final/).")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path. Defaults to <run_dir>/partfield_bucket_visualization.png")
    parser.add_argument("--elev", type=float, default=15.0)
    parser.add_argument("--azim", type=float, default=30.0)
    parser.add_argument("--vertical-axis", choices=("x", "y", "z"), default="y")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or (args.run_dir / "partfield_bucket_visualization.png")
    render(args.run_dir, out_path, args.elev, args.azim, args.vertical_axis)


if __name__ == "__main__":
    main()
