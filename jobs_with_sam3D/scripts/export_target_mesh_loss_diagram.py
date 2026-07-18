#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path


def box(ax, x, y, w, h, text, fc="#f5f7fb"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.5,
        edgecolor="#222222",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)


def arrow(ax, x1, y1, x2, y2):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.5,
        color="#222222",
    )
    ax.add_patch(patch)


def main():
    out_path = Path("jobs_with_sam3D/outputs/diagrams/target_mesh_loss_pipeline.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8), dpi=200)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    box(ax, 0.5, 5.6, 3.0, 1.2, "Source mesh\n(current deformed)")
    box(ax, 0.5, 2.0, 3.0, 1.2, "Target mesh")

    box(ax, 4.2, 5.6, 3.3, 1.2, "Render source views\n+ DINO features")
    box(ax, 4.2, 2.0, 3.3, 1.2, "Render target views\n+ DINO features")
    arrow(ax, 3.5, 6.2, 4.2, 6.2)
    arrow(ax, 3.5, 2.6, 4.2, 2.6)

    box(ax, 8.2, 3.8, 3.2, 1.4, "View-aligned\nfeature comparison")
    arrow(ax, 7.5, 6.2, 8.2, 4.9)
    arrow(ax, 7.5, 2.6, 8.2, 4.1)

    box(ax, 12.0, 5.0, 3.2, 1.1, "Global term\n(1 - cosine CLS)")
    box(ax, 12.0, 3.4, 3.2, 1.1, "Spatial term\n(bidirectional patches)")
    box(ax, 12.0, 1.8, 3.2, 1.1, "Warmup +\nweights applied")
    arrow(ax, 11.4, 4.8, 12.0, 5.55)
    arrow(ax, 11.4, 4.5, 12.0, 3.95)
    arrow(ax, 11.4, 4.2, 12.0, 2.35)

    box(ax, 8.2, 0.7, 3.2, 1.1, "Target mesh\nguidance loss")
    box(ax, 12.0, 0.5, 3.2, 1.5, "Add to total loss:\nSDS + reg + target")
    arrow(ax, 13.6, 1.8, 10.0, 1.8)
    arrow(ax, 11.4, 1.25, 12.0, 1.25)

    ax.text(
        8.0,
        7.35,
        "Target Mesh Correspondence Loss Pipeline",
        fontsize=16,
        weight="bold",
        ha="center",
    )
    ax.text(
        8.0,
        6.95,
        "Simplified V2 (view-aligned DINO global + spatial terms)",
        fontsize=11,
        ha="center",
        color="#333333",
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
