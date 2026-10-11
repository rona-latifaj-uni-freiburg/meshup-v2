"""Build the new comparison figures for report_final from the canonical
2026-08-18 MeshUp-vs-DenseMatcher fair-comparison CSVs and the extended
SAM3D/car surface-quality evaluation run in this session.

Palette follows a small, consistent, colorblind-safe set used across every
chart in this report:
  MeshUp   = deep teal   #1b7a72
  DenseMatcher = warm slate #8a6d3b (muted amber/brown, clearly distinct)
  clean provenance = teal, assisted = amber, unverified = grey
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUCKET_DIR = ROOT / "jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy"
OUT = ROOT / "report_final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MESHUP = "#1b7a72"
DENSEMATCHER = "#b5651d"
CLEAN = "#1b7a72"
ASSISTED = "#c99a2e"
UNVERIFIED = "#8a8f98"
INK = "#20242c"
MUTED = "#6b7280"
GRID = "#e4e7eb"
PAGE_BG = "#fbfaf7"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": PAGE_BG,
        "axes.facecolor": PAGE_BG,
        "savefig.facecolor": PAGE_BG,
    }
)


def per_category_chart() -> None:
    pf = pd.read_csv(BUCKET_DIR / "fair_comparison_meshup_vs_densematcher_pf_summary.csv")
    grouped = []
    for category, rows in pf.groupby("category", sort=True):
        n = rows["n"].astype(float)
        total = n.sum()
        grouped.append(
            {
                "category": category,
                "meshup_auc": (rows["meshup_auc"] * n).sum() / total,
                "meshup_err": (rows["meshup_err"] * n).sum() / total,
                "densematcher_auc": (rows["densematcher_auc"] * n).sum() / total,
                "densematcher_err": (rows["densematcher_err"] * n).sum() / total,
            }
        )
    df = pd.DataFrame(grouped)

    df = df.sort_values("meshup_auc", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 8.6), sharey=True, gridspec_kw={"wspace": 0.06})
    y = np.arange(len(df))
    bar_h = 0.36

    ax = axes[0]
    ax.barh(y + bar_h / 2, df["meshup_auc"], height=bar_h, color=MESHUP, label="This pipeline", zorder=3)
    ax.barh(y - bar_h / 2, df["densematcher_auc"], height=bar_h, color=DENSEMATCHER, label="DenseMatcher", zorder=3)
    ax.set_yticks(y)
    labels = [f"{c}" for c in df["category"]]
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("AUC (higher is better)")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("AUC per category, zero PF selection", fontsize=11, loc="left", color=INK)

    ax2 = axes[1]
    ax2.barh(y + bar_h / 2, df["meshup_err"], height=bar_h, color=MESHUP, zorder=3)
    ax2.barh(y - bar_h / 2, df["densematcher_err"], height=bar_h, color=DENSEMATCHER, zorder=3)
    ax2.set_xlabel("Err, mean normalized geodesic error (lower is better)")
    ax2.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.set_title("Err per category, zero PF selection", fontsize=11, loc="left", color=INK)

    fig.suptitle(
        "This pipeline vs. DenseMatcher on identical directed pairs, per category",
        fontsize=13,
        x=0.02,
        ha="left",
        color=INK,
        y=0.985,
    )
    fig.text(
        0.02,
        0.955,
        "Broccoli and eggplant genuinely lose under zero PF selection.",
        fontsize=9.5,
        color=MUTED,
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=MESHUP),
        plt.Rectangle((0, 0), 1, 1, color=DENSEMATCHER),
    ]
    fig.tight_layout(rect=(0, 0.035, 1, 0.9))
    fig.legend(
        legend_handles,
        ["This pipeline", "DenseMatcher"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.savefig(OUT / "densematcher_category_comparison.png", dpi=200)
    plt.close(fig)
    print("wrote densematcher_category_comparison.png")


def auc_spread_histogram() -> None:
    df = pd.read_csv(BUCKET_DIR / "fair_comparison_meshup_vs_densematcher_matched_rows.csv")
    dm = df.drop_duplicates(subset=["category", "source_id", "target_id", "map_direction"])["densematcher_auc"]
    mu = df["meshup_auc"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.4), sharex=True, gridspec_kw={"hspace": 0.28})
    bins = np.linspace(0, 1, 41)

    ax1.hist(mu, bins=bins, color=MESHUP, zorder=3)
    ax1.axvline(mu.median(), color=INK, ls="--", lw=1.2, zorder=4)
    ax1.text(mu.median() + 0.015, ax1.get_ylim()[1] if False else 0, "", visible=False)
    ax1.set_ylabel("Count")
    ax1.set_title(
        f"This pipeline: n={len(mu)} (pair, PF-setting) rows -- mean {mu.mean():.3f}, median {mu.median():.3f}",
        fontsize=11,
        loc="left",
        color=INK,
    )

    ax2.hist(dm, bins=bins, color=DENSEMATCHER, zorder=3)
    ax2.axvline(dm.median(), color=INK, ls="--", lw=1.2, zorder=4)
    ax2.set_xlabel("AUC (higher is better)")
    ax2.set_ylabel("Count")
    ax2.set_title(
        f"DenseMatcher: n={len(dm)} unique matched pairs -- mean {dm.mean():.3f}, median {dm.median():.3f}",
        fontsize=11,
        loc="left",
        color=INK,
    )

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 1.0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "This pipeline clusters at high AUC; DenseMatcher spreads across the range (dashed line = median)",
        fontsize=12.5,
        x=0.02,
        ha="left",
        color=INK,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.955))
    fig.savefig(OUT / "auc_spread_histogram.png", dpi=200)
    plt.close(fig)
    print("wrote auc_spread_histogram.png")


def sam3d_extended_chart() -> None:
    # (label, chamfer, f_at_002, p95, group)
    rows = [
        ("Horse2→Giraffe4", 0.0134, 0.9923, 0.01287, "animal"),
        ("Fox→Cat2", 0.0224, 0.9460, 0.02087, "animal"),
        ("Panda3→Bear", 0.0199, 0.9730, 0.01831, "animal"),
        ("Panda2→Panda3", 0.0261, 0.8859, 0.02512, "animal"),
        ("Cat2→Chihuahua2", 0.0246, 0.9312, 0.02425, "animal"),
        ("Cat→Fox3", 0.0243, 0.8985, 0.02436, "animal"),
        ("Goldie→Fox", 0.0228, 0.9503, 0.02027, "animal"),
        ("Goldie→Pug", 0.0248, 0.9056, 0.02359, "animal"),
        ("Panther→Horse2", 0.0231, 0.8916, 0.02832, "chain"),
        ("Horse2→D.Giraffe", 0.0189, 0.9692, 0.01894, "chain"),
    ]
    df = pd.DataFrame(rows, columns=["pair", "chamfer", "fscore", "p95", "group"])
    df = df.sort_values(["group", "fscore"], ascending=[True, True]).reset_index(drop=True)
    color = {"animal": MESHUP, "chain": "#4c74a6"}

    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = np.arange(len(df))
    ax.barh(y, df["fscore"], color=[color[g] for g in df["group"]], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["pair"], fontsize=9.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F@0.02 (higher is better)")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for yi, val in zip(y, df["fscore"]):
        ax.text(val + 0.012, yi, f"{val:.3f}", va="center", fontsize=8.5, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color[g]) for g in ("animal", "chain")]
    ax.legend(
        handles,
        ["SAM3D animal pairs (8)", "Cross-dataset chain (2)"],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9.5,
    )
    ax.set_title(
        "Surface-quality F@0.02, every evaluated SAM3D animal pair",
        fontsize=11,
        loc="left",
        color=INK,
    )
    fig.text(
        0.01,
        0.01,
        "No DenseCorr3D ground truth exists for these self-created meshes, so only geometric surface\n"
        "metrics (Chamfer, F@0.02, p95) are reported here, never AUC/Err.",
        fontsize=8.6,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.075, 0.84, 1))
    fig.savefig(OUT / "sam3d_extended_fscore.png", dpi=200)
    plt.close(fig)
    print("wrote sam3d_extended_fscore.png")


if __name__ == "__main__":
    per_category_chart()
    auc_spread_histogram()
    sam3d_extended_chart()
