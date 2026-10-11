"""Standalone, presentation-facing PDF: this pipeline's Final mesh vs. DenseMatcher's
implied warp (see build_densematcher_deformation_comparison.py for the full method),
one page per category, covering every category in the headline fair comparison
(fair_comparison_meshup_vs_densematcher_matched_rows.csv) rather than the 3-pair
subset used in the report figure.

Selection per category: among that category's matched (source, target, PF-setting)
rows, pick the one with the largest (meshup_auc - densematcher_auc) gap that still
resolves to an on-disk run directory, DenseCorr3D source/target mesh, and a
DenseMatcher pair prediction file (falling back to the next-largest-gap row if not).

Pages are ordered by DenseMatcher AUC ascending (worst first) so the deck reads as
a spectrum from catastrophic to competitive DenseMatcher performance, with this
pipeline's output shown alongside every step of that spectrum.

Output: report_final/figures/densematcher_all_categories.pdf
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_review_figures import INK, MUTED, PAGE_BG, draw_text, render_panel, save_rgb  # noqa: E402
from densematcher_compare_lib import (  # noqa: E402
    MATCHED_ROWS_CSV,
    best_azimuth,
    build_run_index,
    load_pair_data,
)

OUT = ROOT / "report_final" / "figures"
PAGE_DIR = OUT / "_densematcher_all_categories_pages"


def select_candidates() -> list[dict]:
    rows = list(csv.DictReader(open(MATCHED_ROWS_CSV)))
    categories = sorted(set(r["category"] for r in rows))
    index = build_run_index()

    selected = []
    skipped = []
    for category in categories:
        candidates = [r for r in rows if r["category"] == category]
        candidates.sort(key=lambda r: -(float(r["meshup_auc"]) - float(r["densematcher_auc"])))
        chosen = None
        for r in candidates:
            run_path = index.get(r["run_name"])
            if run_path is None:
                continue
            if not (run_path / "colored_meshes" / "mesh_initial.ply").is_file():
                continue
            if not (run_path / "colored_meshes" / "mesh_final_position.ply").is_file():
                continue
            try:
                data = load_pair_data(category, r["source_id"], r["target_id"], run_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip candidate {category}/{r['source_id']}->{r['target_id']}: {exc}")
                continue
            chosen = dict(
                category=category,
                source_id=r["source_id"],
                target_id=r["target_id"],
                pf_setting=r["pf_setting"],
                run_path=run_path,
                meshup_auc=float(r["meshup_auc"]),
                meshup_err=float(r["meshup_err"]),
                dm_auc=float(r["densematcher_auc"]),
                dm_err=float(r["densematcher_err"]),
                data=data,
            )
            break
        if chosen is None:
            skipped.append(category)
        else:
            selected.append(chosen)

    if skipped:
        print(f"WARNING: no valid candidate found for categories: {skipped}")
    return selected


def render_page(row: dict, rank: int, total: int) -> Path:
    supersample = 2
    width, height = 2000, 950
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)

    top = 92 * supersample
    panel_w = w // 4

    headers = ("Source", "Ours: Final", "DenseMatcher: warp", "Target")
    for col, header in enumerate(headers):
        draw_text(
            image,
            header,
            (col * panel_w + 20 * supersample, top - 16 * supersample),
            scale=0.62 * supersample,
            color=MUTED,
            thickness=max(1, int(1.6 * supersample)),
        )

    caption = (
        f"{row['category'].replace('_', ' ').capitalize()}   "
        f"AUC {row['meshup_auc']:.2f} vs {row['dm_auc']:.2f}   "
        f"Err {row['meshup_err']:.1f} vs {row['dm_err']:.1f}"
    )
    draw_text(
        image,
        caption,
        (20 * supersample, 42 * supersample),
        scale=0.9 * supersample,
        color=INK,
        thickness=max(1, int(2.2 * supersample)),
    )
    footer = f"{rank} of {total}, ordered worst to best DenseMatcher AUC"
    draw_text(
        image,
        footer,
        (w - 620 * supersample, 42 * supersample),
        scale=0.5 * supersample,
        color=MUTED,
        thickness=max(1, int(1.3 * supersample)),
    )

    data = row["data"]
    azimuth = best_azimuth(data["src"].vertices)
    panel_top = top + 14 * supersample
    panel_h = h - panel_top - 16 * supersample
    for col, mesh in enumerate((data["src"], data["final"], data["dm"], data["target"])):
        render_panel(
            image,
            (col * panel_w, panel_top, panel_w, panel_h),
            mesh,
            "",
            azimuth=azimuth,
            elevation=8.0,
            zoom=0.88,
        )

    page_path = PAGE_DIR / f"{rank:02d}_{row['category']}.png"
    save_rgb(page_path, image, supersample, (width, height))
    return page_path


def main() -> None:
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("Selecting one representative pair per category...")
    selected = select_candidates()
    print(f"Selected {len(selected)} categories.")

    selected.sort(key=lambda r: r["dm_auc"])

    page_paths = []
    for rank, row in enumerate(selected, start=1):
        print(
            f"[{rank}/{len(selected)}] {row['category']}: "
            f"AUC {row['meshup_auc']:.3f} vs {row['dm_auc']:.3f}"
        )
        page_paths.append(render_page(row, rank, len(selected)))

    images = [Image.open(p).convert("RGB") for p in page_paths]
    out_pdf = OUT / "densematcher_all_categories.pdf"
    images[0].save(out_pdf, save_all=True, append_images=images[1:])
    print(f"Wrote {out_pdf} ({len(images)} pages)")


if __name__ == "__main__":
    main()
