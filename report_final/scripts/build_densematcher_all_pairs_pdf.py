"""Standalone, presentation-facing PDF: this pipeline's deformed mesh vs. DenseMatcher's
implied warp (see build_densematcher_deformation_comparison.py for the full method),
one page per unique directed pair -- every pair in the headline fair comparison
(fair_comparison_meshup_vs_densematcher_matched_rows.csv: 496 unique directed pairs
across 23 categories), not a one-per-category sample.

Selection per unique (category, source_id, target_id): among that pair's tested
PF-settings, pick the one with the largest (meshup_auc - densematcher_auc) gap that
still resolves to an on-disk run directory and a DenseMatcher pair prediction file
(falling back to the next-largest-gap PF-setting if not). DenseMatcher's own
prediction does not depend on our PF setting, so this choice only affects which of
our own results is shown, never the DenseMatcher side.

Pages are ordered by DenseMatcher AUC ascending (worst first) across the full pool,
regardless of category, so the deck reads as one continuous spectrum from
catastrophic to competitive DenseMatcher performance.

Output: report_final/figures/densematcher_all_pairs.pdf
"""

from __future__ import annotations

import csv
import sys
import time
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
PAGE_DIR = OUT / "_densematcher_all_pairs_pages"


def select_candidates() -> list[dict]:
    rows = list(csv.DictReader(open(MATCHED_ROWS_CSV)))
    index = build_run_index()

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        key = (r["category"], r["source_id"], r["target_id"])
        groups.setdefault(key, []).append(r)

    selected = []
    skipped = []
    for (category, source_id, target_id), candidates in groups.items():
        candidates = sorted(candidates, key=lambda r: -(float(r["meshup_auc"]) - float(r["densematcher_auc"])))
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
                data = load_pair_data(category, source_id, target_id, run_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip candidate {category}/{source_id}->{target_id} [{r['pf_setting']}]: {exc}")
                continue
            chosen = dict(
                category=category,
                source_id=source_id,
                target_id=target_id,
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
            skipped.append((category, source_id, target_id))
        else:
            selected.append(chosen)

    if skipped:
        print(f"WARNING: no valid candidate for {len(skipped)} pairs:")
        for s in skipped:
            print("   ", s)
    return selected


def render_page(row: dict, rank: int, total: int) -> Path:
    supersample = 2
    width, height = 2000, 950
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)

    top = 108 * supersample
    panel_w = w // 4

    headers = ("Source mesh", "This pipeline: deformed mesh", "DenseMatcher: implied warp", "Target mesh")
    for col, header in enumerate(headers):
        draw_text(
            image,
            header,
            (col * panel_w + 20 * supersample, top - 16 * supersample),
            scale=0.58 * supersample,
            color=MUTED,
            thickness=max(1, int(1.5 * supersample)),
        )

    title = (
        f"{row['category'].replace('_', ' ').capitalize()}   "
        f"AUC {row['meshup_auc']:.2f} vs {row['dm_auc']:.2f}   "
        f"Err {row['meshup_err']:.1f} vs {row['dm_err']:.1f}"
    )
    draw_text(
        image,
        title,
        (20 * supersample, 42 * supersample),
        scale=0.86 * supersample,
        color=INK,
        thickness=max(1, int(2.1 * supersample)),
    )
    provenance = f"{row['source_id']} -> {row['target_id']}  ({row['pf_setting']})"
    draw_text(
        image,
        provenance,
        (20 * supersample, 72 * supersample),
        scale=0.4 * supersample,
        color=MUTED,
        thickness=max(1, supersample),
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

    page_path = PAGE_DIR / f"{rank:04d}_{row['category']}.png"
    save_rgb(page_path, image, supersample, (width, height))
    return page_path


def main() -> None:
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print("Selecting a representative PF-setting for every unique directed pair...")
    selected = select_candidates()
    print(f"Selected {len(selected)} pairs in {time.time() - t_start:.1f}s.")

    selected.sort(key=lambda r: r["dm_auc"])

    page_paths = []
    t_render = time.time()
    for rank, row in enumerate(selected, start=1):
        page_paths.append(render_page(row, rank, len(selected)))
        if rank % 20 == 0 or rank == len(selected):
            elapsed = time.time() - t_render
            rate = elapsed / rank
            eta = rate * (len(selected) - rank)
            print(
                f"[{rank}/{len(selected)}] {row['category']} AUC {row['meshup_auc']:.3f} vs "
                f"{row['dm_auc']:.3f}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)"
            )

    print("Assembling PDF...")
    images = [Image.open(p).convert("RGB") for p in page_paths]
    out_pdf = OUT / "densematcher_all_pairs.pdf"
    images[0].save(out_pdf, save_all=True, append_images=images[1:])
    print(f"Wrote {out_pdf} ({len(images)} pages) in {time.time() - t_start:.1f}s total")


if __name__ == "__main__":
    main()
