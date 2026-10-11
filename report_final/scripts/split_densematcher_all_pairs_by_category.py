"""Split the already-rendered densematcher_all_pairs.pdf page images into one
PDF per category, so a reader can open e.g. just the mushroom pairs instead of
paging through all 491. Reuses the PNGs already written by
build_densematcher_all_pairs_pdf.py under
report_final/figures/_densematcher_all_pairs_pages/ -- no mesh re-rendering,
just regrouping and patching the "X of 491" footer to "X of N" within each
category's own file (the original global-rank text is baked into the raster,
so it's patched in place with an opaque rectangle + redraw rather than
re-rendered from the mesh data).

Output: report_final/figures/densematcher_pairs_by_category/<category>.pdf
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))
from make_review_figures import MUTED, PAGE_BG  # noqa: E402

PAGE_DIR = ROOT / "report_final" / "figures" / "_densematcher_all_pairs_pages"
OUT_DIR = ROOT / "report_final" / "figures" / "densematcher_pairs_by_category"

# Footer text lives at final-image coords (2000x950 pages); see render_page()
# in build_densematcher_all_pairs_pdf.py for the original placement.
PATCH_RECT = (1280, 12, 2000, 60)  # x0, y0, x1, y1
FOOTER_ORIGIN = (1380, 42)


def patch_footer(png_path: Path, rank: int, total: int) -> Path:
    image = cv2.imread(str(png_path))
    x0, y0, x1, y1 = PATCH_RECT
    cv2.rectangle(image, (x0, y0), (x1, y1), PAGE_BG, -1)
    text = f"{rank} of {total}, ordered worst to best DenseMatcher AUC"
    cv2.putText(image, text, FOOTER_ORIGIN, cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1, cv2.LINE_AA)
    patched_path = png_path.parent / f"_patched_{png_path.name}"
    cv2.imwrite(str(patched_path), image)
    return patched_path


def main() -> None:
    pages = sorted(PAGE_DIR.glob("*.png"))
    pages = [p for p in pages if not p.name.startswith("_patched_")]
    if not pages:
        raise SystemExit(f"No page PNGs found under {PAGE_DIR}")

    by_category: dict[str, list[Path]] = defaultdict(list)
    pattern = re.compile(r"^\d+_(.+)\.png$")
    for page in pages:
        m = pattern.match(page.name)
        if not m:
            print(f"WARNING: could not parse category from {page.name}, skipping")
            continue
        by_category[m.group(1)].append(page)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(by_category)} categories, {len(pages)} total pages")

    for category in sorted(by_category):
        group = by_category[category]  # already in global rank order = ascending DenseMatcher AUC
        total = len(group)
        patched = [patch_footer(p, rank, total) for rank, p in enumerate(group, start=1)]
        images = [Image.open(p).convert("RGB") for p in patched]
        out_pdf = OUT_DIR / f"{category}.pdf"
        images[0].save(out_pdf, save_all=True, append_images=images[1:])
        for p in patched:
            p.unlink()
        print(f"  {category}: {total} pages -> {out_pdf}")

    print(f"Wrote {len(by_category)} PDFs to {OUT_DIR}")


if __name__ == "__main__":
    main()
