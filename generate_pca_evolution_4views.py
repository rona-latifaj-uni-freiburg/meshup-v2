#!/usr/bin/env python3
"""
Build per-view PCA evolution grids from MeshUp epoch renders.

For each fixed camera view (front/side/back/side), this script:
1) Collects selected epoch render images.
2) Crops the foreground mesh from white background.
3) Runs test_image_pca.py jointly over the selected epochs for that view.

Output structure mirrors the existing manual workflow:
  <output_dir>/front_view_cropped_inputs/*.png
  <output_dir>/front_view_cropped/combined_pca_grid.png
  ... and similarly for side/back views.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ViewSpec:
    index: int
    input_dir_name: str
    output_dir_name: str
    output_stem: str


DEFAULT_VIEWS: List[ViewSpec] = [
    ViewSpec(0, "front_view_cropped_inputs", "front_view_cropped", "view0"),
    ViewSpec(1, "view_1_side_a_cropped_inputs", "view_1_side_a_cropped", "view1"),
    ViewSpec(2, "view_2_back_cropped_inputs", "view_2_back_cropped", "view2"),
    ViewSpec(3, "view_3_side_b_cropped_inputs", "view_3_side_b_cropped", "view3"),
]


def _parse_epochs_arg(epochs_str: str) -> Optional[List[int]]:
    epochs_str = epochs_str.strip()
    if not epochs_str or epochs_str.lower() == "auto":
        return None
    out: List[int] = []
    for tok in epochs_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return sorted(set(out))


def _list_epoch_dirs(epoch_renders_dir: Path) -> List[Path]:
    dirs = [d for d in epoch_renders_dir.glob("epoch_*") if d.is_dir()]
    return sorted(dirs)


def _epoch_number(epoch_dir: Path) -> int:
    # epoch_00001 -> 1
    return int(epoch_dir.name.split("_")[-1])


def _find_view_image(epoch_dir: Path, view_idx: int) -> Optional[Path]:
    matches = sorted(epoch_dir.glob(f"view_{view_idx}_*.png"))
    if not matches:
        return None
    return matches[0]


def _compute_foreground_bbox(
    img_np: np.ndarray,
    white_threshold: int,
    min_side: int,
    margin_ratio: float,
) -> Tuple[int, int, int, int]:
    # Foreground pixels are non-white due to white background compositing in render.
    fg_mask = np.any(img_np < white_threshold, axis=2)
    ys, xs = np.where(fg_mask)

    h, w = img_np.shape[:2]
    if len(xs) == 0 or len(ys) == 0:
        # Safe fallback: centered square crop.
        side = max(min_side, int(min(h, w) * 0.72))
        cx, cy = w // 2, h // 2
        left = max(0, cx - side // 2)
        top = max(0, cy - side // 2)
        right = min(w, left + side)
        bottom = min(h, top + side)
        return left, top, right, bottom

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    # Expand box with margin and optionally enforce a minimum size.
    bw = max(1, x_max - x_min + 1)
    bh = max(1, y_max - y_min + 1)
    margin_x = int(round(bw * margin_ratio))
    margin_y = int(round(bh * margin_ratio))

    left = max(0, x_min - margin_x)
    right = min(w, x_max + margin_x + 1)
    top = max(0, y_min - margin_y)
    bottom = min(h, y_max + margin_y + 1)

    cur_w = right - left
    cur_h = bottom - top
    if min_side > 0:
        if cur_w < min_side:
            pad = (min_side - cur_w) // 2
            left = max(0, left - pad)
            right = min(w, right + (min_side - (right - left)))
        if cur_h < min_side:
            pad = (min_side - cur_h) // 2
            top = max(0, top - pad)
            bottom = min(h, bottom + (min_side - (bottom - top)))

    return left, top, right, bottom


def _crop_image(
    src_path: Path,
    dst_path: Path,
    white_threshold: int,
    min_side: int,
    margin_ratio: float,
) -> None:
    img = Image.open(src_path).convert("RGB")
    img_np = np.asarray(img)
    left, top, right, bottom = _compute_foreground_bbox(
        img_np,
        white_threshold=white_threshold,
        min_side=min_side,
        margin_ratio=margin_ratio,
    )
    crop = img.crop((left, top, right, bottom))
    crop.save(dst_path)


def _run_combined_pca(
    test_image_pca_script: Path,
    image_paths: Iterable[Path],
    output_dir: Path,
    model: str,
    image_size: int,
) -> None:
    image_list = ",".join(str(p) for p in image_paths)
    cmd = [
        sys.executable,
        str(test_image_pca_script),
        "--images",
        image_list,
        "--output_dir",
        str(output_dir),
        "--model",
        model,
        "--image_size",
        str(image_size),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 4-view PCA evolution grids")
    parser.add_argument("--epoch_renders_dir", type=str, required=True, help="Path to epoch_renders directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to pca_evolution output directory")
    parser.add_argument(
        "--epochs",
        type=str,
        default="1,750,1500,2250,3000",
        help="Comma-separated epoch list, or 'auto' to use all available epochs",
    )
    parser.add_argument("--test_image_pca_script", type=str, default="test_image_pca.py", help="Path to test_image_pca.py")
    parser.add_argument("--model", type=str, default="dinov2_vitl14", help="DINO model for test_image_pca.py")
    parser.add_argument("--image_size", type=int, default=518, help="Image size for test_image_pca.py")
    parser.add_argument("--white_threshold", type=int, default=245, help="Foreground extraction white threshold")
    parser.add_argument("--min_crop_side", type=int, default=0, help="Minimum crop side in pixels (0 disables) ")
    parser.add_argument("--crop_margin_ratio", type=float, default=0.08, help="BBox expansion margin ratio")
    args = parser.parse_args()

    epoch_renders_dir = Path(args.epoch_renders_dir)
    output_dir = Path(args.output_dir)
    test_image_pca_script = Path(args.test_image_pca_script)

    if not epoch_renders_dir.is_dir():
        raise FileNotFoundError(f"epoch_renders directory not found: {epoch_renders_dir}")
    if not test_image_pca_script.is_file():
        raise FileNotFoundError(f"test_image_pca.py not found: {test_image_pca_script}")

    output_dir.mkdir(parents=True, exist_ok=True)

    selected_epochs = _parse_epochs_arg(args.epochs)
    epoch_dirs = _list_epoch_dirs(epoch_renders_dir)
    if not epoch_dirs:
        raise RuntimeError(f"No epoch directories found under {epoch_renders_dir}")

    if selected_epochs is not None:
        selected_set = set(selected_epochs)
        epoch_dirs = [d for d in epoch_dirs if _epoch_number(d) in selected_set]
        if not epoch_dirs:
            raise RuntimeError(f"None of requested epochs were found: {selected_epochs}")

    print(f"Found {len(epoch_dirs)} epochs to process")

    for view in DEFAULT_VIEWS:
        in_dir = output_dir / view.input_dir_name
        out_dir = output_dir / view.output_dir_name
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        cropped_paths: List[Path] = []
        for epoch_dir in epoch_dirs:
            epoch_num = _epoch_number(epoch_dir)
            src = _find_view_image(epoch_dir, view.index)
            if src is None:
                print(f"[WARN] Missing view_{view.index} in {epoch_dir.name}, skipping")
                continue

            dst = in_dir / f"epoch_{epoch_num:05d}_{view.output_stem}_crop.png"
            _crop_image(
                src,
                dst,
                white_threshold=args.white_threshold,
                min_side=args.min_crop_side,
                margin_ratio=args.crop_margin_ratio,
            )
            cropped_paths.append(dst)

        cropped_paths = sorted(cropped_paths)
        if not cropped_paths:
            print(f"[WARN] No cropped inputs for view_{view.index}, skipping PCA")
            continue

        print(f"Running combined PCA for view_{view.index}: {len(cropped_paths)} images")
        _run_combined_pca(
            test_image_pca_script=test_image_pca_script,
            image_paths=cropped_paths,
            output_dir=out_dir,
            model=args.model,
            image_size=args.image_size,
        )

    print(f"Done. PCA evolution outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
