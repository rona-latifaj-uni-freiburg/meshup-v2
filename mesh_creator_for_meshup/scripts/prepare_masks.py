#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def binarize_mask(mask_img: Image.Image, threshold: int) -> Image.Image:
    mask = np.array(mask_img)
    if mask.ndim == 3:
        # Keep the last channel because SAM3D load_mask() uses the last channel for 3-channel masks.
        mask = mask[..., -1]
    binary = (mask >= threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize masks in BASE_PATH/mask so they are binary and size-aligned to BASE_PATH/image"
    )
    parser.add_argument("--base_path", type=str, default=".", help="Project base path")
    parser.add_argument("--threshold", type=int, default=1, help="Foreground threshold in [0,255]")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report what would be changed, without writing files",
    )

    args = parser.parse_args()

    if not (0 <= args.threshold <= 255):
        raise ValueError("--threshold must be in [0, 255]")

    base_path = Path(args.base_path).expanduser().resolve()
    image_dir = base_path / "image"
    mask_dir = base_path / "mask"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

    changed = 0
    skipped = 0

    for image_path in sorted(p for p in image_dir.iterdir() if p.is_file()):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            skipped += 1
            print(f"[SKIP] No matching mask for {image_path.name}")
            continue

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        original_size = mask.size
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

        binary_mask = binarize_mask(mask, threshold=args.threshold)

        if args.dry_run:
            print(
                f"[DRY] {mask_path.name}: size {original_size} -> {binary_mask.size}, threshold={args.threshold}"
            )
            changed += 1
            continue

        binary_mask.save(mask_path)
        changed += 1
        print(f"[OK] Wrote {mask_path.name}: size {original_size} -> {binary_mask.size}")

    print(f"[DONE] Processed={changed}, Skipped(no matching mask)={skipped}")


if __name__ == "__main__":
    main()
