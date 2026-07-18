#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_long_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image
    scale = max_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray(image)
    return np.array(pil.resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8)


def save_binary_mask(mask_bool: np.ndarray, out_path: Path) -> None:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a single-object binary mask using SAM2 automatic mask generator"
    )
    parser.add_argument("--base_path", type=str, default=".", help="Workspace base path")
    parser.add_argument("--image_name", type=str, required=True, help="Image filename in image/")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/sam2.1-hiera-small",
        help="SAM2 model id from Hugging Face",
    )
    parser.add_argument(
        "--prefer_center",
        action="store_true",
        help="Bias selection toward masks near image center (useful for single centered object)",
    )
    parser.add_argument(
        "--points_per_side",
        type=int,
        default=32,
        help="Sampling density for automatic mask generation (lower is faster)",
    )
    parser.add_argument(
        "--crop_n_layers",
        type=int,
        default=1,
        help="Number of crop layers for mask generation (0 is fastest)",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=0,
        help="If >0, resize image so longer side is max_side for faster CPU preview",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut for fast preview: points_per_side=16, crop_n_layers=0, max_side=1024",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path).expanduser().resolve()
    image_path = base_path / "image" / args.image_name
    mask_dir = base_path / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / args.image_name

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_image_rgb(image_path)
    if args.fast:
        args.points_per_side = 16
        args.crop_n_layers = 0
        if args.max_side <= 0:
            args.max_side = 1024
    if args.max_side > 0:
        image = resize_long_side(image, args.max_side)

    h, w = image.shape[:2]

    # Import lazily so script shows a clear error if SAM2 is not installed yet.
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        args.model_id,
        device=device,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=args.crop_n_layers,
        output_mode="binary_mask",
    )

    masks = generator.generate(image)
    if not masks:
        raise RuntimeError("SAM2 produced no masks")

    cx, cy = w / 2.0, h / 2.0

    def score(ann: dict) -> float:
        area = float(ann.get("area", 0.0))
        mask = ann.get("segmentation")
        if mask is None:
            return area

        area_ratio = area / float(max(h * w, 1))
        if area_ratio >= 0.85:
            return 0.0

        penalty = 1.0
        top_frac = float(mask[0, :].mean())
        bottom_frac = float(mask[-1, :].mean())
        left_frac = float(mask[:, 0].mean())
        right_frac = float(mask[:, -1].mean())

        # Penalize masks that look like broad background/ground regions.
        if top_frac > 0.45:
            penalty *= 0.2
        if left_frac > 0.2 and right_frac > 0.2:
            penalty *= 0.3
        if bottom_frac > 0.75 and area_ratio > 0.22:
            penalty *= 0.35
        if area_ratio > 0.6:
            penalty *= 0.3

        bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
        bx, by, bw, bh = [float(v) for v in bbox]
        aspect = bw / max(bh, 1.0)
        if aspect > 5.0:
            penalty *= 0.5

        if penalty <= 0.0:
            return 0.0

        if not args.prefer_center:
            return area * penalty
        bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
        bcx = bx + bw / 2.0
        bcy = by + bh / 2.0
        dist = ((bcx - cx) ** 2 + (bcy - cy) ** 2) ** 0.5
        norm_dist = dist / max((w**2 + h**2) ** 0.5, 1.0)
        return area * (1.0 - 0.6 * norm_dist) * penalty

    best = max(masks, key=score)
    best_mask = best["segmentation"]
    save_binary_mask(best_mask, mask_path)

    print(f"[DONE] Wrote mask: {mask_path}")
    print(f"[INFO] Total candidate masks: {len(masks)}")
    print(f"[INFO] Selected mask area: {int(best.get('area', 0))}")


if __name__ == "__main__":
    main()
