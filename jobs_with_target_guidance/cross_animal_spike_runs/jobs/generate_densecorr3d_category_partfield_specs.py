#!/usr/bin/env python3
"""Generate DenseCorr3D category benchmark manifests and PartField deformation specs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


PROMPTS = {
    "apple": "an apple",
    "backpack": "a backpack",
    "banana": "a banana",
    "bell_pepper": "a bell pepper",
    "bread": "a bread loaf",
    "broccoli": "a broccoli",
    "carrot": "a carrot",
    "cars": "a toy car",
    "celery": "a celery",
    "chairs": "a chair",
    "cucumber": "a cucumber",
    "egg": "an egg",
    "eggplant": "an eggplant",
    "knife": "a knife",
    "mushroom": "a mushroom",
    "onion": "an onion",
    "peach": "a peach",
    "pear": "a pear",
    "potato": "a potato",
    "pumpkin": "a pumpkin",
    "shampoo": "a shampoo bottle",
    "tomato": "a tomato",
    "zucchini": "a zucchini",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", default="external/DenseCorr3D/test_files.txt")
    parser.add_argument("--densecorr3d-root", default="external/DenseCorr3D")
    parser.add_argument("--run-name", default="densecorr3d_nonanimals_test_20260806")
    parser.add_argument("--buckets", default="04,05,06,07,08")
    parser.add_argument("--exclude-category", action="append", default=["animals"])
    parser.add_argument("--category", action="append", help="Optional category filter; may be repeated.")
    parser.add_argument(
        "--manifest-output",
        default="jobs_with_target_guidance/cross_animal_spike_runs/jobs/densecorr3d_nonanimals_test_manifest.tsv",
    )
    parser.add_argument(
        "--category-list-output",
        default="jobs_with_target_guidance/cross_animal_spike_runs/jobs/densecorr3d_nonanimals_test_categories.txt",
    )
    parser.add_argument(
        "--spec-output",
        default="jobs_with_target_guidance/cross_animal_spike_runs/jobs/densecorr3d_nonanimals_test_partfield_fair_pf0408_specs.tsv",
    )
    return parser.parse_args()


def slug_piece(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def read_split(args: argparse.Namespace) -> dict[str, list[str]]:
    root = Path(args.densecorr3d_root)
    allowed = set(args.category or [])
    excluded = set(args.exclude_category or [])
    categories: dict[str, list[str]] = defaultdict(list)
    for raw_line in Path(args.split_file).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        category, object_id = line.split("/", 1)
        if category in excluded:
            continue
        if allowed and category not in allowed:
            continue
        object_dir = root / category / object_id
        required = [object_dir / "simple_mesh.obj", object_dir / "groups.txt"]
        if all(path.is_file() for path in required):
            categories[category].append(object_id)
    return {category: objects for category, objects in sorted(categories.items())}


def build_manifest(categories: dict[str, list[str]], args: argparse.Namespace) -> list[dict[str, str]]:
    feature_root = Path("jobs_with_target_guidance/partfield_features/densecorr3d_categories") / args.run_name
    segment_root = Path("jobs_with_target_guidance/partfield_segments/densecorr3d_categories") / args.run_name
    counters: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    for category, objects in categories.items():
        for object_id in objects:
            index = counters[category]
            counters[category] += 1
            name = f"{slug_piece(category)}_m{index:03d}_{slug_piece(object_id)[:8]}"
            rows.append(
                {
                    "category": category,
                    "object_id": object_id,
                    "name": name,
                    "mesh": str(Path(args.densecorr3d_root) / category / object_id / "simple_mesh.obj"),
                    "groups": str(Path(args.densecorr3d_root) / category / object_id / "groups.txt"),
                    "feature": str(feature_root / category / f"part_feat_{name}_0_batch.npy"),
                    "feature_alt": str(feature_root / category / f"part_feat_{name}_0.npy"),
                    "segment_root": str(segment_root / category),
                    "prompt": PROMPTS.get(category, f"a {category.replace('_', ' ')} object"),
                }
            )
    return rows


def write_manifest(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    output = Path(args.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    categories = sorted({row["category"] for row in rows})
    category_output = Path(args.category_list_output)
    category_output.parent.mkdir(parents=True, exist_ok=True)
    category_output.write_text("\n".join(categories) + "\n")


def write_specs(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    buckets = [bucket.strip() for bucket in args.buckets.split(",") if bucket.strip()]
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    output = Path(args.spec_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "# pair_slug",
        "source_mesh",
        "target_mesh",
        "source_name",
        "target_name",
        "bucket",
        "source_labels",
        "target_labels",
        "source_features",
        "target_features",
        "prompt",
    ]
    spec_rows: list[list[str]] = [fields]
    for category, items in sorted(by_category.items()):
        for source in items:
            for target in items:
                if source["name"] == target["name"]:
                    continue
                pair_slug = f"{slug_piece(category)}__{source['name']}_to_{target['name']}"
                for bucket in buckets:
                    label_dir = Path(target["segment_root"]).parent / category / f"buckets_{bucket}" / "labels"
                    spec_rows.append(
                        [
                            pair_slug,
                            source["mesh"],
                            target["mesh"],
                            source["name"],
                            target["name"],
                            bucket,
                            str(label_dir / f"{source['name']}_partfield_labels.npz"),
                            str(label_dir / f"{target['name']}_partfield_labels.npz"),
                            source["feature"],
                            target["feature"],
                            target["prompt"],
                        ]
                    )
    output.write_text("\n".join("\t".join(row) for row in spec_rows) + "\n")


def main() -> None:
    args = parse_args()
    categories = read_split(args)
    rows = build_manifest(categories, args)
    if not rows:
        raise SystemExit("No DenseCorr3D meshes selected.")
    write_manifest(rows, args)
    write_specs(rows, args)
    directed_pairs = sum(len(items) * (len(items) - 1) for items in categories.values())
    buckets = [bucket.strip() for bucket in args.buckets.split(",") if bucket.strip()]
    print(f"manifest={args.manifest_output}")
    print(f"category_list={args.category_list_output}")
    print(f"spec={args.spec_output}")
    print(f"categories={len(categories)}")
    print(f"meshes={len(rows)}")
    print(f"directed_pairs={directed_pairs}")
    print(f"bucket_settings={len(buckets)} ({', '.join(buckets)})")
    print(f"spec_rows={directed_pairs * len(buckets)}")
    for category, items in categories.items():
        n = len(items)
        print(f"{category}\tmeshes={n}\tdirected_pairs={n * (n - 1)}\truns={n * (n - 1) * len(buckets)}")


if __name__ == "__main__":
    main()
