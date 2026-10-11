#!/usr/bin/env python3
"""Generate PartField-guided fair DenseCorr3D all-10 deformation specs."""

from __future__ import annotations

import argparse
from pathlib import Path


OBJECTS = [
    ("panther", "071b8_toy_animals_017", "a panther"),
    ("animal13cf7", "13cf7_toy_animals_055", "an animal"),
    ("animal188f4", "188f4_toy_animals_022", "an animal"),
    ("moose", "1d6d1_toy_animals_015", "a moose"),
    ("elephant", "2d6b3_toy_animals_009", "an elephant"),
    ("giraffe", "34fb4_toy_animals_019", "a giraffe"),
    ("animal4e94a", "4e94a_toy_animals_020", "an animal"),
    ("bear", "96615_toy_animals_018", "a bear"),
    ("cheetah", "bdfd0_toy_animals_016", "a cheetah"),
    ("dinosaur", "d1889_dinosaur_002", "a dinosaur"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        default="densecorr3d_all10_20260804",
        help="PartField feature/segment run name.",
    )
    parser.add_argument(
        "--buckets",
        default="06,07,08",
        help="Comma-separated bucket tokens to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=(
            "jobs_with_target_guidance/cross_animal_spike_runs/jobs/"
            "densecorr3d_all10_partfield_fair_pf060708_specs.tsv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    buckets = [item.strip() for item in args.buckets.split(",") if item.strip()]
    feature_dir = Path("jobs_with_target_guidance/partfield_features/densecorr3d_animals") / args.run_name
    segment_root = Path("jobs_with_target_guidance/partfield_segments/densecorr3d_animals") / args.run_name / "all_ten"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        [
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
    ]
    for source_name, source_object, _source_prompt in OBJECTS:
        for target_name, target_object, target_prompt in OBJECTS:
            if source_name == target_name:
                continue
            pair_slug = f"{source_name}_to_{target_name}"
            source_mesh = f"external/DenseCorr3D/animals/{source_object}/simple_mesh.obj"
            target_mesh = f"external/DenseCorr3D/animals/{target_object}/simple_mesh.obj"
            source_features = str(feature_dir / f"part_feat_{source_name}_0_batch.npy")
            target_features = str(feature_dir / f"part_feat_{target_name}_0_batch.npy")
            for bucket in buckets:
                label_dir = segment_root / f"buckets_{bucket}" / "labels"
                rows.append(
                    [
                        pair_slug,
                        source_mesh,
                        target_mesh,
                        source_name,
                        target_name,
                        bucket,
                        str(label_dir / f"{source_name}_partfield_labels.npz"),
                        str(label_dir / f"{target_name}_partfield_labels.npz"),
                        source_features,
                        target_features,
                        target_prompt,
                    ]
                )

    output.write_text("\n".join("\t".join(row) for row in rows) + "\n")
    print(f"Wrote {output}")
    print(f"Objects: {len(OBJECTS)}")
    print(f"Directed pairs: {len(OBJECTS) * (len(OBJECTS) - 1)}")
    print(f"Bucket settings: {len(buckets)} ({', '.join(buckets)})")
    print(f"Spec rows: {len(rows) - 1}")
    print(f"Best-PF final rows after evaluation: {len(OBJECTS) * (len(OBJECTS) - 1)}")


if __name__ == "__main__":
    main()
