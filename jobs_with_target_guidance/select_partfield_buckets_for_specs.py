#!/usr/bin/env python3
"""Filter PartField sweep specs through the optional bucket selector.

Input specs keep the existing tab-separated shape used by the DenseCorr3D/SAM3D
jobs:

    pair_slug source_mesh target_mesh source_name target_name bucket ...

The output is the same format, but only rows whose bucket is selected for that
source/target pair are retained.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs_with_target_guidance.partfield_bucket_llm import (  # noqa: E402
    DEFAULT_CALIBRATION_CSV,
    BucketDecision,
    choose_buckets,
    infer_category_from_text,
    decision_to_public_dict,
    infer_category_from_path,
    normalize_category,
    parse_bucket_list,
)


DEFAULT_FIELDS = [
    "pair_slug",
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


def strip_comment_field(value: str) -> str:
    return value[2:] if value.startswith("# ") else value.lstrip("#").strip()


def read_spec(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text().splitlines()
    header: Optional[list[str]] = None
    data_lines: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        fields = line.split("\t")
        if header is None and fields[0].startswith("#"):
            header = [strip_comment_field(field) if idx == 0 else field for idx, field in enumerate(fields)]
            continue
        if fields[0].startswith("#"):
            continue
        data_lines.append(line)

    if header is None:
        header = DEFAULT_FIELDS
    rows: list[dict[str, str]] = []
    for line in data_lines:
        values = line.split("\t")
        if len(values) != len(header):
            raise ValueError(f"Malformed spec row in {path}: expected {len(header)} fields, got {len(values)}: {line}")
        rows.append(dict(zip(header, values)))
    return header, rows


def write_spec(path: Path, header: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = ["\t".join(["# " + header[0], *header[1:]])]
    for row in rows:
        rendered.append("\t".join(row.get(field, "") for field in header))
    path.write_text("\n".join(rendered) + "\n")


def bucket_value(row: dict[str, str]) -> int:
    return int(row["bucket"])


def pair_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("pair_slug", ""),
        row.get("source_mesh", ""),
        row.get("target_mesh", ""),
        row.get("source_name", ""),
        row.get("target_name", ""),
    )


def row_category(row: dict[str, str]) -> Optional[str]:
    source_category = infer_category_from_path(Path(row.get("source_mesh", "")))
    target_category = infer_category_from_path(Path(row.get("target_mesh", "")))
    prompt_category = infer_category_from_text(row.get("prompt", ""))
    return normalize_category(source_category or target_category or prompt_category)


def summarize_decision(decision: BucketDecision, pair: tuple[str, str, str, str, str]) -> dict[str, Any]:
    public = decision_to_public_dict(decision)
    return {
        "pair_slug": pair[0],
        "source_mesh": pair[1],
        "target_mesh": pair[2],
        "source_name": pair[3],
        "target_name": pair[4],
        "bucket": decision.bucket,
        "buckets": decision.buckets,
        "mode": decision.mode,
        "provider": decision.provider,
        "confidence": decision.confidence,
        "rationale": decision.rationale,
        "category_guess": decision.category_guess,
        "warnings": public["warnings"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Input sweep spec TSV.")
    parser.add_argument("--output", required=True, help="Filtered output spec TSV.")
    parser.add_argument("--decisions-output", help="Optional JSONL file with one bucket decision per pair.")
    parser.add_argument("--mode", choices=["auto", "fixed", "range"], default="range")
    parser.add_argument("--user-request", default="Select PartField buckets for this source-target mesh pair.")
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "heuristic", "mock"],
        default="auto",
        help="Use openai for a real LLM, heuristic for offline calibrated tests, or auto for OPENAI_API_KEY if present.",
    )
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--mock-response", help="JSON object used with --provider mock.")
    parser.add_argument("--range-size", type=int, default=3)
    parser.add_argument(
        "--candidate-buckets",
        default="",
        help="Optional global allowed bucket list. Defaults to buckets present for each pair in the input spec.",
    )
    parser.add_argument("--category", action="append", default=[], help="Optional category allow-list; may repeat.")
    parser.add_argument("--limit-pairs", type=int, default=0, help="Stop after this many selected pairs, for smoke tests.")
    parser.add_argument(
        "--max-pairs-per-category",
        type=int,
        default=0,
        help="Optional cap per category, useful when dry-running many DenseCorr3D categories.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print decisions and do not write the filtered spec.")
    parser.add_argument("--calibration-csv", action="append", default=[])
    parser.add_argument("--no-default-calibration", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec)
    header, rows = read_spec(spec_path)
    required = {"pair_slug", "source_mesh", "target_mesh", "bucket"}
    missing = required - set(header)
    if missing:
        raise SystemExit(f"{spec_path} is missing required field(s): {', '.join(sorted(missing))}")

    requested_categories = {normalize_category(category) for category in args.category}
    requested_categories.discard(None)
    global_candidates = parse_bucket_list(args.candidate_buckets, default=[]) if args.candidate_buckets else None
    calibration_paths = [Path(path) for path in args.calibration_csv]
    if not args.no_default_calibration:
        calibration_paths.append(DEFAULT_CALIBRATION_CSV)

    grouped: "OrderedDict[tuple[str, str, str, str, str], list[dict[str, str]]]" = OrderedDict()
    category_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        category = row_category(row)
        if requested_categories and category not in requested_categories:
            continue
        key = pair_key(row)
        if key not in grouped:
            if args.max_pairs_per_category and category:
                if category_counts[category] >= args.max_pairs_per_category:
                    continue
                category_counts[category] += 1
            if args.limit_pairs and len(grouped) >= args.limit_pairs:
                continue
            grouped[key] = []
        grouped[key].append(row)

    kept_rows: list[dict[str, str]] = []
    decisions: list[dict[str, Any]] = []
    for key, pair_rows in grouped.items():
        rows_by_bucket = {bucket_value(row): row for row in pair_rows}
        candidates = sorted(rows_by_bucket)
        if global_candidates is not None:
            candidates = [bucket for bucket in candidates if bucket in global_candidates]
        if not candidates:
            continue
        first = pair_rows[0]
        decision = choose_buckets(
            source_mesh=first["source_mesh"],
            target_mesh=first["target_mesh"],
            prompt=first.get("prompt", ""),
            user_request=args.user_request,
            request_mode=args.mode,
            candidate_buckets=candidates,
            calibration_paths=calibration_paths,
            provider=args.provider,
            model=args.model,
            api_base=args.api_base,
            timeout=args.timeout,
            range_size=args.range_size,
            mock_response=args.mock_response,
        )
        selected = [bucket for bucket in decision.buckets if bucket in rows_by_bucket]
        if not selected:
            selected = [decision.bucket]
        for bucket in selected:
            kept_rows.append(rows_by_bucket[bucket])
        decisions.append(summarize_decision(decision, key))

    if args.dry_run:
        for item in decisions:
            print(json.dumps(item, sort_keys=True))
        return

    write_spec(Path(args.output), header, kept_rows)
    if args.decisions_output:
        decisions_path = Path(args.decisions_output)
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        decisions_path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in decisions) + "\n")

    print(f"input_rows={len(rows)}")
    print(f"selected_pairs={len(grouped)}")
    print(f"output_rows={len(kept_rows)}")
    print(f"output={args.output}")
    if args.decisions_output:
        print(f"decisions={args.decisions_output}")


if __name__ == "__main__":
    main()
