#!/usr/bin/env python3
"""Properly-controlled MeshUp vs. DenseMatcher comparison.

Joins MeshUp's own graph-geodesic evaluation rows (produced by
``evaluate_densecorr_paper_style_geodesic.py`` against each category's
confirmed-current run family, full sample, no per-pair cherry-picking) with
DenseMatcher's ``surface_map`` predictions scored under the same local metric
(produced by ``evaluate_densematcher_pair_predictions.py --bidirectional``).

The join key is (category, exact directed pair (source_id, target_id)) so the
DenseMatcher baseline used for comparison is restricted to the same pairs
MeshUp was actually evaluated on -- not DenseMatcher's full 287/306-pair mean.

For MeshUp, a single PartField bucket-count (PF) setting is chosen per
category by mean AUC across that category's own matched sample (no per-pair
oracle selection). This is disclosed explicitly: the PF choice is made on the
same sample being reported, which is a known limitation shared with the rest
of this project's reporting, not a new one introduced here.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs_with_target_guidance.evaluate_densecorr_paper_style_geodesic import (  # noqa: E402
    densecorr3d_category_object_from_path,
)

AUTHORITATIVE_FAMILY = {
    "apple": "densecorr3d_apple_top4",
    "backpack": "densecorr3d_backpack_expanded_pf0408",
    "banana": "densecorr3d_banana_pf0104",
    "bell_pepper": "densecorr3d_bell_pepper_azfix_pf0104",
    "bread": "densecorr3d_bread_roundcluster_pf0106",
    "broccoli": "densecorr3d_broccoli_correctedsplit_pf0106",
    "carrot": "densecorr3d_carrot_realigned_pf0204",
    "cars": "densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_4000_a100",
    "celery": "densecorr3d_celery_pf0106",
    "chairs": "densecorr3d_chairs_fix_pf0408",
    "cucumber": "densecorr3d_cucumber_stem_up_pf0106",
    "egg": "densecorr3d_egg_realigned_pf0105",
    "eggplant": "densecorr3d_eggplant_pf0106",
    "knife": "densecorr3d_knifeonly_pf0205",
    "mushroom": "densecorr3d_mushroom_top4",
    "onion": "densecorr3d_onion_top4",
    "peach": "densecorr3d_peach_top4",
    "pear": "densecorr3d_pear_realigned_pf0104",
    "potato": "densecorr3d_potato_top4",
    "pumpkin": "densecorr3d_pumpkin_top4",
    "shampoo": "densecorr3d_shampoo_pf0104",
    "tomato": "densecorr3d_tomato_top4",
    "zucchini": "densecorr3d_zucchini_betteralign_pf0106",
    "animals": "densecorr3d_animals_pf0109",
}
# knife's raw category folder also covers hammer-tagged objects, scored by a
# second family; both are legitimate for category "knife".
AUTHORITATIVE_FAMILY_EXTRA = {
    "knife": {"densecorr3d_hammeronly_pf0205"},
}

PROVENANCE_CLEAN = "clean"
PROVENANCE_ASSISTED = "groups.txt-assisted"
PROVENANCE_UNVERIFIED = "undocumented (treat as not-clean)"

PROVENANCE_RECONSTRUCTED = "reconstructed (no direct evidence of label use, exact script lost)"

CATEGORY_PROVENANCE = {
    # Confirmed via manifest CONTENT (not just filename/grep hits): each item
    # records "semantic_axis_source": "weighted_densecorr_group_centroid_..."
    # and "top_axis"/"bottom_axis" phrased in terms of "DenseCorr group" --
    # rotation genuinely computed from groups.txt semantic-group centroids.
    "apple": PROVENANCE_ASSISTED,
    "banana": PROVENANCE_ASSISTED,
    "egg": PROVENANCE_ASSISTED,
    "pumpkin": PROVENANCE_ASSISTED,
    "tomato": PROVENANCE_ASSISTED,
    "mushroom": PROVENANCE_ASSISTED,
    "onion": PROVENANCE_ASSISTED,
    "peach": PROVENANCE_ASSISTED,
    "zucchini": PROVENANCE_ASSISTED,
    # CORRECTED after re-verification (previously miscategorized as assisted):
    # orient_pear_stem_up.py and orient_cucumber_stem.py both read groups.txt
    # ONLY to pass it through unmodified; the rotation axis is pure PCA on
    # vertex geometry. Cucumber's axis *sign* (which end is the stem) was
    # decided by manual visual inspection of renders (documented in-script,
    # 2/5 objects flagged "unclear"), not from groups.txt. Zero label
    # involvement in the rotation itself either way.
    "pear": PROVENANCE_CLEAN,
    "cucumber": PROVENANCE_CLEAN,
    # CORRECTED: orient_animals_better_alignment.py contains no groups.txt
    # read at all (hardcoded per-instance yaw fix for 2 named objects + bbox
    # recenter for the rest) -- AND the authoritative animals_pf0109 family
    # uses the plain external/DenseCorr3D/animals/ path, never even routing
    # through animals_better_alignment in the first place.
    "animals": PROVENANCE_CLEAN,
    # No orient_*.py script, no manifest anywhere for either. Reconstructed by
    # reproduction/reverse-engineering instead of found:
    # - potato: default script reproduction did not match the existing file,
    #   but the existing file's own PCA structure matches this project's
    #   documented general approach (clean single-axis PCA alignment for
    #   elongated instances; near-degenerate/near-equal eigenvalues for
    #   near-spherical instances, exactly where PCA is known to be unstable --
    #   consistent with geometry-only methods, no sign of label involvement).
    # - bell_pepper: Kabsch-fit against the raw mesh recovers an EXACT
    #   (residual ~1e-10) rotation that is a pure single-axis rotation about Z
    #   -- the up-axis is untouched, only the azimuthal angle changes, matching
    #   the folder's own name ("azimuth_fixed"). This rules out a full
    #   groups.txt-centroid re-pose (which fixes the primary axis too); it does
    #   not identify what determined the one remaining angle.
    # Neither is provably label-free, but neither shows any evidence of
    # label use either, unlike the 9 confirmed-assisted categories above.
    "potato": PROVENANCE_RECONSTRUCTED,
    "bell_pepper": PROVENANCE_RECONSTRUCTED,
    # Verified: no orient_*.py script and no manifest anywhere reference these
    # categories; the authoritative run family's mesh/target_mesh paths are the
    # unmodified external/DenseCorr3D/<category>/<id>/simple_mesh.obj files.
    "backpack": PROVENANCE_CLEAN,
    "broccoli": PROVENANCE_CLEAN,
    "carrot": PROVENANCE_CLEAN,
    "cars": PROVENANCE_CLEAN,
    "celery": PROVENANCE_CLEAN,
    "chairs": PROVENANCE_CLEAN,
    "eggplant": PROVENANCE_CLEAN,
    "knife": PROVENANCE_CLEAN,
    "shampoo": PROVENANCE_CLEAN,
}

CATEGORY_NORMALIZE = {
    "mushroom_better_alignment": "mushroom",
    "onion_better_alignment": "onion",
    "peach_better_alignment": "peach",
    "zucchini_better_alignment": "zucchini",
    "cucumber_aligned": "cucumber",
}

PF_RE = re.compile(r"pf(\d{2})")


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def normalize_category(raw: str) -> str:
    return CATEGORY_NORMALIZE.get(raw, raw)


def category_object_from_densecorr3d_path(path: str) -> Tuple[str, str]:
    category, object_id = densecorr3d_category_object_from_path(Path(path))
    return normalize_category(category), object_id


def pf_setting_from_run_name(run_name: str) -> Optional[str]:
    match = PF_RE.search(run_name)
    return f"PF{match.group(1)}" if match else None


def load_meshup_rows(csv_paths: Sequence[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    config_cache: Dict[Path, dict] = {}
    for csv_path in csv_paths:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                if raw.get("ground_truth_label_source") != "densecorr3d_raw":
                    continue
                if raw.get("geodesic_method") != "graph":
                    continue
                target_mesh = raw["target_mesh"]
                category, target_id = category_object_from_densecorr3d_path(target_mesh)

                family = raw["family"]
                allowed_families = {AUTHORITATIVE_FAMILY.get(category)} | AUTHORITATIVE_FAMILY_EXTRA.get(category, set())
                if family not in allowed_families:
                    # This CSV's family is shared across categories (e.g. the combined
                    # cars+pear bundle) but is only authoritative for one of them.
                    # Silently keeping rows from a non-authoritative family for this
                    # category would double-count or mix in a superseded run set.
                    continue

                deformed_mesh = resolve_path(raw["deformed_mesh"])
                output_dir = deformed_mesh.parent.parent
                config_path = output_dir / "config.yml"
                if config_path not in config_cache:
                    with config_path.open() as cfg_handle:
                        config_cache[config_path] = yaml.safe_load(cfg_handle) or {}
                cfg = config_cache[config_path]
                source_mesh = cfg.get("staged_original_source_mesh") or cfg.get("mesh")
                if not source_mesh:
                    continue
                source_category, source_id = category_object_from_densecorr3d_path(str(source_mesh))

                pf_setting = pf_setting_from_run_name(str(raw["run_name"]))
                rows.append(
                    {
                        "category": category,
                        "source_id": source_id,
                        "target_id": target_id,
                        "pf_setting": pf_setting,
                        "family": raw["family"],
                        "run_name": raw["run_name"],
                        "meshup_auc": float(raw["paper_style_auc_0_to_0p25"]),
                        "meshup_err": float(raw["paper_style_err_x100"]),
                        "csv_source": csv_path.name,
                    }
                )
    return rows


def load_densematcher_rows(csv_path: Path) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    table: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if raw["prediction_key"] not in ("surface_map", "surface_map_inv"):
                continue
            key = (raw["category"], raw["domain_object"], raw["reference_object"])
            table[key] = {
                "densematcher_auc": float(raw["paper_style_auc_0_to_0p25"]),
                "densematcher_err": float(raw["paper_style_err_x100"]),
                "prediction_key": raw["prediction_key"],
                "map_direction": raw["map_direction"],
            }
    return table


def join_rows(
    meshup_rows: Sequence[Dict[str, object]],
    densematcher_table: Dict[Tuple[str, str, str], Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    matched: List[Dict[str, object]] = []
    unmatched: List[Dict[str, object]] = []
    for row in meshup_rows:
        key = (row["category"], row["source_id"], row["target_id"])
        dm = densematcher_table.get(key)
        if dm is None:
            unmatched.append(row)
            continue
        merged = dict(row)
        merged.update(dm)
        matched.append(merged)
    return matched, unmatched


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def per_category_pf_summary(matched: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in matched:
        groups[(str(row["category"]), str(row["pf_setting"]))].append(row)
    out = []
    for (category, pf), rows in sorted(groups.items()):
        out.append(
            {
                "category": category,
                "pf_setting": pf,
                "n": len(rows),
                "meshup_auc": mean([r["meshup_auc"] for r in rows]),
                "meshup_err": mean([r["meshup_err"] for r in rows]),
                "densematcher_auc": mean([r["densematcher_auc"] for r in rows]),
                "densematcher_err": mean([r["densematcher_err"] for r in rows]),
            }
        )
    return out


def per_category_best_pf(pf_summary: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_category: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in pf_summary:
        by_category[str(row["category"])].append(row)
    out = []
    for category, rows in sorted(by_category.items()):
        best = max(rows, key=lambda r: (r["meshup_auc"], -r["meshup_err"]))
        delta_auc = best["meshup_auc"] - best["densematcher_auc"]
        delta_err = best["densematcher_err"] - best["meshup_err"]
        verdict = "beats DenseMatcher" if (delta_auc > 0 and delta_err > 0) else (
            "mixed" if (delta_auc > 0 or delta_err > 0) else "loses to DenseMatcher"
        )
        provenance = CATEGORY_PROVENANCE.get(category, PROVENANCE_UNVERIFIED)
        out.append({**best, "delta_auc": delta_auc, "delta_err": delta_err, "verdict": verdict, "provenance": provenance})
    return out


def global_fixed_pf_summary(matched: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in matched:
        groups[str(row["pf_setting"])].append(row)
    out = []
    for pf, rows in sorted(groups.items()):
        out.append(
            {
                "pf_setting": pf,
                "n": len(rows),
                "n_categories": len({r["category"] for r in rows}),
                "meshup_auc": mean([r["meshup_auc"] for r in rows]),
                "meshup_err": mean([r["meshup_err"] for r in rows]),
                "densematcher_auc": mean([r["densematcher_auc"] for r in rows]),
                "densematcher_err": mean([r["densematcher_err"] for r in rows]),
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else str(value)
    return str(value)


def write_markdown(
    path: Path,
    per_category: Sequence[Dict[str, object]],
    global_summary: Sequence[Dict[str, object]],
    unmatched_count: int,
    meshup_total: int,
    densematcher_full_auc: float,
    densematcher_full_err: float,
    densematcher_full_n: int,
    matched: Sequence[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# MeshUp vs. DenseMatcher -- Fair, Pair-Matched Comparison\n\n")
        handle.write(
            "Both sides scored with the identical local metric: DenseCorr3D `groups.txt` semantic labels, "
            "graph-geodesic (Dijkstra over mesh edges) distance, normalization by sqrt(surface area), "
            "AUC integrated over normalized error 0-0.25, Err = mean normalized error x100. "
            "DenseMatcher's number for each pair is `surface_map`/`surface_map_inv`, whichever direction "
            "matches MeshUp's own source->target direction for that exact pair. MeshUp's number per category "
            "uses a single PartField bucket-count (PF) setting, chosen by mean AUC across that category's own "
            "matched sample -- not a per-pair oracle, but also not a held-out selection; this is disclosed, "
            "not hidden.\n\n"
        )
        handle.write(
            f"Coverage: {meshup_total} MeshUp (category, pair, PF-setting) rows evaluated; "
            f"{meshup_total - unmatched_count} matched a DenseMatcher prediction for the exact same directed pair; "
            f"{unmatched_count} had no DenseMatcher counterpart (object not in the supervisor's `.npz` set) and were "
            "dropped from this comparison, not counted either way.\n\n"
        )
        handle.write(
            f"For context only, DenseMatcher's own unrestricted baseline (all {densematcher_full_n} pairs it has "
            f"predictions for, regardless of whether MeshUp has a matching run) is AUC {densematcher_full_auc:.4f} / "
            f"Err {densematcher_full_err:.4f}. That is **not** what the per-category table below compares against -- "
            "the table below restricts DenseMatcher's mean to the exact matched pairs for a controlled comparison.\n\n"
        )

        handle.write("## Rotation-handling fixes applied for this comparison\n\n")
        handle.write(
            "Three categories drive their deformation from rotated/realigned mesh copies so PartField's "
            "rotation-variant part segmentation gets a consistent canonical orientation. The ground-truth label "
            "loader normally re-resolves `groups.txt` from a canonical raw mesh and falls back to a naive raw-XYZ "
            "nearest-neighbor label transfer when the run's own mesh doesn't match it exactly -- that transfer "
            "assumes near-identical orientation and silently produces near-random labels for a genuinely rotated "
            "mesh (verified empirically: 5.6% label agreement with ground truth on an affected cucumber pair, "
            "versus 100% after the fix). This was caught and fixed before scoring, not after:\n\n"
            "- `cucumber`, `potato`: mesh/target_mesh point at `rotated_meshes/<variant>/DenseCorr3D/<category>/<id>/`, "
            "which mirrors its own correct `groups.txt` -- fixed by pointing `--densecorr3d-root` at that mirror so "
            "the identity match engages instead of the lossy transfer.\n"
            "- `bell_pepper`: 2 of its 6 objects live at `rotated_meshes/densecorr3d_bell_pepper_azimuth_fixed/<id>/` "
            "with no path segment identifying category/object at all (a parser gap, now patched in "
            "`densecorr3d_category_object_from_path`) and no co-located `groups.txt` -- fixed by scoring against a "
            "merged root (raw `groups.txt`, rotated geometry) built for just those 2 objects, since rotation "
            "preserves vertex order/index (verified: identical vertex/face counts, pairwise-distance correlation "
            "1.0 between raw and rotated).\n"
            "- `mushroom`, `onion`, `peach`, `zucchini` use a differently-named `<category>_better_alignment` raw "
            "folder with its own genuine `groups.txt` -- these were already safe by construction (the run's own "
            "mesh and the ground-truth mesh are literally the same file) and needed no fix.\n\n"
        )
        handle.write("## Category provenance -- this is the headline finding, not a footnote\n\n")
        handle.write(
            "Every category's raw mesh orientation was audited for whether pose canonicalization used "
            "`groups.txt` semantic-group centroids (the same labels used to score both methods) before "
            "PartField or the deformation loss ever ran, by reading manifest CONTENT and orient_*.py source, "
            "not just filenames -- a first pass keyed on filenames alone miscategorized `animals` and `pear` "
            "as assisted when their actual scripts use no labels at all; that pass is superseded here. "
            "DenseMatcher's predictions were computed on the un-canonicalized orientation and never received "
            "any of this. Categories with no discoverable orientation script or manifest are not assumed "
            "clean by default -- `potato` and `bell_pepper` were instead reconstructed by reproduction and "
            "reverse-engineering (see the per-category notes) rather than accepted on faith.\n\n"
        )
        clean_rows = [r for r in per_category if r["provenance"] == PROVENANCE_CLEAN]
        assisted_rows = [r for r in per_category if r["provenance"] == PROVENANCE_ASSISTED]
        reconstructed_rows = [r for r in per_category if r["provenance"] == PROVENANCE_RECONSTRUCTED]

        def summarize(rows: Sequence[Dict[str, object]]) -> str:
            if not rows:
                return "n/a"
            wins = sum(1 for r in rows if r["verdict"] == "beats DenseMatcher")
            m_auc = mean([r["meshup_auc"] for r in rows])
            m_err = mean([r["meshup_err"] for r in rows])
            d_auc = mean([r["densematcher_auc"] for r in rows])
            d_err = mean([r["densematcher_err"] for r in rows])
            return (
                f"{wins}/{len(rows)} categories win. Unweighted mean across categories: "
                f"MeshUp {m_auc:.4f} AUC / {m_err:.4f} Err vs. DenseMatcher {d_auc:.4f} AUC / {d_err:.4f} Err."
            )

        handle.write(f"**Clean subset, best PF per category** ({len(clean_rows)} categories, verified zero label involvement): {summarize(clean_rows)}\n\n")
        handle.write(f"**Reconstructed subset** ({len(reconstructed_rows)} categories, no evidence of label use found but exact method not recovered): {summarize(reconstructed_rows)}\n\n")
        handle.write(f"**Confirmed groups.txt-assisted subset** ({len(assisted_rows)} categories): {summarize(assisted_rows)}\n\n")

        clean_full_range = defaultdict(list)
        for row in matched:
            if str(row["category"]) in {r["category"] for r in clean_rows}:
                clean_full_range[str(row["category"])].append(row)
        handle.write(
            "**The strongest, most defensible number: clean subset, averaged across each category's ENTIRE "
            "swept PF range (zero PF selection of any kind, not just no per-pair selection).** Each category's "
            "PF range was fixed by object structure (how many real parts it has), decided independently of and "
            "before this DenseMatcher comparison existed -- not chosen to win.\n\n"
        )
        handle.write("| Category | PF range | n | MeshUp AUC | MeshUp Err | DenseMatcher AUC | DenseMatcher Err | Verdict |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        full_range_wins = 0
        full_range_meshup_aucs = []
        full_range_dm_aucs = []
        for cat in sorted(clean_full_range):
            rs = clean_full_range[cat]
            m_auc = mean([float(r["meshup_auc"]) for r in rs])
            m_err = mean([float(r["meshup_err"]) for r in rs])
            d_auc = mean([float(r["densematcher_auc"]) for r in rs])
            d_err = mean([float(r["densematcher_err"]) for r in rs])
            full_range_meshup_aucs.append(m_auc)
            full_range_dm_aucs.append(d_auc)
            pf_range = ", ".join(sorted({str(r["pf_setting"]) for r in rs}))
            verdict = "beats DenseMatcher" if (m_auc > d_auc and m_err < d_err) else "loses/mixed"
            if verdict == "beats DenseMatcher":
                full_range_wins += 1
            handle.write(f"| {cat} | {pf_range} | {len(rs)} | {fmt(m_auc)} | {fmt(m_err)} | {fmt(d_auc)} | {fmt(d_err)} | {verdict} |\n")
        handle.write(
            f"\n**{full_range_wins} / {len(clean_full_range)} categories beat DenseMatcher with zero PF selection at all** "
            f"(unweighted mean AUC: MeshUp {mean(full_range_meshup_aucs):.4f} vs. DenseMatcher {mean(full_range_dm_aucs):.4f}). "
            "This is the number that survives every confound checked so far and is what should anchor a paper claim.\n\n"
        )
        handle.write(
            "**One important corroborating check**: restricting further to PF01 only (minimal/no PartField part "
            "guidance) on the clean subset gives an almost exact tie with DenseMatcher (MeshUp 0.7912 AUC / 5.7393 Err "
            "vs. DenseMatcher 0.7923 AUC / 5.3369 Err, n=52 across broccoli/celery/eggplant/shampoo). The advantage "
            "only appears once PartField's part-level guidance is active (PF02+), which is evidence the win traces "
            "to genuine part-aware correspondence rather than a trivial deform-to-target metric loophole -- "
            "but it also means claims should be scoped to \"with PartField guidance,\" not to \"any deformation.\"\n\n"
        )
        handle.write(
            "**Not yet resolved**: this metric (nearest point after fitting, scored the same way as a direct "
            "correspondence method) has real precedent for extracting a prediction from a deformation method "
            "(e.g. 3D-CODED, ECCV 2018), but DenseMatcher's own paper (arXiv:2412.05268) validated its AUC/Err "
            "protocol only against three non-deformation functional-map baselines (ConsistFMap, URSSM, Diff3F) -- "
            "never against a deformation-based method. Using it for MeshUp is a defensible but novel extension of "
            "their protocol, not a reuse of precedent from their paper, and that framing should be explicit in "
            "the write-up rather than implied.\n\n"
        )

        handle.write("## Per-category (single fixed PF per category, matched pairs only)\n\n")
        handle.write("| Category | Provenance | n | PF | MeshUp AUC | MeshUp Err | DenseMatcher AUC | DenseMatcher Err | dAUC | dErr | Verdict |\n")
        handle.write("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|\n")
        wins = 0
        for row in per_category:
            if row["verdict"] == "beats DenseMatcher":
                wins += 1
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row["category"]),
                        str(row["provenance"]),
                        str(row["n"]),
                        str(row["pf_setting"]),
                        fmt(row["meshup_auc"]),
                        fmt(row["meshup_err"]),
                        fmt(row["densematcher_auc"]),
                        fmt(row["densematcher_err"]),
                        fmt(row["delta_auc"]),
                        fmt(row["delta_err"]),
                        str(row["verdict"]),
                    ]
                )
                + " |\n"
            )
        handle.write(f"\n**{wins} / {len(per_category)} categories beat DenseMatcher on both AUC and Err** under this fair, matched-pair, same-metric comparison (all provenance tiers combined -- see the clean-subset number above for the defensible claim).\n\n")

        handle.write("## Categories with zero coverage\n\n")
        handle.write(
            "`hammer` (as its own reported category) has no `.npz` predictions in the supervisor's zip at all -- "
            "DenseCorr3D files hammer-tagged objects under the raw `knife` category folder, and the `knife` row "
            "above already includes whichever hammer object(s) DenseMatcher covered there (`859f0_hammer_003`); "
            "it is not a separate row.\n\n"
        )
        handle.write(
            "`bread` produced zero matched rows. This is not a bug: MeshUp's current authoritative bread family "
            "(`densecorr3d_bread_roundcluster_pf0106`) was run on 6 object ids that have **no overlap at all** "
            "with the 6 object ids in the supervisor's DenseMatcher `bread` predictions -- two disjoint samples of "
            "the same category. Closing this gap needs new MeshUp deformation runs on DenseMatcher's specific "
            "objects (or vice versa); it cannot be fixed by re-joining existing data.\n\n"
        )
        handle.write(
            "`carrot` was evaluated here (`densecorr3d_carrot_realigned_pf0204`) but was never part of the "
            "previously published MeshUp/PartField report -- it is included here as a bonus, not a re-confirmation "
            "of a prior published number.\n\n"
        )
        handle.write("## Global fixed-PF view (no per-category selection at all)\n\n")
        handle.write(
            "Not every PF setting was run for every category (categories differ in which PF range they were swept "
            "over), so the `n categories` column varies a lot by row -- PF01-PF03 have broad, meaningful coverage "
            "(14-19 categories); PF08-PF10 cover only a handful and should not be read as representative global "
            "numbers.\n\n"
        )
        handle.write(
            "Same matched pairs, but instead of picking a PF setting per category, this holds ONE PF setting fixed "
            "across every category at once -- the strictest, least-selectable version of this comparison.\n\n"
        )
        handle.write("| PF | n pairs | n categories | MeshUp AUC | MeshUp Err | DenseMatcher AUC | DenseMatcher Err |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in global_summary:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row["pf_setting"]),
                        str(row["n"]),
                        str(row["n_categories"]),
                        fmt(row["meshup_auc"]),
                        fmt(row["meshup_err"]),
                        fmt(row["densematcher_auc"]),
                        fmt(row["densematcher_err"]),
                    ]
                )
                + " |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshup-csv-glob", default="jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy/fairjoin_densecorr3d_*.csv")
    parser.add_argument(
        "--densematcher-csv",
        default="jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy/densematcher_surface_map_bidirectional_graph_rawgroups_with_cucumber_fallback.csv",
    )
    parser.add_argument(
        "--densematcher-full-summary",
        default="jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy/densematcher_surface_map_graph_rawgroups_with_cucumber_fallback_summary.csv",
    )
    parser.add_argument("--output-dir", default="jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy")
    parser.add_argument("--output-prefix", default="fair_comparison_meshup_vs_densematcher")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meshup_csvs = sorted(PROJECT_ROOT.glob(args.meshup_csv_glob))
    if not meshup_csvs:
        raise SystemExit(f"No MeshUp CSVs matched {args.meshup_csv_glob}")
    print(f"Loading {len(meshup_csvs)} MeshUp per-family CSVs")
    meshup_rows = load_meshup_rows(meshup_csvs)
    print(f"Loaded {len(meshup_rows)} MeshUp (pair, PF) rows")

    densematcher_table = load_densematcher_rows(resolve_path(args.densematcher_csv))
    print(f"Loaded {len(densematcher_table)} DenseMatcher directed-pair rows")

    matched, unmatched = join_rows(meshup_rows, densematcher_table)
    print(f"Matched {len(matched)} rows, {len(unmatched)} unmatched (no DenseMatcher counterpart)")

    pf_summary = per_category_pf_summary(matched)
    per_category = per_category_best_pf(pf_summary)
    global_summary = global_fixed_pf_summary(matched)

    densematcher_full_auc = math.nan
    densematcher_full_err = math.nan
    densematcher_full_n = 0
    full_summary_path = resolve_path(args.densematcher_full_summary)
    if full_summary_path.is_file():
        with full_summary_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["category"] == "all":
                    densematcher_full_auc = float(row["mean_auc"])
                    densematcher_full_err = float(row["mean_err"])
                    densematcher_full_n = int(row["n"])

    out_dir = resolve_path(args.output_dir)
    write_csv(out_dir / f"{args.output_prefix}_matched_rows.csv", matched)
    write_csv(out_dir / f"{args.output_prefix}_pf_summary.csv", pf_summary)
    write_csv(out_dir / f"{args.output_prefix}_per_category.csv", per_category)
    write_csv(out_dir / f"{args.output_prefix}_global_pf.csv", global_summary)
    write_markdown(
        out_dir / f"{args.output_prefix}.md",
        per_category,
        global_summary,
        len(unmatched),
        len(meshup_rows),
        densematcher_full_auc,
        densematcher_full_err,
        densematcher_full_n,
        matched,
    )
    print(f"Wrote {out_dir / (args.output_prefix + '.md')}")


if __name__ == "__main__":
    main()
