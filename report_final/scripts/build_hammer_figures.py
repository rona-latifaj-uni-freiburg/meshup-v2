"""Rebuild the two hammer figures with a clearer mesh pair and a proper side
profile (az=90, vertical handle) instead of the old near-end-on view, and
replace the flat 2D screenshot triptych with a real 3D render.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fix_animal_colors import recolor_final  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import triptych, four_deformation_sheet  # noqa: E402

OUT_FIG = ROOT / "report_final" / "figures"
OUT_DATA = ROOT / "report_final" / "data"

SEG = ROOT / "jobs_with_target_guidance/partfield_segments/densecorr3d_categories/densecorr3d_hammer_only_20260808/hammeronly/buckets_04/colored"
SOURCE_BUCKET_PLY = SEG / "hammeronly_m000_04083_partfield_04_parts.ply"
TARGET_BUCKET_PLY = SEG / "hammeronly_m003_859f0_partfield_04_parts.ply"

RUN = ROOT / (
    "jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_hammeronly_pf0205/"
    "hmrS0T3_hammeronly_pf03_4000_hard_partfield_chamfer_only_hmrS0T3_pf03_pfonly_best_asym035_"
    "jneighbor1000_jump500_4000ep_4000ep_6224157"
)


def main() -> None:
    # Figure: clean bucket example, clear vertical side profile (az=90).
    triptych(
        OUT_FIG / "hammer_pf04_buckets.png",
        (
            ("Source PF4", SOURCE_BUCKET_PLY),
            ("Target PF4", TARGET_BUCKET_PLY),
        ),
        title="Hammer buckets: stable handle and head",
        azimuth=90.0,
        elevation=10.0,
        width=1200,
        height=760,
    )

    # Figure: real 3D triptych for a completed deformation, bucket colors
    # transferred onto the deformed mesh (topology-preserving, exact).
    final_colored = OUT_DATA / "hammer_s0t3_final_partfield_colored.ply"
    recolor_final(
        SOURCE_BUCKET_PLY,
        RUN / "colored_meshes/mesh_initial.ply",
        RUN / "colored_meshes/mesh_final_position.ply",
        final_colored,
    )
    triptych(
        OUT_FIG / "qual_hammer_result.png",
        (
            ("Source", SOURCE_BUCKET_PLY),
            ("Final", final_colored),
            ("Target", TARGET_BUCKET_PLY),
        ),
        title="Hammer: source, deformed source, and target",
        azimuth=90.0,
        elevation=10.0,
        width=1500,
        height=760,
    )


if __name__ == "__main__":
    main()
