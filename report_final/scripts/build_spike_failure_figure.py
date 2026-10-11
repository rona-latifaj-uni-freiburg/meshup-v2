"""Render the bulldog->dachshund PartField chamfer-only spike failure case.

Before the asymmetric source-to-target weighting and the mutual/unmatched
down-weighting described in Section 5.6 were introduced, a chamfer-only local
term could pull an isolated, "unclaimed" source vertex (one that is not the
nearest neighbor of any target point) arbitrarily far toward its own single
nearest target point, producing a thin spike rather than a smooth surface.
This is the earliest run in the project that shows the failure clearly
(job ids 4807068/4807069, before jump/asym/mutual regularization existed).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import triptych  # noqa: E402

RUN_ROOT = ROOT / "jobs_with_target_guidance" / "artur_soft_runs" / "outputs_dev"
HARD_DIR = RUN_ROOT / "bulldog_to_dachshund_artur_soft_dev_h100_single_hard_partfield_chamfer_only_2500ep_4807068" / "colored_meshes"

OUT = ROOT / "report_final" / "figures" / "bulldog_dachshund_spike_failure.png"


def main() -> None:
    triptych(
        OUT,
        [
            ("Source (bulldog)", HARD_DIR / "mesh_initial.ply"),
            ("Final (PartField chamfer-only, no asym/mutual weighting)", HARD_DIR / "mesh_final_position.ply"),
        ],
        title="Bulldog -> dachshund: an unweighted local Chamfer term collapsing a leg and the head into spikes",
        azimuth=-68.0,
        elevation=8.0,
        width=1500,
        height=680,
    )


if __name__ == "__main__":
    main()
