"""Render a dedicated, high-resolution hero image directly from mesh geometry
(not a video frame grab) for the showcase page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, load_mesh, render_panel, save_rgb, PAGE_BG  # noqa: E402

OUT = ROOT / "report_final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def hero(output: Path, mesh_path: Path, *, azimuth: float, elevation: float, width: int, height: int, zoom: float = 1.0, supersample: int = 3) -> None:
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), PAGE_BG, dtype=np.uint8)
    render_panel(image, (0, 0, w, h), load_mesh(mesh_path), "", azimuth=azimuth, elevation=elevation, zoom=zoom)
    save_rgb(output, image, supersample, (width, height))


def main() -> None:
    horse = ROOT / (
        "jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_selected_best_asym035_jump500_4000/"
        "horse2_to_giraffe4_sam3d_selected_pf08_4000_hard_partfield_chamfer_only_horse2_to_giraffe4_pf08_"
        "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5816558"
    )
    hero(
        OUT / "hero_giraffe.png",
        horse / "partfield_colored_meshes/final_partfield_08.ply",
        azimuth=-68.0,
        elevation=6.0,
        width=2400,
        height=1260,
        zoom=1.05,
    )


if __name__ == "__main__":
    main()
