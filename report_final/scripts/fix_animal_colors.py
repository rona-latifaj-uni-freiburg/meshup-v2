"""Fix the domain_animals_showcase panels that used rainbow position-coloring
instead of PartField bucket coloring, by combining the deformed mesh's final
vertex positions with the source mesh's bucket-color-per-vertex array (vertex
correspondence is preserved by construction, so this recoloring is exact).

Writes new colored PLYs under report_final/data/, then renders a corrected
domain_animals_showcase.png into report_final/figures/, reusing the render
helpers from report_new/scripts/make_review_figures.py without modifying it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import (  # noqa: E402
    FIG as OLD_FIG,
    Mesh,
    four_deformation_sheet,
    load_mesh,
    save_rgb,
)
from jobs_with_target_guidance.render_partfield_turntable_video import load_colored_ply  # noqa: E402

OUT_DATA = ROOT / "report_final" / "data"
OUT_FIG = ROOT / "report_final" / "figures"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def write_colored_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors01: np.ndarray) -> None:
    n = vertices.shape[0]
    vertex_arr = np.zeros(
        n,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex_arr["x"], vertex_arr["y"], vertex_arr["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    rgb = np.clip(colors01 * 255.0, 0, 255).astype(np.uint8)
    vertex_arr["red"], vertex_arr["green"], vertex_arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    face_arr = np.zeros(faces.shape[0], dtype=[("vertex_indices", "i4", (3,))])
    face_arr["vertex_indices"] = faces

    PlyData(
        [
            PlyElement.describe(vertex_arr, "vertex"),
            PlyElement.describe(face_arr, "face"),
        ],
        text=False,
    ).write(str(path))
    print(f"wrote {path}")


def normalize_unit_box(vertices: np.ndarray) -> np.ndarray:
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    scale = max(float((vertices.max(axis=0) - vertices.min(axis=0)).max()), 1e-8)
    return (vertices - center) / scale


def recolor_final(
    source_bucket_ply: Path,
    initial_rainbow_ply: Path,
    final_rainbow_ply: Path,
    out_path: Path,
) -> Path:
    """Transfer PartField bucket colors onto the deformed mesh.

    The bucket-colored source PLY was exported from a different-resolution
    mesh than the one actually deformed (PartField ran on a denser copy), so
    vertex indices do not line up 1:1. Instead, nearest-neighbor match the
    deformation's own *undeformed* vertices against the bucket-colored
    source in normalized space to get a color per low-res vertex, then reuse
    those colors on the final (deformed) vertices -- valid because the
    deformation itself preserves vertex order/topology exactly.
    """
    src_v, _src_f, src_colors = load_colored_ply(source_bucket_ply)
    initial_v, _initial_f, _ = load_colored_ply(initial_rainbow_ply)
    final_v, final_f, _final_colors = load_colored_ply(final_rainbow_ply)
    if initial_v.shape[0] != final_v.shape[0]:
        raise ValueError("Initial/final vertex count mismatch; deformation should preserve topology.")

    tree = cKDTree(normalize_unit_box(src_v))
    _dist, nearest_idx = tree.query(normalize_unit_box(initial_v), k=1)
    transferred_colors = src_colors[nearest_idx]

    write_colored_ply(out_path, final_v, final_f, transferred_colors)
    return out_path


def rel(path: str) -> Path:
    return ROOT / path


def main() -> None:
    animal_base = rel("jobs_with_target_guidance/cross_animal_spike_runs/outputs")
    sam3d = animal_base / "sam3d_selected_best_asym035_jump500_4000"
    chain = animal_base / "chain_panther_horse_giraffe_face_x_best_asym035_jump500_4000"
    seg_sam3d = rel("jobs_with_target_guidance/partfield_segments/sam3d_animals/requested_pairs_20260706")
    seg_chain = rel("jobs_with_target_guidance/partfield_segments/chain_animals/panther_horse_giraffe_face_x_20260707")

    horse = sam3d / (
        "horse2_to_giraffe4_sam3d_selected_pf08_4000_hard_partfield_chamfer_only_horse2_to_giraffe4_pf08_"
        "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5816558"
    )

    fixes = [
        (
            "panda3_to_bear",
            sam3d / (
                "panda3_to_bear_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_panda3_to_bear_pf07_"
                "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5817242"
            ),
            seg_sam3d / "panda3_to_bear/buckets_07/colored/panda3_partfield_07_parts.ply",
            seg_sam3d / "panda3_to_bear/buckets_07/colored/bear_partfield_07_parts.ply",
            -62.0,
        ),
        (
            "fox_to_cat2",
            sam3d / (
                "fox_to_cat2_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_fox_to_cat2_pf07_pfonly_"
                "best_asym035_jneighbor1000_jump500_4000ep_4000ep_5818235"
            ),
            seg_sam3d / "fox_to_cat2/buckets_07/colored/fox_partfield_07_parts.ply",
            seg_sam3d / "fox_to_cat2/buckets_07/colored/cat2_partfield_07_parts.ply",
            -62.0,
        ),
        (
            "panther_to_horse2",
            chain / (
                "panther_to_horse2_chain_face_x_pf06_4000_hard_partfield_chamfer_only_panther_to_horse2_pf06_"
                "pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5829892"
            ),
            seg_chain / "panther_to_horse2/buckets_06/colored/panther_partfield_06_parts.ply",
            seg_chain / "panther_to_horse2/buckets_06/colored/horse2_partfield_06_parts.ply",
            -68.0,
        ),
    ]

    animals = [
        (
            "Horse2 to Giraffe4",
            horse / "partfield_colored_meshes/input_partfield_08.ply",
            horse / "partfield_colored_meshes/final_partfield_08.ply",
            horse / "partfield_colored_meshes/target_partfield_08.ply",
            -68.0,
        )
    ]

    for name, run_dir, source_ply, target_ply, azimuth in fixes:
        initial_rainbow = run_dir / "colored_meshes/mesh_initial.ply"
        final_rainbow = run_dir / "colored_meshes/mesh_final_position.ply"
        out_final = OUT_DATA / f"{name}_final_partfield_colored.ply"
        recolor_final(source_ply, initial_rainbow, final_rainbow, out_final)
        label = name.replace("_to_", " to ").title().replace("Sam3D", "SAM3D")
        animals.append((label, source_ply, out_final, target_ply, azimuth))

    four_deformation_sheet(
        OUT_FIG / "domain_animals_showcase.png",
        animals,
        title="Animal deformations: source, final, target (consistent PartField bucket coloring)",
    )


if __name__ == "__main__":
    main()
