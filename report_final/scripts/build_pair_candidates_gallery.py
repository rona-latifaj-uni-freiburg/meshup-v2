"""Browsing galleries so the report author can pick pairs manually, rather than have
pairs pre-selected. Renders every available mesh in a category as a single-view
thumbnail, labeled with its mesh id, so pairs can be chosen by eye. Not used in the
report itself -- these are throwaway selection aids written to report_final/data/.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report_new" / "scripts"))

from make_review_figures import Mesh, load_mesh, render_panel, save_rgb, draw_text, INK  # noqa: E402
from jobs_with_target_guidance.render_partfield_turntable_video import load_colored_ply, face_colors  # noqa: E402

OUT = ROOT / "report_final" / "data"
OUT.mkdir(parents=True, exist_ok=True)
WHITE = (255, 255, 255)

SEG = ROOT / "jobs_with_target_guidance/partfield_segments/densecorr3d_categories"

# hash -> human-readable name, from the project's own fair-comparison CSV (matched_rows).
NAMES = {
    "35ebd": "toy_car_022", "6bb21": "toy_bus_041", "88c92": "toy_truck_011",
    "94b39": "toy_car_025", "f756d": "toy_bus_019",
    "1437b": "chair_005", "1d367": "chair_010", "649a2": "chair_001",
    "7af72": "chair_015", "fe0fe": "chair_004",
    "3ca56": "backpack_001", "68903": "backpack_024", "b74e3": "backpack_015",
    "ee073": "backpack_002", "0f4d5": "backpack_(expanded)", "ede74": "backpack_(expanded)",
}


def label_for(stem: str) -> str:
    parts = stem.split("_")
    mid = parts[1]
    hashid = parts[2]
    friendly = NAMES.get(hashid, "")
    return f"{mid} ({hashid}{' - ' + friendly if friendly else ''})"


def build_grid_gallery(out_name: str, title: str, entries: list[tuple[str, Mesh]],
                        cols: int, cell: int = 340, az: float = 0.0, elevation: float = 8.0,
                        zoom: float = 0.82) -> None:
    supersample = 2
    rows = (len(entries) + cols - 1) // cols
    title_h = 50
    width, height = cell * cols, title_h + cell * rows
    w, h = width * supersample, height * supersample
    image = np.full((h, w, 3), WHITE, dtype=np.uint8)
    draw_text(image, title, (18 * supersample, 34 * supersample), scale=0.62 * supersample,
              color=INK, thickness=max(1, int(1.6 * supersample)))
    for i, (label, mesh) in enumerate(entries):
        r, c = divmod(i, cols)
        x0 = c * cell * supersample
        y0 = (title_h + r * cell) * supersample
        panel_h = int(cell * 0.86) * supersample
        rect = (x0 + 6 * supersample, y0, cell * supersample - 12 * supersample, panel_h)
        render_panel(image, rect, mesh, "", azimuth=az, elevation=elevation, zoom=zoom)
        draw_text(image, label, (x0 + 10 * supersample, y0 + panel_h + 22 * supersample),
                  scale=0.4 * supersample, color=INK, thickness=max(1, supersample))
    save_rgb(OUT / out_name, image, supersample, (width, height))
    print(f"wrote {out_name}")


def build_object_galleries() -> None:
    specs = [
        ("cars_gallery.png", "Cars -- all 5 available meshes (PF06 bucket colors)",
         SEG / "densecorr3d_nonanimals_test_20260806/cars/buckets_06/colored", 3),
        ("chairs_gallery.png", "Chairs -- all 5 available meshes (PF06 bucket colors)",
         SEG / "densecorr3d_nonanimals_test_20260806/chairs/buckets_06/colored", 3),
        ("backpack_gallery.png", "Backpacks -- all 6 available meshes (PF06 bucket colors)",
         SEG / "densecorr3d_backpack_expanded_20260812/backpack/buckets_06/colored", 3),
    ]
    for out_name, title, folder, cols in specs:
        files = sorted(glob.glob(str(folder / "*.ply")))
        entries = []
        for f in files:
            stem = Path(f).stem
            mesh = load_mesh(Path(f))
            entries.append((label_for(stem), mesh))
        build_grid_gallery(out_name, title, entries, cols=cols)


TOSCA_OBJS = ROOT / "external/TOSCA/meshup_prepared_zup_xfront/objs"


def pos_color(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(0, keepdims=True), v.max(0, keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1
    return ((v - lo) / rng).astype(np.float32)


def parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with path.open() as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:4]]
                faces.append(idx)
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def build_tosca_gallery() -> None:
    categories = {
        "cat": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "dog": [0, 1, 2, 3, 5, 6, 7, 8, 10],
        "gorilla": [1, 5, 8, 14],
        "horse": [0, 5, 6, 7, 10, 14, 15, 17],
        "wolf": [0, 1, 2],
    }
    entries = []
    for cat, ids in categories.items():
        for i in ids:
            obj_path = TOSCA_OBJS / f"{cat}{i}.obj"
            if not obj_path.exists():
                continue
            v, f = parse_obj(obj_path)
            c = pos_color(v)
            mesh = Mesh(v, f, c, face_colors(f, c))
            entries.append((f"{cat}{i}", mesh))
    build_grid_gallery(
        "tosca_gallery.png",
        "TOSCA -- all available meshes across cat / dog / gorilla / horse / wolf "
        "(no michael/victoria/centaur meshes exist in this project's prepared TOSCA assets)",
        entries, cols=7, cell=260, zoom=0.8,
    )


def main() -> None:
    build_object_galleries()
    build_tosca_gallery()


if __name__ == "__main__":
    main()
