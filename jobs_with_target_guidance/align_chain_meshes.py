"""Write aligned chain-animal mesh copies for PartField reruns."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from jobs_with_target_guidance.partfield_segment import load_mesh


def identity(vertices: np.ndarray) -> np.ndarray:
    return vertices.astype(np.float32, copy=True)


def sam3d_z_forward_to_x_forward(vertices: np.ndarray) -> np.ndarray:
    """Rotate SAM3D quadrupeds from length-on-Z into DenseCorr-style X-forward."""
    out = np.empty_like(vertices, dtype=np.float32)
    out[:, 0] = vertices[:, 2]
    out[:, 1] = vertices[:, 1]
    out[:, 2] = -vertices[:, 0]
    return out


TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": identity,
    "sam3d_z_to_x": sam3d_z_forward_to_x_forward,
}


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# aligned chain mesh\n")
        for x, y, z in vertices:
            handle.write(f"v {float(x):.8f} {float(y):.8f} {float(z):.8f}\n")
        for face in faces:
            a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
            handle.write(f"f {a} {b} {c}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transform", required=True, choices=sorted(TRANSFORMS))
    args = parser.parse_args()

    vertices, faces = load_mesh(args.input)
    transformed = TRANSFORMS[args.transform](vertices)
    write_obj(args.output, transformed, faces)

    vmin = transformed.min(axis=0)
    vmax = transformed.max(axis=0)
    extent = vmax - vmin
    print(f"Wrote {args.output}")
    print(f"vertices={len(transformed)} faces={len(faces)}")
    print(f"bbox_min={np.round(vmin, 6).tolist()}")
    print(f"bbox_max={np.round(vmax, 6).tolist()}")
    print(f"bbox_extent={np.round(extent, 6).tolist()}")


if __name__ == "__main__":
    main()
