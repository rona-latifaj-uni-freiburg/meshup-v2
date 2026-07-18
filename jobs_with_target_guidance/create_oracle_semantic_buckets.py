"""Create shared-topology semantic bucket labels for quadruped morph meshes.

This is an oracle-proxy for experiments where source and target meshes share
the same topology. Labels are assigned on a reference mesh using coarse animal
anatomy rules and copied by vertex id to every other mesh.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import struct
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)

BUCKET_NAMES: Tuple[str, ...] = (
    "head_ears",
    "neck_chest",
    "torso_front",
    "torso_mid",
    "rump_rear",
    "tail",
    "front_left_leg",
    "front_right_leg",
    "hind_left_leg",
    "hind_right_leg",
)


Vertex = Tuple[float, float, float]
Face = Tuple[int, int, int]


def load_obj(path: Path) -> Tuple[List[Vertex], List[Face]]:
    vertices: List[Vertex] = []
    faces: List[Face] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                face = []
                for token in line.split()[1:4]:
                    face.append(int(token.split("/")[0]) - 1)
                faces.append((face[0], face[1], face[2]))
    if not vertices or not faces:
        raise ValueError(f"{path} did not contain OBJ vertices and triangular faces.")
    return vertices, faces


def normalized_vertices(vertices: Sequence[Vertex]) -> List[Vertex]:
    mins = [min(v[axis] for v in vertices) for axis in range(3)]
    maxs = [max(v[axis] for v in vertices) for axis in range(3)]
    spans = [max(maxs[axis] - mins[axis], 1e-12) for axis in range(3)]
    return [
        (
            (vertex[0] - mins[0]) / spans[0],
            (vertex[1] - mins[1]) / spans[1],
            (vertex[2] - mins[2]) / spans[2],
        )
        for vertex in vertices
    ]


def make_semantic_labels(reference_vertices: Sequence[Vertex]) -> List[int]:
    """Assign coarse quadruped anatomy labels from normalized reference coords.

    The reference meshes in this project have the animal length along x, height
    along y, and left/right along z. Positive x is the head side.
    """
    norm = normalized_vertices(reference_vertices)
    labels: List[int] = []
    for x, y, z in norm:
        left_side = z >= 0.5

        # Tail first: rear, elevated vertices. This captures the tail tip and
        # most of the raised tail without swallowing the hind feet.
        if x < 0.22 and y > 0.55:
            labels.append(5)
            continue

        # Low vertices are limbs. Splitting by x and z gives stable left/right
        # front/hind buckets on the shared topology.
        if y < 0.43:
            front = x >= 0.58
            if front and left_side:
                labels.append(6)
            elif front:
                labels.append(7)
            elif left_side:
                labels.append(8)
            else:
                labels.append(9)
            continue

        if x >= 0.77:
            labels.append(0)
        elif x >= 0.63:
            labels.append(1)
        elif x >= 0.48:
            labels.append(2)
        elif x >= 0.32:
            labels.append(3)
        else:
            labels.append(4)
    return labels


def face_labels_from_vertices(labels: Sequence[int], faces: Sequence[Face]) -> List[int]:
    face_labels: List[int] = []
    for face in faces:
        votes = Counter(labels[idx] for idx in face)
        face_labels.append(votes.most_common(1)[0][0])
    return face_labels


def npy_int64_bytes(values: Sequence[int] | int, shape: Tuple[int, ...]) -> bytes:
    header = {
        "descr": "<i8",
        "fortran_order": False,
        "shape": shape,
    }
    header_str = repr(header)
    header_len = len(header_str) + 1
    pad = (16 - ((10 + header_len) % 16)) % 16
    header_bytes = (header_str + (" " * pad) + "\n").encode("latin1")
    out = bytearray()
    out.extend(b"\x93NUMPY")
    out.extend(bytes([1, 0]))
    out.extend(struct.pack("<H", len(header_bytes)))
    out.extend(header_bytes)
    if shape == ():
        out.extend(struct.pack("<q", int(values)))  # type: ignore[arg-type]
    else:
        for value in values:  # type: ignore[union-attr]
            out.extend(struct.pack("<q", int(value)))
    return bytes(out)


def write_label_npz(
    path: Path,
    source_labels: Sequence[int],
    target_labels: Sequence[int],
    source_faces: Sequence[Face],
    target_faces: Sequence[Face],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_buckets = len(BUCKET_NAMES)
    source_face_labels = face_labels_from_vertices(source_labels, source_faces)
    target_face_labels = face_labels_from_vertices(target_labels, target_faces)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("source_labels.npy", npy_int64_bytes(source_labels, (len(source_labels),)))
        archive.writestr("target_labels.npy", npy_int64_bytes(target_labels, (len(target_labels),)))
        archive.writestr("labels.npy", npy_int64_bytes(source_labels, (len(source_labels),)))
        archive.writestr("vertex_labels.npy", npy_int64_bytes(source_labels, (len(source_labels),)))
        archive.writestr("source_face_labels.npy", npy_int64_bytes(source_face_labels, (len(source_face_labels),)))
        archive.writestr("target_face_labels.npy", npy_int64_bytes(target_face_labels, (len(target_face_labels),)))
        archive.writestr("face_labels.npy", npy_int64_bytes(source_face_labels, (len(source_face_labels),)))
        archive.writestr("n_buckets.npy", npy_int64_bytes(n_buckets, ()))


def write_single_label_npz(
    path: Path,
    vertex_labels: Sequence[int],
    faces: Sequence[Face],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    face_labels = face_labels_from_vertices(vertex_labels, faces)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("labels.npy", npy_int64_bytes(vertex_labels, (len(vertex_labels),)))
        archive.writestr("vertex_labels.npy", npy_int64_bytes(vertex_labels, (len(vertex_labels),)))
        archive.writestr("face_labels.npy", npy_int64_bytes(face_labels, (len(face_labels),)))
        archive.writestr("n_buckets.npy", npy_int64_bytes(len(BUCKET_NAMES), ()))


def read_npy_int64(data: bytes) -> Tuple[List[int], Tuple[int, ...]]:
    if not data.startswith(b"\x93NUMPY"):
        raise ValueError("Not an NPY byte stream.")
    major = data[6]
    if major != 1:
        raise ValueError(f"Only NPY v1.0 is supported, got major version {major}.")
    header_len = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10 : 10 + header_len].decode("latin1").strip())
    if header.get("descr") not in ("<i8", "|i8"):
        raise ValueError(f"Expected int64 NPY, got {header.get('descr')}.")
    shape = tuple(header["shape"])
    offset = 10 + header_len
    count = 1
    for dim in shape:
        count *= int(dim)
    values = [
        struct.unpack("<q", data[offset + i * 8 : offset + (i + 1) * 8])[0]
        for i in range(count)
    ]
    return values, shape


def read_npz_int64(path: Path, key: str) -> List[int]:
    with zipfile.ZipFile(path, "r") as archive:
        data = archive.read(f"{key}.npy")
    values, _shape = read_npy_int64(data)
    return values


def color_for_label(label: int) -> Tuple[int, int, int]:
    return PALETTE[label % len(PALETTE)]


def write_colored_ply(
    path: Path,
    vertices: Sequence[Vertex],
    faces: Sequence[Face],
    labels: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex, label in zip(vertices, labels):
            red, green, blue = color_for_label(label)
            handle.write(
                f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} {red} {green} {blue}\n"
            )
        for a, b, c in faces:
            handle.write(f"3 {a} {b} {c}\n")


def project(vertex: Vertex, view: str) -> Tuple[float, float, float]:
    x, y, z = vertex
    if view == "side":
        return x, y, z
    if view == "front":
        return z, y, x
    if view == "top":
        return x, z, y
    raise ValueError(view)


def svg_mesh_panel(
    vertices: Sequence[Vertex],
    faces: Sequence[Face],
    labels: Sequence[int],
    title: str,
    view: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> List[str]:
    projected = [project(v, view) for v in vertices]
    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-12)
    span_y = max(max_y - min_y, 1e-12)
    scale = 0.88 * min(width / span_x, height / span_y)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    screen = [
        (
            x0 + width * 0.5 + (p[0] - cx) * scale,
            y0 + height * 0.5 - (p[1] - cy) * scale,
            p[2],
        )
        for p in projected
    ]

    def face_depth(face: Face) -> float:
        return sum(screen[idx][2] for idx in face) / 3.0

    lines = [
        f'<text x="{x0 + 10}" y="{y0 + 20}" font-size="14" font-family="Arial" fill="#111">{title}</text>'
    ]
    for face in sorted(faces, key=face_depth):
        face_label = Counter(labels[idx] for idx in face).most_common(1)[0][0]
        red, green, blue = color_for_label(face_label)
        pts = " ".join(f"{screen[idx][0]:.1f},{screen[idx][1]:.1f}" for idx in face)
        lines.append(
            f'<polygon points="{pts}" fill="rgb({red},{green},{blue})" '
            'stroke="rgba(20,20,20,0.22)" stroke-width="0.25" />'
        )
    return lines


def write_svg_summary(
    path: Path,
    source_vertices: Sequence[Vertex],
    source_faces: Sequence[Face],
    target_vertices: Sequence[Vertex],
    target_faces: Sequence[Face],
    source_labels: Sequence[int],
    target_labels: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1320, 900
    panel_w, panel_h = 400, 250
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<text x="24" y="32" font-size="24" font-family="Arial" fill="#111">Shared-topology oracle semantic buckets</text>',
    ]
    views = ("side", "front", "top")
    for col, view in enumerate(views):
        lines.extend(
            svg_mesh_panel(
                source_vertices,
                source_faces,
                source_labels,
                f"source cat - {view}",
                view,
                24 + col * 430,
                60,
                panel_w,
                panel_h,
            )
        )
        lines.extend(
            svg_mesh_panel(
                target_vertices,
                target_faces,
                target_labels,
                f"target dachshund - {view}",
                view,
                24 + col * 430,
                340,
                panel_w,
                panel_h,
            )
        )
    legend_x = 35
    legend_y = 655
    lines.append('<text x="24" y="635" font-size="18" font-family="Arial" fill="#111">Legend</text>')
    for idx, name in enumerate(BUCKET_NAMES):
        red, green, blue = color_for_label(idx)
        x = legend_x + (idx % 5) * 250
        y = legend_y + (idx // 5) * 42
        lines.append(f'<rect x="{x}" y="{y}" width="22" height="22" fill="rgb({red},{green},{blue})" />')
        lines.append(f'<text x="{x + 32}" y="{y + 17}" font-size="14" font-family="Arial" fill="#222">{idx}: {name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    row_bytes = width * 3
    for row in range(height):
        raw.append(0)
        raw.extend(pixels[row * row_bytes : (row + 1) * row_bytes])
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(raw), level=6)))
    png.extend(chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(png))


def draw_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
) -> None:
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    for y in range(y0, y1):
        base = (y * width + x0) * 3
        for _x in range(x0, x1):
            pixels[base : base + 3] = bytes(color)
            base += 3


def draw_triangle(
    pixels: bytearray,
    width: int,
    height: int,
    pts: Sequence[Tuple[float, float]],
    color: Tuple[int, int, int],
) -> None:
    (x0, y0), (x1, y1), (x2, y2) = pts
    min_x = max(0, int(math.floor(min(x0, x1, x2))))
    max_x = min(width - 1, int(math.ceil(max(x0, x1, x2))))
    min_y = max(0, int(math.floor(min(y0, y1, y2))))
    max_y = min(height - 1, int(math.ceil(max(y0, y1, y2))))
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if abs(area) < 1e-8:
        return
    same_sign_positive = area > 0
    color_bytes = bytes(color)
    for y in range(min_y, max_y + 1):
        py = y + 0.5
        for x in range(min_x, max_x + 1):
            px = x + 0.5
            w0 = (x1 - px) * (y2 - py) - (y1 - py) * (x2 - px)
            w1 = (x2 - px) * (y0 - py) - (y2 - py) * (x0 - px)
            w2 = (x0 - px) * (y1 - py) - (y0 - py) * (x1 - px)
            inside = (w0 >= 0 and w1 >= 0 and w2 >= 0) if same_sign_positive else (w0 <= 0 and w1 <= 0 and w2 <= 0)
            if inside:
                idx = (y * width + x) * 3
                pixels[idx : idx + 3] = color_bytes


def raster_mesh_panel(
    pixels: bytearray,
    canvas_width: int,
    canvas_height: int,
    vertices: Sequence[Vertex],
    faces: Sequence[Face],
    labels: Sequence[int],
    view: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    draw_rect(pixels, canvas_width, canvas_height, x0, y0, x0 + width, y0 + height, (250, 250, 250))
    projected = [project(v, view) for v in vertices]
    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-12)
    span_y = max(max_y - min_y, 1e-12)
    scale = 0.88 * min(width / span_x, height / span_y)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    screen = [
        (
            x0 + width * 0.5 + (p[0] - cx) * scale,
            y0 + height * 0.5 - (p[1] - cy) * scale,
            p[2],
        )
        for p in projected
    ]

    def face_depth(face: Face) -> float:
        return sum(screen[idx][2] for idx in face) / 3.0

    for face in sorted(faces, key=face_depth):
        label = Counter(labels[idx] for idx in face).most_common(1)[0][0]
        draw_triangle(
            pixels,
            canvas_width,
            canvas_height,
            [(screen[idx][0], screen[idx][1]) for idx in face],
            color_for_label(label),
        )
    draw_rect(pixels, canvas_width, canvas_height, x0, y0, x0 + width, y0 + 2, (30, 30, 30))
    draw_rect(pixels, canvas_width, canvas_height, x0, y0 + height - 2, x0 + width, y0 + height, (30, 30, 30))
    draw_rect(pixels, canvas_width, canvas_height, x0, y0, x0 + 2, y0 + height, (30, 30, 30))
    draw_rect(pixels, canvas_width, canvas_height, x0 + width - 2, y0, x0 + width, y0 + height, (30, 30, 30))


def write_png_summary(
    path: Path,
    source_vertices: Sequence[Vertex],
    source_faces: Sequence[Face],
    target_vertices: Sequence[Vertex],
    target_faces: Sequence[Face],
    source_labels: Sequence[int],
    target_labels: Sequence[int],
) -> None:
    width, height = 1320, 760
    pixels = bytearray([255] * width * height * 3)
    panel_w, panel_h = 400, 250
    for col, view in enumerate(("side", "front", "top")):
        raster_mesh_panel(
            pixels,
            width,
            height,
            source_vertices,
            source_faces,
            source_labels,
            view,
            24 + col * 430,
            30,
            panel_w,
            panel_h,
        )
        raster_mesh_panel(
            pixels,
            width,
            height,
            target_vertices,
            target_faces,
            target_labels,
            view,
            24 + col * 430,
            310,
            panel_w,
            panel_h,
        )
    for idx in range(len(BUCKET_NAMES)):
        x = 35 + (idx % 5) * 250
        y = 610 + (idx // 5) * 42
        draw_rect(pixels, width, height, x, y, x + 40, y + 28, color_for_label(idx))
    write_png(path, width, height, pixels)


def counts(labels: Sequence[int]) -> Dict[str, int]:
    counter = Counter(labels)
    return {f"{idx:02d}_{name}": int(counter.get(idx, 0)) for idx, name in enumerate(BUCKET_NAMES)}


def confusion(
    source_oracle: Sequence[int],
    source_partfield: Sequence[int],
    n_oracle: int,
    n_partfield: int,
) -> List[List[int]]:
    table = [[0 for _ in range(n_partfield)] for _ in range(n_oracle)]
    for oracle_label, partfield_label in zip(source_oracle, source_partfield):
        if 0 <= oracle_label < n_oracle and 0 <= partfield_label < n_partfield:
            table[oracle_label][partfield_label] += 1
    return table


def write_report(
    path: Path,
    source_path: Path,
    target_path: Path,
    output_dir: Path,
    source_labels: Sequence[int],
    target_labels: Sequence[int],
    partfield_npz: Path | None,
) -> None:
    source_counts = counts(source_labels)
    target_counts = counts(target_labels)
    data = {
        "source_mesh": str(source_path),
        "target_mesh": str(target_path),
        "n_buckets": len(BUCKET_NAMES),
        "bucket_names": list(BUCKET_NAMES),
        "source_counts": source_counts,
        "target_counts": target_counts,
        "note": "Labels are assigned on the source/reference mesh and copied to the target by vertex id because the meshes share identical topology.",
    }
    if partfield_npz and partfield_npz.is_file():
        partfield_source = read_npz_int64(partfield_npz, "source_labels")
        partfield_target = read_npz_int64(partfield_npz, "target_labels")
        pf_n = max(partfield_source + partfield_target) + 1 if partfield_source or partfield_target else 0
        pf_matches = sum(1 for src, tgt in zip(partfield_source, partfield_target) if src == tgt)
        data["partfield_counts"] = {
            "source": {f"bucket_{idx:02d}": partfield_source.count(idx) for idx in range(pf_n)},
            "target": {f"bucket_{idx:02d}": partfield_target.count(idx) for idx in range(pf_n)},
        }
        data["same_topology_label_agreement"] = {
            "oracle": {
                "matches": len(source_labels),
                "mismatches": 0,
                "agreement": 1.0,
            },
            "partfield": {
                "matches": int(pf_matches),
                "mismatches": int(len(partfield_source) - pf_matches),
                "agreement": float(pf_matches / max(len(partfield_source), 1)),
            },
        }
        data["oracle_x_partfield_source_confusion"] = confusion(
            source_labels,
            partfield_source,
            len(BUCKET_NAMES),
            pf_n,
        )
    (output_dir / "summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    lines = [
        "# Oracle Semantic Bucket Trial",
        "",
        "This is a local oracle-proxy test for the cat-to-dachshund run.",
        "The source and target OBJ files have identical face topology, so the same vertex id denotes the same latent surface point across both meshes.",
        "",
        "## Artifacts",
        "",
        f"- Label pack: `{output_dir / 'labels' / 'cat_to_dachshund_oracle_semantic_labels.npz'}`",
        f"- Source labels: `{output_dir / 'labels' / 'source_cat_oracle_semantic_labels.npz'}`",
        f"- Target labels: `{output_dir / 'labels' / 'target_dachshund_oracle_semantic_labels.npz'}`",
        f"- Source colored mesh: `{output_dir / 'colored' / 'source_cat_oracle_semantic.ply'}`",
        f"- Target colored mesh: `{output_dir / 'colored' / 'target_dachshund_oracle_semantic.ply'}`",
        f"- SVG visual summary: `{output_dir / 'visuals' / 'oracle_semantic_summary.svg'}`",
        f"- PNG visual summary: `{output_dir / 'visuals' / 'oracle_semantic_summary.png'}`",
        f"- Machine-readable summary: `{output_dir / 'summary.json'}`",
        "",
        "## Bucket Counts",
        "",
        "| id | bucket | source vertices | target vertices |",
        "| --- | --- | ---: | ---: |",
    ]
    for idx, name in enumerate(BUCKET_NAMES):
        key = f"{idx:02d}_{name}"
        lines.append(f"| {idx} | {name} | {source_counts[key]} | {target_counts[key]} |")
    if partfield_npz and partfield_npz.is_file():
        lines.extend(
            [
                "",
                "## PartField Comparison",
                "",
                f"Compared against `{partfield_npz}`.",
                "",
                "Because the source and target share topology, corresponding vertices should keep the same semantic bucket.",
                f"The oracle labels match at `4712 / 4712` vertices; PartField matches at `{pf_matches} / {len(partfield_source)}` vertices.",
                "",
                "Rows are oracle buckets and columns are PartField buckets.",
                "",
            ]
        )
        table = data["oracle_x_partfield_source_confusion"]  # type: ignore[index]
        pf_n = len(table[0]) if table else 0
        lines.append("| oracle bucket | " + " | ".join(f"pf{idx:02d}" for idx in range(pf_n)) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in range(pf_n)) + " |")
        for idx, row in enumerate(table):
            lines.append(f"| {idx:02d}_{BUCKET_NAMES[idx]} | " + " | ".join(str(v) for v in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    source_path = Path(args.source)
    target_path = Path(args.target)
    output_dir = Path(args.output_dir)
    source_vertices, source_faces = load_obj(source_path)
    target_vertices, target_faces = load_obj(target_path)
    if len(source_vertices) != len(target_vertices):
        raise ValueError("Oracle transfer requires identical vertex counts.")
    if source_faces != target_faces:
        raise ValueError("Oracle transfer requires identical face topology.")

    source_labels = make_semantic_labels(source_vertices)
    target_labels = list(source_labels)

    labels_path = output_dir / "labels" / "cat_to_dachshund_oracle_semantic_labels.npz"
    source_labels_path = output_dir / "labels" / "source_cat_oracle_semantic_labels.npz"
    target_labels_path = output_dir / "labels" / "target_dachshund_oracle_semantic_labels.npz"
    write_label_npz(labels_path, source_labels, target_labels, source_faces, target_faces)
    write_single_label_npz(source_labels_path, source_labels, source_faces)
    write_single_label_npz(target_labels_path, target_labels, target_faces)
    write_colored_ply(output_dir / "colored" / "source_cat_oracle_semantic.ply", source_vertices, source_faces, source_labels)
    write_colored_ply(output_dir / "colored" / "target_dachshund_oracle_semantic.ply", target_vertices, target_faces, target_labels)
    write_svg_summary(
        output_dir / "visuals" / "oracle_semantic_summary.svg",
        source_vertices,
        source_faces,
        target_vertices,
        target_faces,
        source_labels,
        target_labels,
    )
    write_png_summary(
        output_dir / "visuals" / "oracle_semantic_summary.png",
        source_vertices,
        source_faces,
        target_vertices,
        target_faces,
        source_labels,
        target_labels,
    )
    write_report(
        output_dir / "README.md",
        source_path,
        target_path,
        output_dir,
        source_labels,
        target_labels,
        Path(args.partfield_npz) if args.partfield_npz else None,
    )
    print(json.dumps({
        "labels": str(labels_path),
        "source_labels": str(source_labels_path),
        "target_labels": str(target_labels_path),
        "source_colored": str(output_dir / "colored" / "source_cat_oracle_semantic.ply"),
        "target_colored": str(output_dir / "colored" / "target_dachshund_oracle_semantic.ply"),
        "svg": str(output_dir / "visuals" / "oracle_semantic_summary.svg"),
        "png": str(output_dir / "visuals" / "oracle_semantic_summary.png"),
        "report": str(output_dir / "README.md"),
        "source_counts": counts(source_labels),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partfield-npz", default="")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
