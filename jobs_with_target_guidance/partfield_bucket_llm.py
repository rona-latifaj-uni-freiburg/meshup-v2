#!/usr/bin/env python3
"""Choose PartField bucket counts from a source/target mesh pair.

This is intentionally an optional extension. Existing MeshUp/PartField jobs can
keep setting ``PARTFIELD_N_BUCKETS`` by hand; callers opt in by invoking this
module or by setting the runner hook's ``PARTFIELD_BUCKET_SELECTOR=llm``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import struct
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_CSV = (
    PROJECT_ROOT
    / "jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy/"
    / "fair_comparison_meshup_vs_densematcher_pf_summary.csv"
)

Mode = Literal["fixed", "range"]

CATEGORY_ALIASES = {
    "animal": "animals",
    "animals": "animals",
    "car": "cars",
    "cars": "cars",
    "tomatoe": "tomato",
    "tomatoes": "tomato",
}

COMMON_OBJECT_HINTS = {
    "apple": (2, [1, 2, 3]),
    "backpack": (4, [4, 5, 6]),
    "banana": (2, [1, 2, 3]),
    "bell_pepper": (3, [2, 3, 4]),
    "bread": (2, [1, 2, 3, 4]),
    "broccoli": (2, [1, 2, 3]),
    "carrot": (3, [2, 3, 4]),
    "cars": (6, [4, 5, 6, 7, 8]),
    "celery": (4, [3, 4, 5, 6]),
    "chairs": (8, [6, 7, 8]),
    "cucumber": (2, [1, 2, 3]),
    "egg": (1, [1, 2, 3]),
    "eggplant": (2, [1, 2, 3]),
    "knife": (2, [1, 2, 3, 4]),
    "mushroom": (2, [1, 2, 3]),
    "onion": (2, [1, 2, 3]),
    "peach": (1, [1, 2, 3]),
    "pear": (2, [1, 2, 3]),
    "potato": (1, [1, 2, 3]),
    "pumpkin": (2, [1, 2, 3]),
    "shampoo": (3, [2, 3, 4]),
    "tomato": (2, [1, 2, 3]),
    "zucchini": (2, [1, 2, 3]),
    "animals": (6, [4, 5, 6, 7]),
}

OBJ_RE = re.compile(r"^v\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)")
PF_RE = re.compile(r"PF?0*([0-9]+)", re.IGNORECASE)


@dataclass
class MeshSummary:
    path: str
    name: str
    category_guess: Optional[str]
    exists: bool
    file_format: str
    n_vertices: int = 0
    n_faces: int = 0
    bbox_min: Optional[list[float]] = None
    bbox_max: Optional[list[float]] = None
    bbox_extent: Optional[list[float]] = None
    normalized_extent: Optional[list[float]] = None
    aspect_ratio: Optional[float] = None
    surface_area: Optional[float] = None
    connected_components: Optional[int] = None
    isolated_vertices: Optional[int] = None
    warnings: Optional[list[str]] = None


@dataclass
class BucketPrior:
    category: str
    best_bucket: int
    tested_buckets: list[int]
    best_auc: float
    best_err: float


@dataclass
class BucketDecision:
    bucket: int
    buckets: list[int]
    mode: Mode
    provider: str
    confidence: float
    rationale: str
    category_guess: Optional[str]
    source: MeshSummary
    target: MeshSummary
    warnings: list[str]


def normalize_category(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return CATEGORY_ALIASES.get(key, key)


def parse_bucket_token(value: str) -> Optional[int]:
    match = PF_RE.search(str(value))
    if not match:
        return None
    return int(match.group(1))


def parse_bucket_list(value: str | Sequence[int] | None, default: Sequence[int]) -> list[int]:
    if value is None:
        return sorted(set(int(v) for v in default))
    if isinstance(value, str):
        raw = re.findall(r"\d+", value)
        buckets = [int(piece) for piece in raw]
    else:
        buckets = [int(piece) for piece in value]
    buckets = [bucket for bucket in buckets if bucket > 0]
    if not buckets:
        raise ValueError("No positive bucket counts were provided.")
    return sorted(set(buckets))


def detect_request_mode(request_mode: str, user_request: str) -> Mode:
    if request_mode in {"fixed", "range"}:
        return request_mode  # type: ignore[return-value]
    text = user_request.lower()
    if any(word in text for word in ("range", "sweep", "try", "list", "candidates", "options")):
        return "range"
    return "fixed"


def infer_category_from_path(path: Path) -> Optional[str]:
    parts = path.parts
    for marker in ("DenseCorr3D", "densecorr3d"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return normalize_category(parts[idx + 1])

    for idx, part in enumerate(parts):
        if part == "densecorr3d_categories" and idx + 2 < len(parts):
            return normalize_category(parts[idx + 2])

    text = " ".join(path.parts[-4:]).lower()
    for category in COMMON_OBJECT_HINTS:
        if re.search(rf"\b{re.escape(category)}\b", text.replace("_", " ")):
            return normalize_category(category)
    return None


def infer_category_from_text(text: str) -> Optional[str]:
    normalized = text.lower().replace("_", " ")
    for alias, category in CATEGORY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias.replace('_', ' '))}\b", normalized):
            return category
    for category in COMMON_OBJECT_HINTS:
        label = category.replace("_", " ")
        if re.search(rf"\b{re.escape(label)}\b", normalized):
            return normalize_category(category)
    return None


def round3(values: Iterable[float]) -> list[float]:
    return [round(float(value), 3) for value in values]


def face_indices_from_obj(tokens: Sequence[str], n_vertices: int) -> list[int]:
    out = []
    for token in tokens:
        raw = token.split("/")[0]
        if not raw:
            continue
        idx = int(raw)
        out.append(n_vertices + idx if idx < 0 else idx - 1)
    return [idx for idx in out if 0 <= idx < n_vertices]


def triangle_area(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2])


def parse_obj(path: Path) -> tuple[list[tuple[float, float, float]], list[list[int]], list[str]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    warnings: list[str] = []
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                match = OBJ_RE.match(line)
                if match:
                    vertices.append((float(match.group(1)), float(match.group(2)), float(match.group(3))))
            elif line.startswith("f "):
                face = face_indices_from_obj(line.split()[1:], len(vertices))
                if len(face) >= 3:
                    faces.append(face)
    if not vertices:
        warnings.append("no OBJ vertices were parsed")
    if not faces:
        warnings.append("no OBJ faces were parsed")
    return vertices, faces, warnings


PLY_SCALAR_TYPES = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def parse_ply_header(raw: bytes) -> tuple[list[str], int]:
    marker = b"end_header"
    idx = raw.find(marker)
    if idx < 0:
        raise ValueError("PLY header has no end_header marker")
    newline = raw.find(b"\n", idx)
    if newline < 0:
        newline = idx + len(marker)
    header = raw[:newline].decode("ascii", errors="replace").splitlines()
    return header, newline + 1


def parse_ply(path: Path) -> tuple[list[tuple[float, float, float]], list[list[int]], int, list[str]]:
    data = path.read_bytes()
    header, offset = parse_ply_header(data)
    warnings: list[str] = []
    fmt = "ascii"
    vertex_count = 0
    face_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element: Optional[str] = None

    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
            elif current_element == "face":
                face_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) >= 3 and parts[1] != "list":
                vertex_properties.append((parts[1], parts[2]))
            else:
                warnings.append("list-valued vertex properties are not summarized")

    prop_names = [name for _kind, name in vertex_properties]
    if not {"x", "y", "z"}.issubset(set(prop_names)):
        return [], [], face_count, ["PLY vertex x/y/z properties were not found"]
    xyz_indices = [prop_names.index(axis) for axis in ("x", "y", "z")]

    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    if fmt == "ascii":
        text = data[offset:].decode("utf-8", errors="replace").splitlines()
        for line in text[:vertex_count]:
            parts = line.split()
            if len(parts) > max(xyz_indices):
                vertices.append(tuple(float(parts[idx]) for idx in xyz_indices))  # type: ignore[arg-type]
        for line in text[vertex_count : vertex_count + face_count]:
            parts = line.split()
            if not parts:
                continue
            n = int(parts[0])
            ids = [int(piece) for piece in parts[1 : 1 + n]]
            if len(ids) >= 3:
                faces.append(ids)
    elif fmt in {"binary_little_endian", "binary_big_endian"}:
        endian = "<" if fmt == "binary_little_endian" else ">"
        struct_pieces = []
        for kind, _name in vertex_properties:
            if kind not in PLY_SCALAR_TYPES:
                return [], [], face_count, [f"unsupported PLY vertex property type {kind!r}"]
            struct_pieces.append(PLY_SCALAR_TYPES[kind][0])
        vertex_struct = struct.Struct(endian + "".join(struct_pieces))
        cursor = offset
        for _idx in range(vertex_count):
            row = vertex_struct.unpack_from(data, cursor)
            cursor += vertex_struct.size
            vertices.append(tuple(float(row[idx]) for idx in xyz_indices))  # type: ignore[arg-type]
        warnings.append("binary PLY face connectivity is not summarized")
    else:
        warnings.append(f"unsupported PLY format {fmt!r}")

    return vertices, faces, face_count, warnings


def union_components(n_vertices: int, faces: Sequence[Sequence[int]]) -> tuple[Optional[int], Optional[int]]:
    if not faces or n_vertices <= 0:
        return None, None
    parent = list(range(n_vertices))
    touched = [False] * n_vertices

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for face in faces:
        valid = [idx for idx in face if 0 <= idx < n_vertices]
        if len(valid) < 2:
            continue
        for idx in valid:
            touched[idx] = True
        first = valid[0]
        for idx in valid[1:]:
            union(first, idx)

    roots = {find(idx) for idx, was_touched in enumerate(touched) if was_touched}
    isolated = sum(1 for was_touched in touched if not was_touched)
    return len(roots) + isolated, isolated


def summarize_vertices(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    path: Path,
    file_format: str,
    warnings: list[str],
    face_count_override: Optional[int] = None,
) -> MeshSummary:
    category = infer_category_from_path(path)
    n_vertices = len(vertices)
    n_faces = face_count_override if face_count_override is not None else len(faces)
    bbox_min = bbox_max = extent = normalized_extent = None
    aspect_ratio = surface_area = None
    components = isolated = None

    if vertices:
        mins = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
        maxs = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
        ext = [maxs[axis] - mins[axis] for axis in range(3)]
        max_extent = max(ext) if ext else 0.0
        bbox_min = round3(mins)
        bbox_max = round3(maxs)
        extent = round3(ext)
        normalized_extent = round3((value / max_extent if max_extent > 0 else 0.0) for value in ext)
        nonzero = [value for value in ext if value > 1e-12]
        if nonzero:
            aspect_ratio = round(max(nonzero) / max(min(nonzero), 1e-12), 3)

    if faces and vertices:
        area = 0.0
        for face in faces:
            valid = [idx for idx in face if 0 <= idx < len(vertices)]
            if len(valid) < 3:
                continue
            a = vertices[valid[0]]
            for pos in range(1, len(valid) - 1):
                area += triangle_area(a, vertices[valid[pos]], vertices[valid[pos + 1]])
        surface_area = round(area, 3)
        components, isolated = union_components(len(vertices), faces)

    return MeshSummary(
        path=str(path),
        name=path.stem,
        category_guess=category,
        exists=path.is_file(),
        file_format=file_format,
        n_vertices=n_vertices,
        n_faces=n_faces,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_extent=extent,
        normalized_extent=normalized_extent,
        aspect_ratio=aspect_ratio,
        surface_area=surface_area,
        connected_components=components,
        isolated_vertices=isolated,
        warnings=warnings or None,
    )


def summarize_mesh(path_value: str | Path) -> MeshSummary:
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.is_file():
        return MeshSummary(
            path=str(path),
            name=path.stem,
            category_guess=infer_category_from_path(path),
            exists=False,
            file_format=path.suffix.lower().lstrip(".") or "unknown",
            warnings=["mesh path does not exist"],
        )

    suffix = path.suffix.lower()
    try:
        if suffix == ".obj":
            vertices, faces, warnings = parse_obj(path)
            return summarize_vertices(vertices, faces, path, "obj", warnings)
        if suffix == ".ply":
            vertices, faces, face_count, warnings = parse_ply(path)
            return summarize_vertices(vertices, faces, path, "ply", warnings, face_count_override=face_count)
    except Exception as exc:  # pragma: no cover - keeps Slurm jobs debuggable.
        return MeshSummary(
            path=str(path),
            name=path.stem,
            category_guess=infer_category_from_path(path),
            exists=True,
            file_format=suffix.lstrip(".") or "unknown",
            warnings=[f"mesh summary failed: {exc}"],
        )

    return MeshSummary(
        path=str(path),
        name=path.stem,
        category_guess=infer_category_from_path(path),
        exists=True,
        file_format=suffix.lstrip(".") or "unknown",
        warnings=[f"unsupported mesh extension {suffix!r}"],
    )


def load_calibration(paths: Sequence[Path]) -> dict[str, BucketPrior]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                category = normalize_category(row.get("category"))
                bucket = parse_bucket_token(row.get("pf_setting", ""))
                if not category or bucket is None:
                    continue
                try:
                    auc = float(row.get("meshup_auc", "nan"))
                    err = float(row.get("meshup_err", "nan"))
                except ValueError:
                    continue
                grouped.setdefault(category, []).append({"bucket": bucket, "auc": auc, "err": err})

    priors: dict[str, BucketPrior] = {}
    for category, rows in grouped.items():
        if not rows:
            continue
        best = max(rows, key=lambda item: (item["auc"], -item["err"]))
        priors[category] = BucketPrior(
            category=category,
            best_bucket=int(best["bucket"]),
            tested_buckets=sorted({int(row["bucket"]) for row in rows}),
            best_auc=float(best["auc"]),
            best_err=float(best["err"]),
        )
    return priors


def merge_category_guess(source: MeshSummary, target: MeshSummary, prompt: str, user_request: str) -> Optional[str]:
    guesses = [
        source.category_guess,
        target.category_guess,
        infer_category_from_text(prompt),
        infer_category_from_text(user_request),
        infer_category_from_text(source.name),
        infer_category_from_text(target.name),
    ]
    counts: dict[str, int] = {}
    for guess in guesses:
        guess = normalize_category(guess)
        if guess:
            counts[guess] = counts.get(guess, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def nearest_bucket(bucket: int, candidates: Sequence[int]) -> int:
    return min(candidates, key=lambda candidate: (abs(candidate - bucket), candidate))


def centered_bucket_range(bucket: int, candidates: Sequence[int], size: int) -> list[int]:
    ordered = sorted(set(candidates))
    if bucket not in ordered:
        bucket = nearest_bucket(bucket, ordered)
    idx = ordered.index(bucket)
    left = max(0, idx - size // 2)
    right = min(len(ordered), left + size)
    left = max(0, right - size)
    return ordered[left:right]


def heuristic_decision(
    source: MeshSummary,
    target: MeshSummary,
    prompt: str,
    user_request: str,
    mode: Mode,
    candidates: Sequence[int],
    priors: dict[str, BucketPrior],
    range_size: int,
    provider_name: str = "heuristic",
) -> BucketDecision:
    warnings: list[str] = []
    category = merge_category_guess(source, target, prompt, user_request)
    candidate_set = sorted(set(int(v) for v in candidates))
    best_bucket = None
    rationale = ""
    confidence = 0.45

    if category and category in priors:
        prior = priors[category]
        best_bucket = prior.best_bucket
        tested_overlap = [bucket for bucket in prior.tested_buckets if bucket in candidate_set]
        if tested_overlap:
            candidate_set = tested_overlap
        rationale = (
            f"Matched calibration prior for {category}: best PF{prior.best_bucket:02d} "
            f"from tested {prior.tested_buckets}."
        )
        confidence = 0.78
    elif category and category in COMMON_OBJECT_HINTS:
        best_bucket, hinted_range = COMMON_OBJECT_HINTS[category]
        hinted_overlap = [bucket for bucket in hinted_range if bucket in candidate_set]
        if hinted_overlap:
            candidate_set = hinted_overlap
        rationale = f"Used built-in structural prior for {category}."
        confidence = 0.58
    else:
        source_aspect = source.aspect_ratio or 1.0
        target_aspect = target.aspect_ratio or 1.0
        aspect = max(source_aspect, target_aspect)
        component_count = max(source.connected_components or 1, target.connected_components or 1)
        if component_count >= 6:
            best_bucket = 8
        elif aspect >= 4.0:
            best_bucket = 3
        elif aspect >= 2.0:
            best_bucket = 4
        else:
            best_bucket = 5
        rationale = "Used geometry-only fallback from component count and bounding-box aspect."
        confidence = 0.38

    if not candidate_set:
        candidate_set = sorted(set(int(v) for v in candidates))
    bucket = nearest_bucket(int(best_bucket), candidate_set)
    if bucket != best_bucket:
        warnings.append(f"preferred bucket {best_bucket} was outside candidates; using nearest candidate {bucket}")
    buckets = [bucket] if mode == "fixed" else centered_bucket_range(bucket, candidate_set, max(1, range_size))
    return BucketDecision(
        bucket=bucket,
        buckets=buckets,
        mode=mode,
        provider=provider_name,
        confidence=confidence,
        rationale=rationale,
        category_guess=category,
        source=source,
        target=target,
        warnings=warnings,
    )


def json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "bucket": {"type": "integer", "minimum": 1, "maximum": 64},
            "buckets": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 64},
                "minItems": 1,
                "maxItems": 8,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string", "maxLength": 600},
        },
        "required": ["bucket", "buckets", "confidence", "rationale"],
    }


def build_llm_context(
    source: MeshSummary,
    target: MeshSummary,
    prompt: str,
    user_request: str,
    mode: Mode,
    candidates: Sequence[int],
    priors: dict[str, BucketPrior],
) -> dict[str, Any]:
    category = merge_category_guess(source, target, prompt, user_request)
    relevant_priors = {name: asdict(prior) for name, prior in priors.items() if name == category}
    if not relevant_priors:
        relevant_priors = {
            name: asdict(prior)
            for name, prior in priors.items()
            if name in {"animals", "backpack", "broccoli", "cars", "tomato", "celery", "chairs", "knife"}
        }
    return {
        "task": "Choose a shared PartField bucket count for source-target mesh co-segmentation before MeshUp deformation.",
        "requested_output_mode": mode,
        "candidate_buckets": list(candidates),
        "prompt": prompt,
        "user_request": user_request,
        "category_guess": category,
        "source_mesh": asdict(source),
        "target_mesh": asdict(target),
        "calibration_priors": relevant_priors,
        "guidance": [
            "Use fewer buckets for smooth single-lump objects.",
            "Use more buckets for articulated animals, vehicles, chairs, and repeated structures.",
            "If a category calibration prior matches, prefer it unless the mesh statistics strongly contradict it.",
            "For fixed mode, return buckets as a one-item list containing bucket.",
            "For range mode, return 2-5 candidate bucket counts and set bucket to the best first run.",
        ],
    }


def openai_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def call_openai(context: dict[str, Any], model: str, api_base: str, timeout: float) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You choose PartField bucket counts for mesh co-segmentation. "
                    "Be conservative: too many buckets on smooth objects fragments guidance; too few buckets on "
                    "articulated or repeated-part objects loses useful correspondences. Return only valid JSON."
                ),
            },
            {"role": "user", "content": json.dumps(context, indent=2, sort_keys=True)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "partfield_bucket_decision",
                "strict": True,
                "schema": json_schema(),
            }
        },
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        api_base.rstrip("/") + "/responses",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            raw = handle.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed with HTTP {exc.code}: {detail}") from exc
    response = json.loads(raw)
    text = openai_response_text(response)
    if not text:
        raise RuntimeError(f"OpenAI response did not contain output text: {raw[:500]}")
    return json.loads(text)


def parse_mock_response(value: str) -> dict[str, Any]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--mock-response must be JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("--mock-response must decode to a JSON object")
    return data


def validate_llm_decision(
    raw: dict[str, Any],
    source: MeshSummary,
    target: MeshSummary,
    prompt: str,
    user_request: str,
    mode: Mode,
    candidates: Sequence[int],
    provider: str,
    range_size: int,
) -> BucketDecision:
    warnings: list[str] = []
    candidate_set = sorted(set(int(value) for value in candidates))
    if not candidate_set:
        raise ValueError("candidate bucket list is empty")

    raw_bucket = raw.get("bucket")
    raw_buckets = raw.get("buckets")
    if raw_bucket is None and isinstance(raw_buckets, list) and raw_buckets:
        raw_bucket = raw_buckets[0]
    if raw_bucket is None:
        raise ValueError("LLM response is missing bucket")

    try:
        bucket = int(raw_bucket)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LLM bucket is not an integer: {raw_bucket!r}") from exc

    if isinstance(raw_buckets, list):
        buckets = []
        for item in raw_buckets:
            try:
                buckets.append(int(item))
            except (TypeError, ValueError):
                warnings.append(f"dropped non-integer bucket value {item!r}")
        buckets = sorted(set(buckets))
    else:
        buckets = [bucket]

    if bucket not in candidate_set:
        replacement = nearest_bucket(bucket, candidate_set)
        warnings.append(f"LLM bucket {bucket} was outside candidates; using nearest candidate {replacement}")
        bucket = replacement
    buckets = [value for value in buckets if value in candidate_set]
    if not buckets:
        buckets = [bucket]
    if bucket not in buckets:
        buckets.insert(0, bucket)
    if mode == "fixed":
        buckets = [bucket]
    elif len(buckets) == 1:
        buckets = centered_bucket_range(bucket, candidate_set, max(1, range_size))

    confidence = raw.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.5
        warnings.append("LLM confidence was not numeric")

    rationale = str(raw.get("rationale", "LLM selected the bucket count.")).strip()
    if len(rationale) > 600:
        rationale = rationale[:597] + "..."

    return BucketDecision(
        bucket=bucket,
        buckets=buckets,
        mode=mode,
        provider=provider,
        confidence=confidence,
        rationale=rationale,
        category_guess=merge_category_guess(source, target, prompt, user_request),
        source=source,
        target=target,
        warnings=warnings,
    )


def choose_buckets(
    source_mesh: str | Path,
    target_mesh: str | Path,
    prompt: str,
    user_request: str,
    request_mode: str,
    candidate_buckets: Sequence[int],
    calibration_paths: Sequence[Path],
    provider: str,
    model: str,
    api_base: str,
    timeout: float,
    range_size: int,
    mock_response: Optional[str] = None,
) -> BucketDecision:
    mode = detect_request_mode(request_mode, user_request)
    source = summarize_mesh(source_mesh)
    target = summarize_mesh(target_mesh)
    priors = load_calibration(calibration_paths)

    if provider == "auto":
        provider = "openai" if os.environ.get("OPENAI_API_KEY") else "heuristic"

    if provider == "heuristic":
        return heuristic_decision(source, target, prompt, user_request, mode, candidate_buckets, priors, range_size)

    if provider == "mock":
        if mock_response is None:
            raise ValueError("--provider mock requires --mock-response")
        raw = parse_mock_response(mock_response)
        return validate_llm_decision(
            raw,
            source,
            target,
            prompt,
            user_request,
            mode,
            candidate_buckets,
            "mock",
            range_size,
        )

    if provider != "openai":
        raise ValueError(f"Unsupported provider {provider!r}")

    context = build_llm_context(source, target, prompt, user_request, mode, candidate_buckets, priors)
    raw = call_openai(context, model=model, api_base=api_base, timeout=timeout)
    return validate_llm_decision(
        raw,
        source,
        target,
        prompt,
        user_request,
        mode,
        candidate_buckets,
        "openai",
        range_size,
    )


def decision_to_public_dict(decision: BucketDecision) -> dict[str, Any]:
    return {
        "bucket": decision.bucket,
        "buckets": decision.buckets,
        "mode": decision.mode,
        "provider": decision.provider,
        "confidence": decision.confidence,
        "rationale": decision.rationale,
        "category_guess": decision.category_guess,
        "source": asdict(decision.source),
        "target": asdict(decision.target),
        "warnings": decision.warnings,
    }


def print_decision(decision: BucketDecision, print_key: str) -> None:
    if print_key == "bucket":
        print(decision.bucket)
    elif print_key == "buckets_csv":
        print(",".join(str(bucket) for bucket in decision.buckets))
    elif print_key == "shell":
        print(f"PARTFIELD_N_BUCKETS={decision.bucket}")
        print(f"PARTFIELD_BUCKETS={','.join(str(bucket) for bucket in decision.buckets)}")
        print(f"PARTFIELD_BUCKET_SELECTOR_PROVIDER={decision.provider}")
        print(f"PARTFIELD_BUCKET_SELECTOR_CONFIDENCE={decision.confidence:.3f}")
    else:
        print(json.dumps(decision_to_public_dict(decision), indent=2, sort_keys=True))


def calibration_self_test(
    categories: Sequence[str],
    request_mode: str,
    candidates: Sequence[int],
    calibration_paths: Sequence[Path],
    range_size: int,
) -> int:
    priors = load_calibration(calibration_paths)
    if not priors:
        print("No calibration rows found.", file=sys.stderr)
        return 1
    selected_categories = [normalize_category(category) for category in categories] if categories else sorted(priors)
    selected_categories = [category for category in selected_categories if category]
    print("category,known_best,selected_bucket,selected_buckets,hit,mode,rationale")
    misses = 0
    for category in selected_categories:
        prior = priors.get(category)
        if not prior:
            print(f"{category},,missing,,False,{request_mode},no calibration prior")
            misses += 1
            continue
        candidate_list = [bucket for bucket in candidates if bucket in prior.tested_buckets] or list(candidates)
        source = MeshSummary(
            path=f"synthetic/{category}/source.obj",
            name=f"{category}_source",
            category_guess=category,
            exists=False,
            file_format="obj",
        )
        target = MeshSummary(
            path=f"synthetic/{category}/target.obj",
            name=f"{category}_target",
            category_guess=category,
            exists=False,
            file_format="obj",
        )
        decision = heuristic_decision(
            source,
            target,
            prompt=f"a {category.replace('_', ' ')}",
            user_request=request_mode,
            mode=detect_request_mode(request_mode, request_mode),
            candidates=candidate_list,
            priors=priors,
            range_size=range_size,
            provider_name="calibration-self-test",
        )
        hit = prior.best_bucket in decision.buckets
        misses += 0 if hit else 1
        print(
            f"{category},PF{prior.best_bucket:02d},PF{decision.bucket:02d},"
            f"{'|'.join(f'PF{bucket:02d}' for bucket in decision.buckets)},{hit},"
            f"{decision.mode},{decision.rationale}"
        )
    return misses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", help="Source mesh path (.obj or .ply).")
    parser.add_argument("--target-mesh", help="Target mesh path (.obj or .ply).")
    parser.add_argument("--prompt", default="", help="Target text prompt or object description.")
    parser.add_argument("--user-request", default="", help="Natural-language user request for fixed/range behavior.")
    parser.add_argument("--mode", choices=["auto", "fixed", "range"], default="auto")
    parser.add_argument(
        "--candidate-buckets",
        default="1,2,3,4,5,6,7,8,9,10,12,16,20",
        help="Allowed bucket counts, e.g. 4,5,6 or 04,05,06.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "heuristic", "mock"],
        default=os.environ.get("PARTFIELD_BUCKET_SELECTOR_PROVIDER", "auto"),
    )
    parser.add_argument("--model", default=os.environ.get("PARTFIELD_BUCKET_LLM_MODEL", "gpt-5"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--mock-response", default=os.environ.get("PARTFIELD_BUCKET_SELECTOR_MOCK_RESPONSE"))
    parser.add_argument("--range-size", type=int, default=3)
    parser.add_argument(
        "--calibration-csv",
        action="append",
        default=[],
        help="CSV with category,pf_setting,meshup_auc,meshup_err columns. May be repeated.",
    )
    parser.add_argument("--no-default-calibration", action="store_true")
    parser.add_argument("--output", help="Optional full JSON output path.")
    parser.add_argument("--print-key", choices=["json", "bucket", "buckets_csv", "shell"], default="json")
    parser.add_argument(
        "--self-test-calibration",
        action="store_true",
        help="Offline test: compare heuristic selections with the existing best-PF calibration table.",
    )
    parser.add_argument("--self-test-category", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = parse_bucket_list(args.candidate_buckets, default=[])
    calibration_paths = [Path(path) for path in args.calibration_csv]
    if not args.no_default_calibration:
        calibration_paths.append(DEFAULT_CALIBRATION_CSV)

    if args.self_test_calibration:
        raise SystemExit(
            calibration_self_test(
                args.self_test_category,
                args.mode,
                candidates,
                calibration_paths,
                args.range_size,
            )
        )

    if not args.source_mesh or not args.target_mesh:
        raise SystemExit("--source-mesh and --target-mesh are required unless --self-test-calibration is used.")

    decision = choose_buckets(
        source_mesh=args.source_mesh,
        target_mesh=args.target_mesh,
        prompt=args.prompt,
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
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(decision_to_public_dict(decision), indent=2, sort_keys=True) + "\n")
    print_decision(decision, args.print_key)


if __name__ == "__main__":
    main()
