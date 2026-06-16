"""Part-aware target Chamfer guidance for MeshUp.

The loss keeps vertex identity meaningful by freezing source part labels from
the initial mesh, then only matching each source part to the same target part.
For the SAM3D car meshes, the default schema assumes:

- longitudinal axis: z
- lateral axis: x
- vertical axis: y
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_index(axis: str) -> int:
    try:
        return AXIS_TO_INDEX[axis.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown axis '{axis}'. Expected one of x, y, z.") from exc


def unit_vertices(vertices: torch.Tensor) -> torch.Tensor:
    """Normalize vertices to MeshUp-style unit size while preserving gradients."""
    vmin = vertices.amin(dim=0)
    vmax = vertices.amax(dim=0)
    scale = 2.0 / torch.clamp((vmax - vmin).max(), min=1e-8)
    return (vertices - (vmax + vmin) * 0.5) * scale


def _bbox_coordinates(vertices: torch.Tensor) -> torch.Tensor:
    """Map each axis independently to [0, 1]. Used only for fixed part labels."""
    vertices = vertices.detach()
    vmin = vertices.amin(dim=0)
    vmax = vertices.amax(dim=0)
    return (vertices - vmin) / torch.clamp(vmax - vmin, min=1e-8)


def _sample_vertices(vertices: torch.Tensor, n_points: int) -> torch.Tensor:
    if n_points <= 0 or vertices.shape[0] <= n_points:
        return vertices
    idx = torch.linspace(0, vertices.shape[0] - 1, steps=n_points, device=vertices.device).long()
    return vertices.index_select(0, idx)


def _symmetric_chamfer(src_vertices: torch.Tensor, tgt_vertices: torch.Tensor, n_points: int) -> torch.Tensor:
    src = _sample_vertices(src_vertices, n_points)
    tgt = _sample_vertices(tgt_vertices, n_points)
    dists = torch.cdist(src, tgt, p=2).pow(2)
    return dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()


def _build_bbox_labels(
    vertices: torch.Tensor,
    longitudinal_axis: str,
    lateral_axis: str,
    vertical_axis: str,
    flip_longitudinal: bool = False,
) -> Tuple[torch.Tensor, List[str]]:
    """Generic 3 x 2 x 2 spatial grid for non-car sanity checks."""
    coords = _bbox_coordinates(vertices)
    long = coords[:, _axis_index(longitudinal_axis)]
    lat = coords[:, _axis_index(lateral_axis)]
    up = coords[:, _axis_index(vertical_axis)]
    if flip_longitudinal:
        long = 1.0 - long

    long_bin = torch.clamp((long * 3).long(), min=0, max=2)
    lat_bin = torch.clamp((lat * 2).long(), min=0, max=1)
    up_bin = torch.clamp((up * 2).long(), min=0, max=1)
    labels = long_bin * 4 + up_bin * 2 + lat_bin

    names = []
    for long_name in ["rear", "middle", "front"]:
        for up_name in ["lower", "upper"]:
            for side_name in ["left", "right"]:
                names.append(f"{long_name}_{up_name}_{side_name}")
    return labels, names


def _build_car_labels(
    vertices: torch.Tensor,
    longitudinal_axis: str,
    lateral_axis: str,
    vertical_axis: str,
    flip_longitudinal: bool = False,
) -> Tuple[torch.Tensor, List[str]]:
    """Heuristic car parts from normalized XYZ slices.

    The labels are intentionally rough. They are meant to prevent gross
    roof-to-bumper or wheel-to-hood matches, not replace annotated parts.
    """
    coords = _bbox_coordinates(vertices)
    long = coords[:, _axis_index(longitudinal_axis)]
    lat = coords[:, _axis_index(lateral_axis)]
    up = coords[:, _axis_index(vertical_axis)]
    if flip_longitudinal:
        long = 1.0 - long

    names = [
        "front_left_lower_wheel_zone",
        "front_right_lower_wheel_zone",
        "rear_left_lower_wheel_zone",
        "rear_right_lower_wheel_zone",
        "cabin_roof",
        "front_hood_upper",
        "rear_trunk_upper",
        "front_bumper",
        "rear_bumper",
        "left_side_body",
        "right_side_body",
        "center_body",
    ]
    labels = torch.full((vertices.shape[0],), -1, dtype=torch.long, device=vertices.device)

    lower = up <= 0.40
    left_outer = lat <= 0.32
    right_outer = lat >= 0.68
    front = long >= 0.66
    rear = long <= 0.34

    labels[lower & left_outer & front] = 0
    labels[lower & right_outer & front] = 1
    labels[lower & left_outer & rear] = 2
    labels[lower & right_outer & rear] = 3

    unassigned = labels < 0
    cabin = (up >= 0.58) & (long >= 0.28) & (long <= 0.78) & (lat >= 0.18) & (lat <= 0.82)
    labels[unassigned & cabin] = 4

    unassigned = labels < 0
    labels[unassigned & (up >= 0.42) & front] = 5

    unassigned = labels < 0
    labels[unassigned & (up >= 0.42) & rear] = 6

    unassigned = labels < 0
    labels[unassigned & (long >= 0.82)] = 7

    unassigned = labels < 0
    labels[unassigned & (long <= 0.18)] = 8

    unassigned = labels < 0
    labels[unassigned & (lat <= 0.22)] = 9

    unassigned = labels < 0
    labels[unassigned & (lat >= 0.78)] = 10

    labels[labels < 0] = 11
    return labels, names


def build_part_labels(
    vertices: torch.Tensor,
    schema: str = "car",
    longitudinal_axis: str = "z",
    lateral_axis: str = "x",
    vertical_axis: str = "y",
    flip_longitudinal: bool = False,
) -> Tuple[torch.Tensor, List[str]]:
    schema = schema.lower()
    if schema in {"car", "car_12"}:
        return _build_car_labels(
            vertices,
            longitudinal_axis=longitudinal_axis,
            lateral_axis=lateral_axis,
            vertical_axis=vertical_axis,
            flip_longitudinal=flip_longitudinal,
        )
    if schema in {"bbox", "bbox_3x2x2"}:
        return _build_bbox_labels(
            vertices,
            longitudinal_axis=longitudinal_axis,
            lateral_axis=lateral_axis,
            vertical_axis=vertical_axis,
            flip_longitudinal=flip_longitudinal,
        )
    raise ValueError(f"Unknown part-aware Chamfer schema '{schema}'.")


class PartAwareChamferLoss(nn.Module):
    """Symmetric Chamfer over fixed, matching source/target part buckets."""

    def __init__(
        self,
        source_reference_vertices: torch.Tensor,
        target_vertices: torch.Tensor,
        schema: str = "car",
        longitudinal_axis: str = "z",
        lateral_axis: str = "x",
        vertical_axis: str = "y",
        points_per_part: int = 512,
        min_points_per_part: int = 12,
        global_weight: float = 0.0,
        flip_target_longitudinal: bool = False,
    ) -> None:
        super().__init__()
        if source_reference_vertices.ndim != 2 or source_reference_vertices.shape[1] != 3:
            raise ValueError("source_reference_vertices must have shape (N, 3).")
        if target_vertices.ndim != 2 or target_vertices.shape[1] != 3:
            raise ValueError("target_vertices must have shape (M, 3).")

        source_labels, part_names = build_part_labels(
            source_reference_vertices,
            schema=schema,
            longitudinal_axis=longitudinal_axis,
            lateral_axis=lateral_axis,
            vertical_axis=vertical_axis,
        )
        target_labels, target_part_names = build_part_labels(
            target_vertices,
            schema=schema,
            longitudinal_axis=longitudinal_axis,
            lateral_axis=lateral_axis,
            vertical_axis=vertical_axis,
            flip_longitudinal=flip_target_longitudinal,
        )
        if part_names != target_part_names:
            raise RuntimeError("Source and target part schemas produced different part names.")

        self.schema = schema
        self.part_names = part_names
        self.points_per_part = int(points_per_part)
        self.min_points_per_part = int(min_points_per_part)
        self.global_weight = float(global_weight)

        self.register_buffer("source_labels", source_labels.long())
        self.register_buffer("target_labels", target_labels.long())
        self.register_buffer("target_vertices", unit_vertices(target_vertices.detach()))

        self.active_part_ids = self._compute_active_part_ids()

    def _compute_active_part_ids(self) -> List[int]:
        active = []
        for part_id in range(len(self.part_names)):
            src_count = int((self.source_labels == part_id).sum().item())
            tgt_count = int((self.target_labels == part_id).sum().item())
            if src_count >= self.min_points_per_part and tgt_count >= self.min_points_per_part:
                active.append(part_id)
        return active

    def part_counts(self) -> Dict[str, Dict[str, int]]:
        counts = {}
        for part_id, name in enumerate(self.part_names):
            counts[name] = {
                "source": int((self.source_labels == part_id).sum().item()),
                "target": int((self.target_labels == part_id).sum().item()),
                "active": part_id in self.active_part_ids,
            }
        return counts

    def forward(self, current_vertices: torch.Tensor, return_components: bool = False):
        source_vertices = unit_vertices(current_vertices)
        part_losses = []
        part_components = {}

        for part_id in self.active_part_ids:
            src_mask = self.source_labels == part_id
            tgt_mask = self.target_labels == part_id
            loss = _symmetric_chamfer(
                source_vertices[src_mask],
                self.target_vertices[tgt_mask],
                self.points_per_part,
            )
            part_losses.append(loss)
            part_components[self.part_names[part_id]] = loss.detach()

        if part_losses:
            part_raw = torch.stack(part_losses).mean()
        else:
            part_raw = current_vertices.sum() * 0.0

        global_raw = current_vertices.sum() * 0.0
        if self.global_weight > 0:
            global_raw = _symmetric_chamfer(source_vertices, self.target_vertices, self.points_per_part)

        raw = part_raw + self.global_weight * global_raw
        if not return_components:
            return raw

        return {
            "raw": raw,
            "part_raw": part_raw.detach(),
            "global_raw": global_raw.detach(),
            "parts": part_components,
            "active_parts": len(self.active_part_ids),
        }


def create_part_aware_chamfer_loss(
    source_reference_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    schema: str = "car",
    longitudinal_axis: str = "z",
    lateral_axis: str = "x",
    vertical_axis: str = "y",
    points_per_part: int = 512,
    min_points_per_part: int = 12,
    global_weight: float = 0.0,
    flip_target_longitudinal: bool = False,
) -> PartAwareChamferLoss:
    return PartAwareChamferLoss(
        source_reference_vertices=source_reference_vertices,
        target_vertices=target_vertices,
        schema=schema,
        longitudinal_axis=longitudinal_axis,
        lateral_axis=lateral_axis,
        vertical_axis=vertical_axis,
        points_per_part=points_per_part,
        min_points_per_part=min_points_per_part,
        global_weight=global_weight,
        flip_target_longitudinal=flip_target_longitudinal,
    )

