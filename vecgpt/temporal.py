"""Persistent vector-region state and delta application for animation.

This module contains no semantic inventory and no raster frame cache.
Region ids are sequence-local pointers allocated by a generated program.
Cached payloads are compiled *local vector geometry*, so a parent transform
can move a whole subtree without rebuilding every descendant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

import torch

from vecgpt.scene import Stroke


@dataclass
class RegionState:
    """One persistent node in a frame's variable region tree."""

    region_id: int
    parent_id: int | None
    transform: torch.Tensor  # local (x, y, theta), [3]
    strokes: list[Stroke] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    revision: int = 0

    def clone(self) -> "RegionState":
        return RegionState(
            self.region_id,
            self.parent_id,
            self.transform.clone(),
            [stroke.clone() for stroke in self.strokes],
            list(self.children),
            self.revision,
        )


@dataclass
class VectorFrame:
    """Full declarative vector state at one time, indexed by stable ids."""

    regions: dict[int, RegionState]
    roots: list[int]
    time: float = 0.0

    def clone(self) -> "VectorFrame":
        return VectorFrame(
            {rid: region.clone() for rid, region in self.regions.items()},
            list(self.roots),
            self.time,
        )

    def validate(self) -> None:
        if len(self.regions) != len(set(self.regions)):
            raise ValueError("duplicate region id")
        for rid, region in self.regions.items():
            if rid != region.region_id:
                raise ValueError(f"region key/id mismatch: {rid}")
            if region.parent_id is not None and region.parent_id not in self.regions:
                raise ValueError(f"region {rid} has missing parent {region.parent_id}")
            for child in region.children:
                if child not in self.regions:
                    raise ValueError(f"region {rid} has missing child {child}")
                if self.regions[child].parent_id != rid:
                    raise ValueError(f"parent/child mismatch for {child}")
        if any(rid not in self.regions for rid in self.roots):
            raise ValueError("missing root")


@dataclass
class FrameDelta:
    """Sparse update program; absent ids mean KEEP."""

    upsert: dict[int, RegionState] = field(default_factory=dict)
    delete: set[int] = field(default_factory=set)
    roots: list[int] | None = None
    time: float = 0.0


def _stroke_equal(a: Stroke, b: Stroke) -> bool:
    return torch.equal(a.anchor, b.anchor) and torch.equal(a.segs, b.segs)


def local_geometry_equal(a: RegionState, b: RegionState) -> bool:
    """Equality of cacheable local geometry, excluding parent transforms."""
    return (
        a.children == b.children
        and len(a.strokes) == len(b.strokes)
        and all(_stroke_equal(x, y) for x, y in zip(a.strokes, b.strokes))
    )


def region_equal(a: RegionState, b: RegionState) -> bool:
    return (
        a.parent_id == b.parent_id
        and torch.equal(a.transform, b.transform)
        and local_geometry_equal(a, b)
    )


def diff_frames(previous: VectorFrame, current: VectorFrame) -> FrameDelta:
    """Build the exact sparse delta between two persistent vector frames."""
    previous.validate()
    current.validate()
    upsert = {
        rid: region.clone()
        for rid, region in current.regions.items()
        if rid not in previous.regions
        or not region_equal(previous.regions[rid], region)
    }
    delete = set(previous.regions) - set(current.regions)
    roots = list(current.roots) if previous.roots != current.roots else None
    return FrameDelta(upsert, delete, roots, current.time)


def apply_delta(previous: VectorFrame, delta: FrameDelta) -> VectorFrame:
    """Apply ADD/EDIT/DELETE while structurally sharing unchanged content."""
    regions = dict(previous.regions)
    for rid in delta.delete:
        regions.pop(rid, None)
    for rid, region in delta.upsert.items():
        regions[rid] = region.clone()
    out = VectorFrame(
        regions,
        list(previous.roots if delta.roots is None else delta.roots),
        delta.time,
    )
    out.validate()
    return out


Payload = TypeVar("Payload")


class LocalVectorCache(Generic[Payload]):
    """Cache renderer-specific compiled paths, never completed raster frames."""

    def __init__(self):
        self._entries: dict[int, tuple[RegionState, Payload]] = {}

    def sync(
        self,
        frame: VectorFrame,
        compile_local: Callable[[RegionState], Payload],
    ) -> tuple[dict[int, Payload], set[int]]:
        """Return payloads and ids whose local vector geometry was rebuilt.

        Transform-only edits reuse the old payload. The compositor applies the
        new transform matrix when traversing the tree.
        """
        frame.validate()
        rebuilt: set[int] = set()
        for rid in set(self._entries) - set(frame.regions):
            del self._entries[rid]
        for rid, region in frame.regions.items():
            old = self._entries.get(rid)
            if old is None or not local_geometry_equal(old[0], region):
                self._entries[rid] = (region.clone(), compile_local(region))
                rebuilt.add(rid)
            else:
                # Keep cached geometry but remember the newest transform and
                # hierarchy metadata for the next comparison.
                self._entries[rid] = (region.clone(), old[1])
        return {rid: item[1] for rid, item in self._entries.items()}, rebuilt

