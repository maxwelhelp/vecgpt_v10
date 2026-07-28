"""Regions discovered from geometry. No names, no taxonomy, no anatomy.

The previous attempt hard-coded an animal: body, head, eye, trunk, leg0..3.
The model never saw the names, but *I* fixed the structure, which is the
opposite of the point - concepts are supposed to fall out of training, not
be handed over as a schema.

Here a region is whatever a clustering of the strokes says it is. That
works on any scene: the synthetic stages, a compositional grammar, or real
SVG loaded later. It is the part of SAM's design that actually matters -
it segments without knowing what the things are.

Two consequences worth stating.

**The summary is an ellipse, not a box.** Second moments of the points
inside: centre, major axis, minor axis, angle. A box is the wrong shape for
a thin diagonal limb - it reports a large empty square. An ellipse costs
one more token and fits.

**Stroke anchors are LOCAL to their region.** This is the delta that
matters for generation with no image. Today an absolute anchor can only be
read off pixels, so with the picture removed there is nothing to condition
it on. Given a region, the anchor is an offset inside it, so the model can
place the region first and the strokes relative to it - a decision it can
make from its own prior instead of from a raster it will not have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from vecgpt.scene import Stroke, chain_points

IDENTITY = (0.5, 0.5, 0.0, 0.0, 0.0)  # emitted as nothing; leaks nothing


@dataclass
class Region:
    ellipse: tuple[float, float, float, float, float]  # cx, cy, a, b, theta
    strokes: list[Stroke] = field(default_factory=list)
    depth: int = 0
    explicit: bool = True  # False -> identity frame, not written to the stream
    children: list["Region"] = field(default_factory=list)


def stroke_points(st: Stroke, per_seg: int = 8) -> torch.Tensor:
    return chain_points(st.anchor, st.segs, per_seg).reshape(-1, 2)


def ellipse_of(points: torch.Tensor, min_axis: float = 0.012):
    """(cx, cy, a, b, theta) from the second moments of a point cloud."""
    mu = points.mean(0)
    d = points - mu
    if d.shape[0] < 3:
        return float(mu[0]), float(mu[1]), min_axis, min_axis, 0.0
    cov = (d.T @ d) / max(d.shape[0] - 1, 1)
    cov = cov + torch.eye(2, dtype=cov.dtype) * 1e-9
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    a = float(max(2.0 * evals[0].clamp_min(0).sqrt(), min_axis))
    b = float(max(2.0 * evals[1].clamp_min(0).sqrt(), min_axis))
    th = float(torch.atan2(evecs[1, 0], evecs[0, 0]))
    th = (th + math.pi / 2) % math.pi - math.pi / 2  # ellipse angle is mod pi
    return float(mu[0]), float(mu[1]), a, b, th


def to_local(x: float, y: float, ell) -> tuple[float, float]:
    """Absolute point -> offset in the region's rotated frame."""
    cx, cy, _, _, th = ell
    c, s = math.cos(-th), math.sin(-th)
    dx, dy = x - cx, y - cy
    return dx * c - dy * s, dx * s + dy * c


def from_local(u: float, v: float, ell) -> tuple[float, float]:
    cx, cy, _, _, th = ell
    c, s = math.cos(th), math.sin(th)
    return cx + u * c - v * s, cy + u * s + v * c


def local_theta(theta: float, ell) -> float:
    return (theta - ell[4] + math.pi) % (2 * math.pi) - math.pi


def global_theta(theta: float, ell) -> float:
    return (theta + ell[4] + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------- clustering


def _single_linkage(centres: torch.Tensor, thresh: float) -> list[list[int]]:
    n = centres.shape[0]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = torch.cdist(centres, centres)
    for i in range(n):
        for j in range(i + 1, n):
            if float(d[i, j]) <= thresh:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def build_regions(strokes: list[Stroke], thresh: float = 0.18,
                  max_depth: int = 2, min_split: int = 2) -> list[Region]:
    """Build a coarse-to-fine region forest with real ownership.

    Deterministic given the strokes, and ordered by raster position of the
    region centre, for the same reason stroke order is canonicalised: the
    target must be a function of the picture, not of iteration order.

    A previous implementation returned `[empty_parent, child1, child2, ...]`.
    Serialisation then closed every item independently, so the parent-child
    relationship disappeared and almost every stroke fell back to the
    identity frame. Here children remain nested and every stroke is owned
    exactly once by either a region or the implicit scene root.
    """
    if not strokes:
        return []
    pts = [stroke_points(s) for s in strokes]

    def raster_key(idxs: list[int]):
        p = torch.cat([pts[i] for i in idxs], 0).mean(0)
        return round(float(p[1]) * 64), round(float(p[0]) * 64)

    def make(idxs: list[int], depth: int, explicit_single: bool = False) -> Region:
        cloud = torch.cat([pts[i] for i in idxs], 0)
        ell = ellipse_of(cloud)
        if len(idxs) < 2:
            if explicit_single:
                # A one-stroke object still needs an object workspace. Only
                # centre/orientation are emitted (no axes/extent), so the
                # plan says WHERE to work without copying the stroke's size.
                return Region(ell, [strokes[i] for i in idxs], depth,
                              explicit=True)
            return Region(IDENTITY, [strokes[i] for i in idxs], depth,
                          explicit=False)
        if depth >= max_depth or len(idxs) < min_split:
            return Region(ell, [strokes[i] for i in idxs], depth)
        centres = torch.stack([pts[i].mean(0) for i in idxs])
        scale = float(max(ell[2], ell[3]))
        groups = _single_linkage(centres, thresh * scale)
        if len(groups) < 2:
            return Region(ell, [strokes[i] for i in idxs], depth)

        # Every discovered group becomes a child workspace, including a
        # single complex stroke. Otherwise multi-object scenes whose objects
        # are each represented by one stroke collapse back into one root region.
        out = Region(ell, [], depth)
        subs = [[idxs[k] for k in g] for g in groups]
        subs.sort(key=raster_key)
        for g in subs:
            out.children.append(make(g, depth + 1, explicit_single=True))
        return out

    return [make(list(range(len(strokes))), 0)]


def iter_regions(regions: list[Region]):
    """Depth-first traversal used by diagnostics and tests."""
    for reg in regions:
        yield reg
        yield from iter_regions(reg.children)
