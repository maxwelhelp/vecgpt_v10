"""Scenes from a recursive grammar. No animals, no part names, no taxonomy.

The previous generator hard-coded body / head / eye / trunk / leg0..3. The
model never saw the words, but the structure was mine, which defeats the
point: concepts are supposed to fall out of training, not be handed over.

Here a scene is a random tree. A node draws one shape and attaches children
at points on it. Nothing distinguishes "a head with eyes" from "a blob with
blobs" except the sampled parameters, and the model is never told which is
which. Regions are then recovered from the geometry by clustering
(`vecgpt.regions`), not read off this tree - so the same pipeline works on
scenes it did not generate, including real SVG later.

Generalisation is tested by holding out REGIONS OF THE PARAMETER SPACE
rather than named families: deeper trees, wider branching, size ratios
outside the training range. A model that memorised four generators cannot
pass those; a model that learned composition can.
"""

from __future__ import annotations

import math
import random

import torch

from vecgpt.scene import Stroke, arc_step, canonicalize

TRAIN = dict(depth=(1, 2), branch=(2, 3), ratio=(0.35, 0.6))
OOD = {
    "deeper":   dict(TRAIN, depth=(3, 3)),
    "wider":    dict(TRAIN, branch=(5, 6)),
    "tiny":     dict(TRAIN, ratio=(0.15, 0.22)),
    "chain":    dict(TRAIN, depth=(3, 4), branch=(1, 1)),
}


def _style(rng):
    return [rng.uniform(0.012, 0.03), rng.random() * 0.85, rng.random() * 0.85, rng.random() * 0.85]


def _loop(cx, cy, r, s, rng) -> Stroke:
    """A closed outline: k corners of shared turn. k large -> a circle."""
    k = rng.choice([1, 3, 4, 5, 6, 8])
    if k == 1:
        return Stroke(torch.tensor([cx, cy - r, 0.0]), torch.tensor([[2 * math.pi * r, 1 / r] + s]))
    side = 2 * r * math.sin(math.pi / k) * 0.7
    cr = r * 0.25
    turn = 2 * math.pi / k
    segs = []
    for _ in range(k):
        segs += [[side, 0.0] + s, [turn * cr, 1 / cr] + s]
    return Stroke(torch.tensor([cx - r * 0.6, cy - r, rng.uniform(-0.3, 0.3)]), torch.tensor(segs))


def _open(x, y, ang, length, s, rng) -> Stroke:
    n = rng.randint(1, 3)
    k = rng.uniform(-4, 4)
    segs = [[length / n, k * rng.uniform(0.6, 1.8)] + s for _ in range(n)]
    return Stroke(torch.tensor([x, y, ang]), torch.tensor(segs))


def _node(rng, cx, cy, r, depth, cfg, out):
    s = _style(rng)
    closed = rng.random() < 0.65
    if closed:
        out.append(_loop(cx, cy, r, s, rng))
    else:
        a = rng.uniform(-math.pi, math.pi)
        out.append(_open(cx - r * math.cos(a), cy - r * math.sin(a), a, r * 2.2, s, rng))
    if depth <= 0:
        return
    for _ in range(rng.randint(*cfg["branch"])):
        a = rng.uniform(-math.pi, math.pi)
        cr = r * rng.uniform(*cfg["ratio"])
        d = r + cr * rng.uniform(0.15, 0.75)
        _node(rng, cx + d * math.cos(a), cy + d * math.sin(a), cr, depth - 1, cfg, out)


def _extent(strokes):
    from vecgpt.scene import chain_points

    pts = torch.cat([chain_points(s.anchor, s.segs, 12).reshape(-1, 2) for s in strokes], 0)
    pad = float(max(s.segs[:, 2].max() for s in strokes)) / 2
    lo, hi = pts.amin(0) - pad, pts.amax(0) + pad
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def sample_tree(rng: random.Random, variant: str | None = None, tries: int = 40) -> list[Stroke]:
    cfg = OOD[variant] if variant else TRAIN
    for _ in range(tries):
        out: list[Stroke] = []
        r0 = rng.uniform(0.10, 0.17)
        _node(rng, 0.5, 0.5, r0, rng.randint(*cfg["depth"]), cfg, out)
        x0, y0, x1, y1 = _extent(out)
        w, h, m = x1 - x0, y1 - y0, 0.03
        if w > 1 - 2 * m or h > 1 - 2 * m or not out:
            continue
        dx = (m - x0) if x0 < m else ((1 - m) - x1 if x1 > 1 - m else 0.0)
        dy = (m - y0) if y0 < m else ((1 - m) - y1 if y1 > 1 - m else 0.0)
        moved = []
        for st in out:
            a = st.anchor.clone()
            a[0] += dx
            a[1] += dy
            moved.append(Stroke(a, st.segs.clone()))
        return canonicalize(moved)
    return canonicalize(out) if out else []
