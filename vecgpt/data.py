"""Scene generators and batching.

Same philosophy as before: no shape ever reaches the model as a NAME. A
circle is one arc whose curvature closes it; a polygon is straights plus
corner arcs. The generator only ever emits {length, curvature} numbers.

Two things changed:
  * every scene goes through `canonicalize` before tokenizing, so the
    target sequence is a deterministic function of the picture;
  * held-out shape families (star, spiral, cross, blob) are never shown
    during training - they exist so "did it learn shapes or memorise
    them" is an actual measurement, per the OOD gate in the design doc.
"""

from __future__ import annotations

import math
import random

import torch

from vecgpt.scene import Stroke, canonicalize, chain_points
from vecgpt.tokenizer import encode

STRAIGHT_PROB = 0.15


def _stroke(anchor, segs) -> Stroke:
    return Stroke(torch.tensor(anchor, dtype=torch.float32), torch.tensor(segs, dtype=torch.float32))


def _style(rng, thick=False):
    w = rng.uniform(0.03, 0.06) if thick else rng.uniform(0.012, 0.04)
    return [w, rng.random(), rng.random(), rng.random()]


# ------------------------------------------------------------ primitives


def gen_single(rng, thick=False, n_seg=(1, 1),
               straight_prob: float = STRAIGHT_PROB):
    s = _style(rng, thick)
    n = rng.randint(*n_seg)
    segs = []
    for _ in range(n):
        L = rng.uniform(0.10, 0.30)
        k = 0.0 if rng.random() < straight_prob else rng.uniform(-6, 6)
        segs.append([L, k] + s)
    return [_stroke((rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75), rng.uniform(-math.pi, math.pi)), segs)]


def gen_circle(rng):
    r = rng.uniform(0.08, 0.22)
    m = r + 0.04
    cx, cy = rng.uniform(m, 1 - m), rng.uniform(m, 1 - m)
    a = rng.uniform(-math.pi, math.pi)
    s = _style(rng)
    return [_stroke((cx + r * math.cos(a), cy + r * math.sin(a), a + math.pi / 2), [[2 * math.pi * r, 1 / r] + s])]


def gen_polygon(rng, n_sides=None):
    n = n_sides or rng.randint(3, 6)
    turn = 2 * math.pi / n
    cr = rng.uniform(0.02, 0.05)
    sl = rng.uniform(0.06, 0.16)
    s = _style(rng)
    segs = []
    for _ in range(n):
        segs += [[sl, 0.0] + s, [turn * cr, 1 / cr] + s]
    return [_stroke((rng.uniform(0.25, 0.7), rng.uniform(0.25, 0.7), rng.uniform(-math.pi, math.pi)), segs)]


def gen_zigzag(rng):
    s = _style(rng)
    cr = 0.02
    segs = [[rng.uniform(0.10, 0.22), 0.0] + s]
    for _ in range(rng.randint(1, 3)):
        turn = rng.uniform(math.pi / 6, math.pi * 0.7) * rng.choice([-1, 1])
        segs.append([abs(turn) * cr, math.copysign(1 / cr, turn)] + s)
        segs.append([rng.uniform(0.08, 0.18), 0.0] + s)
    return [_stroke((rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75), rng.uniform(-math.pi, math.pi)), segs)]


def gen_wave(rng):
    s = _style(rng)
    sign = rng.choice([-1, 1])
    segs = []
    for _ in range(rng.randint(3, 5)):
        segs.append([rng.uniform(0.08, 0.18), sign * rng.uniform(2, 9)] + s)
        sign *= -1
    return [_stroke((rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75), rng.uniform(-math.pi, math.pi)), segs)]


TRAIN_SHAPES = (gen_circle, gen_polygon, gen_zigzag, gen_wave)


def gen_shape(rng):
    return rng.choice(TRAIN_SHAPES)(rng)


def gen_multi(rng, lo=2, hi=3):
    out = []
    for _ in range(rng.randint(lo, hi)):
        out += gen_shape(rng)
    return out


# ------------------------------------------- held-out families (OOD gate)


def ood_star(rng):
    n = rng.choice([5, 7])
    s = _style(rng)
    cr = 0.018
    segs = []
    inner, outer = math.pi - math.pi * 2 / n, -(math.pi - math.pi * 0.6)
    for _ in range(n):
        for turn in (outer, inner):
            segs.append([rng.uniform(0.05, 0.09), 0.0] + s)
            segs.append([abs(turn) * cr, math.copysign(1 / cr, turn)] + s)
    return [_stroke((rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.6), rng.uniform(-math.pi, math.pi)), segs)]


def ood_spiral(rng):
    s = _style(rng)
    k = rng.uniform(6, 10)
    segs = []
    for _ in range(rng.randint(4, 6)):
        segs.append([rng.uniform(0.10, 0.16), k] + s)
        k *= 0.72
    return [_stroke((rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.6), rng.uniform(-math.pi, math.pi)), segs)]


def ood_cross(rng):
    s = _style(rng)
    L = rng.uniform(0.15, 0.25)
    cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
    a = rng.uniform(-math.pi, math.pi)
    out = []
    for da in (0.0, math.pi / 2):
        th = a + da
        out.append(_stroke((cx - L / 2 * math.cos(th), cy - L / 2 * math.sin(th), th), [[L, 0.0] + s]))
    return out


def ood_blob(rng):
    s = _style(rng)
    base = rng.uniform(5, 9)
    segs = []
    for _ in range(6):
        segs.append([rng.uniform(0.10, 0.20), base * rng.uniform(0.5, 1.6)] + s)
    return [_stroke((rng.uniform(0.3, 0.65), rng.uniform(0.3, 0.65), rng.uniform(-math.pi, math.pi)), segs)]


OOD_SHAPES = (ood_star, ood_spiral, ood_cross, ood_blob)


OOD_NAMES = ("star", "spiral", "cross", "blob")


def gen_ood(rng, tries: int = 24):
    for _ in range(tries):
        fitted = fit_to_canvas(rng.choice(OOD_SHAPES)(rng))
        if fitted is not None:
            return canonicalize(fitted)
    return canonicalize(fit_to_canvas(ood_cross(rng)) or ood_cross(rng))


def gen_ood_batch(families: list[str], n_per: int, seed: int = 1234) -> dict[str, list[Stroke]]:
    """Generate OOD scenes grouped by family. Returns {name: [scenes]}."""
    result = {}
    for name in families:
        if name not in dict(zip(OOD_NAMES, OOD_SHAPES)):
            raise ValueError(f"unknown OOD family: {name}")
    rng = random.Random(seed)
    for name in families:
        fam = dict(zip(OOD_NAMES, OOD_SHAPES))[name]
        scenes = []
        for _ in range(n_per):
            rng2 = random.Random(rng.randint(0, 2 ** 31 - 1))
            fitted = fit_to_canvas(fam(rng2))
            if fitted is not None:
                scenes.append(canonicalize(fitted))
            else:
                scenes.append(
                    canonicalize(fit_to_canvas(ood_cross(rng2)) or ood_cross(rng2))
                )
        result[name] = scenes
    return result


# ------------------------------------------------------------- curriculum

from vecgpt.grammar import OOD as GRAMMAR_OOD
from vecgpt.grammar import sample_tree

STAGES = {
    # Honest primitive gate: varying position, direction, length, width and
    # RGB, but no curvature yet. The former "stage 1" was 85% random arcs,
    # so failure to infer turn was misreported as failure on simple lines.
    0: lambda r: gen_single(
        r, thick=True, n_seg=(1, 1), straight_prob=1.0
    ),
    1: lambda r: gen_single(r, thick=True, n_seg=(1, 1)),
    2: lambda r: gen_single(r, thick=False, n_seg=(1, 2)),
    3: gen_shape,
    4: gen_multi,
    5: sample_tree,   # recursive compositional trees, no fixed anatomy
}


def scene_bbox(strokes: list[Stroke]) -> tuple[float, float, float, float]:
    pts = torch.cat([chain_points(s.anchor, s.segs, 16).reshape(-1, 2) for s in strokes], 0)
    pad = max(float(max(s.segs[:, 2].max() for s in strokes)) / 2, 0.0)
    lo, hi = pts.amin(0), pts.amax(0)
    return float(lo[0]) - pad, float(lo[1]) - pad, float(hi[0]) + pad, float(hi[1]) + pad


def fit_to_canvas(strokes: list[Stroke], margin: float = 0.03) -> list[Stroke] | None:
    """Translate a scene so it lies inside [margin, 1-margin]^2, or reject.

    This is data hygiene, not cosmetics. Measured: a wave that wandered off
    the canvas had its (canonical) start point at x=1.19, which the x token
    grid cannot even represent - it clamps to 0.996, and reconstructing
    that scene from its OWN tokens scored IoU 0.26. Asking a model to
    predict an anchor that is off-screen and unrepresentable is asking it
    to learn noise.
    """
    x0, y0, x1, y1 = scene_bbox(strokes)
    if x1 - x0 > 1 - 2 * margin or y1 - y0 > 1 - 2 * margin:
        return None
    # Shift by ZERO if the scene already fits; otherwise by the smallest
    # amount that brings the offending side inside. The earlier version of
    # this clamped to `margin - x0` unconditionally, which pinned every
    # scene's bounding box to the top-left corner and destroyed almost all
    # of the anchor's positional variance - the y field's marginal entropy
    # collapsed to ~0 and its "information gain" read 0.01 nats while the
    # model was in fact predicting a near-constant. A data bug that shows
    # up as a suspiciously perfect metric.
    def shift(lo, hi):
        if lo < margin:
            return margin - lo
        if hi > 1 - margin:
            return (1 - margin) - hi
        return 0.0

    dx, dy = shift(x0, x1), shift(y0, y1)
    out = []
    for s in strokes:
        a = s.anchor.clone()
        a[0] += dx
        a[1] += dy
        out.append(Stroke(a, s.segs.clone()))
    return out


def sample_scene(rng: random.Random, stage: int, tries: int = 24) -> list[Stroke]:
    for _ in range(tries):
        fitted = fit_to_canvas(STAGES[stage](rng))
        if fitted is not None:
            return canonicalize(fitted)
    # last resort: a scene guaranteed to fit
    return canonicalize(fit_to_canvas(gen_single(rng, thick=True)) or gen_single(rng, thick=True))


class SceneStream:
    """Infinite generator. Scenes are cheap; rendering is done batched on
    the GPU by the training loop, not here."""

    def __init__(self, stage: int = 1, seed: int = 0):
        self.stage = stage
        self.rng = random.Random(seed)

    def batch(self, n: int):
        return [sample_scene(self.rng, self.stage) for _ in range(n)]


class CachedStream:
    """Returns (scenes, images) from pre-rendered cache (build_cache.py).

    Eliminates render_batch from the training loop entirely.
    Accepts `skip` to resume training from a checkpoint.
    """

    def __init__(self, stage: int, cache_dir: str = "cache", device=None,
                 skip_batches: int = 0, batch_size: int = 32):
        import torch

        path = f"{cache_dir}/stage{stage}.pt"
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.imgs = data["imgs"]  # [N, H, W, 3] uint8
        self.scenes = data["scenes"]  # list of list[Stroke]
        self.n = len(self.scenes)
        self.pos = (skip_batches * batch_size) % self.n
        self.device = device

    def batch(self, n: int):
        end = min(self.pos + n, self.n)
        if end <= self.pos:
            self.pos = 0
            end = n
        scenes = self.scenes[self.pos:end]
        imgs = self.imgs[self.pos:end].float() / 255.0
        if self.device is not None:
            imgs = imgs.to(self.device)
        self.pos = end
        if self.pos >= self.n:
            self.pos = 0
        return scenes, imgs


def collate(scenes: list[list[Stroke]], device=None, hierarchical: bool = True):
    """-> dict of padded [B, T] tensors + key_padding mask."""
    encs = [encode(s, hierarchical=hierarchical) for s in scenes]
    B, T = len(encs), max(e.tokens.numel() for e in encs)
    z = lambda: torch.zeros(B, T, dtype=torch.long)
    tokens, slots, segi, stri = z(), z(), z(), z()
    regi, pari, depth = z(), z(), z()
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, e in enumerate(encs):
        n = e.tokens.numel()
        tokens[i, :n], slots[i, :n] = e.tokens, e.states
        segi[i, :n], stri[i, :n] = e.seg_idx, e.stroke_idx
        regi[i, :n], pari[i, :n] = e.region_idx, e.parent_region_idx
        depth[i, :n] = e.region_depth
        mask[i, :n] = True
    out = dict(tokens=tokens, slots=slots, seg_idx=segi, stroke_idx=stri,
               region_idx=regi, parent_region_idx=pari, region_depth=depth,
               mask=mask)
    return {k: v.to(device) for k, v in out.items()} if device else out
