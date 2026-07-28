"""Scene representation + the geometry that makes targets UNAMBIGUOUS.

A Stroke = anchor (x0, y0, theta0) + an ordered chain of constant-curvature
arcs, each with its own style. Same primitive as before - that part of the
old design was fine.

What is new here is `canonicalize`. The old code trained a regression head
against whatever order the generator happened to emit strokes in, and
whichever end it happened to call "the start". Neither is recoverable from
the image, so the conditional target distribution was multi-modal and any
L2/Huber head converges to the *mean over modes* - which for "starts at A
or starts at B" is the midpoint, and for "curves +k or -k" is a straight
line. That is a property of the loss, not something more training steps
fix.

So we remove the ambiguity from the DATA instead: every scene is rewritten
into exactly one canonical form (fixed traversal direction per stroke,
fixed raster order across strokes) before it is ever tokenized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# segment layout: [length, curvature, width, r, g, b]
SEG_DIM = 6
S_LEN, S_KAPPA, S_WIDTH, S_R, S_G, S_B = range(SEG_DIM)


@dataclass
class Stroke:
    anchor: torch.Tensor  # [3] = (x0, y0, theta0)
    segs: torch.Tensor  # [S, 6]

    def clone(self) -> "Stroke":
        return Stroke(self.anchor.clone(), self.segs.clone())


def arc_step(
    x: torch.Tensor, y: torch.Tensor, th: torch.Tensor, kappa: torch.Tensor, s: torch.Tensor
):
    """Position/heading after arc-length s on a constant-curvature arc.

    sinc form, so k=0 is smooth in value *and* gradient without a branch
    (this was already right in the old code - kept verbatim).
    """
    half = kappa * s / 2
    radial = s * torch.sinc(half / math.pi)
    return (
        x + radial * torch.cos(th + half),
        y + radial * torch.sin(th + half),
        th + kappa * s,
    )


def segment_starts(anchor: torch.Tensor, segs: torch.Tensor) -> torch.Tensor:
    """[S, 3] state (x, y, theta) at the start of each segment."""
    x, y, th = anchor[0], anchor[1], anchor[2]
    out = []
    for i in range(segs.shape[0]):
        out.append(torch.stack([x, y, th]))
        x, y, th = arc_step(x, y, th, segs[i, S_KAPPA], segs[i, S_LEN])
    return torch.stack(out)


def canonical_phase(st: "Stroke", samples: int = 48) -> "Stroke":
    """Re-start a CLOSED stroke at its raster-first outline point.

    Measured, not assumed: the same circle on screen, generated with 64
    different start angles, produced canonical anchor x tokens spanning 39
    bins (r=0.15), so the best possible single guess still carried mae 12
    bins ~ 6 px. Half of stage-4 strokes are closed, and the model's x mae
    tracked the closed fraction almost exactly: 0% -> 1 bin, 44% -> 7,
    51% -> 10. The network was being asked to guess a coin flip, and since
    the anchor is the FIRST token of a stroke, a blurred anchor displaces
    everything drawn after it.

    `canonicalize` fixed the traversal DIRECTION of closed strokes but never
    their starting POINT. The old invariance test only shuffled and reversed
    strokes, so it could not see this.

    Here the start is defined by the geometry: the outline point that is
    first in raster order (min y, then min x), splitting the arc it lands
    in. For a circle that is the top of the circle, whose x is the centre's
    x - a very well determined quantity - so this also removes an
    ill-conditioned target rather than merely a random one.
    """
    S = st.segs.shape[0]
    pts = chain_points(st.anchor, st.segs, samples)  # [S, samples+1, 2]
    p = pts[:, :samples]  # drop each segment's duplicate endpoint
    key = p[..., 1] * 1e4 + p[..., 0]  # y major, x minor
    flat = int(key.reshape(-1).argmin())
    i, j = flat // samples, flat % samples
    starts = segment_starts(st.anchor, st.segs)
    L = st.segs[i, S_LEN]

    # local refine: the coarse grid leaves ~1 bin of jitter, which is enough
    # to move the anchor token and cost real IoU on a 1-3 px stroke
    lo, hi = (j - 1) / samples, (j + 1) / samples
    ts = torch.linspace(max(lo, 0.0), min(hi, 1.0), 33, device=st.segs.device, dtype=st.segs.dtype)
    px, py, _ = arc_step(starts[i, 0], starts[i, 1], starts[i, 2], st.segs[i, S_KAPPA], ts * L)
    j_fine = int((py * 1e4 + px).argmin())
    s = L * ts[j_fine]
    if i == 0 and float(s) < 1e-6:
        return st
    x, y, th = arc_step(starts[i, 0], starts[i, 1], starts[i, 2], st.segs[i, S_KAPPA], s)
    new_anchor = torch.stack([x, y, th])

    head = st.segs[i].clone()
    head[S_LEN] = L - s
    parts = [head[None], st.segs[i + 1 :], st.segs[:i]]
    if float(s) > 1e-6:
        tail = st.segs[i].clone()
        tail[S_LEN] = s
        parts.append(tail[None])
    return Stroke(new_anchor, torch.cat([q for q in parts if q.shape[0]], 0))


def chain_points(anchor: torch.Tensor, segs: torch.Tensor, per_seg: int = 12) -> torch.Tensor:
    """[S, per_seg+1, 2] - one point chain per segment, chained end-to-end.

    Fully vectorised over segments: the per-segment start states are the
    cumulative arc-chain, computed with a small sequential scan over S
    (S is ~1-30, negligible) but every *sample* inside a segment at once.
    """
    S = segs.shape[0]
    x, y, th = anchor[0], anchor[1], anchor[2]
    starts = []
    for i in range(S):
        starts.append(torch.stack([x, y, th]))
        x, y, th = arc_step(x, y, th, segs[i, S_KAPPA], segs[i, S_LEN])
    st = torch.stack(starts)  # [S, 3]

    t = torch.linspace(0.0, 1.0, per_seg + 1, device=segs.device, dtype=segs.dtype)
    s = t[None, :] * segs[:, S_LEN][:, None]  # [S, per_seg+1]
    px, py, _ = arc_step(
        st[:, 0][:, None], st[:, 1][:, None], st[:, 2][:, None], segs[:, S_KAPPA][:, None], s
    )
    return torch.stack([px, py], dim=-1)


def end_state(anchor: torch.Tensor, segs: torch.Tensor) -> torch.Tensor:
    x, y, th = anchor[0], anchor[1], anchor[2]
    for i in range(segs.shape[0]):
        x, y, th = arc_step(x, y, th, segs[i, S_KAPPA], segs[i, S_LEN])
    return torch.stack([x, y, th])


def reverse_stroke(st: Stroke) -> Stroke:
    """Exactly the same drawn curve, traversed the other way.

    Backwards along one arc: start at the end point heading theta_end + pi,
    and the total turn must be -(kappa*L), so curvature negates and length
    is unchanged. Reverse the segment order and you have the identical
    geometry with a different (equally valid) parameterisation - which is
    precisely the ambiguity we are removing.
    """
    e = end_state(st.anchor, st.segs)
    new_anchor = torch.stack([e[0], e[1], wrap_pi(e[2] + math.pi)])
    new_segs = st.segs.flip(0).clone()
    new_segs[:, S_KAPPA] = -new_segs[:, S_KAPPA]
    return Stroke(new_anchor, new_segs)


def wrap_pi(a: torch.Tensor | float):
    if isinstance(a, torch.Tensor):
        return (a + math.pi) % (2 * math.pi) - math.pi
    return (a + math.pi) % (2 * math.pi) - math.pi


def _is_closed(st: Stroke, tol: float = 0.02) -> bool:
    e = end_state(st.anchor, st.segs)
    return bool(torch.linalg.norm(e[:2] - st.anchor[:2]) < tol)


def canonicalize(strokes: list[Stroke], grid: int = 64) -> list[Stroke]:
    """One scene -> exactly one token sequence.

    1. traversal direction: keep the orientation whose START point is
       first in raster order (y, then x). For a *closed* stroke both ends
       coincide, so instead we fix the sign of total turning (positive),
       which is the analogous discrete choice there.
    2. stroke order: raster order of the (now fixed) start points.

    Quantising to the same grid the tokenizer uses avoids ties flipping on
    float noise between otherwise identical scenes.
    """
    out = []
    for st in strokes:
        if _is_closed(st):
            total_turn = float((st.segs[:, S_KAPPA] * st.segs[:, S_LEN]).sum())
            cand = st if total_turn >= 0 else reverse_stroke(st)
            cand = canonical_phase(cand)  # start point, not just direction
        else:
            rev = reverse_stroke(st)
            a = (round(float(st.anchor[1]) * grid), round(float(st.anchor[0]) * grid))
            b = (round(float(rev.anchor[1]) * grid), round(float(rev.anchor[0]) * grid))
            cand = st if a <= b else rev
        out.append(cand)

    out.sort(
        key=lambda s: (
            round(float(s.anchor[1]) * grid),
            round(float(s.anchor[0]) * grid),
            round(float(s.anchor[2]) * grid),
        )
    )
    return out
