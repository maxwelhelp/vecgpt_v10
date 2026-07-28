"""encode / decode / tracker, all walking the SAME grammar from schema.py.

There is no transition logic in this file. Every state change goes through
`schema.step`, so the encoder, the decoder and the incremental tracker used
by `generate` cannot drift apart - which is how both silent bugs in this
project happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vecgpt import schema as S
from vecgpt.regions import (IDENTITY, Region, build_regions, from_local,
                            global_theta, local_theta, to_local)
from vecgpt.scene import S_B, S_G, S_KAPPA, S_LEN, S_R, S_WIDTH, Stroke

VOCAB = S.VOCAB
N_STATES = S.N_STATES
METRIC_FIELDS = S.METRIC_FIELDS
N_METRIC = S.N_METRIC
BOS, EOS, EOL, STY, ENDR = S.BOS, S.EOS, S.EOL, S.STY, S.ENDR


@dataclass
class Encoded:
    tokens: torch.Tensor
    states: torch.Tensor
    seg_idx: torch.Tensor
    stroke_idx: torch.Tensor
    region_idx: torch.Tensor
    parent_region_idx: torch.Tensor
    region_depth: torch.Tensor


class Walker:
    """Grammar position plus the index counters, for ONE sequence.

    Used by encode (to label tokens) and by generate (to drive sampling).
    The counters advance only through `schema.step`, so a token is always
    labelled with the counter values that describe it, never the next
    token's - the off-by-one that a logit-parity test was too blunt to see.
    """

    __slots__ = ("state", "seg", "stroke", "region", "parent_region",
                 "depth", "next_region", "region_stack", "done")

    def __init__(self):
        self.state, self.seg, self.stroke = "TOP", 0, 0
        self.region, self.parent_region, self.depth = 0, 0, 0
        self.next_region = 1
        self.region_stack = []
        self.done = False

    def label(self):
        return (S.STATE_ID[self.state], self.seg, self.stroke, self.region,
                self.parent_region, self.depth)

    def advance(self, token: int):
        """Consume a token; returns the label that DESCRIBES it."""
        if token == BOS:
            return self.label()
        state_before = self.state
        nxt, bump, stack = S.step(self.state, token)

        if stack == "push_region":
            # The opening rx token belongs to the new region, even though its
            # grammar slot is TOP/HEAD in the parent.
            new_region = self.next_region
            self.next_region += 1
            new_depth = self.depth + 1
            lab = (S.STATE_ID[state_before], 0, 0, new_region,
                   self.region, new_depth)
            return_state = "TOP" if state_before == "TOP" else "HEAD"
            self.region_stack.append(
                (return_state, self.region, self.parent_region, self.depth,
                 self.seg, self.stroke)
            )
            self.parent_region = self.region
            self.region = new_region
            self.depth = new_depth
            self.seg = self.stroke = 0
        else:
            lab = self.label()

        if bump == "seg":
            self.seg += 1
        elif bump == "stroke":
            self.seg, self.stroke = 0, self.stroke + 1

        if stack == "pop_region":
            if self.region_stack:
                (nxt, self.region, self.parent_region, self.depth,
                 self.seg, self.stroke) = self.region_stack.pop()
            else:
                # Absolute top-level strokes use an implicit identity frame.
                nxt = "TOP"
                self.region = self.parent_region = self.depth = 0
                self.seg = self.stroke = 0

        if nxt is None:
            self.done = True
            nxt = "TOP"
        self.state = nxt
        return lab


def encode(strokes: list[Stroke], regions: list[Region] | None = None,
           hierarchical: bool = True) -> Encoded:
    if regions is None:
        regions = (build_regions(strokes) if hierarchical
                   else [Region(IDENTITY, list(strokes), depth=0,
                                explicit=False)])
    toks: list[int] = [BOS]
    labs: list[tuple] = []
    w = Walker()
    labs.append(w.advance(BOS))

    def put(tok: int):
        toks.append(tok)
        labs.append(w.advance(tok))

    style = None

    def put_stroke(st: Stroke, ell):
        nonlocal style
        u, v = to_local(float(st.anchor[0]), float(st.anchor[1]), ell)
        put(S.encode_value("x", u))
        put(S.encode_value("y", v))
        put(S.encode_value("theta", local_theta(float(st.anchor[2]), ell)))
        for s in range(st.segs.shape[0]):
            g = st.segs[s]
            sty = (S.QUANTS["width"].q(g[S_WIDTH]), S.QUANTS["color"].q(g[S_R]),
                   S.QUANTS["color"].q(g[S_G]), S.QUANTS["color"].q(g[S_B]))
            if sty != style:
                put(STY)
                put(S.RANGE["width"][0] + sty[0])
                for c in sty[1:]:
                    put(S.RANGE["color"][0] + c)
                style = sty
            L = float(g[S_LEN])
            put(S.encode_value("len", L))
            put(S.encode_value("turn", float(g[S_KAPPA]) * L))
        put(EOL)

    def item_key(item):
        kind, obj = item
        if kind == "stroke":
            x, y = float(obj.anchor[0]), float(obj.anchor[1])
        else:
            x, y = obj.ellipse[0], obj.ellipse[1]
        return round(y * 64), round(x * 64), 0 if kind == "region" else 1

    def put_region(reg: Region):
        # Quantise the region FIRST, then express strokes relative to the
        # quantised frame. Using the exact ellipse here and the decoded one
        # there leaks the frame's own rounding into every stroke inside it:
        # the angle bin alone is 2.8 deg, and it rotates all the local
        # coordinates. Measured, that cost stage-3 ceiling 0.816 -> 0.726.
        # Anchoring to the frame the decoder will actually see makes the
        # error cancel exactly.
        if reg.explicit:
            names = ("rx", "ry", "rt")
            vals = (reg.ellipse[0], reg.ellipse[1], reg.ellipse[4])
            etoks = [S.encode_value(f, v) for f, v in zip(names, vals)]
            ell = (S.decode_value("rx", etoks[0]), S.decode_value("ry", etoks[1]),
                   0.0, 0.0, S.decode_value("rt", etoks[2]))
            for t_ in etoks:
                put(t_)
        else:
            ell = IDENTITY

        items = ([("stroke", st) for st in reg.strokes]
                 + [("region", child) for child in reg.children])
        for kind, obj in sorted(items, key=item_key):
            if kind == "stroke":
                put_stroke(obj, ell)
            else:
                put_region(obj)
        put(ENDR)

    for reg in regions:
        put_region(reg)
    put(EOS)

    t = lambda k: torch.tensor([l[k] for l in labs], dtype=torch.long)
    return Encoded(torch.tensor(toks, dtype=torch.long), t(0), t(1), t(2),
                   t(3), t(4), t(5))


def decode(tokens) -> list[Stroke]:
    """Inverse of encode. Tolerant of truncated or malformed streams."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    out: list[Stroke] = []
    w = Walker()
    ell = IDENTITY
    ell_stack = []
    ebuf: list[float] = []
    anchor: list[float] = []
    segs: list[list[float]] = []
    style = [0.02, 0.3, 0.3, 0.3]
    pend = 0.0

    def close():
        if len(anchor) == 3 and segs:
            gx, gy = from_local(anchor[0], anchor[1], ell)
            out.append(Stroke(torch.tensor([gx, gy, global_theta(anchor[2], ell)]),
                              torch.tensor(segs, dtype=torch.float32)))

    for tok in tokens:
        if tok == BOS:
            continue
        st = w.state
        try:
            if st == "TOP" and tok == EOS:
                break
            f = S.field_of_token(tok)
            if st in ("TOP", "HEAD") and f == "rx":
                ell_stack.append(ell)
                ebuf = [S.decode_value("rx", tok)]
            elif st == "RY":
                ebuf.append(S.decode_value("ry", tok))
            elif st == "RT":
                ell = (ebuf[0], ebuf[1], 0.0, 0.0, S.decode_value("rt", tok))
                ebuf = []
            elif st == "TOP" and f == "x":
                ell = IDENTITY
                close()
                anchor, segs = [S.decode_value("x", tok)], []
            elif st == "HEAD" and f == "x":
                close()
                anchor, segs = [S.decode_value("x", tok)], []
            elif st == "Y":
                anchor.append(S.decode_value("y", tok))
            elif st == "TH":
                anchor.append(S.decode_value("theta", tok))
            elif st in ("SEG0", "SEG") and f == "len":
                pend = S.decode_value("len", tok)
            elif st == "TURN":
                segs.append([pend, S.decode_value("turn", tok) / max(pend, 1e-6)] + list(style))
            elif st == "W":
                style[0] = S.decode_value("width", tok)
            elif st in ("R", "G", "B"):
                style["RGB".index(st) + 1] = S.decode_value("color", tok)
            elif st in ("SEG0", "SEG") and tok == EOL:
                close()
                anchor, segs = [], []
            elif st == "HEAD" and tok == ENDR:
                close()
                anchor, segs = [], []
                ell = ell_stack.pop() if ell_stack else IDENTITY
        except (IndexError, ValueError, TypeError):
            break
        w.advance(tok)
        if w.done:
            break
    close()
    return out


def build_state_mask(device=None):
    return S.build_state_mask(device)


def build_smoothing_matrix(sigma_bins: float = 1.0, device=None):
    return S.build_smoothing(sigma_bins, device)


def metric_ranges():
    return S.metric_ranges()


def metric_field_ids(tokens, states):
    return S.metric_ids(tokens, states)
