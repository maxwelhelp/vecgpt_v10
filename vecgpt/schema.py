"""ONE declaration of the token stream. Everything else is derived.

The sequence structure used to be written out by hand in five places -
`encode`, `decode`, `next_slot`, `build_slot_mask` and `SlotTracker`. Five
sources of truth for one fact, and both silent bugs in this project came
from exactly that: a slot index shifted by one in `generate` (cost a whole
training run, no test failed), and `seg_idx` for the EOL token differing
between `encode` and the tracker (the logit-parity test missed it because
one position's RoPE moved the softmax by less than its tolerance).

Here the grammar is data. Vocabulary layout, logit masks, transitions,
index counters, metric buckets and label smoothing are all computed from
it, so encode and decode cannot disagree - not "are tested not to
disagree", but cannot.

Adding a field, a token type, or a whole new construct (regions, clothoid
segments, text tokens) is an edit to this file only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# --------------------------------------------------------------- quantisers


@dataclass
class Quant:
    """A continuous quantity discretised into bins."""

    name: str
    n: int
    kind: str = "uniform"  # uniform | log | signed | circular
    lo: float = 0.0
    hi: float = 1.0

    def q(self, v: float) -> int:
        v = float(v)
        if self.kind == "uniform":
            t = (v - self.lo) / (self.hi - self.lo)
            return _clip(math.floor(t * self.n), self.n)
        if self.kind == "log":
            v = min(max(v, self.lo), self.hi)
            t = (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
            return _clip(round(t * (self.n - 1)), self.n)
        if self.kind == "signed":
            # odd n on purpose: an exactly-centred bin, so 0 round-trips to 0
            v = min(max(v, -self.hi), self.hi)
            return _clip(round((v + self.hi) / (2 * self.hi / (self.n - 1))), self.n)
        if self.kind == "circular":
            a = (v - self.lo) % (self.hi - self.lo)
            return _clip(math.floor(a / (self.hi - self.lo) * self.n), self.n)
        raise ValueError(self.kind)

    def d(self, i: int) -> float:
        if self.kind == "uniform":
            return self.lo + (i + 0.5) * (self.hi - self.lo) / self.n
        if self.kind == "log":
            lg = math.log(self.lo) + i * (math.log(self.hi) - math.log(self.lo)) / (self.n - 1)
            return math.exp(lg)
        if self.kind == "signed":
            return -self.hi + i * (2 * self.hi / (self.n - 1))
        if self.kind == "circular":
            return self.lo + (i + 0.5) * (self.hi - self.lo) / self.n
        raise ValueError(self.kind)

    @property
    def wraps(self) -> bool:
        return self.kind == "circular"


def _clip(i: int, n: int) -> int:
    return int(min(max(i, 0), n - 1))


# ------------------------------------------------------------------ fields
# Resolutions are measured choices, not guesses. Turning angle is quantised
# DIRECTLY rather than curvature: heading error accumulates in radians, and
# a corner arc is short, so constant-curvature bins gave a triangle corner
# ~9 deg of error and rotated the rest of the shape by up to 27. Length is
# log-spaced because it spans 0.002 (a corner) to 1.8 (a circumference) and
# constant *relative* precision is what matters there.

QUANTS = {
    # Anchors are an OFFSET from the region centre, in the region's frame.
    # Dividing by the ellipse axes as well (a "unit square inside the
    # region") looked tidier but ties every stroke's precision to the
    # frame's own ~4% axis rounding, and measured worse: stage-3 ceiling
    # 0.816 flat -> 0.776 scaled. Translate and rotate only, at a fixed
    # absolute resolution, keeps the frame useful without coupling.
    "x":     Quant("x", 257, "signed", 0.0, 0.62),          # 0.3 px at 64 px
    "y":     Quant("y", 257, "signed", 0.0, 0.62),
    "theta": Quant("theta", 256, "circular", -math.pi, math.pi),
    "len":   Quant("len", 256, "log", 0.002, 1.8),
    "turn":  Quant("turn", 513, "signed", 0.0, 2 * math.pi * 1.02),
    "width": Quant("width", 32, "uniform", 0.0, 0.10),
    "color": Quant("color", 16, "uniform", 0.0, 1.0),
    # region summary: an oriented ellipse from the second moments of the
    # strokes inside. A box is the wrong shape for a thin diagonal limb;
    # an ellipse costs one extra token and actually fits one.
    "rx":    Quant("rx", 256, "uniform", 0.0, 1.0),
    "ry":    Quant("ry", 256, "uniform", 0.0, 1.0),
    # NO axes. `to_local` only translates and rotates, so the ellipse's
    # size was never used for anything - it was pure leakage of the extent
    # of the very strokes that follow. Measured: `len` could be predicted
    # from `ra` ALONE, with no image, to 0.65 bins against a prior of 9.5.
    # Under teacher forcing the model read the answer out of its own
    # context instead of the picture, and colour - the one field nothing in
    # the sequence can supply - stayed dead flat at step 500.
    "rt":    Quant("rt", 32, "circular", -math.pi / 2, math.pi / 2),
}

SPECIALS = ["BOS", "EOS", "EOL", "STY", "ENDR"]


# ----------------------------------------------------------------- grammar
# A state says what may come next. `emit` is a field name (a Quant) or a
# special token name. `to` is the next state. Counter effects are declared
# here too, so the encoder and the incremental tracker step them the same
# way by construction.


@dataclass
class Edge:
    emit: str          # field name or special
    to: str | None     # next state, None ends the sequence
    bump: str = ""     # "" | "seg" | "stroke" | "region"
    stack: str = ""    # "" | "push_region" | "pop_region"


@dataclass
class State:
    name: str
    edges: list[Edge]


def _grammar() -> dict[str, State]:
    S = lambda n, *e: State(n, list(e))
    return {s.name: s for s in [
        # A frame is emitted ONLY when it groups two or more strokes.
        # A region over a single stroke groups nothing - it is that stroke
        # written in different numbers, which is exactly the leak above.
        # Ungrouped strokes run in the identity frame, which is a constant
        # and therefore tells the model nothing it has to be told.
        S("TOP",   Edge("EOS", None),
                   Edge("rx", "RY", stack="push_region"),
                   Edge("x", "Y")),
        S("RY",    Edge("ry", "RT")),
        S("RT",    Edge("rt", "HEAD")),
        # strokes, anchors are LOCAL to the region ellipse
        # HEAD is the body of a region. A child region is a genuinely
        # nested construct, not the next independent item in a flat list.
        S("HEAD",  Edge("ENDR", "TOP", stack="pop_region"),
                   Edge("rx", "RY", stack="push_region"),
                   Edge("x", "Y")),
        S("Y",     Edge("y", "TH")),
        S("TH",    Edge("theta", "SEG0")),
        # style is inherited; only a change costs tokens
        S("SEG0",  Edge("STY", "W"), Edge("len", "TURN")),
        S("TURN",  Edge("turn", "SEG", bump="seg")),
        S("SEG",   Edge("STY", "W"), Edge("EOL", "HEAD", bump="stroke"),
                   Edge("len", "TURN")),
        S("W",     Edge("width", "R")),
        S("R",     Edge("color", "G")),
        S("G",     Edge("color", "B")),
        S("B",     Edge("color", "SEG0")),
    ]}


GRAMMAR = _grammar()
STATE_NAMES = list(GRAMMAR)
STATE_ID = {n: i for i, n in enumerate(STATE_NAMES)}
N_STATES = len(STATE_NAMES)


# -------------------------------------------------------------- vocabulary


def _build_vocab():
    tok_of_special, ranges, cursor = {}, {}, 0
    for s in SPECIALS:
        tok_of_special[s] = cursor
        cursor += 1
    for name, q in QUANTS.items():
        ranges[name] = (cursor, cursor + q.n)
        cursor += q.n
    return tok_of_special, ranges, cursor


SPECIAL_TOK, RANGE, VOCAB = _build_vocab()
for _s, _t in SPECIAL_TOK.items():
    globals()[_s] = _t  # BOS, EOS, EOL, STY, ENDR as module constants


def encode_value(field: str, value: float) -> int:
    lo, _ = RANGE[field]
    return lo + QUANTS[field].q(value)


def decode_value(field: str, token: int) -> float:
    lo, _ = RANGE[field]
    return QUANTS[field].d(token - lo)


def field_of_token(token: int) -> str | None:
    for name, (lo, hi) in RANGE.items():
        if lo <= token < hi:
            return name
    return None


# ---------------------------------------------------- derived: masks, steps


def build_state_mask(device=None) -> torch.Tensor:
    """[N_STATES, VOCAB] bool - the legal continuations of each state."""
    m = torch.zeros(N_STATES, VOCAB, dtype=torch.bool, device=device)
    for name, st in GRAMMAR.items():
        for e in st.edges:
            if e.emit in SPECIAL_TOK:
                m[STATE_ID[name], SPECIAL_TOK[e.emit]] = True
            else:
                lo, hi = RANGE[e.emit]
                m[STATE_ID[name], lo:hi] = True
    return m


def step(state: str, token: int) -> tuple[str | None, str, str]:
    """(next state, counter bump, stack action).

    `pop_region` has a dynamic return state, resolved by Walker's region
    stack. Legality still comes entirely from this table.
    """
    st = GRAMMAR[state]
    for e in st.edges:
        if e.emit in SPECIAL_TOK:
            if token == SPECIAL_TOK[e.emit]:
                return e.to, e.bump, e.stack
        else:
            lo, hi = RANGE[e.emit]
            if lo <= token < hi:
                return e.to, e.bump, e.stack
    raise ValueError(f"token {token} is not legal in state {state}")


def build_smoothing(sigma_bins: float = 1.0, device=None) -> torch.Tensor:
    """Soft targets over neighbouring bins for ordinal fields.

    x / y / theta / len / turn / width / colour are positions on a line,
    not unrelated classes: predicting bin 41 instead of 40 should not cost
    the same as predicting bin 200. Specials stay one-hot; circular fields
    wrap.
    """
    M = torch.zeros(VOCAB, VOCAB, device=device)
    for t in SPECIAL_TOK.values():
        M[t, t] = 1.0
    for name, (lo, hi) in RANGE.items():
        q = QUANTS[name]
        n = hi - lo
        idx = torch.arange(n, device=device).float()
        for i in range(n):
            d = idx - i
            if q.wraps:
                d = (d + n / 2) % n - n / 2
            w = torch.exp(-0.5 * (d / sigma_bins) ** 2)
            M[lo + i, lo:hi] = w / w.sum()
    return M


# --------------------------------------------------------- metric bucketing
# Buckets are token ranges, not states: a state can carry two different
# things (TOP is "EOS or a new region", SEG is "another segment, a style
# change, or end of stroke") and averaging a trivial field with a hard one
# is what let a constant anchor look fine in an earlier version.

METRIC_FIELDS = list(SPECIALS[1:]) + [n for n in QUANTS if n != "color"] + ["r", "g", "b"]
N_METRIC = len(METRIC_FIELDS)
_METRIC_ID = {n: i for i, n in enumerate(METRIC_FIELDS)}


def metric_ranges() -> list[tuple[int, int]]:
    out = []
    for n in METRIC_FIELDS:
        if n in SPECIAL_TOK:
            out.append((SPECIAL_TOK[n], SPECIAL_TOK[n] + 1))
        elif n in ("r", "g", "b"):
            out.append(RANGE["color"])
        else:
            out.append(RANGE[n])
    return out


def metric_ids(tokens: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    f = torch.full_like(tokens, -1)
    for n in SPECIALS[1:]:
        f = torch.where(tokens == SPECIAL_TOK[n], torch.full_like(f, _METRIC_ID[n]), f)
    for n, (lo, hi) in RANGE.items():
        if n == "color":
            continue
        f = torch.where((tokens >= lo) & (tokens < hi), torch.full_like(f, _METRIC_ID[n]), f)
    clo, chi = RANGE["color"]
    is_col = (tokens >= clo) & (tokens < chi)
    for st, n in (("R", "r"), ("G", "g"), ("B", "b")):
        f = torch.where(is_col & (states == STATE_ID[st]), torch.full_like(f, _METRIC_ID[n]), f)
    return f
