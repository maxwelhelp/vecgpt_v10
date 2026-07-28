"""Differentiable teacher-forced decoding for the auxiliary raster loss.

The production representation remains a discrete autoregressive language.
During visual bootstrap only, this module turns the probability distribution
for every *numeric* token into an expected continuous parameter, while taking
the command/region structure from the target sequence.  Rendering the result
gives geometry a loss in the space where geometry matters: the final canvas.

This is deliberately auxiliary.  It does not run during generation and it
does not turn VecGPT into an image tracer.
"""

from __future__ import annotations

import math

import torch

from vecgpt import schema as S
from vecgpt.scene import Stroke
from vecgpt.tokenizer import Walker


def _bin_values(field: str, device, dtype) -> torch.Tensor:
    q = S.QUANTS[field]
    i = torch.arange(q.n, device=device, dtype=dtype)
    if q.kind == "uniform":
        return q.lo + (i + 0.5) * (q.hi - q.lo) / q.n
    if q.kind == "log":
        return torch.exp(
            math.log(q.lo)
            + i * (math.log(q.hi) - math.log(q.lo)) / (q.n - 1)
        )
    if q.kind == "signed":
        return -q.hi + i * (2 * q.hi / (q.n - 1))
    if q.kind == "circular":
        return q.lo + (i + 0.5) * (q.hi - q.lo) / q.n
    raise ValueError(q.kind)


def expected_value(logits: torch.Tensor, field: str,
                   temperature: float = 0.5,
                   straight_through: bool = True) -> torch.Tensor:
    """Differentiable decoded value, circular for wrapped quantities.

    The old forward pass used the distribution mean while production used
    argmax. It could therefore render a good mean-length line while the actual
    argmax token decoded to a dot. Straight-through argmax makes the rendered
    forward value identical to inference and uses softmax only as its backward
    gradient estimator.
    """
    lo, hi = S.RANGE[field]
    q = S.QUANTS[field]
    p = (logits[lo:hi] / temperature).softmax(-1)
    if straight_through:
        hard = torch.zeros_like(p)
        hard[p.argmax()] = 1.0
        p = hard + p - p.detach()
    values = _bin_values(field, logits.device, logits.dtype)
    if not q.wraps:
        return (p * values).sum()
    period = q.hi - q.lo
    phase = (values - q.lo) * (2 * math.pi / period)
    sy = (p * phase.sin()).sum()
    sx = (p * phase.cos()).sum()
    # At initialisation a circular distribution is almost uniform and its
    # resultant vector is ~0, where atan2 has an undefined/huge gradient.
    # A tiny detached reference direction keeps the auxiliary raster path
    # finite until the categorical CE has broken the symmetry.
    mean = torch.atan2(sy, sx + sx.new_tensor(1e-4))
    mean = mean.remainder(2 * math.pi)
    return q.lo + mean * (period / (2 * math.pi))


def soft_decode_batch(logits: torch.Tensor, tokens: torch.Tensor,
                      mask: torch.Tensor) -> list[list[Stroke]]:
    """Decode predicted numeric distributions under the target grammar.

    ``logits[:,t]`` predicts ``tokens[:,t+1]``.  Discrete decisions such as
    EOL/ENDR use the target during this auxiliary path; their own CE remains
    responsible for learning structure.
    """
    scenes: list[list[Stroke]] = []
    for b in range(tokens.shape[0]):
        n = int(mask[b].sum())
        walker = Walker()
        walker.advance(int(tokens[b, 0]))
        out: list[Stroke] = []
        # Token 0 is often illegal at this grammar position and therefore
        # carries -inf; multiplying that by zero would create NaN.
        zero = logits[b, 0].logsumexp(0) * 0.0
        frame = (zero + 0.5, zero + 0.5, zero)
        frame_stack = []
        ebuf: list[torch.Tensor] = []
        anchor: list[torch.Tensor] = []
        segs: list[torch.Tensor] = []
        style = [zero + 0.02, zero + 0.3, zero + 0.3, zero + 0.3]
        pending = zero

        def close():
            nonlocal anchor, segs
            if len(anchor) == 3 and segs:
                cx, cy, ft = frame
                c, s = ft.cos(), ft.sin()
                gx = cx + anchor[0] * c - anchor[1] * s
                gy = cy + anchor[0] * s + anchor[1] * c
                gth = anchor[2] + ft
                out.append(Stroke(
                    torch.stack((gx, gy, gth)),
                    torch.stack(segs),
                ))
            anchor, segs = [], []

        for t in range(1, n):
            tok = int(tokens[b, t])
            state = walker.state
            field = S.field_of_token(tok)
            value = (
                expected_value(logits[b, t - 1], field)
                if field is not None else None
            )

            if state in ("TOP", "HEAD") and field == "rx":
                frame_stack.append(frame)
                ebuf = [value]
            elif state == "RY" and field == "ry":
                ebuf.append(value)
            elif state == "RT" and field == "rt":
                frame = (ebuf[0], ebuf[1], value)
                ebuf = []
            elif state == "TOP" and field == "x":
                close()
                frame = (zero + 0.5, zero + 0.5, zero)
                anchor = [value]
            elif state == "HEAD" and field == "x":
                close()
                anchor = [value]
            elif state == "Y" and field == "y":
                anchor.append(value)
            elif state == "TH" and field == "theta":
                anchor.append(value)
            elif state in ("SEG0", "SEG") and field == "len":
                pending = value
            elif state == "TURN" and field == "turn":
                curvature = value / pending.clamp_min(1e-5)
                segs.append(torch.stack(
                    (pending, curvature, style[0], style[1], style[2], style[3])
                ))
            elif state == "W" and field == "width":
                style[0] = value
            elif state == "R" and field == "color":
                style[1] = value
            elif state == "G" and field == "color":
                style[2] = value
            elif state == "B" and field == "color":
                style[3] = value
            elif state in ("SEG0", "SEG") and tok == S.EOL:
                close()
            elif state == "HEAD" and tok == S.ENDR:
                close()
                frame = frame_stack.pop() if frame_stack else (
                    zero + 0.5, zero + 0.5, zero
                )

            walker.advance(tok)
            if walker.done:
                break
        close()
        scenes.append(out)
    return scenes
