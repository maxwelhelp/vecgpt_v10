"""Stateful Stroke representation for clothoid geometry and sparse deltas."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vecgpt.clothoid import clothoid_end_state
from vecgpt.clothoid_render import render_clothoid_batch
from vecgpt.scene import S_B, S_G, S_KAPPA, S_LEN, S_R, S_WIDTH, Stroke


@dataclass
class StatefulStroke:
    """One Stroke with continuous curvature and sparse state changes.

    ``curvature_jump[i]`` is applied immediately before segment ``i``.
    ``delta_kappa[i]`` changes curvature continuously *during* that segment.
    The first jump is normally zero because ``base_kappa`` defines the
    initial state.
    """

    anchor: torch.Tensor          # [3] = x,y,theta
    base_kappa: torch.Tensor      # scalar
    length: torch.Tensor          # [S]
    delta_kappa: torch.Tensor     # [S]
    curvature_jump: torch.Tensor  # [S], sparse boundary discontinuity
    base_style: torch.Tensor      # [5] = width,r,g,b,alpha
    style_delta: torch.Tensor     # [S,5], applied before segment i

    @property
    def n_segments(self) -> int:
        return int(self.length.shape[0])

    def styles(self) -> torch.Tensor:
        return self.base_style[None] + self.style_delta.cumsum(0)

    def clone(self) -> "StatefulStroke":
        return StatefulStroke(
            self.anchor.clone(), self.base_kappa.clone(),
            self.length.clone(), self.delta_kappa.clone(),
            self.curvature_jump.clone(), self.base_style.clone(),
            self.style_delta.clone(),
        )


@dataclass
class PackedStatefulStrokes:
    anchor: torch.Tensor           # [B,K,4] = x,y,sin(theta),cos(theta)
    base_kappa: torch.Tensor       # [B,K]
    base_style: torch.Tensor       # [B,K,5]
    segment: torch.Tensor          # [B,K,S,2] = length,delta_kappa
    curvature_jump: torch.Tensor   # [B,K,S]
    curvature_change: torch.Tensor # [B,K,S] bool
    style_delta: torch.Tensor      # [B,K,S,5]
    style_change: torch.Tensor     # [B,K,S] bool
    stroke_mask: torch.Tensor      # [B,K]
    segment_mask: torch.Tensor     # [B,K,S]
    counts: torch.Tensor           # [B,K]


def legacy_to_stateful(stroke: Stroke) -> StatefulStroke:
    """Exactly convert piecewise constant-curvature legacy geometry."""
    anchor, g = stroke.anchor, stroke.segs
    n = int(g.shape[0])
    if n < 1:
        raise ValueError("Stroke must contain at least one segment")
    alpha = (
        g[:, S_B + 1] if g.shape[1] > S_B + 1
        else torch.ones_like(g[:, S_LEN])
    )
    styles = torch.stack((
        g[:, S_WIDTH], g[:, S_R], g[:, S_G], g[:, S_B], alpha,
    ), -1)
    style_delta = torch.zeros_like(styles)
    if n > 1:
        style_delta[1:] = styles[1:] - styles[:-1]
    jumps = torch.zeros_like(g[:, S_KAPPA])
    if n > 1:
        jumps[1:] = g[1:, S_KAPPA] - g[:-1, S_KAPPA]
    return StatefulStroke(
        anchor=anchor,
        base_kappa=g[0, S_KAPPA],
        length=g[:, S_LEN],
        delta_kappa=torch.zeros_like(g[:, S_KAPPA]),
        curvature_jump=jumps,
        base_style=styles[0],
        style_delta=style_delta,
    )


def pack_stateful_scenes(
    scenes: list[list[StatefulStroke]],
    max_strokes: int,
    max_segments: int,
    device=None,
) -> PackedStatefulStrokes:
    B = len(scenes)
    anchor = torch.zeros(B, max_strokes, 4, device=device)
    base_kappa = torch.zeros(B, max_strokes, device=device)
    base_style = torch.zeros(B, max_strokes, 5, device=device)
    segment = torch.zeros(B, max_strokes, max_segments, 2, device=device)
    curvature_jump = torch.zeros(
        B, max_strokes, max_segments, device=device
    )
    curvature_change = torch.zeros(
        B, max_strokes, max_segments, dtype=torch.bool, device=device
    )
    style_delta = torch.zeros(
        B, max_strokes, max_segments, 5, device=device
    )
    style_change = torch.zeros(
        B, max_strokes, max_segments, dtype=torch.bool, device=device
    )
    stroke_mask = torch.zeros(
        B, max_strokes, dtype=torch.bool, device=device
    )
    segment_mask = torch.zeros(
        B, max_strokes, max_segments, dtype=torch.bool, device=device
    )
    counts = torch.zeros(B, max_strokes, dtype=torch.long, device=device)

    for b, strokes in enumerate(scenes):
        if len(strokes) > max_strokes:
            raise ValueError("scene exceeds max_strokes context")
        for k, stroke in enumerate(strokes):
            n = stroke.n_segments
            if n < 1 or n > max_segments:
                raise ValueError("stroke exceeds max_segments context")
            a = stroke.anchor.to(device)
            anchor[b, k] = torch.stack(
                (a[0], a[1], a[2].sin(), a[2].cos())
            )
            base_kappa[b, k] = stroke.base_kappa.to(device)
            base_style[b, k] = stroke.base_style.to(device)
            segment[b, k, :n, 0] = stroke.length.to(device)
            segment[b, k, :n, 1] = stroke.delta_kappa.to(device)
            curvature_jump[b, k, :n] = stroke.curvature_jump.to(device)
            curvature_change[b, k, :n] = (
                stroke.curvature_jump.to(device).abs() > 1e-6
            )
            style_delta[b, k, :n] = stroke.style_delta.to(device)
            style_change[b, k, :n] = (
                stroke.style_delta.to(device).abs().amax(-1) > 1e-6
            )
            stroke_mask[b, k] = True
            segment_mask[b, k, :n] = True
            counts[b, k] = n
    return PackedStatefulStrokes(
        anchor, base_kappa, base_style, segment,
        curvature_jump, curvature_change,
        style_delta, style_change,
        stroke_mask, segment_mask, counts,
    )


def stateful_chain_states(
    anchor: torch.Tensor,
    base_kappa: torch.Tensor,
    segment: torch.Tensor,
    curvature_jump: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return start anchors/kappas and end states for every segment."""
    theta = torch.atan2(anchor[..., 2], anchor[..., 3])
    state = torch.cat((anchor[..., :2], theta[..., None]), -1)
    kappa = base_kappa
    starts, start_kappas, ends = [], [], []
    for i in range(segment.shape[-2]):
        kappa = kappa + curvature_jump[..., i]
        starts.append(state)
        start_kappas.append(kappa)
        state, kappa = clothoid_end_state(
            state, kappa, segment[..., i, 0], segment[..., i, 1]
        )
        ends.append(torch.cat((state, kappa[..., None]), -1))
    return (
        torch.stack(starts, -2),
        torch.stack(start_kappas, -1),
        torch.stack(ends, -2),
    )


def render_packed_stateful(
    packed: PackedStatefulStrokes,
    *,
    size: int = 64,
    curve_samples: int = 24,
    softness_px: float = 1.0,
    distance_softmin_px: float = 0.0,
    background: float = 1.0,
    pixel_chunk: int = 512,
) -> torch.Tensor:
    """Render packed chains, preserving Stroke/segment painter order."""
    starts, kappas, _ = stateful_chain_states(
        packed.anchor, packed.base_kappa,
        packed.segment, packed.curvature_jump,
    )
    style = (
        packed.base_style[:, :, None, :]
        + packed.style_delta.cumsum(-2)
    )
    B, K, S, _ = starts.shape
    return render_clothoid_batch(
        starts.reshape(B, K * S, 3),
        kappas.reshape(B, K * S),
        packed.segment[..., 0].reshape(B, K * S),
        packed.segment[..., 1].reshape(B, K * S),
        style[..., 0].reshape(B, K * S),
        style[..., 1:5].reshape(B, K * S, 4),
        valid=packed.segment_mask.reshape(B, K * S),
        size=size, curve_samples=curve_samples,
        softness_px=softness_px,
        distance_softmin_px=distance_softmin_px,
        background=background, pixel_chunk=pixel_chunk,
    )

