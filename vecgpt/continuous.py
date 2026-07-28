"""Hierarchical continuous vector autoencoder.

The scene structure is generated at Stroke granularity.  Geometry inside a
Stroke is emitted in one parallel pass as continuous values, instead of a
long sequence of quantised scalar tokens.  The only categorical decisions
are whether another Stroke exists and how many segments it contains.

There are no semantic slots: ``max_strokes`` and ``max_segments`` are batch
padding/context limits, analogous to an LLM context length.  The recurrent
planner dynamically stops at any Stroke index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vecgpt.scene import S_B, S_G, S_KAPPA, S_LEN, S_R, S_WIDTH, Stroke


@dataclass
class PackedStrokes:
    anchor: torch.Tensor       # [B,K,4] = x,y,sin(theta),cos(theta)
    base_style: torch.Tensor   # [B,K,5] = width,r,g,b,alpha
    segment: torch.Tensor      # [B,K,S,2] = length,turn
    style_delta: torch.Tensor  # [B,K,S,5], sparse changes from prior style
    style_change: torch.Tensor # [B,K,S], true only when style changes
    stroke_mask: torch.Tensor  # [B,K]
    segment_mask: torch.Tensor # [B,K,S]
    counts: torch.Tensor       # [B,K], zero for padding


@dataclass
class ContinuousOutput:
    present_logits: torch.Tensor  # [B,K]
    count_logits: torch.Tensor    # [B,K,S]
    anchor: torch.Tensor          # [B,K,4]
    base_style: torch.Tensor      # [B,K,5]
    segment: torch.Tensor         # [B,K,S,2]
    style_change_logits: torch.Tensor # [B,K,S]
    style_delta: torch.Tensor     # [B,K,S,5]
    latent: torch.Tensor          # [B,D]


def pack_scenes(
    scenes: list[list[Stroke]], max_strokes: int, max_segments: int,
    device=None,
) -> PackedStrokes:
    """Convert canonical Stroke scenes to padded continuous tensors."""
    B = len(scenes)
    anchor = torch.zeros(B, max_strokes, 4, device=device)
    base_style = torch.zeros(B, max_strokes, 5, device=device)
    segment = torch.zeros(B, max_strokes, max_segments, 2, device=device)
    style_delta = torch.zeros(B, max_strokes, max_segments, 5, device=device)
    style_change = torch.zeros(
        B, max_strokes, max_segments, dtype=torch.bool, device=device
    )
    stroke_mask = torch.zeros(B, max_strokes, dtype=torch.bool, device=device)
    segment_mask = torch.zeros(
        B, max_strokes, max_segments, dtype=torch.bool, device=device
    )
    counts = torch.zeros(B, max_strokes, dtype=torch.long, device=device)
    for b, strokes in enumerate(scenes):
        if len(strokes) > max_strokes:
            raise ValueError(
                f"scene has {len(strokes)} strokes, context limit is {max_strokes}"
            )
        for k, st in enumerate(strokes):
            n = int(st.segs.shape[0])
            if n < 1 or n > max_segments:
                raise ValueError(
                    f"stroke has {n} segments, context limit is {max_segments}"
                )
            a, g = st.anchor.to(device), st.segs.to(device)
            anchor[b, k] = torch.stack((
                a[0], a[1], a[2].sin(), a[2].cos()
            ))
            turn = g[:, S_KAPPA] * g[:, S_LEN]
            alpha = (
                g[:, S_B + 1] if g.shape[1] > S_B + 1
                else torch.ones_like(turn)
            )
            styles = torch.stack((
                g[:, S_WIDTH], g[:, S_R], g[:, S_G], g[:, S_B], alpha,
            ), -1)
            base_style[b, k] = styles[0]
            segment[b, k, :n] = torch.stack((g[:, S_LEN], turn), -1)
            if n > 1:
                delta = styles[1:] - styles[:-1]
                changed = delta.abs().amax(-1) > 1e-6
                style_delta[b, k, 1:n] = delta
                style_change[b, k, 1:n] = changed
            stroke_mask[b, k] = True
            segment_mask[b, k, :n] = True
            counts[b, k] = n
    return PackedStrokes(
        anchor, base_style, segment, style_delta, style_change,
        stroke_mask, segment_mask, counts,
    )


def _segment_features(p: PackedStrokes) -> torch.Tensor:
    """Scale continuous geometry to roughly unit variance for the encoder."""
    delta = p.style_delta
    return torch.cat((
        p.segment[..., 0:1] / 0.25,
        p.segment[..., 1:2] / math.pi,
        delta[..., 0:1] / 0.03,
        delta[..., 1:5] / 0.30,
        p.style_change[..., None].to(p.segment.dtype),
    ), -1)


def _style_features(style: torch.Tensor) -> torch.Tensor:
    """Put width on the same numerical scale as RGBA.

    Raw widths are only 0.01--0.06 while colour channels span 0--1. Feeding
    those values directly into one projection made LayerNorm/attention retain
    colour but collapse width toward its dataset mean.
    """
    return torch.cat((style[..., :1] / 0.08, style[..., 1:]), -1)


def _trajectory_states(
    anchor: torch.Tensor, segment: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable end position and heading after every local arc.

    Segment-wise parameter errors are not independent: a small turn error
    rotates every following segment.  Returning accumulated physical states
    lets the loss supervise the actual Stroke trajectory without rasterizing.
    """
    length, turn = segment.unbind(-1)
    theta0 = torch.atan2(anchor[..., 2], anchor[..., 3])
    theta_start = theta0[..., None] + torch.cumsum(
        torch.cat((turn[..., :1] * 0.0, turn[..., :-1]), -1), -1
    )
    # Stable forms of sin(t)/t and (1-cos(t))/t, including t=0.
    local_x = length * torch.sinc(turn / math.pi)
    local_y = length * (
        0.5 * turn * torch.sinc(turn / (2.0 * math.pi)).square()
    )
    ct, st = theta_start.cos(), theta_start.sin()
    dx = ct * local_x - st * local_y
    dy = st * local_x + ct * local_y
    start_xy = anchor[..., :2, None].transpose(-1, -2)
    end_xy = start_xy + torch.cumsum(torch.stack((dx, dy), -1), -2)
    theta_end = theta0[..., None] + torch.cumsum(turn, -1)
    heading = torch.stack((theta_end.sin(), theta_end.cos()), -1)
    return end_xy, heading


class ContinuousVectorEncoder(nn.Module):
    """Vector program -> one compact scene latent."""

    def __init__(
        self, d_model: int = 192, n_heads: int = 6,
        n_layers: int = 3, max_strokes: int = 16, max_segments: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_strokes = max_strokes
        self.max_segments = max_segments
        self.segment_in = nn.Sequential(
            nn.Linear(8, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.anchor_in = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.style_in = nn.Sequential(
            nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.segment_pos = nn.Parameter(
            torch.randn(1, max_segments, d_model) * 0.02
        )
        seg_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.segment_encoder = nn.TransformerEncoder(seg_layer, num_layers=2)
        self.stroke_fuse = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.scene_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.stroke_pos = nn.Parameter(
            torch.randn(1, max_strokes, d_model) * 0.02
        )
        scene_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.scene_encoder = nn.TransformerEncoder(
            scene_layer, num_layers=n_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, p: PackedStrokes) -> torch.Tensor:
        B, K, S, _ = p.segment.shape
        x = self.segment_in(_segment_features(p))
        x = x + self.segment_pos[:, :S, None, :].squeeze(2)
        x = x.reshape(B * K, S, self.d_model)
        pad = (~p.segment_mask).reshape(B * K, S)
        # Transformer attention cannot consume an entirely masked row.
        safe_pad = pad.clone()
        safe_pad[pad.all(-1), 0] = False
        x = self.segment_encoder(x, src_key_padding_mask=safe_pad)
        valid = (~pad).to(x.dtype)
        pooled = (
            x * valid[..., None]
        ).sum(1) / valid.sum(1, keepdim=True).clamp_min(1.0)
        pooled = pooled.reshape(B, K, self.d_model)
        anchor_code = self.anchor_in(p.anchor)
        style_code = self.style_in(_style_features(p.base_style))
        stroke = (
            pooled + anchor_code + style_code
            + self.stroke_fuse(torch.cat((
                pooled, anchor_code, style_code
            ), -1))
        )
        stroke = stroke + self.stroke_pos[:, :K]
        cls = self.scene_cls.expand(B, -1, -1)
        scene = torch.cat((cls, stroke), 1)
        scene_pad = torch.cat((
            torch.zeros(B, 1, dtype=torch.bool, device=stroke.device),
            ~p.stroke_mask,
        ), 1)
        scene = self.scene_encoder(scene, src_key_padding_mask=scene_pad)
        # A direct permutation-stable carrier prevents the CLS attention
        # bottleneck from preserving only the largest/easiest field.  The
        # Transformer still models relationships and order; this residual
        # merely guarantees that exact vector coordinates reach the latent.
        valid_stroke = p.stroke_mask.to(stroke.dtype)
        carrier = (
            stroke * valid_stroke[..., None]
        ).sum(1) / valid_stroke.sum(1, keepdim=True).clamp_min(1.0)
        return self.norm(scene[:, 0] + carrier)


class ContinuousStrokeDecoder(nn.Module):
    """Compact latent -> dynamic Stroke sequence with parallel segments."""

    def __init__(
        self, d_model: int = 192, max_strokes: int = 16,
        max_segments: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_strokes = max_strokes
        self.max_segments = max_segments
        self.init = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh()
        )
        self.planner = nn.GRUCell(d_model, d_model)
        self.stroke_code = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
            nn.GELU(), nn.Linear(2 * d_model, d_model),
        )
        self.present = nn.Linear(d_model, 1)
        self.count = nn.Linear(d_model, max_segments)
        self.anchor_head = nn.Linear(d_model, 4)
        self.base_style_head = nn.Linear(d_model, 5)
        self.segment_pos = nn.Parameter(
            torch.randn(1, max_segments, d_model) * 0.02
        )
        self.segment_decoder = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
            nn.GELU(), nn.Linear(2 * d_model, d_model), nn.GELU(),
        )
        self.segment_head = nn.Linear(d_model, 2)
        self.style_change_head = nn.Linear(d_model, 1)
        self.style_delta_head = nn.Linear(d_model, 5)
        self.step_emb = nn.Parameter(
            torch.randn(max_strokes, d_model) * 0.02
        )

    def forward(self, latent: torch.Tensor) -> ContinuousOutput:
        B = latent.shape[0]
        h = self.init(latent)
        present, counts, anchors, base_styles = [], [], [], []
        segments, style_changes, style_deltas = [], [], []
        for k in range(self.max_strokes):
            code = h + self.stroke_code(h)
            present.append(self.present(code).squeeze(-1))
            counts.append(self.count(code))
            raw_a = self.anchor_head(code)
            xy = raw_a[..., :2].sigmoid()
            direction = F.normalize(raw_a[..., 2:4], dim=-1, eps=1e-4)
            anchors.append(torch.cat((xy, direction), -1))
            raw_style = self.base_style_head(code)
            base_styles.append(torch.cat((
                0.08 * raw_style[..., :1].sigmoid(),
                raw_style[..., 1:].sigmoid(),
            ), -1))

            q = code[:, None, :] + self.segment_pos
            decoded = self.segment_decoder(q)
            raw = self.segment_head(decoded)
            # Explicit physical ranges; every output remains continuous.
            seg = torch.stack((
                1.25 * raw[..., 0].sigmoid(),
                2.0 * math.pi * raw[..., 1].tanh(),
            ), -1)
            segments.append(seg)
            style_changes.append(
                self.style_change_head(decoded).squeeze(-1)
            )
            raw_delta = self.style_delta_head(decoded).tanh()
            style_deltas.append(torch.cat((
                0.03 * raw_delta[..., :1],
                0.30 * raw_delta[..., 1:],
            ), -1))
            h = self.planner(code + self.step_emb[k], h)
        return ContinuousOutput(
            torch.stack(present, 1),
            torch.stack(counts, 1),
            torch.stack(anchors, 1),
            torch.stack(base_styles, 1),
            torch.stack(segments, 1),
            torch.stack(style_changes, 1),
            torch.stack(style_deltas, 1),
            latent,
        )


class ContinuousVectorAutoencoder(nn.Module):
    def __init__(
        self, d_model: int = 192, n_heads: int = 6, n_layers: int = 3,
        max_strokes: int = 16, max_segments: int = 32,
    ):
        super().__init__()
        self.max_strokes = max_strokes
        self.max_segments = max_segments
        self.encoder = ContinuousVectorEncoder(
            d_model, n_heads, n_layers, max_strokes, max_segments
        )
        self.decoder = ContinuousStrokeDecoder(
            d_model, max_strokes, max_segments
        )

    def forward(self, packed: PackedStrokes) -> ContinuousOutput:
        return self.decoder(self.encoder(packed))


@dataclass
class ASTLatent:
    """Variable typed latent tree, padded only for batching."""
    root: torch.Tensor          # [B,D]
    stroke: torch.Tensor        # [B,K,D]
    frame: torch.Tensor         # [B,K,D], x/y/theta typed node
    style: torch.Tensor         # [B,K,D], base style typed node
    segment: torch.Tensor       # [B,K,S,D]
    stroke_mask: torch.Tensor   # [B,K]
    segment_mask: torch.Tensor  # [B,K,S]


class ASTVectorEncoder(nn.Module):
    """Typed ROOT -> STROKE -> SEGMENT encoder.

    Unlike the compact baseline, exact segment information is never forced
    through one scene vector.  Higher nodes aggregate semantics while every
    dynamic leaf retains its own continuous latent.
    """

    def __init__(
        self, d_model=128, n_heads=4, n_layers=2,
        max_strokes=16, max_segments=32,
    ):
        super().__init__()
        self.d_model = d_model
        self.segment_in = nn.Sequential(
            nn.Linear(8, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.anchor_in = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.style_in = nn.Sequential(
            nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.segment_pos = nn.Parameter(
            torch.randn(1, 1, max_segments, d_model) * 0.02
        )
        self.stroke_pos = nn.Parameter(
            torch.randn(1, max_strokes, d_model) * 0.02
        )
        self.root = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        seg_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.segment_layers = nn.TransformerEncoder(
            seg_layer, num_layers=max(1, n_layers)
        )
        scene_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.scene_layers = nn.TransformerEncoder(
            scene_layer, num_layers=max(1, n_layers)
        )
        self.stroke_fuse = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.segment_context = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
        )
        self.frame_context = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
        )
        self.style_context = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, p: PackedStrokes) -> ASTLatent:
        B, K, S, _ = p.segment.shape
        raw_segment = self.segment_in(_segment_features(p))
        x = raw_segment + self.segment_pos[:, :, :S]
        x = x.reshape(B * K, S, self.d_model)
        pad = (~p.segment_mask).reshape(B * K, S)
        safe = pad.clone()
        safe[safe.all(-1), 0] = False
        x = self.segment_layers(x, src_key_padding_mask=safe)
        valid = (~pad).to(x.dtype)
        pooled = (
            x * valid[..., None]
        ).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        pooled = pooled.reshape(B, K, self.d_model)
        anchor = self.anchor_in(p.anchor)
        style = self.style_in(_style_features(p.base_style))
        stroke = (
            pooled + anchor + style
            + self.stroke_fuse(torch.cat((pooled, anchor, style), -1))
            + self.stroke_pos[:, :K]
        )
        root = self.root.expand(B, -1, -1)
        tree = torch.cat((root, stroke), 1)
        tree_pad = torch.cat((
            torch.zeros(B, 1, dtype=torch.bool, device=stroke.device),
            ~p.stroke_mask,
        ), 1)
        tree = self.scene_layers(tree, src_key_padding_mask=tree_pad)
        root_latent = self.norm(tree[:, 0])
        stroke_latent = self.norm(tree[:, 1:])
        frame_latent = self.norm(
            anchor + self.frame_context(stroke_latent)
        )
        style_latent = self.norm(
            style + self.style_context(stroke_latent)
        )
        # Parent-to-child context is additive; leaf identity remains present
        # through the residual x term.
        leaf = x.reshape(B, K, S, self.d_model)
        leaf = self.norm(
            leaf + self.segment_context(stroke_latent)[:, :, None, :]
        )
        return ASTLatent(
            root_latent, stroke_latent, frame_latent, style_latent, leaf,
            p.stroke_mask, p.segment_mask,
        )


class ASTVectorDecoder(nn.Module):
    """Parallel typed-node decoder for a variable AST latent."""

    def __init__(self, d_model=128, max_strokes=16, max_segments=32):
        super().__init__()
        self.max_strokes = max_strokes
        self.max_segments = max_segments
        self.stroke_decode = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
            nn.GELU(), nn.Linear(2 * d_model, d_model), nn.GELU(),
        )
        self.segment_decode = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
            nn.GELU(), nn.Linear(2 * d_model, d_model), nn.GELU(),
        )
        self.frame_decode = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
            nn.GELU(), nn.Linear(2 * d_model, d_model),
        )
        self.style_decode = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
            nn.GELU(), nn.Linear(2 * d_model, d_model),
        )
        self.present = nn.Linear(d_model, 1)
        self.count = nn.Linear(d_model, max_segments)
        self.anchor_head = nn.Linear(d_model, 4)
        self.base_style_head = nn.Linear(d_model, 5)
        self.segment_head = nn.Linear(d_model, 2)
        self.style_change_head = nn.Linear(d_model, 1)
        self.style_delta_head = nn.Linear(d_model, 5)

    def forward(self, latent: ASTLatent) -> ContinuousOutput:
        stroke = latent.stroke + self.stroke_decode(latent.stroke)
        frame = latent.frame + self.frame_decode(latent.frame)
        style = latent.style + self.style_decode(latent.style)
        leaf = latent.segment + self.segment_decode(latent.segment)
        raw_anchor = self.anchor_head(frame)
        anchor = torch.cat((
            raw_anchor[..., :2].sigmoid(),
            F.normalize(raw_anchor[..., 2:4], dim=-1, eps=1e-4),
        ), -1)
        raw_style = self.base_style_head(style)
        base_style = torch.cat((
            0.08 * raw_style[..., :1].sigmoid(),
            raw_style[..., 1:].sigmoid(),
        ), -1)
        raw_segment = self.segment_head(leaf)
        segment = torch.stack((
            1.25 * raw_segment[..., 0].sigmoid(),
            2.0 * math.pi * raw_segment[..., 1].tanh(),
        ), -1)
        raw_delta = self.style_delta_head(leaf).tanh()
        style_delta = torch.cat((
            0.03 * raw_delta[..., :1],
            0.30 * raw_delta[..., 1:],
        ), -1)
        return ContinuousOutput(
            self.present(stroke).squeeze(-1),
            self.count(stroke),
            anchor,
            base_style,
            segment,
            self.style_change_head(leaf).squeeze(-1),
            style_delta,
            latent.root,
        )


class ASTVectorAutoencoder(nn.Module):
    """Recommended vector representation model; compact AE remains baseline."""

    def __init__(
        self, d_model=128, n_heads=4, n_layers=2,
        max_strokes=16, max_segments=32,
    ):
        super().__init__()
        self.max_strokes = max_strokes
        self.max_segments = max_segments
        self.encoder = ASTVectorEncoder(
            d_model, n_heads, n_layers, max_strokes, max_segments
        )
        self.decoder = ASTVectorDecoder(
            d_model, max_strokes, max_segments
        )

    def forward(self, packed: PackedStrokes) -> ContinuousOutput:
        return self.decoder(self.encoder(packed))


def continuous_losses(
    out: ContinuousOutput, target: PackedStrokes,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Direct, topology-aware vector supervision."""
    B, K = target.stroke_mask.shape
    present_target = target.stroke_mask.to(out.present_logits.dtype)
    # Empty slots after the first absent Stroke are explicit stop targets.
    present = F.binary_cross_entropy_with_logits(
        out.present_logits, present_target
    )
    sm = target.stroke_mask
    if sm.any():
        count = F.cross_entropy(
            out.count_logits[sm], target.counts[sm] - 1
        )
        # Losses are expressed in meaningful physical error units.  Raw
        # SmoothL1 made a 0.17-canvas anchor error cost only ~0.014, so the
        # decoder rationally spent its capacity on the unit-scale angle and
        # ignored x/y, length and RGB.  These scales make a visibly bad error
        # comparable across fields without relying on batch statistics.
        anchor_xy = (
            (out.anchor[..., :2][sm] - target.anchor[..., :2][sm]) / 0.25
        ).square().mean()
        anchor_angle = (
            1.0 - (
                out.anchor[..., 2:4][sm] * target.anchor[..., 2:4][sm]
            ).sum(-1)
        ).mean()
    else:
        z = out.present_logits.new_zeros(())
        count = anchor_xy = anchor_angle = z

    gm = target.segment_mask
    if gm.any():
        pred, truth = out.segment[gm], target.segment[gm]
        length = ((pred[:, 0] - truth[:, 0]) / 0.25).square().mean()
        turn = ((pred[:, 1] - truth[:, 1]) / math.pi).square().mean()
        pred_xy, pred_heading = _trajectory_states(out.anchor, out.segment)
        true_xy, true_heading = _trajectory_states(
            target.anchor, target.segment
        )
        trajectory_xy = (
            (pred_xy[gm] - true_xy[gm]) / 0.25
        ).square().mean()
        trajectory_angle = (
            1.0 - (pred_heading[gm] * true_heading[gm]).sum(-1)
        ).mean()
        change = F.binary_cross_entropy_with_logits(
            out.style_change_logits[gm],
            target.style_change[gm].to(out.segment.dtype),
        )
    else:
        z = out.present_logits.new_zeros(())
        length = turn = trajectory_xy = trajectory_angle = change = z
    if sm.any():
        width = (
            (out.base_style[..., 0][sm] - target.base_style[..., 0][sm])
            / 0.015
        ).square().mean()
        color = (
            (out.base_style[..., 1:][sm] - target.base_style[..., 1:][sm])
            / 0.30
        ).square().mean()
    else:
        width = color = out.present_logits.new_zeros(())
    dm = target.style_change & target.segment_mask
    if dm.any():
        delta = (
            (out.style_delta[dm] - target.style_delta[dm])
            / out.style_delta.new_tensor([0.03, 0.30, 0.30, 0.30, 0.30])
        ).square().mean()
    else:
        delta = out.present_logits.new_zeros(())
    total = (
        2.0 * present + 2.0 * count
        + 2.0 * anchor_xy + 2.0 * anchor_angle
        + 2.0 * length + 2.0 * turn + width + color
        + 0.5 * trajectory_xy + 0.5 * trajectory_angle
        + 0.5 * change + delta
    )
    return total, {
        "present": present.detach(), "count": count.detach(),
        "anchor_xy": anchor_xy.detach(), "anchor_angle": anchor_angle.detach(),
        "length": length.detach(), "turn": turn.detach(),
        "trajectory_xy": trajectory_xy.detach(),
        "trajectory_angle": trajectory_angle.detach(),
        "width": width.detach(), "color": color.detach(),
        "style_change": change.detach(), "style_delta": delta.detach(),
    }


def output_to_scenes(
    out: ContinuousOutput, soft_structure: bool = False,
) -> list[list[Stroke]]:
    """Decode predictions without consulting target topology.

    With ``soft_structure=True`` every technical padding segment is rendered,
    but its alpha is the differentiable probability that both its Stroke and
    its segment index exist.  Thus render loss reaches structural logits.
    """
    scenes: list[list[Stroke]] = []
    count_p = out.count_logits.softmax(-1)
    for b in range(out.anchor.shape[0]):
        strokes = []
        for k in range(out.anchor.shape[1]):
            present_p = out.present_logits[b, k].sigmoid()
            if not soft_structure and float(present_p) < 0.5:
                break
            if soft_structure:
                n = out.segment.shape[2]
            else:
                n = int(out.count_logits[b, k].argmax()) + 1
            a = out.anchor[b, k]
            theta = torch.atan2(a[2], a[3])
            seg = out.segment[b, k, :n]
            length, turn = seg[:, 0], seg[:, 1]
            curvature = turn / length.clamp_min(1e-4)
            base = out.base_style[b, k]
            gate = (
                out.style_change_logits[b, k, :n].sigmoid()
                if soft_structure else
                (out.style_change_logits[b, k, :n] >= 0).to(seg.dtype)
            )
            changes = gate[:, None] * out.style_delta[b, k, :n]
            styles = base[None] + changes.cumsum(0)
            styles = torch.cat((
                styles[:, :1].clamp(0.001, 0.08),
                styles[:, 1:].clamp(0.0, 1.0),
            ), -1)
            if soft_structure:
                # P(count >= j+1) for j=0..S-1.
                survival = count_p[b, k].flip(0).cumsum(0).flip(0)[:n]
                alpha = styles[:, 4] * present_p * survival
            else:
                alpha = styles[:, 4]
            g = torch.stack((
                length, curvature, styles[:, 0],
                styles[:, 1], styles[:, 2], styles[:, 3], alpha,
            ), -1)
            strokes.append(Stroke(torch.stack((a[0], a[1], theta)), g))
        scenes.append(strokes)
    return scenes
