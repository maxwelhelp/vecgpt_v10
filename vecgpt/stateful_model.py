"""Typed-AST autoencoder for stateful clothoid Stroke chains."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vecgpt.continuous import ASTLatent
from vecgpt.stateful import PackedStatefulStrokes, stateful_chain_states


@dataclass
class StatefulOutput:
    present_logits: torch.Tensor
    count_logits: torch.Tensor
    anchor: torch.Tensor
    base_kappa: torch.Tensor
    base_style: torch.Tensor
    segment: torch.Tensor
    curvature_change_logits: torch.Tensor
    curvature_jump: torch.Tensor
    style_change_logits: torch.Tensor
    style_delta: torch.Tensor
    latent: torch.Tensor


def _style_features(style):
    return torch.cat((style[..., :1] / 0.08, style[..., 1:]), -1)


def _segment_features(p: PackedStatefulStrokes):
    return torch.cat((
        p.segment[..., 0:1] / 0.25,
        p.segment[..., 1:2] / 8.0,
        p.curvature_jump[..., None] / 12.0,
        p.curvature_change[..., None].to(p.segment.dtype),
        p.style_delta[..., :1] / 0.03,
        p.style_delta[..., 1:] / 0.30,
        p.style_change[..., None].to(p.segment.dtype),
    ), -1)


class StatefulASTEncoder(nn.Module):
    def __init__(
        self, d_model=128, n_heads=4, n_layers=2,
        max_strokes=16, max_segments=32,
    ):
        super().__init__()
        self.d_model = d_model
        self.segment_in = nn.Sequential(
            nn.Linear(10, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.anchor_in = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.kappa_in = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
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
        scene_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.segment_layers = nn.TransformerEncoder(
            seg_layer, num_layers=max(1, n_layers)
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

    def forward(self, p: PackedStatefulStrokes) -> ASTLatent:
        B, K, S, _ = p.segment.shape
        x = self.segment_in(_segment_features(p))
        x = x + self.segment_pos[:, :, :S]
        x = x.reshape(B * K, S, self.d_model)
        pad = (~p.segment_mask).reshape(B * K, S)
        safe = pad.clone()
        safe[safe.all(-1), 0] = False
        x = self.segment_layers(x, src_key_padding_mask=safe)
        valid = (~pad).to(x.dtype)
        pooled = (
            (x * valid[..., None]).sum(1)
            / valid.sum(1, keepdim=True).clamp_min(1)
        ).reshape(B, K, self.d_model)
        frame = (
            self.anchor_in(p.anchor)
            + self.kappa_in((p.base_kappa / 8.0)[..., None])
        )
        style = self.style_in(_style_features(p.base_style))
        stroke = (
            pooled + frame + style
            + self.stroke_fuse(torch.cat((pooled, frame, style), -1))
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
            frame + self.frame_context(stroke_latent)
        )
        style_latent = self.norm(
            style + self.style_context(stroke_latent)
        )
        leaf = x.reshape(B, K, S, self.d_model)
        leaf = self.norm(
            leaf + self.segment_context(stroke_latent)[:, :, None, :]
        )
        return ASTLatent(
            root_latent, stroke_latent, frame_latent, style_latent, leaf,
            p.stroke_mask, p.segment_mask,
        )


class StatefulASTDecoder(nn.Module):
    def __init__(self, d_model=128, max_strokes=16, max_segments=32):
        super().__init__()
        self.max_strokes = max_strokes
        self.max_segments = max_segments
        def block():
            return nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
                nn.GELU(), nn.Linear(2 * d_model, d_model), nn.GELU(),
            )
        self.stroke_decode = block()
        self.frame_decode = block()
        self.style_decode = block()
        self.segment_decode = block()
        self.present = nn.Linear(d_model, 1)
        self.count = nn.Linear(d_model, max_segments)
        self.anchor_head = nn.Linear(d_model, 4)
        self.base_kappa_head = nn.Linear(d_model, 1)
        self.base_style_head = nn.Linear(d_model, 5)
        self.segment_head = nn.Linear(d_model, 2)
        self.curvature_change_head = nn.Linear(d_model, 1)
        self.curvature_jump_head = nn.Linear(d_model, 1)
        self.style_change_head = nn.Linear(d_model, 1)
        self.style_delta_head = nn.Linear(d_model, 5)

    def forward(self, latent: ASTLatent) -> StatefulOutput:
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
            0.60 * raw_segment[..., 0].sigmoid(),
            12.0 * raw_segment[..., 1].tanh(),
        ), -1)
        raw_style_delta = self.style_delta_head(leaf).tanh()
        style_delta = torch.cat((
            0.03 * raw_style_delta[..., :1],
            0.30 * raw_style_delta[..., 1:],
        ), -1)
        return StatefulOutput(
            self.present(stroke).squeeze(-1),
            self.count(stroke),
            anchor,
            12.0 * self.base_kappa_head(frame).squeeze(-1).tanh(),
            base_style,
            segment,
            self.curvature_change_head(leaf).squeeze(-1),
            20.0 * self.curvature_jump_head(leaf).squeeze(-1).tanh(),
            self.style_change_head(leaf).squeeze(-1),
            style_delta,
            latent.root,
        )


class StatefulASTAutoencoder(nn.Module):
    def __init__(
        self, d_model=128, n_heads=4, n_layers=2,
        max_strokes=16, max_segments=32,
    ):
        super().__init__()
        self.max_strokes = max_strokes
        self.max_segments = max_segments
        self.encoder = StatefulASTEncoder(
            d_model, n_heads, n_layers, max_strokes, max_segments
        )
        self.decoder = StatefulASTDecoder(
            d_model, max_strokes, max_segments
        )

    def forward(self, packed):
        return self.decoder(self.encoder(packed))


def effective_curvature_jump(out: StatefulOutput):
    return out.curvature_jump * out.curvature_change_logits.sigmoid()


def effective_style_delta(out: StatefulOutput):
    return out.style_delta * out.style_change_logits.sigmoid()[..., None]


def stateful_losses(
    out: StatefulOutput, target: PackedStatefulStrokes,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dtype = out.present_logits.dtype
    sm, gm = target.stroke_mask, target.segment_mask
    present = F.binary_cross_entropy_with_logits(
        out.present_logits, sm.to(dtype)
    )
    count = F.cross_entropy(
        out.count_logits[sm], target.counts[sm] - 1
    ) if sm.any() else present.new_zeros(())
    anchor_xy = (
        ((out.anchor[..., :2][sm] - target.anchor[..., :2][sm]) / 0.25)
        .square().mean()
    )
    anchor_angle = (
        1.0 - (
            out.anchor[..., 2:4][sm] * target.anchor[..., 2:4][sm]
        ).sum(-1)
    ).mean()
    base_kappa = (
        ((out.base_kappa[sm] - target.base_kappa[sm]) / 8.0)
        .square().mean()
    )
    base_style = (
        (out.base_style[sm] - target.base_style[sm])
        / out.base_style.new_tensor([0.03, 0.30, 0.30, 0.30, 0.30])
    ).square().mean()

    pred_seg, true_seg = out.segment[gm], target.segment[gm]
    length = ((pred_seg[:, 0] - true_seg[:, 0]) / 0.25).square().mean()
    delta_kappa = (
        (pred_seg[:, 1] - true_seg[:, 1]) / 8.0
    ).square().mean()
    curvature_change = F.binary_cross_entropy_with_logits(
        out.curvature_change_logits[gm],
        target.curvature_change[gm].to(dtype),
    )
    # Sparse values are supervised where present; effective zero is
    # supervised everywhere through the trajectory and gate probability.
    cm = target.curvature_change & gm
    curvature_jump = (
        ((out.curvature_jump[cm] - target.curvature_jump[cm]) / 12.0)
        .square().mean()
        if cm.any() else present.new_zeros(())
    )
    effective_jump = (
        (
            (
                effective_curvature_jump(out)[gm]
                - target.curvature_jump[gm]
            ) / 12.0
        ).square().mean()
    )
    style_change = F.binary_cross_entropy_with_logits(
        out.style_change_logits[gm], target.style_change[gm].to(dtype)
    )
    dm = target.style_change & gm
    style_delta = (
        (
            (out.style_delta[dm] - target.style_delta[dm])
            / out.style_delta.new_tensor([0.03, 0.30, 0.30, 0.30, 0.30])
        ).square().mean()
        if dm.any() else present.new_zeros(())
    )
    effective_delta = (
        (
            (
                effective_style_delta(out)[gm]
                - target.style_delta[gm]
            )
            / out.style_delta.new_tensor(
                [0.03, 0.30, 0.30, 0.30, 0.30]
            )
        ).square().mean()
    )

    pred_starts, pred_k, pred_ends = stateful_chain_states(
        out.anchor, out.base_kappa, out.segment,
        effective_curvature_jump(out),
    )
    true_starts, true_k, true_ends = stateful_chain_states(
        target.anchor, target.base_kappa, target.segment,
        target.curvature_jump,
    )
    trajectory_xy = (
        (pred_ends[..., :2][gm] - true_ends[..., :2][gm]) / 0.25
    ).square().mean()
    trajectory_angle = (
        1.0 - torch.cos(
            pred_ends[..., 2][gm] - true_ends[..., 2][gm]
        )
    ).mean()
    trajectory_kappa = (
        (pred_ends[..., 3][gm] - true_ends[..., 3][gm]) / 10.0
    ).square().mean()
    total = (
        2 * present + 2 * count
        + 2 * anchor_xy + 2 * anchor_angle + base_kappa + base_style
        + 2 * length + 2 * delta_kappa
        + 0.5 * curvature_change + curvature_jump + 2 * effective_jump
        + 0.5 * style_change + style_delta + effective_delta
        + trajectory_xy
        + trajectory_angle
        + trajectory_kappa
    )
    return total, {
        "present": present.detach(), "count": count.detach(),
        "anchor_xy": anchor_xy.detach(), "anchor_angle": anchor_angle.detach(),
        "base_kappa": base_kappa.detach(), "base_style": base_style.detach(),
        "length": length.detach(), "delta_kappa": delta_kappa.detach(),
        "curvature_change": curvature_change.detach(),
        "curvature_jump": curvature_jump.detach(),
        "effective_jump": effective_jump.detach(),
        "style_change": style_change.detach(),
        "style_delta": style_delta.detach(),
        "effective_delta": effective_delta.detach(),
        "trajectory_xy": trajectory_xy.detach(),
        "trajectory_angle": trajectory_angle.detach(),
        "trajectory_kappa": trajectory_kappa.detach(),
    }


def output_to_packed(
    out: StatefulOutput, soft_structure: bool = False,
) -> PackedStatefulStrokes:
    B, K, S, _ = out.segment.shape
    counts = out.count_logits.argmax(-1) + 1
    if soft_structure:
        stroke_mask = torch.ones(
            B, K, dtype=torch.bool, device=out.segment.device
        )
        segment_mask = stroke_mask[..., None].expand(B, K, S)
    else:
        stroke_mask = out.present_logits.sigmoid() >= 0.5
        # Canonical Stroke list stops at the first absent technical position.
        stroke_mask = stroke_mask & (
            stroke_mask.to(torch.int64).cumprod(-1).bool()
        )
        idx = torch.arange(S, device=out.segment.device)
        segment_mask = (
            idx[None, None, :] < counts[..., None]
        ) & stroke_mask[..., None]
    curvature_gate = (
        out.curvature_change_logits.sigmoid()
        if soft_structure else
        (out.curvature_change_logits >= 0).to(out.segment.dtype)
    )
    style_gate = (
        out.style_change_logits.sigmoid()
        if soft_structure else
        (out.style_change_logits >= 0).to(out.segment.dtype)
    )
    style_delta = style_gate[..., None] * out.style_delta
    base_style = out.base_style
    if soft_structure:
        # Make raster loss reach the variable-cardinality decisions.  A
        # segment contributes in proportion to P(stroke exists) and
        # P(count >= segment_index + 1).  Convert those per-segment alpha
        # values back to the stateful base + delta representation.
        count_prob = out.count_logits.softmax(-1)
        survival = count_prob.flip(-1).cumsum(-1).flip(-1)
        present_prob = out.present_logits.sigmoid()[..., None]
        expanded_style = (
            out.base_style[:, :, None, :] + style_delta.cumsum(-2)
        )
        expanded_style = torch.cat((
            expanded_style[..., :4],
            expanded_style[..., 4:5]
            * present_prob[..., None]
            * survival[..., None],
        ), -1)
        base_style = expanded_style[:, :, 0]
        style_delta = torch.zeros_like(expanded_style)
        style_delta[:, :, 1:] = (
            expanded_style[:, :, 1:] - expanded_style[:, :, :-1]
        )
    return PackedStatefulStrokes(
        out.anchor, out.base_kappa, base_style, out.segment,
        curvature_gate * out.curvature_jump,
        curvature_gate >= 0.5,
        style_delta,
        style_gate >= 0.5,
        stroke_mask, segment_mask, counts,
    )
