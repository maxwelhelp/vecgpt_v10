"""Dynamic REGION programs with local stateful Stroke geometry.

REGION is a routing/ownership node, not a raster mask and not a semantic
class.  Its Sim(2) frame maps normalized local Stroke coordinates to the
canvas.  Padded tensors are only a batching representation; ``region_mask``
is the generated variable-cardinality prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from vecgpt.continuous import ASTLatent
from vecgpt.stateful import (
    PackedStatefulStrokes,
    StatefulStroke,
    pack_stateful_scenes,
    render_packed_stateful,
)
from vecgpt.stateful_model import (
    StatefulASTDecoder,
    StatefulASTEncoder,
    StatefulOutput,
    output_to_packed,
    stateful_losses,
)


@dataclass
class RegionProgram:
    # [tx, ty, theta, log_scale]
    frame: torch.Tensor
    strokes: list[StatefulStroke]
    diagnostic_kind: str = ""


@dataclass
class PackedRegionPrograms:
    frame: torch.Tensor               # [B,R,5] tx,ty,sin,cos,log_scale
    region_mask: torch.Tensor         # [B,R]
    region_count: torch.Tensor        # [B]
    local: PackedStatefulStrokes      # leading dimension B*R
    batch_size: int
    max_regions: int


@dataclass
class RegionASTOutput:
    region_present_logits: torch.Tensor
    frame: torch.Tensor
    local: StatefulOutput
    region_latent: torch.Tensor
    scene_latent: torch.Tensor


def region_hungarian_indices(
    out: RegionASTOutput, target: PackedRegionPrograms,
):
    """Match active target REGIONs to unordered predicted queries.

    Assignment is deliberately detached/non-differentiable; gradients flow
    through the matched losses after the discrete assignment is selected.
    """
    result = []
    pred_frame = out.frame.detach()
    pred_presence = out.region_present_logits.detach()
    for b in range(target.batch_size):
        gt = torch.nonzero(target.region_mask[b], as_tuple=False).flatten()
        if gt.numel() == 0:
            result.append((torch.empty(0, dtype=torch.long), gt.cpu()))
            continue
        pf = pred_frame[b, :, None, :]
        tf = target.frame[b, gt][None, :, :]
        xy = (pf[..., :2] - tf[..., :2]).abs().sum(-1) / 0.3
        angle = 1.0 - (pf[..., 2:4] * tf[..., 2:4]).sum(-1)
        scale = (pf[..., 4] - tf[..., 4]).abs()
        # Every matched target is present; this term discourages assigning a
        # confident background query to a real REGION.
        presence = -F.logsigmoid(pred_presence[b, :, None])
        cost = xy + angle + scale + presence
        rows, cols = linear_sum_assignment(cost.cpu().numpy())
        result.append((
            torch.as_tensor(rows, dtype=torch.long, device=out.frame.device),
            gt[torch.as_tensor(cols, device=gt.device)],
        ))
    return result


def pack_region_programs(
    scenes: list[list[RegionProgram]],
    max_regions: int,
    max_strokes: int,
    max_segments: int,
    device=None,
) -> PackedRegionPrograms:
    B = len(scenes)
    frame = torch.zeros(B, max_regions, 5, device=device)
    mask = torch.zeros(B, max_regions, dtype=torch.bool, device=device)
    flat: list[list[StatefulStroke]] = []
    for b, regions in enumerate(scenes):
        if len(regions) > max_regions:
            raise ValueError("scene exceeds max_regions context")
        for r in range(max_regions):
            if r < len(regions):
                region = regions[r]
                raw = region.frame.to(device)
                frame[b, r] = torch.stack((
                    raw[0], raw[1], raw[2].sin(), raw[2].cos(), raw[3],
                ))
                mask[b, r] = True
                flat.append(region.strokes)
            else:
                flat.append([])
    local = pack_stateful_scenes(
        flat, max_strokes, max_segments, device
    )
    return PackedRegionPrograms(
        frame, mask, mask.sum(-1), local, B, max_regions
    )


def _reshape_local(
    packed: PackedStatefulStrokes, B: int, R: int,
) -> dict[str, torch.Tensor]:
    return {
        key: value.reshape(B, R, *value.shape[1:])
        for key, value in vars(packed).items()
    }


def regions_to_global(
    packed: PackedRegionPrograms,
) -> PackedStatefulStrokes:
    """Apply all local Sim(2) frames and flatten REGION/STROKE painter order."""
    B, R = packed.batch_size, packed.max_regions
    q = _reshape_local(packed.local, B, R)
    K, S = q["segment"].shape[2:4]
    tx, ty = packed.frame[..., 0], packed.frame[..., 1]
    sn, cs = packed.frame[..., 2], packed.frame[..., 3]
    scale = packed.frame[..., 4].exp().clamp(1e-3, 2.0)

    local_xy = q["anchor"][..., :2] - 0.5
    lx, ly = local_xy[..., 0], local_xy[..., 1]
    gx = tx[..., None] + scale[..., None] * (
        cs[..., None] * lx - sn[..., None] * ly
    )
    gy = ty[..., None] + scale[..., None] * (
        sn[..., None] * lx + cs[..., None] * ly
    )
    lsn, lcs = q["anchor"][..., 2], q["anchor"][..., 3]
    gsn = sn[..., None] * lcs + cs[..., None] * lsn
    gcs = cs[..., None] * lcs - sn[..., None] * lsn
    anchor = torch.stack((gx, gy, gsn, gcs), -1)

    style = torch.cat((
        q["base_style"][..., :1] * scale[..., None, None],
        q["base_style"][..., 1:],
    ), -1)
    style_delta = torch.cat((
        q["style_delta"][..., :1] * scale[..., None, None, None],
        q["style_delta"][..., 1:],
    ), -1)
    segment = torch.cat((
        q["segment"][..., :1] * scale[..., None, None, None],
        q["segment"][..., 1:2] / scale[..., None, None, None],
    ), -1)
    base_kappa = q["base_kappa"] / scale[..., None]
    curvature_jump = (
        q["curvature_jump"] / scale[..., None, None]
    )
    stroke_mask = (
        q["stroke_mask"] & packed.region_mask[..., None]
    )
    segment_mask = (
        q["segment_mask"] & stroke_mask[..., None]
    )

    def flat(value):
        return value.reshape(B, R * K, *value.shape[3:])

    return PackedStatefulStrokes(
        flat(anchor),
        flat(base_kappa),
        flat(style),
        flat(segment),
        flat(curvature_jump),
        flat(q["curvature_change"]),
        flat(style_delta),
        flat(q["style_change"]),
        flat(stroke_mask),
        flat(segment_mask),
        flat(q["counts"]),
    )


def render_region_programs(packed: PackedRegionPrograms, **kwargs):
    return render_packed_stateful(regions_to_global(packed), **kwargs)


class RegionASTAutoencoder(nn.Module):
    """Typed ROOT -> REGION -> local STROKE AST autoencoder."""

    def __init__(
        self, d_model=128, n_heads=4, n_layers=2,
        max_regions=24, max_strokes=4, max_segments=8,
    ):
        super().__init__()
        self.max_regions = max_regions
        self.max_strokes = max_strokes
        self.max_segments = max_segments
        self.local_encoder = StatefulASTEncoder(
            d_model, n_heads, n_layers, max_strokes, max_segments
        )
        self.local_decoder = StatefulASTDecoder(
            d_model, max_strokes, max_segments
        )
        self.frame_in = nn.Sequential(
            nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.region_pos = nn.Parameter(
            torch.randn(1, max_regions, d_model) * 0.02
        )
        self.root = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.scene = nn.TransformerEncoder(layer, max(1, n_layers))
        self.region_fuse = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.present = nn.Linear(d_model, 1)
        self.frame_decode = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
            nn.GELU(), nn.Linear(2 * d_model, 5),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, packed: PackedRegionPrograms) -> RegionASTOutput:
        B, R = packed.batch_size, packed.max_regions
        local_latent = self.local_encoder(packed.local)
        content = local_latent.root.reshape(B, R, -1)
        frame_feature = torch.cat((
            packed.frame[..., :2],
            packed.frame[..., 2:4],
            packed.frame[..., 4:5] / 1.5,
        ), -1)
        frame = self.frame_in(frame_feature)
        region = (
            content + frame
            + self.region_fuse(torch.cat((content, frame), -1))
            + self.region_pos[:, :R]
        )
        tree = torch.cat((self.root.expand(B, -1, -1), region), 1)
        padding = torch.cat((
            torch.zeros(B, 1, dtype=torch.bool, device=region.device),
            ~packed.region_mask,
        ), 1)
        tree = self.scene(tree, src_key_padding_mask=padding)
        scene_latent = self.norm(tree[:, 0])
        region_latent = self.norm(tree[:, 1:])

        raw = self.frame_decode(region_latent)
        decoded_frame = torch.cat((
            raw[..., :2].sigmoid(),
            F.normalize(raw[..., 2:4], dim=-1, eps=1e-4),
            -3.0 + 2.8 * raw[..., 4:5].sigmoid(),
        ), -1)
        return RegionASTOutput(
            self.present(region_latent).squeeze(-1),
            decoded_frame,
            self.local_decoder(local_latent),
            region_latent,
            scene_latent,
        )


def region_output_to_packed(
    out: RegionASTOutput,
    template: PackedRegionPrograms,
    soft_structure: bool = False,
    set_structure: bool = False,
) -> PackedRegionPrograms:
    B, R = template.batch_size, template.max_regions
    local = output_to_packed(out.local, soft_structure=soft_structure)
    if soft_structure:
        region_mask = torch.ones(
            B, R, dtype=torch.bool, device=out.frame.device
        )
        # Feed region STOP probability to raster alpha.
        q = _reshape_local(local, B, R)
        probability = out.region_present_logits.sigmoid()
        base_style = torch.cat((
            q["base_style"][..., :4],
            q["base_style"][..., 4:5] * probability[..., None, None],
        ), -1)
        local = PackedStatefulStrokes(
            q["anchor"].reshape(B * R, *q["anchor"].shape[2:]),
            q["base_kappa"].reshape(B * R, *q["base_kappa"].shape[2:]),
            base_style.reshape(B * R, *base_style.shape[2:]),
            q["segment"].reshape(B * R, *q["segment"].shape[2:]),
            q["curvature_jump"].reshape(
                B * R, *q["curvature_jump"].shape[2:]
            ),
            q["curvature_change"].reshape(
                B * R, *q["curvature_change"].shape[2:]
            ),
            q["style_delta"].reshape(B * R, *q["style_delta"].shape[2:]),
            q["style_change"].reshape(
                B * R, *q["style_change"].shape[2:]
            ),
            q["stroke_mask"].reshape(B * R, *q["stroke_mask"].shape[2:]),
            q["segment_mask"].reshape(
                B * R, *q["segment_mask"].shape[2:]
            ),
            q["counts"].reshape(B * R, *q["counts"].shape[2:]),
        )
    else:
        region_mask = out.region_present_logits >= 0
        if not set_structure:
            region_mask = (
                region_mask
                & region_mask.to(torch.int64).cumprod(-1).bool()
            )
        # Set-mode keeps every independently present query active.
    return PackedRegionPrograms(
        out.frame, region_mask, region_mask.sum(-1),
        local, B, R,
    )


def region_losses(
    out: RegionASTOutput, target: PackedRegionPrograms,
    matching: bool = False,
):
    assignments = (
        region_hungarian_indices(out, target) if matching else None
    )
    layout, layout_terms = region_layout_loss(
        out, target, assignments=assignments
    )
    local, local_terms = region_local_loss(
        out, target, assignments=assignments
    )
    # A REGION frame moves every child Stroke.  Treating its four geometric
    # values like one more leaf attribute lets the much larger local program
    # dominate and produces correct motifs piled at the canvas centre.
    # Weight composition first; local detail is only meaningful in its frame.
    total = layout + local
    return total, {
        **layout_terms,
        **{f"local_{key}": value for key, value in local_terms.items()},
    }


def region_layout_loss(
    out: RegionASTOutput, target: PackedRegionPrograms,
    assignments=None,
):
    """Loss for the global REGION assembly branch only."""
    dtype = out.frame.dtype
    if assignments is None:
        assignments = [(
            torch.nonzero(target.region_mask[b], as_tuple=False).flatten(),
            torch.nonzero(target.region_mask[b], as_tuple=False).flatten(),
        ) for b in range(target.batch_size)]
    labels = torch.zeros_like(out.region_present_logits)
    pred_rows, target_cols = [], []
    for b, (rows, cols) in enumerate(assignments):
        if rows.numel():
            labels[b, rows] = 1
            pred_rows.append(out.frame[b, rows])
            target_cols.append(target.frame[b, cols])
    present = F.binary_cross_entropy_with_logits(
        out.region_present_logits, labels.to(dtype)
    )
    pred_frame = torch.cat(pred_rows) if pred_rows else out.frame[:0].reshape(0, 5)
    true_frame = torch.cat(target_cols) if target_cols else out.frame[:0].reshape(0, 5)
    xy = ((pred_frame[..., :2] - true_frame[..., :2]) / 0.3)
    frame_xy = xy.square().mean() if xy.numel() else out.frame.sum() * 0
    frame_angle = (
        1.0 - (pred_frame[..., 2:4] * true_frame[..., 2:4]).sum(-1)
    ).mean() if pred_frame.numel() else out.frame.sum() * 0
    frame_scale = (
        pred_frame[..., 4] - true_frame[..., 4]
    ).square().mean() if pred_frame.numel() else out.frame.sum() * 0
    total = 2 * present + 10 * frame_xy + 2 * frame_angle + 3 * frame_scale
    return total, {
        "region_present": present.detach(),
        "region_xy": frame_xy.detach(),
        "region_angle": frame_angle.detach(),
        "region_scale": frame_scale.detach(),
    }


def region_local_loss(
    out: RegionASTOutput, target: PackedRegionPrograms, assignments=None,
):
    """Loss for local Stroke/segment programs inside target REGIONs."""
    if assignments is None:
        assignments = [(
            torch.nonzero(target.region_mask[b], as_tuple=False).flatten(),
            torch.nonzero(target.region_mask[b], as_tuple=False).flatten(),
        ) for b in range(target.batch_size)]
    pred_ids, true_ids = [], []
    R = target.max_regions
    for b, (rows, cols) in enumerate(assignments):
        pred_ids.extend((b * R + rows).tolist())
        true_ids.extend((b * R + cols).tolist())
    pred_ids = torch.as_tensor(pred_ids, dtype=torch.long, device=out.frame.device)
    true_ids = torch.as_tensor(true_ids, dtype=torch.long, device=out.frame.device)
    local_out = StatefulOutput(**{
        key: value[pred_ids] for key, value in vars(out.local).items()
    })
    local_target = PackedStatefulStrokes(**{
        key: value[true_ids] for key, value in vars(target.local).items()
    })
    return stateful_losses(local_out, local_target)
