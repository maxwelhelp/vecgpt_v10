"""Spatial raster memory and image-conditioned stateful AST decoding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vecgpt.continuous import ASTLatent
from vecgpt.region_ast import RegionASTOutput
from vecgpt.stateful_model import StatefulASTDecoder, StatefulOutput


class RasterPatchEncoder(nn.Module):
    """ViT-like patch memory; never collapses the image to one bottleneck."""

    def __init__(
        self, d_model=128, patch_size=4, n_heads=4, n_layers=3,
    ):
        super().__init__()
        self.patch_size = patch_size
        if patch_size != 4:
            raise ValueError("current overlapping stem expects patch_size=4")
        hidden = max(16, d_model // 2)
        self.patch = nn.Sequential(
            nn.Conv2d(4, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, d_model, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.foreground = nn.Parameter(torch.randn(d_model) * 0.02)
        self.coord = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.layers = nn.TransformerEncoder(layer, max(1, n_layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("image must be [B,H,W,3] or [B,3,H,W]")
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        if image.shape[1] != 3:
            raise ValueError("image must have three colour channels")
        ink = (1.0 - image).abs().amax(1, keepdim=True)
        # Sparse line art must not be numerically dominated by the constant
        # background.  Background becomes exactly zero; foreground retains
        # both coverage and colour. Sqrt expands low-contrast antialiased
        # pixels without introducing a semantic mask.
        saliency = ink.clamp_min(1e-6).sqrt()
        foreground_rgb = saliency * (image * 2.0 - 1.0)
        visual_input = torch.cat((foreground_rgb, saliency), 1)
        x = self.patch(visual_input)
        B, D, H, W = x.shape
        ink_patch = torch.nn.functional.adaptive_avg_pool2d(
            saliency, (H, W)
        )
        x = x + ink_patch * self.foreground[None, :, None, None]
        x = x.flatten(2).transpose(1, 2)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coordinates = torch.stack((xx, yy), -1).reshape(1, H * W, 2)
        x = x + self.coord(coordinates)
        x = self.norm(self.layers(x))
        # A generic foreground-weighted summary gives the decoder a strong
        # conditioning route from the first update, while all spatial patch
        # tokens remain available for precise cross-attention.
        weights = ink_patch.flatten(2).transpose(1, 2).clamp_min(1e-4)
        summary = (x * weights).sum(1) / weights.sum(1).clamp_min(1e-4)
        return torch.cat((summary[:, None, :], x), 1)


class RasterToStatefulAST(nn.Module):
    """Raster patch memory -> sequential Stroke/SEGMENT typed latents.

    ``max_strokes`` and ``max_segments`` are context lengths. Positions are
    ordered program tokens and the heads predict the prefix STOP decisions;
    they are not persistent semantic object slots.
    """

    def __init__(
        self, d_model=128, n_heads=4, encoder_layers=3,
        decoder_layers=2, patch_size=4,
        max_strokes=4, max_segments=8,
    ):
        super().__init__()
        self.max_strokes = max_strokes
        self.max_segments = max_segments
        self.visual = RasterPatchEncoder(
            d_model, patch_size, n_heads, encoder_layers
        )
        self.stroke_query = nn.Parameter(
            torch.randn(1, max_strokes, d_model) * 0.02
        )
        self.segment_query = nn.Parameter(
            torch.randn(1, 1, max_segments, d_model) * 0.02
        )
        stroke_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True,
            norm_first=True, activation="gelu",
        )
        segment_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.stroke_decoder = nn.TransformerDecoder(
            stroke_layer, max(1, decoder_layers)
        )
        self.segment_decoder = nn.TransformerDecoder(
            segment_layer, max(1, decoder_layers)
        )
        self.frame_type = nn.Parameter(torch.randn(d_model) * 0.02)
        self.style_type = nn.Parameter(torch.randn(d_model) * 0.02)
        self.root = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
        )
        self.ast_decoder = StatefulASTDecoder(
            d_model, max_strokes, max_segments
        )

    def latent(self, image: torch.Tensor) -> ASTLatent:
        memory = self.visual(image)
        B = memory.shape[0]
        summary = memory[:, 0]
        stroke_query = (
            self.stroke_query.expand(B, -1, -1) + summary[:, None, :]
        )
        causal = torch.triu(
            torch.ones(
                self.max_strokes, self.max_strokes,
                dtype=torch.bool, device=memory.device,
            ),
            diagonal=1,
        )
        stroke = self.stroke_decoder(
            stroke_query, memory, tgt_mask=causal
        )
        frame = stroke + self.frame_type
        style = stroke + self.style_type
        segment_query = (
            stroke[:, :, None, :]
            + self.segment_query[:, :, :self.max_segments]
        ).reshape(B, self.max_strokes * self.max_segments, -1)
        segment = self.segment_decoder(segment_query, memory)
        segment = segment.reshape(
            B, self.max_strokes, self.max_segments, -1
        )
        return ASTLatent(
            self.root(summary),
            stroke, frame, style, segment,
            torch.ones(
                B, self.max_strokes, dtype=torch.bool,
                device=memory.device,
            ),
            torch.ones(
                B, self.max_strokes, self.max_segments,
                dtype=torch.bool, device=memory.device,
            ),
        )

    def forward(self, image: torch.Tensor) -> StatefulOutput:
        return self.ast_decoder(self.latent(image))


class RasterToRegionAST(nn.Module):
    """Raster memory -> sequential REGION -> local stateful Stroke AST."""

    def __init__(
        self, d_model=128, n_heads=4, encoder_layers=3,
        decoder_layers=2, patch_size=4,
        max_regions=16, max_strokes=4, max_segments=8,
        unordered_regions=False, autoregressive_children=False,
    ):
        super().__init__()
        self.max_regions = max_regions
        self.max_strokes = max_strokes
        self.max_segments = max_segments
        self.unordered_regions = unordered_regions
        self.autoregressive_children = autoregressive_children
        # DAB-style spatial reference for each technical query. These are
        # coverage anchors, not semantic slots: a query is responsible for a
        # neighbourhood and may represent any learned part.
        cols = max(1, int(round(max_regions ** 0.5)))
        rows = (max_regions + cols - 1) // cols
        grid = torch.tensor([
            ((i % cols + 0.5) / cols, (i // cols + 0.5) / rows)
            for i in range(max_regions)
        ], dtype=torch.float32)
        self.anchor_xy = nn.Parameter(
            torch.logit(grid.clamp(0.05, 0.95))
        )
        self.anchor_log_scale = nn.Parameter(
            torch.full((max_regions, 1), -1.0)
        )
        self.anchor_pos = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.anchor_refine = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3)
        )
        self.visual = RasterPatchEncoder(
            d_model, patch_size, n_heads, encoder_layers
        )
        self.region_query = nn.Parameter(
            torch.randn(1, max_regions, d_model) * 0.02
        )
        self.stroke_query = nn.Parameter(
            torch.randn(1, 1, max_strokes, d_model) * 0.02
        )
        self.segment_query = nn.Parameter(
            torch.randn(1, 1, 1, max_segments, d_model) * 0.02
        )

        def decoder():
            layer = nn.TransformerDecoderLayer(
                d_model, n_heads, 4 * d_model, batch_first=True,
                norm_first=True, activation="gelu",
            )
            return nn.TransformerDecoder(
                layer, max(1, decoder_layers)
            )

        self.region_decoder = decoder()
        self.stroke_decoder = decoder()
        self.segment_decoder = decoder()
        # A recurrent state makes child order causal. The old Transformer
        # branch produced all local Stroke latents independently, allowing
        # symmetric children to average even after REGION matching.
        self.stroke_rnn = nn.GRUCell(2 * d_model, d_model)
        self.segment_rnn = nn.GRUCell(2 * d_model, d_model)
        crop_hidden = max(16, d_model // 2)
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(4, crop_hidden, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(crop_hidden, d_model, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(d_model),
        )
        self.crop_fuse = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.anchor_local = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.present = nn.Linear(d_model, 1)
        self.frame_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
            nn.GELU(), nn.Linear(2 * d_model, 5),
        )
        self.frame_type = nn.Parameter(torch.randn(d_model) * 0.02)
        self.style_type = nn.Parameter(torch.randn(d_model) * 0.02)
        self.local_decoder = StatefulASTDecoder(
            d_model, max_strokes, max_segments
        )
        self.root = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
        )

    @staticmethod
    def _foreground_input(image: torch.Tensor) -> torch.Tensor:
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        ink = (1.0 - image).abs().amax(1, keepdim=True)
        saliency = ink.clamp_min(1e-6).sqrt()
        return torch.cat((saliency * (image * 2.0 - 1.0), saliency), 1)

    def _region_crops(
        self, image: torch.Tensor, frame: torch.Tensor, size: int = 16,
    ) -> torch.Tensor:
        """Rotation/scale-normalized raster view for every REGION."""
        visual = self._foreground_input(image)
        B, R = frame.shape[:2]
        tx, ty = frame[..., 0], frame[..., 1]
        sn, cs = frame[..., 2], frame[..., 3]
        scale = frame[..., 4].exp().clamp(0.015, 1.2)
        affine = torch.stack((
            scale * cs, -scale * sn, 2.0 * tx - 1.0,
            scale * sn, scale * cs, 2.0 * ty - 1.0,
        ), -1).reshape(B * R, 2, 3)
        grid = F.affine_grid(
            affine, (B * R, 4, size, size), align_corners=False
        )
        repeated = visual[:, None].expand(
            B, R, *visual.shape[1:]
        ).reshape(B * R, *visual.shape[1:])
        return F.grid_sample(
            repeated, grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )

    def forward(
        self, image: torch.Tensor,
        teacher_frame: torch.Tensor | None = None,
    ) -> RegionASTOutput:
        memory = self.visual(image)
        B, D = memory.shape[0], memory.shape[-1]
        summary = memory[:, 0]
        anchor_xy = self.anchor_xy.sigmoid()
        anchor_scale = self.anchor_log_scale.exp().clamp(0.08, 1.0)
        anchor = torch.cat((anchor_xy, anchor_scale), -1)
        anchor_frame = torch.cat((
            anchor_xy[None].expand(B, -1, -1),
            torch.zeros(B, self.max_regions, 2, device=memory.device),
            self.anchor_log_scale[None].expand(B, -1, -1),
        ), -1)
        anchor_crop = self._region_crops(image, anchor_frame)
        anchor_feature = self.crop_encoder(anchor_crop).reshape(
            B, self.max_regions, D
        )
        region_query = (
            self.region_query.expand(B, -1, -1)
            + summary[:, None, :]
            + self.anchor_pos(anchor)[None]
            + self.anchor_local(anchor_feature)
        )
        if self.unordered_regions:
            region = self.region_decoder(region_query, memory)
        else:
            causal = torch.triu(
                torch.ones(
                    self.max_regions, self.max_regions,
                    dtype=torch.bool, device=memory.device,
                ), diagonal=1,
            )
            region = self.region_decoder(
                region_query, memory, tgt_mask=causal
            )
        raw_frame = self.frame_head(region)
        # Predict bounded offsets around the query's spatial reference. This
        # keeps a query tied to its neighbourhood while retaining refinement.
        delta = self.anchor_refine(region)
        base_xy = anchor_xy[None]
        base_log_scale = self.anchor_log_scale[None]
        frame = torch.cat((
            (base_xy + 0.35 * torch.tanh(delta[..., :2])).clamp(0.01, 0.99),
            F.normalize(raw_frame[..., 2:4], dim=-1, eps=1e-4),
            base_log_scale + 1.5 * torch.tanh(delta[..., 2:3]),
        ), -1)
        crop_frame = frame if teacher_frame is None else teacher_frame
        crop = self._region_crops(image, crop_frame)
        crop_feature = self.crop_encoder(crop).reshape(
            B, self.max_regions, D
        )
        region_local = region + self.crop_fuse(
            torch.cat((region, crop_feature), -1)
        )

        BR = B * self.max_regions
        if self.autoregressive_children:
            parent = region_local.reshape(BR, D)
            stroke_state = torch.zeros_like(parent)
            strokes = []
            for k in range(self.max_strokes):
                query = self.stroke_query[0, 0, k].expand(BR, -1)
                stroke_state = self.stroke_rnn(
                    torch.cat((parent, query), -1), stroke_state
                )
                strokes.append(stroke_state)
            stroke = torch.stack(strokes, 1).reshape(
                B, self.max_regions, self.max_strokes, D
            )
            stroke_flat = stroke.reshape(BR * self.max_strokes, D)
            segment_state = torch.zeros_like(stroke_flat)
            segments = []
            for s in range(self.max_segments):
                query = self.segment_query[0, 0, 0, s].expand(
                    BR * self.max_strokes, -1
                )
                segment_state = self.segment_rnn(
                    torch.cat((stroke_flat, query), -1), segment_state
                )
                segments.append(segment_state)
            segment = torch.stack(segments, 1).reshape(
                B, self.max_regions, self.max_strokes,
                self.max_segments, D,
            )
        else:
            stroke_query = (
                region_local[:, :, None, :]
                + self.stroke_query[:, :, :self.max_strokes]
            ).reshape(B, self.max_regions * self.max_strokes, D)
            stroke = self.stroke_decoder(
                stroke_query, memory
            ).reshape(B, self.max_regions, self.max_strokes, D)
            segment_query = (
                stroke[:, :, :, None, :]
                + self.segment_query[:, :, :, :self.max_segments]
            ).reshape(
                B, self.max_regions * self.max_strokes
                * self.max_segments, D
            )
            segment = self.segment_decoder(
                segment_query, memory
            ).reshape(
                B, self.max_regions, self.max_strokes,
                self.max_segments, D,
            )
        BR = B * self.max_regions
        local_latent = ASTLatent(
            region.reshape(BR, D),
            stroke.reshape(BR, self.max_strokes, D),
            (stroke + self.frame_type).reshape(
                BR, self.max_strokes, D
            ),
            (stroke + self.style_type).reshape(
                BR, self.max_strokes, D
            ),
            segment.reshape(
                BR, self.max_strokes, self.max_segments, D
            ),
            torch.ones(
                BR, self.max_strokes, dtype=torch.bool,
                device=memory.device,
            ),
            torch.ones(
                BR, self.max_strokes, self.max_segments,
                dtype=torch.bool, device=memory.device,
            ),
        )
        return RegionASTOutput(
            self.present(region).squeeze(-1),
            frame,
            self.local_decoder(local_latent),
            region_local,
            self.root(summary),
        )
