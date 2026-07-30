"""Flat stroke autoencoder and latent diffusion model.

This is the proposed VecGPT hybrid baseline: no semantic REGION slots, no
image encoder in the generative path, and no pixel diffusion.  A padded
stroke set is used only for tensor batching; ``presence`` decides how many
strokes exist.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

try:
    from .geometry import stroke_to_feature, sample_clothoid_params
except ImportError:  # direct execution from inside hybrid_stroke_diffusion/
    from geometry import stroke_to_feature, sample_clothoid_params


class SinusoidalTime(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        x = t.float()[:, None] * freq[None]
        emb = torch.cat((x.sin(), x.cos()), -1)
        return F.pad(emb, (0, self.dim - emb.shape[-1]))


class StrokeAutoencoder(nn.Module):
    """Encode/decode individual clothoid strokes in a shared latent space."""

    def __init__(self, latent_dim: int = 64, hidden: int = 128, udf_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.udf_dim = udf_dim
        self.in_proj = nn.Sequential(
            nn.Linear(10 + udf_dim, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim)
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, 10)
        )
        self.presence = nn.Linear(latent_dim, 1)

    def encode(self, params: torch.Tensor) -> torch.Tensor:
        return self.norm(self.in_proj(stroke_to_feature(params, self.udf_dim)))

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.dec(z)
        # Stable bounded parameterization.  Coordinates and colours are in
        # [0,1], angles/curvatures use bounded ranges, length is positive.
        out = torch.cat((
            raw[..., :2].sigmoid(),
            math.pi * torch.tanh(raw[..., 2:3]),
            0.9 * raw[..., 3:4].sigmoid(),
            12.0 * torch.tanh(raw[..., 4:5]),
            24.0 * torch.tanh(raw[..., 5:6]),
            # Toy corpus uses thin strokes (~0.01--0.045).  A broad 0--0.08
            # range makes width underweighted and produces visibly fat lines.
            0.005 + 0.045 * raw[..., 6:7].sigmoid(),
            raw[..., 7:10].sigmoid(),
        ), -1)
        return out, self.presence(z).squeeze(-1)

    def forward(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(params)
        recon, presence_logits = self.decode(z)
        return recon, presence_logits, z

    def loss(self, params: torch.Tensor, presence: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        recon, logits, z = self(params)
        mask = presence[..., None]
        # Compare fields in normalized units.  Real SVG fits have a much
        # wider curvature range than the toy generator.
        field_scales = params.new_tensor([1., 1., math.pi, .9, 12., 24., .05, 1., 1., 1.])
        field_weights = params.new_tensor([1., 1., 1., 1., 1., 1., 4., 1., 1., 1.])
        normalized_error = ((recon - params) / field_scales).square()
        geom = ((normalized_error * field_weights) * mask).sum() / mask.sum().clamp_min(1.0)
        # The decoder must learn to switch padded entries off.  This is not a
        # semantic slot: it is only variable-cardinality bookkeeping.
        stop = F.binary_cross_entropy_with_logits(logits, presence)
        feat_true = stroke_to_feature(params, self.udf_dim)
        feat_pred = stroke_to_feature(recon, self.udf_dim)
        udf = ((feat_pred[..., 10:] - feat_true[..., 10:]).square() * mask).sum() / mask.sum().clamp_min(1.0)
        target_curve = sample_clothoid_params(params, samples=16)
        pred_curve = sample_clothoid_params(recon, samples=16)
        curve = ((pred_curve - target_curve).square().mean(-1) * presence[..., None]).sum() / (mask.sum().clamp_min(1.0) * pred_curve.shape[-2])
        total = geom + 2.0 * curve + 0.25 * udf + 0.5 * stop
        return total, {"geom": geom.detach(), "curve": curve.detach(), "udf": udf.detach(), "presence": stop.detach(), "z": z.detach()}


class StrokeLatentDiffusion(nn.Module):
    """Transformer denoiser over padded stroke latents.

    Architecture follows StrokeFusion: plain self-attention Transformer
    over a fixed-length padded sequence, no separate count head.
    Cardinality is determined by the diffused presence flag (flag > 0).
    """

    def __init__(self, latent_dim: int = 64, model_dim: int = 128, layers: int = 4, heads: int = 4, cond_dim: int = 0, use_pos: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.in_proj = nn.Linear(latent_dim, model_dim)
        self.time = nn.Sequential(SinusoidalTime(model_dim), nn.Linear(model_dim, model_dim), nn.SiLU())
        self.cond = nn.Linear(cond_dim, model_dim) if cond_dim else None
        self.use_pos = use_pos
        self.pos = nn.Parameter(torch.randn(1, 256, model_dim) * 0.01)
        block = nn.TransformerEncoderLayer(model_dim, heads, 4 * model_dim, batch_first=True, norm_first=False, dropout=0.1, activation='gelu')
        self.net = nn.TransformerEncoder(block, layers)
        self.out = nn.Linear(model_dim, latent_dim)

    def forward(self, noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None):
        h = self.in_proj(noisy) + self.time(t)[:, None]
        if self.use_pos:
            h = h + self.pos[:, : noisy.shape[1]]
        if self.cond is not None and cond is not None:
            h = h + self.cond(cond)[:, None] * 2.0
        h = self.net(h)
        return self.out(h)


def diffusion_loss(denoiser: StrokeLatentDiffusion, clean: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
    """Standard DDPM noise prediction MSE — no masking.

    StrokeFusion uses plain MSE over all dimensions of all padded positions.
    This lets the model learn the full joint distribution including the
    presence flag (±0.5), without the training imbalance caused by masking.
    """
    b = clean.shape[0]
    t = torch.randint(0, 1000, (b,), device=clean.device)
    noise = torch.randn_like(clean)
    a = (1.0 - t.float() / 1000).clamp(0.01, 1.0)
    noisy = a[:, None, None].sqrt() * clean + (1.0 - a[:, None, None]).sqrt() * noise
    pred = denoiser(noisy, t, cond)
    return F.mse_loss(pred, noise)
