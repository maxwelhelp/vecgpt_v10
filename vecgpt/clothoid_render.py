"""Memory-bounded differentiable rasterization of clothoid strokes.

Geometry is sampled with the fixed Gauss-Legendre implementation in
``vecgpt.clothoid``.  Coverage is computed from point-to-*segment* distance;
pixels are processed in chunks so memory does not scale as one monolithic
``B x H x W x strokes x curve_samples`` tensor.
"""

from __future__ import annotations

import torch

from vecgpt.clothoid import clothoid_points
from vecgpt.render import pixel_grid


def _segment_distance_sq(
    pixels: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Squared pixel-to-segment distances.

    Args:
        pixels: ``[C,2]``.
        a, b: ``[B,K,M,2]`` segment endpoints.
    Returns:
        ``[B,K,C,M]``.
    """
    ab = b - a
    ab2 = ab.square().sum(-1).clamp_min(1e-12)
    p = pixels[None, None, :, None, :]
    av = a[:, :, None, :, :]
    abv = ab[:, :, None, :, :]
    t = ((p - av) * abv).sum(-1) / ab2[:, :, None, :]
    closest = av + t.clamp(0.0, 1.0)[..., None] * abv
    return (p - closest).square().sum(-1)


def _reduce_curve_distance(
    distance_sq: torch.Tensor,
    softmin_px: float,
    size: int,
) -> torch.Tensor:
    """Reduce sampled segment distances to one differentiable curve distance."""
    if softmin_px <= 0:
        return distance_sq.amin(-1).clamp_min(1e-12).sqrt()
    temperature = (softmin_px / size) ** 2
    # A weighted average is a smooth approximation from above and avoids the
    # negative-distance bias of -tau*logsumexp when many segments are present.
    weights = torch.softmax(-distance_sq / temperature, dim=-1)
    return (weights * distance_sq).sum(-1).clamp_min(1e-12).sqrt()


def render_clothoid_batch(
    anchor: torch.Tensor,
    kappa0: torch.Tensor,
    length: torch.Tensor,
    delta_kappa: torch.Tensor,
    width: torch.Tensor,
    rgba: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    size: int = 64,
    curve_samples: int = 32,
    softness_px: float = 1.0,
    distance_softmin_px: float = 0.0,
    background: float = 1.0,
    pixel_chunk: int = 1024,
) -> torch.Tensor:
    """Render one clothoid segment per painter-ordered Stroke.

    Shapes are ``anchor=[B,K,3]``, scalar parameters ``[B,K]`` and
    ``rgba=[B,K,4]``.  The result is ``[B,size,size,3]``.  All operations are
    differentiable with respect to geometry, width, colour and alpha.
    """
    if anchor.ndim != 3 or anchor.shape[-1] != 3:
        raise ValueError("anchor must have shape [B,K,3]")
    B, K, _ = anchor.shape
    expected = (B, K)
    for name, value in (
        ("kappa0", kappa0),
        ("length", length),
        ("delta_kappa", delta_kappa),
        ("width", width),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if rgba.shape != (B, K, 4):
        raise ValueError(f"rgba must have shape {(B, K, 4)}")
    if curve_samples < 1 or pixel_chunk < 1:
        raise ValueError("curve_samples and pixel_chunk must be positive")

    valid = (
        torch.ones(expected, dtype=torch.bool, device=anchor.device)
        if valid is None else valid.to(device=anchor.device, dtype=torch.bool)
    )
    if valid.shape != expected:
        raise ValueError(f"valid must have shape {expected}")

    points = clothoid_points(
        anchor, kappa0, length, delta_kappa, curve_samples
    )
    a, b = points[..., :-1, :], points[..., 1:, :]
    pixels = pixel_grid(size, anchor.device, anchor.dtype)
    soft = softness_px / size
    chunks = []
    for start in range(0, pixels.shape[0], pixel_chunk):
        p = pixels[start : start + pixel_chunk]
        distance_sq = _segment_distance_sq(p, a, b)
        distance = _reduce_curve_distance(
            distance_sq, distance_softmin_px, size
        )
        coverage = (
            0.5 + (width[:, :, None] / 2.0 - distance) / soft
        ).clamp(0.0, 1.0)
        chunks.append(coverage * valid[:, :, None].to(anchor.dtype))
    coverage = torch.cat(chunks, dim=-1)

    image = torch.full(
        (B, size * size, 3),
        float(background),
        device=anchor.device,
        dtype=anchor.dtype,
    )
    for stroke in range(K):
        alpha = (
            coverage[:, stroke]
            * rgba[:, stroke, 3, None].clamp(0.0, 1.0)
        )[..., None]
        color = rgba[:, stroke, :3].clamp(0.0, 1.0)[:, None, :]
        image = image * (1.0 - alpha) + color * alpha
    return image.view(B, size, size, 3)

