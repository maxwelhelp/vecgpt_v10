"""Vector-only geometry features for the hybrid stroke model.

The UDF signature here is deliberately *not* an output image.  It is a
small vector of distances from fixed 2-D probes to the sampled clothoid.  It
keeps the useful geometric inductive bias of StrokeFusion while preserving a
fully vector-native representation.
"""

from __future__ import annotations

import torch

from vecgpt.clothoid import clothoid_points


def sample_clothoid_params(params: torch.Tensor, samples: int = 24) -> torch.Tensor:
    """Decode ``[..., 10]`` stroke parameters to sampled curve points.

    Parameter layout: ``x, y, theta, length, kappa, delta_kappa, width,
    r, g, b``.  Alpha and presence are carried separately.
    """
    if params.shape[-1] < 10:
        raise ValueError("stroke params need at least 10 values")
    anchor = params[..., :3]
    length = params[..., 3].clamp_min(1e-4)
    kappa = params[..., 4]
    delta = params[..., 5]
    return clothoid_points(anchor, kappa, length, delta, samples=samples)


def clothoid_bbox(params: torch.Tensor, samples: int = 24) -> torch.Tensor:
    """Return normalized center/size of each curve for the joint diffusion state."""
    pts = sample_clothoid_params(params, samples)
    mn, mx = pts.amin(-2), pts.amax(-2)
    center = (mn + mx) * 0.5
    size = (mx - mn).clamp_min(1e-4)
    return torch.cat((center, size), -1)


def place_clothoids(params: torch.Tensor, target_bbox: torch.Tensor) -> torch.Tensor:
    """Translate decoded local shapes to generated bbox centers.

    Only repositions strokes; does NOT rescale curvature/length.
    Scaling kappa/delta_kappa creates numerical explosions when the DDPM
    produces inconsistently sized latent-bbox pairs.
    """
    out = params.clone()
    old = clothoid_bbox(out)
    out[..., :2] = out[..., :2] + (target_bbox[..., :2] - old[..., :2])
    return out


def curve_udf_signature(
    params: torch.Tensor,
    probes: torch.Tensor | None = None,
    curve_samples: int = 24,
    softmin: float = 0.025,
) -> torch.Tensor:
    """Return a differentiable vector UDF signature for each stroke.

    ``probes`` are points in normalized canvas coordinates.  The result is
    ``[..., P]`` where each value is a soft distance from a probe to the
    clothoid.  Unlike a raster UDF this is not a pixel image and is used only
    as an auxiliary geometric descriptor.
    """
    pts = sample_clothoid_params(params, curve_samples)
    # Canonicalize the curve before measuring probes.  The signature describes
    # shape, not the accidental global position of the stroke.
    mn = pts.amin(dim=-2, keepdim=True)
    mx = pts.amax(dim=-2, keepdim=True)
    center = (mn + mx) * 0.5
    scale = (mx - mn).amax(dim=-1, keepdim=True).clamp_min(1e-4)
    pts = (pts - center) / scale + 0.5
    if probes is None:
        # A deterministic local grid gives a richer shape signature while
        # remaining a vector of distances, not an image tensor.
        g = torch.linspace(0.05, 0.95, 8, device=params.device, dtype=params.dtype)
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        probes = torch.stack((xx, yy), -1).reshape(-1, 2)
    d2 = (probes.to(params)[..., None, :] - pts[..., None, :, :]).square().sum(-1)
    if softmin > 0:
        tau = softmin ** 2
        return (-tau * torch.logsumexp(-d2 / tau, dim=-1)).clamp_min(0).sqrt()
    return d2.amin(-1).clamp_min(1e-10).sqrt()


def stroke_to_feature(params: torch.Tensor, udf_dim: int = 16) -> torch.Tensor:
    """Concatenate normalized vector parameters and geometric UDF features."""
    # Keep the feature dimensions stable when a custom probe count is used.
    probes = None
    if udf_dim != 16:
        side = max(2, int(round(udf_dim ** 0.5)))
        g = torch.linspace(0.08, 0.92, side, device=params.device, dtype=params.dtype)
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        probes = torch.stack((xx, yy), -1).reshape(-1, 2)[:udf_dim]
        if probes.shape[0] < udf_dim:
            probes = torch.cat((probes, probes[:1].expand(udf_dim - probes.shape[0], -1)), 0)
    udf = curve_udf_signature(params, probes=probes)
    return torch.cat((params, udf), dim=-1)
