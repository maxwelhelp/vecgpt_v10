"""Differentiable Euler-spiral (clothoid) geometry.

Curvature varies linearly with arc length:

    kappa(s) = kappa0 + alpha*s
    theta(s) = theta0 + kappa0*s + 0.5*alpha*s^2

The x/y integrals are evaluated with fixed Gauss-Legendre quadrature.  Fixed
nodes keep the operation vectorised and differentiable in every parameter,
without adding a SciPy/Fresnel dependency.
"""

from __future__ import annotations

import torch

from vecgpt.scene import arc_step


_GL8_X = (
    -0.9602898564975363, -0.7966664774136267,
    -0.5255324099163290, -0.1834346424956498,
     0.1834346424956498,  0.5255324099163290,
     0.7966664774136267,  0.9602898564975363,
)
_GL8_W = (
    0.1012285362903763, 0.2223810344533745,
    0.3137066458778873, 0.3626837833783620,
    0.3626837833783620, 0.3137066458778873,
    0.2223810344533745, 0.1012285362903763,
)


def clothoid_points(
    anchor: torch.Tensor,
    kappa0: torch.Tensor,
    length: torch.Tensor,
    delta_kappa: torch.Tensor,
    samples: int = 32,
) -> torch.Tensor:
    """Return ``[..., samples+1, 2]`` points on one clothoid segment."""
    dtype, device = anchor.dtype, anchor.device
    shape = torch.broadcast_shapes(
        anchor.shape[:-1], kappa0.shape, length.shape, delta_kappa.shape
    )
    a = anchor.expand(*shape, 3)
    k0 = kappa0.expand(shape)
    L = length.expand(shape)
    dk = delta_kappa.expand(shape)
    t = torch.linspace(0, 1, samples + 1, device=device, dtype=dtype)
    s = L[..., None] * t
    nodes = torch.tensor(_GL8_X, device=device, dtype=dtype)
    weights = torch.tensor(_GL8_W, device=device, dtype=dtype)
    # Map [-1,1] quadrature nodes to [0,s] independently for every sample.
    u = 0.5 * s[..., None] * (nodes + 1)
    alpha = dk / L.clamp_min(1e-7)
    theta = (
        a[..., 2, None, None]
        + k0[..., None, None] * u
        + 0.5 * alpha[..., None, None] * u.square()
    )
    scale = 0.5 * s
    dx = scale * (theta.cos() * weights).sum(-1)
    dy = scale * (theta.sin() * weights).sum(-1)
    return torch.stack((
        a[..., 0, None] + dx,
        a[..., 1, None] + dy,
    ), -1)


def clothoid_end_state(
    anchor: torch.Tensor,
    kappa0: torch.Tensor,
    length: torch.Tensor,
    delta_kappa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final ``(x,y,theta)`` and curvature."""
    xy = clothoid_points(anchor, kappa0, length, delta_kappa, 1)[..., -1, :]
    theta = (
        anchor[..., 2] + kappa0 * length
        + 0.5 * delta_kappa * length
    )
    return torch.cat((xy, theta[..., None]), -1), kappa0 + delta_kappa


def constant_arc_approximation(
    anchor: torch.Tensor,
    kappa0: torch.Tensor,
    length: torch.Tensor,
    delta_kappa: torch.Tensor,
    n_arcs: int,
    samples_per_arc: int = 8,
) -> torch.Tensor:
    """Approximate one clothoid with midpoint-curvature circular arcs."""
    h = length / n_arcs
    alpha = delta_kappa / length.clamp_min(1e-7)
    state = anchor
    pieces = []
    t = torch.linspace(
        0, 1, samples_per_arc + 1,
        device=anchor.device, dtype=anchor.dtype,
    )
    for j in range(n_arcs):
        kmid = kappa0 + alpha * h * (j + 0.5)
        s = h[..., None] * t
        x, y, _ = arc_step(
            state[..., 0, None], state[..., 1, None],
            state[..., 2, None], kmid[..., None], s,
        )
        part = torch.stack((x, y), -1)
        pieces.append(part if j == 0 else part[..., 1:, :])
        ex, ey, eth = arc_step(
            state[..., 0], state[..., 1], state[..., 2], kmid, h
        )
        state = torch.stack((ex, ey, eth), -1)
    return torch.cat(pieces, -2)
