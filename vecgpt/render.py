"""Batched SDF renderer.

The old renderer was a Python loop over lines, then over segments, then a
[H*W, P] distance field per segment - a batch of 8 scenes meant 8 separate
forward passes. Measured: 17.8 scenes/s. Here the whole batch's segments
are padded into one tensor and the distance field is computed in a single
op, so a P40 runs thousands of scenes/s and the batch dimension is real.

Compositing still walks layers in order (painter's algorithm is inherently
sequential) but each step is one batched image op, and depth is only the
max segment count, not batch x segments.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vecgpt.scene import S_B, S_G, S_KAPPA, S_LEN, S_R, S_WIDTH, Stroke, chain_points


def pixel_grid(size: int, device, dtype) -> torch.Tensor:
    c = (torch.arange(size, device=device, dtype=dtype) + 0.5) / size
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack((xx, yy), -1).reshape(-1, 2)  # [H*W, 2]


def _pack(scenes: list[list[Stroke]], per_seg: int, device, dtype):
    """Pack geometry and RGBA style.

    The canonical Stroke tensor historically contained six columns
    ``(length, curvature, width, r, g, b)``.  A seventh, optional alpha
    column is accepted by the continuous decoder.  Old data therefore keeps
    rendering identically (implicit alpha=1), while soft structural
    existence can be differentiated through alpha without inserting the
    target segment count into the renderer.
    """
    pts_all, sty_all = [], []
    for strokes in scenes:
        pts, sty = [], []
        for st in strokes:
            a = st.anchor.to(device=device, dtype=dtype)
            g = st.segs.to(device=device, dtype=dtype)
            pts.append(chain_points(a, g, per_seg))  # [S, P, 2]
            rgba = torch.ones(g.shape[0], 5, device=device, dtype=dtype)
            rgba[:, :4] = g[:, [S_WIDTH, S_R, S_G, S_B]]
            if g.shape[1] > S_B + 1:
                rgba[:, 4] = g[:, S_B + 1]
            sty.append(rgba)
        if pts:
            pts_all.append(torch.cat(pts, 0))
            sty_all.append(torch.cat(sty, 0))
        else:  # empty scene -> a single degenerate, fully transparent stub
            pts_all.append(torch.zeros(1, per_seg + 1, 2, device=device, dtype=dtype))
            sty_all.append(torch.zeros(1, 5, device=device, dtype=dtype))

    B = len(pts_all)
    S = max(p.shape[0] for p in pts_all)
    P = per_seg + 1
    points = torch.zeros(B, S, P, 2, device=device, dtype=dtype)
    style = torch.zeros(B, S, 5, device=device, dtype=dtype)
    valid = torch.zeros(B, S, dtype=torch.bool, device=device)
    for i, (p, s) in enumerate(zip(pts_all, sty_all)):
        n = p.shape[0]
        points[i, :n], style[i, :n], valid[i, :n] = p, s, True
    return points, style, valid


def render_batch(
    scenes: list[list[Stroke]],
    size: int = 64,
    softness_px: float = 1.0,
    per_seg: int = 12,
    background: float = 1.0,
    device=None,
    dtype=torch.float32,
    chunk: int = 6,
) -> torch.Tensor:
    """-> [B, size, size, 3]. Differentiable in every stroke parameter."""
    device = device or (scenes[0][0].anchor.device if scenes and scenes[0] else torch.device("cpu"))
    out = []
    for i in range(0, len(scenes), chunk):
        out.append(
            _render_chunk(scenes[i : i + chunk], size, softness_px, per_seg, background, device, dtype)
        )
    return torch.cat(out, 0)


def _render_chunk(scenes, size, softness_px, per_seg, background, device, dtype):
    points, style, valid = _pack(scenes, per_seg, device, dtype)
    B, S, P, _ = points.shape
    grid = pixel_grid(size, device, dtype)  # [N, 2]
    N = grid.shape[0]
    soft = softness_px / size

    a = points[:, :, :-1]  # [B,S,P-1,2]
    b = points[:, :, 1:]
    ab = b - a
    ab2 = (ab * ab).sum(-1).clamp_min(1e-12)  # [B,S,P-1]

    g = grid.view(1, 1, N, 1, 2)
    t = ((g - a.unsqueeze(2)) * ab.unsqueeze(2)).sum(-1) / ab2.unsqueeze(2)  # [B,S,N,P-1]
    t = t.clamp(0, 1)
    closest = a.unsqueeze(2) + t.unsqueeze(-1) * ab.unsqueeze(2)
    dist = (g - closest).pow(2).sum(-1).clamp_min(1e-12).sqrt().min(-1).values  # [B,S,N]

    width = style[..., 0].unsqueeze(-1)  # [B,S,1]
    # Analytic antialias ramp, NOT a sigmoid.
    #
    # A sigmoid of scale ~1 px never saturates for a 1-2 px stroke: the
    # centre of a 1.8 px stroke reached only 0.7 coverage, so its colour
    # came out blended with the background (true R=0.758 rendered as
    # 0.818) by an amount that depends on sub-pixel alignment. Colour was
    # then literally not recoverable from the image, and measured: a
    # direct linear probe on the encoder could not beat the uniform prior
    # on the colour bin at all. The ramp is exactly 1.0 in the interior,
    # so an interior pixel IS the stroke's colour.
    coverage = (0.5 + (width / 2 - dist) / soft).clamp(0.0, 1.0)  # [B,S,N]
    coverage = coverage * valid.unsqueeze(-1).to(dtype)

    img = torch.full((B, N, 3), float(background), device=device, dtype=dtype)
    for s in range(S):  # painter's algorithm, one batched op per layer
        al = (
            coverage[:, s] * style[:, s, 4, None].clamp(0.0, 1.0)
        ).unsqueeze(-1)  # [B,N,1]
        img = img * (1 - al) + style[:, s, 1:4].unsqueeze(1) * al
    return img.view(B, size, size, 3)


def render_one(strokes: list[Stroke], size: int = 64, **kw) -> torch.Tensor:
    return render_batch([strokes], size=size, **kw)[0]


def ink_map(img: torch.Tensor, background: float = 1.0) -> torch.Tensor:
    """[B,H,W,3] -> [B,H,W] in [0,1]: how far this pixel is fromthe background.

    Max deviation across channels, so a stroke that only differs from the
    background in one channel still registers.
    """
    return (img - background).abs().amax(-1).clamp(0, 1)


def _ink_moments(ink: torch.Tensor) -> torch.Tensor:
    """Global differentiable shape statistics for sparse line drawings.

    A pixelwise SDF has no attraction gradient when two thin strokes are far
    apart.  Moments do: centroid moves the prediction toward the target,
    covariance teaches extent/orientation, and third moments distinguish
    asymmetric bends.  These are generic geometry coordinates, not named
    primitives or semantic supervision.
    """
    B, H, W = ink.shape
    dtype, device = ink.dtype, ink.device
    x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mass = ink.sum((1, 2)).clamp_min(1e-4)
    p = ink / mass[:, None, None]
    mx = (p * xx).sum((1, 2))
    my = (p * yy).sum((1, 2))
    dx, dy = xx[None] - mx[:, None, None], yy[None] - my[:, None, None]
    return torch.stack((
        mass / float(H * W),
        mx, my,
        (p * dx.square()).sum((1, 2)),
        (p * dy.square()).sum((1, 2)),
        (p * dx * dy).sum((1, 2)),
        (p * dx.pow(3)).sum((1, 2)),
        (p * dy.pow(3)).sum((1, 2)),
        (p * dx.square() * dy).sum((1, 2)),
        (p * dx * dy.square()).sum((1, 2)),
    ), -1)


def foreground_render_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Foreground-normalized multiscale loss for sparse vector strokes.

    Plain mean L1 rewards the 98% white background and made the supposedly
    closed raster loop contribute ~0.007 to a ~3.8 token loss. Every term
    here is normalized per drawing, so a one-pixel line is not 100x less
    important than the canvas surrounding it.
    """
    pred_ink, target_ink = ink_map(pred), ink_map(target)
    dims = (1, 2)
    target_mass = target_ink.sum(dims).clamp_min(1.0)

    coverage_terms = []
    dice_terms = []
    for sigma in (0.0, 1.0, 2.0, 4.0):
        p = pred_ink if sigma == 0 else _gauss_blur(pred_ink, sigma)
        t = target_ink if sigma == 0 else _gauss_blur(target_ink, sigma)
        coverage_terms.append((p - t).abs().sum(dims) / target_mass)
        inter = (p * t).sum(dims)
        dice_terms.append(
            1.0 - (2.0 * inter + 1e-4)
            / (p.square().sum(dims) + t.square().sum(dims) + 1e-4)
        )
    coverage_per = torch.stack(coverage_terms, -1).mean(-1)
    dice_per = torch.stack(dice_terms, -1).mean(-1)

    pm, tm = _ink_moments(pred_ink), _ink_moments(target_ink)
    # Centroid and covariance carry the long-range geometry gradient; mass
    # and third moments are useful but should not dominate it.
    moment_weights = pred.new_tensor(
        [1.0, 3.0, 3.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5]
    )
    moments = (
        F.smooth_l1_loss(pm, tm, reduction="none") * moment_weights
    ).mean()
    centroid_dist2 = (pm[:, 1:3] - tm[:, 1:3]).square().sum(-1)
    # Fine overlap losses are excellent refiners but, while strokes are
    # disjoint, they mostly teach the prediction to disappear. Activate them
    # only after the global geometry terms have brought both drawings close.
    near = torch.exp(-4.0 * centroid_dist2).detach()
    coverage = (near * coverage_per).mean()
    dice = (near * dice_per).mean()

    def probability(x):
        return x / x.sum(dims, keepdim=True).clamp_min(1e-4)

    # Gaussian-kernel MMD is zero for identical ink distributions and has a
    # canvas-wide attraction gradient for disjoint strokes. Dividing by the
    # target self-energy keeps its scale independent of line thickness.
    # Eight-by-eight is enough for this deliberately coarse, canvas-wide
    # attraction term.  Computing a sigma~32 convolution at full 64x64
    # resolution made every training step needlessly expensive.
    coarse_size = 8
    pred_coarse = F.adaptive_avg_pool2d(
        pred_ink[:, None], (coarse_size, coarse_size)
    ).squeeze(1)
    target_coarse = F.adaptive_avg_pool2d(
        target_ink[:, None], (coarse_size, coarse_size)
    ).squeeze(1)
    pp, tp = probability(pred_coarse), probability(target_coarse)
    sigma_global = coarse_size / 2
    p_blur = _gauss_blur(pp, sigma_global)
    t_blur = _gauss_blur(tp, sigma_global)
    e_pp = (pp * p_blur).sum(dims)
    e_tt = (tp * t_blur).sum(dims)
    e_pt = (pp * t_blur).sum(dims)
    transport = ((e_pp + e_tt - 2.0 * e_pt) / e_tt.clamp_min(1e-6)).mean()

    def mean_color(img, ink):
        mass = ink.sum(dims).clamp_min(1e-4)
        return (img * ink[..., None]).sum((1, 2)) / mass[:, None]

    color = F.smooth_l1_loss(
        mean_color(pred, pred_ink), mean_color(target, target_ink)
    )
    total = transport + coverage + dice + 2.0 * moments + 0.5 * color
    return total, {
        "transport": transport.detach(),
        "coverage": coverage.detach(),
        "dice": dice.detach(),
        "moments": moments.detach(),
        "color": color.detach(),
    }


def image_iou(pred: torch.Tensor, target: torch.Tensor, background: float = 1.0) -> torch.Tensor:
    """Threshold-free soft IoU on the ink map. [B,H,W,3] -> [B].

    A hard threshold is the wrong tool here: a 1.5 px stroke never reaches
    full coverage (softness is a similar scale to the stroke half-width),
    so any fixed cut-off silently reports "no ink at all" for pale or thin
    strokes and the metric becomes noise on a handful of pixels.
    """
    a = ink_map(pred, background).flatten(1)
    b = ink_map(target, background).flatten(1)
    inter = torch.minimum(a, b).sum(-1)
    union = torch.maximum(a, b).sum(-1)
    return torch.where(union > 1e-6, inter / union.clamp_min(1e-6), torch.ones_like(union))


def save_grid(rows, path, gap: int = 4, bg: float = 0.85):
    """Save rows of [H,W,3] tensors as one PNG. rows[0] on top."""
    import os

    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cell = rows[0][0].shape[0]
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    W = n_cols * cell + (n_cols + 1) * gap
    H = n_rows * cell + (n_rows + 1) * gap
    canvas = torch.full((H, W, 3), float(bg))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            y, x = gap + r * (cell + gap), gap + c * (cell + gap)
            canvas[y : y + cell, x : x + cell] = img.detach().float().cpu().clamp(0, 1)
    Image.fromarray((canvas * 255).byte().numpy()).save(path)
    return path


def _gauss_blur(m: torch.Tensor, sigma: float) -> torch.Tensor:
    import torch.nn.functional as F

    k = int(sigma * 3) | 1
    x = torch.arange(k, device=m.device, dtype=m.dtype) - k // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    m = m.unsqueeze(1)
    m = F.conv2d(m, g.view(1, 1, 1, -1), padding=(0, k // 2))
    m = F.conv2d(m, g.view(1, 1, -1, 1), padding=(k // 2, 0))
    return m.squeeze(1)


def image_iou_shape(pred, target, sigma_px: float = 2.0, background: float = 1.0):
    """Soft IoU on BLURRED ink maps: a shape metric, not a registration one.

    Strict IoU on 1-3 px strokes is dominated by sub-pixel placement, and it
    mis-ORDERS results. Measured on stage 4:

        offset   strict   blurred(2px)
        0 px     1.000    1.000
        1 px     0.479    0.780
        2 px     0.245    0.612
        4 px     0.114    0.391
        wrong shape, right place
                 0.169    0.263

    A geometrically perfect scene shifted 4 px scores 0.114 strict - WORSE
    than a scene with randomised curvature in the right place (0.169). Any
    ranking built on that number is partly backwards. Blurring at ~2 px
    keeps wrong shapes down at 0.26 while letting a correct-but-nudged
    shape score 0.6-0.8, which is what "did it draw the right thing" means.

    Report both: strict says how well it registers, this says what it drew.
    """
    a = _gauss_blur(ink_map(pred, background), sigma_px).flatten(1)
    b = _gauss_blur(ink_map(target, background), sigma_px).flatten(1)
    inter = torch.minimum(a, b).sum(-1)
    union = torch.maximum(a, b).sum(-1)
    return torch.where(union > 1e-6, inter / union.clamp_min(1e-6), torch.ones_like(union))
