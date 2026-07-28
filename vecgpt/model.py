"""Encoder -> autoregressive decoder -> ONE masked softmax over one vocab.

Kept from the old design (these parts were not the problem):
  * arcs as the primitive, one token per parameter of a stroke;
  * axial RoPE with whole heads assigned to an axis, so a head gets its
    full head_dim of frequencies on the segment axis or the stroke axis
    rather than splitting channels three ways;
  * ~7M-parameter sizing, fp32 (P40's fp16 is 1/64 rate - AMP is a
    pessimisation there).

Replaced:
  * five bespoke output heads in four parameterisations -> one Linear to
    one vocabulary, logits masked to the legal field per position;
  * regression losses on multi-modal targets -> cross-entropy, which
    represents "either +k or -k" as two modes instead of averaging them
    into a straight line;
  * batch-of-1 python loop -> real [B, T] batching everywhere.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import vecgpt.schema as S_
import torch.nn.functional as F

from vecgpt.tokenizer import N_STATES, VOCAB, Walker, build_state_mask


# ------------------------------------------------------------------ RoPE


def rope_angles(pos: torch.Tensor, dim: int, base: float = 10000.0) -> torch.Tensor:
    """pos [B, T] -> [B, T, dim/2]"""
    f = 1.0 / (base ** (torch.arange(0, dim, 2, device=pos.device).float() / dim))
    return pos.float()[..., None] * f


def apply_rope(x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, D], ang [B, T, D/2]"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


def anchor_positions(tokens: torch.Tensor, slots: torch.Tensor,
                     region_idx: torch.Tensor | None = None):
    """[B,T] tokens -> ([B,T,2] xy in [0,1], [B,T] bool 'is known').

    Tracks positions causally while respecting nested local frames.

    The previous implementation treated signed local x/y bins as world
    coordinates and did not apply rx/ry/rt. That made spatial attention
    geometrically wrong precisely when explicit regions were enabled.
    """
    import vecgpt.schema as SC

    rx_lo, rx_hi = SC.RANGE["rx"]
    ry_lo, ry_hi = SC.RANGE["ry"]
    x_lo, x_hi = SC.RANGE["x"]
    y_lo, y_hi = SC.RANGE["y"]
    rt_lo, rt_hi = SC.RANGE["rt"]
    B, T = tokens.shape
    dev = tokens.device
    region_idx = (torch.zeros_like(tokens) if region_idx is None else region_idx)
    R = max(int(region_idx.max().detach()), 0) + 1
    rows = torch.arange(B, device=dev)

    cx = torch.full((B, R), 0.5, device=dev)
    cy = torch.full((B, R), 0.5, device=dev)
    rt = torch.zeros(B, R, device=dev)
    lx = torch.zeros(B, R, device=dev)
    ly = torch.zeros(B, R, device=dev)
    ax = torch.zeros(B, R, device=dev)
    ay = torch.zeros(B, R, device=dev)
    frame_known = torch.zeros(B, R, dtype=torch.bool, device=dev)
    frame_known[:, 0] = True  # implicit scene frame
    anchor_known = torch.zeros_like(frame_known)
    out_xy, out_known = [], []

    def uniform(tok, lo, n):
        return (tok - lo + 0.5).float() / n

    def signed(tok, lo, n, hi):
        return -hi + (tok - lo).float() * (2 * hi / (n - 1))

    for t in range(T):
        tok = tokens[:, t]
        rid = region_idx[:, t].clamp_max(R - 1)

        m = (tok >= rx_lo) & (tok < rx_hi)
        cx[rows[m], rid[m]] = uniform(tok[m], rx_lo, rx_hi - rx_lo)
        frame_known[rows[m], rid[m]] = False
        anchor_known[rows[m], rid[m]] = False

        m = (tok >= ry_lo) & (tok < ry_hi)
        cy[rows[m], rid[m]] = uniform(tok[m], ry_lo, ry_hi - ry_lo)
        frame_known[rows[m], rid[m]] = True

        m = (tok >= rt_lo) & (tok < rt_hi)
        q = SC.QUANTS["rt"]
        rt[rows[m], rid[m]] = q.lo + (
            tok[m] - rt_lo + 0.5
        ).float() * (q.hi - q.lo) / q.n

        m = (tok >= x_lo) & (tok < x_hi)
        q = SC.QUANTS["x"]
        lx[rows[m], rid[m]] = signed(tok[m], x_lo, q.n, q.hi)
        anchor_known[rows[m], rid[m]] = False

        m = (tok >= y_lo) & (tok < y_hi)
        q = SC.QUANTS["y"]
        ly[rows[m], rid[m]] = signed(tok[m], y_lo, q.n, q.hi)
        rr, cc = rows[m], rid[m]
        c, s = rt[rr, cc].cos(), rt[rr, cc].sin()
        ax[rr, cc] = cx[rr, cc] + lx[rr, cc] * c - ly[rr, cc] * s
        ay[rr, cc] = cy[rr, cc] + lx[rr, cc] * s + ly[rr, cc] * c
        anchor_known[rr, cc] = True

        rr = rows
        use_anchor = anchor_known[rr, rid]
        px = torch.where(use_anchor, ax[rr, rid], cx[rr, rid])
        py = torch.where(use_anchor, ay[rr, rid], cy[rr, rid])
        known = use_anchor | frame_known[rr, rid]
        out_xy.append(torch.stack([px, py], -1))
        out_known.append(known)

    return torch.stack(out_xy, 1), torch.stack(out_known, 1)


def turtle_states(tokens: torch.Tensor,
                  region_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Causal world-space pen state after every consumed token.

    The drawing grammar already defines exact SE(2) dynamics. Asking the
    Transformer to rediscover arc integration from token co-occurrences wastes
    capacity and makes the next cross-attention query unaware of where the pen
    currently is. This is analogous to positional encoding: deterministic
    execution state, not a semantic label.

    Returns ``[B,T,4] = (x, y, sin(theta), cos(theta))``.  Region frames are
    respected, so the same function is valid for flat and nested streams.
    """
    import vecgpt.schema as SC

    B, T = tokens.shape
    dev, dtype = tokens.device, torch.float32
    region_idx = torch.zeros_like(tokens) if region_idx is None else region_idx
    R = max(int(region_idx.max().detach()), 0) + 1
    rows = torch.arange(B, device=dev)

    cx = torch.full((B, R), 0.5, device=dev, dtype=dtype)
    cy = torch.full((B, R), 0.5, device=dev, dtype=dtype)
    frame_th = torch.zeros(B, R, device=dev, dtype=dtype)
    lx = torch.zeros(B, R, device=dev, dtype=dtype)
    ly = torch.zeros(B, R, device=dev, dtype=dtype)
    th = torch.zeros(B, R, device=dev, dtype=dtype)
    pending_len = torch.zeros(B, R, device=dev, dtype=dtype)

    ranges = SC.RANGE

    def decode(tok, field):
        lo, _ = ranges[field]
        q = SC.QUANTS[field]
        i = (tok - lo).float()
        if q.kind == "uniform":
            return q.lo + (i + 0.5) * (q.hi - q.lo) / q.n
        if q.kind == "log":
            return torch.exp(
                math.log(q.lo)
                + i * (math.log(q.hi) - math.log(q.lo)) / (q.n - 1)
            )
        if q.kind == "signed":
            return -q.hi + i * (2 * q.hi / (q.n - 1))
        if q.kind == "circular":
            return q.lo + (i + 0.5) * (q.hi - q.lo) / q.n
        raise ValueError(q.kind)

    out = []
    for t in range(T):
        tok = tokens[:, t]
        rid = region_idx[:, t].clamp_max(R - 1)

        for field, dst in (("rx", cx), ("ry", cy), ("rt", frame_th),
                           ("x", lx), ("y", ly), ("theta", th),
                           ("len", pending_len)):
            lo, hi = ranges[field]
            m = (tok >= lo) & (tok < hi)
            if m.any():
                dst[rows[m], rid[m]] = decode(tok[m], field)

        lo, hi = ranges["turn"]
        m = (tok >= lo) & (tok < hi)
        if m.any():
            rr, ri = rows[m], rid[m]
            turn = decode(tok[m], "turn")
            length = pending_len[rr, ri]
            heading = th[rr, ri]
            small = turn.abs() < 1e-5
            safe_turn = torch.where(small, torch.ones_like(turn), turn)
            radius = length / safe_turn
            nx_arc = lx[rr, ri] + radius * (
                torch.sin(heading + turn) - torch.sin(heading)
            )
            ny_arc = ly[rr, ri] + radius * (
                -torch.cos(heading + turn) + torch.cos(heading)
            )
            nx_line = lx[rr, ri] + length * torch.cos(heading)
            ny_line = ly[rr, ri] + length * torch.sin(heading)
            lx[rr, ri] = torch.where(small, nx_line, nx_arc)
            ly[rr, ri] = torch.where(small, ny_line, ny_arc)
            th[rr, ri] = heading + turn

        rr = rows
        ft = frame_th[rr, rid]
        c, s = ft.cos(), ft.sin()
        wx = cx[rr, rid] + lx[rr, rid] * c - ly[rr, rid] * s
        wy = cy[rr, rid] + lx[rr, rid] * s + ly[rr, rid] * c
        wth = th[rr, rid] + ft
        out.append(torch.stack((wx, wy, wth.sin(), wth.cos()), -1))
    return torch.stack(out, 1)


def region_plan_positions(tokens: torch.Tensor,
                          region_idx: torch.Tensor | None) -> torch.Tensor:
    """For every token, index of its dynamic REGION plan token.

    The rt token is the last token of a region header, so its hidden state is
    the first state that contains the complete coarse plan. Region 0 is the
    implicit scene workspace and uses BOS. There is no fixed bank and no
    maximum number of semantic slots: maps are allocated by REGION tokens in
    the sequence itself.
    """
    if region_idx is None:
        return torch.zeros_like(tokens)
    B, T = tokens.shape
    dev = tokens.device
    rows = torch.arange(B, device=dev)
    R = max(int(region_idx.max().detach()), 0) + 1
    latest = torch.full((B, R), -1, dtype=torch.long, device=dev)
    latest[:, 0] = 0
    out = torch.empty_like(tokens)
    lo, hi = S_.RANGE["rt"]
    for t in range(T):
        tok = tokens[:, t]
        rid = region_idx[:, t].clamp_max(R - 1)
        is_plan = (tok >= lo) & (tok < hi)
        latest[rows[is_plan], rid[is_plan]] = t
        out[:, t] = latest[rows, rid]
    return out


class DynamicRegionCrossAttention(nn.Module):
    """Token cross-attention routed by a learned soft heatmap per REGION.

    A region's rt hidden state queries the dense encoder map and produces a
    probability map over pixels. Every descendant stroke token in that
    region reuses this dynamic map while retaining its own content query.
    The routing strength starts near zero, so training begins as ordinary
    global cross-attention and learns locality only when reconstruction
    rewards it.
    """

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.region_q = nn.Linear(d, d)
        self.region_k = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        # A near-zero gate made the alleged masks mathematically irrelevant:
        # even a 100x mask ratio changed an attention logit by <0.1.  Routing
        # is now active from the start; the flat ablation bypasses it entirely.
        self.gate_raw = nn.Parameter(torch.full((n_heads,), 2.0))
        # Half the heads start local around the executed pen position; half
        # remain effectively global. This is deformable attention driven by
        # DSL state, not an object slot or semantic prior.
        local_init = torch.full((n_heads,), -6.0)
        local_init[: max(1, n_heads // 2)] = 0.0
        self.local_raw = nn.Parameter(local_init)
        self.capture = False
        self.last_heatmap = None
        self.routing_masks = None

    def _scope_masks(self, x, mem, region_idx, parent_region_idx, plan_pos):
        """Causal MONet-like masks with competition between siblings.

        Region 0 covers the complete canvas.  Every later region is bounded by
        its parent and receives only the parent's scope not already claimed by
        an earlier sibling.  No semantic class and no fixed slot bank exists;
        the number of masks is the number of REGION tokens in the sequence.
        """
        B, T, D = x.shape
        N = mem.shape[1]
        R = max(int(region_idx.max().detach()), 0) + 1
        rows = torch.arange(B, device=x.device)

        # One causal plan state and one parent id for each dynamic region.
        plan_at = torch.zeros(B, R, dtype=torch.long, device=x.device)
        parent = torch.zeros(B, R, dtype=torch.long, device=x.device)
        for t in range(T):
            rid = region_idx[:, t].clamp_max(R - 1)
            valid = plan_pos[:, t] == t
            if valid.any():
                plan_at[rows[valid], rid[valid]] = t
            parent[rows, rid] = parent_region_idx[:, t].clamp_max(R - 1)

        plans = x.gather(
            1, plan_at.unsqueeze(-1).expand(-1, -1, D)
        )
        rq = self.region_q(plans).view(B, R, self.h, self.dh).transpose(1, 2)
        rk = self.region_k(mem).view(B, N, self.h, self.dh).transpose(1, 2)
        raw = torch.matmul(rq, rk.transpose(-2, -1)) / math.sqrt(self.dh)
        raw = raw.sigmoid()

        masks = [torch.ones(B, self.h, N, device=x.device, dtype=x.dtype)]
        for r in range(1, R):
            pr = parent[:, r].clamp_max(r - 1)
            parent_mask = torch.stack(masks, 2).gather(
                2, pr[:, None, None, None].expand(-1, self.h, 1, N)
            ).squeeze(2)
            scope = parent_mask
            for s in range(1, r):
                sibling = parent[:, s] == pr
                if sibling.any():
                    scope = scope * torch.where(
                        sibling[:, None, None], 1.0 - masks[s], 1.0
                    )
            masks.append(raw[:, :, r] * scope)
        masks = torch.stack(masks, 2)  # [B,H,R,N]
        token_masks = masks.gather(
            2, region_idx.clamp_max(R - 1)[:, None, :, None].expand(
                -1, self.h, -1, N
            )
        )
        return token_masks, masks

    def forward(self, x, mem, plan_pos, region_idx=None,
                parent_region_idx=None, use_region_masks=True,
                query_xy=None, query_local=None):
        B, T, D = x.shape
        N = mem.shape[1]

        def heads(z, proj):
            return proj(z).view(B, -1, self.h, self.dh).transpose(1, 2)

        q = heads(x, self.q)
        k = heads(mem, self.k)
        v = heads(mem, self.v)
        base = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)

        if query_xy is not None:
            grid = int(round(math.sqrt(N)))
            if grid * grid != N:
                raise ValueError(f"spatial memory must be square, got {N} cells")
            c = (
                torch.arange(grid, device=x.device, dtype=x.dtype) + 0.5
            ) / grid
            yy, xx = torch.meshgrid(c, c, indexing="ij")
            mem_xy = torch.stack((xx, yy), -1).reshape(1, 1, N, 2)
            d2 = (
                query_xy[:, None, :, None, :] - mem_xy[:, :, None, :, :]
            ).square().sum(-1)
            local_strength = F.softplus(self.local_raw)[None, :, None, None]
            local_bias = -local_strength * d2 / (0.25 ** 2)
            if query_local is not None:
                local_bias = local_bias * query_local[:, None, :, None].to(x.dtype)
            base = base + local_bias

        if ((use_region_masks or self.capture) and region_idx is not None
                and parent_region_idx is not None):
            heatmap, region_masks = self._scope_masks(
                x, mem, region_idx, parent_region_idx, plan_pos
            )
            self.routing_masks = region_masks
            if self.capture:
                self.last_heatmap = heatmap.detach()
            strength = torch.sigmoid(self.gate_raw)[None, :, None, None]
            # A mask is a routing decision, not a decorative probability map.
            bias = strength * heatmap.clamp_min(1e-6).log()
            score = base + bias if use_region_masks else base
        else:
            self.routing_masks = None
            score = base
        attn = score.softmax(-1)
        out = torch.matmul(attn, v)
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class AxialSelfAttention(nn.Module):
    """Heads are partitioned across axes: `n_seg` heads see the
    segment-within-stroke index, `n_stroke` heads see the stroke-within-
    scene index, the rest see flat sequence position."""

    def __init__(self, d: int, n_heads: int, n_seg: int, n_stroke: int,
                 spatial_bias=False, region_attention=False,
                 n_global_heads=2):
        super().__init__()
        assert n_seg + n_stroke <= n_heads
        self.h, self.dh = n_heads, d // n_heads
        self.n_seg, self.n_stroke = n_seg, n_stroke
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        # Soft spatial locality. Global attention lets the model pick up
        # long-range correlations that are artefacts of the generators
        # ("in this family, six segments after a tight arc comes ...")
        # rather than facts about shape - a plausible contributor to the
        # in-dist/OOD gap. A hard local window would be wrong, because
        # some long-range structure is real (bilateral symmetry, overall
        # proportion), so this is a learnable per-head penalty on distance
        # between stroke anchors: close is cheap, far is available if the
        # model pays for it.
        #
        # lambda is softplus(raw) with raw init very negative, so at step 0
        # the bias is ~0 and the model is EXACTLY the no-bias model. It can
        # only introduce locality if locality helps.
        self.spatial_bias = spatial_bias
        self.region_attention = region_attention
        self.n_global_heads = min(n_global_heads, n_heads)
        if spatial_bias:
            self.lam_raw = nn.Parameter(torch.full((n_heads,), -6.0))

    def forward(self, x, seg_idx, stroke_idx, key_pad, pos_xy=None,
                pos_known=None, region_idx=None, parent_region_idx=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        flat = torch.arange(T, device=x.device).expand(B, T)
        cuts = [(0, self.n_seg, seg_idx), (self.n_seg, self.n_seg + self.n_stroke, stroke_idx),
                (self.n_seg + self.n_stroke, self.h, flat)]
        qs, ks = [], []
        for lo, hi, pos in cuts:
            if hi <= lo:
                continue
            a = rope_angles(pos, self.dh)
            qs.append(apply_rope(q[:, lo:hi], a))
            ks.append(apply_rope(k[:, lo:hi], a))
        q, k = torch.cat(qs, 1), torch.cat(ks, 1)

        allowed = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        allowed = allowed[None, None] & key_pad[:, None, None, :]
        allowed = allowed | (~allowed.any(-1, keepdim=True))  # never all-masked

        if self.region_attention and region_idx is not None:
            qreg = region_idx[:, :, None]
            kreg = region_idx[:, None, :]
            related = (qreg == kreg) | (kreg == 0) | (qreg == 0)

            # Build region -> parent lookup from the per-token structural
            # labels, then let local heads see every ancestor of the query.
            max_reg = max(int(region_idx.max().detach()), 0) + 1
            parent_map = torch.zeros(
                B, max_reg, dtype=region_idx.dtype, device=x.device
            )
            parent_map.scatter_(1, region_idx.clamp_max(max_reg - 1),
                                parent_region_idx.clamp_max(max_reg - 1))
            anc = parent_region_idx
            for _ in range(8):
                related = related | (kreg == anc[:, :, None])
                anc = parent_map.gather(1, anc.clamp_max(max_reg - 1))

            local_allowed = allowed & related[:, None]
            if self.n_global_heads <= 0:
                allowed = local_allowed
            elif self.n_global_heads < self.h:
                allowed = torch.cat(
                    [allowed.expand(B, self.n_global_heads, T, T),
                     local_allowed.expand(B, self.h - self.n_global_heads, T, T)],
                    dim=1,
                )

        if self.spatial_bias and pos_xy is not None:
            dist = torch.cdist(pos_xy, pos_xy)  # [B,T,T]
            pair = (pos_known[:, :, None] & pos_known[:, None, :]).float()
            bias = -F.softplus(self.lam_raw)[None, :, None, None] * (dist * pair)[:, None]
            mask = bias.masked_fill(~allowed, float("-inf"))
        else:
            mask = torch.zeros(1, 1, 1, 1, device=x.device, dtype=q.dtype).expand(
                B, self.h, T, T).masked_fill(~allowed, float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))


class DecoderLayer(nn.Module):
    def __init__(self, d, n_heads, n_seg, n_stroke, mult=4,
                 spatial_bias=False, region_attention=False,
                 n_global_heads=2, dynamic_region_masks=True):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = AxialSelfAttention(
            d, n_heads, n_seg, n_stroke, spatial_bias,
            region_attention, n_global_heads
        )
        self.dynamic_region_masks = dynamic_region_masks
        # Always instantiate the same parameters so a shared pretrain can
        # branch into flat/masked ablations without checkpoint surgery.
        self.xa = DynamicRegionCrossAttention(d, n_heads)
        # Semantic memory is deliberately separate from the spatial canvas.
        # LLM hidden states have no intrinsic x/y topology, while visual
        # feature cells do.  Forcing both through one attention made text
        # concepts pretend to be pixels and, during visual bootstrap, let a
        # tiny foreground signal disappear in hundreds of background cells.
        self.semantic_xa = nn.MultiheadAttention(
            d, n_heads, batch_first=True
        )
        self.ff = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Linear(d * mult, d))

    def forward(self, x, mem, seg_idx, stroke_idx, key_pad, pos_xy=None,
                pos_known=None, region_idx=None, parent_region_idx=None,
                region_plan_pos=None, semantic_mem=None,
                cross_xy=None, cross_local=None):
        x = x + self.sa(
            self.n1(x), seg_idx, stroke_idx, key_pad, pos_xy, pos_known,
            region_idx, parent_region_idx
        )
        h = self.n2(x)
        cross = self.xa(
            h, mem, region_plan_pos, region_idx, parent_region_idx,
            use_region_masks=self.dynamic_region_masks,
            query_xy=cross_xy, query_local=cross_local,
        )
        if semantic_mem is not None:
            cross = cross + self.semantic_xa(
                h, semantic_mem, semantic_mem, need_weights=False
            )[0]
        x = x + cross
        return x + self.ff(self.n3(x))


# --------------------------------------------------------------- encoder


def sincos_2d(grid: int, d: int) -> torch.Tensor:
    """Fixed 2D Fourier positional code, [1, grid*grid, d].

    The decoder's job for x/y is: attend to the cell that holds ink, then
    turn "which cell" into an absolute coordinate bin. With a learned
    random embedding that map is an arbitrary lookup the model has to
    memorise cell by cell. A sinusoidal code makes it a smooth function of
    position, so what it learns for one region transfers to the rest -
    the same argument the design doc already makes for encoding time with
    Fourier features rather than a raw scalar, applied to space.
    """
    assert d % 4 == 0
    q = d // 4
    f = 1.0 / (10000.0 ** (torch.arange(q).float() / q))
    c = torch.arange(grid).float()
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    out = []
    for p in (xx.reshape(-1), yy.reshape(-1)):
        a = p[:, None] * f[None, :]
        out += [a.sin(), a.cos()]
    return torch.cat(out, -1)[None]


class EncoderLayer(nn.Module):
    def __init__(self, d, n_heads, mult=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Linear(d * mult, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.sa(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class ConvEncoder(nn.Module):
    """64 px -> 16x16 = 256 memory cells (stride 4), then self-attention.

    The old encoder used stride 8 on a 48 px canvas: 6x6 = 36 cells, one
    cell per 8x8 px block, for strokes 1-2 px wide. Position could not be
    resolved better than the cell grid no matter how good the attention
    got. Stride 4 at 64 px is 4x the spatial resolution.

    The self-attention layers exist for a specific bootstrapping problem.
    A vector canvas is ~98% background, so at init almost every memory
    cell looks the same and the decoder's cross-attention is near-uniform;
    averaging 256 near-identical cells carries no signal about where the
    stroke is, so there is little gradient pushing the attention to
    sharpen - the same chicken-and-egg that made the old anchor head
    collapse to a constant. Self-attention over the memory lets every cell
    see the whole canvas first, so "where the ink is" is present in the
    cells the decoder reads before its cross-attention has learned
    anything.
    """

    def __init__(self, image_size=64, d=256, base=48, n_layers=2, n_heads=8):
        super().__init__()
        assert image_size % 4 == 0
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 3, 1, 1), nn.GELU(),
            nn.Conv2d(base, base * 2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(base * 2, base * 2, 3, 1, 1), nn.GELU(),
            nn.Conv2d(base * 2, base * 4, 3, 2, 1), nn.GELU(),
            nn.Conv2d(base * 4, d, 1),
        )
        self.grid = image_size // 4
        self.register_buffer("sincos", sincos_2d(self.grid, d), persistent=False)
        self.pos = nn.Parameter(torch.zeros(1, self.grid**2, d))
        self.layers = nn.ModuleList([EncoderLayer(d, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, img):  # [B,H,W,3] -> [B, grid^2, d]
        # Feed INK (background - image), not the raw image. A vector canvas
        # is ~98% background, so the raw input is an almost-constant 1.0
        # with a rare small dip; centring it this way puts the background
        # at exactly 0 and makes a stroke a linear function of its colour.
        # Measured on an isolated colour-reading probe: CE 2.05 -> 1.35
        # nats (uniform prior is 2.77) for the same conv stack and budget.
        x = (1.0 - img).permute(0, 3, 1, 2)
        f = self.net(x).flatten(2).transpose(1, 2) + self.sincos + self.pos
        for layer in self.layers:
            f = layer(f)
        return self.norm(f)


class ForegroundSemanticPool(nn.Module):
    """Learned visual readouts biased toward ink, never toward a class.

    Sparse vector drawings are overwhelmingly background. Ordinary
    cross-attention can minimise teacher-forced CE by averaging all cells
    and learning the marginal stroke prior. These variable-content readout
    tokens are generic set pooling: their queries are learned, but the only
    fixed fact is that non-white pixels contain the drawing. No query means
    "circle", "eye", or any other predeclared concept.
    """

    def __init__(self, d: int, n_heads: int, n_tokens: int = 8):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.dh = d // n_heads
        self.n_tokens = n_tokens
        self.query = nn.Parameter(torch.randn(1, n_tokens, d) * 0.02)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )

    def forward(self, spatial: torch.Tensor, img: torch.Tensor,
                grid: int) -> torch.Tensor:
        B, N, D = spatial.shape
        if N != grid * grid:
            raise ValueError(f"expected {grid * grid} spatial cells, got {N}")
        # Max pooling preserves a one-pixel stroke instead of diluting it.
        ink = (1.0 - img).amax(-1, keepdim=True).permute(0, 3, 1, 2)
        ink = F.adaptive_max_pool2d(ink, (grid, grid)).flatten(2)
        # A small floor keeps empty/near-white canvases well-defined.
        foreground_bias = 2.0 * (ink + 1e-3).log()  # [B,1,N]

        q0 = self.query.expand(B, -1, -1)
        q = self.q(q0).view(B, self.n_tokens, self.h, self.dh).transpose(1, 2)
        k = self.k(spatial).view(B, N, self.h, self.dh).transpose(1, 2)
        v = self.v(spatial).view(B, N, self.h, self.dh).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        score = score + foreground_bias[:, :, None, :]
        out = torch.matmul(score.softmax(-1), v)
        out = self.proj(out.transpose(1, 2).reshape(B, self.n_tokens, D))
        x = self.norm1(q0 + out)
        return self.norm2(x + self.ff(x))


class GlobalVisualEncoder(nn.Module):
    """Spatially preserving visual readout with no semantic assumptions.

    Max pooling alone was almost invariant to line length: a short and a long
    thin Stroke produce the same channel maximum.  That made the branch learn
    location while decoding long lines as dots.  Keep both average (mass and
    extent) and max (thin-feature presence), an explicit 8x8 ink canvas, and
    generic coordinate moments.  These are geometric coordinates like a
    positional encoding, not named objects or primitive classifiers.
    """

    def __init__(self, d: int, n_tokens: int = 8):
        super().__init__()
        c = min(64, max(24, d // 4))
        self.n_tokens = n_tokens
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 5, 2, 2), nn.GELU(),
            nn.Conv2d(c, c, 3, 2, 1), nn.GELU(),
            nn.Conv2d(c, c, 3, 2, 1), nn.GELU(),
        )
        # pooled CNN (average + max), raw 8x8 ink/RGB-deviation canvas, and
        # 20 global appearance/shape coordinates.
        input_dim = c * 4 * 4 * 2 + 4 * 8 * 8 + 20
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 4 * d), nn.GELU(),
            nn.Linear(4 * d, n_tokens * d),
        )
        # The first token is a stable geometry carrier. Without this explicit
        # route, the learned projection spent its capacity on easy RGB/x/y and
        # discarded extent (measured len wrong-image gap 0.012).
        self.geometry_proj = nn.Sequential(
            nn.Linear(20, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        dev = (1.0 - img).permute(0, 3, 1, 2)
        ink = dev.amax(1)
        feat = self.net(dev)
        pooled = torch.cat((
            F.adaptive_avg_pool2d(feat, (4, 4)).flatten(1),
            F.adaptive_max_pool2d(feat, (4, 4)).flatten(1),
        ), 1)
        raw = F.adaptive_avg_pool2d(
            torch.cat((ink[:, None], dev), 1), (8, 8)
        ).flatten(1)

        B, H, W = ink.shape
        dtype, device = ink.dtype, ink.device
        y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        mass = ink.sum((1, 2)).clamp_min(1e-4)
        p = ink / mass[:, None, None]
        mx = (p * xx).sum((1, 2))
        my = (p * yy).sum((1, 2))
        dx, dy = xx[None] - mx[:, None, None], yy[None] - my[:, None, None]
        mean_rgb = (
            img * ink[..., None]
        ).sum((1, 2)) / mass[:, None]
        cxx = (p * dx.square()).sum((1, 2))
        cyy = (p * dy.square()).sum((1, 2))
        cxy = (p * dx * dy).sum((1, 2))
        cov = torch.stack((
            torch.stack((cxx, cxy), -1),
            torch.stack((cxy, cyy), -1),
        ), -2)
        eigval, eigvec = torch.linalg.eigh(cov)
        axis = eigvec[:, :, 1]
        # Canonical undirected axis: positive y, with positive x as tie-break.
        flip = (axis[:, 1] < 0) | (
            (axis[:, 1].abs() < 1e-6) & (axis[:, 0] < 0)
        )
        axis = torch.where(flip[:, None], -axis, axis)
        major, minor = eigval[:, 1], eigval[:, 0]
        length_hint = (12.0 * (major - minor).clamp_min(1e-8)).sqrt()
        width_hint = (12.0 * minor.clamp_min(1e-8)).sqrt()
        anchor_x = mx - 0.5 * axis[:, 0] * length_hint
        anchor_y = my - 0.5 * axis[:, 1] * length_hint
        # Generic curvature coordinate. In the PCA chord frame, fit the ink
        # centreline with v = a*u^2 + b*u + c. For a short circular arc,
        # curvature is approximately 2a and total turn approximately 2aL.
        # This is no named-shape detector; it is the second spatial derivative
        # that the Stroke DSL itself needs.
        u = dx * axis[:, 0, None, None] + dy * axis[:, 1, None, None]
        v = -dx * axis[:, 1, None, None] + dy * axis[:, 0, None, None]
        design = torch.stack((u.square(), u, torch.ones_like(u)), -1).flatten(1, 2)
        weights = p.flatten(1)
        vv = v.flatten(1)
        normal = torch.einsum("bn,bni,bnj->bij", weights, design, design)
        rhs = torch.einsum("bn,bni,bn->bi", weights, design, vv)
        eye = torch.eye(3, device=device, dtype=dtype)[None]
        quad = torch.linalg.solve(normal + 1e-4 * eye, rhs)
        turn_hint = (2.0 * quad[:, 0] * length_hint).clamp(-2.5, 2.5)
        moments = torch.cat((
            torch.stack((
                mass / float(H * W), mx, my, cxx, cyy, cxy,
                quad[:, 0], quad[:, 1], turn_hint,
            ), 1),
            mean_rgb,
            torch.stack((
                major, minor, axis[:, 0], axis[:, 1],
                length_hint, width_hint, anchor_x, anchor_y,
            ), 1),
        ), 1)
        x = torch.cat((pooled, raw, moments), 1)
        learned = self.proj(x).view(B, self.n_tokens, -1)
        geometry = self.geometry_proj(moments)[:, None]
        tokens = torch.cat((geometry, learned[:, 1:]), 1)
        return self.norm(tokens)


# ----------------------------------------------------------------- model


class VecGPT(nn.Module):
    def __init__(self, image_size=64, d=256, n_heads=8, n_layers=6, n_seg_heads=3,
                 n_stroke_heads=3, max_seg=64, max_stroke=32, enc_base=48,
                 n_enc_layers=2, spatial_bias=False, max_region_depth=16,
                 region_attention=False,
                 n_global_heads=2, dynamic_region_masks=True,
                 condition_dim=None):
        super().__init__()
        self.encoder = ConvEncoder(image_size, d, enc_base, n_enc_layers, n_heads)
        self.visual_semantic_pool = ForegroundSemanticPool(d, n_heads)
        self.global_visual_encoder = GlobalVisualEncoder(d)
        # A learned "no conditioning" memory. The image encoder is
        # scaffolding: the target is generation with no source image, from
        # a latent shared with an LLM. Training with the conditioning
        # dropped some of the time means one checkpoint can already do
        # unconditional sampling, and gives classifier-free guidance for
        # free later. Cross-attention never learns to assume a picture is
        # there, which is the assumption that would have to be unlearned.
        self.null_mem = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.null_semantic = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.tok_emb = nn.Embedding(VOCAB, d)
        self.slot_emb = nn.Embedding(N_STATES, d)
        self.turtle_proj = nn.Sequential(
            nn.Linear(4, d), nn.GELU(), nn.Linear(d, d)
        )
        self.condition_dim = d if condition_dim is None else condition_dim
        self.latent_proj = (
            nn.Identity() if self.condition_dim == d
            else nn.Linear(self.condition_dim, d)
        )
        self.latent_norm = nn.LayerNorm(d)
        # Shared-space bootstrap: visual readouts are aligned with the
        # corresponding vector program without any class or object label.
        # Later an LLM adapter targets this same semantic space.
        self.visual_align = nn.Linear(d, d)
        self.program_align = nn.Linear(d, d)
        # Text/LLM states have no x/y topology.  They first write into the
        # same dense spatial workspace produced by the visual encoder; REGION
        # masks therefore remain meaningful when no image exists.
        self.canvas_query = nn.Parameter(
            torch.randn(1, self.encoder.grid ** 2, d) * 0.02
        )
        self.latent_to_canvas = nn.MultiheadAttention(
            d, n_heads, batch_first=True
        )
        self.canvas_ff = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.canvas_norm = nn.LayerNorm(d)
        self.register_buffer(
            "time_freq", 2.0 ** torch.arange(8).float(), persistent=False
        )
        self.time_to_condition = nn.Sequential(
            nn.Linear(16, d), nn.GELU(),
            nn.Linear(d, self.condition_dim),
        )
        # Structural coordinates, not semantic labels. They tell the model
        # which tokens share a workspace and how deeply it is nested; what
        # that workspace represents must still emerge from training.
        self.region_depth_emb = nn.Embedding(max_region_depth, d)
        self.seg_clamp, self.stroke_clamp = max_seg - 1, max_stroke - 1
        self.region_depth_clamp = max_region_depth - 1
        self.use_spatial_bias = spatial_bias
        self.layers = nn.ModuleList(
            [DecoderLayer(
                d, n_heads, n_seg_heads, n_stroke_heads,
                spatial_bias=spatial_bias,
                region_attention=region_attention,
                n_global_heads=n_global_heads,
                dynamic_region_masks=dynamic_region_masks,
            )
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)
        # Short gradient path from the condition to every field distribution.
        # Field masks make one shared bias vector behave as separate x/y/etc.
        # heads, while the contextual decoder still distinguishes repeated
        # fields in multi-stroke programs.
        self.semantic_to_vocab = nn.Linear(d, VOCAB, bias=False)
        # Curriculum-only readout: make the first condition token retain every
        # numeric field, not merely whichever easy attributes dominate a
        # global contrastive objective. Grammar state distinguishes R/G/B and
        # repeated parameter roles without introducing semantic classes.
        self.condition_probe = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, VOCAB)
        )
        self.register_buffer("slot_mask", build_state_mask(), persistent=False)
        self.image_size = image_size
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, img):
        return self.encoder(img)

    def encode_condition(self, img):
        """Return spatial cells and foreground semantic readout tokens."""
        spatial = self.encoder(img)
        semantic = torch.cat((
            self.global_visual_encoder(img),
            self.visual_semantic_pool(spatial, img, self.encoder.grid),
        ), dim=1)
        return spatial, semantic

    def condition_probe_logits(
        self, semantic: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        summary = semantic[:, :1].expand(-1, states.shape[1], -1)
        return self.condition_probe(torch.cat((summary, self.slot_emb(states)), -1))

    def condition_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Project LLM/text states without inventing a false 2D topology.

        ``latent`` can be ``[B,D]`` (one global concept vector) or ``[B,N,D]``
        (a variable sequence of LLM hidden states). These are semantic memory
        tokens. A spatial planning canvas is optional and is created explicitly
        by :meth:`condition_canvas`.
        """
        if latent.ndim == 2:
            latent = latent[:, None, :]
        if latent.ndim != 3 or latent.shape[-1] != self.condition_dim:
            raise ValueError(
                f"expected latent [B,D] or [B,N,D] with D={self.condition_dim}, "
                f"got {tuple(latent.shape)}"
            )
        return self.latent_norm(self.latent_proj(latent))

    def condition_canvas(self, latent: torch.Tensor) -> torch.Tensor:
        """Let semantic states write an optional dense planning canvas."""
        src = self.condition_latent(latent)
        B = src.shape[0]
        pos = (self.encoder.sincos + self.encoder.pos).to(src.dtype)
        q = self.canvas_query.expand(B, -1, -1) + pos.expand(B, -1, -1)
        canvas = q + self.latent_to_canvas(q, src, src, need_weights=False)[0]
        canvas = canvas + self.canvas_ff(canvas)
        return self.canvas_norm(canvas)

    def condition_alignment_loss(
        self,
        semantic: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 0.10,
    ) -> torch.Tensor:
        """Symmetric image/program InfoNCE with no semantic labels."""
        visual = self.visual_align(semantic.mean(1))
        weights = mask.to(semantic.dtype).unsqueeze(-1)
        program = (
            self.tok_emb(tokens) * weights
        ).sum(1) / weights.sum(1).clamp_min(1.0)
        program = self.program_align(program)
        visual = F.normalize(visual, dim=-1)
        program = F.normalize(program, dim=-1)
        score = visual @ program.transpose(0, 1) / temperature
        target = torch.arange(score.shape[0], device=score.device)
        return 0.5 * (
            F.cross_entropy(score, target)
            + F.cross_entropy(score.transpose(0, 1), target)
        )

    def condition_animation(self, latent: torch.Tensor, time: torch.Tensor,
                            previous_latent: torch.Tensor | None = None):
        """Spatial conditioning canvas for a frame at continuous ``time``.

        The same scene/LLM latent is reused across frames; only a compact time
        token and, optionally, previous-frame latent states change.  This is
        the interface needed for persistent vector identity and delta
        animation, not a frame-wise raster video generator.
        """
        if latent.ndim == 2:
            latent = latent[:, None, :]
        if time.ndim == 0:
            time = time.expand(latent.shape[0])
        phase = 2 * math.pi * time.float()[:, None] * self.time_freq[None]
        tf = torch.cat((phase.sin(), phase.cos()), -1).to(latent.dtype)
        time_token = self.time_to_condition(tf)[:, None, :]
        parts = [latent, time_token]
        if previous_latent is not None:
            if previous_latent.ndim == 2:
                previous_latent = previous_latent[:, None, :]
            parts.append(previous_latent)
        return self.condition_latent(torch.cat(parts, 1))

    def null_memory(self, batch: int, device=None):
        """Conditioning-free spatial workspace for prior sampling."""
        dev = device or self.null_mem.device
        pos = (self.encoder.sincos + self.encoder.pos).to(dev)
        return self.null_mem.expand(batch, pos.shape[1], -1) + pos.expand(
            batch, -1, -1
        )

    def null_semantic_memory(self, batch: int, device=None):
        dev = device or self.null_semantic.device
        return self.null_semantic.to(dev).expand(batch, -1, -1)

    def drop_condition(self, mem, p: float):
        """Replace whole examples' memory with the null memory, prob p.
        Returns (memory, keep) so the caller can exclude blinded examples
        from diagnostics - they cannot read the image by construction."""
        if p <= 0:
            return mem, None
        keep = (torch.rand(mem.shape[0], 1, 1, device=mem.device) >= p)
        k = keep.float()
        null = self.null_memory(mem.shape[0], mem.device)
        return mem * k + null * (1 - k), keep[:, :, 0]

    def _hidden(self, mem, tokens, slots, seg_idx, stroke_idx, mask,
                region_idx=None, region_depth=None, parent_region_idx=None,
                semantic_mem=None):
        pen_state = turtle_states(tokens, region_idx)
        x = (self.tok_emb(tokens) + self.slot_emb(slots)
             + self.turtle_proj(pen_state))
        if region_depth is not None:
            x = x + self.region_depth_emb(
                region_depth.clamp(0, self.region_depth_clamp)
            )
        seg = seg_idx.clamp(0, self.seg_clamp)
        stk = stroke_idx.clamp(0, self.stroke_clamp)
        if self.use_spatial_bias:
            pos_xy, pos_known = anchor_positions(tokens, slots, region_idx)
        else:
            pos_xy = pos_known = None
        plan_pos = region_plan_positions(tokens, region_idx)
        # Before x/y establish an anchor the pen sits at the arbitrary frame
        # centre, so locality would be misleading. Once y has been consumed,
        # turtle state is the exact current pen position for every segment.
        nonlocal_states = {
            S_.STATE_ID["TOP"], S_.STATE_ID["RY"], S_.STATE_ID["RT"]
        }
        cross_local = torch.ones_like(slots, dtype=torch.bool)
        for sid in nonlocal_states:
            cross_local &= slots != sid
        for layer in self.layers:
            x = layer(x, mem, seg, stk, mask, pos_xy, pos_known,
                      region_idx, parent_region_idx, plan_pos, semantic_mem,
                      pen_state[..., :2], cross_local)
        return self.norm(x)

    def _run(self, mem, tokens, slots, seg_idx, stroke_idx, mask,
             region_idx=None, region_depth=None, parent_region_idx=None,
             semantic_mem=None):
        logits = self.head(self._hidden(
            mem, tokens, slots, seg_idx, stroke_idx, mask,
            region_idx, region_depth, parent_region_idx, semantic_mem
        ))
        if semantic_mem is not None:
            logits = logits + self.semantic_to_vocab(
                semantic_mem[:, 0]
            )[:, None, :]
        return logits

    @torch.no_grad()
    def region_latents(self, mem, batch, phase="complete",
                       semantic_mem=None):
        """Return frozen hidden vectors for concept-emergence diagnostics.

        `plan` samples the rt token, before the region's detailed strokes.
        `complete` samples ENDR, after the whole region/subtree is known.
        No semantic label is used or injected here.
        """
        if phase not in {"plan", "complete"}:
            raise ValueError(phase)
        h = self._hidden(
            mem, batch["tokens"], batch["slots"], batch["seg_idx"],
            batch["stroke_idx"], batch["mask"], batch["region_idx"],
            batch["region_depth"], batch["parent_region_idx"], semantic_mem
        )
        if phase == "complete":
            take = (batch["tokens"] == S_.ENDR) & batch["mask"]
        else:
            lo, hi = S_.RANGE["rt"]
            take = ((batch["tokens"] >= lo) & (batch["tokens"] < hi)
                    & batch["mask"])
        return [
            (h[i, take[i]], batch["region_idx"][i, take[i]])
            for i in range(h.shape[0])
        ]

    @torch.no_grad()
    def region_heatmaps(self, mem, batch, layer=-1, semantic_mem=None):
        """Return learned REGION-token heatmaps for visual inspection."""
        xa = self.layers[layer].xa
        xa.capture = True
        try:
            self._hidden(
                mem, batch["tokens"], batch["slots"], batch["seg_idx"],
                batch["stroke_idx"], batch["mask"], batch["region_idx"],
                batch["region_depth"], batch["parent_region_idx"],
                semantic_mem
            )
            maps = xa.last_heatmap.mean(1)  # [B,T,N], average heads
        finally:
            xa.capture = False
            xa.last_heatmap = None

        lo, hi = S_.RANGE["rt"]
        take = ((batch["tokens"] >= lo) & (batch["tokens"] < hi)
                & batch["mask"])
        return [
            (maps[i, take[i]], batch["region_idx"][i, take[i]])
            for i in range(maps.shape[0])
        ]

    def logits(self, mem, tokens, slots, seg_idx, stroke_idx, mask,
               region_idx=None, region_depth=None, parent_region_idx=None,
               semantic_mem=None):
        """Teacher forcing. Input positions 0..T-1 predict tokens 1..T.

        The returned logits at position t are already masked to the field
        that MUST appear at t+1 - illegal tokens get -inf, so the softmax
        is a distribution over that field only and its cross-entropy is
        directly comparable across fields.
        """
        lg = self._run(mem, tokens, slots, seg_idx, stroke_idx, mask,
                       region_idx, region_depth, parent_region_idx,
                       semantic_mem)

        # slot of the token being PREDICTED at position t is slots[t+1]
        nxt = torch.cat([slots[:, 1:], slots[:, -1:]], 1)
        allowed = self.slot_mask[nxt]  # [B, T, VOCAB]
        return lg.masked_fill(~allowed, float("-inf"))

    @torch.no_grad()
    def logits_incremental_debug(self, mem, tokens, mask, semantic_mem=None):
        """Replay ground-truth tokens through generate()'s OWN bookkeeping.

        Exists so a test can assert the incremental and batched paths give
        identical distributions. Not used in training.
        """
        B, T = tokens.shape
        walkers = [Walker() for _ in range(B)]
        dev = tokens.device
        toks = tokens[:, :1]
        labs = [w.advance(int(t)) for w, t in zip(walkers, tokens[:, 0])]
        sl, sg, sk = [l[0] for l in labs], [l[1] for l in labs], [l[2] for l in labs]
        slots = torch.tensor(sl, device=dev)[:, None]
        segs = torch.tensor(sg, device=dev)[:, None]
        strokes = torch.tensor(sk, device=dev)[:, None]
        regions = torch.tensor([l[3] for l in labs], device=dev)[:, None]
        parents = torch.tensor([l[4] for l in labs], device=dev)[:, None]
        depths = torch.tensor([l[5] for l in labs], device=dev)[:, None]
        out = []
        for t in range(T - 1):
            m = torch.ones_like(toks, dtype=torch.bool)
            lg = self._run(mem, toks, slots, segs, strokes, m,
                           regions, depths, parents, semantic_mem)[:, -1]
            allowed = self.slot_mask[torch.tensor([S_.STATE_ID[w.state] for w in walkers], device=dev)]
            out.append(lg.masked_fill(~allowed, float("-inf")))
            nxt = tokens[:, t + 1]
            labs = [w.advance(int(t)) for w, t in zip(walkers, nxt)]
            sl, sg, sk = [l[0] for l in labs], [l[1] for l in labs], [l[2] for l in labs]
            toks = torch.cat([toks, nxt[:, None]], 1)
            slots = torch.cat([slots, torch.tensor(sl, device=dev)[:, None]], 1)
            segs = torch.cat([segs, torch.tensor(sg, device=dev)[:, None]], 1)
            strokes = torch.cat([strokes, torch.tensor(sk, device=dev)[:, None]], 1)
            regions = torch.cat(
                [regions, torch.tensor([l[3] for l in labs], device=dev)[:, None]], 1
            )
            parents = torch.cat(
                [parents, torch.tensor([l[4] for l in labs], device=dev)[:, None]], 1
            )
            depths = torch.cat(
                [depths, torch.tensor([l[5] for l in labs], device=dev)[:, None]], 1
            )
        return torch.stack(out, 1)

    @torch.no_grad()
    def generate(self, mem, max_tokens=256, temperature=0.0, top_p=1.0,
                 device=None, semantic_mem=None):
        """Batched sampling driven by the same SlotTracker as encode().
        temperature=0 -> greedy (deterministic, for reconstruction gates).
        """
        from vecgpt.tokenizer import BOS

        device = device or mem.device
        B = mem.shape[0]
        walkers = [Walker() for _ in range(B)]
        bos = [BOS] * B
        labs = [w.advance(t) for w, t in zip(walkers, bos)]
        sl, sg, sk = [l[0] for l in labs], [l[1] for l in labs], [l[2] for l in labs]
        toks = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        slots = torch.tensor(sl, device=device)[:, None]
        segs = torch.tensor(sg, device=device)[:, None]
        strokes = torch.tensor(sk, device=device)[:, None]
        regions = torch.tensor([l[3] for l in labs], device=device)[:, None]
        parents = torch.tensor([l[4] for l in labs], device=device)[:, None]
        depths = torch.tensor([l[5] for l in labs], device=device)[:, None]
        out = [[] for _ in range(B)]

        for _ in range(max_tokens):
            mask = torch.ones_like(toks, dtype=torch.bool)
            lg = self._run(mem, toks, slots, segs, strokes, mask,
                           regions, depths, parents, semantic_mem)[:, -1]
            allowed = self.slot_mask[torch.tensor(
                [S_.STATE_ID[w.state] for w in walkers], device=device
            )]
            lg = lg.masked_fill(~allowed, float("-inf"))

            # Grammar edges are TYPES with very different cardinalities:
            # TOP chooses one EOS token or one of 257 x bins. A flat argmax
            # compares EOS with each bin separately and therefore emits an
            # empty scene even when most probability mass says "start a
            # stroke". Choose an edge by its summed probability, then a value
            # conditional on that edge. This exactly factorizes the existing
            # token softmax; it adds no class, slot, or new output primitive.
            scaled = lg if temperature <= 0 else lg / temperature
            nxt = torch.empty(lg.shape[0], dtype=torch.long, device=device)
            state_rows = {}
            for i, walker in enumerate(walkers):
                state_rows.setdefault(walker.state, []).append(i)
            for state, indices in state_rows.items():
                idx = torch.tensor(indices, dtype=torch.long, device=device)
                rows = scaled[idx]
                groups, spans = [], []
                for edge in S_.GRAMMAR[state].edges:
                    if edge.emit in S_.SPECIAL_TOK:
                        lo = S_.SPECIAL_TOK[edge.emit]
                        hi = lo + 1
                    else:
                        lo, hi = S_.RANGE[edge.emit]
                    spans.append((lo, hi))
                    groups.append(torch.logsumexp(rows[:, lo:hi], 1))
                group_logits = torch.stack(groups, 1)
                if temperature <= 0:
                    gi = group_logits.argmax(1)
                else:
                    gi = torch.multinomial(group_logits.softmax(1), 1).squeeze(1)
                for edge_i, (lo, hi) in enumerate(spans):
                    take = gi == edge_i
                    if not take.any():
                        continue
                    local = rows[take, lo:hi]
                    if temperature <= 0:
                        value = local.argmax(1)
                    else:
                        p = local.softmax(1)
                        if top_p < 1.0:
                            sp, si = p.sort(1, descending=True)
                            keep = (sp.cumsum(1) - sp) < top_p
                            sp = sp * keep
                            p = torch.zeros_like(p).scatter_(
                                1, si, sp / sp.sum(1, keepdim=True).clamp_min(1e-12)
                            )
                        value = torch.multinomial(p, 1).squeeze(1)
                    nxt[idx[take]] = lo + value

            toklist = nxt.tolist()
            for b, t in enumerate(toklist):
                if not walkers[b].done:
                    out[b].append(t)
            labs = [w.advance(t) for w, t in zip(walkers, toklist)]
            sl, sg, sk = [l[0] for l in labs], [l[1] for l in labs], [l[2] for l in labs]

            # Finished sequences used to keep running the full max_tokens
            # budget.  Evaluation then paid 384 decoder forwards for a
            # stage-1 program that normally ends in ~15 tokens.
            if all(w.done for w in walkers):
                break

            toks = torch.cat([toks, nxt[:, None]], 1)
            slots = torch.cat([slots, torch.tensor(sl, device=device)[:, None]], 1)
            segs = torch.cat([segs, torch.tensor(sg, device=device)[:, None]], 1)
            strokes = torch.cat([strokes, torch.tensor(sk, device=device)[:, None]], 1)
            regions = torch.cat(
                [regions, torch.tensor([l[3] for l in labs], device=device)[:, None]], 1
            )
            parents = torch.cat(
                [parents, torch.tensor([l[4] for l in labs], device=device)[:, None]], 1
            )
            depths = torch.cat(
                [depths, torch.tensor([l[5] for l in labs], device=device)[:, None]], 1
            )
            if all(w.done for w in walkers):
                break
        return out


def count_params(m) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6
