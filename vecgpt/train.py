"""Training loop.

The metric that matters here is per-field INFORMATION GAIN:

    gain(field) = H_marginal(field) - CE_model(field)     [nats]

H_marginal is the entropy of that field's target distribution over the run
so far - i.e. exactly what a model that ignores the picture entirely would
pay. So gain > 0 means "this field is being read off the image", gain ~ 0
means "collapsed to the prior", and you can see which field is which
within a few hundred steps instead of guessing from a rendered grid.

That is the same question the old `relative_pred_std` probe asked, but it
is (a) free, (b) computed every step on the training batch, and (c) in
comparable units across all fields, because everything is one softmax now.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from vecgpt.data import OOD_NAMES, SceneStream, collate, gen_ood, gen_ood_batch
from vecgpt.grammar import OOD as TREE_OOD, sample_tree
from vecgpt.model import VecGPT, count_params
from vecgpt.render import (foreground_render_loss, image_iou, image_iou_shape,
                           ink_map, render_batch, save_grid)
from vecgpt.tokenizer import (METRIC_FIELDS, N_METRIC, VOCAB, build_smoothing_matrix,
                              decode, encode, metric_field_ids, metric_ranges)

ARCHITECTURE_VERSION = 20


@dataclass
class Cfg:
    image_size: int = 64
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_seg_heads: int = 3
    n_stroke_heads: int = 3
    enc_base: int = 48
    # Thin strokes need a global visual representation before the decoder
    # queries them.  Zero silently disabled the very encoder blocks model.py
    # documents as necessary for sparse canvases.
    n_enc_layers: int = 2
    spatial_bias: bool = False
    region_attention: bool = False
    n_global_heads: int = 2
    hierarchical_regions: bool = True
    dynamic_region_masks: bool = True
    # Visual reconstruction is a geometry bootstrap, not unconditional
    # generation.  LLM/text latents are trained through the same memory
    # interface later; randomly blinding this task before it works adds an
    # irreducible target.
    cond_dropout: float = 0.0
    condition_dim: int | None = None
    balanced_field_loss: bool = True
    anchor_loss_weight: float = 2.0
    render_loss_weight: float = 1.00
    render_loss_size: int = 32
    # Cross-entropy treats every wrong numeric bin alike.  That is wasteful
    # for coordinates and actively creates a train/inference gap when the
    # differentiable renderer uses a distribution mean but generation uses
    # argmax.  This topology-aware term pulls probability mass toward the
    # target value (circularly for angles) without changing the Stroke DSL.
    numeric_distance_weight: float = 4.0
    grammar_edge_weight: float = 2.0
    condition_probe_weight: float = 0.0
    region_mask_loss_weight: float = 0.10
    # Decoder-level adversarial ranking is kept as an ablation but disabled:
    # it can raise mismatched CE by damaging the shared decoder. Representation
    # InfoNCE below is the stable way to enforce conditional information.
    condition_margin_weight: float = 0.0
    condition_margin: float = 0.50
    alignment_loss_weight: float = 0.25
    prefix_corruption: float = 0.0
    mastery_gates: bool = True
    mastery_shape_gates: dict = field(default_factory=lambda: {
        0: 0.75,
        1: 0.70,
        2: 0.55,
        3: 0.55,
    })
    preview_every: int = 2000
    preview_n: int = 8
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 200
    lr_total_steps: int | None = None
    grad_clip: float = 1.0
    stage_schedule: tuple = ((1, 2000), (2, 6000), (3, 12000), (4, 16000), (5, 24000))
    log_every: int = 100
    eval_every: int = 1000
    eval_n: int = 32
    ckpt_every: int = 2000
    seed: int = 0
    device: str | None = None
    out_dir: str = "runs/vecgpt"
    cache_dir: str = ""  # pre-rendered cache from build_cache.py
    resume: str | None = None
    render_softness_px: float = 1.0
    label_smooth_bins: float = 1.0


class FieldStats:
    """Running per-field marginal distribution + model CE/accuracy."""

    def __init__(self, device):
        # The prior must live ONLY on tokens this field can legally take.
        # A uniform prior over the whole vocabulary inflates H_marginal
        # early on and then decays as real counts arrive, which makes the
        # gain drift downwards while the model is still improving - a
        # metric that lies in exactly the situation it exists for.
        self.counts = torch.zeros(N_METRIC, VOCAB, device=device)
        for i, (lo, hi) in enumerate(metric_ranges()):
            self.counts[i, lo:hi] = 0.5
        self.ce = torch.zeros(N_METRIC, device=device)
        self.correct = torch.zeros(N_METRIC, device=device)
        self.binerr = torch.zeros(N_METRIC, device=device)
        self.n = torch.zeros(N_METRIC, device=device)

    def update(self, logits, targets, tgt_slots, valid):
        fid = metric_field_ids(targets, tgt_slots)
        lp = logits.log_softmax(-1)
        nll = -lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        pred = logits.argmax(-1)
        hit = (pred == targets).float()
        # top-1 is a poor metric for a fine ordinal grid (being one bin off
        # scores the same as being 200 off), so track bin distance too.
        binerr = (pred - targets).abs().float()
        for f in range(N_METRIC):
            m = valid & (fid == f)
            k = int(m.sum())
            if k == 0:
                continue
            self.ce[f] += nll[m].sum()
            self.correct[f] += hit[m].sum()
            self.binerr[f] += binerr[m].sum()
            self.n[f] += k
            self.counts[f].scatter_add_(0, targets[m], torch.ones(k, device=targets.device))

    def report(self):
        rows = []
        for f in range(N_METRIC):
            if self.n[f] == 0:
                continue
            p = self.counts[f] / self.counts[f].sum()
            h = float(-(p * p.clamp_min(1e-12).log()).sum())
            ce = float(self.ce[f] / self.n[f])
            rows.append(dict(field=METRIC_FIELDS[f], ce=ce, h_marginal=h, gain=h - ce,
                             acc=float(self.correct[f] / self.n[f]),
                             mae_bins=float(self.binerr[f] / self.n[f]), n=int(self.n[f])))
        return rows

    def reset_window(self):
        self.ce.zero_()
        self.correct.zero_()
        self.binerr.zero_()
        self.n.zero_()


def make_batch(stream, cfg: Cfg, device, cached: bool = False):
    if cached:
        from vecgpt.data import CachedStream
        assert isinstance(stream, CachedStream)
        scenes, imgs = stream.batch(cfg.batch_size)
    else:
        scenes = stream.batch(cfg.batch_size)
        imgs = render_batch(scenes, size=cfg.image_size, softness_px=cfg.render_softness_px, device=device)
    return scenes, imgs, collate(
        scenes, device, hierarchical=cfg.hierarchical_regions
    )


def _corrupt_numeric_prefix(tokens: torch.Tensor, p: float) -> torch.Tensor:
    """Small legal perturbations for rollout robustness.

    Structure is left untouched, so the grammar labels remain valid.  This is
    enabled only after exact teacher-forced reconstruction has passed its
    sanity gate.
    """
    if p <= 0:
        return tokens
    import vecgpt.schema as SC

    out = tokens.clone()
    choose = torch.rand_like(tokens.float()) < p
    for name, (lo, hi) in SC.RANGE.items():
        m = choose & (tokens >= lo) & (tokens < hi)
        if not m.any():
            continue
        delta = torch.randint(-2, 3, (int(m.sum()),), device=tokens.device)
        value = tokens[m] - lo
        if SC.QUANTS[name].wraps:
            value = (value + delta) % (hi - lo)
        else:
            value = (value + delta).clamp(0, hi - lo - 1)
        out[m] = value + lo
    return out


def _balanced_ce(token_loss, targets, tgt_slots, valid,
                 anchor_weight: float = 2.0):
    """Average fields, not tokens, so long paths do not drown their anchors."""
    fid = metric_field_ids(targets, tgt_slots)
    terms, weights = [], []

    # All structural specials form one group.  Treating EOS, EOL, STY and
    # ENDR as four full fields would over-weight trivial syntax.
    structure = valid & (fid >= 0) & (fid < 4)
    if structure.any():
        terms.append(token_loss[structure].mean())
        # Syntax is one group rather than four, but weight 1 left it at only
        # ~6% of a one-stroke objective. The differentiable renderer uses
        # target structure and therefore cannot teach TOP to choose x instead
        # of EOS; under-weighting this group produced visually empty greedy
        # rollouts while teacher-forced geometry was improving.
        weights.append(4.0)

    anchor_names = {"x", "y", "theta", "rx", "ry", "rt"}
    for i, name in enumerate(METRIC_FIELDS[4:], start=4):
        m = valid & (fid == i)
        if not m.any():
            continue
        terms.append(token_loss[m].mean())
        weights.append(anchor_weight if name in anchor_names else 1.0)
    if not terms:
        return token_loss[valid].mean()
    w = token_loss.new_tensor(weights)
    return (torch.stack(terms) * w).sum() / w.sum()


def _numeric_distance_loss(logits, targets, valid):
    """Ordinal/circular distance between numeric token distributions.

    Exact CE is retained.  This auxiliary term merely tells the model that a
    length one bin away is better than a length 200 bins away, information a
    flat categorical loss throws away.
    """
    import vecgpt.schema as SC

    terms = []
    for name, q in SC.QUANTS.items():
        lo, hi = SC.RANGE[name]
        m = valid & (targets >= lo) & (targets < hi)
        if not m.any():
            continue
        p = logits[m][:, lo:hi].softmax(-1)
        target_i = (targets[m] - lo).to(p.dtype)
        bins = torch.arange(q.n, device=p.device, dtype=p.dtype)[None]
        delta = (bins - target_i[:, None]).abs()
        if q.wraps:
            delta = torch.minimum(delta, q.n - delta)
            scale = max(q.n / 2, 1)
        else:
            scale = max(q.n - 1, 1)
        # Smooth bounded geometry cost. Its unique minimum is the target bin,
        # unlike a loss on the distribution mean.
        cost = (delta / scale).square()
        terms.append((p * cost).sum(-1).mean())
    return torch.stack(terms).mean() if terms else logits.new_zeros(())


def _grammar_edge_loss(logits, targets, target_states, valid):
    """Choose a grammar command independently of its vocabulary cardinality.

    At SEG, one EOL token competes with 256 length bins. Numeric CE teaches the
    exact bin but is a poor command classifier. This loss first compares the
    summed probability of EOL vs STYLE vs NEXT_SEGMENT, matching generate()'s
    hierarchical decision.
    """
    import vecgpt.schema as SC

    terms = []
    for state_name, state in SC.GRAMMAR.items():
        if len(state.edges) <= 1:
            continue
        sid = SC.STATE_ID[state_name]
        m = valid & (target_states == sid)
        if not m.any():
            continue
        rows = logits[m]
        group_logits = []
        target_group = torch.full(
            (rows.shape[0],), -1, dtype=torch.long, device=rows.device
        )
        target_tokens = targets[m]
        for edge_i, edge in enumerate(state.edges):
            if edge.emit in SC.SPECIAL_TOK:
                lo = SC.SPECIAL_TOK[edge.emit]
                hi = lo + 1
            else:
                lo, hi = SC.RANGE[edge.emit]
            group_logits.append(torch.logsumexp(rows[:, lo:hi], -1))
            hit = (target_tokens >= lo) & (target_tokens < hi)
            target_group[hit] = edge_i
        if (target_group < 0).any():
            raise RuntimeError(f"target not covered by grammar state {state_name}")
        terms.append(F.cross_entropy(torch.stack(group_logits, -1), target_group))
    return torch.stack(terms).mean() if terms else logits.new_zeros(())


def _region_mask_loss(model, imgs, b):
    """Foreground coverage for dynamic masks, with no object labels.

    Only leaf regions are used for scene coverage; parents are allowed to
    cover their complete subtree.  Sibling competition itself is enforced by
    the residual scope construction in ``DynamicRegionCrossAttention``.
    """
    masks = model.layers[-1].xa.routing_masks
    if masks is None or masks.shape[2] <= 1:
        return imgs.new_zeros(())
    masks = masks.mean(1)  # [B,R,N]
    B, R, N = masks.shape
    grid = int(round(N ** 0.5))
    if grid * grid != N:
        return imgs.new_zeros(())

    present = torch.zeros(B, R, dtype=torch.bool, device=imgs.device)
    parents = torch.zeros(B, R, dtype=torch.long, device=imgs.device)
    for r in range(R):
        present[:, r] = (b["region_idx"] == r).any(1)
        take = (b["region_idx"] == r) & b["mask"]
        for i in range(B):
            if take[i].any():
                parents[i, r] = b["parent_region_idx"][i, take[i]][0]
    is_parent = torch.zeros_like(present)
    for r in range(1, R):
        for i in range(B):
            if present[i, r]:
                is_parent[i, parents[i, r].clamp_max(R - 1)] = True
    leaves = present & ~is_parent
    leaves[:, 0] = False
    active = leaves.any(1)
    if not active.any():
        return imgs.new_zeros(())

    selected = torch.where(leaves[:, :, None], masks, torch.zeros_like(masks))
    union = 1.0 - torch.prod(1.0 - selected.clamp(0, 1), dim=1)
    target = F.adaptive_max_pool2d(
        ink_map(imgs).unsqueeze(1), (grid, grid)
    ).squeeze(1).flatten(1)
    target = (target > 0.03).to(union.dtype)
    return F.binary_cross_entropy(
        union[active].clamp(1e-5, 1 - 1e-5), target[active]
    )


def loss_fn(model, imgs, b, smooth=None, cond_dropout=0.0,
            balanced_fields=False, anchor_weight=2.0,
            render_loss_weight=0.0, render_loss_size=32,
            numeric_distance_weight=0.0,
            grammar_edge_weight=0.0,
            condition_probe_weight=0.0,
            region_mask_loss_weight=0.0, prefix_corruption=0.0,
            condition_margin_weight=0.0, condition_margin=0.5,
            alignment_loss_weight=0.0):
    mem_clean, semantic_clean = model.encode_condition(imgs)
    mem = mem_clean
    semantic = semantic_clean
    keep = None
    if cond_dropout:
        mem, keep = model.drop_condition(mem, cond_dropout)
        k = keep[:, :, None].to(semantic.dtype)
        semantic = (
            semantic * k
            + model.null_semantic_memory(imgs.shape[0], imgs.device) * (1 - k)
        )
    input_tokens = _corrupt_numeric_prefix(b["tokens"], prefix_corruption)
    lg = model.logits(mem, input_tokens, b["slots"], b["seg_idx"],
                      b["stroke_idx"], b["mask"], b["region_idx"],
                      b["region_depth"], b["parent_region_idx"],
                      semantic_mem=semantic)
    logits = lg[:, :-1]
    targets = b["tokens"][:, 1:]
    tgt_slots = b["slots"][:, 1:]
    valid = b["mask"][:, 1:]
    if smooth is None:
        token_loss = -logits.log_softmax(-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
    else:
        token_loss = -(
            smooth[targets] * logits.log_softmax(-1).nan_to_num(neginf=0.0)
        ).sum(-1)
    token_ce = (
        _balanced_ce(token_loss, targets, tgt_slots, valid, anchor_weight)
        if balanced_fields else token_loss[valid].mean()
    )
    numeric_distance_loss = (
        _numeric_distance_loss(logits, targets, valid)
        if numeric_distance_weight > 0 else imgs.new_zeros(())
    )
    grammar_edge_loss = (
        _grammar_edge_loss(logits, targets, tgt_slots, valid)
        if grammar_edge_weight > 0 else imgs.new_zeros(())
    )
    condition_probe_loss = imgs.new_zeros(())
    if condition_probe_weight > 0:
        probe = model.condition_probe_logits(semantic_clean, tgt_slots)
        allowed_probe = model.slot_mask[tgt_slots]
        probe = probe.masked_fill(~allowed_probe, float("-inf"))
        if smooth is None:
            probe_token_loss = -probe.log_softmax(-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
        else:
            probe_token_loss = -(
                smooth[targets]
                * probe.log_softmax(-1).nan_to_num(neginf=0.0)
            ).sum(-1)
        # Probe only fields measured to ignore the image. Easy x/y/RGB already
        # shape the latent through the main decoder and otherwise dominate this
        # auxiliary objective again.
        probe_terms = []
        import vecgpt.schema as SC
        for field_name in ("theta", "len", "width"):
            lo, hi = SC.RANGE[field_name]
            m = valid & (targets >= lo) & (targets < hi)
            if m.any():
                probe_terms.append(probe_token_loss[m].mean())
        if probe_terms:
            condition_probe_loss = torch.stack(probe_terms).mean()

    # Conditional decoders have a strong teacher-forcing shortcut: predict
    # the marginal token prior and ignore the source. This bounded ranking
    # loss makes that collapse an explicit training violation. It stops
    # contributing once mismatched conditioning is sufficiently worse.
    condition_rank_loss = imgs.new_zeros(())
    if condition_margin_weight > 0 and imgs.shape[0] > 1:
        import vecgpt.schema as SC

        order = torch.roll(
            torch.arange(imgs.shape[0], device=imgs.device), shifts=1
        )
        wrong = model.logits(
            mem[order], input_tokens, b["slots"], b["seg_idx"],
            b["stroke_idx"], b["mask"], b["region_idx"],
            b["region_depth"], b["parent_region_idx"],
            semantic_mem=semantic[order],
        )[:, :-1]
        if smooth is None:
            wrong_loss = -wrong.log_softmax(-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
        else:
            wrong_loss = -(
                smooth[targets]
                * wrong.log_softmax(-1).nan_to_num(neginf=0.0)
            ).sum(-1)
        # Enforce conditioning use for EVERY numeric field independently.
        # The old pooled hinge was satisfied by easy x/y/RGB gains while len
        # still ignored the image completely (measured wrong-image gap 0.012).
        # Per-field hinges cannot hide one dead field behind another.
        rank_terms = []
        for field_name in (
            "x", "y", "theta", "len", "turn", "width",
            "color", "rx", "ry", "rt",
        ):
            lo, hi = SC.RANGE[field_name]
            m = (targets >= lo) & (targets < hi) & valid
            if m.any():
                clean_field = token_loss[m].mean()
                wrong_field = wrong_loss[m].mean()
                rank_terms.append(F.relu(
                    clean_field.detach() + condition_margin - wrong_field
                ))
        if rank_terms:
            condition_rank_loss = torch.stack(rank_terms).mean()

    alignment_loss = (
        model.condition_alignment_loss(semantic_clean, b["tokens"], b["mask"])
        if alignment_loss_weight > 0 and imgs.shape[0] > 1
        else imgs.new_zeros(())
    )

    render_loss = imgs.new_zeros(())
    render_parts = {
        name: imgs.new_zeros(())
        for name in ("transport", "coverage", "dice", "moments", "color")
    }
    if render_loss_weight > 0:
        from vecgpt.soft_decode import soft_decode_batch

        soft_scenes = soft_decode_batch(logits, b["tokens"], b["mask"])
        pred = render_batch(
            soft_scenes, size=render_loss_size,
            softness_px=max(0.75, render_loss_size / 64),
            device=imgs.device,
        )
        target_img = F.interpolate(
            imgs.permute(0, 3, 1, 2),
            size=(render_loss_size, render_loss_size),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)
        render_loss, render_parts = foreground_render_loss(pred, target_img)

    mask_loss = (
        _region_mask_loss(model, imgs, b)
        if region_mask_loss_weight > 0 else imgs.new_zeros(())
    )
    total = (
        token_ce
        + numeric_distance_weight * numeric_distance_loss
        + grammar_edge_weight * grammar_edge_loss
        + condition_probe_weight * condition_probe_loss
        + render_loss_weight * render_loss
        + region_mask_loss_weight * mask_loss
        + condition_margin_weight * condition_rank_loss
        + alignment_loss_weight * alignment_loss
    )
    aux = {
        "token_ce": token_ce.detach(),
        "numeric_distance_loss": numeric_distance_loss.detach(),
        "grammar_edge_loss": grammar_edge_loss.detach(),
        "condition_probe_loss": condition_probe_loss.detach(),
        "render_loss": render_loss.detach(),
        **{
            f"render_{name}": value
            for name, value in render_parts.items()
        },
        "region_mask_loss": mask_loss.detach(),
        "condition_rank_loss": condition_rank_loss.detach(),
        "alignment_loss": alignment_loss.detach(),
    }
    return (total, logits, targets, tgt_slots, valid, keep, mem_clean,
            semantic_clean, aux)


@torch.no_grad()
def diagnostic_ce(model, mem, b, smooth=None, mode="clean",
                  semantic_mem=None):
    """CE total and per field under controlled conditioning interventions.

    Modes:
      clean          correct image and oracle token prefix
      shuffled_image wrong image and oracle token prefix
      perturbed_plan correct image and shuffled rx/ry/rt prefix tokens

    Comparing all three distinguishes image use from plan use. The old
    aggregate gap only showed that the image helped *somewhere* and compared
    clean memory against a training loss that included condition dropout.
    """
    if mode not in {"clean", "shuffled_image", "perturbed_plan"}:
        raise ValueError(mode)

    use_mem = mem
    use_semantic = semantic_mem
    tokens = b["tokens"]
    if mode == "shuffled_image":
        order = torch.randperm(mem.shape[0], device=mem.device)
        use_mem = mem[order]
        if semantic_mem is not None:
            use_semantic = semantic_mem[order]
    elif mode == "perturbed_plan":
        import vecgpt.schema as SC

        tokens = tokens.clone()
        for field in ("rx", "ry", "rt"):
            lo, hi = SC.RANGE[field]
            m = (tokens >= lo) & (tokens < hi) & b["mask"]
            vals = tokens[m]
            if vals.numel() > 1:
                tokens[m] = vals[torch.randperm(vals.numel(), device=vals.device)]

    lg = model.logits(use_mem, tokens, b["slots"], b["seg_idx"],
                      b["stroke_idx"], b["mask"], b["region_idx"],
                      b["region_depth"], b["parent_region_idx"],
                      semantic_mem=use_semantic)
    logits = lg[:, :-1]
    targets = b["tokens"][:, 1:]
    tgt_slots = b["slots"][:, 1:]
    valid = b["mask"][:, 1:]
    if smooth is None:
        token_ce = -logits.log_softmax(-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
    else:
        token_ce = -(
            smooth[targets] * logits.log_softmax(-1).nan_to_num(neginf=0.0)
        ).sum(-1)

    fields = {}
    fid = metric_field_ids(targets, tgt_slots)
    for i, name in enumerate(METRIC_FIELDS):
        m = valid & (fid == i)
        if m.any():
            fields[name] = float(token_ce[m].mean())
    return float(token_ce[valid].mean()), fields


@torch.no_grad()
def dump_preview(model, cfg, device, stage, step, tag=""):
    """Two rows: target / greedy reconstruction.

    Unconditional sampling is intentionally absent while ``cond_dropout=0``:
    that prior receives no training signal, so displaying its guaranteed
    garbage beside reconstruction was actively misleading.
    """
    import random

    from vecgpt.data import sample_scene

    rng = random.Random(4321)
    scenes = [sample_scene(rng, stage) for _ in range(cfg.preview_n)]
    tgt = render_batch(scenes, size=cfg.image_size, softness_px=cfg.render_softness_px,
                       device=device)
    was = model.training
    model.eval()
    mem, semantic = model.encode_condition(tgt)
    token_budget = max(
        32,
        max(len(encode(s, hierarchical=cfg.hierarchical_regions).tokens)
            for s in scenes) + 32,
    )
    recon = [decode(s) for s in model.generate(
        mem, max_tokens=token_budget, temperature=0.0,
        semantic_mem=semantic
    )]
    model.train(was)

    r = lambda sc: render_batch(sc, size=cfg.image_size,
                                softness_px=cfg.render_softness_px, device=device).cpu()
    path = os.path.join(cfg.out_dir, "previews", f"s{stage}_{step:07d}{tag}.png")
    save_grid([list(tgt.cpu()), list(r(recon))], path)
    return path


@torch.no_grad()
def evaluate(model, cfg, device, stage, n=None, ood=False):
    import random
    n = cfg.eval_n if n is None else n

    rng = random.Random(1234)
    from vecgpt.data import sample_scene
    from vecgpt.scene import canonicalize

    if ood:
        if stage == 5:
            family_names = list(TREE_OOD)
            n_per = max(n // len(family_names), 1)
            by_family = {}
            for family in family_names:
                by_family[family] = [
                    sample_tree(random.Random(rng.randint(0, 2 ** 31 - 1)),
                                variant=family)
                    for _ in range(n_per)
                ]
        else:
            family_names = list(OOD_NAMES)
            n_per = max(n // len(family_names), 1)
            by_family = gen_ood_batch(
                family_names, n_per, seed=rng.randint(0, 2 ** 31 - 1)
            )
        scenes = [s for fam in family_names for s in by_family[fam]]
    else:
        scenes = [sample_scene(rng, stage) for _ in range(n)]

    imgs = render_batch(scenes, size=cfg.image_size, softness_px=cfg.render_softness_px, device=device)
    mem, semantic = model.encode_condition(imgs)

    # Greedy AND sampled, because greedy is the wrong decoder for a target
    # that is legitimately multi-modal. A closed stroke (circle, polygon)
    # can start anywhere on its own outline - `canonicalize` fixes the
    # traversal DIRECTION for closed shapes but not the starting point, and
    # a circle is generated as a single arc whose start angle is uniformly
    # random, so no model can recover it from the picture. The argmax over
    # "every point on this circle" is an average of those points, i.e. a
    # point that is not on the circle at all, and everything downstream is
    # then drawn from the wrong place. Sampling commits to ONE valid start
    # and draws a correct shape from it.
    # Greedy only. Sampling everything at T=0.8 was measured WORSE
    # (0.278 vs 0.446): it randomises the many fields that are already
    # exact (width/colour/turn all at mae 0) to chase ambiguity in two
    # anchor tokens. Greedy is the right decoder here.
    token_budget = max(
        32,
        max(len(encode(s, hierarchical=cfg.hierarchical_regions).tokens)
            for s in scenes) + 32,
    )
    recon = [decode(x) for x in model.generate(
        mem, max_tokens=token_budget, temperature=0.0,
        semantic_mem=semantic
    )]
    sampled = recon
    skip_empty = sum(1 for r in recon if not r)
    pred_imgs = render_batch(recon, size=cfg.image_size, softness_px=cfg.render_softness_px, device=device)
    iou = image_iou(pred_imgs, imgs)
    iou_s = image_iou_shape(pred_imgs, imgs)  # blur-tolerant: shape, not registration

    result = dict(
        iou=float(iou.mean()), iou_std=float(iou.std()),
        iou_shape=float(iou_s.mean()),
        empty=skip_empty, n=len(scenes),
        n_strokes_pred=sum(len(r) for r in recon) / max(len(scenes), 1),
        n_strokes_true=sum(len(s) for s in scenes) / max(len(scenes), 1),
    )

    if ood:
        families = {}
        idx = 0
        for fam_name in family_names:
            f_scenes = by_family[fam_name]
            f_n = len(f_scenes)
            f_iou = float(iou[idx:idx + f_n].mean())
            f_iou_s = float(iou_s[idx:idx + f_n].mean())
            f_std = float(iou[idx:idx + f_n].std())
            f_empty = sum(1 for r in recon[idx:idx + f_n] if not r)
            f_pred = sum(len(r) for r in recon[idx:idx + f_n]) / max(f_n, 1)
            f_true = sum(len(s) for s in f_scenes) / max(f_n, 1)
            families[fam_name] = dict(
                iou=f_iou, iou_std=f_std, iou_shape=f_iou_s, empty=f_empty,
                n_strokes_pred=f_pred, n_strokes_true=f_true,
            )
            idx += f_n
        result["families"] = families

    return result


def throughput_probe(model, opt, cfg, device, smooth, n=15):
    from vecgpt.data import CachedStream

    use_cache = bool(cfg.cache_dir)
    stream = (CachedStream(1, cfg.cache_dir, device, batch_size=cfg.batch_size) if use_cache
              else SceneStream(stage=1, seed=999))
    for _ in range(3):
        _, imgs, b = make_batch(stream, cfg, device, cached=use_cache)
        ce, *_ = loss_fn(
            model, imgs, b, smooth,
            cond_dropout=cfg.cond_dropout,
            balanced_fields=cfg.balanced_field_loss,
            anchor_weight=cfg.anchor_loss_weight,
            render_loss_weight=cfg.render_loss_weight,
            render_loss_size=cfg.render_loss_size,
            numeric_distance_weight=cfg.numeric_distance_weight,
            grammar_edge_weight=cfg.grammar_edge_weight,
            condition_probe_weight=cfg.condition_probe_weight,
            region_mask_loss_weight=cfg.region_mask_loss_weight,
            prefix_corruption=cfg.prefix_corruption,
            condition_margin_weight=cfg.condition_margin_weight,
            condition_margin=cfg.condition_margin,
            alignment_loss_weight=cfg.alignment_loss_weight,
        )
        opt.zero_grad(set_to_none=True)
        ce.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        _, imgs, b = make_batch(stream, cfg, device, cached=use_cache)
        ce, *_ = loss_fn(
            model, imgs, b, smooth,
            cond_dropout=cfg.cond_dropout,
            balanced_fields=cfg.balanced_field_loss,
            anchor_weight=cfg.anchor_loss_weight,
            render_loss_weight=cfg.render_loss_weight,
            render_loss_size=cfg.render_loss_size,
            region_mask_loss_weight=cfg.region_mask_loss_weight,
            prefix_corruption=cfg.prefix_corruption,
            condition_margin_weight=cfg.condition_margin_weight,
            condition_margin=cfg.condition_margin,
            alignment_loss_weight=cfg.alignment_loss_weight,
        )
        opt.zero_grad(set_to_none=True)
        ce.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t) / n
    opt.zero_grad(set_to_none=True)
    return dt


def train(cfg: Cfg = Cfg()):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(cfg.out_dir, exist_ok=True)

    model = VecGPT(cfg.image_size, cfg.d_model, cfg.n_heads, cfg.n_layers,
                   cfg.n_seg_heads, cfg.n_stroke_heads, enc_base=cfg.enc_base,
                   n_enc_layers=cfg.n_enc_layers, spatial_bias=cfg.spatial_bias,
                   region_attention=cfg.region_attention,
                   n_global_heads=cfg.n_global_heads,
                   dynamic_region_masks=cfg.dynamic_region_masks,
                   condition_dim=cfg.condition_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    total = sum(s for _, s in cfg.stage_schedule)
    # A short diagnostic schedule used to cosine-decay all the way to zero
    # exactly when conditioning would normally start beating the easy token
    # prior. An explicit value still wins; otherwise short gates keep a useful
    # LR while full curricula retain their natural total-length schedule.
    lr_total = cfg.lr_total_steps or max(total, 10_000)
    smooth = (build_smoothing_matrix(cfg.label_smooth_bins, device)
              if cfg.label_smooth_bins > 0 else None)

    sec_per_step = throughput_probe(model, opt, cfg, device, smooth)
    log_path = os.path.join(cfg.out_dir, "log.jsonl")
    meta = dict(
        params_M=round(count_params(model), 3), vocab=VOCAB, device=str(device),
        architecture_version=ARCHITECTURE_VERSION,
        total_steps=total, sec_per_step=round(sec_per_step, 4),
        scenes_per_sec=round(cfg.batch_size / sec_per_step, 1),
        eta_hours=round(total * sec_per_step / 3600, 2),
        stage_eta_min={str(st): round(n * sec_per_step / 60, 1) for st, n in cfg.stage_schedule},
        torch=torch.__version__, cfg=cfg.__dict__,
    )
    run_name = ("resume_run.json"
                if cfg.resume and os.path.exists(os.path.join(cfg.out_dir, "run.json"))
                else "run.json")
    with open(os.path.join(cfg.out_dir, run_name), "w") as f:
        json.dump(meta, f, indent=2)
    if not cfg.resume:
        open(log_path, "w").close()

    print("=" * 78, flush=True)
    print(f"device={device}  params={meta['params_M']}M  vocab={VOCAB}", flush=True)
    print(f"measured: {sec_per_step*1000:.0f} ms/step  =  {meta['scenes_per_sec']} scenes/s "
          f"(batch {cfg.batch_size})", flush=True)
    print(f"schedule: {total} steps  ->  ETA {meta['eta_hours']:.2f} h   "
          + "  ".join(f"stage{st}~{m}min" for st, m in meta["stage_eta_min"].items()), flush=True)
    print(f"logging to {log_path}  (hand this file back for analysis)", flush=True)
    print("=" * 78, flush=True)

    def jlog(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    jlog(dict(kind="meta", **{k: v for k, v in meta.items() if k != "cfg"}, cfg=cfg.__dict__))

    step, resume_stage_idx, resume_i = 0, 0, 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        if ck.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError(
                f"checkpoint architecture v{ck.get('architecture_version', 10)} "
                f"cannot resume v{ARCHITECTURE_VERSION}; keep it as the flat "
                "baseline or start a new hierarchical run"
            )
        incompatible = model.load_state_dict(ck["model"], strict=False)
        allowed_missing = {
            f"layers.{i}.xa.local_raw" for i in range(len(model.layers))
        }
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise ValueError(
                f"incompatible checkpoint keys: missing={sorted(unexpected_missing)}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        if incompatible.missing_keys:
            print(
                "initialized new local-cross-attention parameters while loading "
                "the compatible decoder checkpoint",
                flush=True,
            )
        if "opt" in ck:
            try:
                opt.load_state_dict(ck["opt"])
                print(f"resumed model AND optimizer from {cfg.resume}", flush=True)
            except ValueError as e:
                print(
                    f"resumed model from {cfg.resume}; optimizer restarted because "
                    f"the parameter set grew ({e})",
                    flush=True,
                )
        else:
            print(f"resumed model from {cfg.resume} -- NO optimizer state in this "
                  f"checkpoint, Adam moments restart from zero. Expect a few hundred "
                  f"steps of degraded loss before it recovers.", flush=True)
        step = int(ck["step"]) + 1
        # locate (stage, step-within-stage) from the global counter, so this
        # works regardless of how the file was named
        acc = 0
        for si, (st, n) in enumerate(cfg.stage_schedule):
            if step < acc + n:
                resume_stage_idx, resume_i = si, step - acc
                break
            acc += n
        else:
            print("checkpoint is past the end of this schedule; nothing to do", flush=True)
            return model
        print(f"continuing at global step {step} = stage "
              f"{cfg.stage_schedule[resume_stage_idx][0]}, step {resume_i} within it",
              flush=True)

    t0 = time.time()
    for stage_idx, (stage, n_steps) in enumerate(cfg.stage_schedule):
        if stage_idx < resume_stage_idx:
            continue
        first_i = resume_i if stage_idx == resume_stage_idx else 0
        # reseed past the point already consumed, so a resume does not replay
        # the exact scenes the run has already trained on
        use_cache = bool(cfg.cache_dir)
        if use_cache:
            from vecgpt.data import CachedStream
            skip = first_i * cfg.batch_size
            stream = CachedStream(stage=stage, cache_dir=cfg.cache_dir,
                                  device=device, skip_batches=first_i,
                                  batch_size=cfg.batch_size)
        else:
            stream = SceneStream(stage=stage, seed=cfg.seed + stage * 1009 + first_i)
        stats = FieldStats(device)
        last_eval = None
        print(f"\n=== stage {stage}: steps {first_i}..{n_steps} ===", flush=True)
        for i in range(first_i, n_steps):
            lr = cfg.lr * min(1.0, (step + 1) / cfg.warmup) * (
                0.5 * (1 + math.cos(math.pi * min(1.0, step / lr_total)))
            )
            for g in opt.param_groups:
                g["lr"] = lr

            _, imgs, b = make_batch(stream, cfg, device, cached=use_cache)
            ce, logits, targets, tgt_slots, valid, keep, mem, semantic, aux = loss_fn(
                model, imgs, b, smooth,
                cond_dropout=cfg.cond_dropout,
                balanced_fields=cfg.balanced_field_loss,
                anchor_weight=cfg.anchor_loss_weight,
                render_loss_weight=cfg.render_loss_weight,
                render_loss_size=cfg.render_loss_size,
                numeric_distance_weight=cfg.numeric_distance_weight,
                grammar_edge_weight=cfg.grammar_edge_weight,
                condition_probe_weight=cfg.condition_probe_weight,
                region_mask_loss_weight=cfg.region_mask_loss_weight,
                prefix_corruption=cfg.prefix_corruption,
                condition_margin_weight=cfg.condition_margin_weight,
                condition_margin=cfg.condition_margin,
                alignment_loss_weight=cfg.alignment_loss_weight,
            )
            opt.zero_grad(set_to_none=True)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            # Examples trained with the conditioning dropped are BLIND by
            # construction; including them in gain/mae reports the model as
            # worse than it is, on a metric whose whole job is to say what
            # the model can read from the image.
            stats.update(logits.detach(), targets, tgt_slots,
                         valid if keep is None else valid & keep)

            if i % cfg.log_every == 0 or i == n_steps - 1:
                rows = stats.report()
                geo = [r for r in rows if r["field"] not in ("eos", "eol")]
                flat = [r["field"] for r in geo if r["gain"] < 0.05]
                clean_ce, clean_fields = diagnostic_ce(
                    model, mem, b, smooth, "clean", semantic
                )
                shuffle_ce, shuffle_fields = diagnostic_ce(
                    model, mem, b, smooth, "shuffled_image", semantic
                )
                plan_ce, plan_fields = diagnostic_ce(
                    model, mem, b, smooth, "perturbed_plan", semantic
                )
                image_gap = shuffle_ce - clean_ce
                plan_gap = plan_ce - clean_ce
                field_controls = {
                    name: {
                        "clean_ce": clean_fields[name],
                        "image_use": shuffle_fields.get(name, clean_fields[name])
                                     - clean_fields[name],
                        "plan_use": plan_fields.get(name, clean_fields[name])
                                    - clean_fields[name],
                    }
                    for name in clean_fields
                }
                print(f"[s{stage} {i:6d}/{n_steps}] ce={clean_ce:.3f} "
                      f"train={float(aux['token_ce']):.3f} "
                      f"ord={float(aux['numeric_distance_loss']):.3f} "
                      f"edge={float(aux['grammar_edge_loss']):.3f} "
                      f"probe={float(aux['condition_probe_loss']):.3f} "
                      f"render={float(aux['render_loss']):.3f} "
                      f"(move={float(aux['render_transport']):.2f} "
                      f"cov={float(aux['render_coverage']):.2f} "
                      f"dice={float(aux['render_dice']):.2f} "
                      f"mom={float(aux['render_moments']):.2f} "
                      f"rgb={float(aux['render_color']):.2f}) "
                      f"mask={float(aux['region_mask_loss']):.3f} "
                      f"rank={float(aux['condition_rank_loss']):.3f} "
                      f"align={float(aux['alignment_loss']):.3f} "
                      f"image_gap={image_gap:+.3f} plan_gap={plan_gap:+.3f} lr={lr:.2e} "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
                print("    " + "  ".join(f"{r['field']}:{r['gain']:+.2f}/{r['mae_bins']:.0f}"
                                         for r in rows), flush=True)
                if flat:
                    print(f"    FLAT (gain<0.05, reading nothing from the image): "
                          f"{', '.join(flat)}", flush=True)
                jlog(dict(kind="train", step=step, stage=stage, stage_step=i,
                          ce=clean_ce, shuffle_ce=shuffle_ce,
                          leak_gap=image_gap, plan_ce=plan_ce,
                          image_gap=image_gap, plan_gap=plan_gap,
                          train_token_ce=float(aux["token_ce"]),
                          numeric_distance_loss=float(aux["numeric_distance_loss"]),
                          grammar_edge_loss=float(aux["grammar_edge_loss"]),
                          condition_probe_loss=float(aux["condition_probe_loss"]),
                          render_loss=float(aux["render_loss"]),
                          render_transport=float(aux["render_transport"]),
                          render_coverage=float(aux["render_coverage"]),
                          render_dice=float(aux["render_dice"]),
                          render_moments=float(aux["render_moments"]),
                          render_color=float(aux["render_color"]),
                          region_mask_loss=float(aux["region_mask_loss"]),
                          condition_rank_loss=float(aux["condition_rank_loss"]),
                          alignment_loss=float(aux["alignment_loss"]),
                          field_controls=field_controls,
                          lr=lr, elapsed_s=round(time.time() - t0, 1), fields=rows, flat=flat))
                stats.reset_window()

            if (i + 1) % cfg.eval_every == 0 or i == n_steps - 1:
                model.eval()
                ind = evaluate(model, cfg, device, stage)
                last_eval = ind
                oodm = evaluate(model, cfg, device, stage, ood=True)
                model.train()
                ceil = tokenizer_ceiling(cfg, device, stage)
                print(f"    EVAL  greedy IoU={ind['iou']:.3f}+-{ind['iou_std']:.3f} "
                      f"| shape IoU={ind.get('iou_shape', float('nan')):.3f} "
                      f"(ceiling {ceil:.3f}, so {100*ind['iou']/max(ceil,1e-6):.0f}% of what "
                      f"this tokenizer allows)  strokes {ind['n_strokes_pred']:.1f}/"
                      f"{ind['n_strokes_true']:.1f}  empty={ind['empty']}/{ind['n']}", flush=True)
                print(f"          OOD greedy={oodm['iou']:.3f} sampled={oodm.get('iou_sampled', float('nan')):.3f} empty={oodm['empty']}/{oodm['n']}  "
                      f"(held-out shape families: star/spiral/cross/blob)", flush=True)
                if "families" in oodm:
                    fams = oodm["families"]
                    parts = []
                    for fn, f in fams.items():
                        parts.append(f"{fn} {f['iou']:.3f}|e{f['empty']}")
                    print(f"          per family: " + "  ".join(parts), flush=True)
                jlog(dict(kind="eval", step=step, stage=stage, stage_step=i,
                          in_dist=ind, ood=oodm, ceiling=ceil))

            if cfg.preview_every and ((i + 1) % cfg.preview_every == 0 or i == n_steps - 1):
                try:
                    pth = dump_preview(model, cfg, device, stage, step)
                    print(f"    preview -> {pth}  "
                          f"(rows: target / reconstruction)", flush=True)
                except Exception as e:  # PIL missing etc. - never kill a run for a picture
                    print(f"    preview skipped: {type(e).__name__}: {e}", flush=True)

            if (i + 1) % cfg.ckpt_every == 0:
                blob = {"model": model.state_dict(), "opt": opt.state_dict(),
                        "architecture_version": ARCHITECTURE_VERSION,
                        "cfg": cfg.__dict__, "step": step, "stage": stage, "stage_step": i + 1}
                torch.save(blob, f"{cfg.out_dir}/stage{stage}_{i+1:06d}.pt")
                torch.save(blob, f"{cfg.out_dir}/latest.pt")  # always a resume point
            step += 1

        gate = cfg.mastery_shape_gates.get(
            stage, cfg.mastery_shape_gates.get(str(stage))
        )
        if cfg.mastery_gates and gate is not None:
            score = (
                float(last_eval["iou_shape"])
                if last_eval is not None else float("-inf")
            )
            if score < float(gate):
                blob = {
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "architecture_version": ARCHITECTURE_VERSION,
                    "cfg": cfg.__dict__, "step": step - 1, "stage": stage,
                    "stage_step": n_steps,
                }
                torch.save(blob, f"{cfg.out_dir}/failed_mastery_gate.pt")
                torch.save(blob, f"{cfg.out_dir}/latest.pt")
                raise RuntimeError(
                    f"stage {stage} mastery gate failed: shape IoU "
                    f"{score:.3f} < {float(gate):.3f}. Refusing to advance "
                    "the curriculum; inspect the preview and fix/extend this "
                    "stage first."
                )

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "architecture_version": ARCHITECTURE_VERSION,
                "cfg": cfg.__dict__, "step": step}, f"{cfg.out_dir}/final.pt")
    print(f"\ndone in {(time.time()-t0)/3600:.2f} h. "
          f"Hand back: {cfg.out_dir}/log.jsonl (+ run.json)", flush=True)
    return model


@torch.no_grad()
def tokenizer_ceiling(cfg, device, stage, n=None):
    """IoU of a scene reconstructed from its OWN tokens - the best score
    any model could reach with this token grid. Reported next to the
    model's IoU so a low number can be attributed to the right thing."""
    import random
    n = cfg.eval_n if n is None else n

    from vecgpt.data import sample_scene
    from vecgpt.tokenizer import encode

    rng = random.Random(1234)
    scenes = [sample_scene(rng, stage) for _ in range(n)]
    a = render_batch(scenes, size=cfg.image_size, softness_px=cfg.render_softness_px, device=device)
    b = render_batch([
        decode(encode(s, hierarchical=cfg.hierarchical_regions).tokens)
        for s in scenes
    ], size=cfg.image_size,
                     softness_px=cfg.render_softness_px, device=device)
    return float(image_iou(b, a).mean())
