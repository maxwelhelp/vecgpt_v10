#!/usr/bin/env python
"""Train VecGPT.

Defaults are sized for a single 24 GB Tesla P40 in fp32 (Pascal's fp16 runs
at 1/64 rate, so AMP is a pessimisation there, not an optimisation).

    python scripts/train.py                       # full curriculum
    python scripts/train.py --stages 1:2000,2:6000
    python scripts/train.py --d-model 384 --n-layers 8 --batch-size 48
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vecgpt.train import Cfg, train


def parse_stages(s: str):
    out = []
    for part in s.split(","):
        stage, steps = part.split(":")
        out.append((int(stage), int(steps)))
    return tuple(out)


def parse_gates(s: str):
    if not s:
        return {}
    return {
        int(part.split(":")[0]): float(part.split(":")[1])
        for part in s.split(",")
    }


def main():
    d = Cfg()
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--image-size", type=int, default=d.image_size)
    p.add_argument("--d-model", type=int, default=d.d_model)
    p.add_argument("--n-layers", type=int, default=d.n_layers)
    p.add_argument("--n-heads", type=int, default=d.n_heads)
    p.add_argument("--n-seg-heads", type=int, default=d.n_seg_heads)
    p.add_argument("--n-stroke-heads", type=int, default=d.n_stroke_heads)
    p.add_argument("--enc-base", type=int, default=d.enc_base)
    p.add_argument("--n-enc-layers", type=int, default=d.n_enc_layers)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--warmup", type=int, default=d.warmup)
    p.add_argument("--lr-total-steps", type=int, default=d.lr_total_steps,
                   help="cosine horizon; default=max(sum of stages, 10000)")
    p.add_argument("--label-smooth-bins", type=float, default=d.label_smooth_bins)
    p.add_argument("--stages", type=str, default="1:2000,2:6000,3:20000,4:30000")
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--eval-n", type=int, default=d.eval_n)
    p.add_argument("--ckpt-every", type=int, default=d.ckpt_every)
    p.add_argument("--spatial-bias", type=int, default=int(d.spatial_bias),
                   help="1/0; learnable per-head attention penalty on anchor distance")
    p.add_argument("--region-attention", type=int, default=int(d.region_attention),
                   help="1/0; local heads see own region and ancestors")
    p.add_argument("--n-global-heads", type=int, default=d.n_global_heads,
                   help="heads left globally causal when region attention is enabled")
    p.add_argument("--hierarchical-regions", type=int,
                   default=int(d.hierarchical_regions),
                   help="1=nested region ownership, 0=one flat identity region")
    p.add_argument("--dynamic-region-masks", type=int,
                   default=int(d.dynamic_region_masks),
                   help="1=REGION tokens learn soft heatmaps over encoder memory")
    p.add_argument("--cond-dropout", type=float, default=d.cond_dropout)
    p.add_argument("--condition-dim", type=int, default=d.condition_dim,
                   help="dimension of external LLM/text latent memory")
    p.add_argument("--balanced-field-loss", type=int,
                   default=int(d.balanced_field_loss),
                   help="1=average fields instead of letting long paths dominate")
    p.add_argument("--anchor-loss-weight", type=float,
                   default=d.anchor_loss_weight)
    p.add_argument("--render-loss-weight", type=float,
                   default=d.render_loss_weight,
                   help="auxiliary differentiable raster loss during visual bootstrap")
    p.add_argument("--render-loss-size", type=int, default=d.render_loss_size)
    p.add_argument("--numeric-distance-weight", type=float,
                   default=d.numeric_distance_weight,
                   help="ordinal/circular auxiliary loss for numeric token bins")
    p.add_argument("--grammar-edge-weight", type=float,
                   default=d.grammar_edge_weight,
                   help="command-level CE independent of numeric-bin cardinality")
    p.add_argument("--condition-probe-weight", type=float,
                   default=d.condition_probe_weight,
                   help="field/state probe forcing the visual latent to retain every parameter")
    p.add_argument("--region-mask-loss-weight", type=float,
                   default=d.region_mask_loss_weight)
    p.add_argument("--condition-margin-weight", type=float,
                   default=d.condition_margin_weight,
                   help="bounded wrong-condition ranking loss")
    p.add_argument("--condition-margin", type=float,
                   default=d.condition_margin)
    p.add_argument("--alignment-loss-weight", type=float,
                   default=d.alignment_loss_weight,
                   help="image/program semantic-space InfoNCE")
    p.add_argument("--prefix-corruption", type=float,
                   default=d.prefix_corruption,
                   help="legal numeric-prefix corruption; enable only after overfit gate")
    p.add_argument("--mastery-gates", type=int, default=int(d.mastery_gates),
                   help="refuse to advance when a simple stage is not mastered")
    p.add_argument("--mastery-shape-gates",
                   default="0:0.75,1:0.70,2:0.55,3:0.55")
    p.add_argument("--preview-every", type=int, default=d.preview_every)
    p.add_argument("--preview-n", type=int, default=d.preview_n)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=d.out_dir)
    p.add_argument("--cache-dir", type=str, default="",
                   help="pre-rendered cache dir (build_cache.py), enables CachedStream")
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint to continue from, e.g. runs/v2/latest.pt")
    a = p.parse_args()

    cfg = Cfg(
        image_size=a.image_size, d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads,
        n_seg_heads=a.n_seg_heads, n_stroke_heads=a.n_stroke_heads, enc_base=a.enc_base,
        n_enc_layers=a.n_enc_layers,
        batch_size=a.batch_size, lr=a.lr, warmup=a.warmup,
        lr_total_steps=a.lr_total_steps,
        label_smooth_bins=a.label_smooth_bins, stage_schedule=parse_stages(a.stages),
        log_every=a.log_every, eval_every=a.eval_every, ckpt_every=a.ckpt_every,
        eval_n=a.eval_n,
        spatial_bias=bool(a.spatial_bias), cond_dropout=a.cond_dropout,
        condition_dim=a.condition_dim,
        balanced_field_loss=bool(a.balanced_field_loss),
        anchor_loss_weight=a.anchor_loss_weight,
        render_loss_weight=a.render_loss_weight,
        render_loss_size=a.render_loss_size,
        numeric_distance_weight=a.numeric_distance_weight,
        grammar_edge_weight=a.grammar_edge_weight,
        condition_probe_weight=a.condition_probe_weight,
        region_mask_loss_weight=a.region_mask_loss_weight,
        condition_margin_weight=a.condition_margin_weight,
        condition_margin=a.condition_margin,
        alignment_loss_weight=a.alignment_loss_weight,
        prefix_corruption=a.prefix_corruption,
        mastery_gates=bool(a.mastery_gates),
        mastery_shape_gates=parse_gates(a.mastery_shape_gates),
        region_attention=bool(a.region_attention), n_global_heads=a.n_global_heads,
        hierarchical_regions=bool(a.hierarchical_regions),
        dynamic_region_masks=bool(a.dynamic_region_masks),
        preview_every=a.preview_every, preview_n=a.preview_n,
        seed=a.seed, device=a.device, out_dir=a.out_dir, resume=a.resume,
        cache_dir=a.cache_dir,
    )
    train(cfg)


if __name__ == "__main__":
    main()
