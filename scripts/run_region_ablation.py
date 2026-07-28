#!/usr/bin/env python
"""Matched A0/A1/B1 region ablation.

All variants use the same model size, scene caches, curriculum and seeds:

  A0 flat         one identity workspace, global attention
  A1 hierarchy    nested ownership, global attention
  B1 region_local nested ownership, local heads + global heads

The script refuses to overwrite an existing run. Use a new --out-dir when
changing the experiment instead of mixing incompatible logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vecgpt.train import Cfg, train


def parse_stages(text: str):
    return tuple(tuple(map(int, part.split(":"))) for part in text.split(","))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="runs/region_ablation")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--variants", default="flat,hierarchy,region_local")
    p.add_argument("--stages", default="1:500,2:2000,3:4000,4:6000,5:7500")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-global-heads", type=int, default=2)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    variants = {
        "flat": dict(hierarchical_regions=False, region_attention=False,
                     dynamic_region_masks=False),
        "hierarchy": dict(hierarchical_regions=True, region_attention=False,
                          dynamic_region_masks=True),
        "region_local": dict(hierarchical_regions=True, region_attention=True,
                             dynamic_region_masks=True),
    }
    chosen = args.variants.split(",")
    unknown = set(chosen) - set(variants)
    if unknown:
        p.error(f"unknown variants: {sorted(unknown)}")

    for variant in chosen:
        for seed_text in args.seeds.split(","):
            seed = int(seed_text)
            out = Path(args.out_dir) / variant / f"s{seed}"
            if (out / "log.jsonl").exists():
                print(f"SKIP existing run: {out}")
                continue
            cfg = Cfg(
                seed=seed,
                out_dir=str(out),
                cache_dir=args.cache_dir,
                stage_schedule=parse_stages(args.stages),
                batch_size=args.batch_size,
                device=args.device,
                n_global_heads=args.n_global_heads,
                **variants[variant],
            )
            print(f"\n=== {variant} seed={seed} -> {out} ===", flush=True)
            train(cfg)


if __name__ == "__main__":
    main()
