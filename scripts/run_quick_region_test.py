#!/usr/bin/env python
"""Short shared-pretrain test for dynamic REGION-token heatmaps.

This is deliberately not the full multi-seed experiment. It trains the
common single-object prefix once, then branches only for stage 4:

  flat            no hierarchy, ordinary global cross-attention
  dynamic_masks   variable REGION tokens with learned soft heatmaps
  dynamic_local   the same plus region-local decoder self-attention
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vecgpt.train import Cfg, train


# Each stage must pass the mastery gate before complexity increases.  The old
# 300-step stage 1 reached shape IoU 0.023 and nevertheless advanced, making
# every later ablation uninterpretable.
PRETRAIN = ((1, 1200), (2, 2000), (3, 3000))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="runs/quick_regions")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--stage4-steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to start a long CPU run")

    base = Path(args.out_dir)
    shared = base / "shared"
    checkpoint = shared / "latest.pt"
    if not checkpoint.exists():
        train(Cfg(
            out_dir=str(shared),
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=args.batch_size,
            stage_schedule=PRETRAIN,
            lr_total_steps=20000,
            hierarchical_regions=False,
            region_attention=False,
            dynamic_region_masks=False,
            ckpt_every=400,
            eval_every=800,
            eval_n=8,
            preview_every=0,
        ))
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = sum(n for _, n in PRETRAIN) - 1
        if int(state.get("step", -1)) != expected:
            raise SystemExit(
                f"incomplete shared checkpoint at step {state.get('step')}; "
                "use a new --out-dir or resume that pretrain explicitly"
            )
        print(f"reuse shared pretrain: {checkpoint}")

    variants = {
        "flat": dict(
            hierarchical_regions=False,
            dynamic_region_masks=False,
            region_attention=False,
        ),
        "dynamic_masks": dict(
            hierarchical_regions=True,
            dynamic_region_masks=True,
            region_attention=False,
        ),
        "dynamic_local": dict(
            hierarchical_regions=True,
            dynamic_region_masks=True,
            region_attention=True,
        ),
    }
    schedule = PRETRAIN + ((4, args.stage4_steps),)
    expected_final_step = sum(n for _, n in schedule) - 1
    for name, flags in variants.items():
        out = base / name / "s0"
        branch_checkpoint = out / "latest.pt"
        resume_from = checkpoint
        if branch_checkpoint.exists():
            state = torch.load(
                branch_checkpoint, map_location="cpu", weights_only=False
            )
            if int(state.get("step", -1)) == expected_final_step:
                print(f"SKIP completed branch: {out}")
                continue
            resume_from = branch_checkpoint
            print(f"resume incomplete branch: {branch_checkpoint}")
        elif (out / "log.jsonl").exists():
            raise SystemExit(
                f"{out} has a log but no checkpoint; use a new --out-dir"
            )
        train(Cfg(
            out_dir=str(out),
            cache_dir=args.cache_dir,
            resume=str(resume_from),
            device=args.device,
            batch_size=args.batch_size,
            stage_schedule=schedule,
            lr_total_steps=20000,
            ckpt_every=400,
            eval_every=400,
            eval_n=8,
            preview_every=400,
            **flags,
        ))

    print("\nAnalyse:")
    print(f"  PYTHONPATH=. python scripts/analyze_region_ablation.py "
          f"--out-dir {base} --variants flat,dynamic_masks,dynamic_local --seeds 0")


if __name__ == "__main__":
    main()
