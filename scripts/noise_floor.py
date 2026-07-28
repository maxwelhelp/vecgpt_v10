#!/usr/bin/env python
"""Noise floor: run identical config with 3 seeds, compute MIN_READABLE.

Usage:
    python scripts/noise_floor.py --out-dir runs/noise  --seeds 0,1,2  --stages 1:1500,2:2500
"""
from __future__ import annotations

import argparse, json, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vecgpt.train import Cfg, train


def collect(log_path: str):
    evals = []
    with open(log_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "eval":
                evals.append(r)
    return evals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="runs/noise")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--stages", type=str, default="1:1500,2:2500")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--cache-dir", type=str, default="")
    p.add_argument("--analyze-only", action="store_true",
                   help="read existing OUT_DIR/sSEED/log.jsonl without training")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    stage_schedule = []
    for part in args.stages.split(","):
        s, n = part.split(":")
        stage_schedule.append((int(s), int(n)))

    all_runs = {}
    for seed in seeds:
        out = f"{args.out_dir}/s{seed}"
        if not args.analyze_only:
            print(f"\n=== SEED {seed} -> {out} ===")
            cfg = Cfg(
                seed=seed,
                out_dir=out,
                stage_schedule=tuple(stage_schedule),
                batch_size=args.batch_size,
                log_every=args.log_every,
                eval_every=args.eval_every,
                label_smooth_bins=1.0,
                cache_dir=args.cache_dir,
            )
            train(cfg)
        all_runs[seed] = collect(f"{out}/log.jsonl")

    # Aggregate at fixed steps
    all_steps = set()
    for rlist in all_runs.values():
        for r in rlist:
            all_steps.add(r["step"])

    print("\n" + "=" * 80)
    print("NOISE FLOOR ANALYSIS")
    print("=" * 80)

    for step in sorted(all_steps)[:10]:
        values = []
        for seed in seeds:
            for r in all_runs[seed]:
                if r["step"] == step:
                    values.append(r["in_dist"]["iou"])
                    break

        if len(values) < 2:
            continue
        mean = st.mean(values)
        sd = st.stdev(values) if len(values) >= 2 else 0.0
        se = sd / (len(values) ** 0.5)
        readable = 2 * sd
        print(f"step {step:5d}: IoU={mean:.3f}  sd={sd:.3f}  "
              f"2*sd={readable:.3f}  n={len(values)}")

    # Compute global MIN_READABLE
    all_ious = []
    step_2sd = []
    for step in sorted(all_steps):
        values = []
        for seed in seeds:
            for r in all_runs[seed]:
                if r["step"] == step:
                    values.append(r["in_dist"]["iou"])
                    break
        if len(values) >= 2:
            sd = st.stdev(values)
            step_2sd.append(2 * sd)
            all_ious.extend(values)

    global_2sd = 2 * st.stdev(all_ious) if len(all_ious) >= 2 else 0.0
    max_2sd = max(step_2sd) if step_2sd else 0.0

    print(f"\n--- SUMMARY ---")
    print(f"MIN_READABLE (max 2*sd across steps): {max_2sd:.4f}")
    print(f"Global 2*sd: {global_2sd:.4f}")
    print(f"Verdict: any IoU difference < {max_2sd:.4f} is indistinguishable from noise")


if __name__ == "__main__":
    main()
