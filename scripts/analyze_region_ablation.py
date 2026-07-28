#!/usr/bin/env python
"""Summarise matched region ablations across seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path


def records(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_ci(values):
    if not values:
        return float("nan"), float("nan")
    mean = st.mean(values)
    if len(values) < 2:
        return mean, float("nan")
    return mean, 1.96 * st.stdev(values) / math.sqrt(len(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="runs/region_ablation")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--variants", default="flat,hierarchy,region_local")
    args = p.parse_args()

    base = Path(args.out_dir)
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"{'variant':14s} {'stage':>5s} {'IoU':>15s} {'shape IoU':>15s} "
          f"{'OOD':>15s} {'image gap':>15s} {'plan gap':>15s}")

    for variant in args.variants.split(","):
        per_seed = []
        for seed in seeds:
            path = base / variant / f"s{seed}" / "log.jsonl"
            if not path.exists():
                continue
            rs = records(path)
            evals = [r for r in rs if r.get("kind") == "eval"]
            trains = [r for r in rs if r.get("kind") == "train"]
            if not evals or not trains:
                continue
            e, tr = evals[-1], trains[-1]
            per_seed.append({
                "stage": e["stage"],
                "iou": e["in_dist"]["iou"],
                "shape": e["in_dist"]["iou_shape"],
                "ood": e["ood"]["iou"],
                "image": tr.get("image_gap", tr.get("leak_gap", float("nan"))),
                "plan": tr.get("plan_gap", float("nan")),
            })

        if not per_seed:
            print(f"{variant:14s} no completed runs")
            continue

        def fmt(key):
            mean, ci = mean_ci([r[key] for r in per_seed])
            return f"{mean:.3f} ± {ci:.3f}" if not math.isnan(ci) else f"{mean:.3f}"

        print(f"{variant:14s} {max(r['stage'] for r in per_seed):5d} "
              f"{fmt('iou'):>15s} {fmt('shape'):>15s} {fmt('ood'):>15s} "
              f"{fmt('image'):>15s} {fmt('plan'):>15s}")


if __name__ == "__main__":
    main()
