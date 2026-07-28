#!/usr/bin/env python
"""Region-aligned linear plan probe: H(f) - H(f|its region plan).

Measures how much information plan tokens (rx, ry, rt) leak about each
stroke field. Reports ratio = (H - H|plan) / H:
  ratio ~ 1.0 => plan is a sufficient statistic (image not needed) [BAD]
  ratio ~ 0.3-0.6 => plan coarsely localizes, details from image [GOOD]
  ratio < 0 => plan gives nothing [expected on stages without explicit regions]

Usage:
    python scripts/leak_probe.py
    python scripts/leak_probe.py --stages 4,5
"""
from __future__ import annotations

import argparse, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from vecgpt.tokenizer import encode
from vecgpt.regions import build_regions
from vecgpt.data import sample_scene
from vecgpt.grammar import sample_tree
import vecgpt.schema as S

PLAN = {"rx", "ry", "rt"}
TARGETS = ["x", "y", "theta", "len", "turn"]
MAX_PLAN = 3


class LeakProbe(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.emb = nn.Embedding(S.VOCAB, 8)
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(MAX_PLAN * 8, n_cls))

    def forward(self, plan_tokens):
        return self.net(self.emb(plan_tokens))


def build_dataset(stage, n=4000):
    rng = random.Random(stage * 1009)
    X = {t: [] for t in TARGETS}
    Y = {t: [] for t in TARGETS}
    for _ in range(n):
        sc = sample_scene(rng, stage) if stage < 5 else sample_tree(rng)
        regs = build_regions(sc)
        e = encode(sc, regs)
        plan_by_region = {}
        for i, t in enumerate(e.tokens):
            if S.field_of_token(int(t)) in PLAN:
                rid = int(e.region_idx[i])
                plan_by_region.setdefault(rid, []).append(int(t))

        # One target per field per owned region. Never pair a stroke with
        # unrelated plan tokens from elsewhere in the scene.
        found = set()
        for i, t in enumerate(e.tokens):
            f = S.field_of_token(int(t))
            rid = int(e.region_idx[i])
            key = (rid, f)
            if f in TARGETS and key not in found:
                found.add(key)
                plan = (plan_by_region.get(rid, []) + [0] * MAX_PLAN)[:MAX_PLAN]
                X[f].append(plan)
                Y[f].append(int(t) - S.RANGE[f][0])
    return {k: torch.tensor(v) for k, v in X.items()}, \
           {k: torch.tensor(v) for k, v in Y.items()}


def probe_field(X_train, y_train, X_test, y_test, n_cls, steps=300):
    model = LeakProbe(n_cls)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)
    bs = min(128, X_train.shape[0])
    for _ in range(steps):
        idx = torch.randint(0, X_train.shape[0], (bs,))
        loss = F.cross_entropy(model(X_train[idx]), y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        test_ce = float(F.cross_entropy(model(X_test), y_test))
    return test_ce


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", type=str, default="1,2,3,4,5")
    p.add_argument("--n", type=int, default=4000)
    args = p.parse_args()
    stages = [int(s) for s in args.stages.split(",")]

    print("=" * 72)
    print("REGION-ALIGNED LINEAR PLAN PROBE: ratio = (H - H|plan) / H")
    print(f"PLAN: {sorted(PLAN)}  TARGETS: {TARGETS}")
    print("This measures linearly readable information, not definitive leakage.")
    print("High ratio is only bad when image_use is near zero or rollout fails.")
    print("=" * 72)

    for stage in stages:
        print(f"\n--- stage {stage} ---")
        X, Y = build_dataset(stage, n=args.n)
        for f in TARGETS:
            x = X[f]
            y = Y[f]
            if y.numel() == 0:
                print(f"  {f:6s}: no samples")
                continue

            n_cls = S.QUANTS[f].n
            n_total = x.shape[0]
            n_train = int(n_total * 0.8)
            perm = torch.randperm(n_total)
            x_train, x_test = x[perm[:n_train]], x[perm[n_train:]]
            y_train, y_test = y[perm[:n_train]], y[perm[n_train:]]

            # Prior entropy on test set
            cnt = torch.bincount(y_test, minlength=n_cls).float()
            p = cnt / cnt.sum().clamp_min(1)
            H = float(-(p * p.clamp_min(1e-9).log()).sum())

            h_given_plan = probe_field(x_train, y_train, x_test, y_test, n_cls, steps=300)
            ratio = (H - h_given_plan) / max(H, 1e-9)

            if ratio > 0.9:
                tag = "PLAN-SUFFICIENT"
            elif ratio > 0.5:
                tag = "STRONG"
            elif ratio > 0.0:
                tag = "OK"
            else:
                tag = "NONE"
            print(f"  {f:6s}: H={H:.2f}  H|plan={h_given_plan:.3f}  "
                  f"ratio={ratio:+.3f}  [{tag}]  (n={n_train}/{n_total-n_train})")

    print(f"\n{'='*72}")
    print("Interpret together with per-field image_use, plan_use and predicted-plan rollout.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
