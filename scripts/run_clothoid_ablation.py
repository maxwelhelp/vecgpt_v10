#!/usr/bin/env python
"""Measure how many constant-curvature arcs one clothoid replaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vecgpt.clothoid import clothoid_points, constant_arc_approximation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--arcs", default="1,2,4,8,16")
    p.add_argument("--samples-per-arc", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs/clothoid_ablation/result.json")
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    torch.manual_seed(71)
    anchor = torch.empty(args.n, 3, device=device)
    anchor[:, :2].uniform_(0.2, 0.8)
    anchor[:, 2].uniform_(-torch.pi, torch.pi)
    k0 = torch.empty(args.n, device=device).uniform_(-8, 8)
    length = torch.empty(args.n, device=device).uniform_(0.12, 0.55)
    dk = torch.empty(args.n, device=device).uniform_(-16, 16)
    rows = []
    for n_arcs in [int(x) for x in args.arcs.split(",")]:
        per = args.samples_per_arc
        target = clothoid_points(
            anchor, k0, length, dk, n_arcs * per
        )
        approx = constant_arc_approximation(
            anchor, k0, length, dk, n_arcs, per
        )
        point_error = (target - approx).norm(dim=-1)
        row = {
            "constant_arcs": n_arcs,
            "arc_parameters": 2 * n_arcs,
            "clothoid_parameters": 3,
            "mean_point_error": float(point_error.mean()),
            "p95_point_error": float(point_error.quantile(0.95)),
            "max_point_error": float(point_error.max()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    # Verify a useful nonzero gradient reaches all three geometric values.
    L = torch.tensor(0.37, device=device, requires_grad=True)
    K = torch.tensor(1.8, device=device, requires_grad=True)
    D = torch.tensor(7.0, device=device, requires_grad=True)
    a = torch.tensor([0.3, 0.4, -0.7], device=device)
    points = clothoid_points(a, K, L, D, 32)
    objective = points[-1, 0] + 0.7 * points[-1, 1]
    grads = torch.autograd.grad(objective, (L, K, D))
    gradient = {
        "d_length": float(grads[0]),
        "d_kappa0": float(grads[1]),
        "d_delta_kappa": float(grads[2]),
        "all_finite_nonzero": all(
            bool(torch.isfinite(g) & (g.abs() > 1e-8)) for g in grads
        ),
    }
    report = {"n": args.n, "results": rows, "gradient": gradient}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"gradient": gradient}), flush=True)
    print(f"report={out}", flush=True)


if __name__ == "__main__":
    main()
