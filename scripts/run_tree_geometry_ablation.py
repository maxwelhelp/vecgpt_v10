#!/usr/bin/env python
"""Toy ablation: Euclidean vs Poincare embeddings of a REGION tree.

This deliberately tests only the claim hyperbolic geometry is meant to
solve: low-dimensional preservation of an exponentially growing hierarchy.
It does not put x/y, angles, colour, or Stroke geometry in hyperbolic space.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def complete_tree(branching: int, depth: int, device):
    parent = [-1]
    levels = [0]
    frontier = [0]
    for d in range(1, depth + 1):
        nxt = []
        for p in frontier:
            for _ in range(branching):
                parent.append(p)
                levels.append(d)
                nxt.append(len(parent) - 1)
        frontier = nxt
    parent = torch.tensor(parent, device=device)
    levels = torch.tensor(levels, device=device)
    n = len(parent)
    ancestors = torch.full((n, depth + 1), -1, device=device)
    ancestors[:, 0] = torch.arange(n, device=device)
    for j in range(1, depth + 1):
        prev = ancestors[:, j - 1]
        valid = prev >= 0
        ancestors[valid, j] = parent[prev[valid]]

    # Exact tree shortest-path distance through lowest common ancestor.
    dist = torch.zeros(n, n, device=device)
    paths = []
    for i in range(n):
        q, path = i, []
        while q >= 0:
            path.append(q)
            q = int(parent[q])
        paths.append(path)
    for i in range(n):
        pos = {node: k for k, node in enumerate(paths[i])}
        for j in range(n):
            for k, node in enumerate(paths[j]):
                if node in pos:
                    dist[i, j] = pos[node] + k
                    break
    return parent, levels, dist


def poincare(raw, eps=1e-4):
    norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    radius = (1.0 - eps) * torch.tanh(norm)
    return raw * (radius / norm)


def poincare_distance(x, i, j, eps=1e-5):
    a, b = x[i], x[j]
    a2 = a.square().sum(-1).clamp_max(1 - eps)
    b2 = b.square().sum(-1).clamp_max(1 - eps)
    delta = (a - b).square().sum(-1)
    z = 1.0 + 2.0 * delta / ((1 - a2) * (1 - b2)).clamp_min(eps)
    return torch.acosh(z.clamp_min(1.0 + eps))


def euclidean_distance(x, i, j):
    return (x[i] - x[j]).square().sum(-1).clamp_min(1e-8).sqrt()


def train_geometry(kind, dim, parent, levels, graph_dist, steps, device):
    torch.manual_seed(17 + dim)
    n = len(parent)
    raw = torch.nn.Parameter(torch.randn(n, dim, device=device) * 0.02)
    log_scale = torch.nn.Parameter(torch.zeros((), device=device))
    opt = torch.optim.Adam((raw, log_scale), lr=3e-2)
    edge_child = torch.arange(1, n, device=device)
    edge_parent = parent[1:]
    distance = poincare_distance if kind == "poincare" else euclidean_distance

    for step in range(steps):
        batch = 2048
        half = batch // 2
        e = torch.randint(0, n - 1, (half,), device=device)
        i = torch.cat((edge_child[e], torch.randint(n, (half,), device=device)))
        j = torch.cat((edge_parent[e], torch.randint(n, (half,), device=device)))
        keep = i != j
        i, j = i[keep], j[keep]
        x = poincare(raw) if kind == "poincare" else raw
        pred = distance(x, i, j) / log_scale.exp().clamp_min(1e-3)
        truth = graph_dist[i, j]
        # Relative error prevents the many far pairs from drowning tree edges.
        loss = F.smooth_l1_loss(
            pred / truth.clamp_min(1), torch.ones_like(truth)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((raw, log_scale), 5.0)
        opt.step()

    with torch.no_grad():
        x = poincare(raw) if kind == "poincare" else raw
        ii, jj = torch.triu_indices(n, n, offset=1, device=device)
        embedded = distance(x, ii, jj) / log_scale.exp()
        truth = graph_dist[ii, jj]
        relative_distortion = (
            (embedded - truth).abs() / truth.clamp_min(1)
        ).mean()

        # For each non-root node, retrieve its parent among nodes one level up.
        correct = total = 0
        for d in range(1, int(levels.max()) + 1):
            children = torch.where(levels == d)[0]
            candidates = torch.where(levels == d - 1)[0]
            ci = children[:, None].expand(-1, len(candidates)).reshape(-1)
            cj = candidates[None, :].expand(len(children), -1).reshape(-1)
            dd = distance(x, ci, cj).reshape(len(children), len(candidates))
            predicted = candidates[dd.argmin(-1)]
            correct += int((predicted == parent[children]).sum())
            total += len(children)
        parent_recall = correct / total
        # Radius should correlate with depth if hierarchy is represented.
        radius = x.norm(dim=-1)
        depth_centered = levels.float() - levels.float().mean()
        radius_centered = radius - radius.mean()
        depth_radius_corr = (
            depth_centered * radius_centered
        ).mean() / (
            depth_centered.square().mean().sqrt()
            * radius_centered.square().mean().sqrt()
        ).clamp_min(1e-8)
    return {
        "kind": kind,
        "dim": dim,
        "relative_distortion": float(relative_distortion),
        "parent_recall": parent_recall,
        "depth_radius_corr": float(depth_radius_corr),
        "scale": float(log_scale.exp()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--branching", type=int, default=3)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--dims", default="2,4,8")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs/tree_geometry_ablation/result.json")
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    parent, levels, graph_dist = complete_tree(
        args.branching, args.depth, device
    )
    results = []
    for dim in [int(x) for x in args.dims.split(",")]:
        for kind in ("euclidean", "poincare"):
            row = train_geometry(
                kind, dim, parent, levels, graph_dist, args.steps, device
            )
            results.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "nodes": len(parent),
        "branching": args.branching,
        "depth": args.depth,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"report={out}", flush=True)


if __name__ == "__main__":
    main()
