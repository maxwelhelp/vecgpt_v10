#!/usr/bin/env python
"""Fast isolated probe: pre-rendered dataset, simple CNN, quick answer.

Usage:
    python scripts/probe_fast.py theta 1
    python scripts/probe_fast.py len 1
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from vecgpt.data import SceneStream
from vecgpt.render import render_batch
import vecgpt.schema as S


class ProbeNet(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def build_dataset(field, stage, n=2000):
    """Pre-render n scenes, return (images, targets)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = S.QUANTS[field]
    st = SceneStream(stage=stage, seed=0)
    imgs_list, tgt_list = [], []

    bs = 64
    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch_n = end - start
        scenes = st.batch(batch_n)
        imgs = render_batch(scenes, size=64, device="cpu")
        if imgs.shape[1] != 3:
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
        imgs_list.append(imgs)

        vals = []
        for s in scenes:
            if field == "theta":
                vals.append(q.q(float(s[0].anchor[2])))
            elif field == "len":
                vals.append(q.q(float(s[0].segs[0, 0])))
            elif field == "turn":
                L = float(s[0].segs[0, 0])
                vals.append(q.q(float(s[0].segs[0, 1]) * L))
            elif field == "x":
                vals.append(q.q(float(s[0].anchor[0])))
            elif field == "y":
                vals.append(q.q(float(s[0].anchor[1])))
            else:
                vals.append(0)
        tgt_list.append(torch.tensor(vals))

    return torch.cat(imgs_list), torch.cat(tgt_list)


def probe(field: str, stage: int, steps: int = 1200):
    q = S.QUANTS[field]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Building dataset for {field} stage {stage}...", flush=True)
    imgs, tgts = build_dataset(field, stage, n=2000)
    imgs, tgts = imgs.to(device), tgts.to(device)
    print(f"Dataset: {imgs.shape[0]} samples", flush=True)

    model = ProbeNet(q.n).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    prior_ce = torch.log(torch.tensor(float(q.n), device=device))
    print(f"Prior CE: {prior_ce:.2f}  (random guess)")
    print(f"{'step':>6s}  {'CE':>8s}  {'mae':>8s}  {'acc%':>8s}", flush=True)

    n = imgs.shape[0]
    bs = 128
    for i in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        lg = model(imgs[idx])
        loss = F.cross_entropy(lg, tgts[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 300 == 0:
            with torch.no_grad():
                lg_all = model(imgs)
                pred = lg_all.argmax(-1)
                mae = (pred - tgts).abs().float().mean()
                acc = (pred == tgts).float().mean()
                ce = float(F.cross_entropy(lg_all, tgts))
            print(f"{i:6d}  {ce:8.3f}  {mae.item():8.1f}  {acc.item():8.3f}", flush=True)

    with torch.no_grad():
        lg_all = model(imgs)
        pred = lg_all.argmax(-1)
        final_mae = (pred - tgts).abs().float().mean().item()
        final_acc = (pred == tgts).float().mean().item()
        final_ce = float(F.cross_entropy(lg_all, tgts))

    print(f"\nFINAL: CE={final_ce:.3f} mae={final_mae:.1f} bins acc={final_acc:.3f}", flush=True)
    if final_mae < 15:
        print("VERDICT: representation ADEQUATE", flush=True)
    else:
        print("VERDICT: representation INADEQUATE", flush=True)
    return final_mae, final_ce


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/probe_fast.py <field> <stage>")
        sys.exit(1)
    probe(sys.argv[1], int(sys.argv[2]))


if __name__ == "__main__":
    main()
