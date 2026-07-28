#!/usr/bin/env python
"""Isolated probe: can a ConvEncoder + linear head learn to predict a field?

Separates "representation is bad" from "budget is too small".
If the probe achieves mae < 15 bins in 1500 steps, the encoder
representation is adequate — the problem is in the full pipeline.

Usage:
    python scripts/probe_field.py theta 3    # probe theta on stage 3
    python scripts/probe_field.py len 3      # control: len on stage 3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from vecgpt.data import SceneStream
from vecgpt.render import render_batch
import vecgpt.schema as S


def probe(field: str, stage: int, steps: int = 1500, batch: int = 16):
    q = S.QUANTS[field]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class ProbeNet(nn.Module):
        def __init__(self, n_cls):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.GELU(),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.GELU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, n_cls),
            )

        def forward(self, x):
            return self.net(x)

    model = ProbeNet(q.n).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    st = SceneStream(stage=stage, seed=0)
    prior_ce = torch.log(torch.tensor(float(q.n)))

    print(f"\n=== PROBE: field={field}  stage={stage}  bins={q.n}  steps={steps} ===")
    print(f"prior CE: {prior_ce:.2f}  (random guess baseline)")
    print(f"{'step':>6s}  {'CE':>8s}  {'mae_bins':>10s}  {'acc%':>8s}")

    for i in range(steps):
        scenes = st.batch(batch)
        imgs = render_batch(scenes, size=64, device=device)

        tgt_vals = []
        for s in scenes:
            if field == "theta":
                val = float(s[0].anchor[2])
            elif field == "len":
                val = float(s[0].segs[0, 0])
            elif field == "turn":
                L = float(s[0].segs[0, 0])
                val = float(s[0].segs[0, 1]) * L
            elif field in ("x", "y"):
                idx = {"x": 0, "y": 1}[field]
                val = float(s[0].anchor[idx])
            else:
                val = 0.0
            tgt_vals.append(q.q(val))

        tgt = torch.tensor(tgt_vals, device=device)
        if imgs.shape[1] != 3:
            imgs = imgs.permute(0, 3, 1, 2).contiguous()

        lg = model(imgs)
        loss = F.cross_entropy(lg, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 300 == 0:
            with torch.no_grad():
                pred = lg.argmax(-1)
                mae = (pred - tgt).abs().float().mean()
                acc = (pred == tgt).float().mean()
            print(f"{i:6d}  {loss.item():8.3f}  {mae.item():10.1f}  {acc.item():8.3f}")

    # Final
    with torch.no_grad():
        scenes = st.batch(256)
        imgs = render_batch(scenes, size=64, device=device)
        if imgs.shape[1] != 3:
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
        tgt_vals = []
        for s in scenes:
            if field == "theta":
                val = float(s[0].anchor[2])
            elif field == "len":
                val = float(s[0].segs[0, 0])
            elif field == "turn":
                L = float(s[0].segs[0, 0])
                val = float(s[0].segs[0, 1]) * L
            else:
                val = 0.0
            tgt_vals.append(q.q(val))
        tgt = torch.tensor(tgt_vals, device=device)
        lg = model(imgs)
        pred = lg.argmax(-1)
        final_mae = (pred - tgt).abs().float().mean()
        final_acc = (pred == tgt).float().mean()
        final_ce = float(F.cross_entropy(lg, tgt))

    print(f"\nFINAL: CE={final_ce:.3f}  mae={final_mae:.1f} bins  acc={final_acc:.3f}")
    if final_mae < 15:
        print("VERDICT: representation is ADEQUATE — problem is budget/pipeline.")
    else:
        print("VERDICT: representation is INADEQUATE — need to fix encoding.")
    return final_mae, final_ce


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/probe_field.py <field> <stage>")
        print("  field: theta, len, turn, x, y")
        print("  stage: 1-5")
        sys.exit(1)

    field = sys.argv[1]
    stage = int(sys.argv[2])
    if field not in S.QUANTS and field != "theta":
        print(f"Unknown field: {field}")
        sys.exit(1)
    if field == "theta":
        field = "theta"  # theta IS in QUANTS

    probe(field, stage)


if __name__ == "__main__":
    main()
