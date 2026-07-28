#!/usr/bin/env python
"""Recover random clothoid strokes using only differentiable render losses."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from vecgpt.clothoid_render import render_clothoid_batch
from vecgpt.render import (
    foreground_render_loss,
    image_iou,
    image_iou_shape,
    save_grid,
)


def _logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(x) - torch.log1p(-x)


def _atanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-0.99999, 0.99999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class LearnableClothoids(nn.Module):
    def __init__(self, target: dict[str, torch.Tensor]):
        super().__init__()
        def noise_like(x, scale):
            return torch.randn_like(x) * scale

        xy = (target["anchor"][..., :2] + noise_like(
            target["anchor"][..., :2], 0.035
        )).clamp(0.05, 0.95)
        self.raw_xy = nn.Parameter(_logit(xy))
        self.theta = nn.Parameter(
            target["anchor"][..., 2] + noise_like(target["kappa0"], 0.20)
        )
        length_unit = (
            target["length"] + noise_like(target["length"], 0.025) - 0.08
        ) / 0.35
        self.raw_length = nn.Parameter(_logit(length_unit))
        kappa = target["kappa0"] + noise_like(target["kappa0"], 0.8)
        self.raw_kappa = nn.Parameter(_atanh(kappa / 8.0))
        delta = target["delta_kappa"] + noise_like(
            target["delta_kappa"], 1.2
        )
        self.raw_delta = nn.Parameter(_atanh(delta / 12.0))
        width_unit = (
            target["width"] + noise_like(target["width"], 0.006) - 0.01
        ) / 0.07
        self.raw_width = nn.Parameter(_logit(width_unit))
        rgba = (
            target["rgba"] + noise_like(target["rgba"], 0.10)
        ).clamp(0.02, 0.98)
        self.raw_rgba = nn.Parameter(_logit(rgba))

    def values(self) -> dict[str, torch.Tensor]:
        xy = self.raw_xy.sigmoid()
        anchor = torch.cat((xy, self.theta[..., None]), -1)
        return {
            "anchor": anchor,
            "kappa0": 8.0 * self.raw_kappa.tanh(),
            "length": 0.08 + 0.35 * self.raw_length.sigmoid(),
            "delta_kappa": 12.0 * self.raw_delta.tanh(),
            "width": 0.01 + 0.07 * self.raw_width.sigmoid(),
            "rgba": self.raw_rgba.sigmoid(),
        }


def make_target(batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    anchor = torch.empty(batch, 1, 3, device=device)
    anchor[..., :2].uniform_(0.32, 0.68)
    anchor[..., 2].uniform_(-math.pi, math.pi)
    return {
        "anchor": anchor,
        "kappa0": torch.empty(batch, 1, device=device).uniform_(-5.0, 5.0),
        "length": torch.empty(batch, 1, device=device).uniform_(0.16, 0.30),
        "delta_kappa": torch.empty(
            batch, 1, device=device
        ).uniform_(-6.0, 6.0),
        "width": torch.empty(batch, 1, device=device).uniform_(0.025, 0.055),
        "rgba": torch.empty(batch, 1, 4, device=device).uniform_(0.08, 0.92),
    }


def render(values, args, background, softmin_px):
    return render_clothoid_batch(
        values["anchor"], values["kappa0"], values["length"],
        values["delta_kappa"], values["width"], values["rgba"],
        size=args.size, curve_samples=args.curve_samples,
        softness_px=args.softness_px,
        distance_softmin_px=softmin_px,
        background=background, pixel_chunk=args.pixel_chunk,
    )


@torch.no_grad()
def metrics(pred, target, pred_white, target_white):
    angle = torch.atan2(
        torch.sin(pred["anchor"][..., 2] - target["anchor"][..., 2]),
        torch.cos(pred["anchor"][..., 2] - target["anchor"][..., 2]),
    ).abs()
    return {
        "strict_iou": float(image_iou(pred_white, target_white).mean()),
        "shape_iou": float(image_iou_shape(pred_white, target_white).mean()),
        "xy_mae": float(
            (pred["anchor"][..., :2] - target["anchor"][..., :2])
            .norm(dim=-1).mean()
        ),
        "theta_mae": float(angle.mean()),
        "length_mae": float((pred["length"] - target["length"]).abs().mean()),
        "kappa0_mae": float((pred["kappa0"] - target["kappa0"]).abs().mean()),
        "delta_kappa_mae": float(
            (pred["delta_kappa"] - target["delta_kappa"]).abs().mean()
        ),
        "width_mae": float((pred["width"] - target["width"]).abs().mean()),
        "rgb_mae": float(
            (pred["rgba"][..., :3] - target["rgba"][..., :3]).abs().mean()
        ),
        "alpha_mae": float(
            (pred["rgba"][..., 3] - target["rgba"][..., 3]).abs().mean()
        ),
    }


def run_seed(seed: int, args, device: torch.device, out_dir: Path):
    torch.manual_seed(seed)
    target = make_target(args.batch, device)
    target_white = render(target, args, 1.0, 0.0).detach()
    target_black = render(target, args, 0.0, 0.0).detach()
    model = LearnableClothoids(target).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    started = time.time()
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, args.steps + 1):
        values = model.values()
        pred_white = render(
            values, args, 1.0, args.distance_softmin_px
        )
        pred_black = render(
            values, args, 0.0, args.distance_softmin_px
        )
        white_loss, _ = foreground_render_loss(pred_white, target_white)
        # Inverting the black-background render gives foreground_render_loss
        # its expected white background while adding independent information
        # that disambiguates RGB from alpha.
        black_loss, _ = foreground_render_loss(
            1.0 - pred_black, 1.0 - target_black
        )
        loss = 0.5 * (white_loss + black_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                values = model.values()
                pred_eval = render(values, args, 1.0, 0.0)
                row = {
                    "step": step,
                    "loss": float(loss),
                    "grad_norm": float(grad_norm),
                    "seconds": time.time() - started,
                    **metrics(values, target, pred_eval, target_white),
                }
            history.append(row)
            print(json.dumps({"seed": seed, **row}), flush=True)
            save_grid(
                [
                    list(target_white[:8].cpu()),
                    list(pred_eval[:8].cpu()),
                ],
                str(out_dir / f"seed{seed}_step{step:04d}.png"),
            )

    final = history[-1]
    final["peak_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda" else 0.0
    )
    # This is a local recovery gate, not arbitrary raster-to-vector inversion.
    final["passed"] = bool(
        final["shape_iou"] >= 0.85
        and final["xy_mae"] <= 0.025
        and final["theta_mae"] <= 0.15
        and final["length_mae"] <= 0.03
        and final["kappa0_mae"] <= 0.60
        and final["delta_kappa_mae"] <= 1.20
        and final["width_mae"] <= 0.008
        and final["rgb_mae"] <= 0.12
        and final["alpha_mae"] <= 0.12
    )
    return {"seed": seed, "history": history, "final": final}


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--seeds", default="0")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--curve-samples", type=int, default=32)
    p.add_argument("--softness-px", type=float, default=1.0)
    p.add_argument("--distance-softmin-px", type=float, default=0.25)
    p.add_argument("--pixel-chunk", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.025)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="runs/clothoid_renderer_gate")
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        run_seed(int(seed), args, device, out_dir)
        for seed in args.seeds.split(",") if seed.strip()
    ]
    report = {
        "config": vars(args),
        "runs": reports,
        "passed": all(run["final"]["passed"] for run in reports),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(
        f"{'PASS' if report['passed'] else 'FAIL'}: "
        f"report={out_dir / 'report.json'}",
        flush=True,
    )
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
