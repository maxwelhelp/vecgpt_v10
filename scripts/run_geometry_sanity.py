#!/usr/bin/env python
"""Mandatory short gate: overfit a fixed set of one-stroke scenes.

This is not a useful trained model.  It answers a narrower question before
any curriculum run is allowed: can the actual encoder/decoder/loss reproduce
simple geometry autoregressively, or is there still an implementation or
optimization defect?
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vecgpt.data import collate, sample_scene
from vecgpt.model import VecGPT
from vecgpt.render import image_iou, image_iou_shape, render_batch, save_grid
from vecgpt.tokenizer import build_smoothing_matrix, decode, encode
from vecgpt.train import ARCHITECTURE_VERSION, loss_fn


@torch.no_grad()
def evaluate(model, scenes, imgs):
    model.eval()
    mem, semantic = model.encode_condition(imgs)
    budget = max(max(len(encode(s, hierarchical=False).tokens) for s in scenes) + 8, 32)
    recon = [decode(s) for s in model.generate(
        mem, max_tokens=budget, semantic_mem=semantic
    )]
    pred = render_batch(recon, size=imgs.shape[1], device=imgs.device)
    return {
        "strict_iou": float(image_iou(pred, imgs).mean()),
        "shape_iou": float(image_iou_shape(pred, imgs).mean()),
        "pred": pred,
    }


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--n-scenes", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="runs/geometry_sanity_v13")
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this environment")
    device = torch.device(args.device)
    torch.manual_seed(7)

    rng = random.Random(7001)
    scenes = [sample_scene(rng, 1) for _ in range(args.n_scenes)]
    all_imgs = render_batch(scenes, size=64, device=device)
    all_batch = collate(scenes, device, hierarchical=False)

    model = VecGPT(
        image_size=64, d=256, n_heads=8, n_layers=6,
        n_seg_heads=3, n_stroke_heads=3, enc_base=48,
        n_enc_layers=2, dynamic_region_masks=False,
        region_attention=False,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    smooth = build_smoothing_matrix(1.0, device)
    history = []

    for step in range(args.steps):
        start = (step * args.batch_size) % args.n_scenes
        ids = [(start + i) % args.n_scenes for i in range(args.batch_size)]
        imgs = all_imgs[ids]
        batch = {k: v[ids] for k, v in all_batch.items()}
        model.train()
        loss, *rest = loss_fn(
            model, imgs, batch, smooth,
            balanced_fields=True,
            anchor_weight=2.0,
            render_loss_weight=0.20,
            render_loss_size=32,
            region_mask_loss_weight=0.0,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.log_every == 0 or step == 0 or step + 1 == args.steps:
            ev = evaluate(model, scenes, all_imgs)
            aux = rest[-1]
            row = {
                "step": step + 1,
                "loss": float(loss),
                "token_ce": float(aux["token_ce"]),
                "render_loss": float(aux["render_loss"]),
                "strict_iou": ev["strict_iou"],
                "shape_iou": ev["shape_iou"],
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    ev = evaluate(model, scenes, all_imgs)
    passed = ev["strict_iou"] >= 0.75 and ev["shape_iou"] >= 0.90
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(
        {"passed": passed, "history": history}, indent=2
    ))
    torch.save({
        "architecture_version": ARCHITECTURE_VERSION,
        "model": model.state_dict(),
        "step": args.steps,
        "sanity_only": True,
    }, out / "final.pt")
    save_grid(
        [list(all_imgs[:8].cpu()), list(ev["pred"][:8].cpu())],
        str(out / "preview.png"),
    )
    print(
        f"{'PASS' if passed else 'FAIL'}: strict={ev['strict_iou']:.3f} "
        f"shape={ev['shape_iou']:.3f}; preview={out / 'preview.png'}",
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
