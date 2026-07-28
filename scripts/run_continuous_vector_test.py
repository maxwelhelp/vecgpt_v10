#!/usr/bin/env python
"""Fast gate for the hierarchical continuous Stroke autoencoder.

Training is directly on vector programs. Evaluation decodes solely from the
compact latent and predicted topology; target structure is never supplied to
the prediction renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vecgpt.continuous import (
    ASTVectorAutoencoder,
    ContinuousOutput,
    ContinuousVectorAutoencoder,
    continuous_losses,
    output_to_scenes,
    pack_scenes,
)
from vecgpt.data import sample_scene
from vecgpt.render import (
    foreground_render_loss,
    image_iou,
    image_iou_shape,
    render_batch,
    save_grid,
)


def make_scenes(rng: random.Random, stages: list[int], n: int):
    return [sample_scene(rng, stages[rng.randrange(len(stages))]) for _ in range(n)]


@torch.no_grad()
def evaluate(model, scenes, device, size=64):
    model.eval()
    packed = pack_scenes(
        scenes, model.max_strokes, model.max_segments, device
    )
    out = model(packed)
    recon = output_to_scenes(out, soft_structure=False)
    target_img = render_batch(scenes, size=size, device=device)
    pred_img = render_batch(recon, size=size, device=device)
    true_present = packed.stroke_mask
    pred_present = out.present_logits.sigmoid() >= 0.5
    present_acc = (pred_present == true_present).float().mean()
    if true_present.any():
        count_acc = (
            out.count_logits.argmax(-1)[true_present] + 1
            == packed.counts[true_present]
        ).float().mean()
    else:
        count_acc = present_acc.new_ones(())
    return {
        "strict_iou": float(image_iou(pred_img, target_img).mean()),
        "shape_iou": float(image_iou_shape(pred_img, target_img).mean()),
        "present_acc": float(present_acc),
        "count_acc": float(count_acc),
        "target": target_img,
        "pred": pred_img,
    }


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--steps", type=int, default=800)
    p.add_argument(
        "--architecture", choices=("ast", "compact"), default="ast"
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-scenes", type=int, default=1024)
    p.add_argument("--eval-scenes", type=int, default=128)
    p.add_argument("--stages", default="0,1,3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--max-strokes", type=int, default=4)
    p.add_argument("--max-segments", type=int, default=24)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--render-weight", type=float, default=0.20)
    p.add_argument("--render-every", type=int, default=4)
    p.add_argument("--render-batch", type=int, default=12)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--resume", default="")
    p.add_argument("--out-dir", default="runs/continuous_vector_gate")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    stages = [int(x) for x in args.stages.split(",") if x.strip()]
    torch.manual_seed(31)
    random.seed(31)

    train_scenes = make_scenes(random.Random(31001), stages, args.train_scenes)
    eval_scenes = make_scenes(random.Random(91007), stages, args.eval_scenes)
    max_k = max(len(x) for x in train_scenes + eval_scenes)
    max_s = max(
        st.segs.shape[0] for scene in train_scenes + eval_scenes for st in scene
    )
    if max_k > args.max_strokes or max_s > args.max_segments:
        raise SystemExit(
            f"context too small: data needs strokes={max_k}, segments={max_s}; "
            f"configured {args.max_strokes}/{args.max_segments}"
        )

    model_cls = (
        ASTVectorAutoencoder
        if args.architecture == "ast" else ContinuousVectorAutoencoder
    )
    model = model_cls(
        d_model=args.d_model, n_heads=4, n_layers=args.layers,
        max_strokes=args.max_strokes, max_segments=args.max_segments,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer"])
        # The CLI controls the resumed run; optimizer state must not silently
        # restore the old peak learning rate.
        for group in opt.param_groups:
            group["lr"] = args.lr
        start_step = int(checkpoint.get("step", 0))
        history = list(checkpoint.get("history", []))
        print(f"resumed {args.resume} at step {start_step}", flush=True)
    started = time.time()

    for step in range(start_step + 1, args.steps + 1):
        if step <= args.warmup:
            lr_scale = step / max(args.warmup, 1)
        else:
            progress = (step - args.warmup) / max(
                args.steps - args.warmup, 1
            )
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in opt.param_groups:
            group["lr"] = args.lr * max(lr_scale, 0.02)
        ids = torch.randint(0, len(train_scenes), (args.batch_size,)).tolist()
        scenes = [train_scenes[i] for i in ids]
        packed = pack_scenes(
            scenes, args.max_strokes, args.max_segments, device
        )
        model.train()
        pred = model(packed)
        direct, terms = continuous_losses(pred, packed)
        render_loss = direct.new_zeros(())
        render_terms = {}
        if args.render_weight > 0 and step % args.render_every == 0:
            rb = min(args.render_batch, args.batch_size)
            soft_pred = ContinuousOutput(
                pred.present_logits[:rb], pred.count_logits[:rb],
                pred.anchor[:rb], pred.base_style[:rb], pred.segment[:rb],
                pred.style_change_logits[:rb], pred.style_delta[:rb],
                pred.latent[:rb],
            )
            pred_img = render_batch(
                output_to_scenes(soft_pred, soft_structure=True),
                size=32, device=device, per_seg=8,
            )
            target_img = render_batch(
                scenes[:rb], size=32, device=device, per_seg=12
            )
            render_loss, render_terms = foreground_render_loss(
                pred_img, target_img
            )
        loss = direct + args.render_weight * render_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, eval_scenes, device)
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "direct": float(direct.detach()),
                "render": float(render_loss.detach()),
                "strict_iou": ev["strict_iou"],
                "shape_iou": ev["shape_iou"],
                "present_acc": ev["present_acc"],
                "count_acc": ev["count_acc"],
                "grad_norm": float(grad),
                "lr": opt.param_groups[0]["lr"],
                "minutes": (time.time() - started) / 60,
                **{k: float(v) for k, v in terms.items()},
                **{f"render_{k}": float(v) for k, v in render_terms.items()},
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            save_grid(
                [list(ev["target"][:8].cpu()), list(ev["pred"][:8].cpu())],
                str(out_dir / f"preview_{step:05d}.png"),
            )
            torch.save({
                "architecture": f"continuous_{args.architecture}_v1",
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": vars(args),
                "step": step,
                "history": history,
            }, out_dir / "latest.pt")

    ev = evaluate(model, eval_scenes, device)
    passed = (
        ev["shape_iou"] >= 0.70
        and ev["present_acc"] >= 0.98
        and ev["count_acc"] >= 0.90
    )
    result = {
        "passed": passed,
        "architecture": f"continuous_{args.architecture}_v1",
        "stages": stages,
        "train_scenes": len(train_scenes),
        "eval_scenes": len(eval_scenes),
        "history": history,
        "final": {k: v for k, v in ev.items() if k not in ("target", "pred")},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    torch.save({
        "architecture": result["architecture"],
        "model": model.state_dict(),
        "config": vars(args),
        "result": result["final"],
    }, out_dir / "final.pt")
    print(
        f"{'PASS' if passed else 'FAIL'}: strict={ev['strict_iou']:.3f} "
        f"shape={ev['shape_iou']:.3f} present={ev['present_acc']:.3f} "
        f"count={ev['count_acc']:.3f}; "
        f"preview={out_dir / f'preview_{args.steps:05d}.png'}",
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
