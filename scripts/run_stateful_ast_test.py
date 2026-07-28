#!/usr/bin/env python
"""Short CUDA gate for the stateful typed Stroke AST.

The target is a native vector program, not a raster.  Raster loss is only an
auxiliary visual constraint and is backpropagated through the clothoid
renderer.  Evaluation uses predicted Stroke presence and segment counts.
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

from vecgpt.render import (
    foreground_render_loss,
    image_iou,
    image_iou_shape,
    save_grid,
)
from vecgpt.stateful import (
    StatefulStroke,
    pack_stateful_scenes,
    render_packed_stateful,
)
from vecgpt.stateful_model import (
    StatefulASTAutoencoder,
    output_to_packed,
    stateful_losses,
)


def sample_stroke(rng: random.Random, max_segments: int) -> StatefulStroke:
    n = rng.randint(1, max_segments)
    anchor = torch.tensor([
        rng.uniform(0.22, 0.72),
        rng.uniform(0.22, 0.72),
        rng.uniform(-math.pi, math.pi),
    ])
    base_kappa = torch.tensor(rng.uniform(-4.0, 4.0))
    length = torch.tensor([rng.uniform(0.045, 0.13) for _ in range(n)])
    delta_kappa = torch.tensor([
        rng.uniform(-2.5, 2.5) if rng.random() < 0.75 else 0.0
        for _ in range(n)
    ])
    curvature_jump = torch.zeros(n)
    for i in range(1, n):
        if rng.random() < 0.18:
            curvature_jump[i] = rng.uniform(-3.0, 3.0)

    style = torch.tensor([
        rng.uniform(0.018, 0.055),
        rng.uniform(0.08, 0.92),
        rng.uniform(0.08, 0.92),
        rng.uniform(0.08, 0.92),
        rng.uniform(0.70, 1.0),
    ])
    deltas = torch.zeros(n, 5)
    current = style.clone()
    for i in range(1, n):
        if rng.random() < 0.18:
            nxt = current + torch.tensor([
                rng.uniform(-0.008, 0.008),
                rng.uniform(-0.15, 0.15),
                rng.uniform(-0.15, 0.15),
                rng.uniform(-0.15, 0.15),
                rng.uniform(-0.12, 0.12),
            ])
            nxt[0].clamp_(0.012, 0.065)
            nxt[1:].clamp_(0.05, 1.0)
            deltas[i] = nxt - current
            current = nxt
    return StatefulStroke(
        anchor, base_kappa, length, delta_kappa,
        curvature_jump, style, deltas,
    )


def make_scenes(
    seed: int, n: int, max_strokes: int, max_segments: int,
) -> list[list[StatefulStroke]]:
    rng = random.Random(seed)
    return [
        [
            sample_stroke(rng, max_segments)
            for _ in range(rng.randint(1, max_strokes))
        ]
        for _ in range(n)
    ]


@torch.no_grad()
def evaluate(model, scenes, device, size=64):
    model.eval()
    packed = pack_stateful_scenes(
        scenes, model.max_strokes, model.max_segments, device
    )
    out = model(packed)
    predicted = output_to_packed(out, soft_structure=False)
    target_img = render_packed_stateful(
        packed, size=size, curve_samples=24, pixel_chunk=512
    )
    pred_img = render_packed_stateful(
        predicted, size=size, curve_samples=24, pixel_chunk=512
    )
    present = out.present_logits.sigmoid() >= 0.5
    present_acc = (present == packed.stroke_mask).float().mean()
    count_acc = (
        (out.count_logits.argmax(-1)[packed.stroke_mask] + 1)
        == packed.counts[packed.stroke_mask]
    ).float().mean()
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-scenes", type=int, default=1536)
    p.add_argument("--eval-scenes", type=int, default=96)
    p.add_argument("--max-strokes", type=int, default=2)
    p.add_argument("--max-segments", type=int, default=6)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--render-weight", type=float, default=0.15)
    p.add_argument("--render-every", type=int, default=4)
    p.add_argument("--render-batch", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="runs/stateful_ast_gate")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    torch.manual_seed(47)
    train = make_scenes(
        47001, args.train_scenes, args.max_strokes, args.max_segments
    )
    validation = make_scenes(
        97003, args.eval_scenes, args.max_strokes, args.max_segments
    )
    model = StatefulASTAutoencoder(
        d_model=args.d_model, n_heads=4, n_layers=args.layers,
        max_strokes=args.max_strokes, max_segments=args.max_segments,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.time()

    for step in range(1, args.steps + 1):
        if step <= args.warmup:
            scale = step / max(args.warmup, 1)
        else:
            progress = (step - args.warmup) / max(
                args.steps - args.warmup, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * max(scale, 0.02)
        ids = torch.randint(
            0, len(train), (args.batch_size,)
        ).tolist()
        scenes = [train[i] for i in ids]
        packed = pack_stateful_scenes(
            scenes, args.max_strokes, args.max_segments, device
        )
        model.train()
        out = model(packed)
        direct, terms = stateful_losses(out, packed)
        render_loss = direct.new_zeros(())
        render_terms = {}
        if args.render_weight > 0 and step % args.render_every == 0:
            rb = min(args.render_batch, args.batch_size)
            fields = {
                key: value[:rb] for key, value in vars(out).items()
            }
            soft = output_to_packed(type(out)(**fields), soft_structure=True)
            pred_img = render_packed_stateful(
                soft, size=32, curve_samples=10, pixel_chunk=256,
                distance_softmin_px=1.0,
            )
            target_fields = {
                key: value[:rb] for key, value in vars(packed).items()
            }
            target_img = render_packed_stateful(
                type(packed)(**target_fields),
                size=32, curve_samples=12, pixel_chunk=256,
            )
            render_loss, render_terms = foreground_render_loss(
                pred_img, target_img
            )
        loss = direct + args.render_weight * render_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, validation, device)
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
                "minutes": (time.time() - started) / 60,
                **{key: float(value) for key, value in terms.items()},
                **{
                    f"render_{key}": float(value)
                    for key, value in render_terms.items()
                },
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            save_grid(
                [
                    list(ev["target"][:8].cpu()),
                    list(ev["pred"][:8].cpu()),
                ],
                str(out_dir / f"preview_{step:05d}.png"),
            )
            torch.save({
                "architecture": "stateful_typed_ast_v1",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(args),
                "step": step,
                "history": history,
            }, out_dir / "latest.pt")

    ev = evaluate(model, validation, device)
    passed = (
        ev["shape_iou"] >= 0.72
        and ev["present_acc"] >= 0.98
        and ev["count_acc"] >= 0.90
    )
    result = {
        "passed": passed,
        "architecture": "stateful_typed_ast_v1",
        "history": history,
        "final": {
            key: value for key, value in ev.items()
            if key not in ("target", "pred")
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"{'PASS' if passed else 'FAIL'}: "
        f"strict={ev['strict_iou']:.3f} shape={ev['shape_iou']:.3f} "
        f"present={ev['present_acc']:.3f} count={ev['count_acc']:.3f}; "
        f"preview={out_dir / f'preview_{args.steps:05d}.png'}",
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
