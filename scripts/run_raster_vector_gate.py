#!/usr/bin/env python
"""Synthetic raster -> stateful vector AST inverse-graphics gate."""

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

from vecgpt.clothoid import clothoid_end_state, clothoid_points
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
    StatefulOutput,
    output_to_packed,
    stateful_losses,
)
from vecgpt.visual import RasterToStatefulAST


def _candidate_stroke(rng: random.Random, max_segments: int):
    total_length = (
        rng.uniform(0.20, 0.38)
        if max_segments == 1 else
        rng.uniform(0.10, 0.12 * max_segments)
    )
    count = min(max_segments, max(1, math.ceil(total_length / 0.12)))
    length = torch.full((count,), total_length / count)
    total_delta = rng.uniform(-3.0, 3.0)
    delta = torch.full((count,), total_delta / count)
    return StatefulStroke(
        torch.tensor([
            rng.uniform(0.22, 0.78),
            rng.uniform(0.22, 0.78),
            rng.uniform(-math.pi, math.pi),
        ]),
        torch.tensor(rng.uniform(-4.0, 4.0)),
        length,
        delta,
        torch.zeros(count),
        torch.tensor([
            rng.uniform(0.035, 0.075),
            rng.uniform(0.05, 0.90),
            rng.uniform(0.05, 0.90),
            rng.uniform(0.05, 0.90),
            1.0,
        ]),
        torch.zeros(count, 5),
    )


def sample_stroke(rng, max_segments):
    for _ in range(30):
        stroke = _candidate_stroke(rng, max_segments)
        theta = stroke.anchor[2]
        anchor = torch.stack((
            stroke.anchor[0], stroke.anchor[1],
            theta.sin(), theta.cos(),
        )).reshape(1, 1, 4)
        points = clothoid_points(
            torch.cat((
                anchor[..., :2],
                torch.atan2(anchor[..., 2:3], anchor[..., 3:4]),
            ), -1),
            stroke.base_kappa.reshape(1, 1),
            stroke.length.sum().reshape(1, 1),
            stroke.delta_kappa.sum().reshape(1, 1),
            24,
        )
        if bool((points > 0.06).all() and (points < 0.94).all()):
            return canonical_direction(stroke)
    return canonical_direction(_candidate_stroke(rng, max_segments))


def canonical_direction(stroke: StatefulStroke) -> StatefulStroke:
    """Choose traversal from the raster-first endpoint.

    Reversing a clothoid changes the initial curvature to ``-kappa_end``
    while preserving curvature-rate sign, hence each reversed segment keeps
    its ``delta_kappa`` sign.
    """
    end, end_kappa = clothoid_end_state(
        stroke.anchor.reshape(1, 3),
        stroke.base_kappa.reshape(1),
        stroke.length.sum().reshape(1),
        stroke.delta_kappa.sum().reshape(1),
    )
    start_key = float(stroke.anchor[1] * 1e4 + stroke.anchor[0])
    end_key = float(end[0, 1] * 1e4 + end[0, 0])
    if start_key <= end_key:
        return stroke
    anchor = torch.stack((
        end[0, 0], end[0, 1],
        (end[0, 2] + math.pi + math.pi) % (2 * math.pi) - math.pi,
    ))
    return StatefulStroke(
        anchor,
        -end_kappa[0],
        stroke.length.flip(0),
        stroke.delta_kappa.flip(0),
        torch.zeros_like(stroke.curvature_jump),
        stroke.base_style.clone(),
        stroke.style_delta.flip(0),
    )


def make_scenes(seed, n, max_strokes, max_segments):
    rng = random.Random(seed)
    scenes = []
    for _ in range(n):
        strokes = [
            sample_stroke(rng, max_segments)
            for _ in range(rng.randint(1, max_strokes))
        ]
        strokes.sort(key=lambda s: (
            round(float(s.anchor[1]) * 64),
            round(float(s.anchor[0]) * 64),
        ))
        scenes.append(strokes)
    return scenes


@torch.no_grad()
def build_raster_cache(scenes, max_strokes, max_segments, device, size):
    images = []
    for start in range(0, len(scenes), 64):
        packed = pack_stateful_scenes(
            scenes[start:start + 64], max_strokes, max_segments, device
        )
        image = render_packed_stateful(
            packed, size=size, curve_samples=20, pixel_chunk=256
        )
        images.append(image.cpu())
    return torch.cat(images)


@torch.no_grad()
def evaluate(model, scenes, images, device):
    model.eval()
    target = pack_stateful_scenes(
        scenes, model.max_strokes, model.max_segments, device
    )
    source = images.to(device)
    out = model(source)
    predicted = output_to_packed(out, soft_structure=False)
    pred_img = render_packed_stateful(
        predicted, size=source.shape[1], curve_samples=24,
        pixel_chunk=512,
    )
    present = out.present_logits >= 0
    present_acc = (present == target.stroke_mask).float().mean()
    count_acc = (
        out.count_logits.argmax(-1)[target.stroke_mask] + 1
        == target.counts[target.stroke_mask]
    ).float().mean()
    return {
        "strict_iou": float(image_iou(pred_img, source).mean()),
        "shape_iou": float(image_iou_shape(pred_img, source).mean()),
        "present_acc": float(present_acc),
        "count_acc": float(count_acc),
        "source": source,
        "pred": pred_img,
    }


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--train-scenes", type=int, default=1536)
    p.add_argument("--eval-scenes", type=int, default=96)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--max-strokes", type=int, default=2)
    p.add_argument("--max-segments", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--encoder-layers", type=int, default=3)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--render-weight", type=float, default=0.15)
    p.add_argument("--render-every", type=int, default=5)
    p.add_argument("--render-batch", type=int, default=8)
    p.add_argument(
        "--input-noise", type=float, default=0.0,
        help="raster augmentation; keep zero until clean-image mastery",
    )
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--stage0-end", type=int, default=300)
    p.add_argument("--stage1-end", type=int, default=700)
    p.add_argument(
        "--no-curriculum", action="store_true",
        help="train on full multi-Stroke data from step one",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default="")
    p.add_argument("--out-dir", default="runs/raster_vector_gate")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    torch.manual_seed(83)
    specifications = (
        [(args.max_strokes, args.max_segments)]
        if args.no_curriculum else
        [(1, 1), (1, args.max_segments),
         (args.max_strokes, args.max_segments)]
    )
    datasets = []
    print("building synthetic raster caches...", flush=True)
    for stage, (strokes, segments) in enumerate(specifications):
        train = make_scenes(
            83001 + stage * 101, args.train_scenes, strokes, segments
        )
        validation = make_scenes(
            93001 + stage * 101, args.eval_scenes, strokes, segments
        )
        train_images = build_raster_cache(
            train, args.max_strokes, args.max_segments,
            device, args.image_size,
        )
        validation_images = build_raster_cache(
            validation, args.max_strokes, args.max_segments,
            device, args.image_size,
        )
        datasets.append((
            train, train_images, validation, validation_images
        ))
        print(
            f"  stage {stage}: strokes<={strokes} segments<={segments} "
            f"train={tuple(train_images.shape)}", flush=True,
        )

    model = RasterToStatefulAST(
        d_model=args.d_model, n_heads=4,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        patch_size=4, max_strokes=args.max_strokes,
        max_segments=args.max_segments,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    start_step = 0
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        history = list(checkpoint.get("history", []))
        print(
            f"resumed {args.resume} at step {start_step}", flush=True
        )
    best_shape = max(
        (row.get("shape_iou", -1.0) for row in history),
        default=-1.0,
    )
    started = time.time()

    for step in range(start_step + 1, args.steps + 1):
        if len(datasets) == 1:
            stage = 0
        elif step <= args.stage0_end:
            stage = 0
        elif step <= args.stage1_end:
            stage = 1
        else:
            stage = 2
        train, train_images, validation, validation_images = datasets[stage]
        if step <= args.warmup:
            scale = step / args.warmup
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
        target = pack_stateful_scenes(
            scenes, args.max_strokes, args.max_segments, device
        )
        image = train_images[ids].to(device)
        noisy = (
            (image + args.input_noise * torch.randn_like(image)).clamp(0, 1)
            if args.input_noise > 0 else image
        )
        model.train()
        out = model(noisy)
        direct, terms = stateful_losses(out, target)
        render_loss = direct.new_zeros(())
        render_terms = {}
        if args.render_weight and step % args.render_every == 0:
            rb = min(args.render_batch, args.batch_size)
            small = StatefulOutput(**{
                key: value[:rb] for key, value in vars(out).items()
            })
            soft = output_to_packed(small, soft_structure=True)
            pred_img = render_packed_stateful(
                soft, size=args.image_size, curve_samples=10,
                pixel_chunk=256, distance_softmin_px=1.0,
            )
            render_loss, render_terms = foreground_render_loss(
                pred_img, image[:rb]
            )
        loss = direct + args.render_weight * render_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(
                model, validation, validation_images, device
            )
            row = {
                "step": step, "stage": stage,
                "loss": float(loss.detach()),
                "direct": float(direct.detach()),
                "render": float(render_loss.detach()),
                "grad_norm": float(grad),
                "minutes": (time.time() - started) / 60,
                **{
                    key: value for key, value in ev.items()
                    if key not in ("source", "pred")
                },
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
                    list(ev["source"][:8].cpu()),
                    list(ev["pred"][:8].cpu()),
                ],
                str(out_dir / f"preview_{step:05d}.png"),
            )
            torch.save({
                "architecture": "raster_to_stateful_ast_v1",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(args), "step": step,
                "history": history,
            }, out_dir / "latest.pt")
            if ev["shape_iou"] > best_shape:
                best_shape = ev["shape_iou"]
                torch.save({
                    "architecture": "raster_to_stateful_ast_v1",
                    "model": model.state_dict(),
                    "config": vars(args), "step": step,
                    "metrics": {
                        key: value for key, value in ev.items()
                        if key not in ("source", "pred")
                    },
                }, out_dir / "best.pt")

    train, train_images, validation, validation_images = datasets[-1]
    ev = evaluate(model, validation, validation_images, device)
    passed = (
        ev["shape_iou"] >= 0.70
        and ev["present_acc"] >= 0.98
        and ev["count_acc"] >= 0.90
    )
    result = {
        "passed": passed,
        "architecture": "raster_to_stateful_ast_v1",
        "history": history,
        "final": {
            key: value for key, value in ev.items()
            if key not in ("source", "pred")
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"{'PASS' if passed else 'FAIL'}: "
        f"shape={ev['shape_iou']:.3f} strict={ev['strict_iou']:.3f} "
        f"present={ev['present_acc']:.3f} count={ev['count_acc']:.3f}",
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
