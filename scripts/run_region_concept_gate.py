#!/usr/bin/env python
"""Complex REGION reconstruction gate.

Human motif names are reported only as an optional post-hoc probe.  They are
not training targets and are deliberately excluded from the pass criterion:
VecGPT concepts may be distributed and need not align with human categories.
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
import torch.nn.functional as F

from vecgpt.complex_data import make_complex_scenes
from vecgpt.region_ast import (
    PackedRegionPrograms,
    RegionASTAutoencoder,
    RegionASTOutput,
    pack_region_programs,
    region_losses,
    region_output_to_packed,
    render_region_programs,
)
from vecgpt.render import (
    foreground_render_loss,
    image_iou,
    image_iou_shape,
    save_grid,
)


def _slice_dataclass(value, n):
    fields = {}
    for key, item in vars(value).items():
        if key == "local":
            # Local leading dimension is B*R.
            fields[key] = type(item)(**{
                k: v[: n * value.max_regions]
                for k, v in vars(item).items()
            })
        elif isinstance(item, torch.Tensor) and item.ndim:
            fields[key] = item[:n]
        elif key == "batch_size":
            fields[key] = n
        else:
            fields[key] = item
    return type(value)(**fields)


def _part_retrieval(latent, scenes):
    vectors, labels, scene_ids = [], [], []
    for b, regions in enumerate(scenes):
        for r, region in enumerate(regions):
            vectors.append(latent[b, r])
            labels.append(region.diagnostic_kind)
            scene_ids.append(b)
    z = F.normalize(torch.stack(vectors), dim=-1)
    similarity = z @ z.T
    ids = torch.tensor(scene_ids, device=z.device)
    similarity.masked_fill_(ids[:, None] == ids[None, :], -torch.inf)
    nearest = similarity.argmax(-1).tolist()
    correct = sum(labels[i] == labels[j] for i, j in enumerate(nearest))
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    majority = max(counts.values()) / len(labels)
    return correct / len(labels), majority


@torch.no_grad()
def evaluate(model, scenes, device, size=64):
    model.eval()
    packed = pack_region_programs(
        scenes, model.max_regions, model.max_strokes,
        model.max_segments, device,
    )
    out = model(packed)
    predicted = region_output_to_packed(
        out, packed, soft_structure=False
    )
    target_img = render_region_programs(
        packed, size=size, curve_samples=20, pixel_chunk=512
    )
    pred_img = render_region_programs(
        predicted, size=size, curve_samples=20, pixel_chunk=512
    )
    present = out.region_present_logits >= 0
    region_acc = (present == packed.region_mask).float().mean()
    true_local = packed.local.stroke_mask
    local_present = out.local.present_logits >= 0
    active = packed.region_mask.reshape(-1)
    stroke_acc = (
        local_present[active] == true_local[active]
    ).float().mean()
    count_acc = (
        out.local.count_logits.argmax(-1)[true_local] + 1
        == packed.local.counts[true_local]
    ).float().mean()
    retrieval, majority = _part_retrieval(out.region_latent, scenes)
    return {
        "strict_iou": float(image_iou(pred_img, target_img).mean()),
        "shape_iou": float(image_iou_shape(pred_img, target_img).mean()),
        "region_present_acc": float(region_acc),
        "stroke_present_acc": float(stroke_acc),
        "segment_count_acc": float(count_acc),
        "part_retrieval": retrieval,
        "part_majority_baseline": majority,
        "target": target_img,
        "pred": pred_img,
    }


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-scenes", type=int, default=1200)
    p.add_argument("--eval-scenes", type=int, default=48)
    p.add_argument("--max-regions", type=int, default=16)
    p.add_argument("--max-strokes", type=int, default=4)
    p.add_argument("--max-segments", type=int, default=8)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--render-weight", type=float, default=0.10)
    p.add_argument("--render-every", type=int, default=5)
    p.add_argument("--render-batch", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default="")
    p.add_argument("--out-dir", default="runs/region_concept_gate")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    torch.manual_seed(71)
    random.seed(71)
    train, _ = make_complex_scenes(71001, args.train_scenes)
    validation, _ = make_complex_scenes(97001, args.eval_scenes)
    needed = max(len(scene) for scene in train + validation)
    if needed > args.max_regions:
        raise SystemExit(
            f"context too small: data needs {needed} regions"
        )

    model = RegionASTAutoencoder(
        d_model=args.d_model, n_heads=4, n_layers=args.layers,
        max_regions=args.max_regions, max_strokes=args.max_strokes,
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
    started = time.time()

    for step in range(start_step + 1, args.steps + 1):
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
        packed = pack_region_programs(
            scenes, args.max_regions, args.max_strokes,
            args.max_segments, device,
        )
        model.train()
        out = model(packed)
        direct, terms = region_losses(out, packed)
        render_loss = direct.new_zeros(())
        render_terms = {}
        if args.render_weight and step % args.render_every == 0:
            rb = min(args.render_batch, args.batch_size)
            small_target = _slice_dataclass(packed, rb)
            small_out = RegionASTOutput(
                out.region_present_logits[:rb], out.frame[:rb],
                type(out.local)(**{
                    key: value[: rb * args.max_regions]
                    for key, value in vars(out.local).items()
                }),
                out.region_latent[:rb], out.scene_latent[:rb],
            )
            soft = region_output_to_packed(
                small_out, small_target, soft_structure=True
            )
            pred_img = render_region_programs(
                soft, size=32, curve_samples=8, pixel_chunk=256,
                distance_softmin_px=1.0,
            )
            target_img = render_region_programs(
                small_target, size=32, curve_samples=10,
                pixel_chunk=256,
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
                "grad_norm": float(grad),
                "minutes": (time.time() - started) / 60,
                **{
                    key: value for key, value in ev.items()
                    if key not in ("target", "pred")
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
                    list(ev["target"][:8].cpu()),
                    list(ev["pred"][:8].cpu()),
                ],
                str(out_dir / f"preview_{step:05d}.png"),
            )
            torch.save({
                "architecture": "region_stateful_ast_v1",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(args),
                "step": step, "history": history,
            }, out_dir / "latest.pt")

    ev = evaluate(model, validation, device)
    passed = (
        ev["shape_iou"] >= 0.70
        and ev["region_present_acc"] >= 0.98
        and ev["stroke_present_acc"] >= 0.95
    )
    result = {
        "passed": passed,
        "architecture": "region_stateful_ast_v1",
        "history": history,
        "final": {
            key: value for key, value in ev.items()
            if key not in ("target", "pred")
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"{'PASS' if passed else 'FAIL'}: "
        f"shape={ev['shape_iou']:.3f} strict={ev['strict_iou']:.3f} "
        f"regions={ev['region_present_acc']:.3f} "
        f"strokes={ev['stroke_present_acc']:.3f} "
        f"retrieval={ev['part_retrieval']:.3f} "
        f"(majority={ev['part_majority_baseline']:.3f})",
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
