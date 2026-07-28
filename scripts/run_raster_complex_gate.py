#!/usr/bin/env python
"""Raster -> hierarchical REGION/Stroke AST gate on complex objects."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vecgpt.complex_data import make_complex_scenes
from vecgpt.region_ast import (
    RegionASTOutput,
    pack_region_programs,
    region_losses,
    region_output_to_packed,
    region_layout_loss,
    region_local_loss,
    region_hungarian_indices,
    render_region_programs,
)
from vecgpt.render import (
    foreground_render_loss,
    image_iou,
    image_iou_shape,
    save_grid,
)
from vecgpt.stateful import PackedStatefulStrokes
from vecgpt.visual import RasterToRegionAST


def slice_packed(packed, n):
    local = PackedStatefulStrokes(**{
        key: value[: n * packed.max_regions]
        for key, value in vars(packed.local).items()
    })
    return type(packed)(
        packed.frame[:n], packed.region_mask[:n],
        packed.region_count[:n], local, n, packed.max_regions,
    )


@torch.no_grad()
def build_cache(
    scenes, max_regions, max_strokes, max_segments, device, size,
):
    images = []
    for start in range(0, len(scenes), 8):
        packed = pack_region_programs(
            scenes[start:start + 8], max_regions,
            max_strokes, max_segments, device,
        )
        images.append(render_region_programs(
            packed, size=size, curve_samples=16, pixel_chunk=256
        ).cpu())
    return torch.cat(images)


def load_vector_prior(model, path, device):
    if not path:
        return 0
    state = torch.load(
        path, map_location=device, weights_only=False
    )["model"]
    own = model.state_dict()
    copied = {}
    for key, value in state.items():
        target = key
        if key.startswith("frame_decode."):
            target = "frame_head." + key[len("frame_decode."):]
        if target in own and own[target].shape == value.shape and (
            target.startswith("local_decoder.")
            or target.startswith("frame_head.")
            or target.startswith("present.")
        ):
            copied[target] = value
    model.load_state_dict(copied, strict=False)
    return len(copied)


@torch.no_grad()
def evaluate(model, scenes, images, device, set_mode=False):
    model.eval()
    target = pack_region_programs(
        scenes, model.max_regions, model.max_strokes,
        model.max_segments, device,
    )
    source = images.to(device)
    out = model(source)
    predicted = region_output_to_packed(
        out, target, soft_structure=False, set_structure=set_mode
    )
    pred_img = render_region_programs(
        predicted, size=source.shape[1],
        curve_samples=20, pixel_chunk=512,
    )
    region_acc = (
        (out.region_present_logits >= 0) == target.region_mask
    ).float().mean()
    active = target.region_mask.reshape(-1)
    local_present = out.local.present_logits >= 0
    stroke_acc = (
        local_present[active] == target.local.stroke_mask[active]
    ).float().mean()
    counts = target.local.stroke_mask
    count_acc = (
        out.local.count_logits.argmax(-1)[counts] + 1
        == target.local.counts[counts]
    ).float().mean()
    return {
        "strict_iou": float(image_iou(pred_img, source).mean()),
        "shape_iou": float(image_iou_shape(pred_img, source).mean()),
        "region_acc": float(region_acc),
        "stroke_acc": float(stroke_acc),
        "count_acc": float(count_acc),
        "source": source, "pred": pred_img,
    }


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--train-scenes", type=int, default=1000)
    p.add_argument("--eval-scenes", type=int, default=24)
    p.add_argument(
        "--detail-level", type=int, choices=(0, 1), default=0,
        help="0 learns macro silhouettes before tiny details",
    )
    p.add_argument("--image-size", type=int, default=48)
    p.add_argument("--max-regions", type=int, default=16)
    p.add_argument("--max-strokes", type=int, default=4)
    p.add_argument("--max-segments", type=int, default=8)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--encoder-layers", type=int, default=3)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--warmup", type=int, default=80)
    p.add_argument("--render-weight", type=float, default=0.08)
    p.add_argument("--render-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument(
        "--teacher-frame-fraction", type=float, default=0.40,
        help="fraction of training using exact REGION frames for local crops",
    )
    p.add_argument(
        "--set-to-sequence", action="store_true",
        help="experimental Hungarian unordered REGION + recurrent child path",
    )
    p.add_argument(
        "--layout-fraction", type=float, default=0.25,
        help="initial fraction reserved for REGION-frame-only training",
    )
    p.add_argument(
        "--local-fraction", type=float, default=0.35,
        help="next fraction reserved for oracle-crop local Stroke training",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--vector-prior",
        default="runs/region_complex_gate_v2/latest.pt",
    )
    p.add_argument("--out-dir", default="runs/raster_complex_gate")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    torch.manual_seed(97)
    train, _ = make_complex_scenes(
        97001, args.train_scenes, detail_level=args.detail_level
    )
    validation, _ = make_complex_scenes(
        99001, args.eval_scenes, detail_level=args.detail_level
    )
    print("building complex raster caches...", flush=True)
    train_images = build_cache(
        train, args.max_regions, args.max_strokes,
        args.max_segments, device, args.image_size,
    )
    validation_images = build_cache(
        validation, args.max_regions, args.max_strokes,
        args.max_segments, device, args.image_size,
    )
    print(
        f"cache ready: {tuple(train_images.shape)}", flush=True
    )

    model = RasterToRegionAST(
        d_model=args.d_model, n_heads=4,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        max_regions=args.max_regions,
        max_strokes=args.max_strokes,
        max_segments=args.max_segments,
        unordered_regions=args.set_to_sequence,
        autoregressive_children=args.set_to_sequence,
    ).to(device)
    copied = load_vector_prior(model, args.vector_prior, device)
    print(f"loaded {copied} vector-prior tensors", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.time()

    def set_stage(stage):
        # Layout learns where parts belong. Local learns how a part is drawn
        # in its frame. End-to-end is the only stage exposed to predicted
        # frames. This prevents the large local AST loss from drowning layout.
        for parameter in model.parameters():
            parameter.requires_grad_(stage == "end_to_end")
        if stage == "layout":
            names = ("visual", "region_query", "region_decoder", "present", "frame_head")
        elif stage == "local":
            names = ("crop_encoder", "crop_fuse", "stroke_query",
                     "segment_query", "stroke_decoder", "segment_decoder",
                     "local_decoder", "frame_type", "style_type")
        else:
            return
        for name, module in model.named_modules():
            if name and any(name == prefix or name.startswith(prefix + ".") for prefix in names):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        for name, parameter in model.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in names):
                parameter.requires_grad_(True)

    layout_end = int(args.steps * args.layout_fraction)
    local_end = int(args.steps * (args.layout_fraction + args.local_fraction))
    last_stage = None

    for step in range(1, args.steps + 1):
        if step <= layout_end:
            stage = "layout"
        elif step <= local_end:
            stage = "local"
        else:
            stage = "end_to_end"
        if stage != last_stage:
            set_stage(stage)
            print(json.dumps({"stage": stage, "step": step}), flush=True)
            last_stage = stage
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
        target = pack_region_programs(
            scenes, args.max_regions, args.max_strokes,
            args.max_segments, device,
        )
        image = train_images[ids].to(device)
        model.train()
        # First isolate local inverse graphics with exact synthetic frames.
        # Then remove the teacher completely, forcing end-to-end predicted
        # REGION routing and exposing the model to its inference distribution.
        teacher_until = int(args.steps * args.teacher_frame_fraction)
        teacher_frame = (
            target.frame if stage in ("layout", "local")
            or step <= teacher_until else None
        )
        out = model(image, teacher_frame=teacher_frame)
        if stage == "layout":
            direct, terms = region_layout_loss(
                out, target,
                assignments=(
                    region_hungarian_indices(out, target)
                    if args.set_to_sequence else None
                ),
            )
        elif stage == "local":
            direct, local_terms = region_local_loss(
                out, target,
                assignments=(
                    region_hungarian_indices(out, target)
                    if args.set_to_sequence else None
                ),
            )
            terms = {f"local_{key}": value for key, value in local_terms.items()}
        else:
            direct, terms = region_losses(
                out, target, matching=args.set_to_sequence
            )
        render_loss = direct.new_zeros(())
        render_terms = {}
        if args.render_weight and step % args.render_every == 0:
            small_target = slice_packed(target, 1)
            local = type(out.local)(**{
                key: value[:args.max_regions]
                for key, value in vars(out.local).items()
            })
            small_out = RegionASTOutput(
                out.region_present_logits[:1], out.frame[:1],
                local, out.region_latent[:1], out.scene_latent[:1],
            )
            soft = region_output_to_packed(
                small_out, small_target, soft_structure=True
            )
            pred_img = render_region_programs(
                soft, size=args.image_size, curve_samples=8,
                pixel_chunk=256, distance_softmin_px=1.0,
            )
            render_loss, render_terms = foreground_render_loss(
                pred_img, image[:1]
            )
        loss = direct + args.render_weight * render_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(
                model, validation, validation_images, device,
                set_mode=args.set_to_sequence,
            )
            row = {
                "step": step, "loss": float(loss.detach()),
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
                "architecture": "raster_to_region_ast_v1",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(args), "step": step,
                "history": history,
            }, out_dir / "latest.pt")

    ev = evaluate(
        model, validation, validation_images, device,
        set_mode=args.set_to_sequence,
    )
    passed = (
        ev["shape_iou"] >= 0.60
        and ev["region_acc"] >= 0.95
        and ev["stroke_acc"] >= 0.90
        and ev["count_acc"] >= 0.90
    )
    result = {
        "passed": passed, "history": history,
        "final": {
            key: value for key, value in ev.items()
            if key not in ("source", "pred")
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"{'PASS' if passed else 'FAIL'}: "
        f"shape={ev['shape_iou']:.3f} strict={ev['strict_iou']:.3f} "
        f"regions={ev['region_acc']:.3f} "
        f"strokes={ev['stroke_acc']:.3f} count={ev['count_acc']:.3f}",
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
