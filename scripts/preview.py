#!/usr/bin/env python
"""Dump a target-vs-reconstruction grid so you can look at it, not just at
numbers.

    python scripts/preview.py runs/vecgpt/final.pt --stage 3 -o preview.png
    python scripts/preview.py runs/vecgpt/final.pt --ood -o ood.png
    python scripts/preview.py --tokenizer-ceiling -o ceiling.png

`--tokenizer-ceiling` needs no checkpoint: it renders each scene against
itself-round-tripped-through-tokens, i.e. the best any model could ever do
with this token grid. Worth looking at before blaming a model.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vecgpt.data import gen_ood, sample_scene
from vecgpt.model import VecGPT
from vecgpt.render import image_iou, render_batch
from vecgpt.tokenizer import decode, encode
from vecgpt.train import Cfg


def to_png(rows, path, cell, gap=4):
    from PIL import Image

    n_rows, n_cols = len(rows), len(rows[0])
    W = n_cols * cell + (n_cols + 1) * gap
    H = n_rows * cell + (n_rows + 1) * gap
    canvas = torch.full((H, W, 3), 0.85)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            y, x = gap + r * (cell + gap), gap + c * (cell + gap)
            canvas[y : y + cell, x : x + cell] = img
    arr = (canvas.clamp(0, 1) * 255).byte().numpy()
    Image.fromarray(arr).save(path)
    print(f"wrote {path}  ({W}x{H})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default=None)
    p.add_argument("--stage", type=int, default=3)
    p.add_argument("--ood", action="store_true", help="held-out shape families")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tokenizer-ceiling", action="store_true")
    p.add_argument("--unconditional", action="store_true",
                   help="sample with the null memory: no image at all")
    p.add_argument("-o", "--out", default="preview.png")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    rng = random.Random(1234)
    scenes = [gen_ood(rng) if a.ood else sample_scene(rng, a.stage) for _ in range(a.n)]
    size = 64

    if a.tokenizer_ceiling:
        recon = [decode(encode(s).tokens) for s in scenes]
        label = "tokenizer round-trip"
    else:
        if not a.checkpoint:
            p.error("give a checkpoint, or use --tokenizer-ceiling")
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        cfg = Cfg(**ck["cfg"]) if isinstance(ck.get("cfg"), dict) else Cfg()
        size = cfg.image_size
        dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = VecGPT(cfg.image_size, cfg.d_model, cfg.n_heads, cfg.n_layers,
                       cfg.n_seg_heads, cfg.n_stroke_heads, enc_base=cfg.enc_base,
                       n_enc_layers=cfg.n_enc_layers,
                       spatial_bias=getattr(cfg, "spatial_bias", False),
                       region_attention=getattr(cfg, "region_attention", False),
                       n_global_heads=getattr(cfg, "n_global_heads", 2),
                       dynamic_region_masks=getattr(
                           cfg, "dynamic_region_masks", True
                       ),
                       condition_dim=getattr(cfg, "condition_dim", None)).to(dev)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            print(f"checkpoint compatibility: missing={missing}, unexpected={unexpected}")
        model.eval()
        imgs = render_batch(scenes, size=size, device=dev)
        with torch.no_grad():
            if a.unconditional:
                mem = model.null_memory(len(scenes), dev)
                semantic = model.null_semantic_memory(len(scenes), dev)
            else:
                mem, semantic = model.encode_condition(imgs)
            temp = a.temperature if not a.unconditional else max(a.temperature, 1.0)
            budget = max(
                32,
                max(len(encode(
                    s, hierarchical=getattr(cfg, "hierarchical_regions", True)
                ).tokens) for s in scenes) + 32,
            )
            seqs = model.generate(
                mem, max_tokens=budget, temperature=temp, top_p=0.9,
                semantic_mem=semantic
            )
        recon = [decode(s) for s in seqs]
        label = f"model @ step {ck.get('step', '?')}"

    tgt = render_batch(scenes, size=size).cpu()
    pred = render_batch(recon, size=size).cpu()  # empty decode -> blank -> IoU 0, honestly
    iou = image_iou(pred, tgt)
    empty = [i for i, r in enumerate(recon) if not r]

    to_png([list(tgt), list(pred)], a.out, cell=size)
    print(f"{label}: mean IoU {iou.mean():.3f}  per-sample "
          f"{[round(float(v), 2) for v in iou]}")
    if empty:
        print(f"  NOTE: {len(empty)} sample(s) decoded to an empty scene "
              f"(blank cell, IoU 0): {empty}")
    print("  top row = target, bottom row = reconstruction")


if __name__ == "__main__":
    main()
