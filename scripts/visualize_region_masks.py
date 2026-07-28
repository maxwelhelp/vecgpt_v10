#!/usr/bin/env python
"""Render the soft heatmaps created by dynamic REGION tokens."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from vecgpt.data import collate, sample_scene
from vecgpt.model import VecGPT
from vecgpt.render import render_batch, save_grid
from vecgpt.train import ARCHITECTURE_VERSION, Cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--stage", type=int, default=4)
    p.add_argument("-n", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("-o", "--out", default="region_heatmaps.png")
    args = p.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ck.get("architecture_version") != ARCHITECTURE_VERSION:
        raise SystemExit(
            f"heatmap visualisation requires architecture v{ARCHITECTURE_VERSION}"
        )
    cfg = Cfg(**ck["cfg"])
    if not cfg.dynamic_region_masks:
        raise SystemExit("checkpoint was trained without dynamic region masks")

    model = VecGPT(
        cfg.image_size, cfg.d_model, cfg.n_heads, cfg.n_layers,
        cfg.n_seg_heads, cfg.n_stroke_heads, enc_base=cfg.enc_base,
        n_enc_layers=cfg.n_enc_layers, spatial_bias=cfg.spatial_bias,
        region_attention=cfg.region_attention,
        n_global_heads=cfg.n_global_heads,
        dynamic_region_masks=cfg.dynamic_region_masks,
        condition_dim=getattr(cfg, "condition_dim", None),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    rng = random.Random(2027)
    scenes = [sample_scene(rng, args.stage) for _ in range(args.n)]
    imgs = render_batch(scenes, size=cfg.image_size, device=device)
    batch = collate(
        scenes, device, hierarchical=cfg.hierarchical_regions
    )
    with torch.no_grad():
        mem, semantic = model.encode_condition(imgs)
        maps = model.region_heatmaps(
            mem, batch, semantic_mem=semantic
        )

    rows = []
    for i, (hm, region_ids) in enumerate(maps):
        row = [imgs[i].cpu()]
        for j in range(hm.shape[0]):
            grid = int(math.sqrt(hm.shape[1]))
            m = hm[j].reshape(1, 1, grid, grid)
            m = F.interpolate(
                m, size=(cfg.image_size, cfg.image_size),
                mode="bilinear", align_corners=False,
            )[0, 0]
            m = m / m.amax().clamp_min(1e-9)
            colour = torch.stack([m, torch.zeros_like(m), 1 - m], -1)
            row.append((0.55 * imgs[i].cpu() + 0.45 * colour.cpu()).clamp(0, 1))
        rows.append(row)

    save_grid(rows, args.out)
    print(f"saved {args.out}; each row is input followed by its dynamic REGION maps")


if __name__ == "__main__":
    main()
