#!/usr/bin/env python
"""Probe whether frozen region latents organise by shape concept.

Labels are used only by this post-training diagnostic. They are never passed
to VecGPT and never used by its reconstruction loss.

The probe varies position, angle, size and colour through the normal scene
generators, builds one centroid per held-out shape family from half the
examples, and reports cosine nearest-centroid accuracy on the other half.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from vecgpt.data import TRAIN_SHAPES, collate, fit_to_canvas
from vecgpt.model import VecGPT
from vecgpt.render import render_batch
from vecgpt.scene import canonicalize
from vecgpt.train import ARCHITECTURE_VERSION, Cfg


def make_model(cfg, device):
    return VecGPT(
        cfg.image_size, cfg.d_model, cfg.n_heads, cfg.n_layers,
        cfg.n_seg_heads, cfg.n_stroke_heads, enc_base=cfg.enc_base,
        n_enc_layers=cfg.n_enc_layers,
        spatial_bias=getattr(cfg, "spatial_bias", False),
        region_attention=getattr(cfg, "region_attention", False),
        n_global_heads=getattr(cfg, "n_global_heads", 2),
        dynamic_region_masks=getattr(cfg, "dynamic_region_masks", True),
        condition_dim=getattr(cfg, "condition_dim", None),
    ).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--n-per-family", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ck.get("architecture_version") != ARCHITECTURE_VERSION:
        raise SystemExit(
            f"concept probe requires architecture v{ARCHITECTURE_VERSION}"
        )
    cfg = Cfg(**ck["cfg"])
    model = make_model(cfg, device)
    model.load_state_dict(ck["model"])
    model.eval()

    rng = random.Random(9182)
    scenes, labels = [], []
    for label, generator in enumerate(TRAIN_SHAPES):
        made = 0
        while made < args.n_per_family:
            scene = fit_to_canvas(generator(rng))
            if scene is None:
                continue
            scenes.append(canonicalize(scene))
            labels.append(label)
            made += 1

    vectors = []
    with torch.no_grad():
        for start in range(0, len(scenes), args.batch_size):
            part = scenes[start:start + args.batch_size]
            imgs = render_batch(part, size=cfg.image_size, device=device)
            batch = collate(
                part, device,
                hierarchical=getattr(cfg, "hierarchical_regions", True),
            )
            mem, semantic = model.encode_condition(imgs)
            latents = model.region_latents(
                mem, batch, "complete", semantic_mem=semantic
            )
            vectors.extend(v[-1].cpu() for v, _ in latents)

    x = F.normalize(torch.stack(vectors), dim=-1)
    y = torch.tensor(labels)
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    for label in range(len(TRAIN_SHAPES)):
        ids = (y == label).nonzero().flatten()
        train_mask[ids[:len(ids) // 2]] = True

    centroids = []
    for label in range(len(TRAIN_SHAPES)):
        centroids.append(F.normalize(x[train_mask & (y == label)].mean(0), dim=0))
    centroids = torch.stack(centroids)
    pred = (x[~train_mask] @ centroids.T).argmax(-1)
    acc = float((pred == y[~train_mask]).float().mean())

    names = [fn.__name__.removeprefix("gen_") for fn in TRAIN_SHAPES]
    print(f"families: {names}")
    print(f"nearest-centroid accuracy: {acc:.3f} "
          f"(chance {1 / len(names):.3f})")
    print("Labels were used only after freezing VecGPT.")


if __name__ == "__main__":
    main()
