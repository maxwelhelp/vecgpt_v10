#!/usr/bin/env python
"""Pre-render scenes + tokenize to disk. Training reads from cache.

Usage:
    python scripts/build_cache.py --stage 1 --n 30000 --out cache
    python scripts/build_cache.py --stage 5 --n 15000 --out cache
"""
from __future__ import annotations

import argparse, pickle, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vecgpt.data import sample_scene, collate
from vecgpt.grammar import sample_tree
from vecgpt.render import render_batch


def build(stage: int, n: int = 30000, size: int = 64, chunk: int = 64, out: str = "cache"):
    Path(out).mkdir(parents=True, exist_ok=True)
    rng = random.Random(stage * 1009)
    gen = (lambda: sample_tree(rng)) if stage == 5 else (lambda: sample_scene(rng, stage))

    t0 = time.time()
    print(f"Stage {stage}: {n} scenes -> {out}/stage{stage}.pt")

    all_imgs, all_scenes = [], []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        batch_scenes = [gen() for _ in range(end - start)]
        imgs = render_batch(batch_scenes, size=size, per_seg=12, device="cpu")
        all_imgs.append((imgs * 255).to(torch.uint8))
        all_scenes.extend(batch_scenes)

        if start % (chunk * 20) == 0 and start > 0:
            elapsed = time.time() - t0
            rate = end / elapsed
            eta = (n - end) / rate
            print(f"  {end}/{n}  {rate:.0f}/s  ETA {eta:.0f}s", flush=True)

    imgs_tensor = torch.cat(all_imgs)
    path = f"{out}/stage{stage}.pt"
    torch.save({"imgs": imgs_tensor, "scenes": all_scenes, "n": len(all_scenes)}, path,
               pickle_protocol=pickle.HIGHEST_PROTOCOL)
    mb = imgs_tensor.numel() / 1e6
    print(f"  saved {path}  ({mb:.0f} MB imgs + {len(all_scenes)} scenes)", flush=True)
    print(f"  time: {(time.time() - t0):.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--n", type=int, default=30000)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--out", type=str, default="cache")
    args = p.parse_args()
    build(args.stage, args.n, args.size, out=args.out)


if __name__ == "__main__":
    main()
