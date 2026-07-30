#!/usr/bin/env python
"""Train the clothoid autoencoder on real SVG vector programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from .model import StrokeAutoencoder
    from .render import render_strokes
    from .vector_data import SVGClothoidDataset
except ImportError:  # direct execution from inside hybrid_stroke_diffusion/
    from model import StrokeAutoencoder
    from render import render_strokes
    from vector_data import SVGClothoidDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing SVG files")
    ap.add_argument("--max-strokes", type=int, default=64)
    ap.add_argument("--segment-points", type=int, default=16)
    ap.add_argument("--split-paths", action="store_true", help="explicitly split SVG paths into overlapping clothoids")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--cache-file", default="", help=".pt cache of parsed clothoid tensors")
    ap.add_argument("--cache-only", action="store_true", help="build/load the parsed cache and exit")
    ap.add_argument("--udf-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="runs/hybrid_vector_ae_tu_berlin")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache_file = Path(args.cache_file) if args.cache_file else out / "parsed_dataset.pt"
    raw_ds = SVGClothoidDataset(args.data, args.max_strokes, segment_points=args.segment_points, limit=args.limit, split_paths=args.split_paths)
    if len(raw_ds) < 2:
        raise SystemExit(f"no SVG files found under {args.data}")
    expected = {"n": len(raw_ds), "max_strokes": args.max_strokes, "segment_points": args.segment_points, "split_paths": args.split_paths, "version": 4}
    payload = None
    if cache_file.exists():
        try:
            candidate = torch.load(cache_file, map_location="cpu", weights_only=False)
            if candidate.get("meta") == expected and "labels" in candidate and "categories" in candidate:
                payload = candidate
                print(f"loaded parsed SVG cache: {cache_file}", flush=True)
        except Exception as exc:
            print(f"ignoring invalid cache {cache_file}: {exc}", flush=True)
    if payload is None:
        rows, masks, labels = [], [], []
        categories = sorted({p.parent.name for p in raw_ds.files})
        category_to_id = {name: i for i, name in enumerate(categories)}
        print(f"parsing {len(raw_ds)} SVGs once (cache: {cache_file})", flush=True)
        for i in range(len(raw_ds)):
            p, m = raw_ds[i]
            rows.append(p); masks.append(m); labels.append(category_to_id[raw_ds.files[i].parent.name])
            if (i + 1) % 250 == 0 or i + 1 == len(raw_ds):
                print(f"  parsed {i + 1}/{len(raw_ds)}", flush=True)
        payload = {"params": torch.stack(rows), "valid": torch.stack(masks),
                   "labels": torch.tensor(labels, dtype=torch.long), "categories": categories, "meta": expected}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_file)
        print(f"saved parsed SVG cache: {cache_file}", flush=True)
    ds = TensorDataset(payload["params"].float(), payload["valid"].float())
    if args.cache_only:
        print(f"cache ready: {len(ds)} SVG samples", flush=True)
        return
    n_val = max(1, len(ds) // 100)
    train, val = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(
        train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    model = StrokeAutoencoder(latent_dim=64, hidden=128, udf_dim=args.udf_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    it = iter(loader); history = []
    for step in range(1, args.steps + 1):
        try:
            params, valid = next(it)
        except StopIteration:
            it = iter(loader); params, valid = next(it)
        params, valid = params.to(device), valid.to(device)
        loss, terms = model.loss(params, valid)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite AE loss at step {step}; check short clothoid lengths and curvature")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            row = {"step": step, "loss": float(loss), **{k: float(v) for k, v in terms.items() if k != "z"}}
            history.append(row); print(json.dumps(row), flush=True)
            torch.save({"model": model.state_dict(), "config": vars(args), "history": history}, out / "latest.pt")
    params, valid = val[0]
    with torch.no_grad():
        recon, logits, _ = model(params[None].to(device))
        img_a = render_strokes(params[None].to(device), valid[None].to(device), size=128)[0].cpu()
        img_b = render_strokes(recon, (logits > 0.0).float(), size=128)[0].cpu()
    from PIL import Image
    grid = torch.cat((img_a, img_b), 1).clamp(0, 1).mul(255).byte().numpy()
    Image.fromarray(grid).save(out / "preview_target_reconstruction.png")


if __name__ == "__main__":
    main()
