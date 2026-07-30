#!/usr/bin/env python
"""Trace every representation boundary and write measurable diagnostics.

This is intentionally vector-only.  It does not render an image to compute
the metrics; rendering is only an optional visual artifact elsewhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_stroke_diffusion.geometry import clothoid_bbox
from hybrid_stroke_diffusion.model import StrokeAutoencoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ae-checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--udf-dim", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    params = cache["params"].float().to(dev)
    valid = cache["valid"].float().to(dev)
    state = torch.load(args.ae_checkpoint, map_location=dev, weights_only=False)
    ae = StrokeAutoencoder(latent_dim=64, hidden=128, udf_dim=args.udf_dim).to(dev)
    ae.load_state_dict(state["model"], strict=True); ae.eval()
    with torch.no_grad():
        recon, logits, z = ae(params[: min(64, len(params))])
        p = params[: recon.shape[0]]; m = valid[: recon.shape[0]]
        active = m > .5
        param_mse = ((recon - p).square().mean(-1)[active]).mean()
        pred_active = logits > 0
        tp = (pred_active & active).sum().float()
        precision = tp / pred_active.sum().clamp_min(1)
        recall = tp / active.sum().clamp_min(1)
        bbox = clothoid_bbox(p)
        metrics = {
            "samples_checked": int(p.shape[0]),
            "active_mean": float(m.sum(-1).mean()),
            "ae_param_mse_active": float(param_mse),
            "ae_presence_precision": float(precision),
            "ae_presence_recall": float(recall),
            "latent_mean": float(z.mean()),
            "latent_std": float(z.std()),
            "bbox_min": float(bbox.min()),
            "bbox_max": float(bbox.max()),
            "bbox_mean_size": [float(x) for x in bbox[..., 2:4].mean((0, 1))],
            "cache_meta": cache.get("meta", {}),
            "categories": cache.get("categories", [])[:32],
        }
    (out / "diagnostics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    dot = '''digraph VecGPTVectorPipeline {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8f1ff"];
  svg [label="SVG / vector strokes"];
  fit [label="clothoid fitting\\nK sampled points"];
  local [label="local shape\\ncurve + tangent + curvature"];
  udf [label="local vector UDF\\n64 probe distances"];
  ae [label="AE encoder\\nshape latent z (64)"];
  bbox [label="bbox\\ncenter + width + height"];
  state [label="joint diffusion token\\nz + bbox + presence"];
  noise [label="DDPM noise / denoise"];
  sample [label="reverse sample\\nset of stroke tokens"];
  decode [label="AE decoder\\nclothoid parameters"];
  place [label="bbox placement\\nscale + translate"];
  render [label="vector renderer\\nPNG only for preview", fillcolor="#fff0d8"];
  svg -> fit -> local; local -> udf; local -> ae; udf -> ae; fit -> bbox;
  ae -> state; bbox -> state; state -> noise -> sample -> decode -> place -> render;
  presence [label="presence\\n-1 inactive / +1 active"];
  presence -> state;
}'''
    (out / "pipeline_graph.dot").write_text(dot, encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"graph -> {out / 'pipeline_graph.dot'}", flush=True)


if __name__ == "__main__":
    main()
