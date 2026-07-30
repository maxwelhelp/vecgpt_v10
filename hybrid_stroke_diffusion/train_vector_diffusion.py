#!/usr/bin/env python
"""Train latent stroke diffusion — v6.

DDPM diffuses [latent(64) | bbox(4)] = 68 dims. Bbox is normalized
to N(0,1) per channel for DDPM compatibility. Presence via AE decoder.
CFG training (cond dropout 10%) but CFG scale=1.0 at sampling (disabled
until conditioning converges).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, TensorDataset

try:
    from .model import StrokeAutoencoder, StrokeLatentDiffusion
    from .geometry import clothoid_bbox, place_clothoids
    from .render import render_strokes
    from .vector_data import SVGClothoidDataset
except ImportError:
    from model import StrokeAutoencoder, StrokeLatentDiffusion
    from geometry import clothoid_bbox, place_clothoids
    from render import render_strokes
    from vector_data import SVGClothoidDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ae-checkpoint", required=True)
    ap.add_argument("--max-strokes", type=int, default=64)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--cache-file", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="runs/hybrid_vector_diffusion_tu_berlin")
    ap.add_argument("--resume", default="")
    ap.add_argument("--conditioned", action="store_true")
    ap.add_argument("--udf-dim", type=int, default=64)
    ap.add_argument("--preview-samples", type=int, default=4)
    ap.add_argument("--preview-steps", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    if args.cache_file:
        cache = torch.load(args.cache_file, map_location="cpu", weights_only=False)
        meta = cache.get("meta", {})
        if meta.get("max_strokes") != args.max_strokes:
            raise SystemExit(f"cache configuration mismatch: {meta}")
        ds = TensorDataset(cache["params"].float(), cache["valid"].float())
        labels = cache.get("labels")
        categories = cache.get("categories", [])
        print(f"loaded parsed vector cache: {args.cache_file} ({len(ds)} samples)", flush=True)
    else:
        ds = SVGClothoidDataset(args.data, args.max_strokes, limit=args.limit)
        labels = None; categories = []
    if args.conditioned and labels is None:
        raise SystemExit("--conditioned requires a cache with labels")
    if args.conditioned:
        ds = TensorDataset(ds.tensors[0], ds.tensors[1], labels.long())
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")

    ae = StrokeAutoencoder(latent_dim=64, hidden=128, udf_dim=args.udf_dim).to(device)
    state = torch.load(args.ae_checkpoint, map_location=device, weights_only=False)
    ae.load_state_dict(state["model"], strict=True); ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)

    # Compute bbox normalization stats from full dataset (one pass)
    print("computing bbox normalization stats...", flush=True)
    bb_all = []
    with torch.no_grad():
        for i in range(0, len(ds.tensors[0]), 128):
            p = ds.tensors[0][i:i+128].to(device)
            v = ds.tensors[1][i:i+128].to(device)
            bb = clothoid_bbox(p)
            bb_all.append(bb[v > 0.5])
    bb_all = torch.cat(bb_all)
    bb_mean = bb_all.mean(0)  # [cx, cy, w, h]
    bb_std = bb_all.std(0).clamp_min(0.01)
    print(f"bbox mean: {bb_mean.tolist()}  std: {bb_std.tolist()}", flush=True)

    cond_dim = len(categories) if args.conditioned else 0
    state_dim = 68  # latent(64) + bbox(4)
    denoiser = StrokeLatentDiffusion(latent_dim=state_dim, model_dim=256, layers=6, heads=8,
                                     cond_dim=cond_dim, use_pos=True).to(device)
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        denoiser.load_state_dict(saved["model"], strict=True)
        print(f"loaded diffusion checkpoint: {args.resume}", flush=True)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    it = iter(loader); history = []
    metrics_path = out / "train_metrics.jsonl"

    for step in range(1, args.steps + 1):
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        if args.conditioned:
            params, valid, labels_batch = batch
            cond = F.one_hot(labels_batch.to(device), num_classes=cond_dim).float()
            drop_mask = (torch.rand(cond.shape[0], 1, device=device) < 0.1).float()
            cond = cond * (1.0 - drop_mask)
        else:
            params, valid = batch; cond = None
        params, valid = params.to(device), valid.to(device)
        perm = torch.argsort(torch.rand(params.shape[0], params.shape[1], device=device), dim=1)
        gather = perm[..., None].expand(-1, -1, params.shape[-1])
        params = params.gather(1, gather)
        valid = valid.gather(1, perm)
        with torch.no_grad():
            z = ae.encode(params)
            bbox = clothoid_bbox(params)
            # Normalize bbox to N(0,1) per channel
            bbox_n = (bbox - bb_mean.to(device)) / bb_std.to(device)
            clean = torch.cat((z, bbox_n), -1)

        t = torch.randint(0, scheduler.config.num_train_timesteps, (clean.shape[0],), device=device).long()
        noise = torch.randn_like(clean)
        noisy = scheduler.add_noise(clean, noise, t)
        pred_noise = denoiser(noisy, t, cond)
        # Weight bbox dims x3 so model learns spatial structure
        err = (pred_noise - noise).square()
        err[..., 64:68] *= 3.0
        loss = err.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)); opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                abar = scheduler.alphas_cumprod[t].to(device)[:, None, None]
                x0_hat = (noisy - (1.0 - abar).sqrt() * pred_noise) / abar.sqrt().clamp_min(1e-4)
                shape_mse = float(((((x0_hat[..., :64] - clean[..., :64]).square()) * valid[..., None]).sum() / valid.sum().clamp_min(1.0) / 64))
                # bbox_mse in normalized space
                bbox_mse_n = float(((((x0_hat[..., 64:68] - clean[..., 64:68]).square()) * valid[..., None]).sum() / valid.sum().clamp_min(1.0) / 4))
                z_hat = x0_hat[..., :64]
                _, ae_logits = ae.decode(z_hat)
                ae_presence = torch.sigmoid(ae_logits)
                ae_recall = float((ae_presence[valid > .5] > 0.5).float().mean()) if (valid > .5).any() else 0.0
                ae_rej = float((ae_presence[valid <= .5] <= 0.5).float().mean()) if (valid <= .5).any() else 0.0

                row = {
                    "step": step, "loss": float(loss),
                    "t_mean": float(t.float().mean()),
                    "x0_shape_mse": shape_mse,
                    "x0_bbox_mse_n": bbox_mse_n,
                    "ae_presence_recall": ae_recall,
                    "ae_presence_rejection": ae_rej,
                    "active_count_mean": float(valid.sum(-1).mean()),
                    "grad_norm": grad_norm,
                }
                if step > 500 and loss > 10.0:
                    print(f"STOPPING: loss exploded to {float(loss):.2f}", flush=True)
                    break
            history.append(row)
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if step % 500 == 0 or step == args.steps or step == 1:
                torch.save({"model": denoiser.state_dict(), "ae_checkpoint": args.ae_checkpoint,
                            "config": vars(args), "history": history,
                            "bb_mean": bb_mean, "bb_std": bb_std},
                           out / "latest.pt")

    # Sampling
    denoiser.eval()
    with torch.no_grad():
        n = max(1, args.preview_samples)
        steps = args.preview_steps
        x = torch.randn(n, args.max_strokes, state_dim, device=device)
        preview_cond = None
        category_names = categories if args.conditioned else None
        if args.conditioned:
            preview_labels = torch.arange(n, device=device) % cond_dim
            preview_cond = F.one_hot(preview_labels, num_classes=cond_dim).float()
        scheduler.set_timesteps(steps, device=device)
        for t in scheduler.timesteps:
            tt = torch.full((n,), int(t), device=device, dtype=torch.long)
            eps = denoiser(x, tt, preview_cond)
            # CFG: amplify conditioning
            if args.conditioned:
                zero_cond = torch.zeros_like(preview_cond)
                eps_uncond = denoiser(x, tt, zero_cond)
                eps = eps_uncond + 2.0 * (eps - eps_uncond)
            x = scheduler.step(eps, t, x).prev_sample

        # Denormalize bbox
        x_final = x.clone()
        x_final[..., 64:68] = x[..., 64:68] * bb_std.to(device) + bb_mean.to(device)

        generated, presence_logits = ae.decode(x_final[..., :64])
        generated = place_clothoids(generated, x_final[..., 64:68])
        presence_prob = torch.sigmoid(presence_logits)
        target_strokes = 15
        generated_valid = torch.zeros_like(presence_prob).scatter_(1,
            presence_prob.topk(target_strokes, dim=-1).indices, 1.0)

        import matplotlib.pyplot as plt
        imgs = render_strokes(generated, generated_valid, size=192, curve_samples=32).clamp(0, 1).cpu()
        target_item = ds[0]
        tp, tv = target_item[:2]
        tp, tv = tp[None].to(device), tv[None].to(device)
        target_recon, target_logits, _ = ae(tp)
        target_img = render_strokes(tp, tv, size=192, curve_samples=32).clamp(0, 1).cpu()[0]
        recon_img = render_strokes(target_recon, (target_logits > 0.0).float(), size=192, curve_samples=32).clamp(0, 1).cpu()[0]
        cols = min(4, n); rows = (n + cols - 1) // cols + 1
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
        axes[0, 0].imshow(target_img); axes[0, 0].set_title("target SVG")
        if cols > 1:
            axes[0, 1].imshow(recon_img); axes[0, 1].set_title("AE reconstruction")
        for c in range(2, cols): axes[0, c].axis("off")
        for i, ax in enumerate(axes.flat[cols:]):
            ax.axis("off")
            if i < n:
                ax.imshow(imgs[i])
                if category_names and preview_labels is not None:
                    ax.set_title(category_names[preview_labels[i].item() % len(category_names)])
                else:
                    ax.set_title(f"sample {i + 1}")
        fig.tight_layout()
        fig.savefig(out / "preview_generated_vectors.png", dpi=140)
        plt.close(fig)
        print(f"preview -> {out / 'preview_generated_vectors.png'}", flush=True)
        print(json.dumps({"generated_active_mean": float(generated_valid.sum(-1).float().mean()),
                          "generated_active_fraction": float(generated_valid.mean())}), flush=True)


if __name__ == "__main__":
    main()
