# Hybrid Stroke Diffusion

This is a separate experimental branch for VecGPT.  It combines the useful
part of StrokeFusion with VecGPT's differentiable clothoid geometry.

## Representation

Each stroke is a continuous vector:

```text
(x, y, theta, length, kappa, delta_kappa, width, r, g, b)
```

Variable cardinality is represented by a padded stroke tensor plus a learned
`presence` value.  There are no semantic REGION slots.

## UDF choice

StrokeFusion uses an image-like UDF map.  This branch uses a vector UDF
signature instead: fixed 2-D probe points query the soft distance to sampled
clothoid points.  It is an auxiliary geometric descriptor, not a generated
pixel image and not the output representation.

## Training stages

1. Train `StrokeAutoencoder` on vector programs.  It reconstructs clothoid
   parameters and the vector UDF signature.
2. Freeze or slowly unfreeze the autoencoder and train
   `StrokeLatentDiffusion` on the sequence of stroke latents.
3. Condition the denoiser on LLM/text latents.
4. For video, diffuse temporal parameter deltas rather than regenerating all
   strokes at every frame.

Run a GPU smoke test from this folder:

```bash
cd hybrid_stroke_diffusion
PYTHONPATH=. /home/maxwelhelp/main/bin/python -u \
  scripts/run_hybrid_stroke_diffusion_smoke.py \
  --device cuda --ae-steps 80 --diffusion-steps 80
```

The run writes `runs/hybrid_stroke_diffusion_smoke/preview_target_vs_reconstruction.png`.
The preview is a vector-rendered target beside the autoencoder reconstruction;
the diffusion smoke loss alone is not yet a quality benchmark.

Run real SVG training from inside this folder with the parent directory on
`PYTHONPATH`:

```bash
PYTHONPATH=..:. /home/maxwelhelp/main/bin/python -m train_vector_ae \
  --data ../data/vector_raw/strokefusion/tu_berlin \
  --limit 2000 --max-strokes 64 --segment-points 16 \
  --steps 3000 --batch 16 --device cuda \
  --out-dir ../runs/hybrid_vector_ae_tu_berlin_smoke
```
