# Hybrid Stroke Diffusion — implementation plan

## Current status

- Flat clothoid stroke autoencoder implemented.
- Vector UDF signature implemented as probe-to-curve distances; no pixel
  representation is diffused.
- Differentiable clothoid renderer connected.
- Latent Transformer denoiser smoke-tested on CUDA.
- At 1000 AE steps / 64 padded strokes: `curve=0.00155`,
  `ae_eval_geom=0.00306`, `presence=0.00565`.
- Downloaded real vector data to `data/vector_raw/strokefusion/`:
  TU-Berlin SVG (19,999 files) and Creative Birds/Creatures NPZ.
- Added `SVGClothoidDataset` and `train_vector_ae.py`; long SVG paths are
  split into local clothoid pieces instead of being forced into one arc.
- Added `train_vector_diffusion.py` for the second stage after the real-data
  autoencoder checkpoint is ready.

## Next architecture work

### 0. Fix style calibration before scaling data

- Train width in log/normalized space and give width a separate loss weight;
  the current toy decoder can make thin strokes too thick even when trajectory
  error is low.
- Add a per-field validation table (`xy`, `theta`, `length`, `curvature`,
  `width`, `RGB`) so a low aggregate MSE cannot hide a style failure.

### 1. Finish the vector autoencoder

- Train on a real vector-program corpus, not only random clothoids.
- Store stroke-local normalized geometry plus a global transform/bounding box.
- Keep the global transform separate from local clothoid shape.
- Validate with rendered trajectory distance, stroke count accuracy and
  vector round-trip, not only parameter MSE.

Training data order:

1. synthetic clothoid programs (geometry sanity);
2. SVG/NPZ/JSON stroke datasets converted to clothoids;
3. Lottie/Manim/vector-animation programs with keyframes;
4. MP4 only as a source for vectorization/tracking, not as the primary
   generative supervision.

Raster frames may provide auxiliary render/UDF checks, but the core model
must learn from vector programs and their parameters.

### 2. Remove the global stroke-count limit

The current `max_strokes` is only a padded tensor limit. Replace it for final
generation with a stream of blocks:

```text
STROKE_BLOCK(<=32 strokes) -> NEXT
STROKE_BLOCK(<=32 strokes) -> NEXT
STROKE_BLOCK(<=32 strokes) -> END
```

The controller predicts `NEXT/END`; blocks are not semantic regions or
object slots. This permits any finite program length while keeping each
diffusion problem bounded.

### 3. Make diffusion production-grade

- Replace the smoke linear schedule with a standard cosine DDPM/DDIM or flow
  matching schedule.
- Mask noise loss by `presence` and include a calibrated presence/birth loss.
- Add conditioning from text/LLM latent.
- Add classifier-free guidance only after unconditional training is stable.
- Do not add VGG to the sparse-line objective.  Do not add KL while the
  autoencoder is deterministic; use it only if we deliberately change to a
  variational latent model.

### 4. Animation

- Encode a base vector scene once.
- Diffuse temporal parameter deltas (`dx, dy, dtheta, dlength, dkappa`).
- Use SAM2 only as an external mask/tracking teacher while preparing data;
  it is not part of the generator.
- Add an optional line correspondence cache for unchanged strokes.

### 5. Refinement and export

- Use the differentiable clothoid renderer for final geometric refinement.
- Export the same parameter stream to SVG, Lottie and the vector DSL.
- Keep raster losses as validation/refinement signals, never as the primary
  representation.
