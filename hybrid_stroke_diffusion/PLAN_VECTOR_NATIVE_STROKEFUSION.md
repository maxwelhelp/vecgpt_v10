# Vector-native StrokeFusion adaptation

This plan is based on reading the implementation in
`external/StrokeFusion/data_process/sketch_data.py`,
`external/StrokeFusion/nets/autoencoder.py`,
`external/StrokeFusion/nets/diffusion.py`,
`external/StrokeFusion/models/stroke_fusion.py`, and
`external/StrokeFusion/models/sketch_diffusion.py`.

The goal is **not** to move VecGPT to pixel diffusion.  We keep clothoid
strokes as the representation and replace the paper's raster UDF branch with
a vector spatial encoder.

## What the reference implementation actually does

1. Each sketch is split into strokes and resampled to 64 points.
2. The whole sketch is normalized, then every stroke is normalized in its own
   local box.  The global box `[cx, cy, w, h]` is stored separately.
3. The reconstruction model has two encoders: a Transformer over stroke
   points and a CNN over a 64x64 distance field. Their features are fused into
   a VAE-like latent and decoded back to both modalities.
4. Diffusion input is a per-stroke record:

   ```text
   [presence, cx, cy, w, h, latent]
   ```

   Presence and box are diffused together with the latent. Sampling uses the
   denoised presence to decide which strokes to render.
5. The reference implementation uses a fixed padded sequence, but its data
   and rendering logic treat inactive strokes as invalid and do not use our
   length-sorting convention. The public sampler supports unconditional random
   generation and optional class conditioning.

## Gaps in the current VecGPT branch

- The AE is an MLP over absolute clothoid parameters, not a local-shape
  encoder plus a separate global box.
- Only 16 global probe distances are supplied. For a small stroke this can
  become almost constant and carry little shape information.
- Presence is predicted by the AE, but is not a denoised diffusion variable.
- Diffusion currently masks the loss for inactive strokes but still generates
  a full latent tensor; the fallback that forces one active stroke is only a
  diagnostic hack.
- Strokes are sorted by length and receive Transformer positional embeddings.
  This introduces an arbitrary order and makes equivalent stroke sets look
  different to the denoiser.
- The reverse sampler is a compact approximation, not the same scheduler as
  the reference implementation.

## Target architecture without pixel UDF

Each stroke is represented by:

```text
local sampled curve:       K x (x, y, tangent_x, tangent_y, curvature)
vector probe signature:    P distances + optional directional distances
global placement:          cx, cy, log_w, log_h, z_order
style:                     width, rgba
presence:                  scalar logit
```

The local curve is normalized into its own bounding box. The global box is
kept outside the shape latent. The probe points are defined in this local
coordinate system, not on the global canvas.

The pixel-CNN branch is replaced by:

```text
curve points -> 1D Transformer / depthwise 1D mixer
probe points -> PointNet/Set encoder with shared MLP + attention pooling
curve <-> probes -> cross-attention
```

This is still vector-only: no image grid, no RGB raster, and no pixel loss is
needed for the base model. A PNG renderer remains only for diagnostics.

## Implementation order

### Phase A — representation fix

1. Add a robust SVG/vector preprocessing cache containing local clothoid
   samples, local bbox, global bbox, probe coordinates and presence.
2. Replace 16 global probes with 32–64 local probes. Add tangent and curvature
   samples to the feature stream.
3. Train an ablation with `MLP`, `1D-Transformer`, and `curve+probe
   cross-attention`. Keep the smallest model whose reconstruction preview is
   unchanged or better.
4. Keep the current AE checkpoint only as a baseline; do not mix latent sizes
   between old and new checkpoints.

### Phase B — joint diffusion state

1. Build one diffusion token per padded stroke:

   ```text
   token_i = [z_shape_i, bbox_i, style_i, presence_i]
   ```

2. Train noise prediction on all continuous fields and BCE/logit denoising on
   presence, with inactive tokens masked only in the shape/style losses.
3. Use a presence-aware sampler; remove the forced-one-stroke fallback.
4. Use a permutation-safe training representation. Preferred first test:
   randomize active stroke order every epoch and remove absolute sequence
   positions. If quality drops, add only a learned canonical spatial encoding,
   not length sorting.
5. Use the same DDPM scheduler for training and reverse sampling. Save the
   scheduler configuration in every checkpoint.

### Phase C — evaluation before text

Run three independent checks:

1. AE reconstruction: target vector vs decoded vector.
2. Unconditional generation: generated stroke count, bbox distribution,
   curve statistics, duplicate rate and rendered montage.
3. Class-conditioned generation: train/evaluate with folder labels, but print
   the requested class name on every preview cell. Do not claim conditioning
   works from a loss number alone.

Use the full/balanced TU-Berlin split; `limit=2000` currently covers only the
first alphabetic classes and cannot test `fish`, `car`, or `frog`.

### Phase D — text and animation

1. Replace the class embedding with a text embedding projected to the same
   conditioning dimension.
2. Train on `(caption, vector-program)` pairs. The LLM/VLM is the semantic
   controller; the vector diffusion model remains the geometry generator.
3. Add temporal tokens and temporal attention only after single-frame
   conditional generation is stable.
4. For video, preserve stroke identity and diffuse deltas/keyframes, not a new
   independent set of strokes per frame.

## Acceptance gates

- AE shape/curve reconstruction must remain at least as good as the current
  real-SVG baseline.
- Presence F1 and generated stroke-count distribution must be reported.
- Unconditional samples must not collapse into a central stroke cloud.
- A category preview must show the requested category outperforming a random
  category control.
- Only after these gates pass do we add text conditioning or video.

