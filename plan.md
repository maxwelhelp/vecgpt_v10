# VecGPT v10 — Full Diagnostic Report

> Исторический отчёт по architecture v10/v11. Его финальный вердикт
> «architecture is sound» отменён результатами `quick_regions` и разбором
> architecture v12. Актуальный план находится в
> `ARCHITECTURE_EXECUTION_PLAN.md`. Конечная цель — генерация векторной
> графики и анимации из общего LLM latent; raster reconstruction используется
> только как bootstrap геометрического decoder.

## Date: 2026-07-27  |  Run: v10_cached (8000 steps, interrupted by power loss)

---

## 1. Architecture Changes Applied

### 1.1 Leakage Fix (ra/rb removal)
- **`schema.py`:** Removed `ra`/`rb` quantisers (96-bin log axes — main leak channel where
  model predicted `len` from `ra` with MAE 0.65 without image). Coarsened `rt` 128→32 bins.
  Grammar: `TOP → rx → ry → rt → HEAD` + `TOP → x → ...` (absolute strokes).
- **`regions.py`:** `IDENTITY = (0.5,0.5,0,0,0)`. `explicit: bool` on `Region`.
  `build_regions()`: ≥2 strokes → explicit region, <2 → identity frame (no plan tokens).
  `min_split` 3→2.
- **`tokenizer.py`:** `encode()` emits rx/ry/rt only for `reg.explicit`.
  `decode()` handles `TOP→rx` and `TOP→x` paths.

### 1.2 Diagnostics
- **`train.py`:** `leakage_ce()` — computes CE with shuffled images vs correct images.
  Gap = shuf_ce − ce. Gap ≈ 0 = model not reading image. Gap > 0 = model uses image.
  Extended `loss_fn` to return clean memory.

### 1.3 Speed
- **`render.py`:** `per_seg` 24→12 (cosmetic, no ceiling impact).
- **`scripts/build_cache.py`:** Pre-renders scenes to uint8 cache.
- **`data.py`:** `CachedStream` — reads from cache, eliminates rendering from training loop.
  Stage 3: 543→107 ms/step (5.1× faster). Stage 1: 324→67 ms/step.

---

## 2. Leakage Probe — Definitive Results

### Probe v1 (broken — overfitting)
- ~17K params on 1600 train samples. Memorised train, failed on holdout.
- Produced ratio −1.5 to −1.9 (H|plan >> H). This was a **probe bug**, not a leakage signal.

### Probe v2 (fixed — tiny linear)
- Architecture: 3 plan tokens → Embedding(8, VOCAB) → Linear(24, n_cls). ~23K params.
- L2 weight decay 1e-3. Train/val 80/20 split. 400 steps, batch 128.

**Results:**

| Stage | Field | H (marginal) | H\|plan (holdout) | ratio | Verdict |
|-------|-------|-------------|-------------------|-------|---------|
| 1 | x | 5.28 | 5.08 | +0.04 | Plan empty (identity). Probe learns nothing. ✓ |
| 1 | y | 5.27 | 4.92 | +0.07 | Same. ✓ |
| 1 | theta | 5.33 | 5.12 | +0.04 | Same. ✓ |
| 1 | len | 4.85 | 3.73 | +0.23 | Slightly positive — probe learns marginal from constant plan, not leakage. |
| 1 | turn | 5.78 | 4.51 | +0.22 | Same as len. |
| 4 | x | 5.31 | 6.39 | −0.20 | Plan does NOT help on holdout. ✓ |
| 4 | y | 5.17 | 5.66 | −0.10 | Same. ✓ |
| 4 | theta | 4.48 | 4.52 | −0.01 | **Zero.** Plan gives NO info about theta. ✓ |
| 4 | len | 5.23 | 6.64 | −0.27 | Plan does NOT help. ✓ |
| 4 | turn | 5.76 | 6.14 | −0.07 | Plan does NOT help. ✓ |

**Conclusion: LEAK IS ELIMINATED.** All ratios are near zero or negative.
Plan (rx, ry, rt) does NOT contain generalisable information about stroke fields.
The original bug (len predicted from ra with MAE 0.65) is fixed.

**Missing:** cold-control with perturbed images. The probe doesn't use images —
it only sees plan tokens. A useful control would be: replace real images with
noise during training and check if plan tokens are still predicted. NOT DONE.

---

## 3. Training Run Analysis (v10_cached, steps 0–8000)

Schedule: `1:500, 2:2000, 3:4000, 4:6000, 5:7500` (total 20000).
Interrupted at global step ~8000 (stage 4, step ~1500/6000).

### 3.1 Key Metrics by Stage

| Step | Stage | gap | IoU | shape IoU | r gain | theta gain | len gain |
|------|-------|-----|-----|-----------|--------|------------|----------|
| 0 | 1 | −0.00 | — | — | −0.27 | −0.19 | −0.22 |
| 499 | 1 | **+0.31** | 0.06 | 0.13 | +0.17 | −0.01 | +0.08 |
| 999 | 2 | — | 0.10 | 0.26 | — | — | — |
| 1999 | 2 | — | 0.26 | 0.47 | — | — | — |
| 2499 | 2 | **+3.35** | 0.34 | 0.53 | +1.39 | **+1.79** | +0.56 |
| 2999 | 3 | — | 0.19 | 0.42 | — | — | — |
| 4499 | 3 | — | 0.33 | 0.59 | — | — | — |
| 5499 | 3 | — | 0.34 | 0.58 | — | — | — |
| 6499 | 3 | **+2.76** | 0.39 | **0.65** | +1.66 | +1.44 | **+2.11** |
| 7499 | 4 | — | 0.23 | 0.45 | — | — | — |
| 8000 | 4 | **+1.37** | — | — | +1.13 | +0.94 | +1.99 |

### 3.2 What Works

1. **Gap > 0 on ALL stages with explicit regions.** Gap goes from 0 → +3.35 on stage 2,
   recovers to +2.76 on stage 3, +1.37 on stage 4. The model reads the image.
   This is the SINGLE most important metric — it confirms the leakage fix works.

2. **Color (r/g/b) — the canary — is alive.** Gains: +0.17→+1.39→+1.66→+1.13.
   Color cannot be derived from context; it MUST come from the image.
   In pre-fix runs, color was dead flat at step 500. Now it's positive at step 400.

3. **Plan (rx/ry/rt) IS read from the image on stage 4.** At step 6500: rx −10.40
   (catastrophic transition). By step 8000: rx +0.91, ry +1.09, rt +1.31.
   The model learns to predict WHERE the region is from pixels — plan architecture works.

4. **Shape IoU reaches 0.65 on stage 3.** The model reconstructs SHAPES well (circles,
   polygons, zigzags). Shape IoU is blur-tolerant — measures structure, not registration.

5. **Theta is NOT dead.** The isolated probe proved theta is readable from the encoder
   (CNN probe: CE 5.55→0.22, MAE 1.3 bins). In the full model, theta gain recovers
   after each stage transition: +1.79 (stage 2) → +1.44 (stage 3) → +0.94 (stage 4 mid).
   The apparent decline is because each stage is harder and transitions reset the field.

6. **Cache works.** 5× speedup eliminates rendering bottleneck. Training is GPU-bound now.

---

## 4. PROBLEMS — What's Broken or Incomplete

### 4.1 CRITICAL: No MIN_READABLE (Task 0 not done)

**All IoU comparisons in this report are uncalibrated.** Without running identical
config with 3+ seeds, we don't know the noise floor. IoU differences of 0.05–0.10
may be pure noise. Specifically:
- "IoU dropped 0.34→0.19 on stage 3 transition" — could be noise
- "Shape IoU reached 0.65" — SE on n=32 is unknown
- All ablation comparisons (Task 3) will be **uninterpretable** without MIN_READABLE

**Required:** 3 runs of `1:1000,2:2000` with seeds 0,1,2. Estimated ~40 min total
on cache. Command in Section 6.

### 4.2 Stage 1 is Wasted Computation

On stage 1 (single thick stroke, 1 segment), gap stays near zero for 300+ steps.
The model learns positional priors (len appears at position 4), not image features.
Color only activates at step 400 of stage 1 (+0.17 at step 499).
In pre-fix runs, stage 1 was 2000 steps — mostly wasted.

**Recommendation:** reduce stage 1 to 200–300 steps. Not urgent, but saves ~10% of
training time with no accuracy cost.

### 4.3 Stage Transitions Cause Full Reset

Every curriculum stage change produces a catastrophic drop:
- Stage 2→3: gap +3.35→+0.77, len +0.56→−9.18, IoU 0.34→0.19
- Stage 3→4: gap +2.76→+0.57, rx −10.40, color +1.66→−2.84

Recovery takes ~300–800 steps. This is normal (new data distribution) but means
~10% of training time is "re-learning" rather than progressing.

**Note:** this was described as normal in the original design doc. It recovers.
But if the model is ever trained on real SVG with continuous difficulty, this
won't happen. Not a bug — just a curriculum artefact.

### 4.4 IoU / Shape IoU Divergence

On stage 3 (step 6500): shape IoU = 0.65 but IoU = 0.39. The model gets the
STRUCTURE right but POSITIONS are off. This means:
- Stroke anchor prediction has higher error than stroke geometry prediction
- The model draws the right shape at slightly wrong coordinates
- This is consistent with `x/y` gains being high (+3.52/+3.66) but absolute
  positioning still having residual error

**Diagnosis:** anchor tokens (x/y) are predicted early in the sequence with less
context than later tokens (len/turn/color). The model knows "this is a circle of
radius 0.15" but places it at (0.51, 0.48) instead of (0.50, 0.50).

### 4.5 len Field Recovery is Slow

`len` takes longest to recover after stage transitions:
- Stage 2→3: len −9.18 at step 2500, still −0.20 at step 2600, reaches +0.68 at step 2700
  → recovery in ~200 steps. Normal.
- But compared to `x` (which recovers faster), `len` is the bottleneck.

This is expected: `len` has 256 log-spaced bins spanning 0.002–1.8. At stage transitions,
the length distribution changes, and the model needs to re-learn the new distribution.

### 4.6 No Perturbed-Image Control for Plan Leakage

The leak probe tests whether plan predicts strokes (it doesn't). But it does NOT
test whether the **model** uses the plan as a shortcut. The `gap` metric (shuffled
images) tests whether the model reads the image — it does. But there's no test for:
"If the image is replaced with noise, can the model still predict plan tokens
from preceding context?"

This matters on stage 4-5 where rx/ry/rt appear alongside stroke tokens. If the
model learns "rx at position P → strokes at positions P+1..P+N", that's sequential
leakage, not image reading. `gap` catches this (shuffled image breaks the plan→stroke
correlation), but we haven't explicitly measured it.

### 4.7 Training Interrupted — Missing Stage 4-5 Data

The run stopped at step ~8000 (stage 4, ~25% complete). Missing:
- Stage 4 completion (rx/ry/rt gains at plateau)
- Stage 5 (recursive trees — real test of plan architecture)
- Final IoU / shape IoU at convergence
- OOD evaluation on deep trees (deeper/wider variants)

The run can be resumed with `--resume runs/v10_cached/latest.pt`.

---

## 5. Tasks Status

| Task | Status | Key Finding |
|------|--------|-------------|
| 0 — Noise floor | **NOT DONE** | Blocks all ablation conclusions. Script written. |
| 1 — Leak probe | **DONE** ✓ | LEAK = 0. Plan does not leak stroke fields. Fix confirmed. |
| 2 — Theta probe | **DONE** ✓ | Theta reads from encoder (mae 1.3 bins). Don't drop from loss. |
| 3 — Region attention | **NOT IMPLEMENTED** | Main open hypothesis. Needs Task 0 first. |
| 4 — Text conditioning | **NOT IMPLEMENTED** | Depends on Task 3. |

---

## 6. Commands to Run

### Task 0 — Noise Floor (~40 min)

```bash
cd "/home/maxwelhelp/Загрузки/vecgpt (1)/vecgpt_v10"

# Pre-build stage 1+2 cache if not already done:
PYTHONPATH=. python scripts/build_cache.py --stage 1 --n 5000 --out cache
PYTHONPATH=. python scripts/build_cache.py --stage 2 --n 10000 --out cache

# Three seeds, short curriculum
for seed in 0 1 2; do
    PYTHONPATH=. python scripts/train.py \
      --stages "1:1000,2:2000" \
      --out-dir runs/noise_s${seed} \
      --cache-dir cache --seed $seed \
      --log-every 200 --eval-every 500 --batch-size 32
done

# Analyse
PYTHONPATH=. python scripts/noise_floor.py --out-dir runs/noise --seeds 0,1,2
```

### Resume interrupted training

```bash
cd "/home/maxwelhelp/Загрузки/vecgpt (1)/vecgpt_v10" && \
PYTHONPATH=. python -u scripts/train.py \
  --stages "1:500,2:2000,3:4000,4:6000,5:7500" \
  --out-dir runs/v10_cached \
  --cache-dir cache \
  --resume runs/v10_cached/latest.pt \
  --batch-size 32 \
  2>&1 | tee -a runs/v10_cached/console.log
```

---

## 7. Verdict

**Architecture is sound.** Leakage is fixed. Model reads the image (gap > 0, color alive).
Plan is not a shortcut — probe confirms no generalisable leak, and plan tokens themselves
are read from pixels on stage 4.

**Main gaps:** (1) No noise floor → ablation results unreadable. (2) No region attention
→ main open hypothesis untested. (3) Training interrupted → missing stage 4 plateau and
all of stage 5.

**Next step:** Task 0, then Task 3 ablation (region attention vs global).

---

## 8. v14 correction — the previous verdict is superseded

The statement above that the architecture was already sound was too strong.
A held-out, variable single-Stroke gate exposed several independent logical
errors that the fixed 32-scene geometry sanity test could not expose (that
test was largely a memorisation/round-trip check).

### Confirmed errors

1. The old preview mixed target, reconstruction, and an **untrained**
   unconditional sample. With `cond_dropout=0`, the third row was guaranteed
   garbage and must not be used to judge reconstruction. New previews contain
   only target/reconstruction.
2. The former stage 1 was not a straight-line gate: `STRAIGHT_PROB=0.15` meant
   85% of examples were curved. Stage 0 is now exactly one variable straight
   Stroke.
3. The old pixel L1 was averaged over an almost entirely white canvas and then
   multiplied by 0.2. Its typical contribution was about `0.007` beside a
   token loss around `3.8`; the advertised differentiable raster loop was
   therefore almost decorative.
4. Greedy decoding compared one EOS token with each of 257 x bins separately.
   It could choose an empty scene even when the total probability of starting a
   Stroke was larger. Generation now chooses a grammar edge/type first and a
   value inside that edge second.
5. The renderer trained the expected numeric value while production generation
   used the argmax bin. A broad length distribution could render an acceptable
   expected line while its argmax decoded to a near-zero line. Numeric fields
   now also receive an ordinal/circular distance loss.

### Implemented Im2Vec-style bootstrap, adapted to Stroke

```
raster
  -> Conv/attention visual encoder (dense spatial + semantic memory)
  -> autoregressive Stroke token distributions
  -> differentiable soft Stroke parameters
  -> differentiable vector renderer
  -> foreground-normalised transport/coverage/moment/colour loss
```

This is training scaffolding only. The production path remains:

```
LLM/text latent -> the same decoder memory interface -> vector program
```

No semantic class slots, named shapes, Bezier primitive, or raster output were
added. Regions remain variable program tokens and can later serve as layers,
stable animation identities, and cache/delta boundaries.

### Measured short A/B gate on Tesla P40

Both runs used held-out variable straight lines, batch 32, no regions:

| Run | Steps | Shape IoU | Visual result |
|---|---:|---:|---|
| foreground raster loss only | 300 | 0.223 | mostly dots |
| + ordinal/circular numeric loss | 300 | 0.285 (peak 0.316) | short lines |
| same checkpoint continued | 600 | 0.433 | position/angle improve; length remains short |

Tokenizer ceiling is about 0.909. Therefore v14 is materially better but still
does **not** pass the required 0.75 primitive mastery gate.

### Correct next order

1. Keep regions, masks, animation, and MP4 conversion disabled.
2. Fix the basic visual readout for extent/width/colour. The present
   max-pooled semantic branch learns foreground location much faster than line
   extent. Replace it with a multiscale, spatially preserving Stroke-parameter
   readout and test against the same held-out stage-0 gate.
3. Require shape IoU >= 0.75 on variable stage 0, then master curved
   single-Stroke stage 1.
4. Only then enable variable hierarchical REGION tokens.
5. After static regions work, add stable temporal identity, KEEP/DELTA
   programs, and compiled-vector caching for Lottie/SVG/keyframes, then
   temporally consistent MP4-derived vector supervision.

The immediate blocker is not a lack of regions and not a need for longer blind
training. It is incomplete recovery of the continuous parameters of one Stroke,
especially length and width, from the visual latent.
