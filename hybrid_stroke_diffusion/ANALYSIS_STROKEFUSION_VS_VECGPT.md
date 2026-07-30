# StrokeFusion vs VecGPT — полный разбор различий

Дата: 2026-07-30

После полного чтения кода `external/StrokeFusion/` и сравнения с `hybrid_stroke_diffusion/`.

---

## 1. Сравнение архитектур: state representation

### Что диффузится

| | StrokeFusion | VecGPT (наш) |
|---|---|---|
| Размерность | `1 + 4 + 64 = 69` | `1 + 4 + 64 = 69` |
| Каналы | `[flag, cx, cy, w, h, latent(64)]` | `[presence, cx, cy, w, h, latent(64)]` |
| Flag convention | `flag = 1.0` (valid) / `-1.0` (pad). Затем `seqs[...,0:1] *= 0.5` → `±0.5` | `presence = valid - 0.5` → `±0.5` |
| Max seq len | 32 | 64 |

Вердикт: **одинаково.** Разница только в количестве слотов (32 vs 64).

---

## 2. Сравнение архитектур: Transformer denoiser

### StrokeFusion (`nets/diffusion.py`)

```python
class TransformerDiffusion(nn.Module):
    def __init__(self, feature_dim=69, emb_size=512, n_layers=16, n_heads=16, dropout=0.1):
        self.input_proj = nn.Linear(69, 512)
        self.time_emb = Sequential(
            SinusoidalPosEmb(512),
            Linear(512, 2048), GELU,    # ×4 расширение
            Linear(2048, 512)
        )
        self.cond_proj = nn.Linear(512, 512)   # cond уже в emb_size
        # TransformerEncoder: batch_first=True, norm_first=False
        encoder_layer = TransformerEncoderLayer(
            d_model=512, nhead=16, dim_feedforward=2048,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=16)
        self.output_proj = nn.Linear(512, 69)
        # НЕТ positional encoding — вообще
```

### VecGPT (`model.py`)

```python
class StrokeLatentDiffusion(nn.Module):
    def __init__(self, latent_dim=69, model_dim=128, layers=4, heads=4):
        self.in_proj = nn.Linear(69, 128)
        self.time = Sequential(
            SinusoidalTime(128),
            Linear(128, 128), SiLU   # без расширения
        )
        self.cond = nn.Linear(cond_dim, 128)   # cond_dim = num_classes (one-hot → Linear)
        self.pos = nn.Parameter(randn(1, 256, 128))   # ЕСТЬ позиционные эмбеддинги
        # TransformerEncoder: batch_first=True, norm_first=True
        block = TransformerEncoderLayer(
            128, 4, 512, batch_first=True, norm_first=True
        )
        self.net = TransformerEncoder(block, layers=4)
        self.out = nn.Linear(128, 69)
```

### Таблица различий

| Параметр | StrokeFusion (TU-Berlin) | VecGPT (наш) | Разница |
|---|---|---|---|
| `emb_size` / `model_dim` | **512** | 128 | 4× |
| `n_layers` | **16** | 4 | 4× |
| `n_heads` | **16** | 4 | 4× |
| `dim_feedforward` | **2048** | 512 | 4× |
| Активация | **GELU** | SiLU | — |
| Time embedding | ×4 расширение | без расширения | — |
| Position encoding | **нет** | есть (`self.pos`) | — |
| `norm_first` | **False** (post-norm) | True (pre-norm) | — |
| Condition | `Embedding(N, 512) → Linear(512,512)` | `Linear(N, 128)` (над one-hot) | — |
| Dropout | **0.1** | нет | — |
| Параметров | **~53M** | ~860K | **60×** |

---

## 3. Сравнение: training loop

### StrokeFusion (`sketch_diffusion.py:compute_loss`)

```python
seqs = batch['seqs']           # [B, 32, 69]
seqs[..., 0:1] *= 0.5          # flag: ±1 → ±0.5

t = randint(0, 1000, (B,))
noise = randn_like(seqs)
noisy = scheduler.add_noise(seqs, noise, t)

cond_embd = self.cond_embd(cond)
noise_pred = self(noisy, t, cond_embd)

mse = F.mse_loss(noise_pred, noise, reduction='none')  # [B, S, D]
weights = 1   # ← ВСЕГДА 1. Весовая версия закомментирована
               # torch.where(seqs[...,0]>0, 1, 0.0005).unsqueeze(-1)
               # Они ПРОБОВАЛИ взвешивать — но отключили.
loss = (mse * weights).mean()
```

### VecGPT (v10, после исправления)

```python
clean = cat(z, bbox, (valid-0.5)[..., None])  # [B, 64, 69]
t = randint(0, 1000, (B,))
noise = randn_like(clean)
noisy = scheduler.add_noise(clean, noise, t)
pred_noise = denoiser(noisy, t, cond)
loss = F.mse_loss(pred_noise, noise)   # [B, 64, 69] mean
```

### Таблица различий

| | StrokeFusion | VecGPT (v10) |
|---|---|---|
| Loss | `MSE(pred, noise)` | `MSE(pred, noise)` |
| Веса | `weights=1` (равные) | неявно равные |
| Scheduler | `DDPMScheduler(1000, "linear")` | `DDPMScheduler(1000, "linear")` |
| Batch size | 256 | 16 |
| Optimizer | AdamW, lr=1e-4, warmup 10 epochs | AdamW, lr=2e-4, no warmup |
| Эпохи | много (max_epochs=-1) | 15K steps ≈ 120 эпох |

Вердикт: **loss одинаковый.** Принципиальных различий в обучении нет.

---

## 4. Сравнение: sampling / inference

### StrokeFusion (`_sample_rec`)

```python
x = randn(B, 32, 69)
for t in reversed(range(1000)):
    noise_pred = self.diffusion_module(x, t_batch, cond_emb)
    x = scheduler.step(noise_pred, t, x).prev_sample
# После DDPM:
for j in range(32):
    flag = seq[j, 0].item()     # порог: > 0
    if flag <= 0: continue
    bbox = seq[j, 1:5]          # cx, cy, w, h
    enc  = seq[j, 5:]           # latent 64
    pts = decode(enc)           # → local points
    scale = max(w, h)
    xs = pts[:,0]*scale + cx    # placement
    ys = pts[:,1]*scale + cy
```

### VecGPT (v10, после исправления)

```python
x = randn(B, 64, 69)
for t in scheduler.timesteps:
    eps = denoiser(x, tt, preview_cond)
    x = scheduler.step(eps, t, x).prev_sample
# После DDPM:
generated_valid = (x[..., 68] > 0).float()   # порог: > 0
generated = ae.decode(x[..., :64])
generated = place_clothoids(generated, x[..., 64:68])
```

### Таблица

| | StrokeFusion | VecGPT (v10) |
|---|---|---|
| DDPM шаги | `reversed(range(1000))` | `scheduler.timesteps` (то же самое) |
| Threshold | `flag > 0` | `presence > 0` |
| Placement | `pts * scale + center` | `place_clothoids()` (scale + translate) |
| Max seq | 32 | 64 |

Вердикт: **одинаково.** Пороговый отбор `> 0`, DDPM одинаковый.

---

## 5. Корень проблемы — почему у них presence разделяется, а у нас нет

### Что происходит у нас (v10 conditioned)

После DDPM (15000 шагов обучения, batch=16, 2000 примеров, 4 слоя × 128 dim):

```
t=0 presence: mean=-0.024, range=[-0.52, 0.69], positive=46%
→ 30 штрихов на скетч вместо ~13
```

Модель НЕ научилась чисто разделять ±0.5. Значения размазаны непрерывно от -0.52 до +0.69.

### Почему у StrokeFusion работает

1. **53M параметров** против 860K. В 60 раз больше capacity.
2. 16 слоёв × 16 голов = 256 attention heads total. У нас 4×4 = 16.
3. Каждый attention head может специализироваться. С 16 головами хватает capacity выделить 1-2 головы под presence без ущерба для геометрии.
4. С 4 головами все заняты геометрией/bbox, presence получает остаточный сигнал.
5. **Даже они пробовали weight=0.0005 для inactive** — но отключили. Потому что с 53M параметров это не нужно.

### Что НЕ является проблемой

- **Loss**: одинаковый `MSE` у обоих. Не в этом дело.
- **Flag convention**: `±0.5` у обоих.
- **DDPM scheduler**: одинаковый.
- **Threshold `> 0`**: одинаковый.
- **Position encoding**: у них нет, у нас есть. Но это вторично.
- **Активация/Time emb**: GELU vs SiLU, ×4 vs без расширения — вторично.

### Что ЯВЛЯЕТСЯ проблемой

**Модель слишком маленькая для задачи.** 860K параметров не хватает чтобы одновременно выучить:
- 64-мерное пространство латентов (геометрия штрихов)
- 4-мерное пространство bbox (позиция + размер)
- бинарное разделение presence (±0.5)

Все 16 attention heads (4 слоя × 4 головы) заняты геометрией — presence получает недостаточно representational capacity.

---

## 6. Предлагаемое решение

**Масштабировать денойзер до разумных размеров.** Не «тупо поднять параметры», а довести до уровня, где presence-канал получает естественную representational capacity через дополнительные attention heads.

### Конкретные изменения

Только одна строка в `train_vector_diffusion.py`:

```python
# Было:
denoiser = StrokeLatentDiffusion(latent_dim=state_dim, model_dim=128, layers=4, heads=4, ...)

# Стало:
denoiser = StrokeLatentDiffusion(latent_dim=state_dim, model_dim=256, layers=6, heads=8, ...)
```

### Почему именно эти значения

| Параметр | Было | Стало | Обоснование |
|---|---|---|---|
| `model_dim` | 128 | **256** | ×2 ширина = больше representational capacity на каждый attention head. При 128-dim каждый head сжимается до 32-dim векторов — слишком мало для meaningful attention patterns |
| `layers` | 4 | **6** | ×1.5 глубина. 4 слоя дают только 4 шага reasoning. 6 слоёв позволяют более глубокие взаимодействия между штрихами |
| `heads` | 4 | **8** | ×2 головы. С 8 головами модель может распределить: 5-6 на геометрию/bbox, 2-3 на presence/структуру |
| Параметров | 860K | **~5M** | В 10× меньше чем StrokeFusion, но в 6× больше чем текущий. Разумный компромисс |

### Chain of reasoning

1. StrokeFusion доказывает что `MSE` + `flag>0` + `DDPM` работают — **архитектура правильная**
2. У них 53M параметров и presence разделяется
3. У нас 860K и presence размазан
4. Единственное существенное различие — **capacity модели**
5. 53M нам не нужно (у них ещё и image encoder), но 860K — слишком мало
6. 6 слоёв × 8 голов × 256 dim = ~5M — разумный минимум для задачи с 69-мерным состоянием на 64 позициях
7. Никаких костылей с весами, отдельными выходами или AE-декодером — **архитектура остаётся чистой**

### Что НЕ меняется

- Loss: `F.mse_loss(pred_noise, noise)` — без весов
- Threshold: `presence > 0` — без count_head
- Scheduler: `DDPMScheduler(1000, "linear")`
- AE: `runs/hybrid_vector_ae_v5/latest.pt` — без изменений
- Кеш: `tu_berlin_m64_s16_n2000_nosplit_v4.pt` — без изменений

---

## 7. Команда на запуск

```bash
cd "/home/maxwelhelp/Загрузки/vecgpt (1)/vecgpt_v10/hybrid_stroke_diffusion"

PYTHONPATH=..:. /home/maxwelhelp/main/bin/python -m train_vector_diffusion \
  --device cuda \
  --data ../data/vector_raw/strokefusion/tu_berlin \
  --limit 2000 \
  --max-strokes 64 \
  --segment-points 16 \
  --cache-file ../data/vector_cache/tu_berlin_m64_s16_n2000_nosplit_v4.pt \
  --ae-checkpoint runs/hybrid_vector_ae_v5/latest.pt \
  --conditioned \
  --steps 15000 \
  --batch 16 \
  --out-dir runs/hybrid_vector_diffusion_conditioned_v3
```

Единственное изменение — `model_dim=256, layers=6, heads=8` в коде (одна строка).

---

## 8. Ожидаемый результат

При достаточной capacity модели presence-канал должен естественно разделиться на ±0.5:

```
t=0 presence: mean≈-0.27, positive≈20-25%  (вместо текущего 46%)
→ 13-16 штрихов на скетч (вместо 30)
→ читаемые силуэты по категориям
```

Если этого не произойдёт — тогда capacity была не единственной проблемой и нужно копать дальше. Но это **логически обоснованный следующий шаг**, а не случайная догадка.

---

*(дополнить результаты после прогона)*
