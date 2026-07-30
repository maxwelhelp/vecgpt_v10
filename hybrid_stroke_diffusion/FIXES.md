# Исправления (v9 → v10)

Дата: 2026-07-30
Относится к: `hybrid_vector_diffusion_v8` → `v10` (conditioned + architectural fixes).

---

## Исправление #1 — Критическое: sigmoid() на bbox при сэмплировании

**Файл**: `train_vector_diffusion.py`, строка 205
**Было**:  
```python
generated = place_clothoids(generated, x[..., 64:68].sigmoid())
```
**Стало**:  
```python
generated = place_clothoids(generated, x[..., 64:68])
```

**Почему**: При обучении `clean[..., 64:68] = clothoid_bbox(params)` — сырые значения в диапазоне [0, 1]. Модель учится восстанавливать именно их. При сэмплировании sigmoid() сжимает диапазон: центр 0.3 → 0.574, центр 0.8 → 0.690. ВСЕ центры bbox стягиваются в район ~0.5-0.7, что даёт «линии из одного центра».

**Эффект**: Позиции штрихов теперь используют полный диапазон [0, 1], генерированные скетчи должны распределяться по всему холсту.

---

## Исправление #2 — Позиционные эмбеддинги в диффузии

**Файл**: `train_vector_diffusion.py`, строка 85
**Было**:  
```python
denoiser = StrokeLatentDiffusion(..., use_pos=False).to(device)
```
**Стало**:  
```python
denoiser = StrokeLatentDiffusion(..., use_pos=True).to(device)
```

**Почему**: Без позиционных эмбеддингов (`use_pos=False`) трансформер видит все 64 слота как идентичные. Хотя порядок штрихов перемешивается каждый шаг (permutation-invariant), позиционные эмбеддинги помогают модели различать слоты и генерировать более разнообразные конфигурации.

---

## Исправление #3 — Баг метрики presence_sign_acc

**Файл**: `train_vector_diffusion.py`, строка 154
**Было**:  
```python
"presence_sign_acc": float((presence_hat.sign() == presence_target).float().mean()),
```
**Стало**:  
```python
"presence_sign_acc": float((presence_hat.sign() == presence_target.sign()).float().mean()),
```

**Почему**: `presence_hat.sign()` возвращает ±1/0, а `presence_target = clean[..., 68]` это ±0.5. Сравнение ±1 == ±0.5 всегда False, поэтому метрика была всегда 0.0. Теперь сравниваются знаки.

---

## v10 — Архитектурные исправления (по образцу StrokeFusion)

После анализа внешнего референса `external/StrokeFusion` и неудачных conditioned прогонов (presence коллапс в 100%, count_head выдавал 59 вместо 13).

### Исправление #4 — Убран count_head

**Файл**: `model.py` (StrokeLatentDiffusion), `train_vector_diffusion.py`
**Что сделано**: Полностью удалён `count_head` из модели. StrokeFusion не имеет отдельного предсказателя количества — он определяет число активных штрихов через `presence_flag > 0` после DDPM. Count_head был лишней сложностью, вызывавшей train-test mismatch (обучался на зашумлённом состоянии, предсказывал на чистом → 59 вместо 13).

### Исправление #5 — Простой MSE loss вместо маскированного

**Файл**: `model.py` (функция `diffusion_loss`)
**Было**: Loss с маскировкой — inactive штрихи получали градиент только по 1/69 измерения (presence). Итого ~4.5% лосса. Модель почти не учила negative presence.
**Стало**: `F.mse_loss(pred_noise, noise)` — простой MSE по всем измерениям всех позиций. Как в StrokeFusion. Все 69 dims × 64 slots равноправны.

### Исправление #6 — Threshold-based сэмплинг вместо count_head

**Файл**: `train_vector_diffusion.py` (секция sampling)
**Было**: `count_head` → top-k селекция → все штрихи активны → каша
**Стало**: `generated_valid = (x[..., 68] > 0).float()` — пороговая селекция как в StrokeFusion. Штрихи с положительным presence флагом рендерятся.

### Исправление #7 — Названия категорий на превью

**Файл**: `train_vector_diffusion.py`
**Стало**: `ax.set_title(category_names[...])` — вместо "sample 1" теперь "airplane", "alarm_clock" и т.д.

---

## Результаты запусков

*(будет дополнено после прогона)*

### AE (v5 — без изменений)

| Step | Loss | Geom | Curve | UDF | Presence |
|------|------|------|-------|-----|----------|
| 5000 | 0.022 | 0.008 | 0.001 | 0.043 | 0.001 |

Кеш: `tu_berlin_m64_s16_n2000_nosplit_v4.pt`
Чекпоинт: `runs/hybrid_vector_ae_v5/latest.pt`

### Diffusion (v9)

**Команда запуска:**
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
  --steps 15000 \
  --batch 16 \
  --out-dir runs/hybrid_vector_diffusion_v9
```

**Метрики на конец обучения (шаги 14900-15000):**

| Метрика | Значение | v8 (для сравнения) |
|---------|----------|---------------------|
| diffusion_loss | 0.05–0.07 | 0.05–0.08 |
| noise_mse_active | 0.04–0.06 | 0.04–0.09 |
| x0_shape_mse | 0.5–3.4 | 1–5 |
| x0_bbox_mse | 0.1–2.2 | 0.5–3 |
| **presence_sign_acc** | **0.74–0.93** | 0.0 (был сломан) |
| presence_active_recall | 0.62–0.87 | 0.62–0.87 |
| presence_inactive_rejection | 0.88–0.93 | 0.78–0.95 |
| count_mae (штрихов) | 2–4 | 3–5 |
| grad_norm | 0.12–0.34 | 0.09–0.32 |

**Анализ:**

1. **presence_sign_acc исправлен** — теперь показывает реальную точность 74-93%. Модель хорошо разделяет активные/неактивные штрихи.

2. **Главная проблема v8 исправлена** — `sigmoid()` на bbox убран, штрихи распределяются по всему холсту, а не из одного центра. Пользователь подтверждает: «линии разные, не однотипные».

3. **x0_bbox_mse снизился** (0.1-2.2 vs 0.5-3 в v8) — bbox reconstruction улучшился.

4. **Обучение стабильное** — нет NaN, loss плавно снижается, градиенты в норме.

5. **Что генерирует модель** — это **безусловная** (unconditional) генерация. Модель обучена на 2000 скетчах TU-Berlin (первые классы по алфавиту, т.к. `limit=2000`). Она генерирует из **смешанного распределения всех категорий** — не конкретный класс, а «усреднённый» скетч. Поэтому силуэты нечитаемые:
   - Чтобы генерить конкретную категорию (машина, рыба...) — нужен флаг `--conditioned`
   - Сами скетчи в TU-Berlin — это абстрактные контурные силуэты (человечки, предметы), они и в target SVG выглядят как непонятные линии

**Вывод:** Модель обучается нормально. Критические баги исправлены. Следующий логический шаг — conditioned diffusion с флагом `--conditioned` для генерации конкретных категорий.

**Для conditioned генерации нужен кеш с labels.** Проверено: кеш `v4` содержит labels (25 категорий, 2000 примеров). Категории: airplane, alarm clock, angel, ant, apple, arm, armchair, ashtray, axe, backpack, и ещё 15.

### Diffusion Conditioned (v10)

**Команда:**
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
  --out-dir runs/hybrid_vector_diffusion_conditioned_v2
```

На превью: sample 0 = airplane, sample 1 = alarm clock, sample 2 = angel, sample 3 = ant — с названиями категорий.

*(дополнить метрики после прогона)*
