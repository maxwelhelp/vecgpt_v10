# VecGPT Typed-AST — каноническая архитектура

Статус: рабочая спецификация новой основной архитектуры.  
Версия документа: 0.6, 2026-07-27.

Этот файл является источником истины для новой ветки VecGPT. Старые
`vecgpt_context.md`, `plan.md` и секции v12/v13 в
`ARCHITECTURE_EXECUTION_PLAN.md` сохраняются как история экспериментов.
Решения из них действуют только тогда, когда явно перенесены сюда.

## 1. Цель и неизменяемые требования

Продуктовая цель:

```text
prompt / LLM hidden states
  -> переменная векторная программа
  -> редактируемая векторная картинка
  -> persistent AST + temporal deltas
  -> векторная анимация произвольной частоты кадров
```

Система не является `image -> SVG` моделью. Raster-to-vector допускается
как bootstrap, средство подготовки MP4 и дополнительный учитель, но на
продуктовом инференсе исходной картинки может не быть.

Инварианты:

- единственный рисующий примитив — `Stroke`;
- REGION, STROKE и SEGMENT имеют переменное количество;
- нет фиксированных semantic slots и списка понятий;
- «человек», «глаз», «круг», «часть объекта» и другие концепты должны
  возникать в скрытых векторах из данных;
- растр используется для loss и проверки, но не является внутренним
  форматом результата;
- синтаксис программы гарантирует grammar, семантику учит модель;
- анимация изменяет persistent-сцену дельтами, а не перерисовывает каждый
  кадр с нуля;
- неизменившиеся поддеревья используют кэш локальной векторной геометрии.

## 2. Полный продуктовый тракт

```text
LLM hidden-state sequence H_text [T,D]
        │
        ├────────────── global cross-attention ──────────────┐
        ▼                                                    │
ROOT latent                                                  │
        │ dynamic frontier expansion                         │
        ▼                                                    │
REGION* latent + local frame + optional attention field      │
        │                                                    │
        ├── REGION*                                          │
        └── STROKE* latent + persistent identity             │
                ├── FRAME  (x,y,sinθ,cosθ,κ)                 │
                ├── STYLE  (width,R,G,B,A)                   │
                └── SEGMENT* (L,Δκ, optional STYLE_DELTA) ◄──┘
        │
        ▼
declarative vector AST
        │
        ├── renderer/exporter
        └── temporal edit decoder
              KEEP / TRANSFORM / GEOMETRY / STYLE
              ADD / REMOVE / FADE
                    │
                    ▼
              Bernstein/keyframe trajectories
                    │
                    ▼
              vector animation
```

## 3. Латентное пространство и связь с LLM

Идея общего пространства с LLM сохраняется, но уточняется.

### 3.1 Что нельзя делать

Нельзя сжимать всю сложную сцену в один `z_scene` и требовать от него
восстановить тысячи точных параметров. Такой bottleneck уже показал
плохую геометрическую масштабируемость.

Нельзя превращать LLM hidden state в заранее размеченные slots
«человек/глаз/фон». Это зашьёт онтологию вместо её обучения.

### 3.2 Что делаем

LLM передаёт последовательность hidden states:

```text
H_text = [h_1, ..., h_T], h_i ∈ R^D
```

Она является общей semantic memory. Динамические узлы AST получают смысл
через cross-attention:

```text
z_root   = RootDecoder(H_text)
z_region = RegionDecoder(z_parent, H_text, siblings)
z_stroke = StrokeDecoder(z_region, H_text, siblings)
z_seg    = GeometryDecoder(z_stroke, local_context)
```

`z_root` может содержать глобальную композицию, стиль и сюжет, но не
является единственным каналом точной геометрии. Каждый REGION/STROKE/
SEGMENT сохраняет собственный latent. Количество latent-токенов растёт
вместе со сложностью сцены, как количество токенов контекста в LLM.

Для подключения конкретной LLM используется adapter/projector:

```text
LLM width -> RMSNorm -> Linear/MLP -> VecGPT model width
```

Этапы подключения:

1. frozen LLM + обучаемый adapter и VecGPT decoder;
2. LoRA верхних LLM-слоёв;
3. при наличии достаточного датасета — совместное дообучение.

### 3.3 Visual inverse-graphics interface

Для построения датасета из изображений и MP4 тот же AST decoder получает
не LLM memory, а spatial visual memory:

```text
raster
  -> overlapping CNN stem
  -> spatial Transformer patch tokens
  -> foreground summary + cross-attention
  -> sequential STROKE/SEGMENT positions + STOP
  -> stateful vector AST
```

Visual tokens не сжимаются в единственный `z`. Foreground summary является
сильным residual conditioning route, а patch tokens сохраняются для точной
локализации. Это закрывает обнаруженный shortcut, при котором learned query
обходил слабый cross-attention и decoder выдавал среднюю линию.

Для sparse line art фон удаляется из статистики входа:

```text
ink = max_channel(abs(background - image))
visual_input = [sqrt(ink) * normalized_RGB, sqrt(ink)]
```

Это не semantic mask и не REGION label. На синтетике background известен.
На реальных данных он оценивается отдельно; noise augmentation разрешается
только после clean-image mastery и с noise-floor compensation.

## 4. Typed AST и переменная топология

### 4.1 Типы узлов

```text
ROOT
REGION
STROKE
FRAME
STYLE
SEGMENT
```

Это типы синтаксиса, а не semantic classes. REGION не означает заранее
заданный объект и не обязан совпадать с SAM-маской.

### 4.2 Dynamic frontier decoder

Дерево строится по уровням или вставками. На каждом шаге существует
переменный список активных узлов `frontier`. Для каждого активного
родителя предсказываются действия:

```text
STOP_CHILDREN
ADD_REGION
ADD_STROKE
ADD_SEGMENT
```

Независимые frontier-узлы обрабатываются параллельно. Внутри одного
родителя дети добавляются до `STOP_CHILDREN`. Grammar маскирует
недопустимые действия.

Вероятностная модель:

```text
p(T, Θ | H) =
  ∏ p(action_n | parent_n, H, partial_tree)
    p(parameters_n | type_n, parent_n, H, partial_tree)
```

`max_nodes`, `max_depth` и `max_segments` являются лимитами контекста и
безопасности, а не количеством semantic slots.

Механизм переменного числа детей имеет прямое основание в Abstract Syntax
Networks. Для sequential field авторы используют горизонтальное состояние
и Bernoulli-решение продолжения:

```text
p(continue_i = 1 | sibling_state_{i-1}, parent_state)
  = sigmoid(f_gen(sibling_state_{i-1}, parent_state))
```

При `continue_i=0` список детей завершается. TRANX выражает ту же идею
grammar-действиями: `APPLYCONSTR` добавляет ребёнка в поле кардинальности
`*`, а `REDUCE` закрывает поле. Для VecGPT:

```text
APPLYCONSTR -> ADD_REGION / ADD_STROKE / ADD_SEGMENT
REDUCE      -> STOP_CHILDREN
```

Источники:

- https://aclanthology.org/P17-1105/
- https://aclanthology.org/D18-2002/

Они подтверждают variable cardinality и syntactic validity, но не решают
continuous geometry, unordered sibling matching, raster loss и animation.
Эти части остаются собственными расширениями VecGPT.

### 4.3 Teacher forcing и честный rollout

Обучение топологии начинается с teacher-forced parent/frontier, но gate
всегда использует собственный rollout модели. Затем добавляются:

- corruption частичного дерева;
- обучение вставке пропущенных узлов;
- scheduled sampling только после освоения чистой задачи;
- stop/complexity penalty;
- сравнение oracle-tree и predicted-tree метрик.

## 5. Геометрия Stroke

### 5.1 Текущий совместимый формат

Рабочая continuous-ветка сейчас использует:

```text
FRAME   = (x,y,sinθ,cosθ)
STYLE   = (width,R,G,B,A)
SEGMENT = (length,turn)
```

Он сохраняется как baseline и миграционный формат.

### 5.2 Целевой stateful clothoid

```text
Stroke state:
  (x, y, θ, κ, width, R, G, B, A)

Segment:
  (L, Δκ)

Sparse boundary event:
  KAPPA_DELTA = jump applied before a segment

κ(s) = κ0 + (Δκ/L)s
θ(s) = θ0 + κ0 s + 0.5(Δκ/L)s²
```

После сегмента:

```text
(x,y,θ) <- clothoid_end_state(...)
κ       <- κ + Δκ
```

Постоянная дуга является точным частным случаем `Δκ=0`. Прямая является
частным случаем `κ=0, Δκ=0`. Поэтому clothoid не добавляет новую
семантическую сущность — он расширяет физику того же Stroke.

STYLE записывается один раз. Изменение задаётся только событием:

```text
STYLE_DELTA = (Δwidth,ΔR,ΔG,ΔB,ΔA)
```

Если style не меняется, событие не генерируется и не вычисляется повторно.

`KAPPA_DELTA` нужен не постоянно. Он сохраняет точную совместимость со
старыми piecewise-arc данными, где прямая или одна дуга мгновенно
переходит в другую кривизну. Если скачка нет, модель не генерирует событие,
а следующий clothoid автоматически начинает с конечной `κ` предыдущего.

### 5.3 G² не возникает автоматически

Последовательность stateful clothoid-сегментов имеет непрерывную кривизну,
только если следующий сегмент начинает с конечной `κ` предыдущего.
Произвольное соединение независимых clothoid не гарантирует `G²`.

Резкий художественный угол можно выразить:

- коротким сегментом большой кривизны;
- началом нового Stroke;
- редким `KAPPA_DELTA`; старые polygon/zigzag данные уже доказали его
  необходимость для точного round-trip.

## 6. Дифференцируемый clothoid renderer

### 6.1 Что уже реализовано

`vecgpt/clothoid.py` использует фиксированную 8-точечную квадратуру
Гаусса–Лежандра. Она:

- работает в чистом PyTorch;
- пропускает autograd во все параметры;
- не зависит от SciPy;
- не делит напрямую на `sqrt(γ)`;
- имеет проверенный предел постоянной дуги;
- дала конечные ненулевые градиенты по `L`, `κ0`, `Δκ`.

Для обучения нейросети это безопаснее неполного Fresnel-скелета.
Аналитическая/рациональная Fresnel-реализация рассматривается позже как
оптимизация скорости, а не обязательное условие корректности.

### 6.2 Целевой sampling-SDF

1. Получить фиксированное число точек квадратурой.
2. Построить отрезки `[P_i,P_{i+1}]`.
3. Считать точное расстояние пикселя до каждого отрезка, а не до точки.
4. Во время обучения использовать контролируемый soft-min.
5. Покрытие:

```text
coverage = clamp(0.5 + (width/2 - distance) / tau, 0, 1)
alpha    = coverage * style_alpha
```

6. Композировать Stroke в painter order.

Расчёт выполняется tile/chunk-блоками. Тензор полного размера
`B×H×W×all_segments×samples` запрещён из-за памяти.

`tau` задаётся относительно размера пикселя и уменьшается curriculum:

```text
tau ≈ c / min(H,W)
```

Hard `min` допустим в forward/eval, но в train передаёт градиент только
ближайшему сегменту. Soft-min должен иметь ограниченную температуру, иначе
далёкие сегменты начнут ошибочно расширять Stroke.

### 6.3 Реализованный renderer gate

Реализовано в `vecgpt/clothoid_render.py`:

- фиксированная Gauss–Legendre геометрия;
- точное point-to-segment distance;
- hard-min для eval и weighted soft-min для train;
- anti-aliased coverage;
- RGBA painter composition;
- pixel chunking без полного `H×W×M` тензора.

CUDA recovery gate оптимизировал 16 случайных Stroke только через два
render loss (белый и чёрный фон, чтобы разделить RGB и alpha):

| Метрика после 250 steps | Значение |
|---|---:|
| strict IoU | 0.9519 |
| shape IoU | 0.9751 |
| XY MAE | 0.00078 |
| theta MAE | 0.0161 rad |
| length MAE | 0.00052 |
| κ0 MAE | 0.3297 |
| Δκ MAE | 0.6366 |
| width MAE | 0.00022 |
| RGB MAE | 0.0505 |
| alpha MAE | 0.0325 |

Отчёт: `runs/clothoid_renderer_gate_v2/report.json`.

RGB и alpha принципиально неоднозначны на одном фоне:
`I=αC+(1-α)B`. Поэтому style recovery gate использует два известных фона.
На native vector supervision параметры STYLE обучаются напрямую.

### 6.4 Ошибки присланного учебного скелета

- `cumsum(dx * L/num_samples)` сдвигает первую точку от `(x0,y0)`;
- это не trapezoidal integration;
- distance-to-point не является distance-to-curve;
- `torch.min` даёт кусочно-гладкий градиент только победителю;
- `meshgrid` не задаёт `indexing`;
- цвет/alpha и painter composition не определены;
- создаётся дорогой `H×W×M` тензор;
- `γ` и `Δκ` смешиваются: в нашей модели `γ=Δκ/L`;
- одна глобальная константа `0.01` для anti-aliasing неверна при другой
  размерности canvas.

## 7. DSL сохраняется, но меняет роль

Typed AST и есть внутренний DSL. Отдельная длинная строка числовых токенов
больше не является главным вычислительным представлением.

DSL нужен для:

- grammar и валидации;
- сохранения/загрузки;
- редактирования;
- экспорта в SVG/Lottie/собственный runtime;
- temporal edit actions;
- отладки и интерпретации.

Дискретными остаются типы узлов и действия. Геометрические параметры
предсказываются continuous heads. При необходимости сериализации они могут
квантоваться только на границе формата.

LLM не обязана печатать сырой SVG. Она создаёт semantic memory, а
специализированный decoder генерирует валидный typed AST.

## 8. REGION и появление концептов

REGION — динамический latent-узел с:

```text
identity
parent pointer
local Sim(2) frame: translation, rotation, uniform log-scale
children
semantic latent
optional spatial attention field
```

Spatial heatmap/mask не является REGION. Это только один из способов
проверить или направить его область ответственности.

Полный affine/STN со shear пока не используется: он усложняет ширину,
кривизну и интерпретацию дочерней геометрии. `Sim(2)` даёт перенос,
вращение и масштаб целой части, сохраняя форму и углы. Более сложная
деформация выражается geometry deltas дочерних Stroke.

При text-to-vector инференсе REGION создаётся из LLM memory и родительского
latent. При visual bootstrap он может дополнительно читать feature map.
Одно и то же downstream-представление используется в обоих случаях.

Чтобы концепты возникали, а не задавались вручную, измеряем:

- устойчивость region latent к переносу/цвету/малой деформации;
- предсказание дочерних узлов из parent latent;
- latent swap между сценами;
- повторное использование похожего latent в разных контекстах;
- linear probes только для анализа, не как обучающие labels;
- OOD-композицию знакомых частей в новой структуре.

Малое фиксированное число REGION не используется: это снова slots.
Сжатие достигается через динамический STOP и MDL/rate penalty.
Сам MDL не гарантирует семантику и способен схлопнуть всю сцену в один
REGION, поэтому он применяется только вместе с:

- cross-frame co-motion/identity;
- transform equivariance;
- subtree reuse;
- intervention/swap consistency;
- fidelity gate, не позволяющим удалить важные детали.

## 9. Функции потерь

```text
L_total =
  λ_topology   L_action_stop_parent
+ λ_param      L_normalized_parameters
+ λ_traj       L_accumulated_SE2_kappa
+ λ_match      L_local_sibling_matching
+ λ_curve      L_arc_length_or_cyclic_alignment
+ λ_render     L_multiscale_render
+ λ_temporal   L_identity_and_motion
+ λ_complexity L_description_length
```

Обязательные свойства:

- физические параметры нормализуются до сравнимого масштаба;
- trajectory loss оценивает накопленное состояние после каждого сегмента;
- для замкнутой кривой учитывается циклический сдвиг start point;
- Sinkhorn/Hungarian применяется локально к неупорядоченным siblings;
- painter/z-order не уничтожается permutation-invariant loss;
- raster loss не заменяет точный program loss;
- complexity loss штрафует бессмысленное дробление одной кривой на много
  коротких сегментов.
- MDL включается после достижения fidelity threshold; иначе самый дешёвый
  код — пустой рисунок или один чрезмерно общий REGION.

SDS/LPIPS допустимы только как поздние semantic/perceptual rewards. Они не
являются геометрической основой.

## 10. Обучение по этапам

### Gate 0 — математика и renderer

Данные: аналитически заданные единичные примитивы.

Проверки:

- line и constant-arc limits;
- clothoid endpoint и curvature;
- finite difference против autograd;
- gradient по position, angle, `L`, `κ0`, `Δκ`, width, RGBA;
- convergence при прямой оптимизации параметров;
- отсутствие NaN около нулевой кривизны;
- память и скорость tile renderer.

### Gate 1 — synthetic vector identity

Вход уже является векторной программой. Raster encoder отсутствует.

```text
synthetic Stroke AST
  -> typed vector encoder
  -> typed latents
  -> decoder
  -> reconstructed Stroke AST
```

Синтетика постепенно включает:

1. одну прямую;
2. одну дугу;
3. одну clothoid;
4. несколько сегментов;
5. STYLE_DELTA;
6. несколько Stroke;
7. REGION-дерево разной глубины;
8. transform/scale/rotation/color combinations.

Этот этап учит физику и синтаксис, но не понятия «человек/глаз».

### Gate 2 — synthetic procedural concepts

Программный генератор создаёт композиции без semantic inventory в модели:

```text
части -> поддеревья -> сцена
```

Названия генераторов не подаются decoder как классы. Они используются
только для построения train/OOD split. Проверяется, возникают ли повторно
используемые region latents и композиционное обобщение.

### Gate 3 — условная генерация

Сначала используется простой learnable condition encoder или текстовые
шаблоны. Затем он заменяется frozen LLM memory через adapter.

```text
description / source program
  -> conditioning memory
  -> generated AST rollout
```

На этом этапе vector autoencoder остаётся geometry prior/teacher.

### Gate 4 — native SVG/Lottie/keyframes

Главный реальный источник до MP4:

- очистка и канонизация;
- преобразование всех поддерживаемых путей в Stroke;
- извлечение групп/transform hierarchy;
- persistent identity;
- sparse state updates;
- keyframe и interpolation parameters.

Нативная структура ценнее покадрового растра, потому что содержит точные
деревья, identity и временные соответствия.

### Gate 5 — temporal model

Первый кадр генерирует полный AST. Далее:

```text
KEEP
REGION_TRANSFORM_DELTA
GEOMETRY_DELTA
STYLE_DELTA
FADE_IN / FADE_OUT
ADD_SUBTREE / REMOVE_SUBTREE
```

Дельты между keyframes параметризуются Bernstein/B-spline trajectories.
Неизменившиеся узлы не декодируются повторно.

### Gate 6 — MP4 bootstrap

MP4 не конвертируется независимо кадр за кадром как окончательная truth.
Пайплайн:

```text
video
  -> sparse tracking / optical flow / segmentation hints
  -> joint temporal vectorization
  -> persistent IDs
  -> fitted keyframes + trajectories
  -> confidence-filtered pseudo-label AST
```

Низкоуверенные фрагменты не используются как точная program supervision;
для них допустимы render/perceptual losses.

### Gate 7 — совместное обучение с LLM

Смешиваются:

- text ↔ native vector animation;
- instruction ↔ edit delta;
- description ↔ static AST;
- video/text ↔ pseudo-label AST;
- vector continuation/inpainting;
- hierarchy and concept interventions.

Сначала frozen LLM, затем LoRA. Полное дообучение разрешается только после
того, как decoder независимо проходит geometry/topology/temporal gates.

## 11. Анимация, persistent identity и кэш

Из старой системы сохраняются `VectorFrame`, `RegionState`, `FrameDelta` и
`LocalVectorCache`.

Ключ:

- identity — sequence-local pointer, не semantic slot;
- отсутствие id в delta означает `KEEP`;
- transform-only update не компилирует descendants заново;
- geometry cache хранит локальные векторные пути, а не готовые растровые
  кадры;
- compositor применяет новый parent transform при обходе дерева.

Для длинного видео храним:

```text
base AST
+ sparse edit intervals
+ trajectory coefficients
+ occasional keyframe/checkpoint
```

## 12. Что переносится из старой архитектуры

### Переносится почти без изменений

- единый примитив `Stroke`;
- локальные системы координат REGION;
- canonicalization и fit-to-canvas data hygiene;
- differentiable render loss и IoU diagnostics;
- curriculum и mastery gates;
- conditioning memory interface для LLM hidden states;
- persistent `RegionState`, frame diff/apply;
- local vector cache;
- synthetic generators и OOD splits;
- grammar validator и exporters.

### Переносится с адаптацией

| Старое | Новое |
|---|---|
| одна длинная token sequence | typed AST + frontier actions |
| общий scalar-token softmax | categorical topology + continuous heads |
| один scene latent | ROOT summary + переменные node latents |
| image encoder как обязательный вход | optional visual teacher/bootstrap |
| bootstrap `build_regions()` | target tree только для teacher forcing |
| REGION heatmap как структура | REGION latent; heatmap только auxiliary |
| полный frame generation | base AST + sparse deltas |
| постоянная дуга `(L,turn)` | stateful `(κ0; L,Δκ)` |
| повтор style в каждом сегменте | base STYLE + sparse STYLE_DELTA |

### Не переносится в ядро

- фиксированные semantic slots;
- обязательный raster canvas перед decoder;
- один bottleneck-вектор для всей точной геометрии;
- квантование каждого continuous scalar как основной способ обучения;
- полная перерисовка неизменившихся кадров;
- teacher-forcing метрика без собственного rollout;
- SAM-подобная маска как определение понятия REGION.

## 13. Миграция без потери baseline

Старая модель не удаляется до прохождения новой всех gates.

Порядок:

1. continuous typed-AST geometry baseline — выполнено;
2. scaled physical features — выполнено;
3. accumulated trajectory loss — выполнено;
4. clothoid geometry toy — выполнено;
5. segment-SDF clothoid renderer — выполнено;
6. stateful clothoid representation + typed autoencoder gate — выполнено;
7. dynamic REGION frontier — следующий этап;
8. topology rollout gate;
9. Lottie/native temporal dataset;
10. sparse temporal decoder;
11. LLM adapter;
12. переключение основного `train.py`;
13. старая AR-модель остаётся сравнительным baseline/export experiment.

Нельзя одновременно менять renderer, topology decoder, LLM conditioning и
temporal model: при провале будет невозможно определить причину.

## 14. Текущие измеренные результаты

Typed-AST, mixed straight/arc/shape CUDA gate:

| Вариант | Steps | Strict IoU | Shape IoU | Present/count |
|---|---:|---:|---:|---:|
| raw physical scale | 600 | 0.492 | 0.637 | 1.000 / 1.000 |
| scaled width/length | 600 | 0.524 | 0.703 | 1.000 / 1.000 |
| + safe trajectory loss | 1000 | 0.539 | 0.717 | 1.000 / 1.000 |

Stateful typed-AST CUDA gate на нативных программах с 1–2 Stroke,
1–6 сегментами, ненулевым `Δκ`, sparse `KAPPA_DELTA` и
`STYLE_DELTA`:

| Вариант | Steps | Strict IoU | Shape IoU | Present/count |
|---|---:|---:|---:|---:|
| stateful clothoid AST v1 | 800 | 0.728 | 0.838 | 1.000 / 1.000 |

В этом gate target topology не подаётся в renderer реконструкции.
`present` и `count` предсказываются моделью. Для train raster loss
вероятности существования Stroke и survival каждого сегмента входят в
его alpha, поэтому визуальный градиент достигает topology heads.

Артефакты:

- `runs/stateful_ast_gate_v1/result.json`;
- `runs/stateful_ast_gate_v1/preview_00800.png`.

Полный локальный набор: `39 passed`. Этот результат закрывает
stateful geometry plumbing, но пока не проверяет REGION-иерархию,
семантические концепты или условную генерацию из LLM.

Complex local-REGION reconstruction gate:

| Вариант | Steps | Strict IoU | Shape IoU | Topology |
|---|---:|---:|---:|---:|
| complex REGION AST | 1000 | 0.376 | 0.751 | 1.000 |

Этот тест использовал процедурно заданные локальные группы и доказывает
масштабирование Sim(2)-геометрии, но **не** самостоятельное открытие
понятий. Человеческие motif names исключены из pass criterion. Emergent
representation проверяется predictive compression, equivariance,
co-motion и causal interventions, а не совпадением с названиями
«глаз/голова».

Первый synthetic raster-to-vector gate:

| Вход | Output | Steps | Strict IoU | Shape IoU | Present/count |
|---|---|---:|---:|---:|---:|
| 32×32 raster, одна кривая | stateful Clothoid AST | 1200 | 0.464 | 0.711 | 1.000 / 1.000 |

Артефакты:

- `runs/raster_vector_single_observable_v7/result.json`;
- `runs/raster_vector_single_observable_v7/preview_01200.png`;
- `runs/raster_vector_single_observable_v7/best.pt`.

Критические найденные ошибки:

- traversal direction raster-неидентифицируем без геометрической
  канонизации endpoint;
- короткий след 3–4 px не позволяет устойчиво разделить `θ/κ0/Δκ`;
- additive raster noise перед `sqrt(ink)` превращал весь фон в сильную
  saliency и создавал train/eval distribution shift;
- norm-first decoder мог обойти cross-attention через unconditional
  learned-query residual; foreground summary добавлен прямо в query.

Raster-to-hierarchical-object gate (coarse silhouettes, пять
диагностических семейств `person/car/cat/house/tree`; названия не являются
входом модели):

| Вариант | Steps | Strict IoU | Shape IoU | REGION / Stroke / count |
|---|---:|---:|---:|---:|
| global REGION decoder, unstable world sort | 800 | 0.053 | 0.184 | .922 / .792 / .856 |
| local normalized crop, unstable world sort | 600 | 0.070 | 0.243 | .948 / .800 / .866 |
| local crop + stable program traversal | 1000 | 0.099 | 0.321 | .964 / .870 / .953 |

Последний gate **не пройден** (`shape threshold=0.60`). Он выявил:

- глобальная сортировка REGION по world-`y/x` противоречива: при малом
  повороте симметричные левые/правые части меняют target position, и MSE
  складывает их в центре;
- canonical source-program traversal устраняет эту конкретную
  неоднозначность; для внешних SVG без канонического порядка потребуется
  permutation matching, а не world-coordinate sort;
- локальный raster crop в frame REGION улучшает маршрутизацию, но не
  закрывает задачу: с oracle frame получено только `shape=0.383`, а с
  oracle frame + oracle crop `0.391` на диагностической подвыборке;
- следовательно, остаточная ошибка находится и в локальном program
  decoding/topology, и в композиции frame; нельзя сводить её только к
  фону, renderer или числу шагов.

Артефакты:

- `runs/raster_complex_objects_v1/result.json`;
- `runs/raster_complex_coarse_v4/result.json`;
- `runs/raster_complex_coarse_v4/preview_01000.png`.

Следующий curriculum должен разнести задачи: (1) frame-only assembly gate
со стабильным traversal/matching, (2) local-program gate на oracle crops,
(3) end-to-end без teacher frame. До прохождения (1) и (2) добавлять
мелкие глаза/суставы, LLM conditioning или видео запрещено.

Clothoid toy:

| Constant arcs approximating one clothoid | Parameters | Mean error |
|---:|---:|---:|
| 1 | 2 | 0.03709 |
| 2 | 4 | 0.00918 |
| 4 | 8 | 0.00231 |
| 8 | 16 | 0.00058 |
| 16 | 32 | 0.000145 |

Это доказывает plumbing и преимущество ёмкости clothoid на синтетике, но
ещё не доказывает генерацию сложных реальных сцен или появление понятий.

## 15. Ближайший критерий готовности

Следующий обязательный gate:

```text
procedural subtree scenes + paired transformations
  -> dynamic ROOT/REGION frontier rollout
  -> ADD_REGION / ADD_STROKE / STOP_CHILDREN
  -> local Sim(2) frames
  -> stateful Stroke children
  -> equivariance / co-motion / subtree-swap checks
```

Условия:

- переменное число REGION без фиксированных semantic slots;
- honest frontier rollout без target tree;
- syntactic validity 100%;
- fidelity не хуже плоского stateful baseline на тех же сценах;
- перенос/поворот/scale поддерева меняет его frame, а не переписывает
  локальную геометрию;
- paired-frame identity и co-motion лучше случайного matching;
- latent subtree swap переносит целую часть и сохраняет локальную форму;
- MDL включается только после fidelity threshold;
- attention heatmap оценивается как diagnostic и не объявляется REGION.

## 16. Правило обновления документа

После каждого архитектурного изменения сюда добавляются:

1. изменённый контракт данных/latent/action;
2. причина изменения и закрываемая логическая дыра;
3. измеренный gate до/после;
4. что стало deprecated;
5. следующий нерешённый риск.

Новая идея не считается частью архитектуры, пока она не записана здесь и
не имеет отдельного проверяемого gate. История численных прогонов может
оставаться в `ARCHITECTURE_EXECUTION_PLAN.md`, но актуальный контракт
всегда должен находиться в этом файле.

### Changelog

- `0.1 / 2026-07-27`: зафиксированы typed-AST, dynamic frontier,
  LLM semantic memory без single-vector bottleneck, stateful clothoid,
  sparse temporal updates, Bernstein trajectories и staged migration.
- `0.2 / 2026-07-27`: после проверки TRANX/Abstract Syntax Networks
  зафиксировано литературное основание variable child cardinality через
  `continue/stop` и `APPLYCONSTR/REDUCE`; continuous heads и animation
  отмечены как собственное расширение VecGPT.
- `0.3 / 2026-07-27`: segment-distance clothoid renderer прошёл CUDA
  parameter-recovery gate (`strict=0.952`, `shape=0.975`); следующий gate
  переключён на stateful clothoid chains внутри typed-AST.
- `0.4 / 2026-07-27`: добавлены sparse `KAPPA_DELTA`, exact legacy
  piecewise-arc compatibility и differentiable stateful-chain renderer;
  REGION frame расширен с SE(2) до Sim(2), жёсткий region bottleneck
  отклонён в пользу dynamic STOP + осторожного MDL.
- `0.5 / 2026-07-27`: stateful typed-AST прошёл CUDA gate на переменной
  topology (`strict=0.728`, `shape=0.838`, `present/count=1.0/1.0`);
  soft raster path теперь обучает topology через probabilistic alpha.
  Следующий изолированный риск — dynamic REGION frontier и возникновение
  устойчивых part-whole latent без semantic labels.
- `0.6 / 2026-07-27`: реализован spatial raster encoder и первый
  `raster -> stateful Clothoid AST` gate (`shape=0.711`). Зафиксированы
  foreground-gated input, direct conditioning residual, геометрическая
  direction canonicalization и запрет noise augmentation до clean mastery.
  Complex Sim(2) reconstruction прошёл `shape=0.751`, но отделён от
  настоящего emergence gate.
- `0.7 / 2026-07-28`: complex raster-to-REGION gate проверен на
  человеческих фигурах и объектах. Добавлены normalized REGION crops,
  coarse-to-detail curriculum и усиленный composition loss. Найдена и
  устранена противоречивая world-coordinate canonicalization, однако
  end-to-end gate остаётся непройденным (`shape=0.321`); зафиксировано
  обязательное разделение frame-only и local-program gates.
- `0.8 / 2026-07-28`: добавлен экспериментальный Hungarian REGION
  matching и recurrent child decoder (`--set-to-sequence`). На тех же
  данных он дал `shape=0.110` против baseline `0.331`. Причина — matching
  меняет query identity между batch, а локальный decoder привязан к hidden
  query; одного Hungarian недостаточно. Эксперимент изолирован флагом и не
  включён по умолчанию. Добавлен assignment unit-test и `scipy` dependency.
- `0.9 / 2026-07-28`: добавлены DAB-подобные spatial anchors и
  anchor-conditioned local crops. Короткий gate (`300` steps) дал
  `shape=0.103`, то есть anchor embedding и локальный crop сами по себе не
  устранили query jumping. Следующий обязательный механизм — DN-style
  denoising queries или эквивалентная привязка query к конкретному GT region
  на layout-stage; SAM2 оставлен только как будущий teacher для реальных
  данных, не как часть синтетического gate.
