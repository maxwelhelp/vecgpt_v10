# Запросы для SciSpace по VecGPT

Ниже не общий поиск «text-to-SVG», а вопросы по оставшимся
архитектурным рискам. Лучше запускать запросы отдельно и просить SciSpace
искать по full text. Для каждого ответа нужны первичные статьи, точные
формулы, ссылка на официальный код и явное указание, что подтверждено
экспериментом, а что является предложением автора ответа.

## 0. Общий контекст, который надо добавлять к каждому запросу

```text
We are designing a prompt/LLM-latent-to-vector-animation model. Its only
rendering primitive is a stateful Stroke. A scene is a variable typed AST:
ROOT -> REGION* -> STROKE* -> FRAME + STYLE + SEGMENT*.

FRAME=(x,y,sin(theta),cos(theta)).
STYLE=(width,RGBA) is emitted once and changed only by sparse STYLE_DELTA.
The proposed geometric segment is a clothoid
(length L, initial curvature kappa_0, curvature change Delta_kappa);
a constant-curvature circular arc is the Delta_kappa=0 special case.

There must be no fixed semantic/object slots and no predefined concepts
such as person, eye, circle, etc. The number and meaning of latent REGION,
STROKE and SEGMENT nodes must emerge from training. Rasterization may be
used for a differentiable training loss but the generated representation
and animation remain vector programs. Video frames should preserve node
identity and encode only temporal deltas for unchanged subtrees.
```

## 1. Самая важная дыра: переменная топология без фиксированных slots

```text
[Insert the common context above.]

Find primary research on neural decoders that generate a variable-cardinality
tree/program/set of continuous geometric primitives without a fixed K object
slot bottleneck. Compare autoregressive insertion/deletion, recursive tree
decoding, adaptive computation, set prediction with learned stop decisions,
and diffusion/flow models over variable-size structured data.

We need to predict ROOT -> REGION* -> STROKE* -> SEGMENT*, including birth,
end, parent-child ownership and continuous parameters. Concepts must emerge
in latent vectors; REGION must not correspond to a predefined class.

Extract:
1. exact mathematical factorization of p(tree, continuous parameters | z);
2. how variable cardinality and parent assignment are trained;
3. whether teacher forcing creates an exposure gap and how it is addressed;
4. permutation/order assumptions;
5. complexity and scaling to hundreds or thousands of nodes;
6. official code links and benchmark evidence.

Exclude papers whose essential solution is a small fixed number of object
slots. Include them only as negative comparisons.
```

## 2. Clothoid: точный differentiable renderer и устойчивые градиенты

```text
[Insert the common context above.]

Find primary papers and implementations for differentiable rasterization of
variable-width clothoid/Euler-spiral strokes, including alpha coverage and
anti-aliasing. We need gradients with respect to
(x0,y0,theta0,L,kappa0,Delta_kappa,width,RGBA), stable both near zero
curvature and near zero curvature rate.

Extract exact equations for:
1. theta(s), x(s), y(s), including correct Fresnel-integral scaling;
2. endpoint and Jacobians with respect to every parameter;
3. distance or coverage from a pixel to a clothoid centreline/offset curve;
4. anti-aliasing and alpha-compositing derivatives;
5. numerical strategy and error bounds (Fresnel functions, quadrature,
   piecewise approximation, implicit differentiation);
6. handling cusps/self-intersections and variable width;
7. GPU implementation and official code.

Compare accuracy, gradient stability and runtime against cubic Bezier
segments and piecewise constant-curvature circular arcs.
```

## 3. Loss для всей кривой, а не независимых параметров

```text
[Insert the common context above.]

Find primary work on losses for learning ordered parametric curves where
small local tangent/curvature errors accumulate along the trajectory.
Compare direct parameter regression, endpoint/pose trajectory losses,
arc-length-aligned point losses, Chamfer/Hausdorff distance, optimal
transport/Sinkhorn matching, signed-distance-field losses, multiscale
differentiable raster losses, tangent/curvature regularization and closure
constraints.

We need losses that preserve exact geometry and topology of thin vector
strokes and remain differentiable for a variable number of segments.

Extract equations, invariances, failure modes, computational cost and
ablation evidence. Specifically answer how to avoid:
- an open predicted loop for a closed target;
- phase/starting-point ambiguity on closed curves;
- many short segments replacing one smooth segment;
- correct raster appearance but wrong editable vector program.
```

## 4. Канонизация, соответствие узлов и неоднозначность программы

```text
[Insert the common context above.]

Find methods for training structured vector graphics when many different
stroke orders, start points, directions, segmentations and layer orders
render to nearly the same image. Focus on canonicalization versus
permutation-invariant matching, cyclic sequence alignment for closed paths,
bipartite/optimal-transport matching of strokes, differentiable sorting,
latent alignment and minimum-description-length objectives.

Extract the matching objective and algorithm, its complexity, whether it
supports nested parent-child trees and how it handles occlusion/alpha layer
order. Include official code and measured ablations.
```

## 5. Самопроизвольные REGION-концепты и иерархическое пространство

```text
[Insert the common context above.]

Find primary work on unsupervised discovery of a variable hierarchical scene
decomposition without class labels and without fixed object slots. REGION is
a learned grouping/routing token, not a SAM mask copied from an input image.
At generation time it must be created from a prompt/LLM latent when no image
exists.

Compare recursive attention/routing, hierarchical VAEs, neural scene graphs,
tree transformers, nonparametric/adaptive object discovery and product
Euclidean-hyperbolic latent spaces.

Extract:
1. objective that prevents empty, duplicate, all-background or tiny masks;
2. how children partition or overlap a parent;
3. how the number/depth of regions is learned;
4. whether masks are necessary or can remain implicit attention fields;
5. evidence that learned nodes correspond to reusable concepts;
6. equations for combining Euclidean physical geometry with hyperbolic tree
   structure without putting x/y coordinates into hyperbolic space.
```

## 6. Векторная анимация: identity, дельты и кэш поддеревьев

```text
[Insert the common context above.]

Find primary papers and official code on temporally coherent stroke-based or
vector animation from video/Lottie/SVG sequences. We need persistent IDs for
REGION/STROKE nodes and a sparse edit stream:
KEEP, TRANSFORM_DELTA, GEOMETRY_DELTA, STYLE_DELTA, FADE, ADD, REMOVE.

Extract exact objectives and algorithms for:
1. matching strokes/regions across frames under motion, occlusion, split and
   merge;
2. birth/death probabilities and persistent identity;
3. using optical flow only as training supervision, not as the final model;
4. keyframe selection and interpolation;
5. penalizing flicker and needless redraws;
6. caching unchanged vector subtrees and predicting residual edits;
7. datasets containing native Lottie/SVG/keyframe structure.

Compare frame-independent vectorization against joint spatiotemporal
optimization and learned delta-program generation.
```

## 7. LLM hidden states -> vector AST, без обязательного растра

```text
[Insert the common context above.]

Find primary research that conditions a structured continuous graphics or
program decoder directly on language-model hidden states or a sequence of
text embeddings. The target is not raster-to-vector: training examples are
native SVG/Lottie/vector programs first, then vectorized video.

Extract:
1. cross-attention/interface between language memory and structured decoder;
2. alignment/pretraining losses between text, scene semantics and geometry;
3. curriculum from vector autoencoding to conditional generation;
4. whether a learned 2D planning field improves spatial relations, and how
   it is generated when no input image exists;
5. prevention of conditioning collapse;
6. scaling to long vector programs and animation.

Separate evidence from raster-first systems and vector-native systems.
```

## Что присылать обратно

Наиболее полезен не пересказ, а:

1. название и ссылка на 5–15 наиболее прямых первичных работ;
2. точные формулы в LaTeX;
3. фрагмент, где описан variable-cardinality/stop/parent mechanism;
4. ссылка на официальный репозиторий и конкретный файл реализации;
5. таблица «идея → какую дыру VecGPT закрывает → ограничение»;
6. противоречащие результаты, если разные статьи дают разные выводы.

