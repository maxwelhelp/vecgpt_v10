import math, random, torch, pytest
from vecgpt.scene import Stroke, chain_points, end_state, reverse_stroke, canonicalize
from vecgpt.render import render_batch, render_one, image_iou
from vecgpt.data import sample_scene, gen_ood


def test_canonical_order_is_stable_under_subpixel_shift():
    """A half-pixel move must not usually reshuffle the stroke order.

    Any total order on continuous positions has boundaries, so this can
    never be 0%. Measured here: ~1% of half-pixel shifts flip an adjacent
    pair. Tolerable for stills; it is the thing to replace with set-level
    matching before delta animation, where a moving object crosses such
    boundaries constantly and a flip desynchronises the whole cache.
    """
    import random

    from vecgpt.data import sample_scene
    from vecgpt.scene import Stroke, canonicalize

    def shifted(scene, d):
        out = []
        for s in scene:
            a = s.anchor.clone()
            a[0] += d
            a[1] += d
            out.append(Stroke(a, s.segs.clone()))
        return out

    sig = lambda S: [tuple(round(float(v), 4) for v in s.segs.flatten()[:4]) for s in S]
    rng = random.Random(7)
    flips = total = 0
    for _ in range(120):
        raw = sample_scene(rng, 4)
        base = canonicalize(raw)
        for d in (0.5 / 64, -0.5 / 64):
            total += 1
            if sig(base) != sig(canonicalize(shifted(raw, d))):
                flips += 1
    assert flips / total < 0.06, f"stroke order unstable: {100*flips/total:.1f}%"


def test_null_memory_generates():
    """Unconditional sampling must run and produce a decodable scene."""
    import torch

    from vecgpt.model import VecGPT
    from vecgpt.tokenizer import decode

    torch.manual_seed(0)
    m = VecGPT(image_size=32, d=64, n_heads=4, n_layers=2, n_seg_heads=1,
               n_stroke_heads=1, enc_base=8)
    m.eval()
    seqs = m.generate(m.null_memory(3), max_tokens=120, temperature=1.0, top_p=0.9)
    assert len(seqs) == 3
    for s in seqs:
        decode(s)  # must not raise


def test_closed_stroke_phase_is_canonical():
    """The same circle drawn from 64 different start angles must produce the
    same anchor token. This is the test that was missing: the old
    invariance test shuffled and reversed strokes but never varied the
    PHASE of a closed one, so it could not see that `canonicalize` fixed
    direction and left the start point to the generator's RNG. Measured
    before the fix: 39 distinct x tokens for r=0.15, i.e. an anchor no
    model can recover, sitting in the first token of the stroke."""
    import math

    import torch

    import vecgpt.schema as S
    from vecgpt.scene import Stroke, canonicalize

    for r in (0.10, 0.15, 0.20):
        toks = set()
        for k in range(64):
            a = -math.pi + 2 * math.pi * k / 64
            st = Stroke(torch.tensor([0.5 + r * math.cos(a), 0.5 + r * math.sin(a), a + math.pi / 2]),
                        torch.tensor([[2 * math.pi * r, 1 / r, 0.03, 0.2, 0.3, 0.8]]))
            toks.add(S.QUANTS["rx"].q(canonicalize([st])[0].anchor[0]))
        assert len(toks) <= 4, f"r={r}: anchor still ambiguous over {len(toks)} bins"

def test_generation_matches_teacher_forcing():
    """The incremental path must give the same logits as the batched one.

    Both now walk the SAME grammar in schema.py, so this guards the
    derivation rather than five hand-kept copies. It is still here because
    the bug it was written for - `generate` labelling each token with the
    NEXT token's state - produced sequences that stayed syntactically valid
    and failed no other test, while generation scored 2% of the ceiling.
    """
    import random

    import torch

    from vecgpt.data import collate, sample_scene
    from vecgpt.model import VecGPT
    from vecgpt.render import render_batch

    torch.manual_seed(0)
    rng = random.Random(0)
    scenes = [sample_scene(rng, 4) for _ in range(2)]
    model = VecGPT(image_size=32, d=64, n_heads=4, n_layers=2, n_seg_heads=1,
                   n_stroke_heads=1, enc_base=8, region_attention=True,
                   n_global_heads=1, spatial_bias=True)
    model.eval()
    imgs = render_batch(scenes, size=32)
    b = collate(scenes)
    assert int(b["region_depth"].max()) >= 2  # explicit root + object child
    with torch.no_grad():
        mem = model.encode(imgs)
        ref = model.logits(
            mem, b["tokens"], b["slots"], b["seg_idx"], b["stroke_idx"],
            b["mask"], b["region_idx"], b["region_depth"],
            b["parent_region_idx"]
        )
        got = model.logits_incremental_debug(mem, b["tokens"], b["mask"])
    for t in range(ref.shape[1] - 1):
        m = b["mask"][:, t]
        if not m.any():
            continue
        assert torch.allclose(ref[:, t][m].softmax(-1), got[:, t][m].softmax(-1), atol=1e-4), t


def test_collate_shapes():
    import random

    from vecgpt.data import collate, sample_scene

    rng = random.Random(6)
    b = collate([sample_scene(rng, 4) for _ in range(4)])
    assert b["tokens"].shape == b["slots"].shape == b["mask"].shape
    assert b["mask"].sum() > 0


def test_turtle_state_executes_straight_segment():
    """The deterministic decoder state must follow the same SE(2) dynamics
    as the renderer; otherwise it would be a misleading positional signal."""
    import vecgpt.schema as S
    from vecgpt.model import turtle_states
    from vecgpt.scene import Stroke, end_state
    from vecgpt.tokenizer import encode

    st = Stroke(
        torch.tensor([0.25, 0.35, 0.4]),
        torch.tensor([[0.2, 0.0, 0.03, 0.2, 0.4, 0.6]]),
    )
    e = encode([st], hierarchical=False)
    state = turtle_states(e.tokens[None], e.region_idx[None])[0]
    turn_lo, turn_hi = S.RANGE["turn"]
    turn_pos = ((e.tokens >= turn_lo) & (e.tokens < turn_hi)).nonzero()[0, 0]
    got = state[turn_pos]
    want = end_state(st.anchor, st.segs)
    assert torch.allclose(got[:2], want[:2], atol=0.01)
    assert torch.allclose(got[2:], torch.stack((want[2].sin(), want[2].cos())),
                          atol=0.03)


def test_external_latent_uses_same_decoder_memory_interface():
    from vecgpt.model import VecGPT

    model = VecGPT(
        image_size=32, d=64, n_heads=4, n_layers=2,
        n_seg_heads=1, n_stroke_heads=1, enc_base=8,
        n_enc_layers=1, condition_dim=96,
    )
    mem = model.condition_latent(torch.randn(3, 5, 96))
    assert mem.shape == (3, 5, 64)
    canvas = model.condition_canvas(torch.randn(3, 5, 96))
    assert canvas.shape == (3, 8 * 8, 64)
    frame_mem = model.condition_animation(
        torch.randn(3, 5, 96), torch.tensor([0.0, 0.5, 1.0])
    )
    assert frame_mem.shape == (3, 6, 64)
    seqs = model.generate(canvas, max_tokens=16, semantic_mem=mem)
    assert len(seqs) == 3


def test_visual_condition_has_distinct_semantic_and_spatial_memories():
    from vecgpt.model import VecGPT

    model = VecGPT(
        image_size=32, d=64, n_heads=4, n_layers=1,
        n_seg_heads=1, n_stroke_heads=1, enc_base=8,
        n_enc_layers=1,
    )
    imgs = torch.ones(2, 32, 32, 3)
    imgs[0, 4:12, 5:7] = 0.0
    imgs[1, 20:28, 24:26] = 0.0
    spatial, semantic = model.encode_condition(imgs)
    assert spatial.shape == (2, 8 * 8, 64)
    assert semantic.shape == (2, 16, 64)
    assert not torch.allclose(semantic[0], semantic[1])


def test_auxiliary_render_loss_reaches_geometry():
    import random

    from vecgpt.data import collate, sample_scene
    from vecgpt.model import VecGPT
    from vecgpt.render import render_batch
    from vecgpt.train import loss_fn

    scenes = [sample_scene(random.Random(i), 1) for i in range(2)]
    imgs = render_batch(scenes, size=32)
    batch = collate(scenes, hierarchical=False)
    model = VecGPT(
        image_size=32, d=64, n_heads=4, n_layers=2,
        n_seg_heads=1, n_stroke_heads=1, enc_base=8, n_enc_layers=1,
        dynamic_region_masks=False,
    )
    loss, *rest = loss_fn(
        model, imgs, batch, balanced_fields=True,
        render_loss_weight=0.2, render_loss_size=24,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(rest[-1]["render_loss"]) > 0
    assert model.turtle_proj[0].weight.grad is not None
    assert model.semantic_to_vocab.weight.grad is not None
    assert model.global_visual_encoder.proj[0].weight.grad is not None


def test_sparse_render_loss_attracts_disjoint_strokes():
    from vecgpt.render import foreground_render_loss, render_batch

    target = Stroke(
        torch.tensor([0.25, 0.25, 0.2]),
        torch.tensor([[0.2, 0.0, 0.035, 0.2, 0.5, 0.8]]),
    )
    anchor = torch.tensor([0.75, 0.75, 1.2], requires_grad=True)
    segs = torch.tensor(
        [[0.2, 0.0, 0.035, 0.2, 0.5, 0.8]], requires_grad=True
    )
    pred = Stroke(anchor, segs)
    target_img = render_batch([[target]], size=32).detach()
    pred_img = render_batch([[pred]], size=32)
    loss, _ = foreground_render_loss(pred_img, target_img)
    loss.backward()
    # Gradient descent must decrease both coordinates toward (0.25, 0.25).
    assert anchor.grad[0] > 0
    assert anchor.grad[1] > 0
    same, _ = foreground_render_loss(target_img, target_img)
    assert float(same) < 1e-6


def test_numeric_distance_respects_bin_topology():
    import vecgpt.schema as S
    from vecgpt.train import _numeric_distance_loss

    lo, _ = S.RANGE["len"]
    target = lo + 120
    base = torch.full((1, 1, S.VOCAB), -20.0)
    valid = torch.ones((1, 1), dtype=torch.bool)
    targets = torch.tensor([[target]])

    near = base.clone()
    near[0, 0, target + 1] = 20.0
    far = base.clone()
    far[0, 0, lo + 1] = 20.0
    assert _numeric_distance_loss(near, targets, valid) < _numeric_distance_loss(
        far, targets, valid
    )
