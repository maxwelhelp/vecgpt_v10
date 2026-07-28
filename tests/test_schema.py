import random

import torch

import vecgpt.schema as S
from vecgpt.data import sample_scene
from vecgpt.grammar import OOD, sample_tree
from vecgpt.regions import build_regions, ellipse_of, from_local, iter_regions, to_local
from vecgpt.render import image_iou, render_one
from vecgpt.tokenizer import Walker, decode, encode


def _scenes(n=12):
    r = random.Random(0)
    out = [sample_scene(r, st) for st in (1, 2, 3, 4) for _ in range(n // 4 + 1)]
    out += [sample_tree(r) for _ in range(n // 2)]
    out += [sample_tree(r, v) for v in OOD]
    return [s for s in out if s]


def test_quantisers_roundtrip():
    for name, q in S.QUANTS.items():
        for t in (0.0, 0.13, 0.5, 0.87, 1.0):
            v = q.lo + t * (q.hi - q.lo) if q.kind != "signed" else (2 * t - 1) * q.hi
            i = q.q(v)
            assert 0 <= i < q.n, (name, v, i)
            assert q.q(q.d(i)) == i, (name, i)


def test_signed_quant_has_exact_zero():
    for name, q in S.QUANTS.items():
        if q.kind == "signed":
            assert q.d(q.q(0.0)) == 0.0, name


def test_grammar_is_reachable_and_closed():
    """Every state must be reachable from TOP and every edge must land in a
    real state - the kind of thing that was previously only true because I
    kept five copies of the transitions in sync by hand."""
    seen, stack = {"TOP"}, ["TOP"]
    while stack:
        st = S.GRAMMAR[stack.pop()]
        for e in st.edges:
            assert e.to is None or e.to in S.GRAMMAR, (st.name, e.emit, e.to)
            if e.to and e.to not in seen:
                seen.add(e.to)
                stack.append(e.to)
    assert seen == set(S.GRAMMAR), set(S.GRAMMAR) - seen


def test_mask_matches_step():
    """The logit mask and the transition function must agree token by
    token. They are both derived from GRAMMAR now, so this is a guard
    against the derivation, not against hand-copied tables."""
    m = S.build_state_mask()
    for name in S.GRAMMAR:
        for tok in range(S.VOCAB):
            legal = bool(m[S.STATE_ID[name], tok])
            try:
                S.step(name, tok)
                ok = True
            except ValueError:
                ok = False
            assert legal == ok, (name, tok)


def test_walker_labels_match_encode():
    for sc in _scenes():
        e = encode(sc)
        w = Walker()
        got = [w.advance(int(t)) for t in e.tokens]
        want = list(zip(e.states.tolist(), e.seg_idx.tolist(),
                        e.stroke_idx.tolist(), e.region_idx.tolist(),
                        e.parent_region_idx.tolist(), e.region_depth.tolist()))
        assert got == want


def test_local_frame_roundtrip():
    rng = random.Random(2)
    for _ in range(50):
        ell = (rng.random(), rng.random(), rng.uniform(0.05, 0.4),
               rng.uniform(0.02, 0.3), rng.uniform(-1.5, 1.5))
        x, y = rng.random(), rng.random()
        u, v = to_local(x, y, ell)
        bx, by = from_local(u, v, ell)
        assert abs(bx - x) < 1e-5 and abs(by - y) < 1e-5


def test_regions_are_deterministic_and_nested():
    rng = random.Random(3)
    for _ in range(15):
        sc = sample_tree(rng)
        a, b = build_regions(sc), build_regions(sc)
        af, bf = list(iter_regions(a)), list(iter_regions(b))
        assert [r.ellipse for r in af] == [r.ellipse for r in bf]
        owned = [st for r in af for st in r.strokes]
        assert len(owned) == len(sc)
        assert len({id(st) for st in owned}) == len(sc)
        assert a[0].depth == 0
        for r in af:
            assert all(c.depth == r.depth + 1 for c in r.children)


def test_token_roundtrip_beats_flat_ceiling():
    for sc in _scenes(16):
        back = decode(encode(sc).tokens)
        iou = float(image_iou(render_one(back, size=64)[None], render_one(sc, size=64)[None])[0])
        assert iou > 0.55, iou


def test_flat_baseline_has_no_region_side_channel():
    rng = random.Random(11)
    sc = sample_scene(rng, 4)
    e = encode(sc, hierarchical=False)
    assert int(e.region_idx.max()) == 0
    assert int(e.region_depth.max()) == 0
    assert not any(S.field_of_token(int(t)) in {"rx", "ry", "rt"}
                   for t in e.tokens)
    assert len(decode(e.tokens)) == len(sc)


def test_grammar_ood_changes_structure():
    r = random.Random(1)
    base = sum(len(sample_tree(random.Random(i))) for i in range(20)) / 20
    wide = sum(len(sample_tree(random.Random(i), "wider")) for i in range(20)) / 20
    deep = sum(len(sample_tree(random.Random(i), "deeper")) for i in range(20)) / 20
    assert wide > base * 1.3 and deep > base
