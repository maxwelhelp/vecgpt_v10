import random

import torch

from vecgpt.data import sample_scene
from vecgpt.render import render_batch
from vecgpt.stateful import (
    StatefulStroke,
    legacy_to_stateful,
    pack_stateful_scenes,
    render_packed_stateful,
    stateful_chain_states,
)


def test_legacy_piecewise_arcs_round_trip_through_sparse_kappa_jumps():
    legacy = [sample_scene(random.Random(501 + i), 3) for i in range(4)]
    stateful = [
        [legacy_to_stateful(stroke) for stroke in scene]
        for scene in legacy
    ]
    packed = pack_stateful_scenes(stateful, max_strokes=2, max_segments=24)
    actual = render_packed_stateful(
        packed, size=48, curve_samples=24, pixel_chunk=311
    )
    expected = render_batch(legacy, size=48, per_seg=24)
    assert torch.allclose(actual, expected, atol=5e-6)


def test_sparse_change_masks_are_not_fixed_slots():
    stroke = StatefulStroke(
        anchor=torch.tensor([0.2, 0.3, 0.0]),
        base_kappa=torch.tensor(0.0),
        length=torch.tensor([0.2, 0.2, 0.2]),
        delta_kappa=torch.tensor([1.0, -1.0, 0.0]),
        curvature_jump=torch.tensor([0.0, 0.0, 3.0]),
        base_style=torch.tensor([0.03, 0.1, 0.2, 0.3, 1.0]),
        style_delta=torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.2, 0.0, 0.0, 0.0],
        ]),
    )
    packed = pack_stateful_scenes([[stroke]], 4, 8)
    assert packed.counts[0, 0] == 3
    assert packed.segment_mask.sum() == 3
    assert packed.curvature_change.sum() == 1
    assert packed.style_change.sum() == 1


def test_stateful_chain_gradients_reach_curvature_and_jumps():
    anchor = torch.tensor(
        [[[0.2, 0.3, 0.0, 1.0]]], requires_grad=True
    )
    base = torch.tensor([[0.5]], requires_grad=True)
    segment = torch.tensor(
        [[[[0.2, 2.0], [0.25, -1.0]]]], requires_grad=True
    )
    jumps = torch.tensor(
        [[[0.0, 0.7]]], requires_grad=True
    )
    _, _, ends = stateful_chain_states(anchor, base, segment, jumps)
    objective = ends[..., -1, :].sum()
    gradients = torch.autograd.grad(
        objective, (anchor, base, segment, jumps)
    )
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(float(g.abs().sum()) > 1e-8 for g in gradients)

