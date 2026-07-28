import random

import torch

from vecgpt.continuous import (
    ASTVectorAutoencoder,
    ContinuousVectorAutoencoder,
    _trajectory_states,
    continuous_losses,
    output_to_scenes,
    pack_scenes,
)
from vecgpt.data import sample_scene
from vecgpt.render import render_batch


def test_continuous_shapes_and_hard_decode():
    scenes = [sample_scene(random.Random(100 + i), i % 4) for i in range(4)]
    packed = pack_scenes(scenes, max_strokes=4, max_segments=24)
    model = ContinuousVectorAutoencoder(
        d_model=64, n_heads=4, n_layers=1,
        max_strokes=4, max_segments=24,
    )
    out = model(packed)
    assert out.anchor.shape == (4, 4, 4)
    assert out.segment.shape == (4, 4, 24, 2)
    assert out.base_style.shape == (4, 4, 5)
    loss, terms = continuous_losses(out, packed)
    assert torch.isfinite(loss)
    assert set(("present", "count", "length", "turn")) <= terms.keys()
    decoded = output_to_scenes(out, soft_structure=False)
    assert len(decoded) == 4


def test_soft_structure_render_reaches_topology_logits():
    scene = [sample_scene(random.Random(77), 1)]
    packed = pack_scenes(scene, max_strokes=2, max_segments=4)
    model = ContinuousVectorAutoencoder(
        d_model=32, n_heads=4, n_layers=1,
        max_strokes=2, max_segments=4,
    )
    out = model(packed)
    soft = output_to_scenes(out, soft_structure=True)
    image = render_batch(soft, size=16, per_seg=4)
    loss = (1.0 - image).mean()
    grads = torch.autograd.grad(
        loss, (out.present_logits, out.count_logits), allow_unused=False
    )
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_ast_latent_keeps_dynamic_stroke_and_segment_nodes():
    scenes = [sample_scene(random.Random(301 + i), 3) for i in range(3)]
    packed = pack_scenes(scenes, max_strokes=3, max_segments=24)
    model = ASTVectorAutoencoder(
        d_model=48, n_heads=4, n_layers=1,
        max_strokes=3, max_segments=24,
    )
    latent = model.encoder(packed)
    assert latent.root.shape == (3, 48)
    assert latent.stroke.shape == (3, 3, 48)
    assert latent.frame.shape == (3, 3, 48)
    assert latent.style.shape == (3, 3, 48)
    assert latent.segment.shape == (3, 3, 24, 48)
    assert torch.equal(latent.stroke_mask, packed.stroke_mask)
    out = model.decoder(latent)
    loss, _ = continuous_losses(out, packed)
    loss.backward()
    assert torch.isfinite(loss)


def test_trajectory_state_integrates_a_quarter_circle():
    anchor = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    segment = torch.tensor([[[[torch.pi / 2.0, torch.pi / 2.0]]]])
    xy, heading = _trajectory_states(anchor, segment)
    assert torch.allclose(
        xy[0, 0, 0], torch.tensor([1.0, 1.0]), atol=1e-5
    )
    assert torch.allclose(
        heading[0, 0, 0], torch.tensor([1.0, 0.0]), atol=1e-5
    )
