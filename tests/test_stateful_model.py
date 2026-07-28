import math

import torch

from vecgpt.stateful import (
    StatefulStroke,
    pack_stateful_scenes,
    render_packed_stateful,
)
from vecgpt.stateful_model import (
    StatefulASTAutoencoder,
    output_to_packed,
    stateful_losses,
)


def _stroke(n=3):
    return StatefulStroke(
        anchor=torch.tensor([0.25, 0.35, 0.2]),
        base_kappa=torch.tensor(0.4),
        length=torch.tensor([0.16, 0.13, 0.11])[:n],
        delta_kappa=torch.tensor([0.7, -0.3, 0.2])[:n],
        curvature_jump=torch.tensor([0.0, 0.8, 0.0])[:n],
        base_style=torch.tensor([0.035, 0.2, 0.4, 0.7, 0.9]),
        style_delta=torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.004, 0.1, -0.1, 0.0, -0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])[:n],
    )


def test_stateful_ast_forward_loss_render_and_backward():
    scenes = [[_stroke(3)], [_stroke(2), _stroke(1)]]
    packed = pack_stateful_scenes(scenes, 3, 4)
    model = StatefulASTAutoencoder(
        d_model=32, n_heads=4, n_layers=1,
        max_strokes=3, max_segments=4,
    )
    out = model(packed)
    assert out.anchor.shape == (2, 3, 4)
    assert out.segment.shape == (2, 3, 4, 2)
    direct, terms = stateful_losses(out, packed)
    soft = output_to_packed(out, soft_structure=True)
    image = render_packed_stateful(
        soft, size=16, curve_samples=6, pixel_chunk=64,
        distance_softmin_px=1.0,
    )
    loss = direct + image.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in terms.values())
    assert model.decoder.present.weight.grad.abs().sum() > 0
    assert model.decoder.count.weight.grad.abs().sum() > 0
    assert model.decoder.segment_head.weight.grad.abs().sum() > 0


def test_soft_structure_alpha_tracks_present_and_count_probabilities():
    packed = pack_stateful_scenes([[_stroke(3)]], 2, 4)
    model = StatefulASTAutoencoder(
        d_model=32, n_heads=4, n_layers=1,
        max_strokes=2, max_segments=4,
    )
    out = model(packed)
    with torch.no_grad():
        out.present_logits.fill_(-20.0)
        out.count_logits.fill_(-20.0)
        out.count_logits[..., 0] = 20.0
        out.base_style[..., 4] = 1.0
        out.style_delta.zero_()
        out.style_change_logits.fill_(-20.0)
    soft = output_to_packed(out, soft_structure=True)
    styles = soft.base_style[:, :, None] + soft.style_delta.cumsum(-2)
    assert styles[..., 4].max() < 1e-7

    with torch.no_grad():
        out.present_logits.fill_(20.0)
    soft = output_to_packed(out, soft_structure=True)
    styles = soft.base_style[:, :, None] + soft.style_delta.cumsum(-2)
    assert styles[0, 0, 0, 4] > 0.999
    assert styles[0, 0, 1:, 4].max() < 1e-7


def test_hard_structure_is_prefix_and_predicted_count():
    packed = pack_stateful_scenes([[_stroke(2)]], 3, 4)
    model = StatefulASTAutoencoder(
        d_model=32, n_heads=4, n_layers=1,
        max_strokes=3, max_segments=4,
    )
    out = model(packed)
    with torch.no_grad():
        out.present_logits[0] = torch.tensor([10.0, -10.0, 10.0])
        out.count_logits.fill_(-10.0)
        out.count_logits[0, 0, 2] = 10.0
    hard = output_to_packed(out, soft_structure=False)
    assert hard.stroke_mask.tolist() == [[True, False, False]]
    assert hard.segment_mask[0, 0].tolist() == [True, True, True, False]
