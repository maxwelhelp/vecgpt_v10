import torch

from vecgpt.clothoid_render import (
    _segment_distance_sq,
    render_clothoid_batch,
)
from vecgpt.render import render_batch
from vecgpt.scene import Stroke


def test_point_to_segment_distance_not_point_distance():
    pixels = torch.tensor([[0.5, 0.25]])
    a = torch.tensor([[[[0.0, 0.0]]]])
    b = torch.tensor([[[[1.0, 0.0]]]])
    distance_sq = _segment_distance_sq(pixels, a, b)
    assert torch.allclose(distance_sq.squeeze(), torch.tensor(0.25**2))


def test_constant_clothoid_renderer_matches_arc_renderer():
    anchor = torch.tensor([[[0.25, 0.35, -0.3]]])
    kappa0 = torch.tensor([[2.0]])
    length = torch.tensor([[0.33]])
    delta = torch.zeros_like(length)
    width = torch.tensor([[0.045]])
    rgba = torch.tensor([[[0.2, 0.4, 0.7, 0.8]]])
    actual = render_clothoid_batch(
        anchor, kappa0, length, delta, width, rgba,
        size=48, curve_samples=32, pixel_chunk=257,
    )
    stroke = Stroke(
        anchor[0, 0],
        torch.tensor([[0.33, 2.0, 0.045, 0.2, 0.4, 0.7, 0.8]]),
    )
    expected = render_batch([[stroke]], size=48, per_seg=32)
    assert torch.allclose(actual, expected, atol=3e-6)


def test_chunking_is_numerically_identical():
    anchor = torch.tensor([[[0.3, 0.4, 0.2]]])
    kappa0 = torch.tensor([[1.2]])
    length = torch.tensor([[0.4]])
    delta = torch.tensor([[3.0]])
    width = torch.tensor([[0.035]])
    rgba = torch.tensor([[[0.1, 0.5, 0.8, 0.9]]])
    one = render_clothoid_batch(
        anchor, kappa0, length, delta, width, rgba,
        size=24, curve_samples=24, distance_softmin_px=0.3,
        pixel_chunk=24 * 24,
    )
    many = render_clothoid_batch(
        anchor, kappa0, length, delta, width, rgba,
        size=24, curve_samples=24, distance_softmin_px=0.3,
        pixel_chunk=37,
    )
    assert torch.equal(one, many)


def test_render_gradients_reach_geometry_and_style():
    anchor = torch.tensor(
        [[[0.25, 0.35, -0.2]]], requires_grad=True
    )
    kappa0 = torch.tensor([[1.0]], requires_grad=True)
    length = torch.tensor([[0.4]], requires_grad=True)
    delta = torch.tensor([[2.5]], requires_grad=True)
    width = torch.tensor([[0.05]], requires_grad=True)
    rgba = torch.tensor(
        [[[0.2, 0.4, 0.7, 0.8]]], requires_grad=True
    )
    image = render_clothoid_batch(
        anchor, kappa0, length, delta, width, rgba,
        size=32, curve_samples=24, distance_softmin_px=0.25,
        pixel_chunk=113,
    )
    y = torch.linspace(0.3, 1.3, 32)[:, None, None]
    x = torch.linspace(0.7, 1.7, 32)[None, :, None]
    objective = (image * x * y).mean()
    gradients = torch.autograd.grad(
        objective, (anchor, kappa0, length, delta, width, rgba)
    )
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(float(g.abs().sum()) > 1e-9 for g in gradients)

