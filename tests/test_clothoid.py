import torch

from vecgpt.clothoid import clothoid_end_state, clothoid_points
from vecgpt.scene import arc_step


def test_clothoid_reduces_to_constant_curvature_arc():
    a = torch.tensor([0.2, 0.3, -0.4])
    k = torch.tensor(2.3)
    length = torch.tensor(0.41)
    p = clothoid_points(a, k, length, torch.tensor(0.0), 32)
    s = torch.linspace(0, float(length), 33)
    x, y, _ = arc_step(a[0], a[1], a[2], k, s)
    expected = torch.stack((x, y), -1)
    assert torch.allclose(p, expected, atol=2e-6)


def test_clothoid_end_and_gradients():
    a = torch.tensor([0.1, 0.2, 0.3])
    k = torch.tensor(1.0, requires_grad=True)
    length = torch.tensor(0.4, requires_grad=True)
    dk = torch.tensor(2.0, requires_grad=True)
    end, end_k = clothoid_end_state(a, k, length, dk)
    assert torch.allclose(end[2], torch.tensor(1.1), atol=1e-6)
    assert torch.allclose(end_k, torch.tensor(3.0), atol=1e-6)
    grads = torch.autograd.grad(end[:2].sum(), (k, length, dk))
    assert all(torch.isfinite(g) and g.abs() > 1e-8 for g in grads)
