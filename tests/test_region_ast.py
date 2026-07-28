import torch

from vecgpt.region_ast import (
    RegionASTAutoencoder,
    RegionProgram,
    pack_region_programs,
    region_losses,
    region_output_to_packed,
    region_hungarian_indices,
    region_layout_loss,
    regions_to_global,
    render_region_programs,
)


def test_region_hungarian_matches_permuted_queries():
    scenes = [[
        RegionProgram(torch.tensor([0.2, 0.3, 0.0, -1.0]), [_line()]),
        RegionProgram(torch.tensor([0.8, 0.7, 0.0, -1.0]), [_line()]),
    ]]
    target = pack_region_programs(scenes, 3, 2, 3)
    model = RegionASTAutoencoder(
        d_model=32, n_heads=4, n_layers=1,
        max_regions=3, max_strokes=2, max_segments=3,
    )
    out = model(target)
    out.frame = torch.cat((
        target.frame[:, 1:2], target.frame[:, 0:1], out.frame[:, 2:3]
    ), 1)
    out.region_present_logits = torch.tensor([[10.0, 10.0, -10.0]])
    assignment = region_hungarian_indices(out, target)
    rows, cols = assignment[0]
    assert rows.tolist() == [0, 1]
    assert cols.tolist() == [1, 0]
    loss, _ = region_layout_loss(out, target, assignments=assignment)
    assert float(loss) < 0.01
from vecgpt.stateful import StatefulStroke


def _line():
    return StatefulStroke(
        torch.tensor([0.1, 0.5, 0.0]),
        torch.tensor(0.0),
        torch.tensor([0.8]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.08, 0.1, 0.4, 0.8, 1.0]),
        torch.zeros(1, 5),
    )


def test_sim2_region_scales_local_geometry_and_style():
    scene = [[
        RegionProgram(torch.tensor([0.3, 0.4, 0.0, torch.log(torch.tensor(0.2))]), [_line()])
    ]]
    packed = pack_region_programs(scene, 2, 2, 3)
    global_packed = regions_to_global(packed)
    assert torch.allclose(
        global_packed.anchor[0, 0, :2], torch.tensor([0.22, 0.4])
    )
    assert torch.allclose(global_packed.segment[0, 0, 0, 0], torch.tensor(0.16))
    assert torch.allclose(global_packed.base_style[0, 0, 0], torch.tensor(0.016))


def test_region_ast_loss_render_and_topology_gradients():
    scenes = [
        [RegionProgram(torch.tensor([0.3, 0.4, 0.2, -1.2]), [_line()])],
        [
            RegionProgram(torch.tensor([0.2, 0.3, -0.3, -1.5]), [_line()]),
            RegionProgram(torch.tensor([0.7, 0.6, 0.5, -1.0]), [_line()]),
        ],
    ]
    packed = pack_region_programs(scenes, 3, 2, 3)
    model = RegionASTAutoencoder(
        d_model=32, n_heads=4, n_layers=1,
        max_regions=3, max_strokes=2, max_segments=3,
    )
    out = model(packed)
    direct, terms = region_losses(out, packed)
    soft = region_output_to_packed(out, packed, soft_structure=True)
    image = render_region_programs(
        soft, size=16, curve_samples=6, pixel_chunk=64,
        distance_softmin_px=1.0,
    )
    loss = direct + image.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in terms.values())
    assert model.present.weight.grad.abs().sum() > 0
    assert model.frame_decode[-1].weight.grad.abs().sum() > 0
    assert model.local_decoder.segment_head.weight.grad.abs().sum() > 0


def test_hard_region_topology_is_generated_prefix():
    packed = pack_region_programs(
        [[RegionProgram(torch.tensor([0.5, 0.5, 0.0, -1.0]), [_line()])]],
        3, 2, 3,
    )
    model = RegionASTAutoencoder(
        d_model=32, n_heads=4, n_layers=1,
        max_regions=3, max_strokes=2, max_segments=3,
    )
    out = model(packed)
    with torch.no_grad():
        out.region_present_logits[0] = torch.tensor([10.0, -10.0, 10.0])
    decoded = region_output_to_packed(out, packed, soft_structure=False)
    assert decoded.region_mask.tolist() == [[True, False, False]]
