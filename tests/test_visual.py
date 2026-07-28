import torch

from vecgpt.stateful import (
    StatefulStroke,
    pack_stateful_scenes,
    render_packed_stateful,
)
from vecgpt.stateful_model import (
    output_to_packed,
    stateful_losses,
)
from vecgpt.region_ast import (
    RegionProgram,
    pack_region_programs,
    region_losses,
    region_output_to_packed,
    render_region_programs,
)
from vecgpt.visual import (
    RasterPatchEncoder,
    RasterToRegionAST,
    RasterToStatefulAST,
)


def _stroke():
    return StatefulStroke(
        torch.tensor([0.2, 0.3, 0.1]),
        torch.tensor(0.2),
        torch.tensor([0.25, 0.20]),
        torch.tensor([0.5, -0.2]),
        torch.zeros(2),
        torch.tensor([0.04, 0.2, 0.5, 0.8, 1.0]),
        torch.zeros(2, 5),
    )


def test_patch_encoder_preserves_spatial_token_sequence():
    encoder = RasterPatchEncoder(
        d_model=32, patch_size=4, n_heads=4, n_layers=1
    )
    memory = encoder(torch.rand(2, 32, 32, 3))
    assert memory.shape == (2, 65, 32)


def test_raster_to_ast_program_and_render_gradients():
    target = pack_stateful_scenes([[_stroke()], [_stroke()]], 2, 4)
    image = render_packed_stateful(
        target, size=32, curve_samples=8, pixel_chunk=128
    )
    model = RasterToStatefulAST(
        d_model=32, n_heads=4, encoder_layers=1, decoder_layers=1,
        patch_size=4, max_strokes=2, max_segments=4,
    )
    out = model(image)
    direct, _ = stateful_losses(out, target)
    predicted = output_to_packed(out, soft_structure=True)
    reconstructed = render_packed_stateful(
        predicted, size=16, curve_samples=6, pixel_chunk=64,
        distance_softmin_px=1.0,
    )
    loss = direct + reconstructed.square().mean()
    loss.backward()
    assert model.visual.patch[0].weight.grad.abs().sum() > 0
    assert model.stroke_query.grad.abs().sum() > 0
    assert model.ast_decoder.segment_head.weight.grad.abs().sum() > 0


def test_raster_to_region_ast_reaches_hierarchy_and_local_geometry():
    scenes = [[
        RegionProgram(torch.tensor([0.4, 0.5, 0.2, -1.2]), [_stroke()])
    ]]
    target = pack_region_programs(scenes, 2, 2, 4)
    image = render_region_programs(
        target, size=32, curve_samples=8, pixel_chunk=128
    )
    model = RasterToRegionAST(
        d_model=32, n_heads=4, encoder_layers=1, decoder_layers=1,
        max_regions=2, max_strokes=2, max_segments=4,
    )
    out = model(image)
    direct, _ = region_losses(out, target)
    predicted = region_output_to_packed(
        out, target, soft_structure=True
    )
    reconstructed = render_region_programs(
        predicted, size=16, curve_samples=6, pixel_chunk=64,
        distance_softmin_px=1.0,
    )
    (direct + reconstructed.square().mean()).backward()
    assert model.region_query.grad.abs().sum() > 0
    assert model.stroke_query.grad.abs().sum() > 0
    assert model.local_decoder.segment_head.weight.grad.abs().sum() > 0


def test_region_crop_is_spatial_and_differentiable():
    model = RasterToRegionAST(
        d_model=32, n_heads=4, encoder_layers=1, decoder_layers=1,
        max_regions=2, max_strokes=1, max_segments=1,
    )
    image = torch.ones(1, 24, 24, 3, requires_grad=True)
    image.data[:, 4:9, 3:8] = 0.0
    frame = torch.tensor([[
        [5 / 24, 6 / 24, 0.0, 1.0, -1.4],
        [19 / 24, 18 / 24, 0.0, 1.0, -1.4],
    ]])
    crop = model._region_crops(image, frame, size=12)
    assert crop[0, 3].mean() > crop[1, 3].mean()
    crop.sum().backward()
    assert image.grad is not None
