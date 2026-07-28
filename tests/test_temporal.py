import torch

from vecgpt.scene import Stroke
from vecgpt.temporal import (
    LocalVectorCache,
    RegionState,
    VectorFrame,
    apply_delta,
    diff_frames,
)


def stroke(x=0.1):
    return Stroke(
        torch.tensor([x, 0.2, 0.0]),
        torch.tensor([[0.2, 0.0, 0.03, 0.1, 0.2, 0.3]]),
    )


def frame(parent_x=0.0, child_x=0.1):
    return VectorFrame(
        {
            1: RegionState(1, None, torch.tensor([parent_x, 0.0, 0.0]), [], [2]),
            2: RegionState(2, 1, torch.tensor([0.0, 0.0, 0.0]), [stroke(child_x)], []),
        },
        [1],
    )


def test_delta_roundtrip_and_implicit_keep():
    a = frame()
    b = frame(child_x=0.3)
    b.time = 0.5
    delta = diff_frames(a, b)
    assert set(delta.upsert) == {2}
    got = apply_delta(a, delta)
    assert torch.equal(got.regions[2].strokes[0].anchor,
                       b.regions[2].strokes[0].anchor)
    assert got.regions[1] is a.regions[1]


def test_parent_motion_reuses_compiled_child_vector_geometry():
    cache = LocalVectorCache()
    compiled = []

    def compile_local(region):
        compiled.append(region.region_id)
        return f"path-{region.region_id}-{len(compiled)}"

    _, rebuilt0 = cache.sync(frame(), compile_local)
    payload0, rebuilt1 = cache.sync(frame(parent_x=0.4), compile_local)
    assert rebuilt0 == {1, 2}
    assert rebuilt1 == set()
    assert payload0[2].startswith("path-2-")

