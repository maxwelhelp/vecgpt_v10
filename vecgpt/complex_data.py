"""Procedural complex vector programs for hierarchy/concept gates.

Names in this module are diagnostics for evaluation only.  The model sees
only REGION frames and numeric local Stroke programs.  Repeated motifs are
deliberately reused under different object layouts, colours and Sim(2)
transforms so latent retrieval and part swapping are measurable.
"""

from __future__ import annotations

import math
import random

import torch

from vecgpt.region_ast import RegionProgram
from vecgpt.stateful import StatefulStroke


def _stroke(anchor, length, kappa, width, rgb, delta_kappa=0.0):
    n = max(1, math.ceil(length / 0.48))
    return StatefulStroke(
        torch.tensor(anchor, dtype=torch.float32),
        torch.tensor(kappa, dtype=torch.float32),
        torch.full((n,), length / n, dtype=torch.float32),
        torch.full((n,), delta_kappa / n, dtype=torch.float32),
        torch.zeros(n),
        torch.tensor([width, *rgb, 1.0], dtype=torch.float32),
        torch.zeros(n, 5),
    )


def _line(a, b, width, rgb):
    dx, dy = b[0] - a[0], b[1] - a[1]
    return _stroke(
        [a[0], a[1], math.atan2(dy, dx)],
        math.hypot(dx, dy), 0.0, width, rgb,
    )


def _circle(radius, width, rgb):
    return _stroke(
        [0.5 + radius, 0.5, math.pi / 2],
        2 * math.pi * radius, 1.0 / radius, width, rgb,
    )


def motif(kind: str, rgb, rng: random.Random) -> list[StatefulStroke]:
    width = rng.uniform(0.045, 0.075)
    if kind in ("head", "joint", "wheel", "crown"):
        radius = {
            "head": 0.38, "joint": 0.30,
            "wheel": 0.34, "crown": 0.42,
        }[kind]
        return [_circle(radius, width, rgb)]
    if kind == "eye":
        dark = tuple(max(0.02, c * 0.18) for c in rgb)
        return [
            _circle(0.34, width, rgb),
            _circle(0.12, width * 1.2, dark),
        ]
    if kind in ("limb", "branch", "whisker"):
        return [_line((0.08, 0.5), (0.92, 0.5), width, rgb)]
    if kind == "tail":
        return [
            _stroke(
                [0.08, 0.62, -0.35], 0.90, -1.4, width, rgb,
                delta_kappa=3.2,
            )
        ]
    if kind in ("mouth", "smile"):
        return [_stroke([0.16, 0.42, -0.35], 0.72, 1.0, width, rgb)]
    if kind in ("torso", "body", "window", "door"):
        inset = 0.10 if kind in ("torso", "body") else 0.16
        return [
            _line((inset, inset), (1 - inset, inset), width, rgb),
            _line((1 - inset, inset), (1 - inset, 1 - inset), width, rgb),
            _line((1 - inset, 1 - inset), (inset, 1 - inset), width, rgb),
            _line((inset, 1 - inset), (inset, inset), width, rgb),
        ]
    if kind in ("ear", "roof"):
        return [
            _line((0.08, 0.84), (0.5, 0.08), width, rgb),
            _line((0.5, 0.08), (0.92, 0.84), width, rgb),
            _line((0.92, 0.84), (0.08, 0.84), width, rgb),
        ]
    if kind == "leaf":
        return [
            _stroke([0.08, 0.5, -0.65], 0.72, 2.0, width, rgb),
            _stroke([0.92, 0.5, math.pi - 0.65], 0.72, 2.0, width, rgb),
        ]
    raise ValueError(f"unknown motif {kind}")


OBJECTS = {
    "person": [
        ("head", 0.0, -0.27, 0.0, 0.25),
        ("eye", -0.065, -0.30, 0.0, 0.055),
        ("eye", 0.065, -0.30, 0.0, 0.055),
        ("smile", 0.0, -0.22, 0.0, 0.09),
        ("torso", 0.0, 0.02, 0.0, 0.27),
        ("limb", -0.25, -0.02, -2.55, 0.24),
        ("limb", 0.25, -0.02, -0.59, 0.24),
        ("limb", -0.12, 0.32, 1.82, 0.28),
        ("limb", 0.12, 0.32, 1.32, 0.28),
        ("joint", -0.36, 0.12, 0.0, 0.045),
        ("joint", 0.36, 0.12, 0.0, 0.045),
    ],
    "cat": [
        ("head", -0.14, -0.17, 0.0, 0.24),
        ("ear", -0.25, -0.37, -0.18, 0.12),
        ("ear", -0.03, -0.38, 0.18, 0.12),
        ("eye", -0.20, -0.18, 0.0, 0.052),
        ("eye", -0.08, -0.18, 0.0, 0.052),
        ("mouth", -0.14, -0.10, 0.0, 0.08),
        ("body", 0.13, 0.12, 0.0, 0.31),
        ("limb", -0.02, 0.34, 1.55, 0.20),
        ("limb", 0.22, 0.35, 1.55, 0.20),
        ("tail", 0.37, 0.02, -0.45, 0.28),
        ("whisker", -0.34, -0.10, 3.0, 0.15),
        ("whisker", 0.04, -0.10, 0.14, 0.15),
    ],
    "house": [
        ("body", 0.0, 0.10, 0.0, 0.42),
        ("roof", 0.0, -0.25, 0.0, 0.48),
        ("door", 0.0, 0.22, 0.0, 0.16),
        ("window", -0.22, 0.06, 0.0, 0.13),
        ("window", 0.22, 0.06, 0.0, 0.13),
        ("wheel", -0.22, 0.06, 0.0, 0.04),
        ("wheel", 0.22, 0.06, 0.0, 0.04),
    ],
    "car": [
        ("body", 0.0, 0.05, 0.0, 0.43),
        ("roof", 0.0, -0.18, 0.0, 0.27),
        ("window", -0.10, -0.13, 0.0, 0.11),
        ("window", 0.10, -0.13, 0.0, 0.11),
        ("wheel", -0.24, 0.26, 0.0, 0.11),
        ("wheel", 0.24, 0.26, 0.0, 0.11),
    ],
    "tree": [
        ("torso", 0.0, 0.18, 0.0, 0.16),
        ("crown", 0.0, -0.17, 0.0, 0.34),
        ("crown", -0.20, -0.08, 0.0, 0.22),
        ("crown", 0.20, -0.08, 0.0, 0.22),
        ("branch", -0.12, 0.02, -2.3, 0.18),
        ("branch", 0.12, 0.00, -0.85, 0.18),
        ("leaf", -0.27, -0.04, -0.3, 0.10),
        ("leaf", 0.27, -0.05, 0.3, 0.10),
    ],
}


def _transform_part(cx, cy, theta, overall, part):
    kind, ox, oy, local_theta, local_scale = part
    cs, sn = math.cos(theta), math.sin(theta)
    tx = cx + overall * (cs * ox - sn * oy)
    ty = cy + overall * (sn * ox + cs * oy)
    return kind, tx, ty, theta + local_theta, overall * local_scale


COARSE_PARTS = {
    "person": {"head", "torso", "limb"},
    "cat": {"head", "ear", "body", "limb", "tail"},
    "house": {"body", "roof", "door", "window"},
    "car": {"body", "roof", "wheel"},
    "tree": {"torso", "crown", "branch"},
}


def sample_complex_scene(
    rng: random.Random, detail_level: int = 1,
) -> tuple[list[RegionProgram], str]:
    object_kind = rng.choice(tuple(OBJECTS))
    theta = rng.uniform(-0.22, 0.22)
    overall = rng.uniform(0.72, 0.92)
    cx, cy = rng.uniform(0.43, 0.57), rng.uniform(0.43, 0.57)
    palette = (
        rng.uniform(0.08, 0.85),
        rng.uniform(0.08, 0.85),
        rng.uniform(0.08, 0.85),
    )
    regions = []
    parts = OBJECTS[object_kind]
    if detail_level <= 0:
        parts = [
            part for part in parts
            if part[0] in COARSE_PARTS[object_kind]
        ]
    for part in parts:
        kind, tx, ty, angle, scale = _transform_part(
            cx, cy, theta, overall, part
        )
        # Small style perturbations prevent exact template hashing.
        rgb = tuple(
            min(0.95, max(0.03, c + rng.uniform(-0.08, 0.08)))
            for c in palette
        )
        local_strokes = motif(kind, rgb, rng)
        local_strokes.sort(key=lambda stroke: (
            round(float(stroke.anchor[1]) * 256),
            round(float(stroke.anchor[0]) * 256),
            round(float(stroke.base_style[0]) * 1024),
        ))
        regions.append(RegionProgram(
            torch.tensor(
                [tx, ty, angle, math.log(scale)], dtype=torch.float32
            ),
            local_strokes,
            diagnostic_kind=kind,
        ))
    # Preserve the canonical source-program traversal. Sorting transformed
    # world coordinates is discontinuous: a tiny object rotation swaps the
    # y-order of symmetric left/right parts and gives one decoder position
    # contradictory targets. Non-canonical external SVG needs matching, not a
    # fragile global-coordinate sort.
    return regions, object_kind


def make_complex_scenes(seed: int, n: int, detail_level: int = 1):
    rng = random.Random(seed)
    pairs = [
        sample_complex_scene(rng, detail_level=detail_level)
        for _ in range(n)
    ]
    return [x[0] for x in pairs], [x[1] for x in pairs]
