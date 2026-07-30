"""Adapters from real SVG stroke programs to clothoid parameter tensors."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from svgpathtools import svg2paths2


def _color(value: str | None) -> tuple[float, float, float]:
    if not value or value.lower() == "none":
        return 0.08, 0.08, 0.08
    value = value.strip().lower()
    names = {"black": (0.0, 0.0, 0.0), "white": (1.0, 1.0, 1.0), "red": (1.0, 0.0, 0.0), "blue": (0.0, 0.0, 1.0), "green": (0.0, 1.0, 0.0)}
    if value in names:
        return names[value]
    if value.startswith("#"):
        h = value[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) >= 6:
            try:
                return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            except ValueError:
                pass
    return 0.08, 0.08, 0.08


def _fit_clothoid(points: np.ndarray, width: float, rgb: tuple[float, float, float]) -> np.ndarray | None:
    if len(points) < 3:
        return None
    d = np.diff(points, axis=0)
    seg = np.linalg.norm(d, axis=1)
    keep = np.r_[True, seg > 1e-6]
    points = points[keep]
    d = np.diff(points, axis=0)
    seg = np.linalg.norm(d, axis=1)
    length = float(seg.sum())
    # Very short normalized paths make alpha=delta_kappa/L explode and cause
    # unstable clothoid derivatives. They are visually sub-pixel at training
    # resolution, so discard them instead of feeding a singular segment.
    if length < 5e-3:
        return None
    theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    ds = np.maximum(seg, 1e-5)
    # A least-squares line for curvature(s) is more stable than using only
    # the first/last turn, especially on noisy exported SVG paths.
    turn = np.diff(theta)
    curv = turn / ds[1:] if len(turn) else np.zeros(1)
    k0 = float(np.median(curv[: max(1, len(curv) // 4)])) if len(curv) else 0.0
    k1 = float(np.median(curv[-max(1, len(curv) // 4):])) if len(curv) else k0
    k0 = float(np.clip(k0, -12.0, 12.0))
    dk = float(np.clip(k1 - k0, -24.0, 24.0))
    x, y = points[0]
    return np.array([x, y, theta[0], np.clip(length, 5e-3, 1.2), k0, dk, np.clip(width, .003, .05), *rgb], dtype=np.float32)


class SVGClothoidDataset(Dataset):
    """Load SVG paths and fit one clothoid segment per path.

    This is intentionally a first real-data adapter.  Later we will split
    long paths into multiple clothoid segments instead of fitting one segment
    to an entire path.
    """

    def __init__(self, root: str | Path, max_strokes: int = 64, path_points: int = 48, segment_points: int = 16, limit: int | None = None, split_paths: bool = False):
        self.files = sorted(Path(root).rglob("*.svg"))
        if limit is not None:
            self.files = self.files[:limit]
        self.max_strokes = max_strokes
        self.path_points = path_points
        self.segment_points = segment_points
        self.split_paths = split_paths

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        paths, attrs, meta = svg2paths2(str(path))
        vb = meta.get("viewBox")
        if vb:
            vals = [float(x) for x in vb.replace(",", " ").split()]
            vx, vy, vw, vh = vals if len(vals) == 4 else (0., 0., 1., 1.)
        else:
            vx = vy = 0.0
            all_end = [complex(seg.end) for p in paths for seg in p]
            vw = max([z.real for z in all_end] + [1.0])
            vh = max([z.imag for z in all_end] + [1.0])
        strokes = []
        for p, attr in zip(paths, attrs):
            # Do not call svgpathtools ``Path.length()`` here.  It performs a
            # numerical quadrature for every path and becomes the dominant CPU
            # cost (and emits IntegrationWarning on exported SVGs).  Sampling
            # below is enough to detect degenerate paths.
            if len(p) == 0:
                continue
            try:
                pts = np.array([p.point(t) for t in np.linspace(0, 1, self.path_points)])
            except (RuntimeError, ValueError, ZeroDivisionError):
                # Some public SVG corpora contain zero-length or malformed
                # subpaths.  One bad path must not abort a whole cache build.
                continue
            xy = np.stack((pts.real, vh - pts.imag), -1).astype(np.float32)
            if not np.isfinite(xy).all() or np.ptp(xy, axis=0).max() < 1e-6:
                continue
            xy[:, 0] = (xy[:, 0] - vx) / max(vw, 1e-6)
            xy[:, 1] = (xy[:, 1] - (vh - (vy + vh))) / max(vh, 1e-6)
            # Normalize to the actual viewBox convention [0,1].
            xy[:, 1] = 1.0 - (pts.imag.astype(np.float32) - vy) / max(vh, 1e-6)
            raw_w = attr.get("stroke-width", "") if attr else ""
            try:
                width = float(raw_w) / max(vw, vh)
            except (ValueError, TypeError):
                width = .012
            rgb = _color(attr.get("stroke") if attr else None)
            # Keep one source SVG path as one training stroke by default.  The
            # previous implementation always split every path into four
            # pieces and then silently truncated to max_strokes; that created
            # a fake cardinality distribution (many samples exactly at 64).
            # Splitting remains available explicitly for a later experiment.
            if self.split_paths:
                stride = max(2, self.segment_points - 4)
                starts = range(0, max(1, len(xy) - 2), stride)
            else:
                starts = (0,)
            for start in starts:
                part = xy[start : start + self.segment_points] if self.split_paths else xy
                if len(part) < 3:
                    continue
                fit = _fit_clothoid(part, width, rgb)
                if fit is not None:
                    strokes.append(fit)
        # Do not silently change a program's topology.  The training command
        # will report overflow; callers can raise max_strokes or filter those
        # examples explicitly.
        strokes = strokes[: self.max_strokes]
        params = np.zeros((self.max_strokes, 10), dtype=np.float32)
        valid = np.zeros((self.max_strokes,), dtype=np.float32)
        if strokes:
            n = len(strokes)
            params[:n] = np.stack(strokes)
            valid[:n] = 1.0
        return torch.from_numpy(params), torch.from_numpy(valid)
