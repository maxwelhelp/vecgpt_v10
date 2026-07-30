#!/usr/bin/env python
"""Run the hybrid smoke test from inside this architecture folder.

Usage from ``hybrid_stroke_diffusion/``:

    PYTHONPATH=. python -u scripts/run_hybrid_stroke_diffusion_smoke.py --device cuda
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file lives two levels below the repository root.  Keep the new branch
# self-contained while reusing the implementation in the root scripts folder.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_hybrid_stroke_diffusion_smoke import main


if __name__ == "__main__":
    main()
