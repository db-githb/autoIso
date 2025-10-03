#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataclasses import dataclass
from pathlib import Path
from autoIsoMASK.mask_main import MaskProcessor
from autoIsoPY.main.utils_main import (
    write_ply,
    load_config, 
    render_loop,
    )

from autoIsoPY.main.utils_autoIso import (
    statcull_anisotropic, 
    modify_model,
)

from autoIsoUTILS.rich_utils import CONSOLE

import time
import shutil
from typing import Optional
from contextlib import contextmanager

def _pct(n, d):  # safe percent string
    return f"{(n/d):.1%}" if d else "n/a"

@contextmanager
def step(console, title, emoji=":arrow_forward:"):
    t0 = time.perf_counter()
    console.log(f"{emoji} [bold]{title}[/bold]")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        console.log(f":white_check_mark: Done [dim]({dt:.2f}s)[/dim]")

def log_totals(console, before, after, noun="splats"):
    culled = before - after
    console.log(
        f"• Removed {culled}/{before} {noun} "
        f"([bold]{_pct(culled, before)}[/bold]) → total: [bold]{after}[/bold]"
    )


@dataclass
class BaseIso:
    """Base class for rendering."""
    load_config: Path
    """Path to model file."""
    output_dir: Path = Path("culled_models/output.ply")
    """Path to output model file."""


@dataclass
class autoIso(BaseIso):
    """Cull using all images in the dataset."""
    stat_thresh: Optional[float] = None 
    
    # main/driver function
    def run_iso(self):

        # load model
        config, pipeline = load_config(
            self.load_config,
            test_mode="inference",
        )
        config.datamanager.dataparser.train_split_fraction = 1.0 # put all images in train split
        config.datamanager.dataparser.downscale_factor = 1
        # Phase 1 — statistical cull
        starting_total = pipeline.model.means.shape[0]
        with step(CONSOLE, "Phase 1 — Statistical cull", emoji=":broom:"):
            cull_mask = statcull_anisotropic(pipeline)
            keep = ~cull_mask
            pipeline.model = modify_model(pipeline.model, keep)
            statcull_total = pipeline.model.means.shape[0]
            log_totals(CONSOLE, starting_total, statcull_total)

        # Phase 2 — render frames
        with step(CONSOLE, "Phase 2 — Rendering frames for mask extraction", emoji=":film_frames:"):
            render_dir = render_loop(config, pipeline)
            CONSOLE.log(":tada: Render complete")

        # Phase 3 — car mask (keep car)
        with step(CONSOLE, "Phase 3 — Apply car mask (keep car)", emoji=":car:"):
            mp = MaskProcessor(Path(render_dir), "car")
            mask_dir = mp.run_mask_processing(0.9, 0.25)
            shutil.rmtree(render_dir)  # delete renders
            CONSOLE.print(f"• Saved masks to: {mask_dir}")