"""Play-band estimation.

The camera sits on the sideline, low and far back, so the flag pitch occupies a
thin horizontal band -- roughly 11% of frame height -- while the bottom half of
every frame is empty foreground grass. The band's position differs per clip
(native y 376-564 on RenegadesVsAnts/GH010007 versus 504-604 on
SpatzenVsAnts/GH020002), so it must be measured per clip rather than configured.

Accumulated frame-difference activity finds it cleanly, and this is the one
place the cheap 480p proxy still earns its keep: we only need to know *which
rows* matter, which survives heavy downscaling. Detection then runs on the
native-resolution crop of that band.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AnalysisConfig
from .frames import Crop, iter_frames


@dataclass(frozen=True)
class Band:
    """The vertical span of the playing area, in native source pixels."""

    top: int
    bottom: int
    peak: int
    #: Column activity profile, kept for diagnostics and the review overlay.
    column_profile: np.ndarray

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def to_crop(self, source_width: int, source_height: int, padding: int) -> Crop:
        """Native-resolution inference strip covering the band."""
        top = max(0, self.top - padding)
        bottom = min(source_height, self.bottom + padding)
        # Detector strides are powers of two; keep the height a multiple of 32
        # so Ultralytics does not silently letterbox and shift coordinates.
        height = max(32, ((bottom - top) // 32) * 32)
        return Crop(x=0, y=top, width=source_width, height=height)


def estimate_band(
    path: Path,
    source_width: int,
    source_height: int,
    cfg: AnalysisConfig,
    *,
    threshold: int = 12,
    activity_fraction: float = 0.45,
) -> tuple[Band, np.ndarray]:
    """Locate the play band from accumulated motion.

    Returns the band and the full accumulated-activity image, which the review
    page renders as a mask overlay so the estimate is inspectable rather than
    implicit.
    """
    width, height = cfg.proxy_width, cfg.proxy_height
    accumulator = np.zeros((height, width), dtype=np.float32)
    previous: np.ndarray | None = None

    for frame in iter_frames(
        path,
        cfg.fps,
        crop=Crop(0, 0, source_width, source_height),
        scale=(width, height),
        gray=True,
    ):
        if previous is not None:
            delta = np.abs(frame.astype(np.int16) - previous.astype(np.int16))
            accumulator += delta > threshold
        previous = frame

    row_profile = accumulator.sum(axis=1)
    if row_profile.max() <= 0:
        raise ValueError(f"no motion found in {path.name}; cannot locate play band")

    peak = int(row_profile.argmax())
    active = np.flatnonzero(row_profile > activity_fraction * row_profile.max())

    scale_y = source_height / height
    band = Band(
        top=int(active[0] * scale_y),
        bottom=int((active[-1] + 1) * scale_y),
        peak=int(peak * scale_y),
        column_profile=accumulator[active[0] : active[-1] + 1].sum(axis=0),
    )
    return band, accumulator
