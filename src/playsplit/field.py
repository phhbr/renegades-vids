"""Field mask -- the relevance gate for detections.

This is the job the original brief's green segmentation was always for. The
activity heatmap in :mod:`playsplit.roi` says *which rows* to run the detector
on; this mask says *which detections count*. They are complementary: the band
is the inference crop, the mask is the relevance gate.

The load-bearing trick is subtracting the red running track before labelling
connected components. Background trees are the same hue as the pitch and touch
it at the frame edges, so a naive "largest green component" swallows the whole
upper half of the frame (measured: 81% of frame area). Cutting the dilated
track first severs the pitch from the trees, from the spectator areas, and from
any adjacent pitch -- which is precisely what the track physically does.

What this mask does *not* remove is people standing on our own pitch: benches,
substitutes and officials along the sideline. Those are the dominant-cluster
step's problem, not the mask's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .frames import Crop, iter_frames


@dataclass
class FieldMask:
    """A binary playing-surface mask at a working resolution."""

    mask: np.ndarray
    width: int
    height: int
    source_width: int
    source_height: int

    @property
    def area_fraction(self) -> float:
        return float(self.mask.mean())

    def contains(self, x: float, y: float) -> bool:
        """Test a point given in *source* pixel coordinates."""
        col = int(x * self.width / self.source_width)
        row = int(y * self.height / self.source_height)
        if not (0 <= col < self.width and 0 <= row < self.height):
            return False
        return bool(self.mask[row, col])

    def contains_many(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`contains` for a batch of source-space points."""
        cols = np.clip((xs * self.width / self.source_width).astype(int), 0, self.width - 1)
        rows = np.clip((ys * self.height / self.source_height).astype(int), 0, self.height - 1)
        return self.mask[rows, cols].astype(bool)


def background_frame(
    path: Path,
    source_width: int,
    source_height: int,
    *,
    width: int = 960,
    height: int = 720,
    samples: int = 24,
    duration: float | None = None,
) -> np.ndarray:
    """Per-pixel median of frames spread across the clip.

    Players, referees and passing spectators are transient, so the median is
    the empty pitch. Segmenting that rather than one arbitrary frame keeps
    bodies from punching holes in the mask.
    """
    if duration is None or duration <= 0:
        raise ValueError("duration is required to spread background samples")

    stack = []
    step = duration / (samples + 1)
    for index in range(1, samples + 1):
        for frame in iter_frames(
            path,
            fps=1.0,
            crop=Crop(0, 0, source_width, source_height),
            scale=(width, height),
            gray=False,
            start=index * step,
            duration=0.5,
        ):
            stack.append(frame)
            break
    if not stack:
        raise ValueError(f"could not sample background frames from {path.name}")
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def _reconstruct(seed: np.ndarray, within: np.ndarray, max_iterations: int = 64) -> np.ndarray:
    """Morphological reconstruction of *seed* under the constraint *within*.

    Repeatedly dilates the seed but keeps only pixels that were already set in
    *within*, so the region regrows to its natural boundary and stops there.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = (seed & within).astype(np.uint8)
    for _ in range(max_iterations):
        grown = (cv2.dilate(current, kernel) & within).astype(np.uint8)
        if np.array_equal(grown, current):
            break
        current = grown
    return current


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior gaps (bare patches, line markings, shadows)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def segment(
    background: np.ndarray,
    source_width: int,
    source_height: int,
    *,
    green_hue: tuple[int, int] = (30, 90),
    green_min_sat: int = 60,
    green_min_val: int = 40,
    red_hue_low: int = 10,
    red_hue_high: int = 170,
    red_min_sat: int = 70,
    red_min_val: int = 50,
    track_dilate_px: int = 15,
) -> tuple[FieldMask, np.ndarray]:
    """Segment the playing surface from a background frame.

    Returns the mask and the detected track pixels (kept for the overlay).
    """
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    green = (
        (hue >= green_hue[0]) & (hue <= green_hue[1])
        & (sat >= green_min_sat) & (val >= green_min_val)
    ).astype(np.uint8)
    track = (
        ((hue <= red_hue_low) | (hue >= red_hue_high))
        & (sat >= red_min_sat) & (val >= red_min_val)
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (track_dilate_px, track_dilate_px)
    )
    severed = (green & (1 - cv2.dilate(track, kernel, iterations=2))).astype(np.uint8)

    clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    severed = cv2.morphologyEx(severed, cv2.MORPH_OPEN, clean)
    severed = cv2.morphologyEx(severed, cv2.MORPH_CLOSE, clean)

    count, labelled, stats, _ = cv2.connectedComponentsWithStats(severed, 8)
    if count < 2:
        raise ValueError("no field component found; check HSV thresholds")

    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    mask = (labelled == largest).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    mask = _fill_holes(mask)

    # Regrow to the true grass boundary. The dilated cut exists only to break
    # connectivity with the trees; keeping it in the gate rejects real players,
    # because the flag pitch is laid out hard against the track and the far
    # line of scrimmage sits within a couple of metres of it (measured: 6 of 32
    # detections survived the gate during a labelled play, nearly all wrongly).
    # Plain dilation would overshoot the other way and swallow the track with
    # its benches and spectators, so reconstruct geodesically instead: grow
    # only into pixels that were green to begin with. The track is not green,
    # so the pitch physically cannot bridge it to reach the adjacent one.
    mask = _reconstruct(seed=mask, within=green)
    mask = _fill_holes(mask)

    height, width = mask.shape
    return (
        FieldMask(mask, width, height, source_width, source_height),
        track,
    )


def write_overlay(
    background: np.ndarray, mask: FieldMask, track: np.ndarray, dest: Path
) -> Path:
    """Save a mask-overlay PNG so the segmentation is inspectable."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    overlay = background.copy()
    field_px = mask.mask > 0
    overlay[field_px] = (
        0.55 * overlay[field_px] + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)
    track_px = track > 0
    overlay[track_px] = (
        0.5 * overlay[track_px] + 0.5 * np.array([0, 0, 255])
    ).astype(np.uint8)
    cv2.imwrite(str(dest), overlay)
    return dest
