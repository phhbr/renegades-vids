"""Thumbnail extraction for the review page.

Thumbnails are cropped to the play band before being scaled down. A full frame
is close to useless here -- players are ~45 px tall in a 1920-wide frame and the
lower half is empty grass, so a downscaled full frame shows nothing a human can
label. Cropping first spends every output pixel on the action.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .frames import Crop

#: Fractions of a candidate's span to sample: just before the snap, mid-play,
#: and just after the whistle.
SAMPLE_POINTS = (0.05, 0.5, 0.95)
SAMPLE_LABELS = ("pre-snap", "mid-play", "post")


def extract(
    clip: Path,
    timestamp: float,
    crop: Crop,
    dest: Path,
    *,
    width: int = 1280,
    force: bool = False,
) -> Path:
    """Write a single band-cropped thumbnail at *timestamp*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        return dest
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{timestamp:.3f}", "-i", str(clip),
            "-frames:v", "1",
            "-vf", f"{crop.to_filter()},scale={width}:-2",
            "-q:v", "3", "-y", str(dest),
        ],
        check=True,
    )
    return dest


def extract_triplet(
    clip: Path,
    start: float,
    end: float,
    crop: Crop,
    out_dir: Path,
    stem: str,
    *,
    width: int = 1280,
    force: bool = False,
) -> list[Path]:
    """Write the pre-snap / mid-play / post thumbnails for one candidate.

    Filenames are keyed by *timestamp*, not by candidate index. Indices shift
    whenever detection changes, so index-keyed caching would silently serve a
    previous run's frames under a new candidate's number.
    """
    span = max(end - start, 0.1)
    paths = []
    for point, label in zip(SAMPLE_POINTS, SAMPLE_LABELS):
        timestamp = start + span * point
        dest = out_dir / f"{stem}__t{timestamp:09.3f}__{label}.jpg"
        paths.append(extract(clip, timestamp, crop, dest, width=width, force=force))
    return paths
