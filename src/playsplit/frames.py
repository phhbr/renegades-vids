"""Analysis-frame decoding.

Frames are piped out of ffmpeg as raw planes straight into numpy. Writing an
intermediate proxy *file* was measured at roughly 3x slower end-to-end and cost
~170 MB per clip, so nothing is ever re-encoded for analysis.

Note on hardware acceleration: ``-hwaccel videotoolbox`` is a pessimisation
here, not an optimisation. Measured on the M4 over 60 s of 1920x1440@59.94
GoPro footage, decoding to 480x360 gray:

    videotoolbox   23.4 s
    software        7.1 s   <- 3.3x faster

The filter graph needs frames in system memory, so every hwaccel frame pays a
GPU->CPU readback that costs more than the decode it saves. We decode in
software and stay decode-bound at ~8.6x realtime.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Crop:
    """A pixel rectangle in source coordinates."""

    x: int
    y: int
    width: int
    height: int

    def to_filter(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


def _build_filter(fps: float, crop: Crop | None, scale: tuple[int, int] | None) -> str:
    parts = [f"fps={fps}"]
    if crop is not None:
        parts.append(crop.to_filter())
    if scale is not None:
        parts.append(f"scale={scale[0]}:{scale[1]}")
    return ",".join(parts)


def iter_frames(
    path: Path,
    fps: float,
    *,
    crop: Crop | None = None,
    scale: tuple[int, int] | None = None,
    gray: bool = True,
    start: float | None = None,
    duration: float | None = None,
) -> Iterator[np.ndarray]:
    """Yield analysis frames as numpy arrays.

    Yields ``(h, w)`` uint8 when *gray*, else ``(h, w, 3)`` in BGR order so the
    output can be handed straight to OpenCV or Ultralytics.
    """
    if crop is not None:
        out_w, out_h = crop.width, crop.height
    else:
        raise ValueError("crop is required to size the raw frame buffer")
    if scale is not None:
        out_w, out_h = scale

    pix_fmt = "gray" if gray else "bgr24"
    channels = 1 if gray else 3
    frame_bytes = out_w * out_h * channels

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-i", str(path),
        "-vf", _build_filter(fps, crop, scale),
        "-an", "-sn",
        "-pix_fmt", pix_fmt,
        "-f", "rawvideo", "-",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8)
            yield frame.reshape((out_h, out_w) if gray else (out_h, out_w, channels))
    finally:
        proc.stdout.close()
        proc.wait()
