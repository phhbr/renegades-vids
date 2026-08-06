"""Cutting plays out of source clips, plus the manifest.

Default is stream copy with the start expanded *backwards* to the preceding
keyframe. GoPro writes a keyframe every 1.000 s exactly (measured), so the
expansion costs at most one second of lead-in, which the pre-buffer already
tolerates. Never shrink forward into the play: a lossless cut that clips the
snap is worse than a slightly early one.

``--reencode`` gives frame-exact boundaries via ``h264_videotoolbox``. Encoding
is the one job VideoToolbox is genuinely good at -- unlike decoding, where it
measured 3.3x slower than software on this footage. It has no CRF mode, so the
bitrate is set explicitly.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .probe import ClipInfo

MANIFEST_FIELDS = [
    "play", "source_clip", "start", "end", "duration",
    "start_tc", "end_tc", "confidence", "tier", "span_conf", "partial", "notes",
]


@dataclass
class Cut:
    """One play to be written out."""

    play: int
    clip: ClipInfo
    start: float
    end: float
    confidence: float = 1.0
    tier: str = "label"
    span_conf: str = "high"
    partial: bool = False
    notes: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def filename(self) -> str:
        """``P07__GH010007__t0134.mp4`` -- sortable, and traceable to source."""
        return f"P{self.play:02d}__{self.clip.path.stem}__t{int(self.start):04d}.mp4"


def timecode(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def keyframe_before(path: Path, timestamp: float, search_s: float = 12.0) -> float:
    """Last keyframe at or before *timestamp*.

    Stream copy can only start on a keyframe. ffmpeg would otherwise seek
    forward to the next one and silently eat the beginning of the play.
    """
    window_start = max(0.0, timestamp - search_s)
    # ``-skip_frame nokey`` makes ffprobe report keyframes only, so there is no
    # second field to line up. Requesting ``frame=pts_time,key_frame`` instead
    # is a trap: ffprobe emits fields in its own fixed order regardless of the
    # order asked for, and reading them the other way round silently returned
    # the window floor every time -- which stretched an 8s play to 20s.
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
            "-read_intervals", f"{window_start}%{timestamp + 0.5}",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    keyframes = [
        float(line)
        for line in result.stdout.split()
        if line and line[0].isdigit() and float(line) <= timestamp + 1e-3
    ]
    return max(keyframes) if keyframes else window_start


def cut_one(
    item: Cut,
    dest: Path,
    *,
    reencode: bool = False,
    bitrate: str = "40M",
    dry_run: bool = False,
) -> tuple[Path, float]:
    """Write one play. Returns the file and the actual start used."""
    if reencode:
        actual_start = item.start
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{item.start:.3f}", "-to", f"{item.end:.3f}",
            "-i", str(item.clip.path),
            "-c:v", "h264_videotoolbox", "-b:v", bitrate,
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-y", str(dest),
        ]
    else:
        actual_start = keyframe_before(item.clip.path, item.start)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{actual_start:.3f}", "-to", f"{item.end:.3f}",
            "-i", str(item.clip.path),
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", "-y", str(dest),
        ]

    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
    return dest, actual_start


def write_manifest(path: Path, cuts: list[Cut], starts: dict[int, float]) -> None:
    """One row per play, with the timecodes needed to find it in the source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for item in cuts:
            actual = starts.get(item.play, item.start)
            writer.writerow(
                {
                    "play": item.play,
                    "source_clip": item.clip.name,
                    "start": f"{actual:.3f}",
                    "end": f"{item.end:.3f}",
                    "duration": f"{item.end - actual:.3f}",
                    "start_tc": timecode(actual),
                    "end_tc": timecode(item.end),
                    "confidence": f"{item.confidence:.2f}",
                    "tier": item.tier,
                    "span_conf": item.span_conf,
                    "partial": "true" if item.partial else "",
                    "notes": item.notes,
                }
            )
