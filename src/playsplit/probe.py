"""ffprobe/ffmpeg wrappers and clip ordering.

Everything here shells out to the Homebrew ffmpeg/ffprobe binaries; no Python
bindings, so the pipeline stays usable offline with nothing but a venv.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv"}

#: GoPro chaptered filenames look like ``GH010007.MP4`` / ``GX010016.MP4``:
#: two letters, a two-digit *chapter*, then a four-digit *session*. Recording a
#: long game produces one session split into several chapters, so the correct
#: chronological order is ``(session, chapter)`` -- NOT the filename order,
#: which would interleave sessions (GH010007 before GH020006 is wrong).
_GOPRO_RE = re.compile(r"^(?P<prefix>G[HXP])(?P<chapter>\d{2})(?P<session>\d{4})$")


@dataclass(frozen=True)
class ClipInfo:
    """Container/stream facts probed from a single clip."""

    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool

    @property
    def name(self) -> str:
        return self.path.name


def sort_key(path: Path) -> tuple[int, int, int, str]:
    """Return a sort key placing clips in true recording order.

    GoPro chapter files sort by ``(session, chapter)``. Anything else falls back
    to modification time, then filename, and is grouped after the GoPro files
    only if it cannot be parsed -- so mixed folders stay deterministic.
    """
    match = _GOPRO_RE.match(path.stem.upper())
    if match:
        return (0, int(match["session"]), int(match["chapter"]), path.name)
    try:
        mtime = int(path.stat().st_mtime)
    except OSError:
        mtime = 0
    return (1, mtime, 0, path.name)


def find_clips(game_dir: Path) -> list[Path]:
    """List a game's source clips in recording order.

    Accepts either the ``raw/`` layout from the spec or a folder of videos, so
    existing ``assets/<tournament>/<matchup>/`` trees work without migration.
    """
    raw = game_dir / "raw"
    search_dir = raw if raw.is_dir() else game_dir
    clips = [
        p
        for p in search_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and not p.name.startswith(".")
    ]
    return sorted(clips, key=sort_key)


def probe(path: Path) -> ClipInfo:
    """Read duration, resolution and frame rate from a clip via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next(s for s in streams if s.get("codec_type") == "video")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    num, _, den = video.get("avg_frame_rate", "0/0").partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0

    return ClipInfo(
        path=path,
        duration=float(data["format"]["duration"]),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        has_audio=has_audio,
    )
