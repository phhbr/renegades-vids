"""Configuration loading.

Every threshold in the pipeline lives here and is overridable from a per-game
``playsplit.toml``. Nothing in the signal modules may hardcode a constant.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# Vendored weights, resolved relative to the repo so Ultralytics never reaches
# for the network. Overridable for experiments with a larger model.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = _REPO_ROOT / "models" / "yolo11n.pt"


@dataclass
class AudioConfig:
    """Whistle detection.

    Referee whistles are narrowband tones in the 2–4.5 kHz range. Two tests
    must both pass: the band must dominate the broadband spectrum (*ratio*),
    and the energy inside the band must be concentrated in a single peak rather
    than spread out (*peak_ratio*) -- wind is broadband and loud, so the ratio
    test alone lets gusts through.
    """

    sample_rate: int = 16_000
    window: int = 1024
    hop: int = 256
    band_low_hz: float = 2000.0
    band_high_hz: float = 4500.0
    broadband_low_hz: float = 200.0
    broadband_high_hz: float = 8000.0
    #: Band-dominance threshold, in robust standard deviations above the noise
    #: floor (median + k * MAD). Deliberately NOT a high percentile: a
    #: percentile threshold rises with whistle density, so a busy clip would
    #: mask its own whistles -- exactly backwards for a recall-first pipeline.
    #:
    #: Lowered 8.0 -> 7.0 once a second game was labelled. 8.0 was fitted to
    #: one clip; on the noisier GH010009 (audio RMS 0.090 vs 0.077) the higher
    #: noise floor lifted the absolute threshold and play-end recall fell to
    #: 10/15, losing a play the pipeline then never saw. That is the original
    #: percentile bug wearing a different hat: a threshold that adapts to the
    #: noise still gets stricter exactly where the signal is weaker. 7.0 gives
    #: full recall on both labelled clips.
    ratio_mad_k: float = 7.0
    #: Minimum peak-to-mean ratio within the band (narrowband-ness test).
    min_peak_ratio: float = 7.0
    #: A whistle must sustain for at least this long.
    min_duration_s: float = 0.12
    #: Two refs and echoes produce doublets; merge anchors closer than this.
    merge_gap_s: float = 2.5


@dataclass
class DetectConfig:
    """Person detection on the native-resolution ROI strip."""

    weights: Path = DEFAULT_WEIGHTS
    device: str = "mps"
    #: Inference size. Must stay at native strip width: downscaling to 640
    #: loses 70-85% of players at this camera distance (measured).
    imgsz: int = 1920
    conf: float = 0.20
    #: Vertical padding added around the auto-detected activity band.
    band_padding_px: int = 40
    #: Ultralytics tracker config. ByteTrack ships with the package, so this
    #: resolves offline.
    tracker: str = "bytetrack.yaml"
    #: Tracking is off by default. Measured on GH010007 it changed nothing that
    #: mattered -- per-track stationarity scored 5/12 legible plays against the
    #: positional occupancy map's 6/12 -- while the tracker's association step
    #: discarded 22% of detections (80663 -> 62576), which cost dispersion
    #: contrast. Kept behind a flag because it is the right tool if detections
    #: ever get dense enough for tracks to survive; at 5 fps in a crowd of ~30
    #: near-identical 45px people they do not (median track life 5.4s).
    use_tracker: bool = False


@dataclass
class SegmentConfig:
    """State machine and cutting."""

    #: Fitted to the user's labelling style, not the brief's defaults (5.0/3.0).
    #: The pre-buffer shrank from 5.0s once contraction-minimum localisation
    #: started landing the snap accurately: the old value was compensating for
    #: a start estimate that ran seconds early, and stacking it on a good
    #: estimate simply made every clip 5s too long.
    pre_buffer_s: float = 1.5
    post_buffer_s: float = 0.7
    min_play_s: float = 2.0
    max_play_s: float = 25.0
    #: How far back from a whistle anchor to search for the snap.
    max_backtrack_s: float = 45.0
    #: Candidates closer than this are merged.
    merge_gap_s: float = 2.0


@dataclass
class AnalysisConfig:
    """Frame sampling for the analysis passes."""

    fps: float = 5.0
    #: Low-res proxy used only for the activity-heatmap band estimation.
    proxy_width: int = 480
    proxy_height: int = 360
    #: Frames at the head of a clip excluded from robust-scaling statistics --
    #: camera handling at clip start otherwise compresses the whole trace.
    warmup_skip_s: float = 25.0
    #: Windows sampled across a clip to locate the play band, and their length.
    #: The band is a static property of the camera, so 8 x 8s settles it as
    #: well as a full decode at a fraction of the cost.
    band_sample_windows: int = 8
    band_window_s: float = 8.0


@dataclass
class Config:
    """Top-level config for one game directory."""

    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    #: Per-clip spans to skip outright, e.g. ``{"GH010007.MP4": [[0, 120]]}``.
    ignore_ranges: dict[str, list[list[float]]] = field(default_factory=dict)


def _apply(target: Any, values: dict[str, Any]) -> None:
    """Recursively overlay a parsed TOML table onto a dataclass instance."""
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"unknown config key: {key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        elif isinstance(current, Path):
            setattr(target, key, Path(value))
        else:
            setattr(target, key, value)


def load(game_dir: Path) -> Config:
    """Load ``<game_dir>/playsplit.toml``, falling back to defaults."""
    config = Config()
    path = game_dir / "playsplit.toml"
    if path.is_file():
        with path.open("rb") as handle:
            _apply(config, tomllib.load(handle))
    return config
