"""Whistle detection.

Whistles are the tightest signal available in this footage -- they mark play
ends to within a few tens of milliseconds, where every video-derived boundary
is quantised to the analysis frame rate. They are *anchors*, never oracles: the
adjacent pitch is fully audible and audio has no notion of our field ROI, so a
whistle only emits a play once detection features corroborate it.

Measured on the sample clips: 21 anchors on RenegadesVsAnts/GH010007 (in-game,
~1 per 25 s) versus 10 on SpatzenVsAnts/GH020002 (warm-up heavy). That second
number is exactly why corroboration is mandatory.
"""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AudioConfig


@dataclass(frozen=True)
class Whistle:
    """A candidate play-end anchor."""

    start: float
    end: float
    strength: float

    @property
    def time(self) -> float:
        """Anchor timestamp -- the onset, which is when the ref reacted."""
        return self.start


def load_audio(path: Path, sample_rate: int, dest: Path) -> np.ndarray:
    """Decode a clip's audio to mono float32 at *sample_rate*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-vn", "-ac", "1", "-ar", str(sample_rate),
                "-c:a", "pcm_s16le", "-y", str(dest),
            ],
            check=True,
        )
    with wave.open(str(dest)) as handle:
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _spectrogram(signal: np.ndarray, cfg: AudioConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (magnitude spectrogram, frequency bins)."""
    frames = np.lib.stride_tricks.sliding_window_view(signal, cfg.window)[:: cfg.hop]
    windowed = frames * np.hanning(cfg.window)
    return np.abs(np.fft.rfft(windowed, axis=1)), np.fft.rfftfreq(cfg.window, 1 / cfg.sample_rate)


def _robust_threshold(values: np.ndarray, k: float) -> float:
    """Noise-floor threshold: median + *k* robust standard deviations.

    The median and MAD are computed over the whole clip but are dominated by
    the ~98% of it that contains no whistle, so the threshold tracks the wind
    and crowd floor rather than the events being detected.
    """
    median = float(np.median(values))
    # 1.4826 rescales MAD to a standard-deviation estimate for normal noise.
    mad = float(np.median(np.abs(values - median))) * 1.4826
    if mad <= 0:
        mad = float(values.std()) or 1e-9
    return median + k * mad


def detect(signal: np.ndarray, cfg: AudioConfig) -> list[Whistle]:
    """Find whistle anchors in a mono signal.

    Merges events closer than ``cfg.merge_gap_s`` -- two referees and stadium
    echo reliably produce doublets, visible as pairs ~2 s apart in the sample
    footage, and each pair marks one play end rather than two.
    """
    spec, freqs = _spectrogram(signal, cfg)
    band = (freqs >= cfg.band_low_hz) & (freqs <= cfg.band_high_hz)
    broad = (freqs >= cfg.broadband_low_hz) & (freqs < cfg.broadband_high_hz)

    band_energy = spec[:, band]
    ratio = band_energy.sum(axis=1) / (spec[:, broad].sum(axis=1) + 1e-9)
    peak_ratio = band_energy.max(axis=1) / (band_energy.mean(axis=1) + 1e-9)

    hot = (ratio > _robust_threshold(ratio, cfg.ratio_mad_k)) & (
        peak_ratio > cfg.min_peak_ratio
    )

    times = np.arange(len(ratio)) * cfg.hop / cfg.sample_rate
    indices = np.flatnonzero(hot)
    if indices.size == 0:
        return []

    # Split into runs of contiguous hot frames, allowing a short dropout.
    max_gap_frames = int(round(0.1 * cfg.sample_rate / cfg.hop))
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > max_gap_frames) + 1)

    events: list[Whistle] = []
    for run in runs:
        start, end = float(times[run[0]]), float(times[run[-1]])
        if end - start < cfg.min_duration_s:
            continue
        events.append(Whistle(start, end, float(ratio[run].max())))

    return _merge(events, cfg.merge_gap_s)


def _merge(events: list[Whistle], gap: float) -> list[Whistle]:
    """Collapse anchors whose onsets fall within *gap* seconds."""
    if not events:
        return []
    merged = [events[0]]
    for event in events[1:]:
        previous = merged[-1]
        if event.start - previous.start <= gap:
            merged[-1] = Whistle(
                previous.start, max(previous.end, event.end),
                max(previous.strength, event.strength),
            )
        else:
            merged.append(event)
    return merged
