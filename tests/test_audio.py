"""Whistle detector tests using synthetic signals."""

from __future__ import annotations

import numpy as np
import pytest

from playsplit.audio import Whistle, _merge, detect
from playsplit.config import AudioConfig


def _tone(signal: np.ndarray, sr: int, start: float, duration: float, freq: float) -> None:
    """Add a pure tone in place."""
    lo, hi = int(start * sr), int((start + duration) * sr)
    t = np.arange(hi - lo) / sr
    signal[lo:hi] += 0.8 * np.sin(2 * np.pi * freq * t)


def _wind(signal: np.ndarray, rng: np.random.Generator, level: float = 0.08) -> None:
    """Add broadband noise, the thing that must NOT trigger a detection.

    The default level puts the tones at a signal-to-noise ratio comparable to
    the sample footage, where whistles clear the band-dominance floor by a wide
    margin and sustain for 0.24-0.58 s.
    """
    signal += level * rng.standard_normal(len(signal))


def test_detects_narrowband_whistle_and_ignores_wind() -> None:
    cfg = AudioConfig()
    sr = cfg.sample_rate
    signal = np.zeros(30 * sr, dtype=np.float32)
    _wind(signal, np.random.default_rng(0))
    for start in (5.0, 12.0, 22.0):
        _tone(signal, sr, start, 0.4, 3200.0)

    found = [w.time for w in detect(signal, cfg)]

    assert len(found) == 3
    for expected, actual in zip((5.0, 12.0, 22.0), found):
        assert actual == pytest.approx(expected, abs=0.15)


def test_wind_alone_produces_no_anchors() -> None:
    """Broadband gusts are loud but not narrowband; they must stay silent."""
    cfg = AudioConfig()
    signal = np.zeros(30 * cfg.sample_rate, dtype=np.float32)
    _wind(signal, np.random.default_rng(2), level=0.4)

    assert detect(signal, cfg) == []


def test_short_blips_are_rejected() -> None:
    """A 50 ms chirp is shorter than a real whistle and must be dropped."""
    cfg = AudioConfig()
    sr = cfg.sample_rate
    signal = np.zeros(20 * sr, dtype=np.float32)
    _wind(signal, np.random.default_rng(1))
    _tone(signal, sr, 8.0, 0.05, 3000.0)

    assert detect(signal, cfg) == []


def test_threshold_does_not_scale_with_whistle_density() -> None:
    """A busy clip must not mask its own whistles.

    This is the failure mode that retired the percentile threshold: with a
    high-percentile cutoff, adding whistles raises the bar and drops
    detections. Recall must not degrade as plays get denser.
    """
    cfg = AudioConfig()
    sr = cfg.sample_rate

    def build(count: int) -> np.ndarray:
        signal = np.zeros(60 * sr, dtype=np.float32)
        _wind(signal, np.random.default_rng(3))
        for index in range(count):
            _tone(signal, sr, 3.0 + index * 3.0, 0.4, 3200.0)
        return signal

    assert len(detect(build(3), cfg)) == 3
    assert len(detect(build(15), cfg)) == 15


def test_doublets_merge_into_one_anchor() -> None:
    """Two refs whistling 1.5 s apart mark one play end, not two."""
    events = [
        Whistle(10.0, 10.3, 1.0),
        Whistle(11.5, 11.8, 1.0),
        Whistle(40.0, 40.4, 1.0),
    ]

    merged = _merge(events, gap=2.5)

    assert [w.time for w in merged] == [10.0, 40.0]
    assert merged[0].end == 11.8


def test_merge_preserves_distinct_anchors() -> None:
    events = [Whistle(10.0, 10.3, 1.0), Whistle(13.0, 13.3, 1.0)]

    assert len(_merge(events, gap=2.5)) == 2


def test_merge_handles_empty_input() -> None:
    assert _merge([], gap=2.5) == []
