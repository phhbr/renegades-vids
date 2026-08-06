"""State machine and tiering tests on synthetic dispersion traces."""

from __future__ import annotations

import numpy as np
import pytest

from playsplit.audio import Whistle
from playsplit.segments import build
from playsplit.statemachine import (
    SegmentConfigSM,
    Tier,
    evidence_for_anchor,
    find_episodes,
)

FPS = 5.0


def trace(duration_s: float = 120.0, level: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """A quiet clip: constant dispersion, no plays."""
    times = np.arange(0, duration_s, 1 / FPS)
    return times, np.full(len(times), level)


def add_play(
    times: np.ndarray,
    values: np.ndarray,
    snap: float,
    duration: float,
    peak: float,
    *,
    formation_s: float = 4.0,
    formation_level: float = 30.0,
) -> None:
    """Insert a formation (settled) followed by a burst (dispersion spike)."""
    settled = (times >= snap - formation_s) & (times < snap)
    values[settled] = formation_level
    live = (times >= snap) & (times < snap + duration)
    count = int(live.sum())
    rise = max(int(count * 0.35), 2)
    shape = np.concatenate([
        np.linspace(formation_level, peak, rise),
        np.linspace(peak, formation_level * 1.6, count - rise),
    ])
    values[live] = shape[:count]


def test_quiet_trace_yields_no_episodes() -> None:
    times, values = trace()

    episodes, _ = find_episodes(times, values, FPS, SegmentConfigSM())

    assert episodes == []


def test_formation_then_burst_is_anchored_and_confident() -> None:
    times, values = trace()
    add_play(times, values, snap=40.0, duration=8.0, peak=300.0)
    whistles = [Whistle(48.5, 48.8, 1.0)]

    cfg = SegmentConfigSM()
    episodes, _ = find_episodes(times, values, FPS, cfg)
    candidates = build(
        episodes, whistles, times, values, FPS, cfg,
        pre_buffer_s=3.0, post_buffer_s=0.7, clip_duration=times[-1],
    )

    # HIGH is retired: the formation rule scored zero true positives on real
    # footage. A formation-plus-burst play is a confident MEDIUM instead.
    assert [c.tier for c in candidates] == [Tier.MEDIUM]
    assert candidates[0].confidence >= 0.6


def test_formation_alone_never_emits_a_play() -> None:
    """The pre-game player-pass check: two lines, but no snap ever comes."""
    times, values = trace()
    settled = (times >= 30.0) & (times < 60.0)
    values[settled] = 30.0
    whistles = [Whistle(61.0, 61.3, 1.0)]

    cfg = SegmentConfigSM()
    episodes, _ = find_episodes(times, values, FPS, cfg)
    candidates = build(
        episodes, whistles, times, values, FPS, cfg,
        pre_buffer_s=3.0, post_buffer_s=0.7, clip_duration=times[-1],
    )

    assert all(c.tier is not Tier.HIGH for c in candidates)
    assert Tier.MEDIUM not in [c.tier for c in candidates]


def test_burst_without_formation_is_medium() -> None:
    times, values = trace()
    add_play(times, values, snap=40.0, duration=8.0, peak=300.0, formation_s=0.2)
    whistles = [Whistle(48.5, 48.8, 1.0)]

    cfg = SegmentConfigSM()
    episodes, _ = find_episodes(times, values, FPS, cfg)
    tiers = [
        c.tier
        for c in build(
            episodes, whistles, times, values, FPS, cfg,
            pre_buffer_s=3.0, post_buffer_s=0.7, clip_duration=times[-1],
        )
    ]

    assert Tier.HIGH not in tiers


def test_anchor_without_burst_still_reaches_review() -> None:
    """Short plays produce no measurable spike; the anchor must carry them."""
    times, values = trace()
    whistles = [Whistle(60.0, 60.3, 1.0)]

    cfg = SegmentConfigSM()
    episodes, _ = find_episodes(times, values, FPS, cfg)
    candidates = build(
        episodes, whistles, times, values, FPS, cfg,
        pre_buffer_s=3.0, post_buffer_s=0.7, clip_duration=times[-1],
    )

    assert [c.tier for c in candidates] == [Tier.LOW]


def test_ignore_ranges_suppress_with_a_reason() -> None:
    times, values = trace()
    add_play(times, values, snap=40.0, duration=8.0, peak=300.0)
    whistles = [Whistle(48.5, 48.8, 1.0)]

    cfg = SegmentConfigSM()
    episodes, _ = find_episodes(times, values, FPS, cfg)
    candidates = build(
        episodes, whistles, times, values, FPS, cfg,
        pre_buffer_s=3.0, post_buffer_s=0.7, clip_duration=times[-1],
        ignore_ranges=[[30.0, 60.0]],
    )

    suppressed = [c for c in candidates if c.tier is Tier.SUPPRESSED]
    assert suppressed and "ignore_range" in suppressed[0].reasons[0]


def test_backtrack_never_crosses_the_previous_play() -> None:
    """Without a floor, a play inherits its predecessor's burst as its own."""
    times, values = trace()
    add_play(times, values, snap=40.0, duration=6.0, peak=400.0)
    add_play(times, values, snap=50.0, duration=3.0, peak=90.0)

    cfg = SegmentConfigSM()
    unfloored = evidence_for_anchor(times, values, 54.0, FPS, cfg, floor=0.0)
    floored = evidence_for_anchor(times, values, 54.0, FPS, cfg, floor=47.0)

    # Unfloored, the anchor claims the *previous* play's much larger burst.
    assert unfloored.peak_time is not None and unfloored.peak_time < 47.0
    assert floored.peak_time is not None and floored.peak_time >= 47.0


def test_every_anchor_gets_a_disposition() -> None:
    """Nothing is silently dropped; segments.json stays auditable."""
    times, values = trace(duration_s=200.0)
    add_play(times, values, snap=40.0, duration=8.0, peak=300.0)
    whistles = [Whistle(t, t + 0.3, 1.0) for t in (48.5, 90.0, 130.0, 170.0)]

    cfg = SegmentConfigSM()
    episodes, _ = find_episodes(times, values, FPS, cfg)
    candidates = build(
        episodes, whistles, times, values, FPS, cfg,
        pre_buffer_s=3.0, post_buffer_s=0.7, clip_duration=times[-1],
    )

    assert {c.anchor_id for c in candidates if c.anchor_id is not None} == {0, 1, 2, 3}


@pytest.mark.parametrize("start,expect_partial", [(0.0, True), (60.0, False)])
def test_clip_boundary_marks_partial(start: float, expect_partial: bool) -> None:
    times, values = trace()
    add_play(times, values, snap=max(start, 2.0), duration=6.0, peak=300.0)
    anchor = max(start, 2.0) + 6.5
    cfg = SegmentConfigSM()
    episodes, _ = find_episodes(times, values, FPS, cfg)

    candidates = build(
        episodes, [Whistle(anchor, anchor + 0.3, 1.0)], times, values, FPS, cfg,
        pre_buffer_s=8.0, post_buffer_s=0.7, clip_duration=times[-1],
    )

    live = [c for c in candidates if c.tier is not Tier.SUPPRESSED]
    assert live and live[0].partial is expect_partial


def test_consecutive_plays_never_overlap() -> None:
    """The pre-buffer must not reach back over the previous play's end.

    Regression: the clamp to the previous end was applied before the buffer was
    subtracted, so every candidate started `pre_buffer` seconds inside its
    predecessor. Two kept clips then contained the same footage, and review
    could not catch it because both looked correct on their own.
    """
    times, values = trace(duration_s=140.0)
    add_play(times, values, snap=30.0, duration=6.0, peak=300.0)
    add_play(times, values, snap=44.0, duration=6.0, peak=300.0)
    whistles = [Whistle(t, t + 0.3, 1.0) for t in (37.0, 51.0)]

    cfg = SegmentConfigSM()
    episodes, _ = find_episodes(times, values, FPS, cfg)
    candidates = [
        c
        for c in build(
            episodes, whistles, times, values, FPS, cfg,
            pre_buffer_s=8.0, post_buffer_s=0.7, clip_duration=times[-1],
        )
        if c.tier is not Tier.SUPPRESSED
    ]

    ordered = sorted(candidates, key=lambda c: c.start)
    for earlier, later in zip(ordered, ordered[1:]):
        assert later.start >= earlier.end - 1e-6, (
            f"{later.start:.2f} starts inside {earlier.start:.2f}-{earlier.end:.2f}"
        )
