"""IDLE → FORMATION → LIVE → DEAD segmentation.

The state machine survives from the original brief; only its inputs changed.
It runs on dominant-cluster dispersion rather than pixel motion, and whistle
anchors corroborate its LIVE → DEAD transitions instead of driving them.

Two rules earn their keep here:

*A whistle never forces DEAD on its own.* Measured on the calibration clip,
18 of 30 anchors are not our play ends -- a second game shares the audio and
is often louder. A whistle only closes a play if the visual signal is already
decaying. Foreign whistles did not in fact land mid-play in that clip, but
precision 0.43 means the safeguard costs nothing and prevents a truncation
that would be invisible in review.

*A play is never emitted from a formation alone.* That is the pre-game
player-pass check from the brief: two lines facing the referees, which looks
exactly like a snap that never comes. FORMATION must be followed by a burst.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .audio import Whistle


class State(str, Enum):
    IDLE = "idle"
    FORMATION = "formation"
    LIVE = "live"
    DEAD = "dead"


class Tier(str, Enum):
    """Disposition of a candidate.

    HIGH is the only tier that may be auto-accepted; everything else goes to
    review. SUPPRESSED rows are still written to segments.json with a reason,
    so nothing is ever silently dropped.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SUPPRESSED = "suppressed"


@dataclass
class SegmentConfigSM:
    """Thresholds for the state machine.

    Separate from the cutting buffers in :class:`playsplit.config.SegmentConfig`
    so detection and presentation stay independently tunable.
    """

    #: Rolling window for the local dispersion baseline.
    baseline_window_s: float = 20.0
    #: Window searched backwards from a whistle anchor for the burst.
    backtrack_s: float = 20.0
    #: Dispersion below this fraction of the burst peak counts as settled.
    formation_max_ratio: float = 0.75
    #: A formation must hold this long before it can arm a play. Swept against
    #: the 13 labelled plays: 3.0s is the shortest value that keeps the
    #: negative control free of auto-accepts. Shorter values admit between-play
    #: lulls, which look settled but are only players walking back.
    formation_min_s: float = 3.0
    #: Fraction of the formation-to-peak rise that marks the snap.
    snap_rise_fraction: float = 0.25
    #: Dispersion above this multiple of the pre-snap level is a burst.
    spike_ratio: float = 1.5
    #: ...and it must also clear the clip's ambient dispersion by this much.
    #: Without the second test, a settled formation that simply relaxes back to
    #: ambient reads as a 2x "burst" -- which is precisely the pre-game
    #: player-pass check the brief warns about: two lines facing the referees,
    #: and no snap ever comes.
    min_peak_over_ambient: float = 1.15
    #: A play is dead once dispersion falls back to this fraction of its peak.
    decay_fraction: float = 0.55
    #: ...and stays there this long.
    decay_hold_s: float = 1.6
    #: A whistle corroborates an end if the visual signal decays within this.
    whistle_confirm_s: float = 3.0
    #: An anchor is attributed to a live episode ending within this of it.
    anchor_attach_s: float = 5.0
    min_play_s: float = 1.5
    max_play_s: float = 25.0


@dataclass
class Episode:
    """A LIVE stretch found by the forward pass."""

    start: float
    end: float
    peak: float
    baseline: float
    formation_s: float

    @property
    def had_formation(self) -> bool:
        return self.formation_s > 0.0


@dataclass
class Candidate:
    """One emitted segment, with the reasoning that produced it."""

    index: int
    start: float
    end: float
    tier: Tier
    confidence: float
    anchor_id: int | None
    anchor_time: float | None
    reasons: list[str] = field(default_factory=list)
    partial: bool = False
    accepted: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start


def _rolling_median(values: np.ndarray, width: int) -> np.ndarray:
    width = max(3, width | 1)
    padded = np.pad(values, (width // 2, width // 2), mode="edge")
    return np.nanmedian(
        np.lib.stride_tricks.sliding_window_view(padded, width), axis=1
    )


def find_episodes(
    times: np.ndarray,
    dispersion: np.ndarray,
    fps: float,
    cfg: SegmentConfigSM,
) -> tuple[list[Episode], list[State]]:
    """Forward pass: walk the dispersion trace through the state machine.

    Returns the LIVE episodes and the per-frame state trace, which the review
    page renders so a rejected play can be explained rather than guessed at.
    """
    baseline = _rolling_median(dispersion, int(cfg.baseline_window_s * fps))
    states = [State.IDLE] * len(dispersion)

    episodes: list[Episode] = []
    state = State.IDLE
    formation_frames = 0
    formation_level = np.nan
    live_start = 0.0
    live_peak = 0.0
    decay_frames = 0

    decay_hold = int(cfg.decay_hold_s * fps)
    formation_min = int(cfg.formation_min_s * fps)

    for index, value in enumerate(dispersion):
        if not np.isfinite(value):
            states[index] = state
            continue
        local = baseline[index] if np.isfinite(baseline[index]) else value

        if state in (State.IDLE, State.DEAD, State.FORMATION):
            settled = value <= cfg.formation_max_ratio * local
            if settled:
                formation_frames += 1
                formation_level = (
                    value if not np.isfinite(formation_level)
                    else 0.7 * formation_level + 0.3 * value
                )
                state = State.FORMATION if formation_frames >= 2 else state
            else:
                reference = formation_level if np.isfinite(formation_level) else local
                if value >= cfg.spike_ratio * max(reference, 1e-6):
                    state = State.LIVE
                    live_start = times[index]
                    live_peak = value
                    decay_frames = 0
                    states[index] = state
                    continue
                formation_frames = 0

        elif state is State.LIVE:
            live_peak = max(live_peak, value)
            if value <= cfg.decay_fraction * live_peak:
                decay_frames += 1
            else:
                decay_frames = 0
            too_long = times[index] - live_start >= cfg.max_play_s
            if decay_frames >= decay_hold or too_long:
                end = times[index]
                if end - live_start >= cfg.min_play_s:
                    episodes.append(
                        Episode(
                            start=live_start,
                            end=end,
                            peak=live_peak,
                            baseline=(
                                formation_level if np.isfinite(formation_level) else local
                            ),
                            formation_s=formation_frames / fps,
                        )
                    )
                state = State.DEAD
                formation_frames = 0
                formation_level = np.nan

        states[index] = state

    if state is State.LIVE and times[-1] - live_start >= cfg.min_play_s:
        # Clip ended mid-play; keep it, the caller marks it partial.
        episodes.append(
            Episode(times[-1], times[-1], live_peak, formation_level, formation_frames / fps)
        )
        episodes[-1] = Episode(
            live_start, times[-1], live_peak,
            formation_level if np.isfinite(formation_level) else 0.0,
            formation_frames / fps,
        )
    return episodes, states


@dataclass
class AnchorEvidence:
    """What the dispersion trace says about one whistle anchor."""

    #: Sustained settled dispersion immediately preceding the burst, seconds.
    formation_s: float
    #: Burst height relative to the pre-burst level.
    spike_ratio: float
    #: Estimated snap time, i.e. where dispersion left the formation level.
    snap: float | None
    #: Time of the dispersion peak.
    peak_time: float | None

    @property
    def spiked(self) -> bool:
        return self.snap is not None


def evidence_for_anchor(
    times: np.ndarray,
    dispersion: np.ndarray,
    anchor: float,
    fps: float,
    cfg: SegmentConfigSM,
    floor: float = 0.0,
    ambient: float | None = None,
) -> AnchorEvidence:
    """Score the backtrack window behind a whistle anchor.

    Deliberately measured against the *peak* inside a fixed window ending at
    the anchor, not against an episode boundary. An earlier version asked
    whether dispersion was low before the episode started, which is true by
    construction -- the episode starts when dispersion rises -- so it called
    almost everything a formation and auto-accepted 13 false plays.

    The peak is the only landmark locatable without already knowing the
    answer, so both the burst height and the settled run are referred to it.
    """
    # *floor* is the previous play's end. Backtracking across it would let a
    # play inherit its predecessor's burst as its own, which put estimated
    # starts a median 9 s away from the labels.
    start = max(anchor - cfg.backtrack_s, floor)
    window = (times >= start) & (times < anchor)
    values = dispersion[window]
    stamps = times[window]
    finite = np.isfinite(values)
    values, stamps = values[finite], stamps[finite]
    if len(values) < 6:
        return AnchorEvidence(0.0, 0.0, None, None)

    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])
    if peak_index < 2:
        # Peak sits at the window edge; the rise happened before we can see it.
        return AnchorEvidence(0.0, 0.0, None, None)

    before = values[:peak_index]
    ratio = peak / max(float(np.median(before)), 1e-6)
    if ambient is None:
        ambient = float(np.nanmedian(dispersion[np.isfinite(dispersion)]))
    if ratio <= cfg.spike_ratio or peak < cfg.min_peak_over_ambient * ambient:
        return AnchorEvidence(0.0, ratio, None, float(stamps[peak_index]))

    # The snap is where the trace leaves the settled level on its way to the
    # peak; the formation is the settled run immediately before it. Both are
    # referred to the same ceiling, which is the tighter of "well below this
    # burst" and "tighter than this clip's ambient spread" -- the second test
    # stops an empty, quiet field from reading as a formation.
    ceiling = cfg.formation_max_ratio * min(peak, ambient)
    limit = int(cfg.max_play_s * fps)

    snap_index = peak_index
    steps = 0
    while snap_index > 0 and values[snap_index - 1] > ceiling and steps < limit:
        snap_index -= 1
        steps += 1

    run = 0
    while snap_index - run - 1 >= 0 and values[snap_index - run - 1] <= ceiling:
        run += 1

    return AnchorEvidence(
        formation_s=run / fps,
        spike_ratio=ratio,
        snap=float(stamps[snap_index]),
        peak_time=float(stamps[peak_index]),
    )
