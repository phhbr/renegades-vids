"""Tiered candidate emission and segments.json I/O.

Tiers implement recall-over-precision explicitly rather than by threshold
choice. Only HIGH may be auto-accepted; MEDIUM and LOW go to review; every
anchor that produced nothing is written out SUPPRESSED with a reason, so the
file explains itself and no anchor vanishes silently.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .audio import Whistle
from .statemachine import (
    Candidate,
    Episode,
    SegmentConfigSM,
    Tier,
    evidence_for_anchor,
)

#: Base confidence per tier. Presentational only -- it orders the review queue
#: and must never gate a decision.
CONFIDENCE = {Tier.HIGH: 0.9, Tier.MEDIUM: 0.6, Tier.LOW: 0.3, Tier.SUPPRESSED: 0.0}

#: Tier.HIGH is no longer assigned. The formation-plus-burst rule that produced
#: it scored zero true positives on the labelled clip while still firing on the
#: negative control -- a sort key anti-correlated with truth is worse than
#: none. Ranking is by duration plausibility instead, the only key measured to
#: separate real plays from false candidates (AUC 0.75). The enum member stays
#: so existing segments.json files still parse.
PLAUSIBLE_DURATION_S = (3.0, 24.0)


def duration_confidence(duration: float) -> float:
    """How play-like a candidate's length is, in [0, 1]."""
    low, high = PLAUSIBLE_DURATION_S
    if low <= duration <= high:
        return 1.0
    reference = low if duration < low else high
    return max(0.0, 1.0 - abs(duration - reference) / reference)


def build(
    episodes: list[Episode],
    whistles: list[Whistle],
    times: np.ndarray,
    dispersion: np.ndarray,
    fps: float,
    cfg: SegmentConfigSM,
    *,
    fine: np.ndarray | None = None,
    pre_buffer_s: float,
    post_buffer_s: float,
    clip_duration: float,
    ignore_ranges: list[list[float]] | None = None,
) -> list[Candidate]:
    """Attach anchors to episodes and assign a tier to each.

    Anchors drive recall: whistle recall measured 1.00 on the calibration clip,
    so every real play has an anchor to ride even when its visual burst is too
    short to register. Episodes drive precision, and also provide the
    visual-only path for plays whose whistle the wind ate.
    """
    ignore_ranges = ignore_ranges or []
    candidates: list[Candidate] = []
    used_episodes: set[int] = set()
    previous_end = 0.0
    #: Median span of anchored candidates so far, used as the fallback play
    #: length once a few plays have been seen on this clip.
    learned: list[float] = []
    learned_median: float | None = None

    def ignored(time: float) -> bool:
        return any(low <= time <= high for low, high in ignore_ranges)

    for anchor_id, whistle in enumerate(whistles):
        if ignored(whistle.time):
            candidates.append(
                _suppressed(anchor_id, whistle, "inside a configured ignore_range")
            )
            continue

        evidence = evidence_for_anchor(
            times, dispersion, whistle.time, fps, cfg,
            floor=previous_end, fine=fine, median_play_s=learned_median,
        )
        match = _nearest_episode(episodes, whistle.time, cfg.anchor_attach_s, used_episodes)
        if match is not None:
            used_episodes.add(match[0])

        if evidence.spiked:
            tier = Tier.MEDIUM
            reasons = [
                f"burst {evidence.spike_ratio:.1f}x, formation "
                f"{evidence.formation_s:.1f}s",
                f"whistle @ {whistle.time:.2f}s corroborates the end",
            ]
        else:
            # No measurable burst. Kept at LOW rather than dropped: short plays
            # do not develop a peak above baseline, and whistle recall is the
            # only signal that reaches them.
            tier = Tier.LOW
            reasons = ["anchor without a measurable burst; span estimated from cadence"]

        start = (
            evidence.snap
            if evidence.snap is not None
            else _quiet_before(times, dispersion, whistle.time, fps, cfg)
        )
        # Guaranteed lookback. A late start omits the snap and destroys the
        # play; an early start only prepends walk-up footage. So the estimate
        # is never trusted to run later than a long play would: whichever of
        # the two is earlier wins. This bounds lateness structurally rather
        # than relying on the estimator, which the dispersion trace cannot
        # support -- it has several minima per play and the deepest is not
        # reliably the snap.
        guaranteed = whistle.time - cfg.guaranteed_lookback_s
        if guaranteed < start:
            reasons.append(
                f"start pulled back to the {cfg.guaranteed_lookback_s:.0f}s "
                "guaranteed lookback"
            )
            start = guaranteed
        start = max(start, previous_end)
        previous_end = whistle.time
        if evidence.span_conf == "high":
            learned.append(whistle.time - start)
            learned_median = float(np.median(learned))
        candidates.append(
            Candidate(
                index=0,
                start=max(0.0, start - pre_buffer_s),
                end=min(whistle.time + post_buffer_s, clip_duration),
                tier=tier,
                confidence=CONFIDENCE[tier],
                anchor_id=anchor_id,
                anchor_time=whistle.time,
                reasons=reasons,
                span_conf=evidence.span_conf,
            )
        )

    # Visual-only episodes: the wind-masked recovery path. Deliberately narrow.
    # On the calibration clip this path produced roughly a third of all
    # candidates and contributed zero unique recall, so it now needs a higher
    # burst, must not duplicate an anchor, and must either have closed cleanly
    # or been flushed at a clip boundary.
    anchor_times = [w.time for w in whistles]
    for position, episode in enumerate(episodes):
        if position in used_episodes:
            continue
        if any(abs(episode.end - a) <= cfg.visual_dedupe_s for a in anchor_times):
            continue
        evidence = evidence_for_anchor(
            times, dispersion, episode.end, fps, cfg, fine=fine,
            median_play_s=learned_median,
        )
        boundary = not episode.closed
        if not boundary and (
            not evidence.spiked or evidence.spike_ratio < cfg.visual_spike_ratio
        ):
            continue
        reason = (
            "episode still live at the clip boundary; force-closed and kept "
            "as a partial play"
            if boundary
            else f"visual burst {evidence.spike_ratio:.1f}x with no whistle; "
            "possible wind-masked play end"
        )
        start = evidence.snap if evidence.snap is not None else episode.start
        candidates.append(
            Candidate(
                index=0,
                start=max(0.0, start - pre_buffer_s),
                end=min(episode.end + (0.0 if boundary else post_buffer_s), clip_duration),
                tier=Tier.LOW,
                confidence=CONFIDENCE[Tier.LOW],
                anchor_id=None,
                anchor_time=None,
                reasons=[reason],
                span_conf=evidence.span_conf,
            )
        )

    candidates.sort(key=lambda c: c.start)
    return _finalise(candidates, cfg, clip_duration)


def _quiet_before(
    times: np.ndarray,
    dispersion: np.ndarray,
    anchor: float,
    fps: float,
    cfg: SegmentConfigSM,
) -> float:
    """Fallback start for an anchor with no burst.

    Walks back from the anchor to where dispersion was last at or below its
    local median -- the best available stand-in for the snap when the play was
    too short to produce a peak.
    """
    window = (times >= anchor - cfg.backtrack_s) & (times < anchor)
    values = dispersion[window]
    stamps = times[window]
    finite = np.isfinite(values)
    values, stamps = values[finite], stamps[finite]
    if len(values) < 4:
        return anchor - cfg.min_play_s
    level = float(np.median(values))
    for index in range(len(values) - 1, -1, -1):
        if values[index] <= level:
            return float(stamps[index])
    return float(stamps[0])


def _suppressed(anchor_id: int, whistle: Whistle, reason: str) -> Candidate:
    return Candidate(
        index=0, start=whistle.time, end=whistle.time,
        tier=Tier.SUPPRESSED, confidence=0.0,
        anchor_id=anchor_id, anchor_time=whistle.time,
        reasons=[reason], accepted=False,
    )


def _nearest_episode(
    episodes: list[Episode], anchor: float, window: float, used: set[int]
) -> tuple[int, Episode] | None:
    """Episode whose end sits closest before (or just after) *anchor*."""
    best: tuple[int, Episode] | None = None
    best_gap = window
    for position, episode in enumerate(episodes):
        if position in used:
            continue
        gap = abs(anchor - episode.end)
        if episode.start - 1.0 <= anchor and gap <= best_gap:
            best, best_gap = (position, episode), gap
    return best


def _finalise(
    candidates: list[Candidate], cfg: SegmentConfigSM, clip_duration: float
) -> list[Candidate]:
    """Enforce durations, mark partials and number the survivors."""
    output: list[Candidate] = []
    for candidate in candidates:
        if candidate.tier is not Tier.SUPPRESSED:
            if candidate.duration < cfg.min_play_s:
                candidate.tier = Tier.SUPPRESSED
                candidate.accepted = False
                candidate.reasons.append(
                    f"span {candidate.duration:.1f}s below min_play_s"
                )
            elif candidate.duration > cfg.max_play_s + 10.0:
                candidate.start = candidate.end - (cfg.max_play_s + 10.0)
                candidate.reasons.append("span clamped to max_play_s + buffers")
            candidate.partial = bool(
                candidate.start <= 0.05 or candidate.end >= clip_duration - 0.05
            )
        output.append(candidate)

    for candidate in output:
        if candidate.tier is not Tier.SUPPRESSED:
            # Blend the structural tier with duration plausibility so review is
            # ordered by the only signal measured to rank true plays highly.
            candidate.confidence = round(
                0.5 * CONFIDENCE[candidate.tier]
                + 0.5 * duration_confidence(candidate.duration),
                3,
            )
    for number, candidate in enumerate(
        [c for c in output if c.tier is not Tier.SUPPRESSED], start=1
    ):
        candidate.index = number
    return output


def write(path: Path, candidates: list[Candidate], *, clip: str, meta: dict) -> None:
    """Write segments.json. Hand-editable: review edits this file, then cut."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip": clip,
        "meta": meta,
        "segments": [
            {**asdict(candidate), "tier": candidate.tier.value} for candidate in candidates
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def read(path: Path) -> list[Candidate]:
    """Read segments.json back, preserving hand edits."""
    payload = json.loads(path.read_text())
    return [
        Candidate(
            index=row["index"], start=row["start"], end=row["end"],
            tier=Tier(row["tier"]), confidence=row["confidence"],
            anchor_id=row["anchor_id"], anchor_time=row["anchor_time"],
            reasons=row["reasons"], partial=row["partial"], accepted=row["accepted"],
            span_conf=row.get("span_conf", "high"),
        )
        for row in payload["segments"]
    ]
