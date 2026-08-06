"""Evaluation metrics against hand-corrected labels."""

from __future__ import annotations

from dataclasses import dataclass

from .audio import Whistle
from .labels import Label

#: A play-end whistle may precede the labelled end by up to this much, because
#: the label is a *clip boundary* and the user pads the end past the whistle.
END_MATCH_BEFORE_S = 4.0
#: It may follow the labelled end only slightly -- a later whistle belongs to
#: the next play, or to the other pitch.
END_MATCH_AFTER_S = 0.5

#: A whistle is genuinely "mid-play" only if it is too late to be the previous
#: play's end and too early to be this one's.
MID_PLAY_AFTER_START_S = 2.0
MID_PLAY_BEFORE_END_S = END_MATCH_BEFORE_S


@dataclass
class WhistleMetrics:
    """Whistle detector performance against labelled play ends."""

    matched_plays: int
    scored_plays: int
    excluded_partial: int
    true_positives: int
    total_anchors: int
    mid_play_whistles: int
    plays_with_mid_play_whistle: int

    @property
    def recall(self) -> float:
        return self.matched_plays / self.scored_plays if self.scored_plays else float("nan")

    @property
    def precision(self) -> float:
        return self.true_positives / self.total_anchors if self.total_anchors else float("nan")


def _matches_end(whistle_time: float, label: Label) -> bool:
    return label.end - END_MATCH_BEFORE_S <= whistle_time <= label.end + END_MATCH_AFTER_S


def whistle_metrics(whistles: list[Whistle], labels: list[Label]) -> WhistleMetrics:
    """Score whistle anchors against labelled play ends.

    Plays truncated by the end of the recording are excluded from the recall
    denominator: no end whistle can exist for them, so counting them as misses
    would blame the detector for the camera stopping.
    """
    keeps = [label for label in labels if label.is_kept]
    scorable = [label for label in keeps if not label.partial]

    matched = sum(1 for label in scorable if any(_matches_end(w.time, label) for w in whistles))
    true_positives = sum(
        1 for w in whistles if any(_matches_end(w.time, label) for label in scorable)
    )

    mid_play = [
        (label, w.time)
        for label in keeps
        for w in whistles
        if label.start + MID_PLAY_AFTER_START_S
        <= w.time
        <= label.end - MID_PLAY_BEFORE_END_S
    ]

    return WhistleMetrics(
        matched_plays=matched,
        scored_plays=len(scorable),
        excluded_partial=len(keeps) - len(scorable),
        true_positives=true_positives,
        total_anchors=len(whistles),
        mid_play_whistles=len(mid_play),
        plays_with_mid_play_whistle=len({id(label) for label, _ in mid_play}),
    )
