"""Session-level segmentation: analyse per chapter, segment per session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import analyze, audio, segments
from .audio import Whistle
from .config import Config
from .session import Session
from .statemachine import Candidate, SegmentConfigSM, find_episodes

#: Median-filter widths. Episodes are detected on the coarse trace, where a
#: burst is unambiguous; the snap is localised on the fine one, because a
#: landmark blurred through a 1.8 s window drifts by seconds.
COARSE_WIDTH = 9
FINE_WIDTH = 3


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    padded = np.pad(values, (width // 2, width // 2), mode="edge")
    return np.nanmedian(
        np.lib.stride_tricks.sliding_window_view(padded, width), axis=1
    )


@dataclass
class SessionAnalysis:
    """A session's features and anchors on one continuous timeline."""

    session: Session
    times: np.ndarray
    coarse: np.ndarray
    fine: np.ndarray
    whistles: list[Whistle]
    realtime: float


def analyse_session(
    session: Session, cfg: Config, *, force: bool = False, log=print
) -> SessionAnalysis:
    """Concatenate per-chapter features onto the session timeline.

    Each chapter is analysed and cached independently -- the expensive YOLO
    pass is untouched and its cache keys are unchanged. Only the assembled
    arrays are new.
    """
    times: list[np.ndarray] = []
    dispersion: list[np.ndarray] = []
    whistles: list[Whistle] = []
    factors: list[float] = []

    for chapter in session.chapters:
        analysis_dir = chapter.info.path.parent / "analysis"
        rows, realtime = analyze.features(
            chapter.info, analysis_dir, cfg, force=force, log=log
        )
        factors.append(realtime)
        if not rows:
            continue
        times.append(np.array([r.time for r in rows]) + chapter.offset)
        dispersion.append(np.array([r.dispersion for r in rows]))

        signal = audio.load_audio(
            chapter.info.path,
            cfg.audio.sample_rate,
            analysis_dir / f"{chapter.info.path.stem}.wav",
        )
        whistles.extend(
            Whistle(w.start + chapter.offset, w.end + chapter.offset, w.strength)
            for w in audio.detect(signal, cfg.audio)
        )

    if not times:
        empty = np.array([])
        return SessionAnalysis(session, empty, empty, empty, [], 0.0)

    stacked_times = np.concatenate(times)
    stacked_disp = np.concatenate(dispersion)
    return SessionAnalysis(
        session=session,
        times=stacked_times,
        coarse=_smooth(stacked_disp, COARSE_WIDTH),
        fine=_smooth(stacked_disp, FINE_WIDTH),
        whistles=sorted(whistles, key=lambda w: w.time),
        realtime=float(np.mean(factors)) if factors else 0.0,
    )


def segment_session(
    analysis: SessionAnalysis, cfg: Config, sm: SegmentConfigSM | None = None
) -> list[Candidate]:
    """Run episode finding and tiering once over the whole session."""
    sm = sm or SegmentConfigSM()
    if analysis.times.size == 0:
        return []

    ignore: list[list[float]] = []
    for chapter in analysis.session.chapters:
        for low, high in cfg.ignore_ranges.get(chapter.info.name, []):
            ignore.append([low + chapter.offset, high + chapter.offset])

    episodes, _ = find_episodes(
        analysis.times, analysis.coarse, cfg.analysis.fps, sm
    )
    return segments.build(
        episodes,
        analysis.whistles,
        analysis.times,
        analysis.coarse,
        cfg.analysis.fps,
        sm,
        fine=analysis.fine,
        pre_buffer_s=cfg.segment.pre_buffer_s,
        post_buffer_s=cfg.segment.post_buffer_s,
        clip_duration=analysis.session.duration,
        ignore_ranges=ignore,
    )
