"""Sessions: the continuous recordings that GoPro chapters actually belong to.

A GoPro splits a recording into ~4 GB chapters, so ``GH010007`` and
``GH020007`` are one unbroken take cut at an arbitrary byte offset -- in this
corpus, mid-play. Segmenting per file therefore invents a play boundary where
none exists: the play that snaps at 530 s of chapter 01 is live 0.2 s into
chapter 02, and per-file analysis emits it as two truncated halves, both
flagged ``partial``.

So analysis and caching stay per file (nothing is re-decoded, cache keys are
unchanged), but the *segmentation input* is a session-level virtual timeline:
each chapter's feature times, whistle anchors and detections are shifted by the
cumulative duration of the chapters before it, then the state machine runs once
over the join. ``partial`` then means what it should -- truncated by the
session ending, not by a chapter ending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .probe import ClipInfo, _GOPRO_RE, find_clips, probe


@dataclass
class Chapter:
    """One file within a session, placed on the session timeline."""

    info: ClipInfo
    number: int
    #: Seconds of preceding chapters; add to a local time to get session time.
    offset: float

    @property
    def start(self) -> float:
        return self.offset

    @property
    def end(self) -> float:
        return self.offset + self.info.duration

    def to_session(self, local: float) -> float:
        return local + self.offset

    def to_local(self, session_time: float) -> float:
        return session_time - self.offset


@dataclass
class Session:
    """A continuous recording, reassembled from its chapters."""

    game: str
    session_id: str
    chapters: list[Chapter]
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(chapter.info.duration for chapter in self.chapters)

    @property
    def name(self) -> str:
        return f"{self.game}/{self.session_id}"

    def locate(self, session_time: float) -> tuple[Chapter, float]:
        """Map a session time back to (chapter, time within that chapter)."""
        for chapter in self.chapters:
            if session_time < chapter.end or chapter is self.chapters[-1]:
                return chapter, max(0.0, session_time - chapter.offset)
        raise ValueError(f"{session_time} lies outside {self.name}")

    def spans_boundary(self, start: float, end: float) -> bool:
        """True when a segment crosses from one chapter into the next."""
        return self.locate(start)[0] is not self.locate(end - 1e-6)[0]


def plan_sessions(paths: list[Path]) -> list[tuple[str, list[tuple[int, Path]], list[str]]]:
    """Group paths into (session id, [(chapter, path)], warnings), in order.

    Pure: no probing, so chapter bookkeeping can be tested without footage.
    Non-GoPro filenames each become their own single-chapter session -- without
    the naming convention there is no evidence that two files are continuous,
    and wrongly joining them is worse than wrongly splitting them.
    """
    buckets: dict[str, list[tuple[int, Path]]] = {}
    order: list[str] = []

    for path in paths:
        match = _GOPRO_RE.match(path.stem.upper())
        key = match["session"] if match else path.stem
        number = int(match["chapter"]) if match else 1
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append((number, path))

    planned = []
    for key in order:
        entries = sorted(buckets[key])
        numbers = [number for number, _ in entries]
        warnings: list[str] = []
        if numbers[0] != 1:
            # Informational, not an error: trimming the opening chapter as
            # pre-game is normal practice, and it is how sessions 0006 and 0002
            # in this corpus came to start at chapter 02. Worth surfacing only
            # so a genuinely lost chapter is not mistaken for missed plays.
            warnings.append(
                f"session {key} starts at chapter {numbers[0]:02d}: opening "
                f"{numbers[0] - 1} chapter(s) absent (usually pre-game trim)"
            )
        if any(b != a + 1 for a, b in zip(numbers, numbers[1:])):
            warnings.append(
                f"session {key} has non-contiguous chapters {numbers}: "
                "a play spanning the gap cannot be reassembled"
            )
        planned.append((key, entries, warnings))
    return planned


def group_sessions(game_dir: Path) -> list[Session]:
    """Group a game folder's clips into sessions, in recording order."""
    sessions: list[Session] = []
    for key, entries, warnings in plan_sessions(find_clips(game_dir)):
        chapters: list[Chapter] = []
        offset = 0.0
        for number, path in entries:
            info = probe(path)
            chapters.append(Chapter(info=info, number=number, offset=offset))
            offset += info.duration
        sessions.append(Session(game_dir.name, key, chapters, warnings))
    return sessions
