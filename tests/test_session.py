"""Session grouping and chapter-boundary handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from playsplit.session import Chapter, Session, plan_sessions
from playsplit.probe import ClipInfo  # noqa: E402


def info(name: str, duration: float = 531.5) -> ClipInfo:
    return ClipInfo(
        path=Path(f"/g/{name}.MP4"), duration=duration,
        width=1920, height=1440, fps=59.94, has_audio=True,
    )


def session(*durations: float) -> Session:
    chapters, offset = [], 0.0
    for index, duration in enumerate(durations, start=1):
        chapters.append(Chapter(info(f"GH{index:02d}0007", duration), index, offset))
        offset += duration
    return Session("game", "0007", chapters)


def test_offsets_place_chapters_end_to_end() -> None:
    assert [c.offset for c in session(531.5, 416.8).chapters] == [0.0, 531.5]


def test_session_time_round_trips() -> None:
    chapter = session(531.5, 416.8).chapters[1]
    assert chapter.to_local(chapter.to_session(10.0)) == pytest.approx(10.0)


def test_locate_maps_back_to_the_right_chapter() -> None:
    sess = session(531.5, 416.8)
    chapter, local = sess.locate(535.0)
    assert chapter.number == 2
    assert local == pytest.approx(3.5)


def test_spans_boundary_detects_a_play_across_the_seam() -> None:
    """The real case: a play snapping at 530s is live in the next chapter."""
    sess = session(531.5, 416.8)
    assert sess.spans_boundary(517.4, 537.7)
    assert not sess.spans_boundary(480.0, 500.0)


def test_missing_first_chapter_warns() -> None:
    plan = plan_sessions([Path("GH020006.MP4"), Path("GH030006.MP4")])
    assert "starts at chapter 02" in plan[0][2][0]


def test_contiguous_chapters_do_not_warn() -> None:
    assert plan_sessions([Path("GH010007.MP4"), Path("GH020007.MP4")])[0][2] == []


def test_chapter_gap_warns() -> None:
    warnings = plan_sessions([Path("GH010007.MP4"), Path("GH030007.MP4")])[0][2]
    assert any("non-contiguous" in w for w in warnings)


def test_sessions_are_separated_by_session_id() -> None:
    plan = plan_sessions(
        [Path("GH010007.MP4"), Path("GH020007.MP4"), Path("GH010008.MP4")]
    )
    assert [key for key, _, _ in plan] == ["0007", "0008"]
    assert [len(entries) for _, entries, _ in plan] == [2, 1]


def test_non_gopro_files_are_never_joined() -> None:
    """Without the naming convention there is no evidence of continuity."""
    assert len(plan_sessions([Path("clip_a.MP4"), Path("clip_b.MP4")])) == 2


def test_real_corpus_chapter_layout() -> None:
    """Pins the actual footage: two sessions are missing their first chapter."""
    names = [
        "GH020006", "GH030006", "GH010007", "GH020007",
        "GH010008", "GH020008", "GH030008",
    ]
    plan = plan_sessions([Path(f"{n}.MP4") for n in names])

    assert [key for key, _, _ in plan] == ["0006", "0007", "0008"]
    assert plan[0][2] and "starts at chapter 02" in plan[0][2][0]
    assert plan[1][2] == [] and plan[2][2] == []
