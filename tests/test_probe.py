"""Clip ordering tests."""

from __future__ import annotations

from pathlib import Path

from playsplit.probe import find_clips, sort_key


def test_gopro_chapters_sort_by_session_then_chapter() -> None:
    """The real recording order, which filename order gets wrong.

    GH010007 sorts before GH020006 alphabetically but was recorded 35 minutes
    later; session must outrank chapter.
    """
    names = ["GH010007", "GH020006", "GH030006", "GH020007", "GH010008"]
    paths = [Path(f"{n}.MP4") for n in names]

    ordered = [p.stem for p in sorted(paths, key=sort_key)]

    assert ordered == ["GH020006", "GH030006", "GH010007", "GH020007", "GH010008"]


def test_gopro_prefixes_are_interchangeable() -> None:
    """GX (AVC) and GH (HEVC) share the chapter/session scheme."""
    paths = [Path("GX020001.MP4"), Path("GX010001.MP4")]

    assert [p.stem for p in sorted(paths, key=sort_key)] == ["GX010001", "GX020001"]


def test_non_gopro_names_sort_after_and_stay_deterministic(tmp_path: Path) -> None:
    gopro = tmp_path / "GH010001.MP4"
    other = tmp_path / "clip_a.MP4"
    for path in (gopro, other):
        path.write_bytes(b"")

    ordered = sorted([other, gopro], key=sort_key)

    assert ordered[0] == gopro


def test_find_clips_prefers_raw_subdir(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "GH010001.MP4").write_bytes(b"")
    (tmp_path / "GH010002.MP4").write_bytes(b"")

    assert [p.name for p in find_clips(tmp_path)] == ["GH010001.MP4"]


def test_find_clips_falls_back_to_flat_layout(tmp_path: Path) -> None:
    """Existing assets/<tournament>/<matchup>/ trees work without migration."""
    (tmp_path / "GH020001.MP4").write_bytes(b"")
    (tmp_path / "GH010001.MP4").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / ".DS_Store").write_bytes(b"")

    assert [p.name for p in find_clips(tmp_path)] == ["GH010001.MP4", "GH020001.MP4"]
