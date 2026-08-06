"""Cutting, naming and manifest tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from playsplit.cut import Cut, MANIFEST_FIELDS, keyframe_before, timecode, write_manifest
from playsplit.probe import ClipInfo

CLIP = ClipInfo(
    path=Path("/games/g1/GH010007.MP4"), duration=531.54,
    width=1920, height=1440, fps=59.94, has_audio=True,
)


def test_filename_is_sortable_and_traceable() -> None:
    assert Cut(7, CLIP, 134.2, 146.0).filename == "P07__GH010007__t0134.mp4"


def test_filename_pads_play_numbers_for_sorting() -> None:
    names = [Cut(n, CLIP, 10.0, 20.0).filename for n in (2, 10)]
    assert names == sorted(names)


@pytest.mark.parametrize(
    "seconds,expected",
    [(0.0, "00:00:00.000"), (82.082, "00:01:22.082"), (3661.5, "01:01:01.500")],
)
def test_timecode(seconds: float, expected: str) -> None:
    assert timecode(seconds) == expected


def test_manifest_records_the_actual_keyframe_start(tmp_path: Path) -> None:
    """The manifest must report where the file really begins, not the request.

    Stream copy snaps the start back to a keyframe, so the requested start and
    the delivered one differ. Recording the request would make every timecode
    in the manifest wrong by up to a second.
    """
    item = Cut(1, CLIP, start=13.0, end=21.0)
    path = tmp_path / "manifest.csv"

    write_manifest(path, [item], starts={1: 12.012})

    row = next(csv.DictReader(path.read_text().splitlines()))
    assert row["start"] == "12.012"
    assert row["duration"] == "8.988"
    assert row["start_tc"] == "00:00:12.012"


def test_manifest_has_the_documented_columns(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(path, [Cut(1, CLIP, 10.0, 20.0)], starts={})

    assert next(csv.reader(path.read_text().splitlines())) == MANIFEST_FIELDS


def test_partial_flag_is_written(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(path, [Cut(1, CLIP, 520.0, 531.54, partial=True)], starts={})

    assert next(csv.DictReader(path.read_text().splitlines()))["partial"] == "true"


def test_keyframe_before_never_runs_past_the_request(tmp_path: Path) -> None:
    """Regression: a misread ffprobe field made this return the window floor.

    ffprobe emits frame fields in its own fixed order, not the order requested,
    so parsing ``pts_time,key_frame`` positionally yielded no keyframes at all
    and the fallback silently stretched an 8s play to 20s. Any real clip must
    return a keyframe within one GOP of the request, never the floor.
    """
    source = Path(
        "assets/Bayernliga_250726_Oberam/RenegadesVsAnts/GH010007.MP4"
    )
    if not source.is_file():
        pytest.skip("sample footage not present in this checkout")

    found = keyframe_before(source, 13.0, search_s=12.0)

    assert found <= 13.0
    assert 13.0 - found <= 1.5, "fell back to the window floor instead of a keyframe"
