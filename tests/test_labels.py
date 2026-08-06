"""Label file validation tests.

Each case here corresponds to a real defect that arrived in the first
hand-corrected pass, or to a rule the eval harness depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from playsplit.labels import Label, LabelError, read, write


def _csv(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "labels.csv"
    path.write_text(
        "# note\nclip,index,start,end,verdict,partial,notes\n" + body
    )
    return path


def test_blank_verdict_is_rejected(tmp_path: Path) -> None:
    """'blank' meant keep in one place and undecided in another; ban it."""
    path = _csv(tmp_path, "c.MP4,1,10,20,,,\n")

    with pytest.raises(LabelError, match="not one of"):
        read(path)


def test_zero_length_row_is_rejected(tmp_path: Path) -> None:
    path = _csv(tmp_path, "c.MP4,1,36.83,36.83,keep,,\n")

    with pytest.raises(LabelError, match="start .* >= end"):
        read(path)


def test_start_after_end_is_rejected(tmp_path: Path) -> None:
    path = _csv(tmp_path, "c.MP4,1,520,518.91,keep,,\n")

    with pytest.raises(LabelError, match="start .* >= end"):
        read(path)


def test_overlapping_keeps_are_rejected(tmp_path: Path) -> None:
    path = _csv(tmp_path, "c.MP4,1,10,25,keep,,\nc.MP4,2,20,30,keep,,\n")

    with pytest.raises(LabelError, match="overlap"):
        read(path)


def test_overlapping_drops_are_allowed(tmp_path: Path) -> None:
    """Drops mark dead time and may overlap anything, including keeps."""
    path = _csv(tmp_path, "c.MP4,1,10,25,drop,,\nc.MP4,2,20,30,keep,,\n")

    rows, _ = read(path)

    assert len(rows) == 2


@pytest.mark.parametrize("start,end", [(100, 145), (100, 101.5)])
def test_implausible_keep_duration_warns_but_loads(
    tmp_path: Path, start: float, end: float
) -> None:
    """Too long or too short is a warning, never a hard error -- a 3.4 s play
    is real, and only a human can tell a long play from two merged ones."""
    path = _csv(tmp_path, f"c.MP4,1,{start},{end},keep,,\n")

    rows, warnings = read(path)

    assert len(rows) == 1
    assert any("outside plausible" in w for w in warnings)


def test_observed_play_durations_do_not_warn(tmp_path: Path) -> None:
    """The real keeps span 3.4-23.3 s; none of that range may be noisy."""
    path = _csv(tmp_path, "c.MP4,1,225.1,228.5,keep,,\nc.MP4,2,358,381.33,keep,,\n")

    _, warnings = read(path)

    assert not [w for w in warnings if "outside plausible" in w]


def test_end_touching_clip_boundary_snaps_and_marks_partial(tmp_path: Path) -> None:
    """A clip that stops mid-play still contains a real, cuttable play."""
    path = _csv(tmp_path, "c.MP4,1,520,531.00,keep,,\n")

    rows, warnings = read(path, clip_duration=531.541333)

    assert rows[0].end == pytest.approx(531.541333)
    assert rows[0].partial is True
    assert any("partial" in w for w in warnings)


def test_start_touching_clip_boundary_marks_partial(tmp_path: Path) -> None:
    path = _csv(tmp_path, "c.MP4,1,0.4,12,keep,,\n")

    rows, _ = read(path, clip_duration=531.5)

    assert rows[0].start == 0.0
    assert rows[0].partial is True


def test_check_rows_are_unresolved_and_not_kept(tmp_path: Path) -> None:
    path = _csv(tmp_path, "c.MP4,1,297.82,336,check,,\n")

    rows, _ = read(path)

    assert rows[0].is_unresolved
    assert not rows[0].is_kept


def test_write_reindexes_sequentially(tmp_path: Path) -> None:
    """The first correction pass arrived with duplicated indices."""
    path = tmp_path / "out.csv"
    write(
        path,
        [
            Label("c.MP4", 8, 10, 20, "keep"),
            Label("c.MP4", 8, 30, 40, "keep"),
            Label("c.MP4", 12, 50, 60, "drop"),
        ],
    )

    rows, _ = read(path)

    assert [r.index for r in rows] == [1, 2, 3]


def test_partial_flag_survives_a_write_read_cycle(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    write(path, [Label("c.MP4", 1, 520, 531.54, "keep", partial=True)])

    rows, _ = read(path)

    assert rows[0].partial is True


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert read(tmp_path / "absent.csv") == ([], [])
