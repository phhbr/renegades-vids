"""Guard: hand-corrected ground truth must stay under version control.

This exists because it already failed once. The ``assets`` ignore rule covered
the label files, so several correction passes lived only on one machine. Ignore
rules regress silently -- a broad pattern added later re-swallows them and
nothing complains until the labels are gone. Compute can regenerate every other
artifact in this repo; it cannot regenerate the user's labelling time.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_GLOB = "*__labels_corrected.csv"


def _label_files() -> list[Path]:
    return sorted(REPO_ROOT.glob(f"assets/**/{LABEL_GLOB}"))


def test_corrected_label_files_are_not_ignored() -> None:
    """Every corrected label file on disk must be tracked by git."""
    labels = _label_files()
    if not labels:
        pytest.skip("no corrected label files present in this checkout")

    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", *[str(p) for p in labels]],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout.split()

    assert not ignored, (
        "these ground-truth files are gitignored and would be lost: "
        + ", ".join(ignored)
    )


def test_corrected_label_files_are_committed() -> None:
    """Not merely un-ignored -- actually present in the index."""
    labels = _label_files()
    if not labels:
        pytest.skip("no corrected label files present in this checkout")

    tracked = set(
        subprocess.run(
            ["git", "ls-files", "assets"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    )

    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in labels
        if str(path.relative_to(REPO_ROOT)) not in tracked
    ]
    assert not missing, f"untracked ground truth: {missing}"


def test_footage_is_not_tracked() -> None:
    """The un-ignore rule must not have widened into the 45 GB of video."""
    tracked = subprocess.run(
        ["git", "ls-files", "assets"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()

    heavy = [name for name in tracked if name.lower().endswith((".mp4", ".mov", ".npz", ".wav"))]
    assert not heavy, f"binary artifacts are tracked: {heavy[:5]}"
