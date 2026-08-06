"""Ground-truth label files.

Labels are bootstrapped from whistle anchors and then corrected by hand: fixing
timestamps against thumbnails is far cheaper than scrubbing raw video. The
corrected file is the tuning target for everything downstream.

Labels record **desired final clip boundaries**, not raw snap/whistle instants
-- end lands a beat after the whistle, start just before the snap. The pipeline
is therefore evaluated on its *post-buffer* output, with the pre/post buffers
fitted as free parameters. Comparing pre-buffer segments against these labels
would report a systematic offset that is not a detection error.

Every hand-edited file is validated on load. The first correction pass arrived
with a zero-length row, duplicated indices and a ``start > end`` row; those are
mechanical mistakes that a human should never have to catch twice.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

FIELDNAMES = ["clip", "index", "start", "end", "verdict", "partial", "notes"]

VERDICTS = {"keep", "drop", "check"}

#: Written into every generated file so the correction pass is self-explanatory.
#: Deliberately offers no blank option -- an empty verdict used to mean "keep"
#: in one place and "undecided" in another, so the loader now rejects it.
HEADER_NOTE = (
    "verdict: keep | drop | check (required, no blanks). "
    "start/end in seconds = desired final clip boundaries "
    "(end ~ whistle + 0-2s; start ~ just before the snap). "
    "Delete rows that are not plays; add rows for plays the detector missed."
)

#: Durations outside this range are surfaced as warnings, not errors -- a 3.4 s
#: play is real, but a 38 s one is usually two plays merged.
PLAUSIBLE_DURATION_S = (2.0, 40.0)


class LabelError(ValueError):
    """A hand-edited label file is malformed in a way tuning must not absorb."""


@dataclass
class Label:
    """One ground-truth play interval."""

    clip: str
    index: int
    start: float
    end: float
    verdict: str
    partial: bool = False
    notes: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_kept(self) -> bool:
        return self.verdict == "keep"

    @property
    def is_unresolved(self) -> bool:
        """CHECK rows are excluded from tuning and metrics until resolved."""
        return self.verdict == "check"


def write(path: Path, labels: list[Label], *, note: str = HEADER_NOTE) -> None:
    """Write a label CSV, re-indexing sequentially."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write(f"# {note}\n")
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index, label in enumerate(labels, start=1):
            writer.writerow(
                {
                    "clip": label.clip,
                    "index": index,
                    "start": f"{label.start:.2f}",
                    "end": f"{label.end:.2f}",
                    "verdict": label.verdict,
                    "partial": "true" if label.partial else "",
                    "notes": label.notes,
                }
            )


#: A label reaching within this many seconds of a clip boundary is treated as
#: truncated by the recording rather than by the referee. Hand-labelled ends
#: land on round numbers ("531"), so an exact-overrun test would miss them.
BOUNDARY_TOLERANCE_S = 1.0


def read(
    path: Path,
    *,
    clip_duration: float | None = None,
    strict: bool = True,
) -> tuple[list[Label], list[str]]:
    """Read and validate a label CSV.

    Returns the labels and a list of non-fatal warnings. Raises
    :class:`LabelError` on structural problems when *strict*. When
    *clip_duration* is given, labels touching either clip boundary are snapped
    to it and flagged ``partial`` -- a clip that starts or stops mid-play still
    contains a real play, and it must be cut and manifested as one.
    """
    if not path.is_file():
        return [], []

    with path.open(newline="") as handle:
        rows = [line for line in handle if not line.lstrip().startswith("#")]

    labels: list[Label] = []
    problems: list[str] = []
    warnings: list[str] = []

    for line_number, row in enumerate(csv.DictReader(rows), start=2):
        if not (row.get("start") or "").strip():
            continue
        raw_verdict = (row.get("verdict") or "").strip().lower()
        if raw_verdict not in VERDICTS:
            problems.append(
                f"line {line_number}: verdict {raw_verdict or '(blank)'!r} "
                f"is not one of {sorted(VERDICTS)}"
            )
            continue

        start, end = float(row["start"]), float(row["end"])
        if start >= end:
            problems.append(f"line {line_number}: start {start} >= end {end}")
            continue

        partial = (row.get("partial") or "").strip().lower() in {"true", "yes", "1"}
        if clip_duration is not None:
            if end >= clip_duration - BOUNDARY_TOLERANCE_S:
                warnings.append(
                    f"row {len(labels) + 1}: end {end:.2f}s reaches the clip end "
                    f"({clip_duration:.2f}s); snapped and marked partial"
                )
                end, partial = clip_duration, True
            if start <= BOUNDARY_TOLERANCE_S:
                warnings.append(
                    f"row {len(labels) + 1}: start {start:.2f}s reaches the clip "
                    "start; snapped and marked partial"
                )
                start, partial = 0.0, True

        labels.append(
            Label(
                clip=(row.get("clip") or "").strip(),
                index=len(labels) + 1,
                start=start,
                end=end,
                verdict=raw_verdict,
                partial=partial,
                notes=(row.get("notes") or "").strip(),
            )
        )

    warnings.extend(_check_durations(labels))
    problems.extend(_check_keep_overlaps(labels))

    if problems and strict:
        raise LabelError(f"{path.name}: " + "; ".join(problems))
    warnings.extend(problems)
    return labels, warnings


def _check_durations(labels: list[Label]) -> list[str]:
    low, high = PLAUSIBLE_DURATION_S
    return [
        f"row {label.index} ({label.verdict}): duration {label.duration:.1f}s "
        f"outside plausible {low:g}-{high:g}s"
        for label in labels
        if label.is_kept and not (low <= label.duration <= high)
    ]


def _check_keep_overlaps(labels: list[Label]) -> list[str]:
    """Kept intervals must not overlap; drops may overlap anything."""
    keeps = sorted((l for l in labels if l.is_kept), key=lambda l: l.start)
    return [
        f"rows {a.index} and {b.index}: kept intervals overlap "
        f"({a.start:.2f}-{a.end:.2f} vs {b.start:.2f}-{b.end:.2f})"
        for a, b in zip(keeps, keeps[1:])
        if b.start < a.end
    ]


def bootstrap_from_whistles(
    clip: str,
    whistle_times: list[float],
    *,
    placeholder_lead_s: float,
    clip_duration: float,
) -> list[Label]:
    """Build first-pass labels: end at the whistle, start a fixed lead earlier.

    The start is a deliberate placeholder -- detection features have not been
    built yet, so there is nothing better to backtrack with. Correcting a
    consistently-wrong start is quicker than correcting an erratic one.
    """
    labels: list[Label] = []
    previous_end = 0.0
    for time in whistle_times:
        start = max(previous_end, time - placeholder_lead_s, 0.0)
        end = min(time, clip_duration)
        if end <= start:
            continue
        labels.append(
            Label(
                clip=clip,
                index=len(labels) + 1,
                start=start,
                end=end,
                verdict="check",
                notes=f"bootstrap: whistle @ {time:.2f}s",
            )
        )
        previous_end = end
    return labels
