"""Ground-truth label files.

Labels are bootstrapped from whistle anchors and then corrected by hand: fixing
timestamps against thumbnails is far cheaper than scrubbing raw video. The
corrected file is the tuning target for everything downstream.

The negative-control clip carries an intentionally empty label set -- the
pipeline must emit no plays from warm-up footage despite its whistles.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

FIELDNAMES = ["clip", "index", "start", "end", "verdict", "notes"]

#: Written into every bootstrap file so the correction pass is self-explanatory.
HEADER_NOTE = (
    "verdict: keep | drop | (blank = undecided). "
    "Correct start/end in seconds; delete rows that are not plays; "
    "add rows for plays the whistle detector missed."
)


@dataclass
class Label:
    """One ground-truth play interval."""

    clip: str
    index: int
    start: float
    end: float
    verdict: str = ""
    notes: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_kept(self) -> bool:
        """Rows are kept unless explicitly dropped, so blanks mean 'yes'."""
        return self.verdict.strip().lower() != "drop"


def write(path: Path, labels: list[Label], *, note: str = HEADER_NOTE) -> None:
    """Write a label CSV, with *note* as a leading comment line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write(f"# {note}\n")
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for label in labels:
            writer.writerow(
                {
                    "clip": label.clip,
                    "index": label.index,
                    "start": f"{label.start:.2f}",
                    "end": f"{label.end:.2f}",
                    "verdict": label.verdict,
                    "notes": label.notes,
                }
            )


def read(path: Path) -> list[Label]:
    """Read a label CSV, tolerating comment lines and blank rows."""
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        rows = [line for line in handle if not line.lstrip().startswith("#")]
    labels: list[Label] = []
    for row in csv.DictReader(rows):
        if not (row.get("start") or "").strip():
            continue
        labels.append(
            Label(
                clip=(row.get("clip") or "").strip(),
                index=int(row.get("index") or len(labels) + 1),
                start=float(row["start"]),
                end=float(row["end"]),
                verdict=(row.get("verdict") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
        )
    return labels


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
    for index, time in enumerate(whistle_times, start=1):
        start = max(previous_end, time - placeholder_lead_s, 0.0)
        end = min(time, clip_duration)
        if end <= start:
            continue
        labels.append(
            Label(
                clip=clip,
                index=index,
                start=start,
                end=end,
                notes=f"bootstrap: whistle @ {time:.2f}s",
            )
        )
        previous_end = end
    return labels
