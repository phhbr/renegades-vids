"""Run the analysis pipeline over every clip in every game folder.

This is the generalization test. Every threshold in the pipeline was fitted to
one clip from one game, so the questions this answers are: does the play band
get found on other camera placements, does the field mask survive different
backgrounds, and is the whistle rate plausible everywhere.

Each clip is isolated -- a chapter that is pure halftime may legitimately have
no motion band, and that must be reported rather than abort the sweep.

Usage: uv run python scripts/sweep.py [--limit N]
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from playsplit import analyze, audio, labels, probe, segments
from playsplit.config import load
from playsplit.statemachine import SegmentConfigSM, Tier, find_episodes

ROOT = Path("assets/Bayernliga_250726_Oberam")
OUT = Path("analysis/sweep.csv")


@dataclass
class Row:
    game: str
    clip: str
    duration_s: float = 0.0
    band_top: int = 0
    band_bottom: int = 0
    band_height: int = 0
    mask_pct: float = 0.0
    detections: int = 0
    on_field_pct: float = 0.0
    median_cluster: float = 0.0
    median_dispersion: float = 0.0
    whistles: int = 0
    whistles_per_min: float = 0.0
    candidates: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    realtime: float = 0.0
    elapsed_s: float = 0.0
    status: str = "ok"
    note: str = ""


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    padded = np.pad(values, (width // 2, width // 2), mode="edge")
    return np.nanmedian(
        np.lib.stride_tricks.sliding_window_view(padded, width), axis=1
    )


def process(game_dir: Path, clip_path: Path) -> Row:
    row = Row(game=game_dir.name, clip=clip_path.name)
    started = time.time()
    try:
        cfg = load(game_dir)
        info = probe.probe(clip_path)
        row.duration_s = round(info.duration, 1)
        analysis_dir = game_dir / "analysis"

        features, realtime = analyze.features(
            info, analysis_dir, cfg, log=lambda _: None
        )
        row.realtime = round(realtime, 1)
        if not features:
            row.status = "no-features"
            return row

        raw = analyze.detections(info, analysis_dir, cfg, log=lambda _: None)
        mask = analyze._mask(info, analysis_dir, False, lambda _: None)
        band = analyze._band(info, analysis_dir, cfg, False, lambda _: None)
        row.band_top, row.band_bottom = band.top, band.bottom
        row.band_height = band.bottom - band.top
        row.mask_pct = round(mask.area_fraction * 100, 1)
        row.detections = len(raw.xs)
        if len(raw.xs):
            row.on_field_pct = round(
                float(mask.contains_many(raw.xs, raw.ys).mean()) * 100, 1
            )

        counts = np.array([f.count for f in features], dtype=float)
        dispersion = np.array([f.dispersion for f in features])
        row.median_cluster = round(float(np.nanmedian(counts[counts > 0])), 1)
        row.median_dispersion = round(float(np.nanmedian(dispersion)), 1)

        signal = audio.load_audio(
            info.path, cfg.audio.sample_rate, analysis_dir / f"{clip_path.stem}.wav"
        )
        whistles = audio.detect(signal, cfg.audio)
        row.whistles = len(whistles)
        row.whistles_per_min = round(len(whistles) / (info.duration / 60), 1)

        times = np.array([f.time for f in features])
        coarse, fine = smooth(dispersion, 9), smooth(dispersion, 3)
        sm = SegmentConfigSM()
        episodes, _ = find_episodes(times, coarse, cfg.analysis.fps, sm)
        candidates = segments.build(
            episodes, whistles, times, coarse, cfg.analysis.fps, sm, fine=fine,
            pre_buffer_s=cfg.segment.pre_buffer_s,
            post_buffer_s=cfg.segment.post_buffer_s,
            clip_duration=info.duration,
            ignore_ranges=cfg.ignore_ranges.get(clip_path.name, []),
        )
        segments.write(
            analysis_dir / f"{clip_path.stem}__segments.json", candidates,
            clip=clip_path.name,
            meta={"anchors": len(whistles), "episodes": len(episodes)},
        )
        row.candidates = sum(1 for c in candidates if c.tier is not Tier.SUPPRESSED)
        row.high = sum(1 for c in candidates if c.tier is Tier.HIGH)
        row.medium = sum(1 for c in candidates if c.tier is Tier.MEDIUM)
        row.low = sum(1 for c in candidates if c.tier is Tier.LOW)

    except Exception as exc:  # one bad chapter must not stop the sweep
        row.status = "error"
        row.note = f"{type(exc).__name__}: {exc}"[:200]
        traceback.print_exc(file=sys.stderr)
    finally:
        row.elapsed_s = round(time.time() - started, 1)
    return row


def main() -> None:
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    jobs: list[tuple[Path, Path]] = []
    for game_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for clip_path in probe.find_clips(game_dir):
            jobs.append((game_dir, clip_path))
    if limit:
        jobs = jobs[:limit]

    rows: list[Row] = []
    for index, (game_dir, clip_path) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {game_dir.name}/{clip_path.name}", flush=True)
        row = process(game_dir, clip_path)
        rows.append(row)
        print(
            f"    {row.status} band={row.band_top}-{row.band_bottom} "
            f"mask={row.mask_pct}% whistles={row.whistles} "
            f"cands={row.candidates} {row.elapsed_s}s",
            flush=True,
        )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
            writer.writeheader()
            for item in rows:
                writer.writerow(asdict(item))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
