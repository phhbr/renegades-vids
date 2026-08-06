"""P2 checkpoint figure -- detection-cluster features against ground truth.

Renders analysis/features.png: the labelled clip on top, the negative control
below. Kept plays are shaded so the contraction -> spike -> decay shape is
judged against labels rather than eyeballs; unresolved CHECK spans are hatched.

Usage: uv run python scripts/feature_report.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from playsplit import analyze, audio, labels, probe
from playsplit.cluster import spread_rate
from playsplit.config import load

ROOT = Path("assets/Bayernliga_250726_Oberam")
OUT = Path("analysis/features.png")

CLIPS = [
    (ROOT / "RenegadesVsAnts", "GH010007.MP4", "GH010007__labels_corrected.csv",
     "RenegadesVsAnts / GH010007 — labelled (12 keeps)"),
    (ROOT / "SpatzenVsAnts", "GH020002.MP4", "GH020002__labels.csv",
     "SpatzenVsAnts / GH020002 — negative control (0 plays expected)"),
]


def _load(game_dir: Path, clip_name: str, label_name: str):
    cfg = load(game_dir)
    info = probe.probe(next(p for p in probe.find_clips(game_dir) if p.name == clip_name))
    analysis_dir = game_dir / "analysis"
    features, realtime = analyze.features(info, analysis_dir, cfg, log=lambda _: None)
    rows, _ = labels.read(analysis_dir / label_name, clip_duration=info.duration)
    signal = audio.load_audio(
        info.path, cfg.audio.sample_rate, analysis_dir / f"{info.path.stem}.wav"
    )
    whistles = audio.detect(signal, cfg.audio)
    return info, features, rows, whistles, realtime


fig, axes = plt.subplots(4, 2, figsize=(19, 13), sharex="col")
fig.suptitle(
    "playsplit P2 — dominant-cluster features vs ground truth",
    fontsize=15, fontweight="bold",
)

for column, (game_dir, clip_name, label_name, title) in enumerate(CLIPS):
    info, features, rows, whistles, realtime = _load(game_dir, clip_name, label_name)

    times = np.array([f.time for f in features])
    counts = np.array([float(f.count) for f in features])
    dispersion = np.array([f.dispersion for f in features])
    centroid = np.array([f.centroid_x for f in features])
    rate = spread_rate(dispersion, fps=5.0)

    keeps = [r for r in rows if r.is_kept]
    checks = [r for r in rows if r.is_unresolved]

    def decorate(ax, *, legend: bool = False) -> None:
        for index, row in enumerate(keeps):
            ax.axvspan(row.start, row.end, color="#22c55e", alpha=0.20,
                       label="labelled play" if index == 0 and legend else None)
        for index, row in enumerate(checks):
            ax.axvspan(row.start, row.end, facecolor="none", edgecolor="#f97316",
                       hatch="///", linewidth=1.0,
                       label="CHECK (unresolved)" if index == 0 and legend else None)
        for index, whistle in enumerate(whistles):
            ax.axvline(whistle.time, color="#eab308", lw=0.8, alpha=0.85,
                       label="whistle anchor" if index == 0 and legend else None)
        ax.grid(alpha=0.2)
        ax.margins(x=0.01)

    ax = axes[0][column]
    ax.plot(times, counts, color="#0f172a", lw=0.8)
    decorate(ax, legend=True)
    ax.set_ylabel("cluster size")
    ax.set_title(f"{title}\n{realtime:.1f}x realtime", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper right")

    ax = axes[1][column]
    ax.plot(times, dispersion, color="#2563eb", lw=0.9)
    decorate(ax)
    ax.set_ylabel("x-dispersion (px)")

    ax = axes[2][column]
    ax.plot(times, rate, color="#7c3aed", lw=0.8)
    ax.axhline(0, color="#888", lw=0.6)
    decorate(ax)
    ax.set_ylabel("spread rate (px/s)")

    ax = axes[3][column]
    ax.plot(times, centroid, color="#0891b2", lw=0.9)
    decorate(ax)
    ax.set_ylabel("cluster centroid x (px)")
    ax.set_xlabel("time (s)")

fig.tight_layout(rect=(0, 0, 1, 0.96))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=110)
print(f"wrote {OUT}")
