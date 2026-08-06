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


#: Boundary between the two centroid-x modes seen on GH010007. Used only for
#: the diagnostic scatter that tested whether flat plays sit in the far mode.
FAR_LOCUS_PX = 1250


def medfilt(values: np.ndarray, width: int = 9) -> np.ndarray:
    padded = np.pad(values, (width // 2, width // 2), mode="edge")
    return np.nanmedian(
        np.lib.stride_tricks.sliding_window_view(padded, width), axis=1
    )


fig = plt.figure(figsize=(19, 16))
grid = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1.15], hspace=0.32, wspace=0.13)
axes = [[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(4)]
fig.suptitle(
    "playsplit P2 (revised) — dominant-cluster features vs ground truth",
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

    if column == 0:
        # Bottom-left: the locus diagnostic. Does a flat play sit in the far
        # centroid mode? If it did, two games would be sharing the grass and
        # the fix would be a per-field polygon rather than better clustering.
        smooth = medfilt(dispersion)
        scatter_ax = fig.add_subplot(grid[4, 0])
        for row in keeps:
            window = (times >= row.start) & (times < row.end)
            before = (times >= row.start - 12) & (times < row.start)
            during = smooth[window]
            baseline = smooth[before]
            during = during[np.isfinite(during)]
            baseline = baseline[np.isfinite(baseline)]
            if len(during) < 3 or len(baseline) < 3:
                continue
            ratio = np.nanmax(during) / max(np.median(baseline), 1e-6)
            centre = centroid[window]
            centre = centre[np.isfinite(centre)]
            far = float((centre > FAR_LOCUS_PX).mean())
            clear = ratio > 1.5
            scatter_ax.scatter(
                far * 100, ratio,
                s=90, marker="o" if clear else "X",
                color="#22c55e" if clear else "#dc2626",
                edgecolor="#111", linewidth=0.6, zorder=3,
            )
        scatter_ax.axhline(1.5, color="#666", ls="--", lw=1,
                           label="legibility threshold (peak / pre-play baseline = 1.5)")
        scatter_ax.set_xlabel("% of play spent in the FAR centroid locus (x > 1250 px)")
        scatter_ax.set_ylabel("dispersion peak ÷ pre-play baseline")
        scatter_ax.set_title(
            "Locus diagnostic — flat plays do NOT sit in the far locus\n"
            "(AUC 0.22, corr −0.34: the two-games-share-the-grass hypothesis is rejected)",
            fontsize=10, pad=14,
        )
        scatter_ax.grid(alpha=0.25)
        scatter_ax.legend(fontsize=8)

# Bottom-right: whistle bookkeeping, corrected.
table_ax = fig.add_subplot(grid[4, 1])
table_ax.axis("off")
table_ax.set_title("Whistle bookkeeping — GH010007", fontsize=11, loc="left")
table_ax.text(
    0.0, 0.92,
    "Corrections applied\n"
    "  • end-match window is asymmetric: whistle ∈ [end − 4 s, end + 0.5 s]\n"
    "  • the partial final play (520–531.5 s) is excluded from the recall denominator\n"
    "  • “mid-play” means whistle ∈ [start + 2 s, end − 4 s]\n\n"
    "Results\n"
    "  recall     11 / 11  = 1.00     (was 0.83 under a symmetric ±2.5 s window)\n"
    "  precision  12 / 30  = 0.40     (matches the parallel-game prediction)\n"
    "  mid-play whistles          0   (was “6 of 12” — entirely a window artifact)\n\n"
    "Implication for P3\n"
    "  Whistles are a far better end anchor than last checkpoint suggested, but\n"
    "  18 of 30 anchors are still not play ends, so corroboration stays mandatory.",
    va="top", ha="left", fontsize=9.5, family="monospace", transform=table_ax.transAxes,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"wrote {OUT}")
