"""Generate the milestone-2 calibration figure from cached probe arrays.

Run via ``uv run python scripts/calibration_report.py``. Consumes the .npy
caches written by the exploration probes and renders analysis/calibration.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path("analysis/calibration.png")

REN_WHISTLES = np.load(SCRATCH / "ren_whistles.npy")
SVA_WHISTLES = np.array(
    [61.7, 93.9, 174.0, 270.8, 285.1, 297.1, 346.8, 409.8, 443.7, 507.3]
)


def smooth(x: np.ndarray, n: int = 5) -> np.ndarray:
    return np.convolve(x, np.ones(n) / n, mode="same")


fig, axes = plt.subplots(3, 2, figsize=(17, 12))
fig.suptitle(
    "playsplit calibration — RenegadesVsAnts/GH010007 (in-game) vs "
    "SpatzenVsAnts/GH020002 (warm-up heavy)",
    fontsize=14,
    fontweight="bold",
)

# --- Row 1: ROI row-activity profiles -------------------------------------
ren_acc = np.load(SCRATCH / "ren_acc3.npy")
sva_acc = np.load(SCRATCH / "sva_acc.npy")

for ax, acc, title, native_h in (
    (axes[0][0], ren_acc, "RenegadesVsAnts GH010007", 1440),
    (axes[0][1], sva_acc, "SpatzenVsAnts GH020002", 1440),
):
    prof = acc.sum(axis=1)
    prof = prof / prof.max()
    ys = np.arange(len(prof)) * (native_h / len(prof))
    ax.plot(prof, ys, color="#2563eb")
    peak = int(prof.argmax())
    band = np.where(prof > 0.45 * prof.max())[0]
    ax.axhspan(
        band[0] * native_h / len(prof),
        band[-1] * native_h / len(prof),
        color="#22c55e",
        alpha=0.22,
        label=f"auto ROI band (native y {int(band[0]*native_h/len(prof))}"
        f"–{int(band[-1]*native_h/len(prof))})",
    )
    ax.axhline(peak * native_h / len(prof), color="#dc2626", ls="--", lw=1, label="peak row")
    ax.invert_yaxis()
    ax.set_title(f"Stage 1 — accumulated motion by row\n{title}")
    ax.set_xlabel("normalised activity")
    ax.set_ylabel("native y (px)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

# --- Row 2: motion energy vs whistles (the negative result) ---------------
ren_row = np.load(SCRATCH / "ren_row3.npy").astype(np.float32)
sva_row = np.load(SCRATCH / "sva_rowprof.npy").astype(np.float32)

for ax, rows, band, whistles, title in (
    (axes[1][0], ren_row, (180, 270), REN_WHISTLES, "RenegadesVsAnts GH010007"),
    (axes[1][1], sva_row, (105, 210), SVA_WHISTLES, "SpatzenVsAnts GH020002"),
):
    sig = smooth(rows[:, band[0] : band[1]].sum(axis=1))
    t = np.arange(len(sig)) / 5.0
    sig = sig / np.percentile(sig, 99)
    ax.plot(t, sig, color="#0f172a", lw=0.9, label="motion energy (ROI band)")
    for i, w in enumerate(whistles):
        ax.axvline(w, color="#f59e0b", alpha=0.75, lw=1.1,
                   label="whistle" if i == 0 else None)
    p50, p99 = np.percentile(sig, [50, 99])
    ax.set_title(
        f"Stage 2 — motion energy vs whistles — {title}\n"
        f"peakiness p99/p50 = {p99/p50:.1f}  →  no per-play structure",
        fontsize=10,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalised motion")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

# --- Row 3: why it fails — detector scale, and decode benchmarks ----------
ax = axes[2][0]
ts = [125, 129, 133, 300, 440]
small = [4, 4, 6, 14, 11]
native = [42, 34, 28, 30, 45]
x = np.arange(len(ts))
ax.bar(x - 0.2, small, 0.4, label="imgsz=640 (downscaled frame)", color="#f87171")
ax.bar(x + 0.2, native, 0.4, label="imgsz=1920 (native ROI strip)", color="#22c55e")
ax.set_xticks(x, [f"t={v}s" for v in ts])
ax.set_ylabel("persons detected")
ax.set_title(
    "Stage 3 — YOLO11n detections\ndownscaling loses 70–85% of players",
    fontsize=10,
)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")

ax = axes[2][1]
labels = [
    "hwaccel\nvideotoolbox",
    "software\ndecode",
    "keyframe\nonly (1 fps)",
    "YOLO11n\nMPS strip",
]
speeds = [2.6, 8.6, 22.0, 16.6]
colors = ["#f87171", "#22c55e", "#93c5fd", "#a78bfa"]
ax.bar(labels, speeds, color=colors)
ax.axhline(3.0, color="#dc2626", ls="--", lw=1.2,
           label="required: 20 min clip in ≤3 min (6.7x)")
ax.axhline(6.7, color="#dc2626", ls=":", lw=1.2)
for i, v in enumerate(speeds):
    ax.text(i, v + 0.4, f"{v}x", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("× realtime")
ax.set_title(
    "Throughput on M4 (1920x1440@59.94, 60 Mbps)\n"
    "videotoolbox is 3.3x SLOWER than software decode",
    fontsize=10,
)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")

fig.tight_layout(rect=(0, 0, 1, 0.97))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=115)
print(f"wrote {OUT}")
