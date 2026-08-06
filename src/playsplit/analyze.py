"""P2 -- compute and cache per-clip detection features."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import detect, field
from .cluster import (
    ClusterConfig,
    FrameFeatures,
    features_for_frame,
    occupancy_map,
    stationary_detections,
)
from .config import Config
from .probe import ClipInfo
from .roi import Band, estimate_band


def _band(clip: ClipInfo, analysis_dir: Path, cfg: Config, force: bool, log) -> Band:
    cache = analysis_dir / f"{clip.path.stem}__band.npz"
    if cache.is_file() and not force:
        data = np.load(cache)
        return Band(
            int(data["top"]), int(data["bottom"]), int(data["peak"]),
            data["column_profile"],
        )
    band, accumulator = estimate_band(
        clip.path, clip.width, clip.height, cfg.analysis
    )
    np.savez_compressed(
        cache, top=band.top, bottom=band.bottom, peak=band.peak,
        column_profile=band.column_profile, accumulator=accumulator,
    )
    log(f"      play band: native y {band.top}–{band.bottom}")
    return band


def _mask(
    clip: ClipInfo, analysis_dir: Path, force: bool, log
) -> field.FieldMask:
    cache = analysis_dir / f"{clip.path.stem}__fieldmask.npz"
    if cache.is_file() and not force:
        data = np.load(cache)
        return field.FieldMask(
            data["mask"], int(data["width"]), int(data["height"]),
            int(data["source_width"]), int(data["source_height"]),
        )
    background = field.background_frame(
        clip.path, clip.width, clip.height, duration=clip.duration
    )
    mask, track = field.segment(background, clip.width, clip.height)
    np.savez_compressed(
        cache, mask=mask.mask, width=mask.width, height=mask.height,
        source_width=mask.source_width, source_height=mask.source_height,
    )
    overlay = field.write_overlay(
        background, mask, track, analysis_dir / f"{clip.path.stem}__fieldmask.png"
    )
    log(f"      field mask: {mask.area_fraction * 100:.1f}% of frame → {overlay.name}")
    return mask


def detections(
    clip: ClipInfo,
    analysis_dir: Path,
    cfg: Config,
    *,
    force: bool = False,
    log=print,
) -> detect.RawDetections:
    """Run (or load) the YOLO pass for a clip.

    This is the only expensive stage. It is cached separately from the features
    so that clustering thresholds can be retuned in seconds.
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)
    cache = analysis_dir / f"{clip.path.stem}__detections.npz"
    if cache.is_file() and not force:
        data = np.load(cache)
        return detect.RawDetections(
            xs=data["xs"], ys=data["ys"], heights=data["heights"],
            frame_index=data["frame_index"], track_id=data["track_id"],
            jersey_hue=data["jersey_hue"], jersey_sat=data["jersey_sat"],
            frame_count=int(data["frame_count"]),
            fps=float(data["fps"]), elapsed_s=float(data["elapsed_s"]),
        )

    band = _band(clip, analysis_dir, cfg, force, log)
    crop = band.to_crop(clip.width, clip.height, cfg.detect.band_padding_px)
    log(f"      inference strip {crop.width}x{crop.height} at y={crop.y}")

    raw = detect.run(clip.path, crop, cfg.detect, cfg.analysis.fps)
    log(
        f"      {raw.frame_count} frames, {len(raw.xs)} detections in "
        f"{raw.elapsed_s:.0f}s = {raw.realtime_factor:.1f}x realtime"
    )
    np.savez_compressed(
        cache, xs=raw.xs, ys=raw.ys, heights=raw.heights,
        frame_index=raw.frame_index, track_id=raw.track_id,
        jersey_hue=raw.jersey_hue, jersey_sat=raw.jersey_sat,
        frame_count=raw.frame_count, fps=raw.fps, elapsed_s=raw.elapsed_s,
    )
    return raw


def features(
    clip: ClipInfo,
    analysis_dir: Path,
    cfg: Config,
    *,
    cluster_cfg: ClusterConfig | None = None,
    force: bool = False,
    log=print,
) -> tuple[list[FrameFeatures], float]:
    """Reduce cached detections to per-frame dominant-cluster features."""
    cluster_cfg = cluster_cfg or ClusterConfig()
    raw = detections(clip, analysis_dir, cfg, force=force, log=log)
    mask = _mask(clip, analysis_dir, force, log)

    if len(raw.xs) == 0:
        return [], raw.realtime_factor

    on_field = mask.contains_many(raw.xs, raw.ys)

    # Suppress people who are not going anywhere. Per-track displacement is the
    # better instrument in principle, but only when tracks live long enough to
    # judge; at 5 fps in this crowd they do not, so the positional occupancy
    # map is the default. Both reduce to the same idea, applied per identity or
    # per location.
    if raw.track_id.size and (raw.track_id >= 0).any():
        parked = stationary_detections(
            raw.xs, raw.ys, raw.heights, raw.track_id, raw.frame_index,
            raw.fps, cluster_cfg,
        )
        static_cells: set[tuple[int, int]] = set()
    else:
        parked = np.zeros(len(raw.xs), dtype=bool)
        static_cells = occupancy_map(
            raw.xs[on_field], raw.ys[on_field], raw.frame_count, cluster_cfg
        )
    kept = on_field & ~parked
    log(
        f"      {len(raw.xs)} detections → {int(on_field.sum())} on field, "
        f"{int(parked.sum())} parked, {len(static_cells)} static cells"
    )

    rows: list[FrameFeatures] = []
    previous_centroid: float | None = None
    for index in range(raw.frame_count):
        selection = (raw.frame_index == index) & kept
        row = features_for_frame(
            index / raw.fps, raw.xs[selection], raw.ys[selection],
            mask, cluster_cfg, previous_centroid, static_cells,
        )
        if row.valid:
            previous_centroid = row.centroid_x
        rows.append(row)
    return rows, raw.realtime_factor
