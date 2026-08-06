"""Dominant-cluster isolation and per-frame play features.

YOLO on the native ROI strip finds 28-45 people per frame, but only ~13 of them
are participants (10 players plus referees). The rest are benches, substitutes
and officials standing on our own pitch, so the field mask cannot remove them.
Naive aggregates over all detections are useless -- the centroid saturates and
dispersion pins to the frame width.

The escalation, cheapest first:

1. gate detections by foot point inside the field mask (:mod:`playsplit.field`)
2. cluster the survivors along the field's long axis and keep the dominant one
3. only if still noisy, add tracking and stationarity suppression

Steps 1 and 2 are implemented here. The field's long axis is horizontal in
image space because the camera sits on the sideline, not behind the endzone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import FieldMask


@dataclass
class ClusterConfig:
    """Thresholds for gating, stationarity suppression and clustering."""

    #: Gap along the long axis that separates two clusters, in source pixels.
    split_gap_px: float = 160.0
    #: Plausible participant count for the dominant cluster (players + refs).
    min_members: int = 4
    max_members: int = 16
    #: Weight on continuity with the previous frame's cluster, in pixels of
    #: centroid distance traded against one extra member.
    continuity_px_per_member: float = 45.0
    #: Sliding horizon over which a track's displacement is judged. Long
    #: enough that linemen holding a formation are not mistaken for furniture
    #: -- a 10 s horizon would suppress exactly the people formation detection
    #: depends on -- and short enough that benches still die.
    stationary_horizon_s: float = 36.0
    #: Displacement below which a track is furniture, in player-heights. A
    #: player is ~1.75 m, so 1.15 heights is about 2 m. Expressing the
    #: threshold in heights rather than pixels makes it perspective-correct:
    #: the same real distance is fewer pixels at the far side of the pitch.
    stationary_move_heights: float = 1.15
    #: Grid cell size for the occupancy map, in source pixels.
    occupancy_cell_px: int = 48
    #: A cell occupied in more than this fraction of frames holds furniture,
    #: not a player. Benches and substitutes stand still for minutes; players
    #: never occupy one 48 px cell for half of a nine-minute clip.
    #:
    #: Swept against the 12 labelled plays on GH010007. At 0.50 every play
    #: shows a dispersion swing of at least 168 px; at 0.33 five plays fall
    #: below 150 px and become hard to read. Suppressing too eagerly costs
    #: more than it saves, because a player who pauses briefly on the line of
    #: scrimmage shares a cell with whoever stood there earlier.
    stationary_occupancy: float = 0.50


@dataclass
class FrameFeatures:
    """Per-frame statistics of the dominant cluster."""

    time: float
    count: int
    centroid_x: float
    centroid_y: float
    dispersion: float
    #: Count of detections gated out by the field mask, for diagnostics.
    rejected: int

    @property
    def valid(self) -> bool:
        return self.count > 0


def gate(
    xs: np.ndarray, ys: np.ndarray, mask: FieldMask
) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep detections whose foot point lands on the playing surface.

    *ys* must already be bbox bottoms: a person's feet, not their centre. Using
    the centre would admit anyone leaning over the touchline.
    """
    if len(xs) == 0:
        return xs, ys, 0
    keep = mask.contains_many(xs, ys)
    return xs[keep], ys[keep], int((~keep).sum())


def stationary_detections(
    xs: np.ndarray,
    ys: np.ndarray,
    heights: np.ndarray,
    track_id: np.ndarray,
    frame_index: np.ndarray,
    fps: float,
    cfg: ClusterConfig,
) -> np.ndarray:
    """Flag detections belonging to a track that is not going anywhere.

    Judged per detection over a sliding horizon rather than per track over its
    whole life, so a substitute who warms up, sits down, then comes on is
    suppressed only while actually parked.

    Sprinters fragment into short-lived tracks at 5 fps. That is by design:
    a fragment spans less than the horizon, so it is never judged stationary
    and stays in the active set. Continuity is only ever needed for people
    standing still, who track trivially at any frame rate.
    """
    flags = np.zeros(len(xs), dtype=bool)
    if len(xs) == 0:
        return flags

    horizon_frames = cfg.stationary_horizon_s * fps
    for identity in np.unique(track_id):
        if identity < 0:
            continue
        member = np.flatnonzero(track_id == identity)
        frames = frame_index[member].astype(float)
        order = np.argsort(frames)
        member, frames = member[order], frames[order]
        px, py = xs[member], ys[member]
        scale = np.median(heights[member]) or 1.0
        limit = cfg.stationary_move_heights * scale

        # Widest displacement within the horizon centred on each detection.
        starts = np.searchsorted(frames, frames - horizon_frames / 2, side="left")
        ends = np.searchsorted(frames, frames + horizon_frames / 2, side="right")
        for position in range(len(member)):
            lo, hi = starts[position], ends[position]
            if frames[hi - 1] - frames[lo] < horizon_frames * 0.6:
                continue  # too short a window to call it parked
            spread = max(np.ptp(px[lo:hi]), np.ptp(py[lo:hi]))
            if spread < limit:
                flags[member[position]] = True
    return flags


def occupancy_map(
    xs: np.ndarray, ys: np.ndarray, frame_count: int, cfg: ClusterConfig
) -> set[tuple[int, int]]:
    """Find grid cells that are almost always occupied by somebody.

    Sideline benches, substitutes and spectators inside the field mask hold
    position for minutes at a time, so their cell is occupied in most frames.
    Players cross a cell in a second or two. Thresholding occupancy therefore
    separates furniture from participants without any tracking -- the cheap
    version of stationarity suppression.
    """
    if len(xs) == 0 or frame_count == 0:
        return set()
    cell = cfg.occupancy_cell_px
    keys = np.stack([(xs // cell).astype(int), (ys // cell).astype(int)], axis=1)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    hot = counts / frame_count > cfg.stationary_occupancy
    return {(int(a), int(b)) for a, b in unique[hot]}


def suppress_stationary(
    xs: np.ndarray, ys: np.ndarray, static_cells: set[tuple[int, int]], cfg: ClusterConfig
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop detections sitting in persistently-occupied cells."""
    if len(xs) == 0 or not static_cells:
        return xs, ys, 0
    cell = cfg.occupancy_cell_px
    keep = np.array(
        [
            (int(x // cell), int(y // cell)) not in static_cells
            for x, y in zip(xs, ys)
        ]
    )
    return xs[keep], ys[keep], int((~keep).sum())


def split_along_axis(values: np.ndarray, gap: float) -> list[np.ndarray]:
    """Group indices into runs separated by more than *gap* along one axis."""
    if len(values) == 0:
        return []
    order = np.argsort(values)
    ordered = values[order]
    breaks = np.flatnonzero(np.diff(ordered) > gap) + 1
    return [group for group in np.split(order, breaks) if len(group)]


def dominant_cluster(
    xs: np.ndarray,
    ys: np.ndarray,
    cfg: ClusterConfig,
    previous_centroid: float | None,
) -> np.ndarray:
    """Pick the cluster most likely to be the play.

    Scores candidate groups by size, penalised by distance from the previous
    frame's cluster so the choice does not flicker between the play and a
    similarly-sized bench. Groups outside the plausible member range are
    considered only if nothing else qualifies.
    """
    groups = split_along_axis(xs, cfg.split_gap_px)
    if not groups:
        return np.array([], dtype=int)

    plausible = [g for g in groups if cfg.min_members <= len(g) <= cfg.max_members]
    candidates = plausible or groups

    def score(group: np.ndarray) -> float:
        value = float(len(group))
        if previous_centroid is not None:
            distance = abs(float(xs[group].mean()) - previous_centroid)
            value -= distance / cfg.continuity_px_per_member
        return value

    return max(candidates, key=score)


def features_from_active(time: float, xs: np.ndarray, rejected: int) -> FrameFeatures:
    """Summarise the active participant set directly.

    With foot-point gating and per-track stationarity applied, what remains
    inside the field mask *is* the participant set, so its statistics can be
    taken as-is. This deliberately replaces picking a "dominant" sub-cluster:
    gap-based splitting fragments the players at exactly the wrong moment,
    when receivers spread at the snap, and keeping one fragment holds
    dispersion flat through the burst it is supposed to measure.
    """
    if len(xs) == 0:
        return FrameFeatures(time, 0, float("nan"), float("nan"), float("nan"), rejected)
    return FrameFeatures(
        time=time,
        count=len(xs),
        centroid_x=float(xs.mean()),
        centroid_y=float("nan"),
        dispersion=float(xs.std()) if len(xs) > 1 else 0.0,
        rejected=rejected,
    )


def features_for_frame(
    time: float,
    xs: np.ndarray,
    ys: np.ndarray,
    mask: FieldMask,
    cfg: ClusterConfig,
    previous_centroid: float | None,
    static_cells: set[tuple[int, int]] | None = None,
) -> FrameFeatures:
    """Gate, suppress furniture, cluster and summarise one frame's detections."""
    gated_x, gated_y, rejected = gate(xs, ys, mask)
    if static_cells:
        gated_x, gated_y, dropped = suppress_stationary(
            gated_x, gated_y, static_cells, cfg
        )
        rejected += dropped
    if len(gated_x) == 0:
        return FrameFeatures(time, 0, float("nan"), float("nan"), float("nan"), rejected)

    members = dominant_cluster(gated_x, gated_y, cfg, previous_centroid)
    if len(members) == 0:
        return FrameFeatures(time, 0, float("nan"), float("nan"), float("nan"), rejected)

    cluster_x, cluster_y = gated_x[members], gated_y[members]
    return FrameFeatures(
        time=time,
        count=len(members),
        centroid_x=float(cluster_x.mean()),
        centroid_y=float(cluster_y.mean()),
        dispersion=float(cluster_x.std()),
        rejected=rejected + (len(gated_x) - len(members)),
    )


def spread_rate(dispersion: np.ndarray, fps: float, smooth: int = 3) -> np.ndarray:
    """Rate of change of dispersion -- the snap signature.

    A snap turns two tight facing lines into receivers running apart, so
    dispersion rises sharply. This derivative is the burst feature that raw
    pixel motion failed to provide.
    """
    filled = _interpolate_gaps(dispersion)
    kernel = np.ones(smooth) / smooth
    smoothed = np.convolve(filled, kernel, mode="same")
    return np.gradient(smoothed) * fps


def _interpolate_gaps(values: np.ndarray) -> np.ndarray:
    """Linearly bridge frames where no cluster was found."""
    result = values.astype(np.float64).copy()
    missing = ~np.isfinite(result)
    if missing.all():
        return np.zeros_like(result)
    indices = np.arange(len(result))
    result[missing] = np.interp(indices[missing], indices[~missing], result[~missing])
    return result
