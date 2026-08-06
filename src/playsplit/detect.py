"""Person detection over a clip, pipelined against decode.

Decode and inference are separate processes competing for the same wall clock.
Run serially they compose as 1/(1/8.6 + 1/16.6) = 5.7x realtime, under the
6.7x the performance target needs. Run as producer/consumer with a bounded
queue, wall clock is bounded by the slower stage (~8.6x) instead of the sum.
The queue bound matters: an unbounded one would buffer a whole clip of
1920x256 BGR frames, which is several GB.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DetectConfig
from .frames import Crop, iter_frames

#: Frames held in flight between decode and inference.
QUEUE_DEPTH = 16
_SENTINEL = object()


@dataclass
class RawDetections:
    """Every person detection in a clip, in source coordinates.

    Cached so that gating, stationarity suppression and clustering can all be
    retuned without paying for another YOLO pass -- the expensive stage runs
    once per clip, calibration iterates on top of it.
    """

    #: Foot points: bbox bottom-centre x, bbox bottom y.
    xs: np.ndarray
    ys: np.ndarray
    #: Bounding-box heights. Double as a local scale: a player is ~1.75 m, so
    #: height in pixels converts displacement to metres at that image depth,
    #: which matters because this camera has strong perspective foreshortening.
    heights: np.ndarray
    #: Index into the frame sequence for each detection.
    frame_index: np.ndarray
    #: ByteTrack identity, or -1 where the tracker did not assign one.
    track_id: np.ndarray
    #: Mean jersey hue/saturation with grass suppressed. Passive metadata only
    #: -- logged so the labels can later say whether it separates rosters.
    jersey_hue: np.ndarray
    jersey_sat: np.ndarray
    frame_count: int
    fps: float
    elapsed_s: float

    @property
    def video_s(self) -> float:
        return self.frame_count / self.fps

    @property
    def realtime_factor(self) -> float:
        return self.video_s / self.elapsed_s if self.elapsed_s else float("inf")

    def frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Foot points detected in one frame."""
        selection = self.frame_index == index
        return self.xs[selection], self.ys[selection]


def _produce(
    path: Path, fps: float, crop: Crop, sink: queue.Queue, error: list[BaseException]
) -> None:
    try:
        for frame in iter_frames(path, fps, crop=crop, gray=False):
            sink.put(frame)
    except BaseException as exc:  # surfaced on the consumer thread
        error.append(exc)
    finally:
        sink.put(_SENTINEL)


def _frames_pipelined(
    path: Path, fps: float, crop: Crop
) -> Iterator[np.ndarray]:
    """Yield frames decoded on a background thread.

    ffmpeg decoding releases the GIL, so the decoder genuinely overlaps with
    Torch inference rather than time-slicing against it.
    """
    channel: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
    errors: list[BaseException] = []
    worker = threading.Thread(
        target=_produce, args=(path, fps, crop, channel, errors), daemon=True
    )
    worker.start()
    try:
        while True:
            item = channel.get()
            if item is _SENTINEL:
                break
            yield item
    finally:
        worker.join(timeout=5.0)
    if errors:
        raise errors[0]


def run(
    path: Path,
    crop: Crop,
    detect_cfg: DetectConfig,
    fps: float,
    *,
    progress=None,
) -> RawDetections:
    """Detect every person in the ROI strip, in source coordinates."""
    from ultralytics import YOLO

    if not Path(detect_cfg.weights).is_file():
        raise FileNotFoundError(
            f"model weights missing at {detect_cfg.weights}. They are vendored "
            "in models/ so nothing downloads at a venue with no internet."
        )
    model = YOLO(str(detect_cfg.weights))

    columns: dict[str, list[np.ndarray]] = {
        key: [] for key in ("x", "y", "h", "i", "id", "hue", "sat")
    }
    count = 0
    started = time.time()

    for index, frame in enumerate(_frames_pipelined(path, fps, crop)):
        count = index + 1
        # Tracking is opt-in; see DetectConfig.use_tracker for why it is off.
        # Analysis fps is never raised for it -- that would blow the wall-clock
        # budget, and fragmenting sprinters are harmless, since suppression
        # only ever needs continuity for people standing still.
        if detect_cfg.use_tracker:
            result = model.track(
                frame, imgsz=detect_cfg.imgsz, conf=detect_cfg.conf, classes=[0],
                device=detect_cfg.device, tracker=detect_cfg.tracker,
                persist=True, verbose=False,
            )
        else:
            result = model.predict(
                frame, imgsz=detect_cfg.imgsz, conf=detect_cfg.conf, classes=[0],
                device=detect_cfg.device, verbose=False,
            )
        boxes = result[0].boxes

        if not len(boxes):
            if progress is not None:
                progress(index)
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        # Foot point in *source* coordinates: bbox bottom-centre, shifted by
        # the crop origin so it can be tested against the full-frame mask.
        # Centres would admit anyone leaning over the touchline.
        columns["x"].append((xyxy[:, 0] + xyxy[:, 2]) / 2 + crop.x)
        columns["y"].append(xyxy[:, 3] + crop.y)
        columns["h"].append(xyxy[:, 3] - xyxy[:, 1])
        columns["i"].append(np.full(len(xyxy), index))
        columns["id"].append(
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else np.full(len(xyxy), -1)
        )
        hue, sat = _jersey_colour(frame, xyxy)
        columns["hue"].append(hue)
        columns["sat"].append(sat)

        if progress is not None:
            progress(index)

    def stack(key: str, dtype=np.float64) -> np.ndarray:
        parts = columns[key]
        return np.concatenate(parts).astype(dtype) if parts else np.array([], dtype=dtype)

    return RawDetections(
        xs=stack("x"), ys=stack("y"), heights=stack("h"),
        frame_index=stack("i", int), track_id=stack("id", int),
        jersey_hue=stack("hue"), jersey_sat=stack("sat"),
        frame_count=count, fps=fps, elapsed_s=time.time() - started,
    )


def _jersey_colour(frame: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean hue and saturation of each detection's torso, grass suppressed.

    Sampled from the upper-middle of the box, where a jersey is, and with
    green pixels dropped so the pitch behind a thin player does not dominate.
    Recorded as metadata only -- at 20-45 px tall these statistics are weak,
    and at least one roster here wears green on green grass.
    """
    import cv2

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hues = np.zeros(len(xyxy))
    sats = np.zeros(len(xyxy))
    for row, (x1, y1, x2, y2) in enumerate(xyxy):
        cx = (x1 + x2) / 2
        half = max((x2 - x1) / 4, 1.0)
        top = y1 + (y2 - y1) * 0.2
        bottom = y1 + (y2 - y1) * 0.55
        patch = hsv[
            max(int(top), 0) : min(int(bottom) + 1, height),
            max(int(cx - half), 0) : min(int(cx + half) + 1, width),
        ]
        if patch.size == 0:
            continue
        flat = patch.reshape(-1, 3)
        not_grass = ~((flat[:, 0] >= 30) & (flat[:, 0] <= 90) & (flat[:, 1] >= 60))
        sample = flat[not_grass] if not_grass.any() else flat
        hues[row] = float(sample[:, 0].mean())
        sats[row] = float(sample[:, 1].mean())
    return hues, sats
