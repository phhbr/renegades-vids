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
    #: Bounding-box heights, a proxy for distance from camera.
    heights: np.ndarray
    #: Index into the frame sequence for each detection.
    frame_index: np.ndarray
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

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_h: list[np.ndarray] = []
    all_i: list[np.ndarray] = []
    count = 0
    started = time.time()

    for index, frame in enumerate(_frames_pipelined(path, fps, crop)):
        count = index + 1
        boxes = model.predict(
            frame,
            imgsz=detect_cfg.imgsz,
            conf=detect_cfg.conf,
            classes=[0],
            device=detect_cfg.device,
            verbose=False,
        )[0].boxes

        if len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            # Foot point in *source* coordinates: bbox bottom-centre, shifted
            # by the crop origin so it can be tested against the full-frame
            # mask. Centres would admit anyone leaning over the touchline.
            all_x.append((xyxy[:, 0] + xyxy[:, 2]) / 2 + crop.x)
            all_y.append(xyxy[:, 3] + crop.y)
            all_h.append(xyxy[:, 3] - xyxy[:, 1])
            all_i.append(np.full(len(xyxy), index))

        if progress is not None:
            progress(index)

    empty = np.array([], dtype=np.float64)
    return RawDetections(
        xs=np.concatenate(all_x) if all_x else empty,
        ys=np.concatenate(all_y) if all_y else empty,
        heights=np.concatenate(all_h) if all_h else empty,
        frame_index=np.concatenate(all_i).astype(int) if all_i else empty.astype(int),
        frame_count=count,
        fps=fps,
        elapsed_s=time.time() - started,
    )
