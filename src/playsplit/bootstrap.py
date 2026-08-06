"""P1 -- bootstrap a label file and its correction page for one clip."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import audio, labels, thumbs
from .config import Config
from .probe import ClipInfo
from .roi import Band, estimate_band

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class Shot:
    src: str
    label: str
    time: float


@dataclass
class Candidate:
    index: int
    start: float
    end: float
    notes: str
    shots: list[Shot]


def run(
    clip: ClipInfo,
    analysis_dir: Path,
    cfg: Config,
    *,
    placeholder_lead_s: float = 15.0,
    force: bool = False,
    log=print,
) -> tuple[Path, Path]:
    """Detect whistles, write bootstrap labels, and render the correction page.

    Returns the paths of the label CSV and the HTML page.
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stem = clip.path.stem

    log(f"[1/4] decoding audio for {clip.name}")
    signal = audio.load_audio(
        clip.path, cfg.audio.sample_rate, analysis_dir / f"{stem}.wav"
    )

    log("[2/4] detecting whistle anchors")
    whistles = audio.detect(signal, cfg.audio)
    log(f"      {len(whistles)} anchors ({len(whistles) / (clip.duration / 60):.1f}/min)")

    log("[3/4] estimating play band from accumulated motion")
    band_cache = analysis_dir / f"{stem}__band.npz"
    if band_cache.is_file() and not force:
        cached = np.load(band_cache)
        band = Band(
            int(cached["top"]), int(cached["bottom"]), int(cached["peak"]),
            cached["column_profile"],
        )
    else:
        band, accumulator = estimate_band(
            clip.path, clip.width, clip.height, cfg.analysis
        )
        np.savez_compressed(
            band_cache, top=band.top, bottom=band.bottom, peak=band.peak,
            column_profile=band.column_profile, accumulator=accumulator,
        )
    log(f"      native y {band.top}–{band.bottom} (peak {band.peak})")

    rows = labels.bootstrap_from_whistles(
        clip.name,
        [w.time for w in whistles],
        placeholder_lead_s=placeholder_lead_s,
        clip_duration=clip.duration,
    )
    label_file = analysis_dir / f"{stem}__labels.csv"
    labels.write(label_file, rows)

    log(f"[4/4] extracting {len(rows) * 3} thumbnails")
    crop = band.to_crop(clip.width, clip.height, cfg.detect.band_padding_px)
    shot_dir = analysis_dir / "thumbs"
    candidates: list[Candidate] = []
    for row in rows:
        paths = thumbs.extract_triplet(
            clip.path, row.start, row.end, crop, shot_dir, stem, force=force,
        )
        span = max(row.end - row.start, 0.1)
        candidates.append(
            Candidate(
                index=row.index, start=row.start, end=row.end, notes=row.notes,
                shots=[
                    Shot(
                        src=str(path.relative_to(analysis_dir)),
                        label=label,
                        time=row.start + span * point,
                    )
                    for path, label, point in zip(
                        paths, thumbs.SAMPLE_LABELS, thumbs.SAMPLE_POINTS
                    )
                ],
            )
        )

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape()
    )
    page = analysis_dir / f"{stem}__labels.html"
    page.write_text(
        env.get_template("labels.html.j2").render(
            clip=clip.name,
            duration=clip.duration,
            band=band,
            candidates=candidates,
            label_file=label_file.name,
            lead=placeholder_lead_s,
        )
    )
    return label_file, page
