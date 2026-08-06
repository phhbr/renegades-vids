"""playsplit command line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import bootstrap as bootstrap_mod
from . import config as config_mod
from . import probe

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


@app.command("clips")
def list_clips(game_dir: Path = typer.Argument(..., help="Game directory")) -> None:
    """List a game's clips in true recording order."""
    clips = probe.find_clips(game_dir)
    if not clips:
        console.print(f"[red]no video files found in {game_dir}")
        raise typer.Exit(1)

    table = Table("#", "clip", "duration", "resolution", "fps", "audio")
    total = 0.0
    for index, path in enumerate(clips, start=1):
        info = probe.probe(path)
        total += info.duration
        table.add_row(
            str(index), info.name, f"{info.duration:.1f}s",
            f"{info.width}x{info.height}", f"{info.fps:.2f}",
            "yes" if info.has_audio else "no",
        )
    console.print(table)
    console.print(f"{len(clips)} clips, {total / 60:.1f} min total")


@app.command("bootstrap")
def bootstrap(
    game_dir: Path = typer.Argument(..., help="Game directory"),
    clip: str = typer.Option(..., "--clip", help="Clip filename to label"),
    lead: float = typer.Option(15.0, "--lead", help="Placeholder pre-whistle lead (s)"),
    force: bool = typer.Option(False, "--force", help="Recompute cached artifacts"),
) -> None:
    """Generate a bootstrap label file and correction page for one clip."""
    cfg = config_mod.load(game_dir)
    matches = [p for p in probe.find_clips(game_dir) if p.name == clip]
    if not matches:
        console.print(f"[red]clip {clip!r} not found in {game_dir}")
        raise typer.Exit(1)

    info = probe.probe(matches[0])
    analysis_dir = game_dir / "analysis"
    label_file, page = bootstrap_mod.run(
        info, analysis_dir, cfg, placeholder_lead_s=lead, force=force,
        log=lambda message: console.print(f"[dim]{message}"),
    )
    console.print(f"\n[green]labels[/] {label_file}")
    console.print(f"[green]review[/] {page}")


@app.command("features")
def features(
    game_dir: Path = typer.Argument(..., help="Game directory"),
    clip: str = typer.Option(..., "--clip", help="Clip filename"),
    force: bool = typer.Option(False, "--force", help="Recompute cached artifacts"),
) -> None:
    """Compute detection-cluster features for one clip."""
    from . import analyze

    cfg = config_mod.load(game_dir)
    matches = [p for p in probe.find_clips(game_dir) if p.name == clip]
    if not matches:
        console.print(f"[red]clip {clip!r} not found in {game_dir}")
        raise typer.Exit(1)

    info = probe.probe(matches[0])
    rows, realtime = analyze.features(
        info, game_dir / "analysis", cfg, force=force,
        log=lambda message: console.print(f"[dim]{message}"),
    )
    valid = [f for f in rows if f.valid]
    console.print(
        f"[green]{len(rows)}[/] frames, {len(valid)} with a cluster, "
        f"{realtime:.1f}x realtime"
    )


@app.command("segment")
def segment(
    game_dir: Path = typer.Argument(..., help="Game directory"),
    clip: str = typer.Option(..., "--clip", help="Clip filename"),
    force: bool = typer.Option(False, "--force", help="Recompute cached artifacts"),
) -> None:
    """Emit tiered candidate segments for one clip."""
    import numpy as np

    from . import analyze, audio, segments
    from .statemachine import SegmentConfigSM, Tier, find_episodes

    cfg = config_mod.load(game_dir)
    matches = [p for p in probe.find_clips(game_dir) if p.name == clip]
    if not matches:
        console.print(f"[red]clip {clip!r} not found in {game_dir}")
        raise typer.Exit(1)

    info = probe.probe(matches[0])
    analysis_dir = game_dir / "analysis"
    rows, realtime = analyze.features(
        info, analysis_dir, cfg, force=force,
        log=lambda message: console.print(f"[dim]{message}"),
    )
    times = np.array([r.time for r in rows])
    dispersion = np.array([r.dispersion for r in rows])
    width = 9
    padded = np.pad(dispersion, (width // 2, width // 2), mode="edge")
    smoothed = np.nanmedian(
        np.lib.stride_tricks.sliding_window_view(padded, width), axis=1
    )

    signal = audio.load_audio(
        info.path, cfg.audio.sample_rate, analysis_dir / f"{info.path.stem}.wav"
    )
    whistles = audio.detect(signal, cfg.audio)

    sm = SegmentConfigSM()
    episodes, _ = find_episodes(times, smoothed, cfg.analysis.fps, sm)
    candidates = segments.build(
        episodes, whistles, times, smoothed, cfg.analysis.fps, sm,
        pre_buffer_s=cfg.segment.pre_buffer_s,
        post_buffer_s=cfg.segment.post_buffer_s,
        clip_duration=info.duration,
        ignore_ranges=cfg.ignore_ranges.get(clip, []),
    )

    destination = analysis_dir / f"{info.path.stem}__segments.json"
    segments.write(
        destination, candidates, clip=clip,
        meta={
            "anchors": len(whistles),
            "episodes": len(episodes),
            "realtime_factor": realtime,
            "pre_buffer_s": cfg.segment.pre_buffer_s,
            "post_buffer_s": cfg.segment.post_buffer_s,
        },
    )

    table = Table("tier", "count")
    for tier in Tier:
        table.add_row(tier.value, str(sum(1 for c in candidates if c.tier is tier)))
    console.print(table)
    console.print(f"[green]segments[/] {destination}")


@app.command("cut")
def cut(
    game_dir: Path = typer.Argument(..., help="Game directory"),
    from_labels: bool = typer.Option(
        False, "--from-labels",
        help="Cut the hand-corrected label spans instead of detected segments.",
    ),
    reencode: bool = typer.Option(False, "--reencode", help="Frame-exact boundaries"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only, write nothing"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing plays"),
) -> None:
    """Cut accepted plays into plays/ and write the manifest.

    Automated segments are not wired in yet: span quality has not cleared its
    checkpoint, and a clip that clips the snap cannot be fixed by a keystroke
    in review. ``--from-labels`` needs no span estimation at all, so it ships
    finished games today and exercises the whole ffmpeg path end to end.
    """
    from . import cut as cutter
    from . import labels as labels_mod

    if not from_labels:
        console.print(
            "[yellow]automated segments are gated pending span quality; "
            "re-run with --from-labels[/]"
        )
        raise typer.Exit(1)

    analysis_dir = game_dir / "analysis"
    plays_dir = game_dir / "plays"
    items: list[cutter.Cut] = []
    number = 0

    for path in probe.find_clips(game_dir):
        info = probe.probe(path)
        label_file = analysis_dir / f"{path.stem}__labels_corrected.csv"
        rows, warnings = labels_mod.read(label_file, clip_duration=info.duration)
        for warning in warnings:
            console.print(f"[yellow]{path.name}: {warning}")
        for row in rows:
            if not row.is_kept:
                continue
            number += 1
            items.append(
                cutter.Cut(
                    play=number, clip=info, start=row.start, end=row.end,
                    partial=row.partial, notes=row.notes,
                )
            )

    if not items:
        console.print(f"[red]no kept labels found under {analysis_dir}")
        raise typer.Exit(1)

    table = Table("play", "clip", "start", "end", "dur", "partial")
    starts: dict[int, float] = {}
    for item in items:
        destination = plays_dir / item.filename
        if destination.exists() and not force and not dry_run:
            console.print(f"[dim]exists, skipping {destination.name} (use --force)")
            starts[item.play] = item.start
            continue
        _, actual = cutter.cut_one(
            item, destination, reencode=reencode, dry_run=dry_run
        )
        starts[item.play] = actual
        table.add_row(
            str(item.play), item.clip.name, f"{actual:.2f}", f"{item.end:.2f}",
            f"{item.end - actual:.1f}s", "yes" if item.partial else "",
        )

    console.print(table)
    if dry_run:
        console.print(f"[yellow]dry run: {len(items)} plays planned, nothing written")
        return

    manifest = plays_dir / "manifest.csv"
    cutter.write_manifest(manifest, items, starts)
    console.print(f"[green]{len(items)} plays[/] → {plays_dir}")
    console.print(f"[green]manifest[/] {manifest}")


if __name__ == "__main__":
    app()
