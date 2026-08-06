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


if __name__ == "__main__":
    app()
