import json
from pathlib import Path

import typer

from cortex2_engine.io.study_inventory import (
    run_inventory,
)

app = typer.Typer(
    help=(
        "Cortex 2.0 — Reliability-Aware Personalized "
        "Health Phenotyping Engine"
    )
)


@app.command()
def version():
    """Show Cortex 2.0 version."""
    typer.echo("Cortex 2.0 0.1.0")


@app.command()
def info():
    """Show the Cortex 2.0 processing framework."""
    typer.echo(
        "\n"
        "Observability\n"
        "  -> Feature extraction\n"
        "  -> Analysis-ready features\n"
        "  -> Personal baseline\n"
        "  -> Behavioral deviations\n"
        "  -> Clinical alignment\n"
        "  -> Reliability/degradation\n"
        "  -> Transdiagnostic transportability\n"
    )


@app.command()
def inventory(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        readable=True,
    )
):
    """Validate a study and build its canonical inventory."""

    result = run_inventory(
        config
    )

    typer.echo(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
