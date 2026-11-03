import datetime as dt
import json
from pathlib import Path
from typing import Annotated

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
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            readable=True,
        ),
    ],
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


@app.command()
def features(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, readable=True),
    ],
    start: Annotated[
        str | None,
        typer.Option("--start", help="First local date (YYYY-MM-DD), inclusive."),
    ] = None,
    end: Annotated[
        str | None,
        typer.Option("--end", help="Last local date (YYYY-MM-DD), inclusive."),
    ] = None,
):
    """Compute daily features and write the long parquet table."""

    from cortex2_engine.adapters.factory import build_adapter, build_study_episodes
    from cortex2_engine.config import load_study_config
    from cortex2_engine.features.daily_driver import write_daily_features

    try:
        start_date = dt.date.fromisoformat(start) if start else None
        end_date = dt.date.fromisoformat(end) if end else None
    except ValueError as exc:
        raise typer.BadParameter(f"dates must be YYYY-MM-DD: {exc}") from exc

    if start_date and end_date and end_date < start_date:
        raise typer.BadParameter("--end must not precede --start")

    study = load_study_config(config)

    path = write_daily_features(
        build_adapter(study),
        build_study_episodes(study),
        study,
        start_date=start_date,
        end_date=end_date,
    )

    typer.echo(str(path))


if __name__ == "__main__":
    app()
