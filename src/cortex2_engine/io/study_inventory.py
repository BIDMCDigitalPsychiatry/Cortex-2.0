import json
from datetime import datetime, timezone
from pathlib import Path

from cortex2_engine.config import load_study_config
from cortex2_engine.io.inventory import build_raw_file_manifest
from cortex2_engine.windows.clinical_windows import (
    build_daily_grid,
    load_clinical_windows,
)


def run_inventory(config_path):

    config = load_study_config(
        config_path
    )

    root = Path(config.output_dir)
    inv = root / "inventory"
    prov = root / "provenance"

    inv.mkdir(
        parents=True,
        exist_ok=True,
    )

    prov.mkdir(
        parents=True,
        exist_ok=True,
    )

    participants = load_clinical_windows(
        config.windows
    )

    daily = build_daily_grid(
        participants
    )

    files = build_raw_file_manifest(
        config
    )

    participants.to_parquet(
        inv / "participants.parquet",
        index=False,
    )

    daily.to_parquet(
        inv / "daily_grid.parquet",
        index=False,
    )

    files.to_parquet(
        inv / "raw_file_manifest.parquet",
        index=False,
    )

    metadata = {
        "study":
            config.name,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "timezone":
            config.timezone,

        "participants":
            int(len(participants)),

        "participant_days":
            int(len(daily)),

        "raw_files":
            int(len(files)),

        "streams":
            sorted(
                config.streams.keys()
            ),
    }

    with (
        prov / "inventory.json"
    ).open("w") as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    return metadata
