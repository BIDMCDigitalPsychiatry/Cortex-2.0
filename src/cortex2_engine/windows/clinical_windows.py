from pathlib import Path

import pandas as pd

from cortex2_engine.config import WindowsConfig


def load_clinical_windows(
    config: WindowsConfig,
) -> pd.DataFrame:

    path = Path(config.path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        config.participant_id_column,
        config.sensor_id_column,
        config.start_column,
        config.end_column,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required window columns: {missing}"
        )

    out = df[required].copy()

    out.columns = [
        "participant_id",
        "sensor_id",
        "clinical_start",
        "clinical_end",
    ]

    out["participant_id"] = (
        out["participant_id"].astype(str)
    )

    out["sensor_id"] = (
        out["sensor_id"].astype(str)
    )

    out["clinical_start"] = pd.to_datetime(
        out["clinical_start"],
        errors="raise",
    ).dt.normalize()

    out["clinical_end"] = pd.to_datetime(
        out["clinical_end"],
        errors="raise",
    ).dt.normalize()

    bad = (
        out["clinical_end"]
        < out["clinical_start"]
    )

    if bad.any():
        raise ValueError(
            "Clinical end precedes clinical start for "
            f"{bad.sum()} participant episodes."
        )

    duplicates = out.duplicated(
        ["participant_id", "sensor_id"],
        keep=False,
    )

    if duplicates.any():
        examples = (
            out.loc[
                duplicates,
                ["participant_id", "sensor_id"]
            ]
            .head()
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate participant/sensor episodes found. "
            f"Examples: {examples}"
        )

    out["window_days"] = (
        out["clinical_end"]
        - out["clinical_start"]
    ).dt.days + 1

    return out.sort_values(
        ["participant_id", "sensor_id"]
    ).reset_index(drop=True)


def build_daily_grid(
    windows: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for r in windows.itertuples(index=False):

        dates = pd.date_range(
            r.clinical_start,
            r.clinical_end,
            freq="D",
        )

        part = pd.DataFrame({
            "participant_id":
                r.participant_id,

            "sensor_id":
                r.sensor_id,

            "clinical_date":
                dates,

            "clinical_day":
                range(1, len(dates) + 1),
        })

        rows.append(part)

    if not rows:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "sensor_id",
                "clinical_date",
                "clinical_day",
            ]
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )
