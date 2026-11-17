from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(
    "data/external/geolife/raw/Geolife Trajectories 1.3/Data"
)

OUT_DIR = Path(
    "outputs/trip_distance_validation/geolife/full_cohort"
)

MIN_DURATION_MIN = 10.0
MIN_POINTS = 300
MAX_CONTIGUOUS_GAP_S = 5.0
MAX_MEDIAN_DT_S = 5.0


def read_plt(path: Path) -> pd.DataFrame:
    columns = [
        "latitude",
        "longitude",
        "unused",
        "altitude_ft",
        "serial_date",
        "date",
        "time",
    ]

    df = pd.read_csv(
        path,
        skiprows=6,
        header=None,
        names=columns,
    )

    if df.empty:
        return df

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str)
        + " "
        + df["time"].astype(str),
        errors="coerce",
    )

    return (
        df.dropna(
            subset=[
                "timestamp",
                "latitude",
                "longitude",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(
        path,
        sep="\t",
    )

    labels.columns = [
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        for c in labels.columns
    ]

    labels["start_time"] = pd.to_datetime(
        labels["start_time"],
        errors="coerce",
    )

    labels["end_time"] = pd.to_datetime(
        labels["end_time"],
        errors="coerce",
    )

    return labels.dropna(
        subset=[
            "start_time",
            "end_time",
            "transportation_mode",
        ]
    )


def load_user_points(
    user_dir: Path,
) -> pd.DataFrame:
    frames = []

    for path in sorted(
        (user_dir / "Trajectory").glob("*.plt")
    ):
        df = read_plt(path)

        if df.empty:
            continue

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def split_dense_blocks(
    df: pd.DataFrame,
) -> list[pd.DataFrame]:
    if len(df) < 2:
        return []

    dt_s = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    new_block = (
        dt_s.isna()
        |
        (dt_s > MAX_CONTIGUOUS_GAP_S)
        |
        (dt_s <= 0)
    )

    block_id = new_block.cumsum()

    blocks = []

    for _, block in df.groupby(
        block_id,
        sort=False,
    ):
        block = block.reset_index(
            drop=True
        )

        if len(block) < 2:
            continue

        blocks.append(block)

    return blocks


def block_metrics(
    block: pd.DataFrame,
) -> dict:
    dt = (
        block["timestamp"]
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    duration_s = (
        block["timestamp"].iloc[-1]
        - block["timestamp"].iloc[0]
    ).total_seconds()

    return {
        "n_points":
            len(block),
        "duration_min":
            duration_s / 60.0,
        "median_dt_s":
            float(dt.median()),
        "p95_dt_s":
            float(dt.quantile(0.95)),
        "max_dt_s":
            float(dt.max()),
    }


def main() -> None:
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    segment_dir = (
        OUT_DIR / "segments"
    )

    segment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    user_dirs = sorted(
        [
            p
            for p in ROOT.iterdir()
            if p.is_dir()
            and p.name.isdigit()
        ]
    )

    rows = []

    users_with_labels = 0
    labels_examined = 0
    candidate_blocks = 0

    print("=" * 100)
    print("BUILDING GEOLIFE VALIDATION COHORT")
    print("=" * 100)

    for user_dir in tqdm(
        user_dirs,
        desc="GeoLife users",
    ):
        labels_path = (
            user_dir / "labels.txt"
        )

        if not labels_path.exists():
            continue

        users_with_labels += 1

        labels = read_labels(
            labels_path
        )

        points = load_user_points(
            user_dir
        )

        if points.empty:
            continue

        for label_index, label in (
            labels.iterrows()
        ):
            labels_examined += 1

            mask = (
                (
                    points["timestamp"]
                    >= label["start_time"]
                )
                &
                (
                    points["timestamp"]
                    <= label["end_time"]
                )
            )

            labeled = (
                points.loc[mask]
                .copy()
                .reset_index(drop=True)
            )

            if len(labeled) < 2:
                continue

            blocks = split_dense_blocks(
                labeled
            )

            for block_index, block in enumerate(
                blocks
            ):
                candidate_blocks += 1

                metrics = block_metrics(
                    block
                )

                if (
                    metrics["n_points"]
                    < MIN_POINTS
                ):
                    continue

                if (
                    metrics["duration_min"]
                    < MIN_DURATION_MIN
                ):
                    continue

                if (
                    metrics["median_dt_s"]
                    > MAX_MEDIAN_DT_S
                ):
                    continue

                segment_id = (
                    f"user_{user_dir.name}_"
                    f"label_{label_index:04d}_"
                    f"block_{block_index:03d}"
                )

                output_file = (
                    segment_dir
                    / f"{segment_id}.parquet"
                )

                block = block.copy()

                block[
                    "user_id"
                ] = user_dir.name

                block[
                    "segment_id"
                ] = segment_id

                block[
                    "transportation_mode"
                ] = str(
                    label[
                        "transportation_mode"
                    ]
                ).strip()

                block.to_parquet(
                    output_file,
                    index=False,
                )

                rows.append(
                    {
                        "segment_id":
                            segment_id,
                        "user_id":
                            user_dir.name,
                        "label_index":
                            label_index,
                        "block_index":
                            block_index,
                        "transportation_mode":
                            str(
                                label[
                                    "transportation_mode"
                                ]
                            ).strip(),
                        **metrics,
                        "output_file":
                            str(output_file),
                    }
                )

    cohort = pd.DataFrame(
        rows
    )

    cohort.to_csv(
        OUT_DIR
        / "validation_cohort.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("VALIDATION COHORT SUMMARY")
    print("=" * 100)

    print(
        f"Users with labels: "
        f"{users_with_labels}"
    )

    print(
        f"Label intervals examined: "
        f"{labels_examined:,}"
    )

    print(
        f"Dense candidate blocks: "
        f"{candidate_blocks:,}"
    )

    print(
        f"Accepted validation blocks: "
        f"{len(cohort):,}"
    )

    if cohort.empty:
        raise RuntimeError(
            "No validation blocks met criteria."
        )

    print()
    print("BLOCKS BY TRANSPORTATION MODE")
    print("-" * 100)

    mode_summary = (
        cohort.groupby(
            "transportation_mode"
        )
        .agg(
            n_blocks=(
                "segment_id",
                "count",
            ),
            n_users=(
                "user_id",
                "nunique",
            ),
            median_duration_min=(
                "duration_min",
                "median",
            ),
            median_points=(
                "n_points",
                "median",
            ),
            median_dt_s=(
                "median_dt_s",
                "median",
            ),
        )
        .sort_values(
            "n_blocks",
            ascending=False,
        )
    )

    print(
        mode_summary.to_string()
    )

    print()
    print("BLOCKS BY USER")
    print("-" * 100)

    user_summary = (
        cohort.groupby(
            "user_id"
        )
        .agg(
            n_blocks=(
                "segment_id",
                "count",
            ),
            n_modes=(
                "transportation_mode",
                "nunique",
            ),
            total_duration_min=(
                "duration_min",
                "sum",
            ),
        )
        .sort_values(
            "n_blocks",
            ascending=False,
        )
    )

    print(
        user_summary.head(
            20
        ).to_string()
    )

    print()
    print("Saved cohort to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
