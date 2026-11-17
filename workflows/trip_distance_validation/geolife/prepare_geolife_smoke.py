from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/external/geolife/raw")
OUT = Path(
    "outputs/trip_distance_validation/geolife/smoke"
)

N_SEGMENTS = 3

MIN_POINTS = 300
MIN_DURATION_MIN = 5.0
MAX_MEDIAN_DT_S = 5.0


def find_data_root(root: Path) -> Path:
    candidates = [
        p
        for p in root.rglob("Data")
        if p.is_dir()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find GeoLife Data directory under {root}"
        )

    for candidate in candidates:
        if any(
            child.is_dir() and child.name.isdigit()
            for child in candidate.iterdir()
        ):
            return candidate

    return candidates[0]


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

    for col in [
        "latitude",
        "longitude",
        "altitude_ft",
        "serial_date",
    ]:
        df[col] = pd.to_numeric(
            df[col],
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
        str(col).strip().lower().replace(" ", "_")
        for col in labels.columns
    ]

    expected = {
        "start_time",
        "end_time",
        "transportation_mode",
    }

    missing = expected - set(labels.columns)

    if missing:
        raise ValueError(
            f"{path}: missing label columns {sorted(missing)}"
        )

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


def load_user_points(user_dir: Path) -> pd.DataFrame:
    files = sorted(
        (user_dir / "Trajectory").glob("*.plt")
    )

    frames = []

    for path in files:
        df = read_plt(path)

        if df.empty:
            continue

        df["source_file"] = path.name
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


def segment_metrics(
    segment: pd.DataFrame,
) -> dict:
    timestamps = segment["timestamp"]

    dt_s = (
        timestamps.diff()
        .dt.total_seconds()
        .dropna()
    )

    duration_min = (
        (
            timestamps.iloc[-1]
            - timestamps.iloc[0]
        ).total_seconds()
        / 60.0
    )

    return {
        "n_points": len(segment),
        "duration_min": duration_min,
        "median_dt_s": (
            float(dt_s.median())
            if not dt_s.empty
            else np.nan
        ),
        "p95_dt_s": (
            float(dt_s.quantile(0.95))
            if not dt_s.empty
            else np.nan
        ),
        "max_dt_s": (
            float(dt_s.max())
            if not dt_s.empty
            else np.nan
        ),
    }


def main() -> None:
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_root = find_data_root(ROOT)

    print("=" * 100)
    print("GEOLIFE LABELED HIGH-RESOLUTION SMOKE PREPARATION")
    print("=" * 100)
    print(f"Data root: {data_root}")

    selected = []
    summary_rows = []

    user_dirs = sorted(
        [
            p
            for p in data_root.iterdir()
            if p.is_dir()
            and p.name.isdigit()
        ]
    )

    n_users_with_labels = 0
    n_labels_examined = 0

    for user_dir in user_dirs:
        labels_path = user_dir / "labels.txt"

        if not labels_path.exists():
            continue

        n_users_with_labels += 1

        labels = read_labels(labels_path)

        if labels.empty:
            continue

        points = load_user_points(user_dir)

        if points.empty:
            continue

        for label_index, label in labels.iterrows():
            n_labels_examined += 1

            mask = (
                (points["timestamp"] >= label["start_time"])
                &
                (points["timestamp"] <= label["end_time"])
            )

            segment = (
                points.loc[mask]
                .copy()
                .reset_index(drop=True)
            )

            if len(segment) < 2:
                continue

            metrics = segment_metrics(segment)

            if metrics["n_points"] < MIN_POINTS:
                continue

            if metrics["duration_min"] < MIN_DURATION_MIN:
                continue

            if (
                not np.isfinite(metrics["median_dt_s"])
                or metrics["median_dt_s"] > MAX_MEDIAN_DT_S
            ):
                continue

            segment_id = (
                f"user_{user_dir.name}_"
                f"label_{label_index:04d}"
            )

            segment["user_id"] = user_dir.name
            segment["segment_id"] = segment_id
            segment["transportation_mode"] = str(
                label["transportation_mode"]
            ).strip()

            output_file = OUT / f"{segment_id}.csv"

            segment.to_csv(
                output_file,
                index=False,
            )

            summary_rows.append(
                {
                    "segment_id": segment_id,
                    "user_id": user_dir.name,
                    "transportation_mode": str(
                        label["transportation_mode"]
                    ).strip(),
                    "start_time": segment["timestamp"].iloc[0],
                    "end_time": segment["timestamp"].iloc[-1],
                    **metrics,
                    "output_file": str(output_file),
                }
            )

            selected.append(segment_id)

            print()
            print(
                f"Selected {len(selected)}/{N_SEGMENTS}: "
                f"{segment_id}"
            )
            print(
                f"  mode={summary_rows[-1]['transportation_mode']} | "
                f"points={metrics['n_points']:,} | "
                f"duration={metrics['duration_min']:.1f} min | "
                f"median_dt={metrics['median_dt_s']:.1f} s | "
                f"p95_dt={metrics['p95_dt_s']:.1f} s"
            )

            if len(selected) >= N_SEGMENTS:
                break

        if len(selected) >= N_SEGMENTS:
            break

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUT / "smoke_segments.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"Users with labels encountered: "
        f"{n_users_with_labels}"
    )
    print(
        f"Label intervals examined: "
        f"{n_labels_examined}"
    )
    print(
        f"Segments selected: "
        f"{len(selected)}"
    )

    if not summary.empty:
        print()
        print(
            summary[
                [
                    "segment_id",
                    "transportation_mode",
                    "n_points",
                    "duration_min",
                    "median_dt_s",
                    "p95_dt_s",
                    "max_dt_s",
                ]
            ].to_string(index=False)
        )

    if len(selected) < N_SEGMENTS:
        raise RuntimeError(
            f"Only found {len(selected)} qualifying segments; "
            f"expected {N_SEGMENTS}."
        )

    print()
    print("Saved smoke segments to:")
    print(OUT)


if __name__ == "__main__":
    main()
