from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "../cortex2_outputs/digital_clinic/quality"
)

HOURLY = (
    ROOT
    / "validation/CHO370_hourly_quality.parquet"
)

hourly = pd.read_parquet(HOURLY)


# ============================================================
# 1. Hourly -> daily DQ reconciliation
# ============================================================

print("=" * 70)
print("HOURLY -> DAILY RECONCILIATION")
print("=" * 70)


for stream, daily_file in [
    ("gps", "gps_daily.parquet"),
    ("accelerometer", "accelerometer_daily.parquet"),
]:

    h = hourly[
        hourly["stream"].eq(stream)
    ].copy()

    h["date_key"] = (
        pd.to_datetime(
            h["hour_start_utc"],
            utc=True,
        )
        .dt.strftime("%Y-%m-%d")
    )

    reconstructed = (
        h.groupby(
            "date_key",
            as_index=False,
        )
        .agg(
            hourly_occupied=(
                "n_occupied_bins",
                "sum",
            ),
            hourly_bins=(
                "n_bins",
                "sum",
            ),
            all_hours_unavailable=(
                "source_unavailable",
                "all",
            ),
        )
    )

    reconstructed[
        "reconstructed_daily_dq"
    ] = (
        reconstructed["hourly_occupied"]
        / reconstructed["hourly_bins"]
    )

    reconstructed.loc[
        reconstructed[
            "all_hours_unavailable"
        ],
        "reconstructed_daily_dq",
    ] = np.nan

    daily = pd.read_parquet(
        ROOT / daily_file
    )

    daily = daily[
        daily["participant_id"]
        .astype(str)
        .eq("CHO370")
    ].copy()

    daily["date_key"] = (
        pd.to_datetime(
            daily["date"],
            utc=True,
        )
        .dt.strftime("%Y-%m-%d")
    )

    merged = reconstructed.merge(
        daily[
            [
                "date_key",
                "data_quality",
            ]
        ],
        on="date_key",
        how="inner",
        validate="one_to_one",
    )

    comparable = merged[
        merged["data_quality"].notna()
        &
        merged[
            "reconstructed_daily_dq"
        ].notna()
    ].copy()

    comparable["abs_diff"] = (
        comparable["data_quality"]
        -
        comparable[
            "reconstructed_daily_dq"
        ]
    ).abs()

    print(f"\n{stream.upper()}")
    print(
        "comparable days:",
        len(comparable),
    )
    print(
        "max abs difference:",
        comparable["abs_diff"].max(),
    )
    print(
        "matches:",
        int(
            np.isclose(
                comparable["data_quality"],
                comparable[
                    "reconstructed_daily_dq"
                ],
                atol=1e-12,
                rtol=0,
            ).sum()
        ),
        "/",
        len(comparable),
    )


# ============================================================
# 2. GPS / ACC gap alignment
# ============================================================

print("\n" + "=" * 70)
print("GPS / ACC HOURLY GAP ALIGNMENT")
print("=" * 70)


available = hourly[
    ~hourly["source_unavailable"]
].copy()

wide = (
    available.pivot(
        index=[
            "episode_id",
            "hour_index",
            "hour_start_utc",
        ],
        columns="stream",
        values="hour_observed",
    )
    .reset_index()
)

for stream in [
    "gps",
    "accelerometer",
    "screen",
]:
    wide[stream] = (
        wide[stream]
        .fillna(False)
        .astype(bool)
    )


wide["gps_gap"] = ~wide["gps"]
wide["acc_gap"] = ~wide["accelerometer"]


print("\nGPS observed x ACC observed:")

print(
    pd.crosstab(
        wide["gps"],
        wide["accelerometer"],
        margins=True,
    ).to_string()
)


print("\nGap counts:")

print(
    "GPS gap hours:",
    int(wide["gps_gap"].sum()),
)

print(
    "ACC gap hours:",
    int(wide["acc_gap"].sum()),
)

print(
    "Both GPS + ACC gap:",
    int(
        (
            wide["gps_gap"]
            &
            wide["acc_gap"]
        ).sum()
    ),
)

print(
    "GPS gap while ACC observed:",
    int(
        (
            wide["gps_gap"]
            &
            wide["accelerometer"]
        ).sum()
    ),
)

print(
    "ACC gap while GPS observed:",
    int(
        (
            wide["acc_gap"]
            &
            wide["gps"]
        ).sum()
    ),
)


gps_gap = wide[
    wide["gps_gap"]
].copy()

print(
    "\nGPS-gap hours with screen events:",
    int(
        gps_gap["screen"].sum()
    ),
    "/",
    len(gps_gap),
)


# ============================================================
# 3. Contiguous gap runs
# ============================================================

def gap_runs(df, gap_column):

    z = (
        df[
            [
                "hour_index",
                "hour_start_utc",
                gap_column,
            ]
        ]
        .sort_values("hour_index")
        .reset_index(drop=True)
    )

    groups = (
        z[gap_column]
        .ne(
            z[gap_column].shift()
        )
        .cumsum()
    )

    rows = []

    for _, g in z[
        z[gap_column]
    ].groupby(
        groups[z[gap_column]]
    ):

        rows.append({
            "start_utc":
                g["hour_start_utc"].iloc[0],

            "end_utc":
                g["hour_start_utc"].iloc[-1]
                + pd.Timedelta(hours=1),

            "duration_hours":
                len(g),
        })

    return pd.DataFrame(rows)


for label, column in [
    ("GPS", "gps_gap"),
    ("ACC", "acc_gap"),
]:

    runs = gap_runs(
        wide,
        column,
    )

    print(f"\n{label} gap runs:")

    print(
        "number of runs:",
        len(runs),
    )

    if len(runs):

        print(
            "median duration (h):",
            runs[
                "duration_hours"
            ].median(),
        )

        print(
            "longest duration (h):",
            runs[
                "duration_hours"
            ].max(),
        )

        print("\nLongest 10:")

        print(
            runs.sort_values(
                "duration_hours",
                ascending=False,
            )
            .head(10)
            .to_string(index=False)
        )
