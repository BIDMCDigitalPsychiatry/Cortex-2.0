import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd() / "src"))

from cortex2_engine.features.gps_degradation import (
    downsample_points_to_time_bins,
)
from cortex2_engine.features.gps_literature import (
    compute_literature_gps_features,
)


BASE = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic"
)

ALIGNMENT = (
    BASE
    / "multires_clinical_benchmark"
    / "gps_multires_weekly_alignment_resolution_aware.parquet"
)

RAW = (
    BASE
    / "feature_reliability"
    / "gps_clinical"
    / "gps_clinical_weekly_raw_points.parquet"
)

OUTDIR = (
    BASE
    / "multires_clinical_benchmark"
    / "saeb_replication"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PARQUET = (
    OUTDIR
    / "saeb_phq_14d_5min_features.parquet"
)

OUT_CSV = (
    OUTDIR
    / "saeb_phq_14d_5min_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_DAYS = 14

# Saeb-compatible sampling condition
BIN_SIZE_MIN = 5
BIN_SIZE_MS = (
    BIN_SIZE_MIN
    * 60
    * 1000
)

RANDOM_SEED = 0


# ============================================================
# CLINICAL BASELINE
# ============================================================

clinical = pd.read_parquet(
    ALIGNMENT,
    columns=[
        "participant_id",
        "assessment_date",
        "resolution",
        "depression_score",
        "anxiety_score",
    ],
)

clinical = clinical.loc[
    clinical["resolution"]
    .astype(str)
    .eq("native")
].copy()

clinical["assessment_date"] = pd.to_datetime(
    clinical["assessment_date"],
    utc=True,
)

clinical = clinical.dropna(
    subset=[
        "participant_id",
        "assessment_date",
        "depression_score",
    ]
)

# Saeb-style cross-sectional replication:
# one baseline assessment per participant.
clinical = (
    clinical
    .sort_values(
        [
            "participant_id",
            "assessment_date",
        ]
    )
    .groupby(
        "participant_id",
        as_index=False,
    )
    .first()
)

print()
print("=" * 80)
print("SAEB 14-DAY PHQ-9 REPLICATION")
print("=" * 80)

print(
    f"Participants with baseline PHQ-9: "
    f"{clinical['participant_id'].nunique():,}"
)

print(
    f"PHQ-9 range: "
    f"{clinical['depression_score'].min():.0f}"
    f"–"
    f"{clinical['depression_score'].max():.0f}"
)

print(
    f"Sampling condition: {BIN_SIZE_MIN} minutes"
)

print(
    f"Window: assessment date + {WINDOW_DAYS} days"
)


# ============================================================
# RESUME SUPPORT
# ============================================================

if OUT_PARQUET.exists():

    existing = pd.read_parquet(
        OUT_PARQUET
    )

    completed = set(
        existing[
            "participant_id"
        ].astype(str)
    )

    rows = existing.to_dict(
        orient="records"
    )

    print(
        f"\nResuming: "
        f"{len(completed):,} participants already complete."
    )

else:

    completed = set()
    rows = []


# ============================================================
# PARTICIPANT LOOP
# ============================================================

total = len(clinical)

overall_start = time.perf_counter()


for i, row in clinical.iterrows():

    pid = str(
        row["participant_id"]
    )

    if pid in completed:
        continue

    t0 = time.perf_counter()

    assessment_date = row[
        "assessment_date"
    ]

    window_start = assessment_date

    window_end = (
        assessment_date
        + pd.Timedelta(
            days=WINDOW_DAYS
        )
    )

    start_ms = int(
        window_start.timestamp()
        * 1000
    )

    end_ms = int(
        window_end.timestamp()
        * 1000
    )


    # --------------------------------------------------------
    # Read only this participant and this time window.
    # --------------------------------------------------------

    try:

        gps = pd.read_parquet(
            RAW,
            filters=[
                (
                    "participant_id",
                    "==",
                    pid,
                ),
                (
                    "timestamp",
                    ">=",
                    start_ms,
                ),
                (
                    "timestamp",
                    "<",
                    end_ms,
                ),
            ],
        )

    except Exception:

        # Safe fallback if parquet filtering by timestamp
        # is unsupported for this file layout.

        gps = pd.read_parquet(
            RAW,
            filters=[
                (
                    "participant_id",
                    "==",
                    pid,
                )
            ],
        )

        gps = gps.loc[
            gps["timestamp"]
            .between(
                start_ms,
                end_ms,
                inclusive="left",
            )
        ].copy()


    raw_points = len(gps)

    observed_days = (
        gps["date"].nunique()
        if "date" in gps.columns
        else np.nan
    )


    # --------------------------------------------------------
    # 5-minute Saeb-compatible sampling representation
    # --------------------------------------------------------

    if raw_points:

        sampled = (
            downsample_points_to_time_bins(
                gps,
                window_start_ms=start_ms,
                window_end_ms=end_ms,
                bin_size_ms=BIN_SIZE_MS,
                seed=RANDOM_SEED,
            )
        )

    else:

        sampled = gps.copy()


    sampled_points = len(
        sampled
    )


    # --------------------------------------------------------
    # Literature-compatible feature extraction
    # --------------------------------------------------------

    features = (
        compute_literature_gps_features(
            sampled,
            random_state=RANDOM_SEED,
        )
    )


    # --------------------------------------------------------
    # Save clinical + sensing metadata
    # --------------------------------------------------------

    result = {
        "participant_id": pid,
        "assessment_date": assessment_date,
        "window_start": window_start,
        "window_end": window_end,

        "depression_score": float(
            row["depression_score"]
        ),

        "anxiety_score": (
            float(
                row["anxiety_score"]
            )
            if pd.notna(
                row["anxiety_score"]
            )
            else np.nan
        ),

        "sampling_resolution_min": (
            BIN_SIZE_MIN
        ),

        "raw_gps_points": (
            raw_points
        ),

        "sampled_gps_points": (
            sampled_points
        ),

        "observed_gps_days": (
            observed_days
        ),

        **features,
    }


    rows.append(
        result
    )


    # --------------------------------------------------------
    # Checkpoint after each participant
    # --------------------------------------------------------

    out = pd.DataFrame(
        rows
    )

    out.to_parquet(
        OUT_PARQUET,
        index=False,
    )

    out.to_csv(
        OUT_CSV,
        index=False,
    )


    elapsed = (
        time.perf_counter()
        - t0
    )

    done = len(
        set(
            out[
                "participant_id"
            ].astype(str)
        )
    )

    print(
        f"[{done:3d}/{total}] "
        f"{pid} | "
        f"PHQ={row['depression_score']:.0f} | "
        f"days={observed_days} | "
        f"raw={raw_points:,} | "
        f"5m={sampled_points:,} | "
        f"clusters="
        f"{features['literature_n_clusters']} | "
        f"{elapsed:.1f}s"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

out = pd.DataFrame(
    rows
)

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)

print(
    f"Participants: "
    f"{out['participant_id'].nunique():,}"
)

print(
    f"Participants with GPS points: "
    f"{(out['sampled_gps_points'] > 0).sum():,}"
)

print(
    f"Median observed GPS days: "
    f"{out['observed_gps_days'].median():.1f}"
)

print(
    f"Median 5-min GPS samples: "
    f"{out['sampled_gps_points'].median():,.0f}"
)

print(
    f"Total runtime: "
    f"{time.perf_counter() - overall_start:.1f}s"
)

print()
print("Saved:")
print(OUT_PARQUET)
print(OUT_CSV)
