"""
Multi-resolution GPS clinical benchmark.

Purpose
-------
Quantify how temporal GPS sampling resolution affects:

1. Cortex-2.0 GPS feature values
2. Technical fidelity relative to native GPS
3. Daily EMA alignment
4. Weekly questionnaire alignment

Important
---------
Temporal resolution is manipulated by retaining at most one real
GPS observation per occupied time bin.

No interpolation is performed.

The validated Cortex GPS semantics remain fixed:
- UTC day windows
- max supported temporal gap = 60 seconds
- accuracy-aware spatial semantics
- trajectory plausibility <= 1000 km/h
- fixed-anchor location episodes
- compatibility multiplier = 2.0
- time-weighted location features

Thus this first experiment changes sampling resolution ONLY.
"""

from pathlib import Path
import argparse
import hashlib
import gc
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


from cortex2_engine.features.gps_degradation import (
    downsample_points_to_time_bins,
)

from cortex2_engine.features.gps_observability import (
    compute_gps_temporal_observability,
)

from cortex2_engine.features.gps_intervals import (
    build_gps_intervals,
)

from cortex2_engine.features.gps_spatial_semantics import (
    add_gps_spatial_semantics,
)

from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility,
)

from cortex2_engine.features.gps_mobility_features import (
    compute_daily_mobility_features,
)

from cortex2_engine.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)

from cortex2_engine.features.gps_place_candidates import (
    build_place_candidates,
)

from cortex2_engine.features.gps_location_features import (
    compute_daily_location_features,
)


# ============================================================
# CONFIG
# ============================================================

DC = Path("/shared/Digital Clinic")

OUT_DIR = (
    DC
    / "cortex2_outputs"
    / "digital_clinic"
    / "multires_clinical_benchmark"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


GPS_CLINICAL_DIR = (
    DC
    / "cortex2_outputs"
    / "digital_clinic"
    / "feature_reliability"
    / "gps_clinical"
)


DAILY_EMA_DIR = (
    DC
    / "cortex2_outputs"
    / "digital_clinic"
    / "clinical"
    / "daily_ema"
)


DAILY_EMA_RAW = (
    DAILY_EMA_DIR
    / "daily_ema_gps_raw_points.parquet"
)


DAILY_EMA_CLINICAL = (
    DAILY_EMA_DIR
    / "daily_ema_gps_features.parquet"
)


WEEKLY_CLINICAL = (
    GPS_CLINICAL_DIR
    / "gps_clinical_weekly_features_with_home.parquet"
)


MAX_GAP_MS = 60_000

COMPATIBILITY_MULTIPLIER = 2.0

MAX_PLAUSIBLE_SPEED_KMH = 1000.0


DEFAULT_RESOLUTIONS_MIN = [
    0,      # native
    1,
    5,
    10,
    15,
    30,
    60,
    120,
    240,
]


RAW_COLUMNS = [
    "timestamp",
    "latitude",
    "longitude",
    "accuracy",
    "participant_id",
    "date",
    "day_start_ms",
    "day_end_ms",
]


CORE_FEATURES = [
    "supported_displacement_distance_m",
    "supported_radius_of_gyration_m",
    "location_entropy_nats",
    "effective_number_places",
    "dominant_place_fraction",
    "n_observed_places",
]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--smoke",
    action="store_true",
    help=(
        "Run a small validation subset before the full benchmark."
    ),
)

parser.add_argument(
    "--max-participants",
    type=int,
    default=None,
)

parser.add_argument(
    "--support-mode",
    choices=[
        "fixed",
        "resolution-aware",
    ],
    default="fixed",
    help=(
        "fixed: preserve validated Cortex 60-second support semantics; "
        "resolution-aware: use max(60 s, 2 × imposed sampling interval)."
    ),
)

args = parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def stable_seed(
    participant_id,
    date,
    resolution_min,
):
    text = (
        f"{participant_id}|"
        f"{date}|"
        f"{resolution_min}"
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


def resolution_label(minutes):
    if minutes == 0:
        return "native"

    return f"{minutes}min"


def find_weekly_raw_path():
    """
    Find an already-extracted clinical GPS raw-point table.

    We deliberately do not scan the full raw Digital Clinic GPS
    directory here.
    """

    patterns = [
        "*raw*points*.parquet",
        "*raw*gps*.parquet",
        "*gps*raw*.parquet",
    ]

    candidates = []

    for pattern in patterns:
        candidates.extend(
            GPS_CLINICAL_DIR.glob(pattern)
        )

    candidates = [
        p
        for p in set(candidates)
        if p.is_file()
    ]

    if not candidates:
        return None

    # Prefer the largest extracted raw table.
    candidates.sort(
        key=lambda p: p.stat().st_size,
        reverse=True,
    )

    return candidates[0]


def load_raw_sources():
    sources = []

    weekly_raw = find_weekly_raw_path()

    if weekly_raw is not None:
        sources.append(weekly_raw)

    if DAILY_EMA_RAW.exists():
        sources.append(DAILY_EMA_RAW)

    sources = list(
        dict.fromkeys(sources)
    )

    if not sources:
        raise FileNotFoundError(
            "No extracted clinical raw GPS parquet "
            "table could be found."
        )

    print("\nRaw GPS sources:")

    frames = []

    for path in sources:

        print(f"  {path}")

        # Read schema only -- do NOT load the parquet just to inspect columns.
        schema_cols = pq.read_schema(path).names

        use_cols = [
            c
            for c in RAW_COLUMNS
            if c in schema_cols
        ]

        required = {
            "timestamp",
            "latitude",
            "longitude",
            "accuracy",
            "participant_id",
        }

        if not required.issubset(
            use_cols
        ):
            print(
                "    skipped: missing required columns"
            )
            continue

        if args.smoke:
            # Smoke test must stay genuinely small.
            # Read only the first parquet row group.
            parquet_file = pq.ParquetFile(path)

            if parquet_file.num_row_groups == 0:
                continue

            frame = (
                parquet_file
                .read_row_group(
                    0,
                    columns=use_cols,
                )
                .to_pandas()
            )

            print(
                f"    smoke read: "
                f"{len(frame):,} rows "
                f"from 1/{parquet_file.num_row_groups:,} row groups"
            )

        else:
            frame = pd.read_parquet(
                path,
                columns=use_cols,
            )

        frames.append(
            frame
        )

    if not frames:
        raise RuntimeError(
            "No usable raw GPS source."
        )

    raw = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Normalize date
    # --------------------------------------------------------

    raw["timestamp"] = pd.to_numeric(
        raw["timestamp"],
        errors="coerce",
    )

    raw = raw.loc[
        raw["timestamp"].notna()
    ].copy()

    raw["timestamp"] = (
        raw["timestamp"]
        .astype(np.int64)
    )

    if "date" not in raw.columns:

        raw["date"] = (
            pd.to_datetime(
                raw["timestamp"],
                unit="ms",
                utc=True,
            )
            .dt.strftime("%Y-%m-%d")
        )

    else:

        raw["date"] = (
            pd.to_datetime(
                raw["date"],
                errors="coerce",
            )
            .dt.strftime("%Y-%m-%d")
        )

    # --------------------------------------------------------
    # UTC day windows if not already supplied
    # --------------------------------------------------------

    dates = pd.to_datetime(
        raw["date"],
        errors="coerce",
        utc=True,
    )

    if "day_start_ms" not in raw.columns:

        raw["day_start_ms"] = (
            dates.astype("int64")
            // 1_000_000
        )

    if "day_end_ms" not in raw.columns:

        raw["day_end_ms"] = (
            raw["day_start_ms"]
            + 86_400_000
        )

    raw = (
        raw.drop_duplicates(
            subset=[
                "participant_id",
                "timestamp",
                "latitude",
                "longitude",
            ]
        )
        .sort_values(
            [
                "participant_id",
                "date",
                "timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return raw


def identify_assignments_and_places(
    first,
    second,
):
    """
    Guard against tuple-order ambiguity.
    """

    def assignment_score(df):
        if not isinstance(
            df,
            pd.DataFrame,
        ):
            return -1

        score = 0

        for c in [
            "place_candidate_id",
            "duration_ms",
        ]:
            if c in df.columns:
                score += 1

        return score

    if (
        assignment_score(first)
        >= assignment_score(second)
    ):
        assignments = first
        places = second

    else:
        assignments = second
        places = first

    return assignments, places


def compute_one_day(
    day_raw,
    *,
    start_ms,
    end_ms,
    max_gap_ms,
):
    """
    Reuse the validated Cortex-2.0 GPS feature path.
    """

    row = {}

    timestamps = (
        day_raw["timestamp"]
        if len(day_raw)
        else []
    )

    observability = (
        compute_gps_temporal_observability(
            timestamps,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            max_gap_ms=max_gap_ms,
        )
    )

    for key, value in observability.items():
        row[
            f"gps_observability_{key}"
        ] = value

    row["n_raw_gps_points"] = len(
        day_raw
    )

    row["has_raw_gps"] = (
        len(day_raw) > 0
    )

    if len(day_raw) == 0:

        row["gps_feature_status"] = (
            "no_raw_gps"
        )

        for feature in CORE_FEATURES:
            row[feature] = np.nan

        return row

    # --------------------------------------------------------
    # Intervals
    # --------------------------------------------------------

    intervals = build_gps_intervals(
        day_raw[
            [
                "timestamp",
                "latitude",
                "longitude",
                "accuracy",
            ]
        ],
        window_start_ms=start_ms,
        window_end_ms=end_ms,
        max_gap_ms=max_gap_ms,
    )

    row["n_gps_intervals"] = len(
        intervals
    )

    if len(intervals) == 0:

        row["gps_feature_status"] = (
            "no_intervals"
        )

        for feature in CORE_FEATURES:
            row[feature] = np.nan

        return row

    # --------------------------------------------------------
    # Spatial semantics
    # --------------------------------------------------------

    edges = add_gps_spatial_semantics(
        intervals
    )

    edges = (
        add_gps_trajectory_plausibility(
            edges,
            max_plausible_speed_kmh=(
                MAX_PLAUSIBLE_SPEED_KMH
            ),
        )
    )

    # --------------------------------------------------------
    # Mobility
    # --------------------------------------------------------

    mobility = (
        compute_daily_mobility_features(
            edges
        )
    )

    row.update(
        mobility
    )

    supported_duration_ms = (
        mobility.get(
            "supported_duration_ms"
        )
    )

    if supported_duration_ms is None:

        supported_duration_ms = (
            observability.get(
                "observed_duration_ms"
            )
        )

    # --------------------------------------------------------
    # Location episodes / places
    # --------------------------------------------------------

    episodes = (
        build_anchor_compatible_location_episodes(
            edges
        )
    )

    if len(episodes):

        a, b = build_place_candidates(
            episodes,
            compatibility_multiplier=(
                COMPATIBILITY_MULTIPLIER
            ),
        )

        assignments, places = (
            identify_assignments_and_places(
                a,
                b,
            )
        )

    else:

        assignments = pd.DataFrame(
            columns=[
                "place_candidate_id",
                "duration_ms",
            ]
        )

        places = pd.DataFrame()

    row["n_place_candidates_model"] = (
        len(places)
    )

    # --------------------------------------------------------
    # Location features
    # --------------------------------------------------------

    location = (
        compute_daily_location_features(
            assignments,
            supported_duration_ms=(
                supported_duration_ms
            ),
        )
    )

    row.update(
        location
    )

    row["gps_feature_status"] = "ok"

    return row


# ============================================================
# LOAD RAW GPS
# ============================================================
# ============================================================
# LOAD / PROCESS RAW GPS
# ============================================================

print()
print("=" * 90)
print("LOAD / PROCESS RAW GPS")
print("=" * 90)


weekly_raw = find_weekly_raw_path()

RAW_SOURCES = []

if weekly_raw is not None:
    RAW_SOURCES.append(weekly_raw)

if DAILY_EMA_RAW.exists():
    RAW_SOURCES.append(DAILY_EMA_RAW)

RAW_SOURCES = list(dict.fromkeys(RAW_SOURCES))

if not RAW_SOURCES:
    raise FileNotFoundError(
        "No extracted clinical raw GPS parquet tables found."
    )

print("\nRaw GPS sources:")

for p in RAW_SOURCES:
    print(f"  {p}")


def normalize_raw(frame):

    if frame.empty:
        return frame

    frame = frame.copy()

    frame["participant_id"] = (
        frame["participant_id"]
        .astype(str)
    )

    frame["timestamp"] = pd.to_numeric(
        frame["timestamp"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["timestamp"].notna()
    ].copy()

    frame["timestamp"] = (
        frame["timestamp"]
        .astype(np.int64)
    )

    if "date" not in frame.columns:

        frame["date"] = (
            pd.to_datetime(
                frame["timestamp"],
                unit="ms",
                utc=True,
            )
            .dt.strftime("%Y-%m-%d")
        )

    else:

        frame["date"] = (
            pd.to_datetime(
                frame["date"],
                errors="coerce",
            )
            .dt.strftime("%Y-%m-%d")
        )

    dates = pd.to_datetime(
        frame["date"],
        errors="coerce",
        utc=True,
    )

    if "day_start_ms" not in frame.columns:

        frame["day_start_ms"] = (
            dates.astype("int64")
            // 1_000_000
        )

    if "day_end_ms" not in frame.columns:

        frame["day_end_ms"] = (
            frame["day_start_ms"]
            + 86_400_000
        )

    frame = (
        frame.drop_duplicates(
            subset=[
                "participant_id",
                "timestamp",
                "latitude",
                "longitude",
            ]
        )
        .sort_values(
            [
                "participant_id",
                "date",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    return frame


def load_participant_raw(participant_id):

    frames = []

    for source in RAW_SOURCES:

        schema_cols = pq.read_schema(source).names

        use_cols = [
            c
            for c in RAW_COLUMNS
            if c in schema_cols
        ]

        table = pq.read_table(
            source,
            columns=use_cols,
            filters=[
                (
                    "participant_id",
                    "=",
                    str(participant_id),
                )
            ],
        )

        if table.num_rows:

            frames.append(
                table.to_pandas()
            )

    if not frames:
        return pd.DataFrame()

    return normalize_raw(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )


def append_day_features(
    feature_rows,
    participant_id,
    date,
    day_raw,
):

    start_ms = int(
        day_raw["day_start_ms"].iloc[0]
    )

    end_ms = int(
        day_raw["day_end_ms"].iloc[0]
    )

    for resolution_min in DEFAULT_RESOLUTIONS_MIN:

        if resolution_min == 0:

            sampled = (
                day_raw[
                    [
                        "timestamp",
                        "latitude",
                        "longitude",
                        "accuracy",
                    ]
                ]
                .copy()
            )

        else:

            sampled = (
                downsample_points_to_time_bins(
                    day_raw[
                        [
                            "timestamp",
                            "latitude",
                            "longitude",
                            "accuracy",
                        ]
                    ],
                    window_start_ms=start_ms,
                    window_end_ms=end_ms,
                    bin_size_ms=(
                        resolution_min
                        * 60_000
                    ),
                    seed=stable_seed(
                        participant_id,
                        date,
                        resolution_min,
                    ),
                )
            )

        if args.support_mode == "fixed":

            current_max_gap_ms = MAX_GAP_MS

        else:

            if resolution_min == 0:

                current_max_gap_ms = MAX_GAP_MS

            else:

                current_max_gap_ms = max(
                    MAX_GAP_MS,
                    2
                    * resolution_min
                    * 60_000,
                )

        result = compute_one_day(
            sampled,
            start_ms=start_ms,
            end_ms=end_ms,
            max_gap_ms=current_max_gap_ms,
        )

        result.update({
            "support_mode":
                args.support_mode,

            "max_gap_ms":
                current_max_gap_ms,
            "participant_id":
                participant_id,

            "date":
                date,

            "resolution_min":
                resolution_min,

            "resolution":
                resolution_label(
                    resolution_min
                ),

            "original_n_points":
                len(day_raw),

            "sampled_n_points":
                len(sampled),
        })

        feature_rows.append(result)


# ============================================================
# DETERMINE CLINICALLY RELEVANT PARTICIPANTS
# ============================================================

weekly_ids = set()

if WEEKLY_CLINICAL.exists():

    tmp = pd.read_parquet(
        WEEKLY_CLINICAL,
        columns=["participant_id"],
    )

    weekly_ids = set(
        tmp["participant_id"]
        .dropna()
        .astype(str)
        .unique()
    )


ema_ids = set()

if DAILY_EMA_CLINICAL.exists():

    tmp = pd.read_parquet(
        DAILY_EMA_CLINICAL,
        columns=["participant_id"],
    )

    ema_ids = set(
        tmp["participant_id"]
        .dropna()
        .astype(str)
        .unique()
    )


participants = sorted(
    weekly_ids | ema_ids
)

if args.max_participants is not None:

    participants = participants[
        :args.max_participants
    ]


# ============================================================
# SMOKE MODE
# ============================================================

feature_rows = []

CHECKPOINT_DIR = (
    OUT_DIR
    / f"participant_checkpoints_{args.support_mode.replace('-', '_')}"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if args.smoke:

    raw = load_raw_sources()

    raw["participant_id"] = (
        raw["participant_id"]
        .astype(str)
    )

    available = set(
        raw["participant_id"].unique()
    )

    priority = [
        p
        for p in participants
        if (
            p in available
            and p in ema_ids
            and p in weekly_ids
        )
    ]

    priority += [
        p
        for p in participants
        if (
            p in available
            and p not in priority
        )
    ]

    smoke_participants = priority[:3]

    raw = raw.loc[
        raw["participant_id"].isin(
            smoke_participants
        )
    ].copy()

    raw = normalize_raw(raw)

    print(
        f"\nSmoke participants: "
        f"{len(smoke_participants)}"
    )

    for participant_id in smoke_participants:

        participant_raw = raw.loc[
            raw["participant_id"].eq(
                participant_id
            )
        ].copy()

        groups = list(
            participant_raw.groupby(
                "date",
                sort=True,
            )
        )

        if not groups:
            continue

        n_take = min(
            3,
            len(groups),
        )

        indices = np.linspace(
            0,
            len(groups) - 1,
            n_take,
            dtype=int,
        )

        for i in sorted(set(indices)):

            date, day_raw = groups[i]

            print(
                f"{participant_id} | {date}"
            )

            append_day_features(
                feature_rows,
                participant_id,
                date,
                day_raw,
            )


# ============================================================
# FULL / LIMITED PARTICIPANT MODE
# ============================================================

else:

    print(
        f"\nClinical participants to process: "
        f"{len(participants):,}"
    )

    for participant_number, participant_id in enumerate(
        participants,
        start=1,
    ):

        print(
            f"[{participant_number:,}/"
            f"{len(participants):,}] "
            f"{participant_id}",
            flush=True,
        )

        checkpoint = (
            CHECKPOINT_DIR
            / f"{participant_id}.parquet"
        )

        if checkpoint.exists():

            existing = pd.read_parquet(
                checkpoint
            )

            feature_rows.extend(
                existing.to_dict(
                    orient="records"
                )
            )

            print(
                "    checkpoint exists - skipped"
            )

            continue

        participant_raw = (
            load_participant_raw(
                participant_id
            )
        )

        if participant_raw.empty:

            print(
                "    no extracted raw GPS"
            )
            continue

        groups = participant_raw.groupby(
            "date",
            sort=True,
        )

        n_days = (
            participant_raw["date"]
            .nunique()
        )

        print(
            f"    raw points: "
            f"{len(participant_raw):,} | "
            f"days: {n_days:,}"
        )

        participant_feature_rows = []

        for date, day_raw in groups:

            append_day_features(
                participant_feature_rows,
                participant_id,
                date,
                day_raw,
            )

        participant_features = pd.DataFrame(
            participant_feature_rows
        )

        participant_features.to_parquet(
            checkpoint,
            index=False,
        )

        feature_rows.extend(
            participant_feature_rows
        )

        print(
            f"    checkpoint saved: "
            f"{len(participant_features):,} rows"
        )

        del participant_raw
        del participant_features
        gc.collect()


features = pd.DataFrame(
    feature_rows
)

if features.empty:
    raise RuntimeError(
        "No multiresolution GPS features were generated."
    )

features = (
    features.sort_values(
        [
            "participant_id",
            "date",
            "resolution_min",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# FEATURE FIDELITY RELATIVE TO NATIVE
# ============================================================

print()
print("=" * 90)
print("FEATURE FIDELITY")
print("=" * 90)

native_cols = [
    "participant_id",
    "date",
]

native = (
    features.loc[
        features["resolution_min"].eq(0),
        native_cols + [
            c
            for c in CORE_FEATURES
            if c in features.columns
        ],
    ]
    .copy()
)

native = native.rename(
    columns={
        c: f"{c}__native"
        for c in CORE_FEATURES
        if c in native.columns
    }
)

features = features.merge(
    native,
    on=[
        "participant_id",
        "date",
    ],
    how="left",
)


for feature in CORE_FEATURES:

    if feature not in features.columns:
        continue

    native_feature = (
        f"{feature}__native"
    )

    if native_feature not in features.columns:
        continue

    features[
        f"{feature}__absolute_error"
    ] = (
        features[feature]
        - features[native_feature]
    ).abs()

    denominator = (
        features[native_feature]
        .abs()
        .replace(
            0,
            np.nan,
        )
    )

    features[
        f"{feature}__relative_error"
    ] = (
        features[
            f"{feature}__absolute_error"
        ]
        / denominator
    )


# ============================================================
# DAILY EMA ALIGNMENT
# ============================================================

print()
print("=" * 90)
print("DAILY EMA ALIGNMENT")
print("=" * 90)

ema_alignment = pd.DataFrame()

if DAILY_EMA_CLINICAL.exists():

    ema = pd.read_parquet(
        DAILY_EMA_CLINICAL,
        columns=[
            "participant_id",
            "date",
            "anxiety",
            "mood",
            "daily_function",
        ],
    )

    ema["date"] = (
        pd.to_datetime(
            ema["date"],
            errors="coerce",
        )
        .dt.strftime("%Y-%m-%d")
    )

    ema_alignment = features.merge(
        ema,
        on=[
            "participant_id",
            "date",
        ],
        how="inner",
    )

    print(
        f"Aligned rows: "
        f"{len(ema_alignment):,}"
    )

    print(
        f"Participants: "
        f"{ema_alignment['participant_id'].nunique():,}"
    )


# ============================================================
# WEEKLY / QUESTIONNAIRE 7-DAY ALIGNMENT
# ============================================================

print()
print("=" * 90)
print("WEEKLY QUESTIONNAIRE ALIGNMENT")
print("=" * 90)

weekly_alignment = pd.DataFrame()

if WEEKLY_CLINICAL.exists():

    weekly = pd.read_parquet(
        WEEKLY_CLINICAL,
        columns=[
            "participant_id",
            "assessment_date",
            "depression_score",
            "anxiety_score",
        ],
    )

    weekly[
        "assessment_date"
    ] = pd.to_datetime(
        weekly[
            "assessment_date"
        ],
        errors="coerce",
    )

    temp = features.copy()

    temp["date_dt"] = (
        pd.to_datetime(
            temp["date"],
            errors="coerce",
        )
    )

    weekly_rows = []

    for _, assessment in weekly.iterrows():

        participant_id = (
            assessment[
                "participant_id"
            ]
        )

        assessment_date = (
            assessment[
                "assessment_date"
            ]
        )

        if pd.isna(
            assessment_date
        ):
            continue

        start_date = (
            assessment_date
            - pd.Timedelta(
                days=6
            )
        )

        subset = temp.loc[
            temp[
                "participant_id"
            ].eq(
                participant_id
            )
            & temp[
                "date_dt"
            ].between(
                start_date,
                assessment_date,
                inclusive="both",
            )
        ]

        if subset.empty:
            continue

        for resolution_min, group in (
            subset.groupby(
                "resolution_min"
            )
        ):

            row = {
                "participant_id":
                    participant_id,

                "assessment_date":
                    assessment_date,

                "resolution_min":
                    resolution_min,

                "resolution":
                    resolution_label(
                        resolution_min
                    ),

                "depression_score":
                    assessment[
                        "depression_score"
                    ],

                "anxiety_score":
                    assessment[
                        "anxiety_score"
                    ],

                "n_feature_days":
                    group[
                        "date"
                    ].nunique(),
            }

            for feature in CORE_FEATURES:

                if feature not in group.columns:
                    continue

                values = (
                    pd.to_numeric(
                        group[feature],
                        errors="coerce",
                    )
                    .dropna()
                )

                if len(values):

                    row[
                        f"{feature}_mean"
                    ] = values.mean()

                    row[
                        f"{feature}_median"
                    ] = values.median()

                else:

                    row[
                        f"{feature}_mean"
                    ] = np.nan

                    row[
                        f"{feature}_median"
                    ] = np.nan

            weekly_rows.append(
                row
            )

    weekly_alignment = (
        pd.DataFrame(
            weekly_rows
        )
    )

    print(
        f"Aligned assessment-resolution rows: "
        f"{len(weekly_alignment):,}"
    )

    if len(weekly_alignment):

        print(
            f"Participants: "
            f"{weekly_alignment['participant_id'].nunique():,}"
        )


# ============================================================
# SAVE
# ============================================================

suffix = (
    "_smoke"
    if args.smoke
    else ""
)

support_suffix = (
    "_fixed"
    if args.support_mode == "fixed"
    else "_resolution_aware"
)

suffix = support_suffix + suffix

FEATURE_OUT = (
    OUT_DIR
    / f"gps_multires_daily_features{suffix}.parquet"
)

EMA_OUT = (
    OUT_DIR
    / f"gps_multires_daily_ema_alignment{suffix}.parquet"
)

WEEKLY_OUT = (
    OUT_DIR
    / f"gps_multires_weekly_alignment{suffix}.parquet"
)


features.to_parquet(
    FEATURE_OUT,
    index=False,
)

if len(ema_alignment):

    ema_alignment.to_parquet(
        EMA_OUT,
        index=False,
    )

if len(weekly_alignment):

    weekly_alignment.to_parquet(
        WEEKLY_OUT,
        index=False,
    )


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 90)
print("BENCHMARK COMPLETE")
print("=" * 90)

print(
    f"Daily feature rows: "
    f"{len(features):,}"
)

print(
    f"Participants: "
    f"{features['participant_id'].nunique():,}"
)

print()
print("Rows by resolution:")

print(
    features[
        "resolution"
    ]
    .value_counts(
        sort=False
    )
    .to_string()
)

print()
print("Feature status by resolution:")

print(
    pd.crosstab(
        features[
            "resolution"
        ],
        features[
            "gps_feature_status"
        ],
    ).to_string()
)

print()
print("Saved:")
print(FEATURE_OUT)

if len(ema_alignment):
    print(EMA_OUT)

if len(weekly_alignment):
    print(WEEKLY_OUT)
