#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import pyarrow.dataset as ds

from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_anchor_episodes import build_anchor_compatible_location_episodes
from cortex2_engine.features.gps_place_candidates import build_place_candidates


# ============================================================
# PATHS / PARAMETERS
# ============================================================

RAW_PATH = (
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "gps_clinical_weekly_raw_points.parquet"
)

ANCHOR_PATH = (
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "participant_home_anchors.csv"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_TIMEZONE = "America/New_York"

MAX_GAP_MS = 60_000
COMPATIBILITY_MULTIPLIER = 1.0

HOME_REFERENCE_DAYS = 14
NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 6
MIN_NIGHT_OBSERVED_MS = 2 * 60 * 60 * 1000
MIN_VALID_NIGHTS = 5
MIN_HOME_CONFIDENCE = 0.50
MIN_DAILY_SUPPORTED_MS = 6 * 60 * 60 * 1000

HOME_COLOR = "#18a84b"
PLACE_B_COLOR = "#de7f00"
PLACE_C_COLOR = "#159d9a"

AWAY_COLOR = "#5899e8"
UNKNOWN_COLOR = "#d9dde2"
INVALID_COLOR = "#9ca3af"
OTHER_COLOR = "#98a6ba"


# ============================================================
# PARTICIPANT
# ============================================================

def choose_participant():

    df = pd.read_csv(
        ANCHOR_PATH
    )

    df = df[
        df["home_status"].eq("observed")
        & df["home_confidence"].notna()
        & (
            df["home_n_valid_nights"]
            >= MIN_VALID_NIGHTS
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No observed home anchors."
        )

    df["score"] = (
        df["home_confidence"]
        - 0.75
    ).abs()

    row = df.sort_values(
        [
            "score",
            "home_n_valid_nights",
        ],
        ascending=[
            True,
            False,
        ],
    ).iloc[0]

    return str(
        row["participant_id"]
    )


# ============================================================
# LOAD RAW
# ============================================================

def load_raw(pid):

    dataset = ds.dataset(
        RAW_PATH,
        format="parquet",
    )

    table = dataset.to_table(
        filter=(
            ds.field(
                "participant_id"
            )
            == pid
        )
    )

    df = table.to_pandas()

    if df.empty:
        raise RuntimeError(
            f"No GPS for {pid}"
        )

    for col in [
        "timestamp",
        "latitude",
        "longitude",
        "accuracy",
        "day_start_ms",
        "day_end_ms",
    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
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
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SUPPORTED INTERVALS + MICROSEGMENTS
# ============================================================

def build_segments_and_supported(raw):

    segment_parts = []
    supported_by_day = {}

    for day_start, g in raw.groupby(
        "day_start_ms",
        sort=True,
    ):

        start_ms = int(
            day_start
        )

        end_ms = int(
            g[
                "day_end_ms"
            ].iloc[0]
        )

        intervals = (
            build_gps_intervals(
                g[
                    [
                        "timestamp",
                        "latitude",
                        "longitude",
                        "accuracy",
                    ]
                ].copy(),
                window_start_ms=(
                    start_ms
                ),
                window_end_ms=(
                    end_ms
                ),
                max_gap_ms=(
                    MAX_GAP_MS
                ),
            )
        )

        if intervals.empty:

            supported_by_day[
                start_ms
            ] = pd.DataFrame()

            continue

        supported = intervals[
            intervals[
                "supported"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

        supported_by_day[
            start_ms
        ] = supported.copy()

        if supported.empty:
            continue

        spatial = (
            add_gps_spatial_semantics(
                supported
            )
        )

        seg = (
            build_anchor_compatible_location_episodes(
                spatial
            )
        )

        if isinstance(
            seg,
            tuple,
        ):
            seg = seg[0]

        if (
            seg is None
            or seg.empty
        ):
            continue

        seg = seg.copy()

        seg[
            "utc_day_start_ms"
        ] = start_ms

        segment_parts.append(
            seg
        )

    if not segment_parts:
        raise RuntimeError(
            "No location microsegments."
        )

    return (
        pd.concat(
            segment_parts,
            ignore_index=True,
        ),
        supported_by_day,
    )


# ============================================================
# PLACE CLUSTERING
# ============================================================

def cluster_places(segments):

    assignments, places = (
        build_place_candidates(
            segments.copy(),
            compatibility_multiplier=(
                COMPATIBILITY_MULTIPLIER
            ),
        )
    )

    if assignments.empty:
        raise RuntimeError(
            "No place assignments."
        )

    for col in [
        "episode_start_ms",
        "episode_end_ms",
        "anchor_latitude",
        "anchor_longitude",
        "anchor_accuracy_m",
        "accuracy",
        "duration_ms",
        "utc_day_start_ms",
    ]:

        if (
            col not in assignments.columns
            and col in segments.columns
        ):

            assignments[col] = (
                segments[
                    col
                ].to_numpy()
            )

    return (
        assignments,
        places,
    )


# ============================================================
# TIME HELPERS
# ============================================================

def utc_ms(ts):

    return int(
        ts.tz_convert(
            "UTC"
        )
        .timestamp()
        * 1000
    )


# ============================================================
# HOME INFERENCE
# ============================================================

def infer_home(assignments):

    starts = pd.to_numeric(
        assignments[
            "episode_start_ms"
        ],
        errors="coerce",
    ).to_numpy(float)

    ends = pd.to_numeric(
        assignments[
            "episode_end_ms"
        ],
        errors="coerce",
    ).to_numpy(float)

    pids = pd.to_numeric(
        assignments[
            "place_candidate_id"
        ],
        errors="coerce",
    ).to_numpy(float)

    ok = (
        np.isfinite(starts)
        & np.isfinite(ends)
        & np.isfinite(pids)
        & (
            ends > starts
        )
    )

    starts = starts[
        ok
    ]

    ends = ends[
        ok
    ]

    pids = (
        pids[
            ok
        ].astype(int)
    )

    first_local_date = (
        pd.to_datetime(
            starts.min(),
            unit="ms",
            utc=True,
        )
        .tz_convert(
            LOCAL_TIMEZONE
        )
        .date()
    )

    reference_dates = [
        first_local_date
        + pd.Timedelta(
            days=i
        )
        for i
        in range(
            HOME_REFERENCE_DAYS
        )
    ]

    night_rows = []

    for local_date in (
        reference_dates
    ):

        night_start = (
            pd.Timestamp(
                f"{local_date} "
                f"{NIGHT_START_HOUR:02d}:00:00"
            )
            .tz_localize(
                LOCAL_TIMEZONE
            )
        )

        night_end = (
            pd.Timestamp(
                f"{local_date} "
                f"{NIGHT_END_HOUR:02d}:00:00"
            )
            .tz_localize(
                LOCAL_TIMEZONE
            )
        )

        ns = utc_ms(
            night_start
        )

        ne = utc_ms(
            night_end
        )

        overlap = np.maximum(
            0,
            np.minimum(
                ends,
                ne,
            )
            - np.maximum(
                starts,
                ns,
            ),
        )

        observed_ms = float(
            overlap.sum()
        )

        valid_night = (
            observed_ms
            >= MIN_NIGHT_OBSERVED_MS
        )

        winner = np.nan

        if valid_night:

            tmp = pd.DataFrame(
                {
                    "place_candidate_id":
                        pids,

                    "overlap_ms":
                        overlap,
                }
            )

            tmp = (
                tmp[
                    tmp[
                        "overlap_ms"
                    ] > 0
                ]
                .groupby(
                    "place_candidate_id",
                    as_index=False,
                )[
                    "overlap_ms"
                ]
                .sum()
                .sort_values(
                    [
                        "overlap_ms",
                        "place_candidate_id",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
            )

            if not tmp.empty:

                winner = int(
                    tmp.iloc[0][
                        "place_candidate_id"
                    ]
                )

        night_rows.append(
            {
                "local_date":
                    pd.Timestamp(
                        local_date
                    ),

                "valid_night":
                    bool(
                        valid_night
                    ),

                "observed_hours":
                    observed_ms
                    / 3_600_000,

                "winner":
                    winner,
            }
        )

    nights = pd.DataFrame(
        night_rows
    )

    valid = nights[
        nights[
            "valid_night"
        ]
        & nights[
            "winner"
        ].notna()
    ].copy()

    if (
        len(valid)
        < MIN_VALID_NIGHTS
    ):

        raise RuntimeError(
            "Too few valid nights."
        )

    valid[
        "winner"
    ] = (
        valid[
            "winner"
        ].astype(int)
    )

    counts = (
        valid.groupby(
            "winner"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    home_id = int(
        counts.index[0]
    )

    confidence = (
        counts.iloc[0]
        / len(valid)
    )

    if (
        confidence
        < MIN_HOME_CONFIDENCE
    ):

        raise RuntimeError(
            "Home confidence too low."
        )

    return (
        home_id,
        confidence,
        nights,
        valid,
    )


# ============================================================
# SPATIAL HELPERS
# ============================================================

def meters_xy(
    lat,
    lon,
    lat0,
    lon0,
):

    R = 6371000.0

    lat = np.asarray(
        lat,
        dtype=float,
    )

    lon = np.asarray(
        lon,
        dtype=float,
    )

    x = (
        np.deg2rad(
            lon - lon0
        )
        * R
        * np.cos(
            np.deg2rad(
                lat0
            )
        )
    )

    y = (
        np.deg2rad(
            lat - lat0
        )
        * R
    )

    return x, y


def hull(points):

    pts = np.unique(
        np.asarray(
            points
        ),
        axis=0,
    )

    if len(pts) < 3:
        return pts

    pts = pts[
        np.lexsort(
            (
                pts[:, 1],
                pts[:, 0],
            )
        )
    ]

    def cross(
        o,
        a,
        b,
    ):

        return (
            (
                a[0]
                - o[0]
            )
            * (
                b[1]
                - o[1]
            )
            - (
                a[1]
                - o[1]
            )
            * (
                b[0]
                - o[0]
            )
        )

    lower = []

    for p in pts:

        while (
            len(lower) >= 2
            and cross(
                lower[-2],
                lower[-1],
                p,
            ) <= 0
        ):
            lower.pop()

        lower.append(
            tuple(p)
        )

    upper = []

    for p in pts[::-1]:

        while (
            len(upper) >= 2
            and cross(
                upper[-2],
                upper[-1],
                p,
            ) <= 0
        ):
            upper.pop()

        upper.append(
            tuple(p)
        )

    return np.array(
        lower[:-1]
        + upper[:-1]
    )


def add_scale_bar(
    ax,
    length_m=100,
):

    xmin, xmax = (
        ax.get_xlim()
    )

    ymin, ymax = (
        ax.get_ylim()
    )

    x0 = (
        xmin
        + 0.08
        * (
            xmax - xmin
        )
    )

    y0 = (
        ymin
        + 0.10
        * (
            ymax - ymin
        )
    )

    ax.plot(
        [
            x0,
            x0 + length_m,
        ],
        [
            y0,
            y0,
        ],
        color="black",
        linewidth=2,
    )

    ax.text(
        x0
        + length_m / 2,
        y0
        + 0.04
        * (
            ymax - ymin
        ),
        f"{length_m} m",
        ha="center",
        va="bottom",
        fontsize=8,
    )


# ============================================================
# SELECT 3 DISPLAY PLACES
# ============================================================

def select_places(
    assignments,
    home_id,
    valid_nights,
):

    chosen = [
        home_id
    ]

    counts = (
        valid_nights.groupby(
            "winner"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    for pid in (
        counts.index
    ):

        pid = int(
            pid
        )

        if (
            pid
            not in chosen
        ):
            chosen.append(
                pid
            )

        if (
            len(chosen)
            == 3
        ):
            return chosen

    totals = (
        assignments.groupby(
            "place_candidate_id"
        )[
            "duration_ms"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for pid in (
        totals.index
    ):

        pid = int(
            pid
        )

        if (
            pid
            not in chosen
        ):
            chosen.append(
                pid
            )

        if (
            len(chosen)
            == 3
        ):
            break

    return chosen


# ============================================================
# REPRESENTATIVE DAY
# ============================================================

def get_supported_ms(
    supported
):

    if (
        "duration_ms"
        in supported.columns
    ):

        return float(
            pd.to_numeric(
                supported[
                    "duration_ms"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    candidates = [
        (
            "interval_start_ms",
            "interval_end_ms",
        ),
        (
            "start_ms",
            "end_ms",
        ),
        (
            "timestamp_start_ms",
            "timestamp_end_ms",
        ),
    ]

    for (
        start_col,
        end_col,
    ) in candidates:

        if (
            start_col
            in supported.columns
            and end_col
            in supported.columns
        ):

            return float(
                (
                    pd.to_numeric(
                        supported[
                            end_col
                        ],
                        errors="coerce",
                    )
                    - pd.to_numeric(
                        supported[
                            start_col
                        ],
                        errors="coerce",
                    )
                )
                .clip(
                    lower=0
                )
                .fillna(0)
                .sum()
            )

    raise RuntimeError(
        "Cannot identify supported "
        "interval duration."
    )


def choose_example_day(
    assignments,
    supported_by_day,
    home_id,
):

    rows = []

    for (
        day_start_ms,
        supported,
    ) in supported_by_day.items():

        if (
            supported is None
            or supported.empty
        ):
            continue

        supported_ms = (
            get_supported_ms(
                supported
            )
        )

        supported_h = (
            supported_ms
            / 3_600_000
        )

        if (
            supported_ms
            < MIN_DAILY_SUPPORTED_MS
        ):
            continue

        g = assignments[
            assignments[
                "utc_day_start_ms"
            ]
            .astype(int)
            .eq(
                int(
                    day_start_ms
                )
            )
        ].copy()

        home_ms = float(
            pd.to_numeric(
                g.loc[
                    g[
                        "place_candidate_id"
                    ]
                    .astype(int)
                    .eq(
                        home_id
                    ),
                    "duration_ms",
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        fraction = (
            home_ms
            / supported_ms
            if supported_ms > 0
            else np.nan
        )

        rows.append(
            {
                "day_start_ms":
                    int(
                        day_start_ms
                    ),

                "supported_h":
                    supported_h,

                "home_h":
                    home_ms
                    / 3_600_000,

                "home_fraction":
                    fraction,
            }
        )

    d = pd.DataFrame(
        rows
    )

    if d.empty:
        raise RuntimeError(
            "No valid example day."
        )

    preferred = d[
        d[
            "supported_h"
        ].between(
            18,
            22,
            inclusive="both",
        )
    ].copy()

    if not preferred.empty:

        target_fraction = (
            preferred[
                "home_fraction"
            ].median()
        )

        preferred[
            "score"
        ] = (
            (
                preferred[
                    "home_fraction"
                ]
                - target_fraction
            ).abs()
            +
            0.03
            * (
                preferred[
                    "supported_h"
                ]
                - 20
            ).abs()
        )

        return (
            preferred.sort_values(
                "score"
            )
            .iloc[0]
        )

    d[
        "support_distance"
    ] = (
        d[
            "supported_h"
        ]
        - 20
    ).abs()

    return (
        d.sort_values(
            "support_distance"
        )
        .iloc[0]
    )


def supported_interval_columns(
    df
):

    candidates = [
        (
            "interval_start_ms",
            "interval_end_ms",
        ),
        (
            "start_ms",
            "end_ms",
        ),
        (
            "timestamp_start_ms",
            "timestamp_end_ms",
        ),
    ]

    for a, b in candidates:

        if (
            a in df.columns
            and b in df.columns
        ):

            return (
                a,
                b,
            )

    raise RuntimeError(
        f"Supported interval columns "
        f"not found: {list(df.columns)}"
    )


# ============================================================
# RUN
# ============================================================

pid = (
    choose_participant()
)

raw = load_raw(
    pid
)

(
    segments,
    supported_by_day,
) = (
    build_segments_and_supported(
        raw
    )
)

(
    assignments,
    places,
) = (
    cluster_places(
        segments
    )
)

(
    home_id,
    confidence,
    nights,
    valid_nights,
) = infer_home(
    assignments
)

selected = select_places(
    assignments,
    home_id,
    valid_nights,
)

letters = {
    selected[0]:
        "A",
}

if (
    len(selected)
    > 1
):
    letters[
        selected[1]
    ] = "B"

if (
    len(selected)
    > 2
):
    letters[
        selected[2]
    ] = "C"

colors = {
    selected[0]:
        HOME_COLOR,
}

if (
    len(selected)
    > 1
):
    colors[
        selected[1]
    ] = PLACE_B_COLOR

if (
    len(selected)
    > 2
):
    colors[
        selected[2]
    ] = PLACE_C_COLOR


example = (
    choose_example_day(
        assignments,
        supported_by_day,
        home_id,
    )
)

day_start_ms = int(
    example[
        "day_start_ms"
    ]
)

day_end_ms = (
    day_start_ms
    + 24
    * 60
    * 60
    * 1000
)

utc_date = (
    pd.to_datetime(
        day_start_ms,
        unit="ms",
        utc=True,
    )
    .strftime(
        "%Y-%m-%d"
    )
)


# ============================================================
# INSPECT DISPLAY PLACE C OUTLIER
# ============================================================

if (
    len(selected)
    >= 3
):

    c_id = selected[2]

    c = assignments[
        assignments[
            "place_candidate_id"
        ]
        .astype(int)
        .eq(
            c_id
        )
    ].copy()

    lat0 = float(
        np.median(
            c[
                "anchor_latitude"
            ]
        )
    )

    lon0 = float(
        np.median(
            c[
                "anchor_longitude"
            ]
        )
    )

    cx, cy = meters_xy(
        c[
            "anchor_latitude"
        ],
        c[
            "anchor_longitude"
        ],
        lat0,
        lon0,
    )

    distance = np.sqrt(
        cx ** 2
        + cy ** 2
    )

    far_i = int(
        np.argmax(
            distance
        )
    )

    far_row = (
        c.iloc[
            far_i
        ]
    )

    print()
    print(
        "Candidate C farthest "
        "assigned microsegment:"
    )

    print(
        "  distance from cluster median:",
        round(
            float(
                distance[
                    far_i
                ]
            ),
            1,
        ),
        "m",
    )

    print(
        "  episode start:",
        pd.to_datetime(
            far_row[
                "episode_start_ms"
            ],
            unit="ms",
            utc=True,
        ),
    )

    for col in [
        "anchor_accuracy_m",
        "accuracy",
        "seed_support_radius_m",
    ]:

        if (
            col
            in c.columns
        ):

            print(
                f"  {col}:",
                far_row[
                    col
                ],
            )


# ============================================================
# FIGURE
# ============================================================

plt.rcParams.update(
    {
        "font.family":
            "Arial",

        "font.size":
            11,
    }
)

fig = plt.figure(
    figsize=(
        16,
        6.5,
    )
)

outer = fig.add_gridspec(
    1,
    3,
    width_ratios=[
        1.42,
        1.05,
        1.15,
    ],
    left=0.045,
    right=0.975,
    bottom=0.15,
    top=0.82,
    wspace=0.32,
)


# ============================================================
# PANEL 1
# ============================================================

panel_a = (
    outer[
        0,
        0,
    ]
    .subgridspec(
        3,
        2,
        width_ratios=[
            0.95,
            1.05,
        ],
        hspace=0.30,
        wspace=0.20,
    )
)

ax_overview = (
    fig.add_subplot(
        panel_a[
            :,
            0,
        ]
    )
)

zoom_axes = [
    fig.add_subplot(
        panel_a[
            i,
            1,
        ]
    )
    for i in range(
        len(
            selected
        )
    )
]

places2 = (
    places.copy()
)

places2[
    "place_candidate_id"
] = (
    places2[
        "place_candidate_id"
    ].astype(int)
)

home_row = (
    places2[
        places2[
            "place_candidate_id"
        ].eq(
            home_id
        )
    ]
    .iloc[0]
)

home_lat = float(
    home_row[
        "representative_latitude"
    ]
)

home_lon = float(
    home_row[
        "representative_longitude"
    ]
)

for candidate in (
    selected
):

    row = (
        places2[
            places2[
                "place_candidate_id"
            ].eq(
                candidate
            )
        ]
        .iloc[0]
    )

    x, y = meters_xy(
        [
            float(
                row[
                    "representative_latitude"
                ]
            )
        ],
        [
            float(
                row[
                    "representative_longitude"
                ]
            )
        ],
        home_lat,
        home_lon,
    )

    ax_overview.scatter(
        x[0] / 1000,
        y[0] / 1000,
        s=230,
        facecolor="white",
        edgecolor=(
            colors[
                candidate
            ]
        ),
        linewidth=2.5,
        zorder=3,
    )

    ax_overview.text(
        x[0] / 1000,
        y[0] / 1000,
        letters[
            candidate
        ],
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        zorder=4,
    )

ax_overview.set_xlabel(
    "East / west (km)"
)

ax_overview.set_ylabel(
    "North / south (km)"
)

ax_overview.set_title(
    "Geographic overview",
    fontsize=11,
)

ax_overview.grid(
    alpha=0.15
)

# TRUE spatial aspect
ax_overview.set_aspect(
    "equal",
    adjustable="box",
)


# real cluster zooms
for (
    ax,
    candidate,
) in zip(
    zoom_axes,
    selected,
):

    g = assignments[
        assignments[
            "place_candidate_id"
        ]
        .astype(int)
        .eq(
            candidate
        )
    ].copy()

    lat0 = float(
        np.median(
            g[
                "anchor_latitude"
            ]
        )
    )

    lon0 = float(
        np.median(
            g[
                "anchor_longitude"
            ]
        )
    )

    x, y = meters_xy(
        g[
            "anchor_latitude"
        ],
        g[
            "anchor_longitude"
        ],
        lat0,
        lon0,
    )

    ax.scatter(
        x,
        y,
        s=9,
        alpha=0.40,
        color=(
            colors[
                candidate
            ]
        ),
    )

    hh = hull(
        np.c_[
            x,
            y,
        ]
    )

    if (
        len(hh)
        >= 3
    ):

        ax.add_patch(
            Polygon(
                hh,
                closed=True,
                facecolor=(
                    colors[
                        candidate
                    ]
                ),
                edgecolor=(
                    colors[
                        candidate
                    ]
                ),
                alpha=0.10,
                linewidth=1.8,
                linestyle="--",
            )
        )

    ax.text(
        0.05,
        0.85,
        letters[
            candidate
        ],
        transform=(
            ax.transAxes
        ),
        fontsize=14,
        fontweight="bold",
        color=(
            colors[
                candidate
            ]
        ),
    )

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in (
        ax.spines.values()
    ):
        spine.set_color(
            "#bbbbbb"
        )

    # choose a sensible scale bar from the inset span
    ax.figure.canvas.draw()

    xmin, xmax = (
        ax.get_xlim()
    )

    span = (
        xmax - xmin
    )

    if span < 250:
        scale_m = 25
    elif span < 600:
        scale_m = 50
    elif span < 1500:
        scale_m = 100
    else:
        scale_m = 250

    add_scale_bar(
        ax,
        length_m=scale_m,
    )


n_places = int(
    places2[
        "place_candidate_id"
    ].nunique()
)

fig.text(
    0.17,
    0.86,
    (
        f"1. Candidate places "
        f"(3 of {n_places})"
    ),
    fontsize=17,
    fontweight="bold",
    ha="center",
)


# ============================================================
# PANEL 2 — ALL 14 NIGHTS
# ============================================================

ax2 = fig.add_subplot(
    outer[
        0,
        1,
    ]
)

display_places = list(
    selected
)

winner_places = [
    int(x)
    for x in valid_nights[
        "winner"
    ].unique()
]

for p in (
    winner_places
):
    if (
        p
        not in display_places
    ):
        display_places.append(
            p
        )

# rows:
# A/B/C/Other + insufficient
labels = {}

for p in (
    display_places
):

    if p in letters:
        labels[
            p
        ] = letters[
            p
        ]
    else:
        labels[
            p
        ] = "Other"

ordered_places = list(
    reversed(
        display_places
    )
)

y_lookup = {
    p: i + 1
    for i, p
    in enumerate(
        ordered_places
    )
}

invalid_y = 0


ax2.axhspan(
    y_lookup[
        home_id
    ] - 0.35,
    y_lookup[
        home_id
    ] + 0.35,
    color=HOME_COLOR,
    alpha=0.07,
)


for i, row in (
    nights.reset_index(
        drop=True
    )
    .iterrows()
):

    x = i + 1

    if (
        not bool(
            row[
                "valid_night"
            ]
        )
        or pd.isna(
            row[
                "winner"
            ]
        )
    ):

        ax2.scatter(
            x,
            invalid_y,
            s=75,
            facecolor="white",
            edgecolor=INVALID_COLOR,
            linewidth=1.7,
            zorder=3,
        )

    else:

        winner = int(
            row[
                "winner"
            ]
        )

        color = (
            colors[
                winner
            ]
            if winner
            in colors
            else OTHER_COLOR
        )

        ax2.scatter(
            x,
            y_lookup[
                winner
            ],
            s=90,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )


yticks = [
    invalid_y
] + [
    y_lookup[
        p
    ]
    for p in (
        ordered_places
    )
]

yticklabels = [
    "Insufficient"
] + [
    labels[
        p
    ]
    for p in (
        ordered_places
    )
]

ax2.set_yticks(
    yticks
)

ax2.set_yticklabels(
    yticklabels
)


ax2.set_xticks(
    np.arange(
        1,
        HOME_REFERENCE_DAYS
        + 1,
    )
)

ax2.set_xticklabels(
    [
        d.strftime(
            "%b %d"
        )
        for d in nights[
            "local_date"
        ]
    ],
    rotation=45,
    ha="right",
    fontsize=8,
)


ax2.set_xlim(
    0.5,
    HOME_REFERENCE_DAYS
    + 0.5,
)

ax2.set_xlabel(
    "Reference-period night"
)

ax2.set_ylabel(
    "Nighttime winner"
)

ax2.set_title(
    "2. Night-by-night home inference",
    fontsize=17,
    fontweight="bold",
)

ax2.grid(
    axis="x",
    alpha=0.10,
)

home_wins = int(
    (
        valid_nights[
            "winner"
        ].astype(int)
        == home_id
    ).sum()
)

ax2.text(
    0.5,
    -0.25,
    (
        f"A wins "
        f"{home_wins} of "
        f"{len(valid_nights)} "
        f"valid nights  →  HOME"
    ),
    transform=(
        ax2.transAxes
    ),
    ha="center",
    fontsize=12.5,
    fontweight="bold",
    color=HOME_COLOR,
)


# ============================================================
# PANEL 3 — UTC DAY
# ============================================================

ax3 = fig.add_subplot(
    outer[
        0,
        2,
    ]
)

supported = (
    supported_by_day[
        day_start_ms
    ].copy()
)

start_col, end_col = (
    supported_interval_columns(
        supported
    )
)


# whole day initially unobserved
ax3.add_patch(
    Rectangle(
        (
            0,
            0.38,
        ),
        24,
        0.28,
        facecolor=(
            UNKNOWN_COLOR
        ),
        edgecolor="none",
    )
)


# all supported intervals = blue denominator
for row in (
    supported.itertuples(
        index=False
    )
):

    s = max(
        float(
            getattr(
                row,
                start_col
            )
        ),
        day_start_ms,
    )

    e = min(
        float(
            getattr(
                row,
                end_col
            )
        ),
        day_end_ms,
    )

    if e <= s:
        continue

    x0 = (
        s
        - day_start_ms
    ) / 3_600_000

    width = (
        e - s
    ) / 3_600_000

    ax3.add_patch(
        Rectangle(
            (
                x0,
                0.38,
            ),
            width,
            0.28,
            facecolor=(
                AWAY_COLOR
            ),
            edgecolor="none",
        )
    )


# home overlays = green
g = assignments[
    assignments[
        "utc_day_start_ms"
    ]
    .astype(int)
    .eq(
        day_start_ms
    )
].copy()

home_g = g[
    g[
        "place_candidate_id"
    ]
    .astype(int)
    .eq(
        home_id
    )
]

home_ms = 0.0

for row in (
    home_g.itertuples(
        index=False
    )
):

    s = max(
        float(
            row.episode_start_ms
        ),
        day_start_ms,
    )

    e = min(
        float(
            row.episode_end_ms
        ),
        day_end_ms,
    )

    if e <= s:
        continue

    x0 = (
        s
        - day_start_ms
    ) / 3_600_000

    width = (
        e - s
    ) / 3_600_000

    home_ms += (
        e - s
    )

    ax3.add_patch(
        Rectangle(
            (
                x0,
                0.38,
            ),
            width,
            0.28,
            facecolor=(
                HOME_COLOR
            ),
            edgecolor="white",
            linewidth=0.3,
        )
    )


supported_ms = (
    get_supported_ms(
        supported
    )
)

supported_h = (
    supported_ms
    / 3_600_000
)

home_h = (
    home_ms
    / 3_600_000
)

home_fraction = (
    home_ms
    / supported_ms
)

ax3.set_xlim(
    0,
    24,
)

ax3.set_ylim(
    0,
    1,
)

ax3.set_xticks(
    [
        0,
        6,
        12,
        18,
        24,
    ]
)

ax3.set_xticklabels(
    [
        "0",
        "6",
        "12",
        "18",
        "24",
    ]
)

ax3.set_xlabel(
    "Hour of UTC analysis day"
)

ax3.set_yticks([])

for side in [
    "left",
    "right",
    "top",
]:

    ax3.spines[
        side
    ].set_visible(
        False
    )

ax3.set_title(
    (
        "3. Daily home time\n"
        f"{utc_date} UTC"
    ),
    fontsize=17,
    fontweight="bold",
)

ax3.text(
    12,
    0.19,
    f"{home_h:.1f} h home",
    ha="center",
    fontsize=16,
    fontweight="bold",
)

ax3.text(
    12,
    0.10,
    (
        f"{home_h:.1f} / "
        f"{supported_h:.1f} supported h"
        f"  =  "
        f"{100 * home_fraction:.1f}%"
    ),
    ha="center",
    fontsize=12,
)


legend_items = [
    (
        HOME_COLOR,
        "Home",
    ),
    (
        AWAY_COLOR,
        "Supported, not home",
    ),
    (
        UNKNOWN_COLOR,
        "Unobserved",
    ),
]

legend_x = [
    0.8,
    6.6,
    17.0,
]

for x0, (
    color,
    label,
) in zip(
    legend_x,
    legend_items,
):

    ax3.add_patch(
        Rectangle(
            (
                x0,
                0.82,
            ),
            0.9,
            0.055,
            color=color,
            clip_on=False,
        )
    )

    ax3.text(
        x0 + 1.1,
        0.848,
        label,
        va="center",
        fontsize=9.3,
    )


# ============================================================
# TITLE + METHOD CAPTION
# ============================================================

fig.suptitle(
    "Home time",
    fontsize=25,
    fontweight="bold",
    y=0.965,
)

fig.text(
    0.5,
    0.91,
    (
        f"Real participant {pid}  |  "
        f"home confidence = "
        f"{confidence:.2f}"
    ),
    ha="center",
    fontsize=11,
)

fig.text(
    0.5,
    0.035,
    (
        "Home = candidate place winning the most valid "
        "00:00–06:00 local nights during the first 14 days.  "
        "Dashed cluster outlines are visualization footprints only."
    ),
    ha="center",
    fontsize=10.5,
)


# ============================================================
# SAVE
# ============================================================

out_png = (
    OUT_DIR
    / f"home_time_mechanism_v3_{pid}.png"
)

out_pdf = (
    OUT_DIR
    / f"home_time_mechanism_v3_{pid}.pdf"
)

fig.savefig(
    out_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    out_pdf,
    bbox_inches="tight",
)

plt.close(
    fig
)


print()
print(
    "Participant:",
    pid,
)

print(
    "Persistent places:",
    n_places,
)

print(
    "Valid nights:",
    len(
        valid_nights
    ),
    "/",
    HOME_REFERENCE_DAYS,
)

print(
    "Home wins:",
    home_wins,
)

print(
    "Home confidence:",
    confidence,
)

print(
    "Example UTC day:",
    utc_date,
)

print(
    "Supported hours:",
    supported_h,
)

print(
    "Home hours:",
    home_h,
)

print()
print(
    "Saved:"
)

print(
    out_png
)

print(
    out_pdf
)
