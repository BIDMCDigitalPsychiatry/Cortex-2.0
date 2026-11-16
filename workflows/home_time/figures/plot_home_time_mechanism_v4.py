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


def choose_participant():
    df = pd.read_csv(ANCHOR_PATH)

    df = df[
        df["home_status"].eq("observed")
        & df["home_confidence"].notna()
        & (df["home_n_valid_nights"] >= MIN_VALID_NIGHTS)
    ].copy()

    df["score"] = (
        df["home_confidence"] - 0.75
    ).abs()

    return str(
        df.sort_values(
            ["score", "home_n_valid_nights"],
            ascending=[True, False],
        ).iloc[0]["participant_id"]
    )


def load_raw(pid):
    dataset = ds.dataset(
        RAW_PATH,
        format="parquet",
    )

    df = dataset.to_table(
        filter=(
            ds.field("participant_id")
            == pid
        )
    ).to_pandas()

    for c in [
        "timestamp",
        "latitude",
        "longitude",
        "accuracy",
        "day_start_ms",
        "day_end_ms",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
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
        .reset_index(drop=True)
    )


def build_segments_and_supported(raw):
    segment_parts = []
    supported_by_day = {}

    for day_start, g in raw.groupby(
        "day_start_ms",
        sort=True,
    ):
        start_ms = int(day_start)
        end_ms = int(
            g["day_end_ms"].iloc[0]
        )

        intervals = build_gps_intervals(
            g[
                [
                    "timestamp",
                    "latitude",
                    "longitude",
                    "accuracy",
                ]
            ].copy(),
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            max_gap_ms=MAX_GAP_MS,
        )

        if intervals.empty:
            supported_by_day[start_ms] = pd.DataFrame()
            continue

        supported = intervals[
            intervals["supported"]
            .fillna(False)
            .astype(bool)
        ].copy()

        supported_by_day[start_ms] = supported.copy()

        if supported.empty:
            continue

        spatial = add_gps_spatial_semantics(
            supported
        )

        seg = build_anchor_compatible_location_episodes(
            spatial
        )

        if isinstance(seg, tuple):
            seg = seg[0]

        if seg is None or seg.empty:
            continue

        seg = seg.copy()
        seg["utc_day_start_ms"] = start_ms

        segment_parts.append(seg)

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


def cluster_places(segments):
    assignments, places = build_place_candidates(
        segments.copy(),
        compatibility_multiplier=(
            COMPATIBILITY_MULTIPLIER
        ),
    )

    for c in [
        "episode_start_ms",
        "episode_end_ms",
        "anchor_latitude",
        "anchor_longitude",
        "duration_ms",
        "utc_day_start_ms",
    ]:
        if (
            c not in assignments.columns
            and c in segments.columns
        ):
            assignments[c] = (
                segments[c].to_numpy()
            )

    return assignments, places


def utc_ms(ts):
    return int(
        ts.tz_convert("UTC")
        .timestamp()
        * 1000
    )


def infer_home(assignments):
    starts = pd.to_numeric(
        assignments["episode_start_ms"],
        errors="coerce",
    ).to_numpy(float)

    ends = pd.to_numeric(
        assignments["episode_end_ms"],
        errors="coerce",
    ).to_numpy(float)

    pids = pd.to_numeric(
        assignments["place_candidate_id"],
        errors="coerce",
    ).to_numpy(float)

    ok = (
        np.isfinite(starts)
        & np.isfinite(ends)
        & np.isfinite(pids)
        & (ends > starts)
    )

    starts = starts[ok]
    ends = ends[ok]
    pids = pids[ok].astype(int)

    first_local_date = (
        pd.to_datetime(
            starts.min(),
            unit="ms",
            utc=True,
        )
        .tz_convert(LOCAL_TIMEZONE)
        .date()
    )

    reference_dates = [
        first_local_date
        + pd.Timedelta(days=i)
        for i in range(HOME_REFERENCE_DAYS)
    ]

    rows = []

    for local_date in reference_dates:
        night_start = pd.Timestamp(
            f"{local_date} 00:00:00"
        ).tz_localize(
            LOCAL_TIMEZONE
        )

        night_end = pd.Timestamp(
            f"{local_date} 06:00:00"
        ).tz_localize(
            LOCAL_TIMEZONE
        )

        ns = utc_ms(night_start)
        ne = utc_ms(night_end)

        overlap = np.maximum(
            0,
            np.minimum(ends, ne)
            - np.maximum(starts, ns),
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
                    tmp["overlap_ms"] > 0
                ]
                .groupby(
                    "place_candidate_id",
                    as_index=False,
                )["overlap_ms"]
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

        rows.append(
            {
                "local_date":
                    pd.Timestamp(local_date),
                "valid_night":
                    valid_night,
                "winner":
                    winner,
            }
        )

    nights = pd.DataFrame(rows)

    valid = nights[
        nights["valid_night"]
        & nights["winner"].notna()
    ].copy()

    valid["winner"] = (
        valid["winner"].astype(int)
    )

    counts = (
        valid.groupby("winner")
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

    return (
        home_id,
        confidence,
        nights,
        valid,
    )


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
        np.deg2rad(lon - lon0)
        * R
        * np.cos(
            np.deg2rad(lat0)
        )
    )

    y = (
        np.deg2rad(lat - lat0)
        * R
    )

    return x, y


def convex_hull(points):
    pts = np.unique(
        np.asarray(points),
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

    def cross(o, a, b):
        return (
            (a[0] - o[0])
            * (b[1] - o[1])
            - (a[1] - o[1])
            * (b[0] - o[0])
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

        lower.append(tuple(p))

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

        upper.append(tuple(p))

    return np.array(
        lower[:-1]
        + upper[:-1]
    )


def select_places(
    assignments,
    home_id,
    valid_nights,
):
    selected = [home_id]

    counts = (
        valid_nights.groupby(
            "winner"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    for p in counts.index:
        p = int(p)

        if p not in selected:
            selected.append(p)

        if len(selected) == 3:
            return selected

    totals = (
        assignments.groupby(
            "place_candidate_id"
        )["duration_ms"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for p in totals.index:
        p = int(p)

        if p not in selected:
            selected.append(p)

        if len(selected) == 3:
            break

    return selected


def supported_duration_ms(df):
    if "duration_ms" in df.columns:
        return float(
            pd.to_numeric(
                df["duration_ms"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    for a, b in [
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
    ]:
        if (
            a in df.columns
            and b in df.columns
        ):
            return float(
                (
                    pd.to_numeric(
                        df[b],
                        errors="coerce",
                    )
                    - pd.to_numeric(
                        df[a],
                        errors="coerce",
                    )
                )
                .clip(lower=0)
                .fillna(0)
                .sum()
            )

    raise RuntimeError(
        "Supported interval bounds missing."
    )


def supported_bounds(df):
    for a, b in [
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
    ]:
        if (
            a in df.columns
            and b in df.columns
        ):
            return a, b

    raise RuntimeError(
        "Supported interval bounds missing."
    )


def choose_example_day(
    assignments,
    supported_by_day,
    home_id,
):
    rows = []

    for day_start_ms, supported in (
        supported_by_day.items()
    ):
        if (
            supported is None
            or supported.empty
        ):
            continue

        support_ms = (
            supported_duration_ms(
                supported
            )
        )

        support_h = (
            support_ms
            / 3_600_000
        )

        if (
            support_ms
            < MIN_DAILY_SUPPORTED_MS
        ):
            continue

        g = assignments[
            assignments[
                "utc_day_start_ms"
            ].astype(int)
            .eq(
                int(day_start_ms)
            )
        ]

        home_ms = float(
            pd.to_numeric(
                g.loc[
                    g[
                        "place_candidate_id"
                    ].astype(int)
                    .eq(home_id),
                    "duration_ms",
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        rows.append(
            {
                "day_start_ms":
                    int(day_start_ms),
                "supported_h":
                    support_h,
                "home_h":
                    home_ms
                    / 3_600_000,
                "home_fraction":
                    home_ms
                    / support_ms,
            }
        )

    d = pd.DataFrame(rows)

    preferred = d[
        d["supported_h"].between(
            18,
            22,
        )
    ].copy()

    if not preferred.empty:
        target = (
            preferred[
                "home_fraction"
            ].median()
        )

        preferred["score"] = (
            (
                preferred[
                    "home_fraction"
                ]
                - target
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
            ).iloc[0]
        )

    d["score"] = (
        d["supported_h"]
        - 20
    ).abs()

    return (
        d.sort_values(
            "score"
        ).iloc[0]
    )


# ============================================================
# BUILD REAL DATA
# ============================================================

pid = choose_participant()

raw = load_raw(pid)

segments, supported_by_day = (
    build_segments_and_supported(
        raw
    )
)

assignments, places = (
    cluster_places(
        segments
    )
)

home_id, confidence, nights, valid_nights = (
    infer_home(
        assignments
    )
)

selected = select_places(
    assignments,
    home_id,
    valid_nights,
)

letters = {
    selected[0]: "A",
    selected[1]: "B",
    selected[2]: "C",
}

colors = {
    selected[0]:
        HOME_COLOR,
    selected[1]:
        PLACE_B_COLOR,
    selected[2]:
        PLACE_C_COLOR,
}

example = choose_example_day(
    assignments,
    supported_by_day,
    home_id,
)

day_start_ms = int(
    example["day_start_ms"]
)

day_end_ms = (
    day_start_ms
    + 24 * 60 * 60 * 1000
)

utc_date = (
    pd.to_datetime(
        day_start_ms,
        unit="ms",
        utc=True,
    )
    .strftime("%Y-%m-%d")
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
        6.4,
    )
)

outer = fig.add_gridspec(
    1,
    3,
    width_ratios=[
        1.45,
        1.0,
        1.12,
    ],
    left=0.04,
    right=0.98,
    bottom=0.18,
    top=0.82,
    wspace=0.30,
)


# ============================================================
# PANEL 1
# ============================================================

panel1 = (
    outer[
        0,
        0,
    ]
    .subgridspec(
        3,
        2,
        width_ratios=[
            0.9,
            1.1,
        ],
        hspace=0.28,
        wspace=0.17,
    )
)

ax_map = fig.add_subplot(
    panel1[
        :,
        0,
    ]
)

zoom_axes = [
    fig.add_subplot(
        panel1[
            i,
            1,
        ]
    )
    for i
    in range(3)
]


places2 = places.copy()

places2[
    "place_candidate_id"
] = (
    places2[
        "place_candidate_id"
    ].astype(int)
)

home_row = places2[
    places2[
        "place_candidate_id"
    ].eq(
        home_id
    )
].iloc[0]

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


# Actual relative coordinates
overview = {}

for candidate in selected:
    row = places2[
        places2[
            "place_candidate_id"
        ].eq(candidate)
    ].iloc[0]

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

    overview[candidate] = (
        x[0] / 1000,
        y[0] / 1000,
    )


hx, hy = (
    overview[
        home_id
    ]
)


# connecting lines + actual distances
for candidate in selected:

    if candidate == home_id:
        continue

    x, y = (
        overview[
            candidate
        ]
    )

    ax_map.plot(
        [
            hx,
            x,
        ],
        [
            hy,
            y,
        ],
        linestyle="--",
        color="#9ca3af",
        linewidth=1.25,
        zorder=1,
    )



# place markers
for candidate in selected:

    x, y = (
        overview[
            candidate
        ]
    )

    ax_map.scatter(
        x,
        y,
        s=280,
        facecolor="white",
        edgecolor=(
            colors[
                candidate
            ]
        ),
        linewidth=2.7,
        zorder=3,
    )

    ax_map.text(
        x,
        y,
        letters[
            candidate
        ],
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        zorder=4,
    )


ax_map.set_aspect(
    "equal",
    adjustable="datalim",
)

ax_map.set_xticks([])
ax_map.set_yticks([])

for spine in (
    ax_map.spines.values()
):
    spine.set_visible(
        False
    )


# north indicator
xmin, xmax = (
    ax_map.get_xlim()
)

ymin, ymax = (
    ax_map.get_ylim()
)

nx = (
    xmin
    + 0.10
    * (
        xmax - xmin
    )
)

ny = (
    ymax
    - 0.12
    * (
        ymax - ymin
    )
)

ax_map.annotate(
    "",
    xy=(
        nx,
        ny,
    ),
    xytext=(
        nx,
        ny
        - 0.08
        * (
            ymax - ymin
        ),
    ),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#444444",
        linewidth=1.2,
    ),
)

ax_map.text(
    nx,
    ny
    + 0.02
    * (
        ymax - ymin
    ),
    "N",
    ha="center",
    fontsize=9,
    fontweight="bold",
)

ax_map.set_title(
    "Relative locations",
    fontsize=10.5,
)


# real cluster zooms
for ax, candidate in zip(
    zoom_axes,
    selected,
):

    g = assignments[
        assignments[
            "place_candidate_id"
        ].astype(int)
        .eq(candidate)
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
        s=10,
        alpha=0.42,
        color=(
            colors[
                candidate
            ]
        ),
    )

    hh = convex_hull(
        np.c_[
            x,
            y,
        ]
    )

    if len(hh) >= 3:
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
                linewidth=1.7,
                linestyle="--",
                alpha=0.10,
            )
        )

    ax.text(
        0.05,
        0.84,
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

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in (
        ax.spines.values()
    ):
        spine.set_color(
            "#c5c5c5"
        )


fig.text(
    0.165,
    0.86,
    "1. Candidate places",
    ha="center",
    fontsize=17,
    fontweight="bold",
)


# ============================================================
# PANEL 2
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

for p in (
    valid_nights[
        "winner"
    ]
    .astype(int)
    .unique()
):

    if p not in display_places:
        display_places.append(
            p
        )


ordered = list(
    reversed(
        display_places
    )
)

y_lookup = {
    p: i + 1
    for i, p
    in enumerate(
        ordered
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
        not row[
            "valid_night"
        ]
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
        )

    else:

        winner = int(
            row[
                "winner"
            ]
        )

        c = (
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
            color=c,
            edgecolor="white",
            linewidth=0.8,
        )


ax2.set_yticks(
    [
        invalid_y
    ]
    + [
        y_lookup[p]
        for p in ordered
    ]
)

ax2.set_yticklabels(
    [
        "Insufficient"
    ]
    + [
        letters[p]
        if p in letters
        else "Other"
        for p in ordered
    ]
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
    14.5,
)

ax2.set_xlabel(
    ""
)

# NO vertical y-axis title
ax2.set_ylabel(
    ""
)

ax2.set_title(
    "2. Night-by-night home inference",
    fontsize=17,
    fontweight="bold",
    pad=22,
)

ax2.text(
    0.5,
    1.015,
    "Place with most observed time, 12 AM–6 AM",
    transform=(
        ax2.transAxes
    ),
    ha="center",
    va="bottom",
    fontsize=10,
    color="#4b5563",
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
    -0.17,
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
# PANEL 3
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
    supported_bounds(
        supported
    )
)


# background = unobserved
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


# supported denominator
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


# overlay home time
g = assignments[
    assignments[
        "utc_day_start_ms"
    ].astype(int)
    .eq(
        day_start_ms
    )
]

home_g = g[
    g[
        "place_candidate_id"
    ].astype(int)
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


support_ms = (
    supported_duration_ms(
        supported
    )
)

support_h = (
    support_ms
    / 3_600_000
)

home_h = (
    home_ms
    / 3_600_000
)

fraction = (
    home_ms
    / support_ms
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
        f"{support_h:.1f} supported h"
        f"  =  "
        f"{100 * fraction:.1f}%"
    ),
    ha="center",
    fontsize=12,
)


# compact legend
legend = [
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
    0.9,
    6.5,
    17.0,
]

for x0, (
    c,
    label,
) in zip(
    legend_x,
    legend,
):

    ax3.add_patch(
        Rectangle(
            (
                x0,
                0.82,
            ),
            0.9,
            0.055,
            color=c,
        )
    )

    ax3.text(
        x0 + 1.1,
        0.848,
        label,
        va="center",
        fontsize=9.2,
    )


# ============================================================
# TITLES / CAPTION
# ============================================================

fig.suptitle(
    "Home time",
    fontsize=25,
    fontweight="bold",
    y=0.97,
)

fig.text(
    0.5,
    0.915,
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
    0.012,
    (
        "Home = candidate place winning the most valid "
        "00:00–06:00 local nights during the first 14 days."
    ),
    ha="center",
    fontsize=10.5,
)


# ============================================================
# SAVE NEW FILE
# ============================================================

out_png = (
    OUT_DIR
    / f"home_time_mechanism_v5_{pid}.png"
)

out_pdf = (
    OUT_DIR
    / f"home_time_mechanism_v5_{pid}.pdf"
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

plt.close(fig)

print()
print("Participant:", pid)
print("Saved:")
print(out_png)
print(out_pdf)
