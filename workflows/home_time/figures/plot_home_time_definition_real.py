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
    "feature_reliability/gps_clinical/gps_clinical_weekly_raw_points.parquet"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARTICIPANT_ID = "RSH634"
LOCAL_TIMEZONE = "America/New_York"

REFERENCE_START_LOCAL_DATE = "2025-04-23"
HOME_REFERENCE_DAYS = 14
EXAMPLE_LOCAL_DATE = "2025-05-02"

MAX_GAP_MS = 60_000
COMPATIBILITY_MULTIPLIER = 1.0
NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 6
MIN_NIGHT_OBSERVED_MS = 2 * 60 * 60 * 1000
MIN_VALID_NIGHTS = 5
MIN_HOME_CONFIDENCE = 0.50

HOME_COLOR = "#1ca64b"
AWAY_COLOR = "#5f9ae6"
UNKNOWN_COLOR = "#cfd3d7"
OTHER_COLORS = ["#d67f11", "#18a999", "#7a52cc", "#cc4f7a"]


def load_participant_raw(raw_path, participant_id):
    dataset = ds.dataset(raw_path, format="parquet")
    table = dataset.to_table(
        filter=(ds.field("participant_id") == participant_id),
    )
    df = table.to_pandas()

    if df.empty:
        raise RuntimeError(
            f"No raw GPS rows found for {participant_id}"
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

    df = (
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

    return df


def maybe_df(x):
    if isinstance(x, tuple):
        return x[0]
    return x


def utc_ms(ts):
    return int(
        ts.tz_convert("UTC").timestamp()
        * 1000
    )


def local_midnight(date_str):
    return pd.Timestamp(
        f"{date_str} 00:00:00"
    ).tz_localize(
        LOCAL_TIMEZONE
    )


def meters_xy(lat, lon, lat0, lon0):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    r = 6371000.0

    x = (
        np.deg2rad(lon - lon0)
        * r
        * np.cos(
            np.deg2rad(
                (lat + lat0) / 2.0
            )
        )
    )

    y = (
        np.deg2rad(lat - lat0)
        * r
    )

    return x, y


def convex_hull(points):
    pts = np.unique(
        np.asarray(points),
        axis=0,
    )

    if len(pts) <= 2:
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


def build_microsegments(raw):
    all_segments = []

    for day_start_ms, day_raw in raw.groupby(
        "day_start_ms",
        sort=True,
    ):
        day_start_ms = int(
            day_start_ms
        )

        day_end_ms = int(
            day_raw[
                "day_end_ms"
            ].iloc[0]
        )

        intervals = build_gps_intervals(
            day_raw[
                [
                    "timestamp",
                    "latitude",
                    "longitude",
                    "accuracy",
                ]
            ].copy(),
            window_start_ms=day_start_ms,
            window_end_ms=day_end_ms,
            max_gap_ms=MAX_GAP_MS,
        )

        if (
            intervals is None
            or len(intervals) == 0
        ):
            continue

        supported = intervals[
            intervals[
                "supported"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

        if supported.empty:
            continue

        spatial = (
            add_gps_spatial_semantics(
                supported
            )
        )

        segments = (
            build_anchor_compatible_location_episodes(
                spatial
            )
        )

        segments = maybe_df(
            segments
        )

        if (
            segments is None
            or len(segments) == 0
        ):
            continue

        segments = segments.copy()

        if (
            "anchor_latitude"
            not in segments.columns
        ):
            raise RuntimeError(
                "anchor_latitude missing "
                "from microsegments"
            )

        if (
            "anchor_longitude"
            not in segments.columns
        ):
            raise RuntimeError(
                "anchor_longitude missing "
                "from microsegments"
            )

        segments[
            "utc_day_start_ms"
        ] = day_start_ms

        all_segments.append(
            segments
        )

    if not all_segments:
        raise RuntimeError(
            "No supported microsegments."
        )

    return pd.concat(
        all_segments,
        ignore_index=True,
    )


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
        "duration_ms",
    ]:
        if (
            col not in assignments.columns
            and col in segments.columns
        ):
            assignments[col] = (
                segments[col].to_numpy()
            )

    return assignments, places


def infer_home(assignments):
    starts = pd.to_numeric(
        assignments[
            "episode_start_ms"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    ends = pd.to_numeric(
        assignments[
            "episode_end_ms"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    place_ids = pd.to_numeric(
        assignments[
            "place_candidate_id"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(starts)
        & np.isfinite(ends)
        & np.isfinite(place_ids)
        & (ends > starts)
    )

    starts = starts[valid]
    ends = ends[valid]
    place_ids = (
        place_ids[valid]
        .astype(int)
    )

    ref_dates = [
        (
            local_midnight(
                REFERENCE_START_LOCAL_DATE
            )
            + pd.Timedelta(days=i)
        ).date()
        for i in range(
            HOME_REFERENCE_DAYS
        )
    ]

    valid_nights = []
    night_duration_rows = []

    for local_date in ref_dates:
        night_start = pd.Timestamp(
            f"{local_date} "
            f"{NIGHT_START_HOUR:02d}:00:00"
        ).tz_localize(
            LOCAL_TIMEZONE
        )

        night_end = pd.Timestamp(
            f"{local_date} "
            f"{NIGHT_END_HOUR:02d}:00:00"
        ).tz_localize(
            LOCAL_TIMEZONE
        )

        ns = utc_ms(
            night_start
        )

        ne = utc_ms(
            night_end
        )

        overlap = np.maximum(
            0.0,
            np.minimum(
                ends,
                ne,
            )
            - np.maximum(
                starts,
                ns,
            ),
        )

        observed = float(
            overlap.sum()
        )

        if (
            observed
            < MIN_NIGHT_OBSERVED_MS
        ):
            continue

        tmp = pd.DataFrame(
            {
                "place_candidate_id":
                    place_ids,

                "night_overlap_ms":
                    overlap,
            }
        )

        tmp = (
            tmp[
                tmp[
                    "night_overlap_ms"
                ] > 0
            ]
            .groupby(
                "place_candidate_id",
                as_index=False,
            )[
                "night_overlap_ms"
            ]
            .sum()
            .sort_values(
                [
                    "night_overlap_ms",
                    "place_candidate_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )

        if tmp.empty:
            continue

        winner = int(
            tmp.iloc[0][
                "place_candidate_id"
            ]
        )

        valid_nights.append(
            winner
        )

        night_duration_rows.append(
            tmp
        )

    if (
        len(valid_nights)
        < MIN_VALID_NIGHTS
    ):
        raise RuntimeError(
            "Too few valid nights."
        )

    counts = (
        pd.Series(
            valid_nights
        )
        .value_counts()
    )

    home_place_id = int(
        counts.index[0]
    )

    home_confidence = (
        counts.iloc[0]
        / len(valid_nights)
    )

    if (
        home_confidence
        < MIN_HOME_CONFIDENCE
    ):
        raise RuntimeError(
            "Low home confidence."
        )

    all_night = pd.concat(
        night_duration_rows,
        ignore_index=True,
    )

    night_by_place = (
        all_night.groupby(
            "place_candidate_id",
            as_index=False,
        )[
            "night_overlap_ms"
        ]
        .sum()
    )

    night_by_place[
        "hours"
    ] = (
        night_by_place[
            "night_overlap_ms"
        ]
        / 3_600_000
    )

    return (
        home_place_id,
        home_confidence,
        night_by_place,
    )


def choose_top_places(
    assignments,
    home_place_id,
):
    totals = (
        assignments.groupby(
            "place_candidate_id",
            as_index=False,
        )[
            "duration_ms"
        ]
        .sum()
        .sort_values(
            "duration_ms",
            ascending=False,
        )
    )

    chosen = [
        home_place_id
    ]

    for pid in totals[
        "place_candidate_id"
    ]:
        pid = int(pid)

        if pid not in chosen:
            chosen.append(
                pid
            )

        if len(chosen) == 3:
            break

    return chosen


def build_day_timeline(
    assignments,
    home_place_id,
    local_date,
):
    day_start = local_midnight(
        local_date
    )

    day_end = pd.Timestamp(
        f"{local_date} 23:59:59.999999"
    ).tz_localize(
        LOCAL_TIMEZONE
    )

    start_ms = utc_ms(
        day_start
    )

    end_ms = utc_ms(
        day_end
    )

    pieces = []

    for row in assignments.itertuples():
        s = float(
            row.episode_start_ms
        )

        e = float(
            row.episode_end_ms
        )

        ss = max(
            s,
            start_ms,
        )

        ee = min(
            e,
            end_ms,
        )

        if ee <= ss:
            continue

        kind = (
            "home"
            if int(
                row.place_candidate_id
            ) == home_place_id
            else "away"
        )

        pieces.append(
            (
                ss,
                ee,
                kind,
            )
        )

    pieces.sort(
        key=lambda z: z[0]
    )

    timeline = []

    cursor = start_ms

    for s, e, kind in pieces:
        if s > cursor:
            timeline.append(
                (
                    cursor,
                    s,
                    "unknown",
                )
            )

        timeline.append(
            (
                s,
                e,
                kind,
            )
        )

        cursor = max(
            cursor,
            e,
        )

    if cursor < end_ms:
        timeline.append(
            (
                cursor,
                end_ms,
                "unknown",
            )
        )

    return (
        timeline,
        start_ms,
    )


def main():
    print(
        "Loading participant:",
        PARTICIPANT_ID,
    )

    raw = load_participant_raw(
        RAW_PATH,
        PARTICIPANT_ID,
    )

    print(
        "Raw points:",
        len(raw),
    )

    segments = (
        build_microsegments(
            raw
        )
    )

    print(
        "Microsegments:",
        len(segments),
    )

    assignments, places = (
        cluster_places(
            segments
        )
    )

    print(
        "Persistent places:",
        len(places),
    )

    (
        home_place_id,
        home_confidence,
        night_by_place,
    ) = infer_home(
        assignments
    )

    print(
        "Home place:",
        home_place_id,
    )

    print(
        "Home confidence:",
        round(
            home_confidence,
            3,
        ),
    )

    top_places = (
        choose_top_places(
            assignments,
            home_place_id,
        )
    )

    home_row = places[
        places[
            "place_candidate_id"
        ].astype(int)
        .eq(
            home_place_id
        )
    ].iloc[0]

    lat0 = float(
        home_row[
            "representative_latitude"
        ]
    )

    lon0 = float(
        home_row[
            "representative_longitude"
        ]
    )

    labels = {}

    other_letters = [
        "B",
        "C",
        "D",
    ]

    j = 0

    for pid in top_places:
        if pid == home_place_id:
            labels[pid] = "A"
        else:
            labels[pid] = (
                other_letters[j]
            )
            j += 1

    fig = plt.figure(
        figsize=(
            16,
            5.5,
        )
    )

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[
            1.25,
            0.9,
            1.25,
        ],
        wspace=0.28,
    )


    # ========================================================
    # PANEL 1
    # ========================================================

    ax1 = fig.add_subplot(
        gs[0, 0]
    )

    for i, pid in enumerate(
        top_places
    ):
        g = assignments[
            assignments[
                "place_candidate_id"
            ].astype(int)
            .eq(pid)
        ]

        if g.empty:
            continue

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

        color = (
            HOME_COLOR
            if pid == home_place_id
            else OTHER_COLORS[
                max(
                    0,
                    i - 1,
                )
            ]
        )

        ax1.scatter(
            x,
            y,
            s=12,
            alpha=0.35,
            color=color,
        )

        if len(x) >= 3:
            hull = convex_hull(
                np.c_[
                    x,
                    y,
                ]
            )

            if len(hull) >= 3:
                ax1.add_patch(
                    Polygon(
                        hull,
                        closed=True,
                        fill=False,
                        edgecolor=color,
                        linewidth=2,
                    )
                )

        ax1.text(
            np.median(x),
            np.median(y),
            labels[pid],
            fontsize=15,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="circle,pad=0.25",
                fc="white",
                ec=color,
                lw=1.5,
            ),
        )

    ax1.set_title(
        "Candidate places",
        fontsize=17,
        weight="bold",
    )

    ax1.set_xlabel(
        "Relative location"
    )

    ax1.set_ylabel(
        "Relative location"
    )

    ax1.grid(
        alpha=0.12
    )


    # ========================================================
    # PANEL 2
    # ========================================================

    ax2 = fig.add_subplot(
        gs[0, 1]
    )

    nbp = night_by_place[
        night_by_place[
            "place_candidate_id"
        ].astype(int)
        .isin(
            top_places
        )
    ].copy()

    nbp[
        "place_candidate_id"
    ] = (
        nbp[
            "place_candidate_id"
        ].astype(int)
    )

    nbp[
        "label"
    ] = (
        nbp[
            "place_candidate_id"
        ]
        .map(labels)
    )

    nbp[
        "order"
    ] = (
        nbp[
            "place_candidate_id"
        ]
        .map(
            {
                pid: i
                for i, pid
                in enumerate(
                    top_places
                )
            }
        )
    )

    nbp = nbp.sort_values(
        "order"
    )

    colors = [
        HOME_COLOR
        if pid == home_place_id
        else "#9aa9bf"
        for pid in nbp[
            "place_candidate_id"
        ]
    ]

    ypos = np.arange(
        len(nbp)
    )

    ax2.barh(
        ypos,
        nbp["hours"],
        color=colors,
        height=0.58,
    )

    ax2.set_yticks(
        ypos
    )

    ax2.set_yticklabels(
        nbp["label"]
    )

    ax2.invert_yaxis()

    ax2.set_xlabel(
        "Nighttime hours"
    )

    ax2.set_title(
        "12 AM–6 AM",
        fontsize=17,
        weight="bold",
    )

    for yy, hh in zip(
        ypos,
        nbp["hours"],
    ):
        ax2.text(
            hh
            + max(
                nbp[
                    "hours"
                ].max()
                * 0.02,
                0.1,
            ),
            yy,
            f"{hh:.1f} h",
            va="center",
            fontsize=11,
        )

    ax2.text(
        0.5,
        -0.12,
        "A → HOME",
        transform=ax2.transAxes,
        ha="center",
        fontsize=14,
        weight="bold",
        color=HOME_COLOR,
    )

    ax2.grid(
        axis="x",
        alpha=0.12,
    )


    # ========================================================
    # PANEL 3
    # ========================================================

    ax3 = fig.add_subplot(
        gs[0, 2]
    )

    timeline, day_start_ms = (
        build_day_timeline(
            assignments,
            home_place_id,
            EXAMPLE_LOCAL_DATE,
        )
    )

    home_h = 0.0
    away_h = 0.0
    unknown_h = 0.0

    for s, e, kind in timeline:
        x0 = (
            s
            - day_start_ms
        ) / 3_600_000

        width = (
            e - s
        ) / 3_600_000

        if kind == "home":
            color = HOME_COLOR
            home_h += width

        elif kind == "away":
            color = AWAY_COLOR
            away_h += width

        else:
            color = UNKNOWN_COLOR
            unknown_h += width

        ax3.add_patch(
            Rectangle(
                (
                    x0,
                    0.38,
                ),
                width,
                0.30,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
            )
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
            "12 AM",
            "6 AM",
            "12 PM",
            "6 PM",
            "12 AM",
        ]
    )

    ax3.set_yticks([])

    ax3.set_title(
        pd.Timestamp(
            EXAMPLE_LOCAL_DATE
        ).strftime(
            "%b %d"
        ),
        fontsize=17,
        weight="bold",
    )

    ax3.text(
        12,
        0.18,
        f"{home_h:.1f} h at home",
        ha="center",
        fontsize=17,
        weight="bold",
    )

    ax3.spines[
        [
            "left",
            "right",
            "top",
        ]
    ].set_visible(
        False
    )


    # ========================================================
    # ARROWS + TITLE
    # ========================================================

    fig.text(
        0.365,
        0.50,
        "→",
        fontsize=30,
        ha="center",
        va="center",
    )

    fig.text(
        0.665,
        0.50,
        "→",
        fontsize=30,
        ha="center",
        va="center",
    )

    fig.suptitle(
        "Home time",
        fontsize=24,
        weight="bold",
        y=0.98,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    out_png = (
        OUT_DIR
        / f"home_time_visual_{PARTICIPANT_ID}.png"
    )

    out_pdf = (
        OUT_DIR
        / f"home_time_visual_{PARTICIPANT_ID}.pdf"
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

    print()
    print("Saved:")
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
