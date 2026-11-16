#!/usr/bin/env python3
"""Create a clinician-facing, real-data Cortex-2.0 home-time figure for JRI670.

The figure uses only real Digital Clinic outputs/raw GPS:
  1) observed GPS points in the selected 14-day window,
  2) nighttime (00:00-06:00 local) evidence used to identify home,
  3) daily supported home / supported elsewhere / unobserved time,
  4) the actual home-time equation and participant-specific values.

Nothing is simulated. If a required real-data column cannot be found, the script
stops rather than inventing a value.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator


DEFAULT_OUTPUT_ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic"
)
DEFAULT_RAW_GPS = (
    DEFAULT_OUTPUT_ROOT
    / "feature_reliability"
    / "gps_clinical"
    / "gps_clinical_weekly_raw_points.parquet"
)
DEFAULT_WORKFLOW = Path("workflows/home_time/build/build_weekly_home_time.py")


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "figure.titlesize": 23,
})


def find_single(root: Path, filename: str) -> Path:
    hits = list(root.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename!r} under {root}")
    if len(hits) > 1:
        print(f"[INFO] Multiple {filename} files found; using newest:")
        hits = sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)
        for p in hits[:5]:
            print("   ", p)
    return hits[0]


def pick_col(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not identify {label}. Tried: {', '.join(candidates)}\n"
        f"Available columns: {', '.join(df.columns)}"
    )


def detect_local_timezone(workflow: Path) -> str:
    if workflow.exists():
        text = workflow.read_text(errors="ignore")
        patterns = [
            r'ZoneInfo\(["\']([^"\']+/[^"\']+)["\']\)',
            r'(?:LOCAL_)?TIMEZONE\s*=\s*["\']([^"\']+/[^"\']+)["\']',
            r'TZ\s*=\s*["\']([^"\']+/[^"\']+)["\']',
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1)

    raise RuntimeError(
        "Could not recover an explicit local timezone from "
        f"{workflow}. Re-run with --timezone <IANA zone>, using the same zone as "
        "build_weekly_home_time.py."
    )


def latlon_to_local_m(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
):
    r = 6_371_000.0
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)

    x = r * (lon_r - lon0_r) * math.cos(lat0_r)
    y = r * (lat_r - lat0_r)

    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--participant",
        default="JRI670",
    )

    parser.add_argument(
        "--start",
        default="2025-04-23",
    )

    parser.add_argument(
        "--end",
        default="2025-05-07",
    )

    parser.add_argument(
        "--timezone",
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--raw-gps",
        type=Path,
        default=DEFAULT_RAW_GPS,
    )

    parser.add_argument(
        "--workflow",
        type=Path,
        default=DEFAULT_WORKFLOW,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("home_time_JRI670_real_data.png"),
    )

    args = parser.parse_args()

    participant = args.participant

    start_utc = pd.Timestamp(
        args.start,
        tz="UTC",
    )

    end_utc = pd.Timestamp(
        args.end,
        tz="UTC",
    )

    start_ms = int(
        start_utc.timestamp()
        * 1000
    )

    end_ms = int(
        end_utc.timestamp()
        * 1000
    )

    tz_name = (
        args.timezone
        or detect_local_timezone(
            args.workflow
        )
    )

    print(
        "Local timezone used by visualization:",
        tz_name,
    )

    anchors_path = find_single(
        args.output_root,
        "participant_home_anchors.csv",
    )

    daily_path = find_single(
        args.output_root,
        "daily_home_time.csv",
    )

    print(
        "Home anchors:",
        anchors_path,
    )

    print(
        "Daily home time:",
        daily_path,
    )

    print(
        "Raw GPS:",
        args.raw_gps,
    )

    anchors = pd.read_csv(
        anchors_path
    )

    daily = pd.read_csv(
        daily_path
    )

    pid_col_a = pick_col(
        anchors,
        [
            "participant_id",
            "participant",
            "lamp_id",
            "id",
        ],
        "participant ID in home anchors",
    )

    pid_col_d = pick_col(
        daily,
        [
            "participant_id",
            "participant",
            "lamp_id",
            "id",
        ],
        "participant ID in daily home table",
    )

    anchor_rows = anchors[
        anchors[
            pid_col_a
        ].astype(
            str
        ).eq(
            participant
        )
    ].copy()

    if anchor_rows.empty:
        raise RuntimeError(
            f"No home anchor found for participant {participant}"
        )

    if len(anchor_rows) > 1:
        print(
            f"[INFO] {len(anchor_rows)} home-anchor rows found; using first."
        )

    anchor = anchor_rows.iloc[0]

    home_lat_col = pick_col(
        anchor_rows,
        [
            "home_representative_latitude",
            "representative_latitude",
            "home_seed_latitude",
            "seed_latitude",
            "home_latitude",
            "latitude",
        ],
        "home latitude",
    )

    home_lon_col = pick_col(
        anchor_rows,
        [
            "home_representative_longitude",
            "representative_longitude",
            "home_seed_longitude",
            "seed_longitude",
            "home_longitude",
            "longitude",
        ],
        "home longitude",
    )

    home_lat = float(
        anchor[
            home_lat_col
        ]
    )

    home_lon = float(
        anchor[
            home_lon_col
        ]
    )

    radius_col = next(
        (
            c
            for c in [
                "home_seed_support_radius_m",
                "seed_support_radius_m",
                "home_support_radius_m",
                "support_radius_m",
            ]
            if (
                c in anchor_rows.columns
                and pd.notna(
                    anchor[c]
                )
            )
        ),
        None,
    )

    home_radius_m = (
        float(
            anchor[
                radius_col
            ]
        )
        if radius_col
        else None
    )

    home_place_col = pick_col(
        anchor_rows,
        [
            "home_place_candidate_id",
            "home_place_id",
            "place_candidate_id",
        ],
        "home place ID",
    )

    home_place_id = int(
        anchor[
            home_place_col
        ]
    )

    night_frac_col = next(
        (
            c
            for c in [
                "home_nighttime_fraction",
                "home_night_fraction",
            ]
            if c in anchor_rows.columns
        ),
        None,
    )

    night_obs_col = next(
        (
            c
            for c in [
                "nighttime_observed_duration_ms",
                "night_observed_duration_ms",
            ]
            if c in anchor_rows.columns
        ),
        None,
    )

    home_night_col = next(
        (
            c
            for c in [
                "home_nighttime_duration_ms",
                "home_night_duration_ms",
            ]
            if c in anchor_rows.columns
        ),
        None,
    )

    try:
        gps = pd.read_parquet(
            args.raw_gps,
            filters=[
                (
                    "participant_id",
                    "==",
                    participant,
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

        gps = pd.read_parquet(
            args.raw_gps
        )

        gps = gps[
            gps[
                "participant_id"
            ].astype(
                str
            ).eq(
                participant
            )
            &
            pd.to_numeric(
                gps[
                    "timestamp"
                ],
                errors="coerce",
            ).between(
                start_ms,
                end_ms - 1,
            )
        ].copy()

    if gps.empty:
        raise RuntimeError(
            "No raw GPS points found for selected participant/window"
        )

    for c in [
        "timestamp",
        "latitude",
        "longitude",
    ]:

        gps[c] = pd.to_numeric(
            gps[c],
            errors="coerce",
        )

    gps = gps.dropna(
        subset=[
            "timestamp",
            "latitude",
            "longitude",
        ]
    ).copy()

    gps[
        "datetime_utc"
    ] = pd.to_datetime(
        gps[
            "timestamp"
        ],
        unit="ms",
        utc=True,
    )

    gps[
        "datetime_local"
    ] = gps[
        "datetime_utc"
    ].dt.tz_convert(
        tz_name
    )

    gps[
        "local_date"
    ] = (
        gps[
            "datetime_local"
        ]
        .dt.date
        .astype(str)
    )

    gps[
        "hour_local"
    ] = (
        gps[
            "datetime_local"
        ]
        .dt.hour
    )

    gps[
        "is_nighttime"
    ] = gps[
        "hour_local"
    ].between(
        0,
        5,
    )

    daily_p = daily[
        daily[
            pid_col_d
        ].astype(
            str
        ).eq(
            participant
        )
    ].copy()

    date_col = pick_col(
        daily_p,
        [
            "date",
            "local_date",
            "day",
            "study_date",
        ],
        "daily date",
    )

    daily_p[
        date_col
    ] = pd.to_datetime(
        daily_p[
            date_col
        ],
        errors="coerce",
    )

    daily_p = daily_p[
        daily_p[
            date_col
        ].ge(
            start_utc.tz_localize(
                None
            )
        )
        &
        daily_p[
            date_col
        ].lt(
            end_utc.tz_localize(
                None
            )
        )
    ].copy()

    home_duration_col = pick_col(
        daily_p,
        [
            "home_duration_ms",
        ],
        "daily home duration",
    )

    supported_col = pick_col(
        daily_p,
        [
            "supported_duration_ms",
            "observed_duration_ms",
        ],
        "daily supported/observed duration",
    )

    home_fraction_col = next(
        (
            c
            for c in [
                "time_at_home_fraction",
                "home_fraction",
            ]
            if c in daily_p.columns
        ),
        None,
    )

    daily_p[
        home_duration_col
    ] = pd.to_numeric(
        daily_p[
            home_duration_col
        ],
        errors="coerce",
    )

    daily_p[
        supported_col
    ] = pd.to_numeric(
        daily_p[
            supported_col
        ],
        errors="coerce",
    )

    all_dates = pd.date_range(
        start=start_utc.date(),
        end=(
            end_utc
            - pd.Timedelta(
                days=1
            )
        ).date(),
        freq="D",
    )

    day_frame = pd.DataFrame({
        "date":
            all_dates
    })

    merge_cols = [
        date_col,
        home_duration_col,
        supported_col,
    ]

    if home_fraction_col:
        merge_cols.append(
            home_fraction_col
        )

    merge = daily_p[
        merge_cols
    ].copy()

    merge = merge.rename(
        columns={
            date_col:
                "date"
        }
    )

    merge[
        "date"
    ] = pd.to_datetime(
        merge[
            "date"
        ]
    ).dt.normalize()

    day_frame = day_frame.merge(
        merge,
        on="date",
        how="left",
    )

    day_ms = (
        24
        * 60
        * 60
        * 1000.0
    )

    day_frame[
        "home_h"
    ] = (
        day_frame[
            home_duration_col
        ]
        / 3_600_000.0
    )

    day_frame[
        "elsewhere_h"
    ] = (
        (
            day_frame[
                supported_col
            ]
            - day_frame[
                home_duration_col
            ]
        )
        .clip(
            lower=0
        )
        / 3_600_000.0
    )

    day_frame[
        "unobserved_h"
    ] = (
        (
            day_ms
            - day_frame[
                supported_col
            ]
        )
        .clip(
            lower=0
        )
        / 3_600_000.0
    )

    no_row = day_frame[
        supported_col
    ].isna()

    day_frame.loc[
        no_row,
        [
            "home_h",
            "elsewhere_h",
        ],
    ] = 0.0

    day_frame.loc[
        no_row,
        "unobserved_h",
    ] = 24.0

    fig = plt.figure(
        figsize=(
            16,
            9,
        ),
        constrained_layout=False,
    )

    gs = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.97,
        top=0.86,
        bottom=0.10,
        wspace=0.20,
        hspace=0.34,
    )

    fig.suptitle(
        "How Cortex-2.0 derives time at home from raw GPS",
        x=0.055,
        ha="left",
        fontweight="bold",
    )

    fig.text(
        0.055,
        0.905,
        (
            f"Real Digital Clinic example: {participant} | "
            f"{args.start} to {args.end} | no simulated data"
        ),
        fontsize=12,
        ha="left",
    )

    ax1 = fig.add_subplot(
        gs[
            0,
            0,
        ]
    )

    x_all, y_all = latlon_to_local_m(
        gps[
            "latitude"
        ].to_numpy(
            float
        ),
        gps[
            "longitude"
        ].to_numpy(
            float
        ),
        home_lat,
        home_lon,
    )

    night = gps[
        gps[
            "is_nighttime"
        ]
    ]

    x_n, y_n = latlon_to_local_m(
        night[
            "latitude"
        ].to_numpy(
            float
        ),
        night[
            "longitude"
        ].to_numpy(
            float
        ),
        home_lat,
        home_lon,
    )

    finite = (
        np.isfinite(
            x_all
        )
        &
        np.isfinite(
            y_all
        )
    )

    if finite.any():

        r = np.hypot(
            x_all[
                finite
            ],
            y_all[
                finite
            ],
        )

        lim = float(
            np.nanpercentile(
                r,
                95,
            )
        )

        lim = max(
            250.0,
            min(
                lim,
                5000.0,
            ),
        )

    else:

        lim = 1000.0

    nearby = (
        np.hypot(
            x_all,
            y_all,
        )
        <= lim
    )

    nearby_n = (
        np.hypot(
            x_n,
            y_n,
        )
        <= lim
    )

    ax1.scatter(
        x_all[
            nearby
        ],
        y_all[
            nearby
        ],
        s=3,
        alpha=0.12,
        label="All observed GPS",
    )

    ax1.scatter(
        x_n[
            nearby_n
        ],
        y_n[
            nearby_n
        ],
        s=5,
        alpha=0.45,
        label="12 AM–6 AM",
    )

    ax1.scatter(
        [0],
        [0],
        s=90,
        marker="*",
        label="Home anchor",
        zorder=5,
    )

    if (
        home_radius_m is not None
        and np.isfinite(
            home_radius_m
        )
        and home_radius_m > 0
    ):

        ax1.add_patch(
            Circle(
                (
                    0,
                    0,
                ),
                home_radius_m,
                fill=False,
                linewidth=2.0,
            )
        )

        ax1.text(
            0.02,
            0.03,
            (
                f"Home support radius = "
                f"{home_radius_m:.0f} m"
            ),
            transform=ax1.transAxes,
            fontsize=10,
        )

    ax1.set_xlim(
        -lim,
        lim,
    )

    ax1.set_ylim(
        -lim,
        lim,
    )

    ax1.set_aspect(
        "equal",
        adjustable="box",
    )

    ax1.set_xlabel(
        "Meters east/west of inferred home"
    )

    ax1.set_ylabel(
        "Meters north/south of inferred home"
    )

    ax1.set_title(
        "1. Start with observed GPS",
        loc="left",
        fontweight="bold",
    )

    ax1.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
    )

    ax2 = fig.add_subplot(
        gs[
            0,
            1,
        ]
    )

    ax2.axis(
        "off"
    )

    ax2.set_title(
        "2. Identify the participant's home",
        loc="left",
        fontweight="bold",
    )

    ax2.text(
        0.02,
        0.82,
        (
            "Among candidate places, choose the place with the\n"
            "greatest observed occupancy from 12 AM–6 AM local time."
        ),
        fontsize=15,
        va="top",
    )

    ax2.text(
        0.02,
        0.58,
        "Home = arg max",
        fontsize=17,
        fontweight="bold",
    )

    ax2.text(
        0.26,
        0.585,
        "nighttime supported duration(place)",
        fontsize=15,
    )

    ax2.text(
        0.02,
        0.38,
        (
            f"Selected home place ID: "
            f"{home_place_id}"
        ),
        fontsize=13,
    )

    if (
        night_obs_col
        and pd.notna(
            anchor[
                night_obs_col
            ]
        )
    ):

        ax2.text(
            0.02,
            0.29,
            (
                "Observed nighttime duration: "
                f"{float(anchor[night_obs_col]) / 3_600_000:.1f} h"
            ),
            fontsize=13,
        )

    if (
        home_night_col
        and pd.notna(
            anchor[
                home_night_col
            ]
        )
    ):

        ax2.text(
            0.02,
            0.20,
            (
                "Nighttime duration at selected home: "
                f"{float(anchor[home_night_col]) / 3_600_000:.1f} h"
            ),
            fontsize=13,
        )

    if (
        night_frac_col
        and pd.notna(
            anchor[
                night_frac_col
            ]
        )
    ):

        ax2.text(
            0.02,
            0.11,
            (
                "Share of observed nighttime spent there: "
                f"{100 * float(anchor[night_frac_col]):.1f}%"
            ),
            fontsize=13,
        )

    ax2.text(
        0.02,
        0.01,
        (
            "This is different from simply calling the most "
            "common daytime location 'home'."
        ),
        fontsize=10.5,
    )

    ax3 = fig.add_subplot(
        gs[
            1,
            0,
        ]
    )

    x = np.arange(
        len(
            day_frame
        )
    )

    ax3.bar(
        x,
        day_frame[
            "home_h"
        ],
        label="Supported at home",
    )

    ax3.bar(
        x,
        day_frame[
            "elsewhere_h"
        ],
        bottom=day_frame[
            "home_h"
        ],
        label="Supported elsewhere",
    )

    ax3.bar(
        x,
        day_frame[
            "unobserved_h"
        ],
        bottom=(
            day_frame[
                "home_h"
            ]
            + day_frame[
                "elsewhere_h"
            ]
        ),
        label="Unobserved",
    )

    ax3.set_ylim(
        0,
        24,
    )

    ax3.set_ylabel(
        "Hours of day"
    )

    ax3.set_xticks(
        x
    )

    ax3.set_xticklabels(
        [
            d.strftime(
                "%b %d"
            )
            for d in day_frame[
                "date"
            ]
        ],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax3.yaxis.set_major_locator(
        MaxNLocator(
            integer=True,
            nbins=5,
        )
    )

    ax3.set_title(
        "3. Missing time remains unknown",
        loc="left",
        fontweight="bold",
    )

    ax3.legend(
        frameon=False,
        fontsize=9,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
    )

    ax4 = fig.add_subplot(
        gs[
            1,
            1,
        ]
    )

    ax4.axis(
        "off"
    )

    ax4.set_title(
        "4. Calculate time at home",
        loc="left",
        fontweight="bold",
    )

    total_home_h = float(
        day_frame[
            "home_h"
        ].sum()
    )

    total_supported_h = float(
        (
            day_frame[
                "home_h"
            ]
            + day_frame[
                "elsewhere_h"
            ]
        ).sum()
    )

    total_unobs_h = float(
        day_frame[
            "unobserved_h"
        ].sum()
    )

    home_frac = (
        total_home_h
        / total_supported_h
        if total_supported_h > 0
        else np.nan
    )

    ax4.text(
        0.02,
        0.78,
        (
            "Home duration = Σ supported intervals "
            "assigned to the home place"
        ),
        fontsize=15,
        fontweight="bold",
        va="top",
    )

    ax4.text(
        0.02,
        0.58,
        (
            "Observed home time in this 14-day window: "
            f"{total_home_h:.1f} h"
        ),
        fontsize=14,
    )

    ax4.text(
        0.02,
        0.46,
        (
            "Total supported GPS time: "
            f"{total_supported_h:.1f} h"
        ),
        fontsize=14,
    )

    ax4.text(
        0.02,
        0.34,
        (
            "Unobserved time: "
            f"{total_unobs_h:.1f} h"
        ),
        fontsize=14,
    )

    if np.isfinite(
        home_frac
    ):

        ax4.text(
            0.02,
            0.20,
            (
                f"Time at home = "
                f"{total_home_h:.1f} / "
                f"{total_supported_h:.1f} = "
                f"{100 * home_frac:.1f}% "
                "of supported time"
            ),
            fontsize=14,
        )

    ax4.text(
        0.02,
        0.04,
        (
            "Key point: unobserved GPS time is not "
            "automatically labeled as home or away."
        ),
        fontsize=11,
    )

    fig.text(
        0.055,
        0.025,
        (
            "Cortex-2.0 home-time derivation: real participant data; "
            "candidate places are built from temporally supported, "
            "accuracy-aware location episodes."
        ),
        fontsize=9.5,
    )

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.out,
        dpi=220,
        bbox_inches="tight",
    )

    pdf = args.out.with_suffix(
        ".pdf"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "\nSaved:"
    )

    print(
        " ",
        args.out.resolve(),
    )

    print(
        " ",
        pdf.resolve(),
    )

    print(
        "\nParticipant-specific values used:"
    )

    print(
        "  home_place_id:",
        home_place_id,
    )

    print(
        "  home anchor:",
        home_lat,
        home_lon,
    )

    print(
        "  home support radius (m):",
        home_radius_m,
    )

    if night_frac_col:

        print(
            "  home_nighttime_fraction:",
            anchor[
                night_frac_col
            ],
        )

    print(
        "  14-day home hours:",
        round(
            total_home_h,
            3,
        ),
    )

    print(
        "  14-day supported hours:",
        round(
            total_supported_h,
            3,
        ),
    )

    print(
        "  14-day unobserved hours:",
        round(
            total_unobs_h,
            3,
        ),
    )

    print(
        "  home fraction of supported time:",
        home_frac,
    )


if __name__ == "__main__":
    main()
