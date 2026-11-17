from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# CONFIG
# ============================================================

PARTICIPANT = "AMO327"

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

DAILY_FEATURES = (
    ROOT
    / "gps_clinical_weekly_daily_features.parquet"
)

OUT_DIR = (
    ROOT
    / "home_time"
    / "figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_OUT = (
    OUT_DIR
    / "AMO327_location_entropy_real_data.png"
)

PDF_OUT = (
    OUT_DIR
    / "AMO327_location_entropy_real_data.pdf"
)

CSV_OUT = (
    OUT_DIR
    / "AMO327_location_entropy_daily_summary.csv"
)


# ============================================================
# LOAD EXACT PRODUCTION OUTPUT
# ============================================================

df = pd.read_parquet(
    DAILY_FEATURES,
    filters=[
        (
            "participant_id",
            "==",
            PARTICIPANT,
        )
    ],
)

df["date"] = pd.to_datetime(
    df["date"]
).dt.date

# Same 14-day period used in the previous figures.
start_date = pd.Timestamp(
    "2023-11-29"
).date()

end_date = pd.Timestamp(
    "2023-12-12"
).date()

df = df[
    (df["date"] >= start_date)
    &
    (df["date"] <= end_date)
].copy()

# Same participant-date may appear in overlapping clinical windows.
# Production feature values are identical for the same participant-day;
# retain one row for the visualization.
daily = (
    df
    .sort_values(
        [
            "date",
            "clinical_window_id",
        ]
    )
    .drop_duplicates(
        subset=[
            "participant_id",
            "date",
        ],
        keep="first",
    )
    .sort_values(
        "date"
    )
    .reset_index(
        drop=True
    )
)

daily.to_csv(
    CSV_OUT,
    index=False,
)


# ============================================================
# REPRESENTATIVE DAY
# ============================================================

valid = daily[
    daily[
        "location_entropy_nats"
    ].notna()
].copy()

# Choose observed entropy closest to the median.
median_entropy = valid[
    "location_entropy_nats"
].median()

example_i = (
    (
        valid[
            "location_entropy_nats"
        ]
        - median_entropy
    )
    .abs()
    .idxmin()
)

example = daily.loc[
    example_i
]

example_date = example[
    "date"
]

entropy = float(
    example[
        "location_entropy_nats"
    ]
)

effective_places = float(
    example[
        "effective_number_places"
    ]
)

dominant_fraction = float(
    example[
        "dominant_place_fraction"
    ]
)

n_places = int(
    example[
        "n_observed_places"
    ]
)

assigned_fraction = float(
    example[
        "place_time_fraction_of_supported"
    ]
)

coverage = float(
    example[
        "supported_fraction_of_day"
    ]
)


# ============================================================
# FIGURE STYLE
# ============================================================

plt.rcParams[
    "font.family"
] = "Arial"

plt.rcParams[
    "font.size"
] = 10.5


fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        17.5,
        6.0,
    ),
)


# ============================================================
# PANEL A
# DOMINANT PLACE VS ALL OTHER ASSIGNED PLACES
# ============================================================

ax = axes[0]

other_fraction = (
    1.0
    - dominant_fraction
)

ax.barh(
    [0],
    [
        dominant_fraction * 100
    ],
    height=0.45,
    color="0.30",
)

ax.barh(
    [0],
    [
        other_fraction * 100
    ],
    left=[
        dominant_fraction * 100
    ],
    height=0.45,
    color="0.75",
)

ax.text(
    dominant_fraction * 50,
    0,
    (
        f"Dominant place\n"
        f"{dominant_fraction * 100:.1f}%"
    ),
    ha="center",
    va="center",
    color="white",
    fontsize=10,
)

ax.text(
    dominant_fraction * 100
    + other_fraction * 50,
    0,
    (
        f"Other places\n"
        f"{other_fraction * 100:.1f}%"
    ),
    ha="center",
    va="center",
    fontsize=10,
)

ax.set_xlim(
    0,
    100,
)

ax.set_ylim(
    -0.7,
    0.7,
)

ax.set_yticks([])

ax.set_xlabel(
    "% of assigned place time"
)

ax.set_title(
    (
        "A. How concentrated was location time?\n"
        f"{pd.Timestamp(example_date).strftime('%b %d')}"
    )
)

ax.text(
    0.02,
    0.05,
    (
        f"Observed places = {n_places}\n"
        f"Assigned place time = "
        f"{assigned_fraction * 100:.1f}% of supported GPS"
    ),
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)

ax.spines[
    "left"
].set_visible(False)


# ============================================================
# PANEL B
# ENTROPY INTERPRETATION
# ============================================================

ax = axes[1]

ax.axis(
    "off"
)

ax.set_title(
    "B. Cortex-2.0 entropy summary"
)

ax.text(
    0.05,
    0.80,
    r"$H=-\sum_i p_i \ln(p_i)$",
    transform=ax.transAxes,
    fontsize=19,
)

ax.text(
    0.05,
    0.66,
    (
        "pᵢ = fraction of assigned observed time\n"
        "spent at detected place i"
    ),
    transform=ax.transAxes,
    fontsize=10,
)

ax.text(
    0.05,
    0.47,
    f"Entropy",
    transform=ax.transAxes,
    fontsize=10,
)

ax.text(
    0.95,
    0.47,
    f"{entropy:.2f} nats",
    transform=ax.transAxes,
    ha="right",
    fontsize=16,
    fontweight="bold",
)

ax.text(
    0.05,
    0.34,
    "Effective number of places",
    transform=ax.transAxes,
    fontsize=10,
)

ax.text(
    0.95,
    0.34,
    f"{effective_places:.2f}",
    transform=ax.transAxes,
    ha="right",
    fontsize=16,
    fontweight="bold",
)

ax.text(
    0.05,
    0.19,
    (
        "Effective places = exp(H)\n"
        "Higher entropy means observed time is distributed\n"
        "more evenly across places."
    ),
    transform=ax.transAxes,
    fontsize=9.5,
)


# ============================================================
# PANEL C
# DAILY PRODUCTION ENTROPY
# ============================================================

ax = axes[2]

x = np.arange(
    len(
        daily
    )
)

vals = pd.to_numeric(
    daily[
        "location_entropy_nats"
    ],
    errors="coerce",
).to_numpy()

example_pos = int(
    daily.index[
        daily[
            "date"
        ].eq(
            example_date
        )
    ][0]
)

for i, value in enumerate(
    vals
):

    if not np.isfinite(
        value
    ):
        continue

    ax.bar(
        i,
        value,
        width=0.72,
        color=(
            "0.25"
            if i == example_pos
            else "0.72"
        ),
    )

ax.text(
    example_pos,
    vals[
        example_pos
    ] + 0.03,
    f"{vals[example_pos]:.2f}",
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
)

labels = [
    pd.Timestamp(d)
    .strftime(
        "%b %d"
    )
    for d in daily[
        "date"
    ]
]

ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels,
    rotation=45,
    ha="right",
)

ax.set_ylabel(
    "Location entropy (nats)"
)

ax.set_title(
    "C. Daily Cortex-2.0 location entropy"
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)

ax.text(
    0.98,
    0.97,
    (
        "14-day window\n"
        f"{daily['location_entropy_nats'].notna().sum()} "
        "days with computed entropy\n"
        f"Median daily entropy = "
        f"{daily['location_entropy_nats'].median():.2f} nats"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
)


# ============================================================
# COVERAGE STRIP
# ============================================================

pos = ax.get_position()

coverage_ax = fig.add_axes(
    [
        pos.x0,
        pos.y0 - 0.052,
        pos.width,
        0.025,
    ]
)

coverage_ax.set_xlim(
    -0.5,
    len(daily) - 0.5,
)

coverage_ax.set_ylim(
    0,
    1,
)

for i, cov in enumerate(
    daily[
        "supported_fraction_of_day"
    ]
):

    if not np.isfinite(
        cov
    ):
        continue

    shade = (
        0.92
        - 0.72
        * float(cov)
    )

    shade = min(
        max(
            shade,
            0.18,
        ),
        0.92,
    )

    coverage_ax.add_patch(
        Rectangle(
            (
                i - 0.36,
                0.12,
            ),
            0.72,
            0.76,
            facecolor=str(
                shade
            ),
            edgecolor="none",
        )
    )

coverage_ax.set_xticks([])
coverage_ax.set_yticks([])

for spine in coverage_ax.spines.values():
    spine.set_visible(False)

coverage_ax.text(
    -0.02,
    0.5,
    "GPS coverage",
    transform=coverage_ax.transAxes,
    ha="right",
    va="center",
    fontsize=8,
)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    (
        "AMO327: How Cortex-2.0 summarizes location entropy\n"
        "Entropy captures how evenly observed time is "
        "distributed across detected places"
    ),
    fontsize=16,
)

fig.subplots_adjust(
    top=0.78,
    bottom=0.27,
    wspace=0.38,
)

fig.savefig(
    PNG_OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


print("=" * 80)
print("LOCATION ENTROPY FIGURE")
print("=" * 80)
print("Representative day:", example_date)
print("Entropy:", entropy)
print("Effective places:", effective_places)
print("Observed places:", n_places)
print("Dominant-place fraction:", dominant_fraction)
print("Assigned-place fraction:", assigned_fraction)
print("GPS coverage:", coverage)
print()
print("Saved:")
print(PNG_OUT)
print(PDF_OUT)
