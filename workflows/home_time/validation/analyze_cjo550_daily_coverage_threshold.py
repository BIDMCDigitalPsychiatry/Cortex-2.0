from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PID = "CJO550"

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time"
)

INPUT = ROOT / "daily_home_time.csv"

OUT_DIR = (
    ROOT
    / "home_reference_stability"
    / PID
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_OUT = (
    OUT_DIR
    / "CJO550_daily_coverage_threshold.csv"
)

PNG_OUT = (
    OUT_DIR
    / "CJO550_daily_coverage_threshold.png"
)


# ============================================================
# LOAD CJO550 DAILY HOME-TIME DATA
# ============================================================

df = pd.read_csv(
    INPUT,
    low_memory=False,
)

df = df[
    df["participant_id"]
    .astype(str)
    .eq(PID)
].copy()

if df.empty:
    raise RuntimeError(
        f"{PID} not found."
    )


# ============================================================
# IDENTIFY SUPPORTED GPS DURATION
# ============================================================

df["supported_duration_ms"] = pd.to_numeric(
    df["supported_duration_ms"],
    errors="coerce",
)

df["raw_home_duration_ms"] = pd.to_numeric(
    df["raw_home_duration_ms"],
    errors="coerce",
)

df["supported_hours"] = (
    df["supported_duration_ms"]
    / 3_600_000
)


# ============================================================
# THRESHOLDS TO TEST
# ============================================================

THRESHOLDS = [
    2,
    4,
    6,
    8,
    10,
    12,
]


rows = []

for threshold in THRESHOLDS:

    d = df[
        df["supported_hours"]
        >= threshold
    ].copy()

    if d.empty:

        rows.append(
            {
                "minimum_daily_coverage_hours":
                    threshold,

                "n_days_retained":
                    0,

                "retained_pct":
                    0,

                "weighted_home_time_pct":
                    np.nan,

                "mean_daily_home_time_pct":
                    np.nan,
            }
        )

        continue


    # --------------------------------------------------------
    # Weighted home-time estimate:
    # total observed time at home / total supported GPS time
    #
    # This is preferable to treating every day equally when
    # daily observed durations differ.
    # --------------------------------------------------------

    total_supported = (
        d[
            "supported_duration_ms"
        ].sum()
    )

    total_home = (
        d[
            "raw_home_duration_ms"
        ].sum()
    )

    weighted_home_pct = (
        100
        * total_home
        / total_supported
        if total_supported > 0
        else np.nan
    )


    # Daily fraction, calculated from the observed support.
    daily_fraction = (
        d[
            "raw_home_duration_ms"
        ]
        / d[
            "supported_duration_ms"
        ]
    )

    mean_daily_pct = (
        100
        * daily_fraction.mean()
    )


    rows.append(
        {
            "minimum_daily_coverage_hours":
                threshold,

            "n_days_retained":
                len(d),

            "retained_pct":
                100
                * len(d)
                / len(df),

            "weighted_home_time_pct":
                weighted_home_pct,

            "mean_daily_home_time_pct":
                mean_daily_pct,
        }
    )


results = pd.DataFrame(
    rows
)

results.to_csv(
    CSV_OUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 95)
print("CJO550 DAILY GPS COVERAGE SENSITIVITY")
print("=" * 95)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ============================================================
# FIGURE
# ============================================================

plt.rcParams[
    "font.family"
] = "Arial"

plt.rcParams[
    "font.size"
] = 10.5

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9.5, 4.2),
)


# ------------------------------------------------------------
# A — home-time estimate
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    results[
        "minimum_daily_coverage_hours"
    ],
    results[
        "weighted_home_time_pct"
    ],
    marker="o",
    linewidth=2,
)

for _, r in results.iterrows():

    ax.text(
        r[
            "minimum_daily_coverage_hours"
        ],
        r[
            "weighted_home_time_pct"
        ] + 1,
        f'{r["weighted_home_time_pct"]:.1f}%',
        ha="center",
        fontsize=8.5,
    )

ax.axvline(
    6,
    linestyle="--",
    linewidth=1,
    color="0.5",
)

ax.set_xlabel(
    "Minimum supported GPS per day (hours)"
)

ax.set_ylabel(
    "Estimated home time (%)"
)

ax.set_title(
    "A. Stability of home-time estimate"
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


# ------------------------------------------------------------
# B — retained days
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    results[
        "minimum_daily_coverage_hours"
    ],
    results[
        "retained_pct"
    ],
    marker="o",
    linewidth=2,
)

for _, r in results.iterrows():

    ax.text(
        r[
            "minimum_daily_coverage_hours"
        ],
        r[
            "retained_pct"
        ] + 2,
        (
            f'{r["retained_pct"]:.0f}%'
            f'\n(n={int(r["n_days_retained"])})'
        ),
        ha="center",
        fontsize=8,
    )

ax.axvline(
    6,
    linestyle="--",
    linewidth=1,
    color="0.5",
)

ax.set_ylim(
    0,
    110,
)

ax.set_xlabel(
    "Minimum supported GPS per day (hours)"
)

ax.set_ylabel(
    "Study days retained (%)"
)

ax.set_title(
    "B. Data retained"
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


fig.suptitle(
    (
        "CJO550: sensitivity of home-time measurement "
        "to daily GPS coverage requirement"
    ),
    fontsize=13,
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.92,
    ]
)

fig.savefig(
    PNG_OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print()
print("Saved:")
print(CSV_OUT)
print(PNG_OUT)
