from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical"
)

INPUT_PATH = ROOT / "gps_clinical_weekly_features_with_home.parquet"

OUTPUT_DIR = ROOT / "weekly_phq_group_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_dumbbell_clean.png"
PDF_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_dumbbell_clean.pdf"
CSV_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_dumbbell_clean_stats.csv"


# =============================================================================
# HELPERS
# =============================================================================

def mean_ci95(values):
    x = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    n = len(x)

    if n == 0:
        return np.nan, np.nan, np.nan

    mean = float(
        x.mean()
    )

    if n == 1:
        return mean, mean, mean

    se = float(
        x.std(ddof=1)
        / np.sqrt(n)
    )

    margin = (
        1.96
        * se
    )

    return (
        mean,
        mean - margin,
        mean + margin,
    )


def format_value(
    value,
    kind,
):
    if kind == "percent":
        return f"{value:.1f}%"

    if kind == "km":
        return f"{value:.1f}"

    return f"{value:.2f}"


def format_delta(
    value,
    kind,
):
    if kind == "percent":
        return f"{value:+.1f} pp"

    if kind == "km":
        return f"{value:+.1f} km/day"

    return f"{value:+.2f}"


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_parquet(
    INPUT_PATH
).copy()

print("=" * 90)
print("CLEAN WEEKLY PHQ MOBILITY FIGURE")
print("=" * 90)

print(
    "Rows:",
    len(df),
)

print(
    "Participants:",
    df[
        "participant_id"
    ].nunique(),
)


# =============================================================================
# VARIABLES
# =============================================================================

df[
    "mobility_diversity"
] = pd.to_numeric(
    df[
        "location_entropy_mean"
    ],
    errors="coerce",
)

df[
    "time_at_home"
] = pd.to_numeric(
    df[
        "time_at_home_pct"
    ],
    errors="coerce",
)

df[
    "movement_distance"
] = (
    pd.to_numeric(
        df[
            "displacement_distance_mean_m"
        ],
        errors="coerce",
    )
    / 1000.0
)

df[
    "depression_score"
] = pd.to_numeric(
    df[
        "depression_score"
    ],
    errors="coerce",
)

df = df[
    df[
        "depression_score"
    ].notna()
].copy()

df[
    "phq_group"
] = np.where(
    df[
        "depression_score"
    ] <= 4,
    "Lower",
    "Elevated",
)


GROUP_ORDER = [
    "Lower",
    "Elevated",
]


GROUP_COLORS = {
    "Lower":
        "#5B9FC5",

    "Elevated":
        "#E29A64",
}


FEATURES = [
    {
        "column":
            "mobility_diversity",

        "title":
            "Mobility diversity",

        "kind":
            "plain",

        "xlabel":
            "Weekly mean mobility diversity",

        "letter":
            "A",

        "cap_display":
            False,
    },

    {
        "column":
            "time_at_home",

        "title":
            "Time at home",

        "kind":
            "percent",

        "xlabel":
            "Weekly mean time at home (%)",

        "letter":
            "B",

        "cap_display":
            False,
    },

    {
        "column":
            "movement_distance",

        "title":
            "Daily movement distance",

        "kind":
            "km",

        "xlabel":
            "Weekly mean movement distance (km/day)",

        "letter":
            "C",

        "cap_display":
            True,
    },
]


# =============================================================================
# SUMMARY
# =============================================================================

stats_rows = []


for spec in FEATURES:

    col = spec[
        "column"
    ]

    sub = df[
        df[
            col
        ].notna()
    ].copy()


    lower = sub.loc[
        sub[
            "phq_group"
        ].eq(
            "Lower"
        ),
        col,
    ]


    elevated = sub.loc[
        sub[
            "phq_group"
        ].eq(
            "Elevated"
        ),
        col,
    ]


    (
        lower_mean,
        lower_ci_low,
        lower_ci_high,
    ) = mean_ci95(
        lower
    )


    (
        elevated_mean,
        elevated_ci_low,
        elevated_ci_high,
    ) = mean_ci95(
        elevated
    )


    stats_rows.append(
        {
            "feature":
                spec[
                    "title"
                ],

            "lower_mean":
                lower_mean,

            "lower_ci_low":
                lower_ci_low,

            "lower_ci_high":
                lower_ci_high,

            "elevated_mean":
                elevated_mean,

            "elevated_ci_low":
                elevated_ci_low,

            "elevated_ci_high":
                elevated_ci_high,

            "difference":
                (
                    elevated_mean
                    - lower_mean
                ),

            "n_lower":
                len(lower),

            "n_elevated":
                len(elevated),
        }
    )


stats_df = pd.DataFrame(
    stats_rows
)


stats_df.to_csv(
    CSV_PATH,
    index=False,
)


print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    stats_df.to_string(
        index=False
    )
)


# =============================================================================
# PARTICIPANT-LEVEL MEANS
# =============================================================================

participant_tables = {}


for spec in FEATURES:

    col = spec[
        "column"
    ]

    participant_tables[
        col
    ] = (
        df[
            df[
                col
            ].notna()
        ]
        .groupby(
            [
                "participant_id",
                "phq_group",
            ],
            as_index=False,
        )[
            col
        ]
        .mean()
    )


# =============================================================================
# FIGURE
# =============================================================================

plt.rcParams.update(
    {
        "font.size":
            12,

        "axes.titlesize":
            18,

        "axes.labelsize":
            13,

        "xtick.labelsize":
            11,

        "ytick.labelsize":
            13,
    }
)


fig, axes = plt.subplots(
    3,
    1,
    figsize=(
        10.5,
        9.2,
    ),
)


fig.suptitle(
    "Mobility patterns by depression symptom level",
    fontsize=22,
    fontweight="bold",
    y=0.975,
)


rng = np.random.default_rng(
    20261102
)


for ax, spec in zip(
    axes,
    FEATURES,
):

    col = spec[
        "column"
    ]


    participant_means = (
        participant_tables[
            col
        ]
    )


    stat = (
        stats_df.loc[
            stats_df[
                "feature"
            ].eq(
                spec[
                    "title"
                ]
            )
        ]
        .iloc[0]
    )


    y_positions = {
        "Lower":
            1,

        "Elevated":
            0,
    }


    # -------------------------------------------------------------------------
    # Faint participant-level points
    # -------------------------------------------------------------------------

    for group in GROUP_ORDER:

        group_data = (
            participant_means[
                participant_means[
                    "phq_group"
                ].eq(
                    group
                )
            ]
        )


        jitter = rng.normal(
            0,
            0.035,
            size=len(
                group_data
            ),
        )


        ax.scatter(
            group_data[
                col
            ],
            (
                y_positions[
                    group
                ]
                + jitter
            ),
            s=36,
            alpha=0.55,
            color=(
                GROUP_COLORS[
                    group
                ]
            ),
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )


    # -------------------------------------------------------------------------
    # Overall means + CI
    # -------------------------------------------------------------------------

    lower_mean = stat[
        "lower_mean"
    ]

    elevated_mean = stat[
        "elevated_mean"
    ]


    ax.errorbar(
        lower_mean,
        1,
        xerr=[
            [
                lower_mean
                - stat[
                    "lower_ci_low"
                ]
            ],
            [
                stat[
                    "lower_ci_high"
                ]
                - lower_mean
            ],
        ],
        fmt="o",
        markersize=10,
        color="#173650",
        ecolor="#173650",
        elinewidth=3,
        capsize=6,
        capthick=2.5,
        zorder=5,
    )


    ax.errorbar(
        elevated_mean,
        0,
        xerr=[
            [
                elevated_mean
                - stat[
                    "elevated_ci_low"
                ]
            ],
            [
                stat[
                    "elevated_ci_high"
                ]
                - elevated_mean
            ],
        ],
        fmt="o",
        markersize=10,
        color="#173650",
        ecolor="#173650",
        elinewidth=3,
        capsize=6,
        capthick=2.5,
        zorder=5,
    )


    # -------------------------------------------------------------------------
    # Difference annotation
    # -------------------------------------------------------------------------

    difference = stat[
        "difference"
    ]


    if spec[
        "kind"
    ] == "percent":

        summary_text = (
            f"{lower_mean:.1f}%  →  "
            f"{elevated_mean:.1f}%"
            f"      Δ {difference:+.1f} pp"
        )

    elif spec[
        "kind"
    ] == "km":

        summary_text = (
            f"{lower_mean:.1f}  →  "
            f"{elevated_mean:.1f} km/day"
            f"      Δ {difference:+.1f} km/day"
        )

    else:

        summary_text = (
            f"{lower_mean:.2f}  →  "
            f"{elevated_mean:.2f}"
            f"      Δ {difference:+.2f}"
        )


    ax.text(
        0.5,
        1.10,
        summary_text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#173650",
    )


    # -------------------------------------------------------------------------
    # Title + panel letter
    # -------------------------------------------------------------------------

    ax.set_title(
        spec[
            "title"
        ],
        loc="left",
        fontweight="bold",
        pad=26,
    )


    ax.text(
        -0.08,
        1.16,
        spec[
            "letter"
        ],
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
    )


    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------

    ax.set_yticks(
        [
            1,
            0,
        ]
    )


    ax.set_yticklabels(
        [
            "Lower PHQ-9",
            "Elevated PHQ-9",
        ]
    )


    ax.set_xlabel(
        spec[
            "xlabel"
        ]
    )


    ax.grid(
        axis="x",
        alpha=0.16,
    )


    ax.spines[
        "top"
    ].set_visible(
        False
    )


    ax.spines[
        "right"
    ].set_visible(
        False
    )


    ax.spines[
        "left"
    ].set_visible(
        False
    )


    ax.tick_params(
        axis="y",
        length=0,
    )


    # -------------------------------------------------------------------------
    # X limits
    # -------------------------------------------------------------------------

    values = (
        participant_means[
            col
        ]
        .dropna()
        .to_numpy()
    )


    if spec[
        "cap_display"
    ]:

        display_max = float(
            np.nanpercentile(
                values,
                99,
            )
        )


        data_min = float(
            np.nanmin(
                values
            )
        )


        span = (
            display_max
            - data_min
        )


        pad = (
            0.08
            * span
        )


        ax.set_xlim(
            max(
                0,
                data_min - pad,
            ),
            display_max + pad,
        )


        n_beyond = int(
            np.sum(
                values
                > display_max
            )
        )


        if n_beyond > 0:

            ax.text(
                0.99,
                0.04,
                (
                    f"{n_beyond} participant-level value"
                    + (
                        "s"
                        if n_beyond != 1
                        else ""
                    )
                    + " above display range"
                ),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#666666",
            )


    else:

        data_min = float(
            np.nanmin(
                values
            )
        )

        data_max = float(
            np.nanmax(
                values
            )
        )


        span = (
            data_max
            - data_min
        )


        pad = (
            0.08
            * span
            if span > 0
            else 1
        )


        if spec[
            "kind"
        ] == "percent":

            ax.set_xlim(
                max(
                    0,
                    data_min - pad,
                ),
                min(
                    100,
                    data_max + pad,
                ),
            )

        else:

            ax.set_xlim(
                data_min - pad,
                data_max + pad,
            )


plt.tight_layout(
    rect=[
        0.06,
        0.04,
        1,
        0.93,
    ],
    h_pad=2.5,
)


plt.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    PDF_PATH,
    bbox_inches="tight",
)

plt.close()


print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print(PNG_PATH)
print(PDF_PATH)
print(CSV_PATH)
