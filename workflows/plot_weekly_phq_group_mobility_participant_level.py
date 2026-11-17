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

PNG_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_participant_level.png"
PDF_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_participant_level.pdf"
CSV_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_participant_level_stats.csv"


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
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mean = float(
        x.mean()
    )

    if n == 1:
        return (
            mean,
            mean,
            mean,
        )

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


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_parquet(
    INPUT_PATH
).copy()

print("=" * 90)
print("PARTICIPANT-LEVEL PHQ MOBILITY FIGURE")
print("=" * 90)

print(
    "Weekly rows:",
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

        "xlabel":
            "Participant mean mobility diversity",

        "kind":
            "plain",

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

        "xlabel":
            "Participant mean time at home (%)",

        "kind":
            "percent",

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

        "xlabel":
            "Participant mean movement distance (km/day)",

        "kind":
            "km",

        "letter":
            "C",

        "cap_display":
            True,
    },
]


# =============================================================================
# PARTICIPANT × PHQ GROUP MEANS
# =============================================================================

participant_tables = {}

stats_rows = []


for spec in FEATURES:

    col = spec[
        "column"
    ]


    participant_means = (
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
        )
        .agg(
            value=(
                col,
                "mean",
            ),

            n_weeks=(
                col,
                "count",
            ),
        )
    )


    participant_tables[
        col
    ] = participant_means


    lower = (
        participant_means.loc[
            participant_means[
                "phq_group"
            ].eq(
                "Lower"
            ),
            "value",
        ]
    )


    elevated = (
        participant_means.loc[
            participant_means[
                "phq_group"
            ].eq(
                "Elevated"
            ),
            "value",
        ]
    )


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


    paired = (
        participant_means
        .pivot(
            index="participant_id",
            columns="phq_group",
            values="value",
        )
        .dropna()
    )


    stats_rows.append(
        {
            "feature":
                spec[
                    "title"
                ],

            "lower_participant_mean":
                lower_mean,

            "lower_ci_low":
                lower_ci_low,

            "lower_ci_high":
                lower_ci_high,

            "elevated_participant_mean":
                elevated_mean,

            "elevated_ci_low":
                elevated_ci_low,

            "elevated_ci_high":
                elevated_ci_high,

            "difference_elevated_minus_lower":
                (
                    elevated_mean
                    - lower_mean
                ),

            "n_lower_participants":
                len(lower),

            "n_elevated_participants":
                len(elevated),

            "n_participants_with_both_states":
                len(paired),
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
print("PARTICIPANT-LEVEL SUMMARY")
print("=" * 90)

print(
    stats_df.to_string(
        index=False
    )
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
        9.0,
    ),
)


fig.suptitle(
    "Mobility patterns by depression symptom level",
    fontsize=20,
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


    # =========================================================================
    # PARTICIPANT-LEVEL DOTS
    # =========================================================================

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
            0.032,
            size=len(
                group_data
            ),
        )


        ax.scatter(
            group_data[
                "value"
            ],
            (
                y_positions[
                    group
                ]
                + jitter
            ),
            s=34,
            alpha=0.42,
            color=(
                GROUP_COLORS[
                    group
                ]
            ),
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )


    # =========================================================================
    # PARTICIPANT-LEVEL GROUP MEANS + CI
    # =========================================================================

    lower_mean = stat[
        "lower_participant_mean"
    ]

    elevated_mean = stat[
        "elevated_participant_mean"
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


    # =========================================================================
    # SUMMARY TEXT
    # =========================================================================

    difference = stat[
        "difference_elevated_minus_lower"
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
        1.095,
        summary_text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#173650",
    )


    # =========================================================================
    # LABELS
    # =========================================================================

    ax.set_title(
        spec[
            "title"
        ],
        loc="left",
        fontweight="bold",
        pad=25,
    )


    ax.text(
        -0.095,
        1.12,
        spec[
            "letter"
        ],
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
        ha="right",
    )


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


    # =========================================================================
    # STYLE
    # =========================================================================

    ax.grid(
        axis="x",
        alpha=0.14,
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


    # =========================================================================
    # DISPLAY RANGE
    # =========================================================================

    values = (
        participant_means[
            "value"
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


        display_min = float(
            np.nanmin(
                values
            )
        )


        span = (
            display_max
            - display_min
        )


        pad = (
            0.08
            * span
        )


        xmax = (
            display_max
            + pad
        )


        ax.set_xlim(
            max(
                0,
                display_min - pad,
            ),
            xmax,
        )


        n_above = int(
            np.sum(
                values
                > xmax
            )
        )


        if n_above > 0:

            ax.annotate(
                "",
                xy=(
                    0.995,
                    0.16,
                ),
                xytext=(
                    0.955,
                    0.16,
                ),
                xycoords=ax.transAxes,
                arrowprops={
                    "arrowstyle":
                        "-|>",

                    "color":
                        "#777777",

                    "lw":
                        1.4,
                },
            )


            ax.text(
                0.95,
                0.16,
                (
                    f"{n_above} off-scale"
                ),
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=8.5,
                color="#777777",
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
        0.075,
        0.04,
        1,
        0.935,
    ],
    h_pad=2.35,
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
