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

PNG_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_dumbbell.png"
PDF_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_dumbbell.pdf"
CSV_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_dumbbell_stats.csv"


# =============================================================================
# HELPERS
# =============================================================================

def mean_ci95(x):
    x = pd.to_numeric(
        x,
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


def format_value(
    value,
    kind,
):

    if kind == "percent":
        return f"{value:.1f}%"

    if kind == "km":
        return f"{value:.1f} km/day"

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
print("WEEKLY PHQ GROUP MOBILITY DUMBBELL")
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
# CLINICAL FEATURES
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
    "Lower PHQ-9 (0–4)",
    "Elevated PHQ-9 (5+)",
)


GROUP_ORDER = [
    "Lower PHQ-9 (0–4)",
    "Elevated PHQ-9 (5+)",
]


GROUP_COLORS = {
    "Lower PHQ-9 (0–4)":
        "#579BC2",

    "Elevated PHQ-9 (5+)":
        "#DD9560",
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
            GROUP_ORDER[0]
        ),
        col,
    ]


    elevated = sub.loc[
        sub[
            "phq_group"
        ].eq(
            GROUP_ORDER[1]
        ),
        col,
    ]


    (
        lower_mean,
        lower_low,
        lower_high,
    ) = mean_ci95(
        lower
    )


    (
        elevated_mean,
        elevated_low,
        elevated_high,
    ) = mean_ci95(
        elevated
    )


    participant_means = (
        sub.groupby(
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


    paired = (
        participant_means
        .pivot(
            index="participant_id",
            columns="phq_group",
            values=col,
        )
        .dropna()
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
                lower_low,

            "lower_ci_high":
                lower_high,

            "elevated_mean":
                elevated_mean,

            "elevated_ci_low":
                elevated_low,

            "elevated_ci_high":
                elevated_high,

            "difference":
                (
                    elevated_mean
                    - lower_mean
                ),

            "n_lower_rows":
                len(lower),

            "n_elevated_rows":
                len(elevated),

            "n_paired_participants":
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
print("SUMMARY")
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
            17,

        "axes.labelsize":
            12,

        "xtick.labelsize":
            11,

        "ytick.labelsize":
            12,
    }
)


fig, axes = plt.subplots(
    3,
    1,
    figsize=(
        10,
        10,
    ),
)


fig.suptitle(
    "Mobility patterns by depression symptom level",
    fontsize=23,
    fontweight="bold",
    y=0.98,
)


fig.text(
    0.5,
    0.945,
    (
        "Lower PHQ-9 symptoms (0–4) "
        "versus elevated symptoms (5+)"
    ),
    ha="center",
    fontsize=13,
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


    sub = df[
        df[
            col
        ].notna()
    ].copy()


    participant_means = (
        sub.groupby(
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


    paired = (
        participant_means
        .pivot(
            index="participant_id",
            columns="phq_group",
            values=col,
        )
        .dropna()
    )


    y_positions = {
        GROUP_ORDER[0]:
            1,

        GROUP_ORDER[1]:
            0,
    }


    # --------------------------------------------------------
    # Participant-level paired lines
    # --------------------------------------------------------

    for participant_id, row in paired.iterrows():

        ax.plot(
            [
                row[
                    GROUP_ORDER[0]
                ],
                row[
                    GROUP_ORDER[1]
                ],
            ],
            [
                1,
                0,
            ],
            color="#CED3D8",
            linewidth=1,
            alpha=0.45,
            zorder=1,
        )


    # --------------------------------------------------------
    # Participant-level means
    # --------------------------------------------------------

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
            0.045,
            len(
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
            s=42,
            alpha=0.65,
            color=(
                GROUP_COLORS[
                    group
                ]
            ),
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )


    # --------------------------------------------------------
    # Overall means + CI
    # --------------------------------------------------------

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
        markersize=11,
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
        markersize=11,
        color="#173650",
        ecolor="#173650",
        elinewidth=3,
        capsize=6,
        capthick=2.5,
        zorder=5,
    )


    # --------------------------------------------------------
    # Main difference line
    # --------------------------------------------------------

    ax.plot(
        [
            lower_mean,
            elevated_mean,
        ],
        [
            1,
            0,
        ],
        color="#173650",
        linewidth=2.5,
        zorder=4,
    )


    difference = (
        elevated_mean
        - lower_mean
    )


    annotation = (
        f"{format_value(lower_mean, spec['kind'])}"
        "  →  "
        f"{format_value(elevated_mean, spec['kind'])}"
        "     "
        f"Δ {format_delta(difference, spec['kind'])}"
    )


    ax.text(
        0.5,
        1.10,
        annotation,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#173650",
    )


    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

    ax.set_title(
        spec[
            "title"
        ],
        fontweight="bold",
        loc="left",
        pad=28,
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
    )


    ax.set_yticks(
        [
            1,
            0,
        ]
    )


    ax.set_yticklabels(
        [
            "Lower",
            "Elevated",
        ]
    )


    ax.set_xlabel(
        spec[
            "xlabel"
        ]
    )


    ax.grid(
        axis="x",
        alpha=0.18,
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


plt.tight_layout(
    rect=[
        0.06,
        0.04,
        1,
        0.91,
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
