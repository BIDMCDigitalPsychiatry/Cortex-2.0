from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical"
)

INPUT = (
    ROOT
    / "gps_clinical_weekly_features_with_home.parquet"
)

OUT_DIR = (
    ROOT
    / "weekly_phq_group_figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_OUT = (
    OUT_DIR
    / "weekly_phq_clinical_mobility_summary.png"
)

PDF_OUT = (
    OUT_DIR
    / "weekly_phq_clinical_mobility_summary.pdf"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(
    INPUT
).copy()

df = df[
    df[
        "depression_score"
    ].notna()
].copy()


df[
    "depression_group"
] = np.where(
    df[
        "depression_score"
    ] <= 4,
    "Lower depression",
    "Elevated depression",
)


# ============================================================
# CLINICAL FEATURES
# ============================================================

df[
    "mobility_diversity"
] = (
    df[
        "location_entropy_mean"
    ]
)

df[
    "time_at_home"
] = (
    df[
        "time_at_home_pct"
    ]
)

df[
    "movement_distance"
] = (
    df[
        "displacement_distance_mean_m"
    ]
    / 1000.0
)


FEATURES = [
    {
        "column":
            "mobility_diversity",

        "title":
            "Mobility diversity",

        "ylabel":
            "Mobility diversity",

        "difference_unit":
            "",

        "decimals":
            2,
    },

    {
        "column":
            "time_at_home",

        "title":
            "Time at home",

        "ylabel":
            "Time at home (%)",

        "difference_unit":
            " percentage points",

        "decimals":
            1,
    },

    {
        "column":
            "movement_distance",

        "title":
            "Daily movement distance",

        "ylabel":
            "Movement distance (km/day)",

        "difference_unit":
            " km/day",

        "decimals":
            1,
    },
]


GROUP_ORDER = [
    "Lower depression",
    "Elevated depression",
]


COLORS = {
    "Lower depression":
        "#4A90B8",

    "Elevated depression":
        "#E59A61",
}


# ============================================================
# HELPERS
# ============================================================

def mean_ci(values):

    x = (
        pd.Series(
            values
        )
        .dropna()
        .astype(float)
    )

    mean = x.mean()

    se = (
        x.std(
            ddof=1
        )
        / np.sqrt(
            len(x)
        )
    )

    ci = (
        1.96
        * se
    )

    return (
        float(mean),
        float(mean - ci),
        float(mean + ci),
    )


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for feature in FEATURES:

    col = feature[
        "column"
    ]

    lower = (
        df.loc[
            df[
                "depression_group"
            ].eq(
                "Lower depression"
            ),
            col,
        ]
        .dropna()
    )

    elevated = (
        df.loc[
            df[
                "depression_group"
            ].eq(
                "Elevated depression"
            ),
            col,
        ]
        .dropna()
    )


    lower_mean = float(
        lower.mean()
    )

    elevated_mean = float(
        elevated.mean()
    )

    difference = (
        elevated_mean
        - lower_mean
    )


    summary_rows.append(
        {
            "feature":
                feature[
                    "title"
                ],

            "lower_mean":
                lower_mean,

            "elevated_mean":
                elevated_mean,

            "difference":
                difference,

            "n_lower":
                len(lower),

            "n_elevated":
                len(elevated),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print("=" * 90)
print("CLINICAL MOBILITY SUMMARY")
print("=" * 90)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# FIGURE
# ============================================================

plt.rcParams.update(
    {
        "font.family":
            "DejaVu Sans",

        "font.size":
            12,

        "axes.titlesize":
            16,

        "axes.labelsize":
            12,

        "xtick.labelsize":
            11,

        "ytick.labelsize":
            11,
    }
)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        15.5,
        5.6,
    ),
)


fig.suptitle(
    "Mobility patterns differ by depression symptom level",
    fontsize=21,
    fontweight="bold",
    y=1.03,
)


fig.text(
    0.5,
    0.965,
    (
        "Weekly mobility during lower PHQ-9 symptoms (0–4) "
        "versus elevated symptoms (5+)"
    ),
    ha="center",
    fontsize=12,
)


rng = np.random.default_rng(
    20261102
)


panel_labels = [
    "A",
    "B",
    "C",
]


for (
    ax,
    feature,
    panel,
) in zip(
    axes,
    FEATURES,
    panel_labels,
):

    col = feature[
        "column"
    ]


    group_means = {}


    for x_pos, group in enumerate(
        GROUP_ORDER
    ):

        values = (
            df.loc[
                df[
                    "depression_group"
                ].eq(
                    group
                ),
                col,
            ]
            .dropna()
            .astype(float)
            .to_numpy()
        )


        # Individual weekly observations
        jitter = rng.normal(
            0,
            0.055,
            size=len(values),
        )

        ax.scatter(
            np.full(
                len(values),
                x_pos,
            )
            + jitter,
            values,
            s=31,
            alpha=0.72,
            color=COLORS[
                group
            ],
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )


        mean, low, high = (
            mean_ci(
                values
            )
        )

        group_means[
            group
        ] = mean


        # Mean + CI
        ax.errorbar(
            x_pos,
            mean,
            yerr=[
                [
                    mean
                    - low
                ],
                [
                    high
                    - mean
                ],
            ],
            fmt="_",
            color="#15324A",
            markersize=34,
            markeredgewidth=3.5,
            elinewidth=2.8,
            capsize=0,
            zorder=5,
        )


    lower_mean = (
        group_means[
            "Lower depression"
        ]
    )

    elevated_mean = (
        group_means[
            "Elevated depression"
        ]
    )

    difference = (
        elevated_mean
        - lower_mean
    )


    decimals = feature[
        "decimals"
    ]


    if (
        feature[
            "column"
        ]
        == "mobility_diversity"
    ):

        difference_text = (
            f"{lower_mean:.2f} → "
            f"{elevated_mean:.2f}\n"
            f"Difference {difference:+.2f}"
        )

    elif (
        feature[
            "column"
        ]
        == "time_at_home"
    ):

        difference_text = (
            f"{lower_mean:.1f}% → "
            f"{elevated_mean:.1f}%\n"
            f"Difference {difference:+.1f} pp"
        )

    else:

        difference_text = (
            f"{lower_mean:.1f} → "
            f"{elevated_mean:.1f} km/day\n"
            f"Difference {difference:+.1f} km/day"
        )


    ax.text(
        0.5,
        0.96,
        difference_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#15324A",
        bbox={
            "boxstyle":
                "round,pad=0.4",

            "facecolor":
                "white",

            "edgecolor":
                "#D8DEE3",

            "alpha":
                0.94,
        },
    )


    ax.set_title(
        feature[
            "title"
        ],
        pad=14,
        fontweight="bold",
    )


    ax.set_ylabel(
        feature[
            "ylabel"
        ]
    )


    ax.set_xticks(
        [
            0,
            1,
        ]
    )

    ax.set_xticklabels(
        [
            "Lower",
            "Elevated",
        ]
    )


    ax.set_xlabel(
        "Depression symptoms"
    )


    ax.grid(
        axis="y",
        alpha=0.20,
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


    ax.text(
        -0.14,
        1.06,
        panel,
        transform=ax.transAxes,
        fontsize=19,
        fontweight="bold",
    )


# ============================================================
# LEGEND
# ============================================================

handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor=(
            COLORS[
                "Lower depression"
            ]
        ),
        markeredgecolor="white",
        markersize=9,
        label="Lower PHQ-9 (0–4)",
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor=(
            COLORS[
                "Elevated depression"
            ]
        ),
        markeredgecolor="white",
        markersize=9,
        label="Elevated PHQ-9 (5+)",
    ),
]


fig.legend(
    handles=handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(
        0.5,
        -0.03,
    ),
    fontsize=12,
)


# ============================================================
# TAKEAWAY
# ============================================================

fig.text(
    0.5,
    -0.08,
    (
        "Elevated depression symptoms were accompanied by "
        "less diverse mobility, more time at home, "
        "and lower daily movement."
    ),
    ha="center",
    fontsize=12,
    fontweight="bold",
    color="#15324A",
)


plt.tight_layout(
    rect=[
        0,
        0.08,
        1,
        0.93,
    ]
)


plt.savefig(
    PNG_OUT,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    PDF_OUT,
    bbox_inches="tight",
)

plt.close()


print()
print("Saved:")
print(" ", PNG_OUT)
print(" ", PDF_OUT)
