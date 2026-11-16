from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


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

PNG_PATH = OUTPUT_DIR / "weekly_phq_home_time_robustness.png"
PDF_PATH = OUTPUT_DIR / "weekly_phq_home_time_robustness.pdf"
CSV_PATH = OUTPUT_DIR / "weekly_phq_home_time_robustness_stats.csv"


# =============================================================================
# SETTINGS
# =============================================================================

N_BOOT = 5000
RANDOM_SEED = 20261102

GROUP_ORDER = [
    "Lower",
    "Elevated",
]

GROUP_COLORS = {
    "Lower": "#5B9FC5",
    "Elevated": "#E29A64",
}


# =============================================================================
# HELPERS
# =============================================================================

def format_p(p):
    if pd.isna(p):
        return "p = NA"

    if p < 0.001:
        return "p < 0.001"

    return f"p = {p:.3f}"


def cluster_bootstrap_group_means(
    data,
    value_col,
    n_boot=5000,
    seed=20261102,
):
    rng = np.random.default_rng(seed)

    participants = (
        data["participant_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    participant_frames = {
        pid: data.loc[
            data["participant_id"]
            .astype(str)
            .eq(pid)
        ].copy()
        for pid in participants
    }

    lower_boot = []
    elevated_boot = []
    difference_boot = []

    for _ in range(n_boot):

        sampled_ids = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        pieces = []

        for bootstrap_index, pid in enumerate(sampled_ids):

            piece = (
                participant_frames[pid]
                .copy()
            )

            piece["_boot_cluster"] = (
                f"{pid}_{bootstrap_index}"
            )

            pieces.append(piece)

        boot = pd.concat(
            pieces,
            ignore_index=True,
        )

        lower = pd.to_numeric(
            boot.loc[
                boot["phq_group"].eq("Lower"),
                value_col,
            ],
            errors="coerce",
        ).dropna()

        elevated = pd.to_numeric(
            boot.loc[
                boot["phq_group"].eq("Elevated"),
                value_col,
            ],
            errors="coerce",
        ).dropna()

        if (
            len(lower) == 0
            or len(elevated) == 0
        ):
            continue

        lower_mean = float(
            lower.mean()
        )

        elevated_mean = float(
            elevated.mean()
        )

        lower_boot.append(
            lower_mean
        )

        elevated_boot.append(
            elevated_mean
        )

        difference_boot.append(
            elevated_mean
            - lower_mean
        )

    def ci(values):
        values = np.asarray(
            values,
            dtype=float,
        )

        return (
            float(
                np.nanpercentile(
                    values,
                    2.5,
                )
            ),
            float(
                np.nanpercentile(
                    values,
                    97.5,
                )
            ),
        )

    return (
        ci(lower_boot),
        ci(elevated_boot),
        ci(difference_boot),
    )


def clustered_ols(
    data,
    value_col,
):
    x = data[
        [
            "participant_id",
            "phq_group",
            value_col,
        ]
    ].dropna().copy()

    x["elevated"] = (
        x["phq_group"]
        .eq("Elevated")
        .astype(int)
    )

    X = sm.add_constant(
        x[
            [
                "elevated",
            ]
        ]
    )

    y = pd.to_numeric(
        x[
            value_col
        ],
        errors="coerce",
    )

    model = sm.OLS(
        y,
        X,
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                x[
                    "participant_id"
                ]
        },
    )

    beta = float(
        result.params[
            "elevated"
        ]
    )

    p_value = float(
        result.pvalues[
            "elevated"
        ]
    )

    ci = (
        result
        .conf_int()
        .loc[
            "elevated"
        ]
    )

    return {
        "beta":
            beta,

        "ci_low":
            float(
                ci.iloc[0]
            ),

        "ci_high":
            float(
                ci.iloc[1]
            ),

        "p_value":
            p_value,
    }


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_parquet(
    INPUT_PATH
).copy()

print("=" * 100)
print("WEEKLY PHQ HOME-TIME ROBUSTNESS FIGURE")
print("=" * 100)

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
# PHQ GROUPS
# =============================================================================

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


# =============================================================================
# HOME-TIME REPRESENTATIONS
# =============================================================================

# 1. Aggregate weekly home-time percentage
df[
    "weekly_home_pct"
] = pd.to_numeric(
    df[
        "time_at_home_pct"
    ],
    errors="coerce",
)

# 2. Equivalent hours/day
df[
    "equivalent_home_hours_per_day"
] = (
    df[
        "weekly_home_pct"
    ]
    / 100.0
    * 24.0
)

# 3. Mean daily home percentage
# Stored as fraction, so convert to percent
df[
    "mean_daily_home_pct"
] = (
    pd.to_numeric(
        df[
            "mean_daily_home_fraction"
        ],
        errors="coerce",
    )
    * 100.0
)


FEATURES = [
    {
        "column":
            "weekly_home_pct",

        "title":
            "Aggregate weekly time at home",

        "xlabel":
            "Time at home (%)",

        "kind":
            "percent",

        "letter":
            "A",
    },

    {
        "column":
            "equivalent_home_hours_per_day",

        "title":
            "Equivalent home hours/day",

        "xlabel":
            "Equivalent hours at home per day",

        "kind":
            "hours",

        "letter":
            "B",
    },

    {
        "column":
            "mean_daily_home_pct",

        "title":
            "Mean daily time at home",

        "xlabel":
            "Mean daily time at home (%)",

        "kind":
            "percent",

        "letter":
            "C",
    },
]


# =============================================================================
# STATS
# =============================================================================

stats_rows = []


for feature_index, spec in enumerate(
    FEATURES
):

    col = spec[
        "column"
    ]

    sub = df[
        [
            "participant_id",
            "phq_group",
            col,
        ]
    ].dropna().copy()

    lower = pd.to_numeric(
        sub.loc[
            sub[
                "phq_group"
            ].eq(
                "Lower"
            ),
            col,
        ],
        errors="coerce",
    ).dropna()

    elevated = pd.to_numeric(
        sub.loc[
            sub[
                "phq_group"
            ].eq(
                "Elevated"
            ),
            col,
        ],
        errors="coerce",
    ).dropna()

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

    (
        lower_boot_ci,
        elevated_boot_ci,
        difference_boot_ci,
    ) = cluster_bootstrap_group_means(
        data=sub,
        value_col=col,
        n_boot=N_BOOT,
        seed=(
            RANDOM_SEED
            + feature_index
        ),
    )

    inference = clustered_ols(
        data=sub,
        value_col=col,
    )

    stats_rows.append(
        {
            "feature":
                spec[
                    "title"
                ],

            "n_rows":
                len(sub),

            "n_participants":
                sub[
                    "participant_id"
                ].nunique(),

            "lower_mean":
                lower_mean,

            "lower_ci_low":
                lower_boot_ci[0],

            "lower_ci_high":
                lower_boot_ci[1],

            "elevated_mean":
                elevated_mean,

            "elevated_ci_low":
                elevated_boot_ci[0],

            "elevated_ci_high":
                elevated_boot_ci[1],

            "difference":
                difference,

            "difference_ci_low":
                difference_boot_ci[0],

            "difference_ci_high":
                difference_boot_ci[1],

            "clustered_beta":
                inference[
                    "beta"
                ],

            "clustered_ci_low":
                inference[
                    "ci_low"
                ],

            "clustered_ci_high":
                inference[
                    "ci_high"
                ],

            "p_value":
                inference[
                    "p_value"
                ],
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
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    stats_df[
        [
            "feature",
            "n_rows",
            "n_participants",
            "lower_mean",
            "elevated_mean",
            "difference",
            "difference_ci_low",
            "difference_ci_high",
            "p_value",
        ]
    ]
    .to_string(
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
    "Home-time patterns by depression symptom level",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)


rng = np.random.default_rng(
    RANDOM_SEED
)


for ax, spec in zip(
    axes,
    FEATURES,
):

    col = spec[
        "column"
    ]

    sub = df[
        [
            "participant_id",
            "phq_group",
            col,
        ]
    ].dropna().copy()

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
    # WEEKLY OBSERVATIONS
    # -------------------------------------------------------------------------

    for group in GROUP_ORDER:

        group_data = (
            sub[
                sub[
                    "phq_group"
                ].eq(
                    group
                )
            ]
        )

        jitter = rng.normal(
            0,
            0.030,
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
            s=23,
            alpha=0.30,
            color=(
                GROUP_COLORS[
                    group
                ]
            ),
            edgecolor="none",
            zorder=2,
        )


    # -------------------------------------------------------------------------
    # GROUP MEANS + CLUSTER BOOTSTRAP CI
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
    # SUMMARY TEXT
    # -------------------------------------------------------------------------

    difference = stat[
        "difference"
    ]

    p_value = stat[
        "p_value"
    ]


    if spec[
        "kind"
    ] == "percent":

        summary_text = (
            f"{lower_mean:.1f}%  →  "
            f"{elevated_mean:.1f}%"
            f"      Δ {difference:+.1f} pp"
            f"      {format_p(p_value)}"
        )


    else:

        summary_text = (
            f"{lower_mean:.1f}  →  "
            f"{elevated_mean:.1f} h/day"
            f"      Δ {difference:+.1f} h/day"
            f"      {format_p(p_value)}"
        )


    ax.text(
        0.5,
        1.095,
        summary_text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color="#173650",
    )


    # -------------------------------------------------------------------------
    # LABELS
    # -------------------------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # STYLE
    # -------------------------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # AXIS LIMITS
    # -------------------------------------------------------------------------

    if spec[
        "kind"
    ] == "percent":

        ax.set_xlim(
            0,
            100,
        )

    else:

        ax.set_xlim(
            0,
            24,
        )


plt.tight_layout(
    rect=[
        0.075,
        0.04,
        1,
        0.94,
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
print("=" * 100)
print("SAVED")
print("=" * 100)

print(PNG_PATH)
print(PDF_PATH)
print(CSV_PATH)
