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

PNG_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_clustered.png"
PDF_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_clustered.pdf"
CSV_PATH = OUTPUT_DIR / "weekly_phq_group_mobility_clustered_stats.csv"


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


FEATURES = [
    {
        "column": "mobility_diversity",
        "title": "Mobility diversity",
        "xlabel": "Weekly mean mobility diversity",
        "kind": "plain",
        "letter": "A",
    },
    {
        "column": "time_at_home",
        "title": "Time at home",
        "xlabel": "Weekly mean time at home (%)",
        "kind": "percent",
        "letter": "B",
    },
    {
        "column": "movement_distance",
        "title": "Daily movement distance",
        "xlabel": "Weekly mean movement distance (km/day)",
        "kind": "km",
        "letter": "C",
    },
]


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
    """
    Participant-cluster bootstrap.

    Resamples participants with replacement and keeps all eligible weekly
    observations belonging to each sampled participant.

    Returns bootstrap 95% CIs for:
      - Lower group mean
      - Elevated group mean
      - Elevated - Lower difference
    """

    rng = np.random.default_rng(seed)

    participants = (
        data["participant_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    participant_frames = {
        pid: data.loc[
            data["participant_id"].astype(str).eq(pid)
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

            piece = participant_frames[pid].copy()

            # Give duplicated resampled participants unique cluster IDs.
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

    lower_ci = ci(
        lower_boot
    )

    elevated_ci = ci(
        elevated_boot
    )

    difference_ci = ci(
        difference_boot
    )

    return (
        lower_ci,
        elevated_ci,
        difference_ci,
    )


def clustered_ols(
    data,
    value_col,
):
    """
    OLS:
        feature ~ elevated

    with participant-clustered standard errors.

    Because elevated is binary:
        intercept = Lower mean
        elevated coefficient = Elevated mean - Lower mean
    """

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

    ci = result.conf_int().loc[
        "elevated"
    ]

    return {
        "clustered_beta":
            beta,

        "clustered_ci_low":
            float(
                ci.iloc[0]
            ),

        "clustered_ci_high":
            float(
                ci.iloc[1]
            ),

        "clustered_p_value":
            p_value,
    }


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_parquet(
    INPUT_PATH
).copy()

print("=" * 100)
print("WEEKLY PHQ MOBILITY — CLUSTER-AWARE DESCRIPTIVE COMPARISON")
print("=" * 100)

print(
    "Input rows:",
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


print()
print("=" * 100)
print("PHQ GROUP COUNTS")
print("=" * 100)

print(
    df[
        "phq_group"
    ]
    .value_counts()
    .to_string()
)


# =============================================================================
# STATISTICS
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

            "n_lower_weeks":
                len(lower),

            "n_elevated_weeks":
                len(elevated),

            "lower_mean":
                lower_mean,

            "lower_cluster_boot_ci_low":
                lower_boot_ci[0],

            "lower_cluster_boot_ci_high":
                lower_boot_ci[1],

            "elevated_mean":
                elevated_mean,

            "elevated_cluster_boot_ci_low":
                elevated_boot_ci[0],

            "elevated_cluster_boot_ci_high":
                elevated_boot_ci[1],

            "difference_elevated_minus_lower":
                difference,

            "difference_cluster_boot_ci_low":
                difference_boot_ci[0],

            "difference_cluster_boot_ci_high":
                difference_boot_ci[1],

            "clustered_ols_beta":
                inference[
                    "clustered_beta"
                ],

            "clustered_ols_ci_low":
                inference[
                    "clustered_ci_low"
                ],

            "clustered_ols_ci_high":
                inference[
                    "clustered_ci_high"
                ],

            "clustered_ols_p_value":
                inference[
                    "clustered_p_value"
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
            "difference_elevated_minus_lower",
            "difference_cluster_boot_ci_low",
            "difference_cluster_boot_ci_high",
            "clustered_ols_p_value",
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
    "Mobility patterns by depression symptom level",
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


    # =========================================================================
    # WEEKLY OBSERVATIONS
    # =========================================================================

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


    # =========================================================================
    # POOLED WEEKLY MEANS + PARTICIPANT-CLUSTER BOOTSTRAP CIs
    # =========================================================================

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
                    "lower_cluster_boot_ci_low"
                ]
            ],
            [
                stat[
                    "lower_cluster_boot_ci_high"
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
                    "elevated_cluster_boot_ci_low"
                ]
            ],
            [
                stat[
                    "elevated_cluster_boot_ci_high"
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
    # SUMMARY LABEL
    # =========================================================================

    difference = stat[
        "difference_elevated_minus_lower"
    ]

    p_value = stat[
        "clustered_ols_p_value"
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


    elif spec[
        "kind"
    ] == "km":

        summary_text = (
            f"{lower_mean:.1f}  →  "
            f"{elevated_mean:.1f} km/day"
            f"      Δ {difference:+.1f} km/day"
            f"      {format_p(p_value)}"
        )


    else:

        summary_text = (
            f"{lower_mean:.2f}  →  "
            f"{elevated_mean:.2f}"
            f"      Δ {difference:+.2f}"
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

    values = pd.to_numeric(
        sub[
            col
        ],
        errors="coerce",
    ).dropna().to_numpy()


    if spec[
        "kind"
    ] == "percent":

        ax.set_xlim(
            0,
            100,
        )


    else:

        data_min = float(
            np.nanmin(
                values
            )
        )

        data_max = float(
            np.nanpercentile(
                values,
                99,
            )
        )


        span = (
            data_max
            - data_min
        )


        pad = (
            0.07
            * span
            if span > 0
            else 1
        )


        ax.set_xlim(
            max(
                0,
                data_min - pad,
            ),
            data_max + pad,
        )


        n_offscale = int(
            np.sum(
                values
                > (
                    data_max
                    + pad
                )
            )
        )


        if n_offscale > 0:

            ax.text(
                0.985,
                0.08,
                (
                    f"{n_offscale} weekly observation"
                    + (
                        "s"
                        if n_offscale != 1
                        else ""
                    )
                    + " off-scale →"
                ),
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=8.5,
                color="#777777",
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
