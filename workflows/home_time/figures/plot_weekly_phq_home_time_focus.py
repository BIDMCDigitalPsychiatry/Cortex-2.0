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

PNG_PATH = OUTPUT_DIR / "weekly_phq_home_time_focus.png"
PDF_PATH = OUTPUT_DIR / "weekly_phq_home_time_focus.pdf"
CSV_PATH = OUTPUT_DIR / "weekly_phq_home_time_focus_stats.csv"


# =============================================================================
# SETTINGS
# =============================================================================

N_BOOT = 5000
RANDOM_SEED = 20261102

COLORS = {
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

    for _ in range(n_boot):

        sampled_ids = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        pieces = []

        for boot_index, pid in enumerate(sampled_ids):

            piece = participant_frames[pid].copy()

            piece["_boot_cluster"] = (
                f"{pid}_{boot_index}"
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

        lower_boot.append(
            float(
                lower.mean()
            )
        )

        elevated_boot.append(
            float(
                elevated.mean()
            )
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

    result = sm.OLS(
        y,
        X,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                x[
                    "participant_id"
                ]
        },
    )

    return {
        "beta":
            float(
                result.params[
                    "elevated"
                ]
            ),

        "p_value":
            float(
                result.pvalues[
                    "elevated"
                ]
            ),
    }


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_parquet(
    INPUT_PATH
).copy()

print("=" * 100)
print("FOCUSED TIME-AT-HOME FIGURE")
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
# HOME-TIME FEATURES
# =============================================================================

df[
    "weekly_home_pct"
] = pd.to_numeric(
    df[
        "time_at_home_pct"
    ],
    errors="coerce",
)

df[
    "equivalent_home_hours"
] = (
    df[
        "weekly_home_pct"
    ]
    / 100.0
    * 24.0
)

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


# =============================================================================
# MAIN ANALYSIS: EQUIVALENT HOURS/DAY
# =============================================================================

main = df[
    [
        "participant_id",
        "phq_group",
        "equivalent_home_hours",
    ]
].dropna().copy()

lower = pd.to_numeric(
    main.loc[
        main[
            "phq_group"
        ].eq(
            "Lower"
        ),
        "equivalent_home_hours",
    ],
    errors="coerce",
).dropna()

elevated = pd.to_numeric(
    main.loc[
        main[
            "phq_group"
        ].eq(
            "Elevated"
        ),
        "equivalent_home_hours",
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
    lower_ci,
    elevated_ci,
) = cluster_bootstrap_group_means(
    data=main,
    value_col="equivalent_home_hours",
    n_boot=N_BOOT,
    seed=RANDOM_SEED,
)

main_inference = clustered_ols(
    data=main,
    value_col="equivalent_home_hours",
)


# =============================================================================
# ROBUSTNESS: MEAN DAILY HOME PERCENT
# =============================================================================

robust = df[
    [
        "participant_id",
        "phq_group",
        "mean_daily_home_pct",
    ]
].dropna().copy()

robust_lower = pd.to_numeric(
    robust.loc[
        robust[
            "phq_group"
        ].eq(
            "Lower"
        ),
        "mean_daily_home_pct",
    ],
    errors="coerce",
).dropna()

robust_elevated = pd.to_numeric(
    robust.loc[
        robust[
            "phq_group"
        ].eq(
            "Elevated"
        ),
        "mean_daily_home_pct",
    ],
    errors="coerce",
).dropna()

robust_lower_mean = float(
    robust_lower.mean()
)

robust_elevated_mean = float(
    robust_elevated.mean()
)

robust_difference = (
    robust_elevated_mean
    - robust_lower_mean
)

robust_inference = clustered_ols(
    data=robust,
    value_col="mean_daily_home_pct",
)


# =============================================================================
# SAVE STATS
# =============================================================================

stats = pd.DataFrame(
    [
        {
            "measure":
                "Equivalent home hours/day",

            "n_rows":
                len(main),

            "n_participants":
                main[
                    "participant_id"
                ].nunique(),

            "lower_mean":
                lower_mean,

            "elevated_mean":
                elevated_mean,

            "difference":
                difference,

            "p_value":
                main_inference[
                    "p_value"
                ],
        },

        {
            "measure":
                "Mean daily home percentage",

            "n_rows":
                len(robust),

            "n_participants":
                robust[
                    "participant_id"
                ].nunique(),

            "lower_mean":
                robust_lower_mean,

            "elevated_mean":
                robust_elevated_mean,

            "difference":
                robust_difference,

            "p_value":
                robust_inference[
                    "p_value"
                ],
        },
    ]
)

stats.to_csv(
    CSV_PATH,
    index=False,
)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    stats.to_string(
        index=False
    )
)


# =============================================================================
# FIGURE
# =============================================================================

plt.rcParams.update(
    {
        "font.size":
            13,

        "axes.titlesize":
            20,

        "axes.labelsize":
            14,

        "xtick.labelsize":
            12,

        "ytick.labelsize":
            14,
    }
)


fig, ax = plt.subplots(
    figsize=(
        11,
        5.8,
    )
)


fig.suptitle(
    "Time at home by depression symptom level",
    fontsize=22,
    fontweight="bold",
    y=0.98,
)


rng = np.random.default_rng(
    RANDOM_SEED
)


y_positions = {
    "Lower":
        1,

    "Elevated":
        0,
}


# -----------------------------------------------------------------------------
# WEEKLY OBSERVATIONS
# -----------------------------------------------------------------------------

for group in [
    "Lower",
    "Elevated",
]:

    group_data = (
        main[
            main[
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
            "equivalent_home_hours"
        ],
        (
            y_positions[
                group
            ]
            + jitter
        ),
        s=28,
        alpha=0.28,
        color=COLORS[
            group
        ],
        edgecolor="none",
        zorder=2,
    )


# -----------------------------------------------------------------------------
# GROUP MEANS + CLUSTER BOOTSTRAP CI
# -----------------------------------------------------------------------------

ax.errorbar(
    lower_mean,
    1,
    xerr=[
        [
            lower_mean
            - lower_ci[0]
        ],
        [
            lower_ci[1]
            - lower_mean
        ],
    ],
    fmt="o",
    markersize=12,
    color="#173650",
    ecolor="#173650",
    elinewidth=3.5,
    capsize=7,
    capthick=3,
    zorder=5,
)


ax.errorbar(
    elevated_mean,
    0,
    xerr=[
        [
            elevated_mean
            - elevated_ci[0]
        ],
        [
            elevated_ci[1]
            - elevated_mean
        ],
    ],
    fmt="o",
    markersize=12,
    color="#173650",
    ecolor="#173650",
    elinewidth=3.5,
    capsize=7,
    capthick=3,
    zorder=5,
)


# -----------------------------------------------------------------------------
# MAIN RESULT ANNOTATION
# -----------------------------------------------------------------------------

main_text = (
    f"{lower_mean:.1f} → "
    f"{elevated_mean:.1f} h/day"
    f"      Δ {difference:+.1f} h/day"
    f"      {format_p(main_inference['p_value'])}"
)


ax.text(
    0.5,
    1.10,
    main_text,
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=14,
    fontweight="bold",
    color="#173650",
)


# -----------------------------------------------------------------------------
# ROBUSTNESS NOTE
# -----------------------------------------------------------------------------

robust_text = (
    "Alternative daily aggregation: "
    f"{robust_lower_mean:.1f}% → "
    f"{robust_elevated_mean:.1f}% "
    f"(Δ {robust_difference:+.1f} pp; "
    f"{format_p(robust_inference['p_value'])})"
)


ax.text(
    0.5,
    -0.24,
    robust_text,
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=11,
    color="#666666",
)


# -----------------------------------------------------------------------------
# LABELS
# -----------------------------------------------------------------------------

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
    "Equivalent hours at home per day"
)

ax.set_xlim(
    0,
    24,
)


# -----------------------------------------------------------------------------
# STYLE
# -----------------------------------------------------------------------------

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


plt.tight_layout(
    rect=[
        0.06,
        0.12,
        1,
        0.90,
    ]
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
