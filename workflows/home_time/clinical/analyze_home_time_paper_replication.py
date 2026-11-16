from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian


# ============================================================
# PATHS
# ============================================================

DC = Path("/shared/Digital Clinic")

TEMPORAL_DIR = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "home_time/phq_temporal_windows"
)

PHQ_HOME_FILE = (
    TEMPORAL_DIR
    / "phq8_phq9_home_time_strict_14d_cohort.csv"
)

DEMOG_ROOT = (
    DC / "demographics"
)

CROSSWALK = (
    DEMOG_ROOT
    / "[deprecated] pt_start_dates.xlsx"
)

OUT = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "home_time/paper_replication"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# NORMALIZATION
# ============================================================

def norm_text(x):

    if pd.isna(x):
        return None

    s = unicodedata.normalize(
        "NFKD",
        str(x),
    )

    s = (
        s.encode(
            "ascii",
            "ignore",
        )
        .decode()
        .lower()
        .strip()
    )

    s = re.sub(
        r"[^a-z0-9]+",
        " ",
        s,
    )

    s = re.sub(
        r"\s+",
        " ",
        s,
    ).strip()

    return s or None


def norm_col(x):
    return norm_text(x) or ""


def find_name_col(cols):

    preferred = [
        "full name",
        "first and last",
        "first last",
    ]

    for term in preferred:

        for c in cols:

            if term in norm_col(c):
                return c

    for c in cols:

        if norm_col(c) == "name":
            return c

    return None


def find_age_col(cols):

    for c in cols:

        n = norm_col(c)

        if (
            "age" in n
            and "enrollment" in n
        ):
            return c

    return None


# ============================================================
# BUILD CLEAN AGE TABLE
# ============================================================

records = []

for year in [
    2023,
    2024,
    2025,
    2026,
]:

    files = sorted(
        [
            p
            for p in DEMOG_ROOT.iterdir()
            if str(year) in p.name
            and p.suffix.lower()
            in {
                ".xlsx",
                ".xls",
                ".csv",
            }
        ]
    )

    for path in files:

        if path.name.startswith(
            "[deprecated]"
        ):
            continue

        datasets = []

        if path.suffix.lower() in {
            ".xlsx",
            ".xls",
        }:

            xl = pd.ExcelFile(
                path
            )

            for sheet in xl.sheet_names:

                try:
                    d = pd.read_excel(
                        path,
                        sheet_name=sheet,
                    )

                    datasets.append(
                        d
                    )

                except Exception:
                    pass

        else:

            for enc in [
                "utf-8",
                "utf-8-sig",
                "cp1252",
                "latin1",
            ]:

                try:

                    d = pd.read_csv(
                        path,
                        encoding=enc,
                        low_memory=False,
                    )

                    datasets.append(
                        d
                    )

                    break

                except UnicodeDecodeError:
                    continue

                except Exception:
                    continue

        for d in datasets:

            name_col = find_name_col(
                d.columns
            )

            age_col = find_age_col(
                d.columns
            )

            if (
                name_col is None
                or age_col is None
            ):
                continue

            tmp = pd.DataFrame(
                {
                    "name_norm":
                        d[name_col].apply(
                            norm_text
                        ),

                    "age":
                        pd.to_numeric(
                            d[age_col],
                            errors="coerce",
                        ),

                    "source_year":
                        year,
                }
            )

            tmp = tmp[
                tmp["name_norm"].notna()
                & tmp["age"].between(
                    0,
                    120,
                )
            ]

            records.append(
                tmp
            )


age_raw = pd.concat(
    records,
    ignore_index=True,
)


# Exclude names with inconsistent age values.
age_consistency = (
    age_raw.groupby(
        "name_norm"
    )["age"]
    .nunique()
)

clean_age_names = set(
    age_consistency.index[
        age_consistency.eq(1)
    ]
)

age_clean = (
    age_raw[
        age_raw[
            "name_norm"
        ].isin(
            clean_age_names
        )
    ]
    .groupby(
        "name_norm",
        as_index=False,
    )
    .agg(
        age=("age", "first"),
        source_year=(
            "source_year",
            "min",
        ),
    )
)


# ============================================================
# NAME -> CLINIC ID CROSSWALK
# ============================================================

cw = pd.read_excel(
    CROSSWALK,
    sheet_name="6.10pt_start_dates",
)

cw = cw[
    [
        "Name",
        "Clinic ID",
    ]
].copy()

cw["name_norm"] = (
    cw["Name"].apply(
        norm_text
    )
)

cw["participant_id"] = (
    cw["Clinic ID"]
    .astype(str)
    .str.strip()
)

cw = cw[
    cw["name_norm"].notna()
    & cw["participant_id"].notna()
    & cw["participant_id"].ne("")
    & cw["participant_id"].ne("nan")
].copy()

name_id_counts = (
    cw.groupby(
        "name_norm"
    )["participant_id"]
    .nunique()
)

unique_names = set(
    name_id_counts.index[
        name_id_counts.eq(1)
    ]
)

cw_clean = (
    cw[
        cw["name_norm"].isin(
            unique_names
        )
    ]
    .drop_duplicates(
        subset=[
            "name_norm"
        ]
    )
    [
        [
            "name_norm",
            "participant_id",
        ]
    ]
)

age_map = (
    age_clean.merge(
        cw_clean,
        on="name_norm",
        how="inner",
    )
    [
        [
            "participant_id",
            "age",
            "source_year",
        ]
    ]
    .drop_duplicates(
        subset=[
            "participant_id"
        ]
    )
)


# ============================================================
# LOAD STRICT 14-DAY PHQ/HOME DATA
# ============================================================

df = pd.read_csv(
    PHQ_HOME_FILE,
    low_memory=False,
)

required = {
    "participant_id",
    "assessment_date",
    "phq8_total",
    "phq9_total",
    "home_14d_all_pct",
    "home_14d_weekday_matched_pct",
    "home_14d_weekend_pct",
}

missing = (
    required
    - set(df.columns)
)

if missing:
    raise RuntimeError(
        f"Missing expected columns: {missing}"
    )

df["assessment_date"] = (
    pd.to_datetime(
        df["assessment_date"],
        errors="coerce",
    )
)

df = df.merge(
    age_map,
    on="participant_id",
    how="left",
)

# Save deidentified analysis table.
df[
    [
        "participant_id",
        "assessment_date",
        "age",
        "phq8_total",
        "phq9_total",
        "home_14d_all_pct",
        "home_14d_weekday_matched_pct",
        "home_14d_weekend_pct",
    ]
].to_csv(
    OUT
    / "paper_replication_analysis_dataset.csv",
    index=False,
)


# ============================================================
# COHORT SUMMARY
# ============================================================

n_segments = len(df)
n_participants = (
    df["participant_id"]
    .nunique()
)

age_participants = (
    df[
        [
            "participant_id",
            "age",
        ]
    ]
    .dropna()
    .drop_duplicates(
        "participant_id"
    )
)

print("=" * 100)
print("PAPER-STYLE HOME-TIME REPLICATION")
print("=" * 100)

print(
    "14-day segments:",
    n_segments,
)

print(
    "Participants:",
    n_participants,
)

print(
    "Participants with age:",
    len(age_participants),
)

print(
    "Age coverage:",
    f"{100 * len(age_participants) / n_participants:.1f}%"
)

print(
    "Age mean:",
    round(
        age_participants[
            "age"
        ].mean(),
        1,
    )
)

print(
    "Age median:",
    round(
        age_participants[
            "age"
        ].median(),
        1,
    )
)

print(
    "Age range:",
    (
        age_participants[
            "age"
        ].min(),
        age_participants[
            "age"
        ].max(),
    ),
)

print(
    "PHQ-8 mean:",
    round(
        df[
            "phq8_total"
        ].mean(),
        2,
    )
)

print(
    "PHQ-8 range:",
    (
        df[
            "phq8_total"
        ].min(),
        df[
            "phq8_total"
        ].max(),
    ),
)


# ============================================================
# FIGURE 1 — COHORT CHARACTERISTICS
# ============================================================

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        12.5,
        4.2,
    ),
)

# ------------------------------------------------------------
# A: segments per participant
# ------------------------------------------------------------

segment_counts = (
    df.groupby(
        "participant_id"
    )
    .size()
)

max_segments = int(
    segment_counts.max()
)

bins_a = np.arange(
    0.5,
    max_segments + 1.5,
    1,
)

axes[0].hist(
    segment_counts,
    bins=bins_a,
    edgecolor="black",
    linewidth=0.6,
    alpha=0.75,
)

axes[0].set_xlabel(
    "Number of 14-day segments per participant"
)

axes[0].set_ylabel(
    "Number of participants"
)

axes[0].text(
    -0.13,
    1.03,
    "(A)",
    transform=axes[0].transAxes,
    fontweight="bold",
)


# ------------------------------------------------------------
# B: PHQ-8 distribution
# ------------------------------------------------------------

phq8 = (
    pd.to_numeric(
        df["phq8_total"],
        errors="coerce",
    )
    .dropna()
)

bins_b = np.arange(
    -0.5,
    25.5,
    1,
)

axes[1].hist(
    phq8,
    bins=bins_b,
    edgecolor="black",
    linewidth=0.6,
    alpha=0.75,
)

axes[1].set_xlabel(
    "PHQ-8 total score"
)

axes[1].set_ylabel(
    "Number of questionnaires"
)

axes[1].set_xlim(
    -0.5,
    24.5,
)

axes[1].text(
    -0.13,
    1.03,
    "(B)",
    transform=axes[1].transAxes,
    fontweight="bold",
)


# ------------------------------------------------------------
# C: age distribution, one value per participant
# ------------------------------------------------------------

age_values = (
    age_participants[
        "age"
    ]
    .dropna()
)

age_min = (
    np.floor(
        age_values.min() / 5
    ) * 5
)

age_max = (
    np.ceil(
        age_values.max() / 5
    ) * 5
)

bins_c = np.arange(
    age_min,
    age_max + 5,
    5,
)

axes[2].hist(
    age_values,
    bins=bins_c,
    edgecolor="black",
    linewidth=0.6,
    alpha=0.75,
)

axes[2].set_xlabel(
    "Age at enrollment"
)

axes[2].set_ylabel(
    "Number of participants"
)

axes[2].text(
    -0.13,
    1.03,
    "(C)",
    transform=axes[2].transAxes,
    fontweight="bold",
)

fig.tight_layout(
    w_pad=2.8
)

fig.savefig(
    OUT
    / "cohort_characteristics_paper_style.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "cohort_characteristics_paper_style.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# VISUAL LINEAR FIT
# ============================================================

def add_scatter_fit(
    ax,
    x,
    y,
    xlabel,
    title,
    panel,
):

    d = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).dropna()

    ax.scatter(
        d["x"],
        d["y"],
        s=8,
        alpha=0.4,
        edgecolors="none",
    )

    if (
        len(d) >= 2
        and d["x"].nunique() >= 2
    ):

        coef = np.polyfit(
            d["x"],
            d["y"],
            1,
        )

        xx = np.linspace(
            d["x"].min(),
            d["x"].max(),
            200,
        )

        yy = (
            coef[0] * xx
            + coef[1]
        )

        ax.plot(
            xx,
            yy,
            linewidth=1.5,
        )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        "Home time (%)"
    )

    ax.set_title(
        title,
        fontsize=11,
        pad=7,
    )

    ax.text(
        -0.16,
        1.03,
        panel,
        transform=ax.transAxes,
        fontweight="bold",
    )


# ============================================================
# FIGURE 2 — PAPER-STYLE 2 × 3 ASSOCIATION FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        12.5,
        7.2,
    ),
)

home_cols = [
    (
        "home_14d_all_pct",
        "Entire 14-day period",
    ),
    (
        "home_14d_weekday_matched_pct",
        "Weekdays only",
    ),
    (
        "home_14d_weekend_pct",
        "Weekends only",
    ),
]

# Age row
for j, (
    home_col,
    title,
) in enumerate(home_cols):

    add_scatter_fit(
        axes[0, j],
        df["age"],
        df[home_col],
        "Age at enrollment",
        title,
        f"({chr(65 + j)})",
    )

# PHQ-8 row
for j, (
    home_col,
    title,
) in enumerate(home_cols):

    add_scatter_fit(
        axes[1, j],
        df["phq8_total"],
        df[home_col],
        "PHQ-8 total score",
        title,
        f"({chr(68 + j)})",
    )

fig.tight_layout(
    h_pad=2.4,
    w_pad=2.0,
)

fig.savefig(
    OUT
    / "home_time_age_phq8_paper_style.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "home_time_age_phq8_paper_style.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================

# ============================================================
# FIGURE 2B — PHQ-9 LOW VS HIGH ONLY
# ============================================================

phq9_low_high = df[
    (
        df["phq9_total"].between(
            0,
            4,
            inclusive="both",
        )
    )
    |
    (
        df["phq9_total"] >= 10
    )
].copy()

phq9_low_high["phq_group"] = np.where(
    phq9_low_high["phq9_total"] <= 4,
    "Low\nPHQ-9 0–4",
    "High\nPHQ-9 ≥10",
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(11.8, 4.1),
    sharey=True,
)

rng = np.random.default_rng(42)

for j, (home_col, title) in enumerate(home_cols):

    ax = axes[j]

    plot_df = phq9_low_high[
        [
            "phq_group",
            home_col,
        ]
    ].dropna().copy()

    groups = [
        "Low\nPHQ-9 0–4",
        "High\nPHQ-9 ≥10",
    ]

    for x_pos, group in enumerate(groups):

        vals = (
            plot_df.loc[
                plot_df["phq_group"].eq(group),
                home_col,
            ]
            .astype(float)
            .to_numpy()
        )

        # jittered individual observations
        jitter = rng.normal(
            loc=0,
            scale=0.055,
            size=len(vals),
        )

        ax.scatter(
            np.full(len(vals), x_pos) + jitter,
            vals,
            s=18,
            alpha=0.45,
            edgecolors="none",
            zorder=2,
        )

        # mean + 95% CI
        mean = np.mean(vals)

        if len(vals) > 1:
            se = np.std(
                vals,
                ddof=1,
            ) / np.sqrt(len(vals))

            ci = 1.96 * se
        else:
            ci = np.nan

        ax.errorbar(
            x_pos,
            mean,
            yerr=ci,
            fmt="o",
            markersize=7,
            capsize=4,
            linewidth=1.5,
            zorder=4,
        )

        ax.text(
            x_pos,
            3,
            f"n={len(vals)}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        groups
    )

    ax.set_xlim(
        -0.45,
        1.45,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_title(
        title,
        fontsize=11,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        -0.14,
        1.03,
        f"({chr(65+j)})",
        transform=ax.transAxes,
        fontweight="bold",
    )

axes[0].set_ylabel(
    "Home time (%)"
)

fig.suptitle(
    "Home time in low vs high PHQ-9 groups",
    fontsize=13,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.93],
    w_pad=2.0,
)

fig.savefig(
    OUT / "home_time_phq9_low_high_groups.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT / "home_time_phq9_low_high_groups.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# PARTICIPANT-CLUSTERED GEE
# ============================================================

def fit_gee(
    data,
    y_col,
    predictors,
    label,
):

    cols = (
        [
            "participant_id",
            y_col,
        ]
        + predictors
    )

    d = (
        data[
            cols
        ]
        .dropna()
        .copy()
    )

    result_rows = []

    if (
        len(d) == 0
        or d[
            "participant_id"
        ].nunique() < 2
    ):
        return result_rows

    X = sm.add_constant(
        d[predictors],
        has_constant="add",
    )

    model = sm.GEE(
        endog=d[y_col],
        exog=X,
        groups=d[
            "participant_id"
        ],
        family=Gaussian(),
        cov_struct=Exchangeable(),
    )

    fit = model.fit(
        maxiter=200
    )

    ci = fit.conf_int()

    for predictor in predictors:

        result_rows.append(
            {
                "model":
                    label,

                "outcome":
                    y_col,

                "predictor":
                    predictor,

                "n_segments":
                    len(d),

                "n_participants":
                    d[
                        "participant_id"
                    ].nunique(),

                "beta":
                    fit.params[
                        predictor
                    ],

                "ci_low":
                    ci.loc[
                        predictor
                    ].iloc[0],

                "ci_high":
                    ci.loc[
                        predictor
                    ].iloc[1],

                "p_value":
                    fit.pvalues[
                        predictor
                    ],
            }
        )

    return result_rows


stats = []

for home_col, label in home_cols:

    # Unadjusted age
    stats += fit_gee(
        df,
        home_col,
        [
            "age",
        ],
        f"{label}: age unadjusted",
    )

    # Unadjusted PHQ-8
    stats += fit_gee(
        df,
        home_col,
        [
            "phq8_total",
        ],
        f"{label}: PHQ-8 unadjusted",
    )

    # Age + PHQ-8 together
    stats += fit_gee(
        df,
        home_col,
        [
            "age",
            "phq8_total",
        ],
        f"{label}: age + PHQ-8",
    )

    # PHQ-9 sensitivity
    stats += fit_gee(
        df,
        home_col,
        [
            "age",
            "phq9_total",
        ],
        f"{label}: age + PHQ-9 sensitivity",
    )


stats = pd.DataFrame(
    stats
)

stats.to_csv(
    OUT
    / "paper_style_clustered_gee_stats.csv",
    index=False,
)


# ============================================================
# DESCRIPTIVE HOME-TIME SUMMARY
# ============================================================

desc = []

for home_col, label in home_cols:

    x = (
        pd.to_numeric(
            df[home_col],
            errors="coerce",
        )
        .dropna()
    )

    desc.append(
        {
            "time_frame":
                label,

            "n_segments":
                len(x),

            "n_participants":
                df.loc[
                    df[
                        home_col
                    ].notna(),
                    "participant_id",
                ]
                .nunique(),

            "mean_home_pct":
                x.mean(),

            "median_home_pct":
                x.median(),

            "q25_home_pct":
                x.quantile(
                    0.25
                ),

            "q75_home_pct":
                x.quantile(
                    0.75
                ),
        }
    )

desc = pd.DataFrame(
    desc
)

desc.to_csv(
    OUT
    / "paper_style_home_time_descriptives.csv",
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 100)
print("HOME-TIME DESCRIPTIVES")
print("=" * 100)

print(
    desc.round(2)
    .to_string(
        index=False
    )
)

print()
print("=" * 100)
print("CLUSTERED GEE RESULTS")
print("=" * 100)

show = stats.copy()

for c in [
    "beta",
    "ci_low",
    "ci_high",
]:
    show[c] = (
        show[c]
        .round(3)
    )

show[
    "p_value"
] = (
    show[
        "p_value"
    ]
    .map(
        lambda x:
            f"{x:.4g}"
            if pd.notna(x)
            else "NA"
    )
)

print(
    show.to_string(
        index=False
    )
)

print()
print("Saved outputs:")

for path in sorted(
    OUT.iterdir()
):
    print(
        " ",
        path
    )

