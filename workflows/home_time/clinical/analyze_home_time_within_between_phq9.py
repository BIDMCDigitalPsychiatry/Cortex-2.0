from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time"
)

INPUT = (
    ROOT
    / "phq_temporal_windows"
    / "phq8_phq9_home_time_all_assessments.csv"
)

OUT = ROOT / "within_person_clinical_alignment"
OUT.mkdir(parents=True, exist_ok=True)

RESULTS_OUT = OUT / "phq9_home_time_within_between_results.csv"
DATA_OUT = OUT / "phq9_home_time_within_between_dataset.csv"
FIG_OUT = OUT / "phq9_home_time_within_person_windows.png"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT, low_memory=False)

df["assessment_date"] = pd.to_datetime(
    df["assessment_date"],
    errors="coerce"
)

df["phq9_total"] = pd.to_numeric(
    df["phq9_total"],
    errors="coerce"
)

WINDOWS = {
    "3-day": "home_3d_pct",
    "7-day": "home_7d_pct",
    "14-day": "home_14d_all_pct",
}

for col in WINDOWS.values():
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# TIME SINCE FIRST ASSESSMENT
# ============================================================

first_dates = (
    df.groupby("participant_id")["assessment_date"]
    .transform("min")
)

df["study_time_days"] = (
    df["assessment_date"] - first_dates
).dt.days


# ============================================================
# ANALYSIS
# ============================================================

all_results = []
analysis_frames = []

for label, home_col in WINDOWS.items():

    x = df[
        [
            "participant_id",
            "assessment_date",
            "study_time_days",
            "phq9_total",
            home_col,
        ]
    ].dropna().copy()

    # --------------------------------------------------------
    # Need repeated observations to estimate within-person
    # change.
    # --------------------------------------------------------

    counts = (
        x.groupby("participant_id")
        .size()
    )

    repeated_ids = counts[
        counts >= 2
    ].index

    x = x[
        x["participant_id"].isin(repeated_ids)
    ].copy()

    if x.empty:
        print(f"{label}: no usable repeated data.")
        continue

    # --------------------------------------------------------
    # Within/between decomposition
    #
    # between = participant's typical home time
    # within  = current home time - personal typical home time
    # --------------------------------------------------------

    x["home_person_mean"] = (
        x.groupby("participant_id")[home_col]
        .transform("mean")
    )

    x["home_within"] = (
        x[home_col]
        - x["home_person_mean"]
    )

    # Scale effects to 10 percentage-point home-time change
    x["home_within_10"] = x["home_within"] / 10.0
    x["home_between_10"] = x["home_person_mean"] / 10.0

    # Time per 30 days for readable coefficient
    x["study_time_30d"] = (
        x["study_time_days"] / 30.0
    )

    x["window"] = label
    analysis_frames.append(x)

    n_obs = len(x)
    n_participants = x["participant_id"].nunique()
    median_obs = (
        x.groupby("participant_id")
        .size()
        .median()
    )

    # ========================================================
    # PRIMARY: GEE
    # ========================================================

    gee = smf.gee(
        formula=(
            "phq9_total ~ "
            "home_within_10 + "
            "home_between_10 + "
            "study_time_30d"
        ),
        groups="participant_id",
        data=x,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    ).fit()

    for term, interpretation in [
        (
            "home_within_10",
            "Within-person: +10 percentage points home time"
        ),
        (
            "home_between_10",
            "Between-person: +10 percentage points typical home time"
        ),
    ]:

        beta = float(gee.params[term])
        se = float(gee.bse[term])
        p = float(gee.pvalues[term])

        ci_low = beta - 1.96 * se
        ci_high = beta + 1.96 * se

        all_results.append(
            {
                "window": label,
                "model": "GEE",
                "effect": interpretation,
                "term": term,
                "beta_phq9_points": beta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p,
                "n_assessments": n_obs,
                "n_participants": n_participants,
                "median_assessments_per_participant": median_obs,
            }
        )

    # ========================================================
    # SENSITIVITY: RANDOM-INTERCEPT MIXED MODEL
    # ========================================================

    try:
        mixed = smf.mixedlm(
            formula=(
                "phq9_total ~ "
                "home_within_10 + "
                "home_between_10 + "
                "study_time_30d"
            ),
            data=x,
            groups=x["participant_id"],
        ).fit(
            reml=False,
            method="lbfgs",
            maxiter=1000,
            disp=False,
        )

        for term, interpretation in [
            (
                "home_within_10",
                "Within-person: +10 percentage points home time"
            ),
            (
                "home_between_10",
                "Between-person: +10 percentage points typical home time"
            ),
        ]:

            beta = float(mixed.params[term])
            se = float(mixed.bse[term])
            p = float(mixed.pvalues[term])

            ci_low = beta - 1.96 * se
            ci_high = beta + 1.96 * se

            all_results.append(
                {
                    "window": label,
                    "model": "MixedLM",
                    "effect": interpretation,
                    "term": term,
                    "beta_phq9_points": beta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p,
                    "n_assessments": n_obs,
                    "n_participants": n_participants,
                    "median_assessments_per_participant": median_obs,
                }
            )

    except Exception as e:
        print(
            f"{label}: MixedLM failed: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame(all_results)

results.to_csv(
    RESULTS_OUT,
    index=False
)

if analysis_frames:
    pd.concat(
        analysis_frames,
        ignore_index=True
    ).to_csv(
        DATA_OUT,
        index=False
    )


# ============================================================
# PRINT PRIMARY RESULTS
# ============================================================

primary = results[
    (results["model"] == "GEE")
    &
    (results["term"] == "home_within_10")
].copy()

print()
print("=" * 110)
print("WITHIN-PERSON HOME TIME -> PHQ-9")
print("=" * 110)

if not primary.empty:

    display_cols = [
        "window",
        "beta_phq9_points",
        "ci_low",
        "ci_high",
        "p_value",
        "n_assessments",
        "n_participants",
        "median_assessments_per_participant",
    ]

    print(
        primary[display_cols].to_string(
            index=False,
            float_format=lambda z: f"{z:.4f}",
        )
    )

print()
print(
    "Interpretation: beta = PHQ-9 point change associated with "
    "spending 10 percentage points more time at home than that "
    "participant's own typical level."
)


# ============================================================
# FIGURE: WITHIN-PERSON EFFECT ACROSS WINDOWS
# ============================================================

if not primary.empty:

    order = ["3-day", "7-day", "14-day"]

    primary["window"] = pd.Categorical(
        primary["window"],
        categories=order,
        ordered=True,
    )

    primary = primary.sort_values("window")

    y = np.arange(len(primary))

    beta = primary["beta_phq9_points"].to_numpy()
    low = primary["ci_low"].to_numpy()
    high = primary["ci_high"].to_numpy()

    xerr = np.vstack([
        beta - low,
        high - beta,
    ])

    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(
        figsize=(7.2, 4.5)
    )

    ax.errorbar(
        beta,
        y,
        xerr=xerr,
        fmt="o",
        capsize=4,
        linewidth=1.7,
        markersize=7,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
        color="0.5",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(primary["window"])

    ax.set_xlabel(
        "PHQ-9 change per +10 percentage points\n"
        "of home time above personal usual level"
    )

    ax.set_ylabel(
        "Home-time window before PHQ-9"
    )

    ax.set_title(
        "Within-person association between home time and PHQ-9"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for yi, row in enumerate(
        primary.itertuples()
    ):
        ax.text(
            row.ci_high + 0.04,
            yi,
            f"p={row.p_value:.3f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        FIG_OUT,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


print()
print("Saved:")
print(RESULTS_OUT)
print(DATA_OUT)
print(FIG_OUT)
