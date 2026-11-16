from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/daily_longitudinal"
)

DATA_FILE = ROOT / "home_t_function_t1_analysis_dataset.csv"
OUT_FILE = ROOT / "home_functioning_fixed_effects_results.csv"

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# ============================================================
# FORWARD MODEL
#
# home[t] -> functioning[t+1]
# ============================================================

forward_df = df.dropna(
    subset=[
        "participant_id",
        "functioning_t1",
        "functioning_dev_t",
        "home_dev_10",
        "weekend",
        "gps_hours_c",
        "study_day_10",
    ]
).copy()

forward = smf.ols(
    formula=(
        "functioning_t1 ~ "
        "home_dev_10 + "
        "functioning_dev_t + "
        "weekend + "
        "gps_hours_c + "
        "study_day_10 + "
        "C(participant_id)"
    ),
    data=forward_df
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": forward_df["participant_id"]
    }
)

# ============================================================
# FORWARD WITHOUT PRIOR FUNCTIONING
# ============================================================

forward_no_prior = smf.ols(
    formula=(
        "functioning_t1 ~ "
        "home_dev_10 + "
        "weekend + "
        "gps_hours_c + "
        "study_day_10 + "
        "C(participant_id)"
    ),
    data=forward_df
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": forward_df["participant_id"]
    }
)

# ============================================================
# BUILD REVERSE-DIRECTION DATA
#
# functioning[t] -> home[t+1]
# ============================================================

base = df[
    [
        "participant_id",
        "date",
        "home_pct",
        "gps_hours",
        "functioning_t",
    ]
].drop_duplicates(
    subset=[
        "participant_id",
        "date"
    ]
).copy()

# person means based on available analysis days
base["home_mean"] = (
    base.groupby("participant_id")["home_pct"]
    .transform("mean")
)

base["function_mean"] = (
    base.groupby("participant_id")["functioning_t"]
    .transform("mean")
)

base["home_dev"] = (
    base["home_pct"]
    - base["home_mean"]
)

base["function_dev"] = (
    base["functioning_t"]
    - base["function_mean"]
)

# Home at t+1
future_home = base[
    [
        "participant_id",
        "date",
        "home_pct",
    ]
].copy()

future_home["date"] = (
    future_home["date"]
    - pd.Timedelta(days=1)
)

future_home = future_home.rename(
    columns={
        "home_pct": "home_t1"
    }
)

rev = base.merge(
    future_home,
    on=[
        "participant_id",
        "date"
    ],
    how="inner"
)

rev["home_t1_mean"] = (
    rev.groupby("participant_id")["home_t1"]
    .transform("mean")
)

rev["home_t1_dev_10"] = (
    rev["home_t1"]
    - rev["home_t1_mean"]
) / 10

rev["function_dev"] = pd.to_numeric(
    rev["function_dev"],
    errors="coerce"
)

rev["weekend"] = (
    rev["date"].dt.dayofweek >= 5
).astype(int)

first = (
    rev.groupby("participant_id")["date"]
    .transform("min")
)

rev["study_day_10"] = (
    (rev["date"] - first).dt.days
    / 10
)

rev["gps_hours_c"] = (
    rev["gps_hours"]
    - rev["gps_hours"].mean()
)

rev = rev.dropna(
    subset=[
        "home_t1_dev_10",
        "function_dev",
        "participant_id",
        "study_day_10",
    ]
)

reverse = smf.ols(
    formula=(
        "home_t1_dev_10 ~ "
        "function_dev + "
        "weekend + "
        "gps_hours_c + "
        "study_day_10 + "
        "C(participant_id)"
    ),
    data=rev
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": rev["participant_id"]
    }
)

# ============================================================
# EXTRACT RESULTS
# ============================================================

def result_row(name, model, term, n, npart):

    beta = float(
        model.params[term]
    )

    se = float(
        model.bse[term]
    )

    return {
        "model": name,
        "term": term,
        "n_pairs": n,
        "n_participants": npart,
        "beta": beta,
        "se": se,
        "ci_low": beta - 1.96 * se,
        "ci_high": beta + 1.96 * se,
        "p_value": float(
            model.pvalues[term]
        ),
    }


results = pd.DataFrame(
    [
        result_row(
            "forward_with_prior_function",
            forward,
            "home_dev_10",
            len(forward_df),
            forward_df[
                "participant_id"
            ].nunique(),
        ),

        result_row(
            "forward_without_prior_function",
            forward_no_prior,
            "home_dev_10",
            len(forward_df),
            forward_df[
                "participant_id"
            ].nunique(),
        ),

        result_row(
            "reverse_function_to_nextday_home",
            reverse,
            "function_dev",
            len(rev),
            rev[
                "participant_id"
            ].nunique(),
        ),
    ]
)

results.to_csv(
    OUT_FILE,
    index=False
)

print()
print("=" * 100)
print("WITHIN-PERSON FIXED-EFFECTS RESULTS")
print("=" * 100)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("Forward home coefficient with prior functioning:")
print(
    forward.summary().tables[1]
    .as_text()
    .split("C(participant_id)")[0]
)

print()
print("Saved:")
print(OUT_FILE)
