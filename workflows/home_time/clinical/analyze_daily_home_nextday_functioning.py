from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic"
)

HOME_FILE = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "daily_home_time.csv"
)

FUNC_FILE = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "daily_functioning"
    / "daily_functioning_clean.csv"
)

OUT = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "daily_longitudinal"
)

OUT.mkdir(parents=True, exist_ok=True)

DATA_OUT = OUT / "home_t_function_t1_analysis_dataset.csv"
STATS_OUT = OUT / "home_t_function_t1_primary_model_stats.csv"

# ============================================================
# LOAD HOME
# ============================================================

home = pd.read_csv(
    HOME_FILE,
    low_memory=False
)

home["date"] = pd.to_datetime(
    home["date"],
    errors="coerce"
).dt.normalize()

home["time_at_home_pct"] = pd.to_numeric(
    home["time_at_home_pct"],
    errors="coerce"
)

home["supported_duration_ms"] = pd.to_numeric(
    home["supported_duration_ms"],
    errors="coerce"
)

home["gps_hours"] = (
    home["supported_duration_ms"]
    / 1000
    / 60
    / 60
)

home = (
    home[
        [
            "participant_id",
            "date",
            "time_at_home_pct",
            "gps_hours",
            "home_time_status",
        ]
    ]
    .dropna(
        subset=[
            "participant_id",
            "date",
            "time_at_home_pct",
        ]
    )
)

# One row per participant-day
home = (
    home.groupby(
        ["participant_id", "date"],
        as_index=False
    )
    .agg(
        home_pct=("time_at_home_pct", "mean"),
        gps_hours=("gps_hours", "max"),
        home_time_status=("home_time_status", "first"),
    )
)

# ============================================================
# LOAD FUNCTIONING
# ============================================================

func = pd.read_csv(
    FUNC_FILE,
    low_memory=False
)

func["date"] = pd.to_datetime(
    func["assessment_date"],
    errors="coerce"
).dt.normalize()

func["functioning"] = pd.to_numeric(
    func["functioning_score"],
    errors="coerce"
)

func = (
    func[
        [
            "participant_id",
            "date",
            "functioning",
        ]
    ]
    .dropna()
    .groupby(
        ["participant_id", "date"],
        as_index=False
    )
    .agg(
        functioning=("functioning", "mean")
    )
)

# ============================================================
# BUILD EXACT t -> t+1 PAIRS
# ============================================================

# outcome day t+1
func_t1 = func.rename(
    columns={
        "date": "outcome_date",
        "functioning": "functioning_t1",
    }
).copy()

func_t1["date"] = (
    func_t1["outcome_date"]
    - pd.Timedelta(days=1)
)

df = home.merge(
    func_t1[
        [
            "participant_id",
            "date",
            "outcome_date",
            "functioning_t1",
        ]
    ],
    on=[
        "participant_id",
        "date",
    ],
    how="inner"
)

# Add functioning on exposure day t
func_t = func.rename(
    columns={
        "functioning": "functioning_t"
    }
)

df = df.merge(
    func_t[
        [
            "participant_id",
            "date",
            "functioning_t",
        ]
    ],
    on=[
        "participant_id",
        "date",
    ],
    how="left"
)

# ============================================================
# PRIMARY QC: >= 6 HOURS GPS
# ============================================================

df = df[
    df["gps_hours"] >= 6
].copy()

# ============================================================
# STUDY DAY
# ============================================================

first_date = (
    df.groupby("participant_id")["date"]
    .transform("min")
)

df["study_day"] = (
    df["date"] - first_date
).dt.days

df["weekend"] = (
    df["date"].dt.dayofweek >= 5
).astype(int)

# ============================================================
# WITHIN / BETWEEN DECOMPOSITION
# ============================================================

person_home = (
    df.groupby("participant_id")["home_pct"]
    .transform("mean")
)

df["home_mean"] = person_home

df["home_dev"] = (
    df["home_pct"]
    - df["home_mean"]
)

# Scale per +10 percentage points
df["home_mean_10"] = (
    df["home_mean"] / 10
)

df["home_dev_10"] = (
    df["home_dev"] / 10
)

# Person-center prior-day functioning
person_func = (
    df.groupby("participant_id")["functioning_t"]
    .transform("mean")
)

df["functioning_mean"] = person_func

df["functioning_dev_t"] = (
    df["functioning_t"]
    - df["functioning_mean"]
)

# Center nuisance covariates
df["gps_hours_c"] = (
    df["gps_hours"]
    - df["gps_hours"].mean()
)

df["study_day_10"] = (
    df["study_day"] / 10
)

# ============================================================
# MODEL DATA
# ============================================================

model_df = df.dropna(
    subset=[
        "functioning_t1",
        "functioning_t",
        "home_dev_10",
        "home_mean_10",
        "gps_hours_c",
        "study_day_10",
        "weekend",
    ]
).copy()

# Require at least 3 usable observations per person
counts = (
    model_df.groupby("participant_id")
    .size()
)

keep_ids = counts[
    counts >= 3
].index

model_df = model_df[
    model_df["participant_id"].isin(keep_ids)
].copy()

model_df.to_csv(
    DATA_OUT,
    index=False
)

print()
print("=" * 100)
print("PRIMARY DAILY LAGGED HOME -> FUNCTIONING ANALYSIS")
print("=" * 100)

print("Usable paired days:", len(model_df))
print(
    "Participants:",
    model_df["participant_id"].nunique()
)

print()
print("Pairs per participant:")
print(
    model_df.groupby("participant_id")
    .size()
    .describe()
)

print()
print("Home-time deviation distribution (% points):")
print(
    model_df["home_dev"].describe()
)

print()
print("Functioning[t+1] distribution:")
print(
    model_df["functioning_t1"].describe()
)

# ============================================================
# PRIMARY MODEL
#
# Random intercept only first.
# ============================================================

formula = (
    "functioning_t1 ~ "
    "home_dev_10 + "
    "home_mean_10 + "
    "functioning_dev_t + "
    "functioning_mean + "
    "weekend + "
    "gps_hours_c + "
    "study_day_10"
)

model = smf.mixedlm(
    formula,
    data=model_df,
    groups=model_df["participant_id"],
    re_formula="1"
)

fit = model.fit(
    reml=False,
    method="lbfgs",
    maxiter=1000
)

print()
print("=" * 100)
print("PRIMARY RANDOM-INTERCEPT MIXED MODEL")
print("=" * 100)
print(fit.summary())

# ============================================================
# EXTRACT PRIMARY EFFECT
# ============================================================

term = "home_dev_10"

beta = float(
    fit.params[term]
)

se = float(
    fit.bse[term]
)

ci_low = beta - 1.96 * se
ci_high = beta + 1.96 * se

try:
    p = float(
        fit.pvalues[term]
    )
except Exception:
    p = np.nan

print()
print("=" * 100)
print("PRIMARY ESTIMAND")
print("=" * 100)

print(
    "Within-person effect of +10 percentage points "
    "more home time than usual:"
)

print("beta =", f"{beta:.4f}")
print(
    "95% CI =",
    f"{ci_low:.4f}",
    "to",
    f"{ci_high:.4f}"
)
print("p =", f"{p:.4f}")

# ============================================================
# SECONDARY: WITHOUT PRIOR FUNCTIONING
#
# Useful to see how much lagged outcome adjustment changes
# the estimate.
# ============================================================

formula_no_prior = (
    "functioning_t1 ~ "
    "home_dev_10 + "
    "home_mean_10 + "
    "weekend + "
    "gps_hours_c + "
    "study_day_10"
)

fit_no_prior = smf.mixedlm(
    formula_no_prior,
    data=model_df,
    groups=model_df["participant_id"],
    re_formula="1"
).fit(
    reml=False,
    method="lbfgs",
    maxiter=1000
)

beta2 = float(
    fit_no_prior.params["home_dev_10"]
)

se2 = float(
    fit_no_prior.bse["home_dev_10"]
)

p2 = float(
    fit_no_prior.pvalues["home_dev_10"]
)

# ============================================================
# RANDOM SLOPE SENSITIVITY
# ============================================================

random_slope_ok = False
rs_beta = np.nan
rs_se = np.nan
rs_p = np.nan
rs_slope_sd = np.nan

try:

    rs = smf.mixedlm(
        formula,
        data=model_df,
        groups=model_df["participant_id"],
        re_formula="~home_dev_10"
    ).fit(
        reml=False,
        method="lbfgs",
        maxiter=2000
    )

    random_slope_ok = bool(rs.converged)

    rs_beta = float(
        rs.params["home_dev_10"]
    )

    rs_se = float(
        rs.bse["home_dev_10"]
    )

    rs_p = float(
        rs.pvalues["home_dev_10"]
    )

    # Random-effect covariance matrix:
    # intercept first, slope second
    rs_slope_sd = float(
        np.sqrt(
            rs.cov_re.iloc[1, 1]
        )
    )

    print()
    print("=" * 100)
    print("RANDOM-SLOPE SENSITIVITY")
    print("=" * 100)

    print("Converged:", rs.converged)
    print(rs.summary())

except Exception as e:

    print()
    print("=" * 100)
    print("RANDOM-SLOPE SENSITIVITY")
    print("=" * 100)

    print(
        "Random-slope model did not fit reliably:"
    )
    print(str(e))

# ============================================================
# SAVE SUMMARY
# ============================================================

stats = pd.DataFrame(
    [
        {
            "model": "primary_random_intercept_with_prior_function",
            "n_participants": model_df["participant_id"].nunique(),
            "n_pairs": len(model_df),
            "beta_home_dev_per_10pp": beta,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p,
            "random_slope_sd": np.nan,
            "converged": fit.converged,
        },
        {
            "model": "random_intercept_without_prior_function",
            "n_participants": model_df["participant_id"].nunique(),
            "n_pairs": len(model_df),
            "beta_home_dev_per_10pp": beta2,
            "se": se2,
            "ci_low": beta2 - 1.96 * se2,
            "ci_high": beta2 + 1.96 * se2,
            "p_value": p2,
            "random_slope_sd": np.nan,
            "converged": fit_no_prior.converged,
        },
        {
            "model": "random_slope_sensitivity",
            "n_participants": model_df["participant_id"].nunique(),
            "n_pairs": len(model_df),
            "beta_home_dev_per_10pp": rs_beta,
            "se": rs_se,
            "ci_low": (
                rs_beta - 1.96 * rs_se
                if pd.notna(rs_beta)
                else np.nan
            ),
            "ci_high": (
                rs_beta + 1.96 * rs_se
                if pd.notna(rs_beta)
                else np.nan
            ),
            "p_value": rs_p,
            "random_slope_sd": rs_slope_sd,
            "converged": random_slope_ok,
        },
    ]
)

stats.to_csv(
    STATS_OUT,
    index=False
)

print()
print("=" * 100)
print("MODEL COMPARISON")
print("=" * 100)

print(
    stats.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("Saved:")
print(DATA_OUT)
print(STATS_OUT)
