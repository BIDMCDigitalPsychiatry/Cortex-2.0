from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time"
)

DATA = (
    ROOT
    / "paper_replication"
    / "paper_replication_analysis_dataset.csv"
)

SEG = (
    ROOT
    / "phq_temporal_windows"
    / "phq8_phq9_home_time_strict_14d_cohort.csv"
)

OUT = (
    ROOT
    / "paper_replication"
    / "paper_replication_zero_transition_sensitivity.csv"
)

df = pd.read_csv(DATA, low_memory=False)
seg = pd.read_csv(SEG, low_memory=False)

seg["home_14d_all_pct"] = pd.to_numeric(
    seg["home_14d_all_pct"],
    errors="coerce",
)

transition_ids = set(
    seg.loc[
        np.isclose(
            seg["home_14d_all_pct"],
            0.0,
            atol=1e-9,
        ),
        "participant_id",
    ].astype(str)
)

df["participant_id"] = df["participant_id"].astype(str)

print("=" * 100)
print("HOME-TRANSITION SENSITIVITY ANALYSIS")
print("=" * 100)

print("Full dataset:")
print("  segments:", len(df))
print("  participants:", df["participant_id"].nunique())

sens = df[
    ~df["participant_id"].isin(
        transition_ids
    )
].copy()

print()
print("After excluding participants with 0%-home transition:")
print("  segments:", len(sens))
print("  participants:", sens["participant_id"].nunique())
print("  participants excluded:", len(transition_ids))


def gee(data, outcome, predictor, label):

    x = data[
        [
            "participant_id",
            outcome,
            predictor,
        ]
    ].dropna().copy()

    X = sm.add_constant(
        x[[predictor]]
    )

    model = sm.GEE(
        x[outcome],
        X,
        groups=x["participant_id"],
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    )

    fit = model.fit()

    beta = fit.params[predictor]
    se = fit.bse[predictor]
    p = fit.pvalues[predictor]

    ci = fit.conf_int().loc[predictor]

    return {
        "analysis": label,
        "outcome": outcome,
        "predictor": predictor,
        "n_segments": len(x),
        "n_participants": x["participant_id"].nunique(),
        "beta": beta,
        "se": se,
        "ci_low": ci.iloc[0],
        "ci_high": ci.iloc[1],
        "p": p,
    }


results = []

outcomes = [
    "home_14d_all_pct",
    "home_14d_weekday_matched_pct",
    "home_14d_weekend_pct",
]

for outcome in outcomes:

    for predictor in [
        "phq8_total",
        "phq9_total",
        "age",
    ]:

        if predictor not in df.columns:
            continue

        results.append(
            gee(
                df,
                outcome,
                predictor,
                "full",
            )
        )

        results.append(
            gee(
                sens,
                outcome,
                predictor,
                "exclude_transition_participants",
            )
        )


res = pd.DataFrame(results)

print()
print("=" * 100)
print("GEE RESULTS")
print("=" * 100)

print(
    res[
        [
            "analysis",
            "outcome",
            "predictor",
            "n_segments",
            "n_participants",
            "beta",
            "ci_low",
            "ci_high",
            "p",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

res.to_csv(
    OUT,
    index=False,
)

print()
print("Saved:", OUT)
