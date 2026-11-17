from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical"
)

INPUT = ROOT / "gps_clinical_weekly_features_with_home.parquet"

OUTDIR = ROOT / "full_cohort_phq_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "phq_four_group_gps_summary.csv"


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(INPUT).copy()

df["depression_score"] = pd.to_numeric(
    df["depression_score"],
    errors="coerce",
)

df["time_at_home_pct"] = pd.to_numeric(
    df["time_at_home_pct"],
    errors="coerce",
)

df["home_hours_per_day"] = (
    df["time_at_home_pct"] / 100.0 * 24.0
)

df["movement_km_per_day"] = (
    pd.to_numeric(
        df["displacement_distance_mean_m"],
        errors="coerce",
    )
    / 1000.0
)

df["mobility_diversity"] = pd.to_numeric(
    df["location_entropy_mean"],
    errors="coerce",
)


# ============================================================
# FOUR PHQ GROUPS
# ============================================================

bins = [-0.001, 4, 9, 14, 27]

labels = [
    "Minimal (0–4)",
    "Mild (5–9)",
    "Moderate (10–14)",
    "Moderately severe / severe (15–27)",
]

df["phq_group"] = pd.cut(
    df["depression_score"],
    bins=bins,
    labels=labels,
    ordered=True,
)


features = {
    "Movement distance (km/day)": "movement_km_per_day",
    "Time at home (h/day)": "home_hours_per_day",
    "Mobility diversity": "mobility_diversity",
}


# ============================================================
# CLUSTER BOOTSTRAP
# Resample participants, preserving all repeated weekly rows
# ============================================================

def cluster_bootstrap_mean_ci(
    data,
    value_col,
    participant_col="participant_id",
    n_boot=5000,
    seed=42,
):
    clean = data[
        [participant_col, value_col]
    ].dropna().copy()

    participants = clean[
        participant_col
    ].unique()

    if len(participants) == 0:
        return np.nan, np.nan

    grouped = {
        pid: clean.loc[
            clean[participant_col].eq(pid),
            value_col,
        ].to_numpy()
        for pid in participants
    }

    rng = np.random.default_rng(seed)

    boot_means = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):

        sampled_ids = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        total = 0.0
        count = 0

        for pid in sampled_ids:
            values = grouped[pid]
            total += values.sum()
            count += len(values)

        boot_means[i] = (
            total / count
            if count > 0
            else np.nan
        )

    boot_means = boot_means[
        np.isfinite(boot_means)
    ]

    return (
        np.percentile(
            boot_means,
            2.5,
        ),
        np.percentile(
            boot_means,
            97.5,
        ),
    )


# ============================================================
# SUMMARY
# ============================================================

clinical = df.dropna(
    subset=["depression_score"]
).copy()

print("=" * 100)
print("PHQ-9 FOUR-GROUP GPS SUMMARY")
print("=" * 100)

print(
    "PHQ windows:",
    len(clinical),
)

print(
    "Participants:",
    clinical["participant_id"].nunique(),
)

print()
print("=" * 100)
print("PHQ GROUP DISTRIBUTION")
print("=" * 100)

for group in labels:

    x = clinical[
        clinical["phq_group"].eq(group)
    ]

    print(
        f"{group:<40}"
        f" windows={len(x):>4}"
        f" | participants="
        f"{x['participant_id'].nunique():>3}"
    )


rows = []

for feature_label, col in features.items():

    print()
    print("=" * 100)
    print(feature_label.upper())
    print("=" * 100)

    for group in labels:

        x = clinical.loc[
            clinical["phq_group"].eq(group),
            [
                "participant_id",
                col,
            ],
        ].dropna()

        values = x[col]

        if len(values) == 0:
            continue

        mean = values.mean()
        median = values.median()

        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)

        ci_low, ci_high = (
            cluster_bootstrap_mean_ci(
                x,
                value_col=col,
                n_boot=5000,
                seed=42,
            )
        )

        n_windows = len(x)
        n_participants = (
            x["participant_id"].nunique()
        )

        rows.append(
            {
                "feature": feature_label,
                "phq_group": group,
                "n_windows": n_windows,
                "n_participants": n_participants,
                "mean": mean,
                "median": median,
                "q25": q25,
                "q75": q75,
                "cluster_boot_ci_low": ci_low,
                "cluster_boot_ci_high": ci_high,
            }
        )

        print(
            f"{group:<40}"
            f" n={n_windows:>4}"
            f" | participants={n_participants:>3}"
            f" | mean={mean:>8.3f}"
            f" | median={median:>8.3f}"
            f" | 95% CI="
            f"[{ci_low:.3f}, {ci_high:.3f}]"
        )


summary = pd.DataFrame(rows)

summary.to_csv(
    OUT,
    index=False,
)


print()
print("=" * 100)
print("COMPACT DOSE-RESPONSE")
print("=" * 100)

for feature_label in features:

    sub = summary[
        summary["feature"].eq(
            feature_label
        )
    ]

    print()
    print(feature_label)

    for _, row in sub.iterrows():

        print(
            f"  {row['phq_group']:<40}"
            f"{row['mean']:.3f} "
            f"[{row['cluster_boot_ci_low']:.3f}, "
            f"{row['cluster_boot_ci_high']:.3f}]"
        )


print()
print("=" * 100)
print("SAVED")
print("=" * 100)
print(OUT)
