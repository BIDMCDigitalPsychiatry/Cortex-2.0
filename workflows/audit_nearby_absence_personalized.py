from pathlib import Path

import numpy as np
import pandas as pd


PATH = Path(
    "../cortex2_outputs/digital_clinic/"
    "quality/hourly_missingness/"
    "nearby_absence_run_risk_set.parquet"
)

x = pd.read_parquet(PATH)


THRESHOLDS = [
    1,
    3,
    6,
    12,
    24,
]


def bootstrap_ci(
    values,
    n_boot=10000,
    seed=2026,
):

    v = np.asarray(
        values,
        dtype=float,
    )

    v = v[
        np.isfinite(v)
    ]

    if len(v) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(
        n_boot
    )

    for i in range(n_boot):

        means[i] = rng.choice(
            v,
            size=len(v),
            replace=True,
        ).mean()

    return (
        np.quantile(
            means,
            0.025,
        ),
        np.quantile(
            means,
            0.975,
        ),
    )


rows = []
percentile_rows = []


for episode_id, g in x.groupby(
    "episode_id",
    sort=False,
):

    cases = g[
        g["gps_specific_onset"]
    ].copy()

    controls = g[
        ~g["gps_specific_onset"]
    ].copy()

    cases = cases[
        cases[
            "prior_nearby_absence_run_hours"
        ].notna()
    ]

    controls = controls[
        controls[
            "prior_nearby_absence_run_hours"
        ].notna()
    ]

    if (
        len(cases) == 0
        or len(controls) == 0
    ):
        continue

    case_v = (
        cases[
            "prior_nearby_absence_run_hours"
        ]
        .astype(float)
        .to_numpy()
    )

    ctrl_v = (
        controls[
            "prior_nearby_absence_run_hours"
        ]
        .astype(float)
        .to_numpy()
    )

    row = {
        "episode_id":
            episode_id,

        "n_onsets":
            len(case_v),

        "n_controls":
            len(ctrl_v),

        "case_mean":
            case_v.mean(),

        "control_mean":
            ctrl_v.mean(),

        "mean_delta":
            case_v.mean()
            - ctrl_v.mean(),

        "case_median":
            np.median(case_v),

        "control_median":
            np.median(ctrl_v),

        "median_delta":
            np.median(case_v)
            - np.median(ctrl_v),
    }

    for t in THRESHOLDS:

        case_prop = np.mean(
            case_v >= t
        )

        ctrl_prop = np.mean(
            ctrl_v >= t
        )

        row[
            f"case_ge_{t}h"
        ] = case_prop

        row[
            f"control_ge_{t}h"
        ] = ctrl_prop

        row[
            f"delta_ge_{t}h"
        ] = (
            case_prop
            - ctrl_prop
        )

    rows.append(row)

    # --------------------------------------------------------
    # Personalized empirical percentile
    #
    # Compare each onset's run length with the distribution
    # of no-onset risk hours from the SAME episode.
    #
    # Mid-rank handles ties, especially the large number of
    # zero-hour runs.
    # --------------------------------------------------------

    onset_percentiles = []

    for value in case_v:

        less = np.sum(
            ctrl_v < value
        )

        equal = np.sum(
            ctrl_v == value
        )

        percentile = (
            less
            + 0.5 * equal
        ) / len(ctrl_v)

        onset_percentiles.append(
            percentile
        )

    percentile_rows.append({
        "episode_id":
            episode_id,

        "n_onsets":
            len(
                onset_percentiles
            ),

        "mean_onset_percentile":
            np.mean(
                onset_percentiles
            ),

        "median_onset_percentile":
            np.median(
                onset_percentiles
            ),

        "prop_onsets_ge_p75":
            np.mean(
                np.asarray(
                    onset_percentiles
                ) >= 0.75
            ),

        "prop_onsets_ge_p90":
            np.mean(
                np.asarray(
                    onset_percentiles
                ) >= 0.90
            ),

        "prop_onsets_ge_p95":
            np.mean(
                np.asarray(
                    onset_percentiles
                ) >= 0.95
            ),
    })


d = pd.DataFrame(
    rows
)

p = pd.DataFrame(
    percentile_rows
)


print("=" * 76)
print("EPISODE-BALANCED ABSENCE THRESHOLDS")
print("=" * 76)


for t in THRESHOLDS:

    col = (
        f"delta_ge_{t}h"
    )

    v = d[col].dropna()

    lo, hi = bootstrap_ci(v)

    print()
    print(
        f"Nearby absence >= {t} h"
    )

    print(
        "Eligible episodes:",
        len(v),
    )

    print(
        "Mean onset-control difference:",
        f"{100*v.mean():.2f}",
        "percentage points",
    )

    print(
        "Median difference:",
        f"{100*v.median():.2f}",
        "percentage points",
    )

    print(
        "Bootstrap 95% CI:",
        (
            f"{100*lo:.2f}",
            f"{100*hi:.2f}",
        ),
    )

    print(
        "Episodes with higher onset prevalence:",
        f"{100*np.mean(v > 0):.1f}%",
    )

    print(
        "Episodes with lower onset prevalence:",
        f"{100*np.mean(v < 0):.1f}%",
    )


print()
print("=" * 76)
print("PERSONALIZED ABSENCE-RUN PERCENTILES")
print("=" * 76)

v = p[
    "mean_onset_percentile"
].dropna()

lo, hi = bootstrap_ci(v)

print(
    "Eligible episodes:",
    len(v),
)

print(
    "Mean episode-level onset percentile:",
    f"{v.mean():.3f}",
)

print(
    "Median episode-level onset percentile:",
    f"{v.median():.3f}",
)

print(
    "Bootstrap 95% CI for mean:",
    f"[{lo:.3f}, {hi:.3f}]",
)

print(
    "Episodes with onset percentile > 0.50:",
    f"{100*np.mean(v > 0.50):.1f}%",
)

print(
    "Episodes with onset percentile > 0.75:",
    f"{100*np.mean(v > 0.75):.1f}%",
)


for col, label in [
    (
        "prop_onsets_ge_p75",
        "Onsets >= participant-specific P75",
    ),
    (
        "prop_onsets_ge_p90",
        "Onsets >= participant-specific P90",
    ),
    (
        "prop_onsets_ge_p95",
        "Onsets >= participant-specific P95",
    ),
]:

    z = p[col].dropna()

    print()
    print(label)

    print(
        "Episode-balanced mean:",
        f"{100*z.mean():.2f}%",
    )

    print(
        "Median:",
        f"{100*z.median():.2f}%",
    )


print()
print("=" * 76)
print("MOST POSITIVE MEAN-RUN CONTRASTS")
print("=" * 76)

print(
    d[
        [
            "episode_id",
            "n_onsets",
            "case_mean",
            "control_mean",
            "mean_delta",
        ]
    ]
    .sort_values(
        "mean_delta",
        ascending=False,
    )
    .head(15)
    .to_string(
        index=False
    )
)


print()
print("=" * 76)
print("MOST NEGATIVE MEAN-RUN CONTRASTS")
print("=" * 76)

print(
    d[
        [
            "episode_id",
            "n_onsets",
            "case_mean",
            "control_mean",
            "mean_delta",
        ]
    ]
    .sort_values(
        "mean_delta"
    )
    .head(15)
    .to_string(
        index=False
    )
)


OUT_DIR = (
    PATH.parent
    / "nearby_absence_personalized"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

d.to_csv(
    OUT_DIR
    / "episode_threshold_contrasts.csv",
    index=False,
)

p.to_csv(
    OUT_DIR
    / "episode_percentiles.csv",
    index=False,
)

print()
print("Saved:")
print(OUT_DIR)
