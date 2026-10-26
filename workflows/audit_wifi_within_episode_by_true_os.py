from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "../cortex2_outputs/digital_clinic"
)

RISK_PATH = (
    ROOT
    / "quality/hourly_missingness/"
    "gps_onset_contextual_risk_set.parquet"
)

OS_PATH = (
    ROOT
    / "inventory/"
    "risk_episode_lamp_analytics_os.csv"
)

OUTDIR = (
    ROOT
    / "quality/hourly_missingness/"
    "onset_models/true_os_interaction"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD
# ============================================================

x = pd.read_parquet(
    RISK_PATH
)

os_map = pd.read_csv(
    OS_PATH
)

x = x.merge(
    os_map[
        [
            "episode_id",
            "phone_os",
        ]
    ],
    on="episode_id",
    how="left",
    validate="many_to_one",
)

x = x[
    x["phone_os"].isin(
        [
            "iOS",
            "Android",
        ]
    )
].copy()


x["prev_wifi"] = (
    pd.to_numeric(
        x[
            "ctx_prev_unique_wifi_id_count"
        ],
        errors="coerce",
    )
    .fillna(0)
    .gt(0)
    .astype(float)
)


# ============================================================
# EPISODE-LEVEL CONTRASTS
# ============================================================

rows = []


for outcome_name, outcome, competing in [
    (
        "gps_specific",
        "gps_specific_onset",
        "shared_onset",
    ),
    (
        "shared",
        "shared_onset",
        "gps_specific_onset",
    ),
]:

    d = x[
        ~x[competing]
    ].copy()


    for (
        phone_os,
        episode_id,
    ), g in d.groupby(
        [
            "phone_os",
            "episode_id",
        ]
    ):

        onset = g[
            g[outcome]
        ]

        control = g[
            ~g[outcome]
        ]


        if (
            len(onset) == 0
            or len(control) == 0
        ):
            continue


        onset_wifi = (
            onset[
                "prev_wifi"
            ].mean()
        )

        control_wifi = (
            control[
                "prev_wifi"
            ].mean()
        )


        rows.append({
            "outcome":
                outcome_name,

            "phone_os":
                phone_os,

            "episode_id":
                episode_id,

            "onsets":
                len(onset),

            "control_hours":
                len(control),

            "onset_wifi_fraction":
                onset_wifi,

            "control_wifi_fraction":
                control_wifi,

            "delta_wifi":
                onset_wifi
                - control_wifi,
        })


ep = pd.DataFrame(
    rows
)


# ============================================================
# BOOTSTRAP EPISODE-LEVEL MEAN DELTA
# ============================================================

rng = np.random.default_rng(
    42
)


summary_rows = []


for (
    outcome,
    phone_os,
), g in ep.groupby(
    [
        "outcome",
        "phone_os",
    ]
):

    values = (
        g[
            "delta_wifi"
        ]
        .dropna()
        .to_numpy()
    )


    boot = []

    if len(values) > 0:

        for _ in range(
            10000
        ):

            sample = rng.choice(
                values,
                size=len(values),
                replace=True,
            )

            boot.append(
                sample.mean()
            )


    boot = np.asarray(
        boot
    )


    summary_rows.append({
        "outcome":
            outcome,

        "phone_os":
            phone_os,

        "episodes":
            len(values),

        "mean_delta":
            values.mean(),

        "median_delta":
            np.median(
                values
            ),

        "ci_low":
            np.quantile(
                boot,
                0.025,
            ),

        "ci_high":
            np.quantile(
                boot,
                0.975,
            ),

        "pct_episodes_positive":
            100
            * np.mean(
                values > 0
            ),

        "pct_episodes_negative":
            100
            * np.mean(
                values < 0
            ),
    })


summary = pd.DataFrame(
    summary_rows
)


for col in [
    "mean_delta",
    "median_delta",
    "ci_low",
    "ci_high",
]:

    summary[
        col
    ] *= 100


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("WITHIN-EPISODE WIFI CONTRASTS BY TRUE OS")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda v:
            f"{v:.2f}",
    )
)


print()
print(
    "Delta = Wi-Fi evidence at onset "
    "minus Wi-Fi evidence during that "
    "same episode's no-onset hours."
)

print(
    "Negative = Wi-Fi less common before onset."
)

print(
    "Positive = Wi-Fi more common before onset."
)


ep.to_csv(
    OUTDIR
    / "wifi_within_episode_contrasts.csv",
    index=False,
)

summary.to_csv(
    OUTDIR
    / "wifi_within_episode_summary.csv",
    index=False,
)


print()
print("Saved:")
print(OUTDIR)
