from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "../cortex2_outputs/digital_clinic"
)

RISK_PATH = (
    ROOT
    / "quality/hourly_missingness"
    / "gps_onset_contextual_risk_set.parquet"
)

CONTEXT_DIR = (
    ROOT
    / "quality/hourly_context/episodes"
)


# ============================================================
# COMPLETE HOURLY CONTEXT
# ============================================================

frames = []

for path in sorted(
    CONTEXT_DIR.glob("*.parquet")
):

    z = pd.read_parquet(
        path,
        columns=[
            "episode_id",
            "hour_index",
            "nearby_observation_present",
            "nearby_device_episode_evidence",
        ],
    )

    z = z.sort_values(
        "hour_index"
    ).copy()

    observed = (
        z[
            "nearby_observation_present"
        ]
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    evidence = bool(
        z[
            "nearby_device_episode_evidence"
        ]
        .iloc[0]
    )

    n = len(z)

    # Consecutive hours with no nearby positive evidence
    # ending at the CURRENT hour.
    run = np.zeros(
        n,
        dtype=float,
    )

    count = 0

    for i in range(n):

        if observed[i]:
            count = 0
        else:
            count += 1

        run[i] = count

    # For risk hour h, we only use information through h-1.
    prior_run = np.full(
        n,
        np.nan,
    )

    if n > 1:
        prior_run[1:] = run[:-1]

    # Has at least one nearby observation occurred
    # before the current hour?
    prior_seen = np.zeros(
        n,
        dtype=bool,
    )

    seen = False

    for i in range(n):

        prior_seen[i] = seen

        if observed[i]:
            seen = True

    # If the stream has no episode evidence, or if we have
    # never yet observed nearby data in this episode, duration
    # is not identifiable and remains NA.
    valid = (
        evidence
        & prior_seen
    )

    prior_run[
        ~valid
    ] = np.nan

    z[
        "prior_nearby_absence_run_hours"
    ] = prior_run

    z[
        "prior_nearby_run_observable"
    ] = valid

    frames.append(
        z[
            [
                "episode_id",
                "hour_index",
                "prior_nearby_absence_run_hours",
                "prior_nearby_run_observable",
            ]
        ]
    )


ctx = pd.concat(
    frames,
    ignore_index=True,
)


# ============================================================
# MERGE RISK SET
# ============================================================

x = pd.read_parquet(
    RISK_PATH
)

x = x.merge(
    ctx,
    on=[
        "episode_id",
        "hour_index",
    ],
    how="left",
    validate="one_to_one",
)


# GPS-specific onset and clean no-onset controls.
x = x[
    ~x["shared_onset"]
].copy()


x[
    "outcome"
] = np.where(
    x[
        "gps_specific_onset"
    ],
    "gps_specific_onset",
    "no_onset",
)


# ============================================================
# DESCRIPTIVES
# ============================================================

print("=" * 76)
print("PRIOR NEARBY-EVIDENCE ABSENCE RUN")
print("=" * 76)

for outcome in [
    "no_onset",
    "gps_specific_onset",
]:

    z = x[
        x["outcome"]
        == outcome
    ]

    v = (
        z[
            "prior_nearby_absence_run_hours"
        ]
        .dropna()
        .astype(float)
    )

    print()
    print(outcome)
    print(
        "risk hours:",
        f"{len(z):,}",
    )

    print(
        "observable run:",
        f"{len(v):,}",
        f"({100 * len(v) / len(z):.2f}%)"
        if len(z)
        else "",
    )

    if len(v) == 0:
        continue

    print(
        "mean:",
        round(
            v.mean(),
            2,
        )
    )

    print(
        "median:",
        round(
            v.median(),
            2,
        )
    )

    print(
        "p75:",
        round(
            v.quantile(
                0.75
            ),
            2,
        )
    )

    print(
        "p90:",
        round(
            v.quantile(
                0.90
            ),
            2,
        )
    )

    print(
        "p95:",
        round(
            v.quantile(
                0.95
            ),
            2,
        )
    )

    for threshold in [
        1,
        3,
        6,
        12,
        24,
    ]:

        n = int(
            (v >= threshold)
            .sum()
        )

        print(
            f">= {threshold:2d} h:",
            f"{n:,}",
            f"({100*n/len(v):.2f}%)",
        )


# ============================================================
# EPISODE-BALANCED CONTRAST
# ============================================================

rows = []

for episode_id, g in x.groupby(
    "episode_id",
    sort=False,
):

    cases = (
        g[
            g[
                "gps_specific_onset"
            ]
        ][
            "prior_nearby_absence_run_hours"
        ]
        .dropna()
    )

    controls = (
        g[
            ~g[
                "gps_specific_onset"
            ]
        ][
            "prior_nearby_absence_run_hours"
        ]
        .dropna()
    )

    if (
        len(cases) == 0
        or len(controls) == 0
    ):
        continue

    rows.append({
        "episode_id":
            episode_id,

        "n_onsets":
            len(cases),

        "onset_mean":
            cases.mean(),

        "control_mean":
            controls.mean(),

        "delta":
            cases.mean()
            - controls.mean(),
    })


d = pd.DataFrame(
    rows
)


print()
print("=" * 76)
print("EPISODE-BALANCED CONTRAST")
print("=" * 76)

print(
    "Eligible episodes:",
    len(d),
)

if len(d):

    values = (
        d["delta"]
        .to_numpy()
    )

    print(
        "Mean delta:",
        round(
            values.mean(),
            3,
        ),
        "hours",
    )

    print(
        "Median delta:",
        round(
            np.median(values),
            3,
        ),
        "hours",
    )

    print(
        "Episodes with longer absence before onset:",
        f"{100*np.mean(values > 0):.1f}%",
    )

    print(
        "Episodes with shorter absence before onset:",
        f"{100*np.mean(values < 0):.1f}%",
    )


# ============================================================
# SAVE
# ============================================================

OUT = (
    ROOT
    / "quality/hourly_missingness"
    / "nearby_absence_run_risk_set.parquet"
)

x.to_parquet(
    OUT,
    index=False,
)

print()
print("Saved:")
print(OUT)
