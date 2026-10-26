from pathlib import Path

import numpy as np
import pandas as pd


PATH = Path(
    "../cortex2_outputs/digital_clinic/"
    "quality/hourly_missingness/"
    "gps_onset_contextual_risk_set.parquet"
)

x = pd.read_parquet(PATH)


# ------------------------------------------------------------
# Derived variables
# ------------------------------------------------------------

x[
    "prev_battery_observed_num"
] = (
    x[
        "ctx_prev_battery_observed"
    ]
    .astype(float)
)

x[
    "prev_device_usage_observed_num"
] = (
    x[
        "ctx_prev_device_usage_observed"
    ]
    .astype(float)
)

x[
    "prev_nearby_observed_num"
] = (
    x[
        "ctx_prev_nearby_observed"
    ]
    .astype(float)
)


VARIABLES = {
    "prev_acc_dq":
        "ACC DQ",

    "prev_battery_observed_num":
        "Battery/screen evidence",

    "ctx_prev_battery_last":
        "Battery level",

    "prev_device_usage_observed_num":
        "Device-usage evidence",

    "prev_nearby_observed_num":
        "Nearby-device evidence",

    "ctx_prev_unlock_count":
        "Unlock count",

    "ctx_prev_screen_wake_count":
        "Screen wakes",
}


def bootstrap_ci(
    values,
    *,
    n_boot=10000,
    seed=2026,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        means[i] = sample.mean()

    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def episode_contrasts(
    outcome,
    variable,
):

    rows = []

    for episode_id, g in (
        x.groupby(
            "episode_id",
            sort=False,
        )
    ):

        cases = g[
            g[outcome]
        ]

        controls = g[
            ~g[
                "any_gps_onset"
            ]
        ]

        if (
            len(cases) == 0
            or len(controls) == 0
        ):
            continue

        case_values = pd.to_numeric(
            cases[variable],
            errors="coerce",
        ).dropna()

        control_values = pd.to_numeric(
            controls[variable],
            errors="coerce",
        ).dropna()

        if (
            len(case_values) == 0
            or len(control_values) == 0
        ):
            continue

        rows.append({
            "episode_id":
                episode_id,

            "n_onsets":
                len(cases),

            "case_mean":
                case_values.mean(),

            "control_mean":
                control_values.mean(),

            "delta":
                (
                    case_values.mean()
                    - control_values.mean()
                ),
        })

    return pd.DataFrame(
        rows
    )


print("=" * 76)
print("ONSET CONCENTRATION")
print("=" * 76)

for outcome in [
    "gps_specific_onset",
    "shared_onset",
]:

    counts = (
        x[
            x[outcome]
        ]
        .groupby(
            "episode_id"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    total = int(
        counts.sum()
    )

    print(
        f"\n{outcome}"
    )

    print(
        "Episodes with onset:",
        len(counts),
    )

    print(
        "Total onsets:",
        total,
    )

    if len(counts):

        for k in [
            1,
            4,
            10,
        ]:

            kk = min(
                k,
                len(counts),
            )

            n = int(
                counts.head(
                    kk
                ).sum()
            )

            print(
                f"Top {kk} episodes: "
                f"{n:,} "
                f"({100*n/total:.1f}%)"
            )


print(
    "\n"
    + "=" * 76
)

print(
    "EPISODE-BALANCED WITHIN-EPISODE CONTRASTS"
)

print(
    "=" * 76
)

print(
    "\nDelta = mean at onset "
    "- mean during no-onset risk hours "
    "within the same episode."
)

print(
    "Each episode contributes ONE delta "
    "regardless of its number of onsets."
)


for outcome in [
    "gps_specific_onset",
    "shared_onset",
]:

    print(
        "\n"
        + "#" * 76
    )

    print(
        outcome.upper()
    )

    print(
        "#" * 76
    )

    for variable, label in (
        VARIABLES.items()
    ):

        d = episode_contrasts(
            outcome,
            variable,
        )

        if len(d) == 0:
            print(
                f"\n{label}: "
                "no eligible episodes"
            )
            continue

        values = (
            d["delta"]
            .to_numpy()
        )

        low, high = bootstrap_ci(
            values
        )

        negative = (
            100
            * np.mean(
                values < 0
            )
        )

        positive = (
            100
            * np.mean(
                values > 0
            )
        )

        print(
            f"\n{label}"
        )

        print(
            "  episodes:",
            len(d),
        )

        print(
            "  mean delta:",
            f"{np.mean(values):.4f}",
        )

        print(
            "  median delta:",
            f"{np.median(values):.4f}",
        )

        print(
            "  bootstrap 95% CI:",
            f"[{low:.4f}, {high:.4f}]",
        )

        print(
            "  episodes with negative delta:",
            f"{negative:.1f}%",
        )

        print(
            "  episodes with positive delta:",
            f"{positive:.1f}%",
        )
