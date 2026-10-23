"""Compute natural GPS feature variability in the independent holdout cohort.

Uses ONLY the seven full-data reference days per participant.

Primary normalization scale is frozen from calibration:
    median absolute difference across all 21 pairs of the
    participant's seven well-observed reference days.

No degradation data are read or used here.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

REFERENCE_PATH = (
    ROOT
    / "gps_reliability_holdout_reference_location_features.parquet"
)

OUT_PATH = (
    ROOT
    / "gps_reliability_holdout_reference_feature_variability_by_participant.csv"
)


FEATURES = [
    "entropy",
    "dominant_fraction",
    "effective_places",
]


reference = pd.read_parquet(
    REFERENCE_PATH
)

# Canonical short names used by the reliability analysis.
reference["entropy"] = (
    reference["location_entropy_nats"]
)

reference["dominant_fraction"] = (
    reference["dominant_place_fraction"]
)

reference["effective_places"] = (
    reference["effective_number_places"]
)


required = {
    "participant_id",
    "date",
    *FEATURES,
}


missing = (
    required
    - set(reference.columns)
)


if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}\n"
        f"Available columns: {sorted(reference.columns)}"
    )


if len(reference) != 112:
    raise RuntimeError(
        f"Expected 112 reference rows, found {len(reference)}."
    )


if (
    reference["participant_id"].nunique()
    != 16
):
    raise RuntimeError(
        "Expected 16 holdout participants."
    )


day_counts = (
    reference
    .groupby("participant_id")
    .size()
)


if not (
    day_counts
    == 7
).all():
    raise RuntimeError(
        "Every participant must have exactly 7 reference days."
    )


rows = []


for participant_id, x in reference.groupby(
    "participant_id",
    sort=True,
):

    row = {
        "participant_id":
            participant_id,

        "reference_days":
            len(x),
    }


    for feature in FEATURES:

        values = pd.to_numeric(
            x[feature],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )


        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"{participant_id}: non-finite {feature} value."
            )


        pairwise_differences = np.array(
            [
                abs(a - b)
                for a, b
                in combinations(
                    values,
                    2,
                )
            ],
            dtype=float,
        )


        if len(
            pairwise_differences
        ) != 21:
            raise RuntimeError(
                f"{participant_id} {feature}: "
                f"expected 21 pairwise differences, "
                f"found {len(pairwise_differences)}."
            )


        median_value = float(
            np.median(
                values
            )
        )


        row[
            f"{feature}_pairwise_median_abs_diff"
        ] = float(
            np.median(
                pairwise_differences
            )
        )


        row[
            f"{feature}_pairwise_q25_abs_diff"
        ] = float(
            np.quantile(
                pairwise_differences,
                0.25,
            )
        )


        row[
            f"{feature}_pairwise_q75_abs_diff"
        ] = float(
            np.quantile(
                pairwise_differences,
                0.75,
            )
        )


        row[
            f"{feature}_reference_iqr"
        ] = float(
            np.quantile(
                values,
                0.75,
            )
            - np.quantile(
                values,
                0.25,
            )
        )


        row[
            f"{feature}_reference_mad"
        ] = float(
            np.median(
                np.abs(
                    values
                    - median_value
                )
            )
        )


        row[
            f"{feature}_reference_min"
        ] = float(
            np.min(
                values
            )
        )


        row[
            f"{feature}_reference_median"
        ] = median_value


        row[
            f"{feature}_reference_max"
        ] = float(
            np.max(
                values
            )
        )


    rows.append(
        row
    )


variability = pd.DataFrame(
    rows
)


variability.to_csv(
    OUT_PATH,
    index=False,
)


print("=" * 80)
print("RAPHE GPS HOLDOUT NATURAL FEATURE VARIABILITY")
print("=" * 80)

print(
    "Participants:",
    len(
        variability
    ),
)

print(
    "Reference days:",
    len(
        reference
    ),
)


print()
print("PRIMARY PAIRWISE-MEDIAN NORMALIZATION SCALE")

for feature in FEATURES:

    column = (
        f"{feature}_pairwise_median_abs_diff"
    )

    values = variability[
        column
    ]


    print()
    print(feature)

    print(
        "  median:",
        f"{values.median():.4f}",
    )

    print(
        "  q25:",
        f"{values.quantile(0.25):.4f}",
    )

    print(
        "  q75:",
        f"{values.quantile(0.75):.4f}",
    )

    print(
        "  min:",
        f"{values.min():.4f}",
    )

    print(
        "  max:",
        f"{values.max():.4f}",
    )

    print(
        "  zero denominators:",
        int(
            (values <= 0).sum()
        ),
    )


print()
print("PARTICIPANT-SPECIFIC SCALES")

display_columns = [
    "participant_id",
    "entropy_pairwise_median_abs_diff",
    "dominant_fraction_pairwise_median_abs_diff",
    "effective_places_pairwise_median_abs_diff",
]

print(
    variability[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("Output:")
print(OUT_PATH)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
