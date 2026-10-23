"""Strict integrity audit of the frozen GPS reliability holdout matrix."""

from pathlib import Path

import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

EXTREME_DIR = (
    ROOT
    / "reliability_holdout_degradation_runs"
)

INTERMEDIATE_DIR = (
    ROOT
    / "reliability_holdout_intermediate_missingness_runs"
)


COVERAGES = {
    0.90,
    0.75,
    0.50,
    0.25,
}

PATTERNS = {
    "contiguous",
    "distributed",
}

K_VALUES = {
    2,
    4,
    8,
    16,
}

REPLICATES = set(
    range(5)
)


# ============================================================
# FILE INVENTORY
# ============================================================

extreme_files = sorted(
    EXTREME_DIR.glob(
        "coverage_*_*_rep_*.parquet"
    )
)

intermediate_files = sorted(
    INTERMEDIATE_DIR.glob(
        "coverage_*_k*_rep_*.parquet"
    )
)


print("=" * 80)
print("RAPHE GPS INDEPENDENT HOLDOUT MATRIX AUDIT")
print("=" * 80)

print(
    "Extreme files:",
    len(extreme_files),
)

print(
    "Intermediate files:",
    len(intermediate_files),
)


if len(extreme_files) != 40:
    raise RuntimeError(
        f"Expected 40 extreme files, found {len(extreme_files)}."
    )


if len(intermediate_files) != 80:
    raise RuntimeError(
        f"Expected 80 intermediate files, found {len(intermediate_files)}."
    )


# ============================================================
# READ + AUDIT
# ============================================================

all_frames = []

extreme_conditions = set()
intermediate_conditions = set()

bad_files = []


for kind, files in [
    (
        "extreme",
        extreme_files,
    ),
    (
        "intermediate",
        intermediate_files,
    ),
]:

    for path in files:

        try:

            df = pd.read_parquet(
                path
            )

        except Exception as exc:

            bad_files.append(
                (
                    path.name,
                    repr(exc),
                )
            )

            continue


        if len(df) != 112:

            raise RuntimeError(
                f"{path.name}: expected 112 rows, "
                f"found {len(df)}."
            )


        if (
            df[
                "participant_id"
            ].nunique()
            != 16
        ):

            raise RuntimeError(
                f"{path.name}: expected 16 participants."
            )


        participant_counts = (
            df
            .groupby(
                "participant_id"
            )
            .size()
        )


        if not (
            participant_counts
            == 7
        ).all():

            raise RuntimeError(
                f"{path.name}: every participant "
                "must contribute exactly 7 days."
            )


        if (
            "reference_match_missing"
            in df.columns
        ):

            missing_reference = (
                df[
                    "reference_match_missing"
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            )

            if missing_reference != 0:

                raise RuntimeError(
                    f"{path.name}: "
                    f"{missing_reference} missing reference matches."
                )


        coverage_values = (
            pd.to_numeric(
                df[
                    "target_coverage"
                ],
                errors="coerce",
            )
            .dropna()
            .unique()
        )


        if len(
            coverage_values
        ) != 1:

            raise RuntimeError(
                f"{path.name}: expected one target coverage."
            )


        coverage = float(
            coverage_values[0]
        )


        if coverage not in COVERAGES:

            raise RuntimeError(
                f"{path.name}: unexpected coverage {coverage}."
            )


        replicate_values = (
            pd.to_numeric(
                df[
                    "replicate"
                ],
                errors="coerce",
            )
            .dropna()
            .unique()
        )


        if len(
            replicate_values
        ) != 1:

            raise RuntimeError(
                f"{path.name}: expected one replicate."
            )


        replicate = int(
            replicate_values[0]
        )


        if replicate not in REPLICATES:

            raise RuntimeError(
                f"{path.name}: unexpected replicate {replicate}."
            )


        if kind == "extreme":

            pattern_values = (
                df[
                    "missingness_pattern"
                ]
                .astype(str)
                .unique()
            )


            if len(
                pattern_values
            ) != 1:

                raise RuntimeError(
                    f"{path.name}: expected one pattern."
                )


            pattern = str(
                pattern_values[0]
            )


            if pattern not in PATTERNS:

                raise RuntimeError(
                    f"{path.name}: unexpected pattern {pattern}."
                )


            condition = (
                coverage,
                pattern,
                replicate,
            )


            extreme_conditions.add(
                condition
            )


        else:

            n_run_column = None

            for candidate in [
                "n_runs_design",
                "n_runs",
                "missing_run_count_design",
            ]:

                if candidate in df.columns:

                    n_run_column = candidate
                    break


            if n_run_column is None:

                raise RuntimeError(
                    f"{path.name}: could not find "
                    "intermediate run-count column."
                )


            k_values = (
                pd.to_numeric(
                    df[
                        n_run_column
                    ],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )


            if len(
                k_values
            ) != 1:

                raise RuntimeError(
                    f"{path.name}: expected one k value."
                )


            k = int(
                k_values[0]
            )


            if k not in K_VALUES:

                raise RuntimeError(
                    f"{path.name}: unexpected k={k}."
                )


            condition = (
                coverage,
                k,
                replicate,
            )


            intermediate_conditions.add(
                condition
            )


        df[
            "audit_condition_type"
        ] = kind

        df[
            "audit_source_file"
        ] = path.name


        all_frames.append(
            df
        )


if bad_files:

    raise RuntimeError(
        f"Unreadable files: {bad_files}"
    )


# ============================================================
# CONDITION MATRIX
# ============================================================

expected_extreme = {
    (
        coverage,
        pattern,
        replicate,
    )
    for coverage in COVERAGES
    for pattern in PATTERNS
    for replicate in REPLICATES
}


expected_intermediate = {
    (
        coverage,
        k,
        replicate,
    )
    for coverage in COVERAGES
    for k in K_VALUES
    for replicate in REPLICATES
}


missing_extreme = (
    expected_extreme
    - extreme_conditions
)

unexpected_extreme = (
    extreme_conditions
    - expected_extreme
)


missing_intermediate = (
    expected_intermediate
    - intermediate_conditions
)

unexpected_intermediate = (
    intermediate_conditions
    - expected_intermediate
)


# ============================================================
# COMBINE
# ============================================================

combined = pd.concat(
    all_frames,
    ignore_index=True,
)


expected_rows = (
    120
    * 112
)


if len(
    combined
) != expected_rows:

    raise RuntimeError(
        f"Expected {expected_rows:,} total rows, "
        f"found {len(combined):,}."
    )


if (
    combined[
        "participant_id"
    ].nunique()
    != 16
):

    raise RuntimeError(
        "Combined matrix does not contain exactly 16 participants."
    )


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 80)
print("CONDITION MATRIX")
print("=" * 80)

print(
    "Unique extreme conditions:",
    len(
        extreme_conditions
    ),
)

print(
    "Missing extreme:",
    sorted(
        missing_extreme
    ),
)

print(
    "Unexpected extreme:",
    sorted(
        unexpected_extreme
    ),
)

print(
    "Unique intermediate conditions:",
    len(
        intermediate_conditions
    ),
)

print(
    "Missing intermediate:",
    sorted(
        missing_intermediate
    ),
)

print(
    "Unexpected intermediate:",
    sorted(
        unexpected_intermediate
    ),
)


print()
print("=" * 80)
print("COMBINED MATRIX")
print("=" * 80)

print(
    "Rows:",
    f"{len(combined):,}",
)

print(
    "Participants:",
    combined[
        "participant_id"
    ].nunique(),
)

print(
    "Source files:",
    combined[
        "audit_source_file"
    ].nunique(),
)

print(
    "Extreme rows:",
    int(
        (
            combined[
                "audit_condition_type"
            ]
            == "extreme"
        ).sum()
    ),
)

print(
    "Intermediate rows:",
    int(
        (
            combined[
                "audit_condition_type"
            ]
            == "intermediate"
        ).sum()
    ),
)


print()
print("=" * 80)
print("FINAL")
print("=" * 80)


passed = (
    len(extreme_conditions) == 40
    and len(intermediate_conditions) == 80
    and not missing_extreme
    and not unexpected_extreme
    and not missing_intermediate
    and not unexpected_intermediate
    and len(combined) == 13440
)


print(
    "HOLDOUT MATRIX INTEGRITY:",
    "PASS" if passed else "FAIL",
)


if not passed:
    raise RuntimeError(
        "Holdout degradation matrix failed integrity audit."
    )


print("=" * 80)
