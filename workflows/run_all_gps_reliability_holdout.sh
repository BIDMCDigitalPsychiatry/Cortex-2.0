#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -w node001
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --job-name=raphe_gps_holdout
#SBATCH --output=gps_reliability_holdout_%j.out
#SBATCH --error=gps_reliability_holdout_%j.err

set -euo pipefail


source /cm/shared/apps/miniforge3/26.3.2-3/etc/profile.d/conda.sh
conda activate /home/akargara/envs/socialmedia

cd "/shared/Digital Clinic/RAPHE"


ROOT="/shared/Digital Clinic/raphe_outputs/digital_clinic/feature_reliability/gps_observability"

EXTREME_DIR="${ROOT}/reliability_holdout_degradation_runs"
INTERMEDIATE_DIR="${ROOT}/reliability_holdout_intermediate_missingness_runs"

mkdir -p \
  "${EXTREME_DIR}" \
  "${INTERMEDIATE_DIR}"


# ============================================================
# A completed file must:
# - exist
# - be readable parquet
# - contain exactly 112 participant-days
# - contain exactly 16 participants
# ============================================================

is_valid_run() {

    local path="$1"

    python       workflows/check_gps_reliability_holdout_run.py       "${path}"
}


run_extreme() {

    local coverage="$1"
    local label="$2"
    local pattern="$3"
    local replicate="$4"

    local rep_label

    rep_label=$(printf "%02d" "${replicate}")

    local out="${EXTREME_DIR}/coverage_${label}_${pattern}_rep_${rep_label}.parquet"


    if is_valid_run "${out}"; then

        echo
        echo "SKIP valid extreme:"
        echo "${out}"

        return
    fi


    echo
    echo "============================================================"
    echo "RUN extreme"
    echo "coverage=${coverage}"
    echo "pattern=${pattern}"
    echo "replicate=${replicate}"
    echo "============================================================"


    python \
      workflows/run_gps_reliability_holdout_degradation_condition.py \
      --coverage "${coverage}" \
      --pattern "${pattern}" \
      --replicate "${replicate}"


    if ! is_valid_run "${out}"; then

        echo "ERROR: output failed integrity check:"
        echo "${out}"

        exit 1
    fi
}


run_intermediate() {

    local coverage="$1"
    local label="$2"
    local k="$3"
    local replicate="$4"

    local rep_label
    local k_label

    rep_label=$(printf "%02d" "${replicate}")
    k_label=$(printf "%02d" "${k}")

    local out="${INTERMEDIATE_DIR}/coverage_${label}_k${k_label}_rep_${rep_label}.parquet"


    if is_valid_run "${out}"; then

        echo
        echo "SKIP valid intermediate:"
        echo "${out}"

        return
    fi


    echo
    echo "============================================================"
    echo "RUN intermediate"
    echo "coverage=${coverage}"
    echo "k=${k}"
    echo "replicate=${replicate}"
    echo "============================================================"


    python \
      workflows/run_gps_reliability_holdout_intermediate_condition.py \
      --coverage "${coverage}" \
      --n-runs "${k}" \
      --replicate "${replicate}"


    if ! is_valid_run "${out}"; then

        echo "ERROR: output failed integrity check:"
        echo "${out}"

        exit 1
    fi
}


# ============================================================
# EXTREMES
#
# 4 coverages × 2 patterns × 5 replicates = 40
# ============================================================

for spec in \
    "0.90 090" \
    "0.75 075" \
    "0.50 050" \
    "0.25 025"
do

    read -r coverage label <<< "${spec}"


    for pattern in \
        contiguous \
        distributed
    do

        for replicate in \
            0 1 2 3 4
        do

            run_extreme \
                "${coverage}" \
                "${label}" \
                "${pattern}" \
                "${replicate}"

        done

    done

done


# ============================================================
# INTERMEDIATE STRUCTURES
#
# 4 coverages × 4 k values × 5 replicates = 80
# ============================================================

for spec in \
    "0.90 090" \
    "0.75 075" \
    "0.50 050" \
    "0.25 025"
do

    read -r coverage label <<< "${spec}"


    for k in \
        2 4 8 16
    do

        for replicate in \
            0 1 2 3 4
        do

            run_intermediate \
                "${coverage}" \
                "${label}" \
                "${k}" \
                "${replicate}"

        done

    done

done


# ============================================================
# FINAL INTEGRITY CHECK
# ============================================================

python - <<'PY'
from pathlib import Path

import pandas as pd


root = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

extreme_dir = (
    root
    / "reliability_holdout_degradation_runs"
)

intermediate_dir = (
    root
    / "reliability_holdout_intermediate_missingness_runs"
)


extreme_files = sorted(
    extreme_dir.glob(
        "coverage_*_*_rep_*.parquet"
    )
)

intermediate_files = sorted(
    intermediate_dir.glob(
        "coverage_*_k*_rep_*.parquet"
    )
)


def audit(
    files,
):

    valid = 0
    bad = []

    for path in files:

        try:

            df = pd.read_parquet(
                path,
                columns=[
                    "participant_id",
                ],
            )

            if (
                len(df) == 112
                and df[
                    "participant_id"
                ].nunique() == 16
            ):

                valid += 1

            else:

                bad.append(
                    path.name
                )

        except Exception:

            bad.append(
                path.name
            )

    return (
        valid,
        bad,
    )


extreme_valid, extreme_bad = audit(
    extreme_files
)

intermediate_valid, intermediate_bad = audit(
    intermediate_files
)


print()
print("=" * 80)
print("FINAL HOLDOUT DEGRADATION MATRIX")
print("=" * 80)

print(
    "Extreme files:",
    len(
        extreme_files
    ),
)

print(
    "Valid extreme files:",
    extreme_valid,
)

print(
    "Intermediate files:",
    len(
        intermediate_files
    ),
)

print(
    "Valid intermediate files:",
    intermediate_valid,
)

print(
    "Total valid conditions:",
    extreme_valid
    + intermediate_valid,
)

print(
    "Bad extreme files:",
    extreme_bad,
)

print(
    "Bad intermediate files:",
    intermediate_bad,
)


if (
    extreme_valid != 40
    or intermediate_valid != 80
    or extreme_bad
    or intermediate_bad
):

    raise RuntimeError(
        "Holdout degradation matrix incomplete."
    )


print()
print("EXPECTED MATRIX COMPLETE: 120 / 120")
print("=" * 80)
PY
