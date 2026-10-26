#!/bin/bash
#SBATCH -p cpu
#SBATCH -w node001
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --job-name=cortex2_gps_intermediate
#SBATCH --output=gps_intermediate_%j.out
#SBATCH --error=gps_intermediate_%j.err

set -euo pipefail

source /cm/shared/apps/miniforge3/26.3.2-3/etc/profile.d/conda.sh
conda activate /home/akargara/envs/socialmedia

cd "/shared/Digital Clinic/Cortex 2.0"

RUN_DIR="/shared/Digital Clinic/cortex2_outputs/digital_clinic/feature_reliability/gps_observability/intermediate_missingness_runs"

mkdir -p "$RUN_DIR"

for coverage in 0.90 0.75 0.50 0.25; do

    case "$coverage" in
        0.90) label="090" ;;
        0.75) label="075" ;;
        0.50) label="050" ;;
        0.25) label="025" ;;
    esac

    for n_runs in 2 4 8 16; do

        klabel=$(printf "%02d" "$n_runs")

        for replicate in 0 1 2 3 4; do

            outfile="$RUN_DIR/coverage_${label}_k${klabel}_rep_$(printf '%02d' "$replicate").parquet"

            if [ -f "$outfile" ]; then
                echo "SKIP existing: $outfile"
                continue
            fi

            echo
            echo "============================================================"
            echo "coverage=$coverage k=$n_runs replicate=$replicate"
            echo "============================================================"

            python workflows/run_gps_intermediate_missingness_condition.py \
                --coverage "$coverage" \
                --n-runs "$n_runs" \
                --replicate "$replicate"

        done
    done
done

count=$(
    find "$RUN_DIR" \
        -name "coverage_*_k*_rep_*.parquet" \
        -type f \
        | wc -l
)

echo
echo "Completed intermediate run files: $count / 80"
