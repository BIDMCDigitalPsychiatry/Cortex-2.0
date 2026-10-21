#!/bin/bash

#SBATCH -p cpu
#SBATCH -w node001
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --job-name=raphe_gps_validation
#SBATCH --output=gps_validation_%j.out
#SBATCH --error=gps_validation_%j.err

set -e

source /cm/shared/apps/miniforge3/26.3.2-3/etc/profile.d/conda.sh
conda activate /home/akargara/envs/socialmedia
cd "/shared/Digital Clinic/RAPHE"

RUN_DIR="/shared/Digital Clinic/raphe_outputs/digital_clinic/feature_reliability/gps_observability/validation_degradation_runs"

mkdir -p "$RUN_DIR"

for coverage in 0.90 0.75 0.50 0.25; do

  case "$coverage" in
    0.90) label="090" ;;
    0.75) label="075" ;;
    0.50) label="050" ;;
    0.25) label="025" ;;
  esac

  for pattern in distributed contiguous; do
    for replicate in 0 1 2 3 4; do

      outfile="$RUN_DIR/coverage_${label}_${pattern}_rep_$(printf '%02d' "$replicate").parquet"

      if [ -f "$outfile" ]; then
        echo "============================================================"
        echo "SKIP: already complete"
        echo "$outfile"
        echo "============================================================"
        continue
      fi

      echo "============================================================"
      echo "coverage=$coverage"
      echo "pattern=$pattern"
      echo "replicate=$replicate"
      echo "============================================================"

      python workflows/run_gps_validation_degradation_condition.py \
        --coverage "$coverage" \
        --pattern "$pattern" \
        --replicate "$replicate"

    done
  done
done

echo
echo "============================================================"
echo "VALIDATION MATRIX COMPLETE"
echo "============================================================"

count=$(
  find "$RUN_DIR" \
    -name "coverage_*_rep_*.parquet" \
    -type f \
    | wc -l
)

echo "Completed run files: $count / 40"
