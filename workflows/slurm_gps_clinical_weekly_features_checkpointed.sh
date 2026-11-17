#!/bin/bash

#SBATCH -p cpu
#SBATCH -w node001
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=c2_gps_clin
#SBATCH --output=gps_clin_features_%j.out
#SBATCH --error=gps_clin_features_%j.err

set -euo pipefail

source /cm/shared/apps/miniforge3/26.3.2-3/etc/profile.d/conda.sh
conda activate /home/akargara/envs/socialmedia

cd "/shared/Digital Clinic/Cortex-2.0"

export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Cortex 2.0 clinical GPS feature build"
echo "Participant-level checkpoint/resume enabled"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version
echo "Started: $(date)"
echo "============================================================"

python -u \
  workflows/run_gps_clinical_weekly_features_checkpointed.py

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
