#!/bin/bash
#SBATCH --job-name=cortex2_acc_manifest
#SBATCH --partition=cpu
#SBATCH --nodelist=node001
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/acc_manifest_%j.out
#SBATCH --error=logs/acc_manifest_%j.err

set -euo pipefail

cd "/shared/Digital Clinic/Cortex 2.0"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"

PYTHON="/home/akargara/envs/socialmedia/bin/python"

$PYTHON workflows/build_source_file_manifest.py \
  --stream accelerometer

echo "End: $(date)"
