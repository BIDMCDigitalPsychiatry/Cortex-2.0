#!/bin/bash
#SBATCH --job-name=cortex2_acc_quality
#SBATCH --partition=cpu
#SBATCH --nodelist=node001
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/acc_quality_%j.out
#SBATCH --error=logs/acc_quality_%j.err

set -euo pipefail

cd "/shared/Digital Clinic/Cortex 2.0"

PYTHON="/home/akargara/envs/socialmedia/bin/python"

$PYTHON workflows/build_cohort_daily_quality.py \
  --stream accelerometer
