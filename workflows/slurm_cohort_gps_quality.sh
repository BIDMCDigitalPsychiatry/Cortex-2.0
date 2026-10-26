#!/bin/bash
#SBATCH --job-name=cortex2_gps_quality
#SBATCH --partition=cpu
#SBATCH --nodelist=node001
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gps_quality_%j.out
#SBATCH --error=logs/gps_quality_%j.err

set -euo pipefail

cd "/shared/Digital Clinic/Cortex 2.0"

PYTHON="/home/akargara/envs/socialmedia/bin/python"

$PYTHON workflows/build_cohort_daily_quality.py \
  --stream gps
