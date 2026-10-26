#!/bin/bash
#SBATCH --job-name=cortex2_cortex_source
#SBATCH --partition=cpu
#SBATCH --nodelist=node001
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cortex_source_%j.out
#SBATCH --error=logs/cortex_source_%j.err

set -euo pipefail

cd "/shared/Digital Clinic/Cortex 2.0"

PYTHON="/home/akargara/envs/socialmedia/bin/python"

$PYTHON workflows/validate_cortex_dq_5_source_aware.py \
  --stream gps

$PYTHON workflows/validate_cortex_dq_5_source_aware.py \
  --stream accelerometer
