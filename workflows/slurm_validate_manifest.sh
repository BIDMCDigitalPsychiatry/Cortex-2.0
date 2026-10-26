#!/bin/bash
#SBATCH --job-name=cortex2_manifest_validation
#SBATCH --partition=cpu
#SBATCH --nodelist=node001
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/manifest_validation_%j.out
#SBATCH --error=logs/manifest_validation_%j.err

set -euo pipefail

cd "/shared/Digital Clinic/Cortex 2.0"

/home/akargara/envs/socialmedia/bin/python \
  workflows/validate_manifest_equivalence.py
