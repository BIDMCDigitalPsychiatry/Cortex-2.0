#!/bin/bash
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=cortex2_weekly
#SBATCH --output=cortex2_weekly_%j.out
#SBATCH --error=cortex2_weekly_%j.err

source /cm/shared/apps/miniforge3/26.3.2-3/etc/profile.d/conda.sh
conda activate /home/akargara/envs/socialmedia

cd "/shared/Digital Clinic/Cortex-2.0"

export PYTHONPATH="/shared/Digital Clinic/Cortex-2.0/src:${PYTHONPATH:-}"

echo "===== ENVIRONMENT CHECK ====="
which python
python --version
echo "PYTHONPATH=$PYTHONPATH"

python - <<'PY'
import cortex2_engine
print("cortex2_engine import OK")
print(cortex2_engine.__file__)
PY

echo "===== FEATURE BUILD ====="

python workflows/run_gps_clinical_weekly_features_checkpointed.py
