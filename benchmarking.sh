#!/bin/bash
#SBATCH --job-name=FAABPsPolarity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --partition=rome
#SBATCH --cpus-per-task=128

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load numba/0.58.1-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a

# ---------------------------
# THREADING / NUMBA SETTINGS
# ---------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# (optional but useful: force Numba to use threading layer)
export NUMBA_THREADING_LAYER=omp

PROJECT_HOME="$HOME/FAABPsTransportFieldSim"
SCRATCH_ROOT="${TMPDIR:-/tmp}/FAABPsTransportFieldSim_${SLURM_JOB_ID}"
PROJECT_SCRATCH="$SCRATCH_ROOT/project"

mkdir -p "$PROJECT_SCRATCH"

rsync -a \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'data' \
  --exclude 'visualizations' \
  "$PROJECT_HOME/" "$PROJECT_SCRATCH/"

cd "$PROJECT_SCRATCH"
mkdir -p data visualizations

echo "Running from: $(pwd)"
echo "TMPDIR: ${TMPDIR:-/tmp}"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# ---------------------------
# TIMING START
# ---------------------------
start=$(date +%s.%N)

python main.py

end=$(date +%s.%N)

elapsed=$(awk "BEGIN {print $end - $start}")

# ---------------------------
# SAVE RESULT
# ---------------------------
RESULT_FILE="$PROJECT_HOME/benchmarks.txt"
echo "cpus=${SLURM_CPUS_PER_TASK}, time=${elapsed}" >> "$RESULT_FILE"

echo "Elapsed time: $elapsed seconds"

# ---------------------------
# COPY OUTPUT BACK
# ---------------------------
mkdir -p "$PROJECT_HOME/data" "$PROJECT_HOME/visualizations"
rsync -a data/ "$PROJECT_HOME/data/"
rsync -a visualizations/ "$PROJECT_HOME/visualizations/"

echo "Done."