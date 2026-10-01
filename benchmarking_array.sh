#!/bin/bash
#SBATCH --job-name=FAABPs_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --partition=rome
#SBATCH --array=0-41

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load numba/0.58.1-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_THREADING_LAYER=omp

# ---------------------------
# TASK → (N, run_id) MAPPING
# ---------------------------
N_LIST=(16000 16000 16000 16000 16000 16000 250 250 250 250 250 250 375 375 375 375 375 375 500 500 500 500 500 500 750 750 750 750 750 750 1000 1000 1000 1000 1000 1000 1500 1500 1500 1500 1500 1500 2000 2000 2000 2000 2000 2000 3000 3000 3000 3000 3000 3000 4000 4000 4000 4000 4000 4000 6000 6000 6000 6000 6000 6000 8000 8000 8000 8000 8000 8000 12000 12000 12000 12000 12000 12000)
N=${N_LIST[$SLURM_ARRAY_TASK_ID]}
RUN_ID=$(( SLURM_ARRAY_TASK_ID % 6 + 1 ))
OUTPUT_NAME="results_${N}_run${RUN_ID}"

# ---------------------------
# ISOLATED SCRATCH PER TASK
# ---------------------------
PROJECT_HOME="$HOME/FAABPsTransportFieldSim"
SCRATCH_ROOT="${TMPDIR:-/tmp}/FAABPs_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

# ---------------------------
# RUN
# ---------------------------
mkdir -p "$PROJECT_HOME/logsSwarm"
LOG_FILE="$PROJECT_HOME/logsSwarm/task_${SLURM_ARRAY_TASK_ID}.txt"

echo "Task $SLURM_ARRAY_TASK_ID: N=$N run=$RUN_ID output=$OUTPUT_NAME" | tee -a "$LOG_FILE"
echo "Running from: $(pwd)"

start=$(date +%s.%N)
python main.py "$N" "$OUTPUT_NAME" >> "$LOG_FILE" 2>&1
end=$(date +%s.%N)
elapsed=$(awk "BEGIN {print $end - $start}")

echo "N=${N}, run=${RUN_ID}, elapsed: $elapsed seconds" >> "$LOG_FILE"

# ---------------------------
# SHARED RESULT LINE
# Short appends are atomic on ext4/Lustre; safe for one line per task.
# ---------------------------
RESULT_FILE="$PROJECT_HOME/benchmarks.txt"
echo "N=${N}: run=${RUN_ID}, cpus=${SLURM_CPUS_PER_TASK}, output=${OUTPUT_NAME}, time=${elapsed}" >> "$RESULT_FILE"

# ---------------------------
# COPY OUTPUT BACK
# ---------------------------
mkdir -p "$PROJECT_HOME/data" "$PROJECT_HOME/visualizations"
rsync -a data/ "$PROJECT_HOME/data/"
rsync -a visualizations/ "$PROJECT_HOME/visualizations/"

echo "Task $SLURM_ARRAY_TASK_ID done."
