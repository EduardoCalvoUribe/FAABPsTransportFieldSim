#!/bin/bash
#SBATCH --job-name=FAABPs_wallprune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --partition=rome
# G=20 perfect maze: M = (G-1)^2 = 361 interior walls.
# Pruning schedule doubles each run: 0,1,2,4,8,16,32,64,128,256,361 → 11 tasks.
#SBATCH --array=0-10

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
# EXPERIMENT CONFIGURATION
# Each array task is one prune_step (number of interior walls removed).
# prune_step=0: full maze; prune_step=361: all interior walls removed.
# The pruning order is fixed by PRUNE_SEED so every task is reproducible.
# ---------------------------
MAZE_GRID_SIZE=20
PRUNE_SEED=7

# Doubling schedule: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 361 (=M)
PRUNE_STEPS=(0 1 2 4 8 16 32 64 128 256 361)
PRUNE_STEP=${PRUNE_STEPS[$SLURM_ARRAY_TASK_ID]}
PRUNE_STEP_PAD=$(printf "%03d" "$PRUNE_STEP")
OUTPUT_NAME="wallprune_G${MAZE_GRID_SIZE}_step${PRUNE_STEP_PAD}"

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
mkdir -p "$PROJECT_HOME/logsWallPrune"
LOG_FILE="$PROJECT_HOME/logsWallPrune/task_${SLURM_ARRAY_TASK_ID}.txt"

echo "Task $SLURM_ARRAY_TASK_ID: G=$MAZE_GRID_SIZE prune_step=$PRUNE_STEP output=$OUTPUT_NAME" | tee -a "$LOG_FILE"
echo "Running from: $(pwd)"

start=$(date +%s.%N)
python fig6/main_fig6.py "$MAZE_GRID_SIZE" "$PRUNE_STEP" "$OUTPUT_NAME" \
  --prune-seed "$PRUNE_SEED" >> "$LOG_FILE" 2>&1
end=$(date +%s.%N)
elapsed=$(awk "BEGIN {print $end - $start}")

echo "G=${MAZE_GRID_SIZE}, prune_step=${PRUNE_STEP}, elapsed: $elapsed seconds" >> "$LOG_FILE"

# ---------------------------
# SHARED RESULT LINE
# Grep the machine-readable summary from the log and append to results file.
# Short appends are atomic on ext4/Lustre; safe for one line per task.
# ---------------------------
RESULT_FILE="$PROJECT_HOME/wallprune_results.txt"
grep "RESULT_LINE:" "$LOG_FILE" | tail -1 >> "$RESULT_FILE" || true
echo "elapsed=${elapsed}s task=${SLURM_ARRAY_TASK_ID}" >> "$RESULT_FILE"

# ---------------------------
# COPY OUTPUT BACK
# ---------------------------
mkdir -p "$PROJECT_HOME/data" "$PROJECT_HOME/visualizations"
rsync -a data/ "$PROJECT_HOME/data/"
rsync -a visualizations/ "$PROJECT_HOME/visualizations/"

echo "Task $SLURM_ARRAY_TASK_ID done."
