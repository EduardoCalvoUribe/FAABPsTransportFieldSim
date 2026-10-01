#!/bin/bash
#SBATCH --job-name=FAABPs_optpath
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --partition=rome
#SBATCH --array=0-5

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
# Maze grid size is fixed; each array task is one independent repeat.
# ---------------------------
MAZE_GRID_SIZE=10
RUN_ID=$(( SLURM_ARRAY_TASK_ID + 1 ))
OUTPUT_NAME="optimal_G${MAZE_GRID_SIZE}_run${RUN_ID}"

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
mkdir -p "$PROJECT_HOME/logsOptPath"
LOG_FILE="$PROJECT_HOME/logsOptPath/task_${SLURM_ARRAY_TASK_ID}.txt"

echo "Task $SLURM_ARRAY_TASK_ID: MAZE_GRID_SIZE=$MAZE_GRID_SIZE run=$RUN_ID output=$OUTPUT_NAME" | tee -a "$LOG_FILE"
echo "Running from: $(pwd)"

start=$(date +%s.%N)
python main.py "$MAZE_GRID_SIZE" "$OUTPUT_NAME" >> "$LOG_FILE" 2>&1
end=$(date +%s.%N)
elapsed=$(awk "BEGIN {print $end - $start}")

echo "MAZE_GRID_SIZE=${MAZE_GRID_SIZE}, run=${RUN_ID}, elapsed: $elapsed seconds" >> "$LOG_FILE"

# ---------------------------
# SHARED RESULT LINE
# Short appends are atomic on ext4/Lustre; safe for one line per task.
# ---------------------------
RESULT_FILE="$PROJECT_HOME/optimal_path_results.txt"
echo "G=${MAZE_GRID_SIZE}: run=${RUN_ID}, cpus=${SLURM_CPUS_PER_TASK}, output=${OUTPUT_NAME}, time=${elapsed}" >> "$RESULT_FILE"

# ---------------------------
# COPY OUTPUT BACK
# Includes simulation data, the path-analysis text file, and the maze PNG
# (both written by optimal_path.py at the start of each run).
# ---------------------------
mkdir -p "$PROJECT_HOME/data" "$PROJECT_HOME/visualizations"
rsync -a data/ "$PROJECT_HOME/data/"
rsync -a visualizations/ "$PROJECT_HOME/visualizations/"

# optimal_path analysis files are written to the scratch project root
cp -f "optimal_path_G${MAZE_GRID_SIZE}.txt" "$PROJECT_HOME/" 2>/dev/null || true
cp -f "optimal_path_G${MAZE_GRID_SIZE}.png" "$PROJECT_HOME/" 2>/dev/null || true

echo "Task $SLURM_ARRAY_TASK_ID done."
