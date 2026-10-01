#!/bin/bash
#SBATCH --job-name=FAABPs_maze
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=rome
#SBATCH --array=0-39

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
# TASK → (MAZE_GRID_SIZE, run_id) MAPPING
# 7 maze sizes × 5 runs = 35 tasks (0-34)
# G=2:  N=40,    BOX=120
# G=5:  N=250,   BOX=300
# G=10: N=1000,  BOX=600
# G=20: N=4000,  BOX=1200
# G=30: N=9000,  BOX=1800
# G=40: N=16000, BOX=2400
# G=50: N=25000, BOX=3000
# ---------------------------
MAZE_LIST=(2 2 2 2 2 5 5 5 5 5 10 10 10 10 10 20 20 20 20 20 30 30 30 30 30 40 40 40 40 40 50 50 50 50 50 60 60 60 60 60)
G=${MAZE_LIST[$SLURM_ARRAY_TASK_ID]}
RUN_ID=$(( SLURM_ARRAY_TASK_ID % 5 + 1 ))
OUTPUT_NAME="maze_G${G}_run${RUN_ID}"

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
mkdir -p "$PROJECT_HOME/logs"
LOG_FILE="$PROJECT_HOME/logs/task_${SLURM_ARRAY_TASK_ID}.txt"

echo "Task $SLURM_ARRAY_TASK_ID: G=$G run=$RUN_ID output=$OUTPUT_NAME" | tee -a "$LOG_FILE"
echo "Running from: $(pwd)"

start=$(date +%s.%N)
python main.py "$G" "$OUTPUT_NAME" >> "$LOG_FILE" 2>&1
end=$(date +%s.%N)
elapsed=$(awk "BEGIN {print $end - $start}")

echo "G=${G}, run=${RUN_ID}, elapsed: $elapsed seconds" >> "$LOG_FILE"

# ---------------------------
# SHARED RESULT LINE
# Short appends are atomic on ext4/Lustre; safe for one line per task.
# ---------------------------
RESULT_FILE="$PROJECT_HOME/benchmarks.txt"
echo "G=${G}: run=${RUN_ID}, cpus=${SLURM_CPUS_PER_TASK}, output=${OUTPUT_NAME}, time=${elapsed}" >> "$RESULT_FILE"

# ---------------------------
# COPY OUTPUT BACK
# ---------------------------
mkdir -p "$PROJECT_HOME/data" "$PROJECT_HOME/visualizations"
rsync -a data/ "$PROJECT_HOME/data/"
rsync -a visualizations/ "$PROJECT_HOME/visualizations/"

echo "Task $SLURM_ARRAY_TASK_ID done."
