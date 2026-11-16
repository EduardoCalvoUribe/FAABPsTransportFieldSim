#!/bin/bash
#SBATCH --job-name=FAABPs_mazescale
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=120:00:00
#SBATCH --partition=rome
# 9 maze sizes x 10 unique mazes = 90 tasks (task = size_index*10 + run-1)
#
# CPUs are sized per group of tasks; command-line options override the #SBATCH
# defaults, so submit the groups separately:
#   sbatch --cpus-per-task=4  --array=0-19  --time=12:00:00  snellius_maze_scaling_v2.sh   # G=2,5
#   sbatch --cpus-per-task=8  --array=20-39 --time=48:00:00  snellius_maze_scaling_v2.sh   # G=10,20
#   sbatch --cpus-per-task=16 --array=40-49 --time=120:00:00 snellius_maze_scaling_v2.sh   # G=30
#   sbatch --cpus-per-task=32 --array=50-89 --time=120:00:00 snellius_maze_scaling_v2.sh   # G=35,40,45,50
# (Plain `sbatch snellius_maze_scaling_v2.sh` would run only the defaults below:
#  tasks 0-89 on 32 CPUs.)
#SBATCH --array=0-89

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
# TASK MAPPING (size-major: 10 mazes per size)
#   G in 2 5 10 20 30 35 40 45 50
#   RUN_ID 1..10, each with its own maze seed
# ---------------------------
GRIDS=(2 5 10 20 30 35 40 45 50)
RUNS_PER_G=10

# Max steps per G = 2 x (upper envelope of the log-log fit to figures/fig6/d data).
# Fit: steps ~ 10^3.5056 * area^0.8966 (area = G^2); the envelope is scaled by 4.08,
# the largest observed trial/fit ratio (the G=30 outlier, 5.83M steps).
# Rounded up. END_WHEN_GOAL_REACHED=True, so this is only a safety cap.
declare -A MAX_STEPS=(
  [2]=100000
  [5]=500000
  [10]=1700000
  [20]=5700000
  [30]=11700000
  [35]=15400000
  [40]=19600000
  [45]=24200000
  [50]=29200000
)

T=$SLURM_ARRAY_TASK_ID
G=${GRIDS[$(( T / RUNS_PER_G ))]}
RUN_ID=$(( T % RUNS_PER_G + 1 ))
# Unique, reproducible maze seed per (G, run). 42 (the old fixed seed) is never used.
MAZE_SEED=$(( 100000 + G * 100 + RUN_ID ))
N_STEPS=${MAX_STEPS[$G]}

PROJECT_HOME="$HOME/FAABPsTransportFieldSim"
OUTPUT_NAME="mazescale_G${G}_run${RUN_ID}_seed${MAZE_SEED}"
CMD=(python main.py "$G" "$OUTPUT_NAME" --maze-seed "$MAZE_SEED" --n-steps "$N_STEPS" --no-shortcut)
LOG_DIR="$PROJECT_HOME/logsMazeScale"
DESC="maze G=$G run=$RUN_ID maze_seed=$MAZE_SEED n_steps=$N_STEPS"

# ---------------------------
# ISOLATED NODE-LOCAL SCRATCH PER TASK (cleaned up on exit)
# ---------------------------
SCRATCH_ROOT="${TMPDIR:-/tmp}/FAABPs_${SLURM_ARRAY_JOB_ID}_${T}"
PROJECT_SCRATCH="$SCRATCH_ROOT/project"
trap 'rm -rf "$SCRATCH_ROOT"' EXIT

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
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/mazescale_${SLURM_ARRAY_JOB_ID}_task_${T}.txt"

echo "Task $T: $DESC output=$OUTPUT_NAME" | tee -a "$LOG_FILE"
echo "Running from: $(pwd)"

start=$(date +%s.%N)
"${CMD[@]}" >> "$LOG_FILE" 2>&1
end=$(date +%s.%N)
elapsed=$(awk "BEGIN {print $end - $start}")

echo "$DESC, elapsed: $elapsed seconds" >> "$LOG_FILE"

# ---------------------------
# SHARED RESULT LINES (single short append per task)
# ---------------------------
echo "G=${G}: run=${RUN_ID}, maze_seed=${MAZE_SEED}, cpus=${SLURM_CPUS_PER_TASK}, output=${OUTPUT_NAME}, time=${elapsed}" \
  >> "$PROJECT_HOME/benchmarks.txt"
# Machine-readable line for figures/fig6/d/plot_maze_scaling.py.
# final_step == n_steps means the cap was hit (goal NOT reached).
FINAL_STEP=$(grep -E "^Final step:" "$LOG_FILE" | tail -1 | awk '{print $3}')
echo "MAZE_RESULT: G=${G} run=${RUN_ID} maze_seed=${MAZE_SEED} final_step=${FINAL_STEP:-NA} n_steps=${N_STEPS} output=${OUTPUT_NAME}" \
  >> "$PROJECT_HOME/maze_results_v2.txt"

# ---------------------------
# COPY OUTPUT BACK
# ---------------------------
mkdir -p "$PROJECT_HOME/data" "$PROJECT_HOME/visualizations"
rsync -a data/ "$PROJECT_HOME/data/"
rsync -a visualizations/ "$PROJECT_HOME/visualizations/"

echo "Task $T done."
