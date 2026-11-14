#!/bin/bash
#SBATCH --job-name=FAABPs_extra
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --partition=rome
# 5 (maze G=35, runs 1-5) + 44 (wall-pruning: 11 steps x 4 extra runs) = 49 tasks
#SBATCH --array=0-48

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
# TASK MAPPING
#   0-4   : maze scaling, G=35, run_id 1-5          (main.py)
#   5-48  : wall pruning, G=20, runs 2-5 (run 1 = existing data)
#           ordered run-major: 11 prune steps per run  (fig6/main_fig6.py)
# ---------------------------
T=$SLURM_ARRAY_TASK_ID
N_MAZE_TASKS=5

PROJECT_HOME="$HOME/FAABPsTransportFieldSim"

if (( T < N_MAZE_TASKS )); then
  EXPERIMENT=maze
  G=35
  RUN_ID=$(( T + 1 ))
  OUTPUT_NAME="maze_G${G}_run${RUN_ID}"
  CMD=(python main.py "$G" "$OUTPUT_NAME")
  LOG_DIR="$PROJECT_HOME/logs"
  DESC="maze G=$G run=$RUN_ID"
else
  EXPERIMENT=wallprune
  W=$(( T - N_MAZE_TASKS ))
  PRUNE_STEPS=(0 1 2 4 8 16 32 64 128 256 361)
  PRUNE_SEED=7                       # same pruning order as the original runs
  G=20
  PRUNE_STEP=${PRUNE_STEPS[$(( W % 11 ))]}
  RUN_ID=$(( W / 11 + 2 ))           # runs 2..5
  OUTPUT_NAME="wallprune_G${G}_step$(printf '%03d' "$PRUNE_STEP")_run${RUN_ID}"
  CMD=(python fig6/main_fig6.py "$G" "$PRUNE_STEP" "$OUTPUT_NAME" --prune-seed "$PRUNE_SEED")
  LOG_DIR="$PROJECT_HOME/logsWallPrune"
  DESC="wallprune G=$G step=$PRUNE_STEP run=$RUN_ID"
fi

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
LOG_FILE="$LOG_DIR/extra_${SLURM_ARRAY_JOB_ID}_task_${T}.txt"

echo "Task $T: $DESC output=$OUTPUT_NAME" | tee -a "$LOG_FILE"
echo "Running from: $(pwd)"

start=$(date +%s.%N)
"${CMD[@]}" >> "$LOG_FILE" 2>&1
end=$(date +%s.%N)
elapsed=$(awk "BEGIN {print $end - $start}")

echo "$DESC, elapsed: $elapsed seconds" >> "$LOG_FILE"

# ---------------------------
# SHARED RESULT LINE (single short append per task)
# ---------------------------
if [[ $EXPERIMENT == maze ]]; then
  echo "G=${G}: run=${RUN_ID}, cpus=${SLURM_CPUS_PER_TASK}, output=${OUTPUT_NAME}, time=${elapsed}" \
    >> "$PROJECT_HOME/benchmarks.txt"
  # Machine-readable line for figures/fig6/d/plot_maze_scaling.py
  FINAL_STEP=$(grep -E "^Final step:" "$LOG_FILE" | tail -1 | awk '{print $3}')
  echo "MAZE_RESULT: G=${G} run=${RUN_ID} final_step=${FINAL_STEP:-NA} output=${OUTPUT_NAME}" \
    >> "$PROJECT_HOME/maze_results.txt"
else
  {
    grep "RESULT_LINE:" "$LOG_FILE" | tail -1 | sed "s/\$/ run=${RUN_ID}/" || true
    echo "elapsed=${elapsed}s task=${T} run=${RUN_ID}"
  } >> "$PROJECT_HOME/wallprune_results.txt"
fi

# ---------------------------
# COPY OUTPUT BACK
# ---------------------------
mkdir -p "$PROJECT_HOME/data" "$PROJECT_HOME/visualizations"
rsync -a data/ "$PROJECT_HOME/data/"
rsync -a visualizations/ "$PROJECT_HOME/visualizations/"

echo "Task $T done."
