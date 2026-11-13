#!/bin/bash
#SBATCH --job-name=FAABPs_swarm_scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=rome
#SBATCH --array=0-233%15

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
# TASK → (G, N, run_id) MAPPING
# ---------------------------
# 3 maze sizes × 13 density-matched N values × 6 runs = 234 tasks
#
# Base N values from the 10×10 swarm-scaling experiment:
#   250 375 500 750 1000 1500 2000 3000 4000 6000 8000 12000 16000
# Density-matched: N_G = BASE_N * G² / 100  (integer arithmetic)
#   G=5:  ×0.25  →   62   93  125  187   250   375   500   750  1000  1500  2000   3000   4000
#   G=20: ×4.00  → 1000 1500 2000 3000  4000  6000  8000 12000 16000 24000 32000  48000  64000
#   G=30: ×9.00  → 2250 3375 4500 6750  9000 13500 18000 27000 36000 54000 72000 108000 144000
#
# Layout: 78 tasks per maze (13 N values × 6 runs), mazes in order [5, 20, 30]
# ---------------------------

MAZE_SIZES=(5 20 30)
BASE_N=(250 375 500 750 1000 1500 2000 3000 4000 6000 8000 12000 16000)

TASKS_PER_MAZE=78   # 13 N values × 6 runs
RUNS_PER_N=6

maze_idx=$(( SLURM_ARRAY_TASK_ID / TASKS_PER_MAZE ))
within=$(( SLURM_ARRAY_TASK_ID % TASKS_PER_MAZE ))
n_idx=$(( within / RUNS_PER_N ))
RUN_ID=$(( within % RUNS_PER_N + 1 ))

G=${MAZE_SIZES[$maze_idx]}
N=$(( ${BASE_N[$n_idx]} * G * G / 100 ))
OUTPUT_NAME="swarmscale_G${G}_N${N}_run${RUN_ID}"

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
mkdir -p "$PROJECT_HOME/logsSwarmScaling"
LOG_FILE="$PROJECT_HOME/logsSwarmScaling/task_${SLURM_ARRAY_TASK_ID}.txt"

echo "Task $SLURM_ARRAY_TASK_ID: G=$G N=$N run=$RUN_ID output=$OUTPUT_NAME" | tee -a "$LOG_FILE"
echo "Running from: $(pwd)"

start=$(date +%s.%N)
python main2.py "$G" "$OUTPUT_NAME" --n-particles "$N" >> "$LOG_FILE" 2>&1
end=$(date +%s.%N)
elapsed=$(awk "BEGIN {print $end - $start}")

echo "G=${G}, N=${N}, run=${RUN_ID}, elapsed: $elapsed seconds" >> "$LOG_FILE"

# ---------------------------
# SHARED RESULT LINE
# Short appends are atomic on ext4/Lustre; safe for one line per task.
# ---------------------------
RESULT_FILE="$PROJECT_HOME/benchmarks_swarm_scaling.txt"
echo "G=${G}: N=${N}, run=${RUN_ID}, cpus=${SLURM_CPUS_PER_TASK}, output=${OUTPUT_NAME}, time=${elapsed}" >> "$RESULT_FILE"

# ---------------------------
# COPY OUTPUT BACK
# ---------------------------
mkdir -p "$PROJECT_HOME/data" "$PROJECT_HOME/visualizations"
rsync -a data/ "$PROJECT_HOME/data/"
rsync -a visualizations/ "$PROJECT_HOME/visualizations/"

echo "Task $SLURM_ARRAY_TASK_ID done."
