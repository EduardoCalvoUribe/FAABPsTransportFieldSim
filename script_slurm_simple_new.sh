#!/bin/bash
#SBATCH --job-name=FAABPs
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=72:00:00
#SBATCH --array=0-5                  # 3 replicas × 2 halves = 6 array tasks
#SBATCH --output=/home/vbekker/active_brawnian-curvity-faabp_clean/main_snellius/logs/abp_%A_%a.out
#SBATCH --error=/home/vbekker/active_brawnian-curvity-faabp_clean/main_snellius/logs/abp_%A_%a.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=vecchio.bekker@ru.nl


set -euo pipefail
export OMP_NUM_THREADS=1

ROOT_DIR="/home/vbekker/active_brawnian-curvity-faabp_clean"
BASEFOLDER="$ROOT_DIR/main_snellius/results"
LOGFOLDER="$ROOT_DIR/main_snellius/logs"
BINARY="$ROOT_DIR/target/release/matan_spp"

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found at $BINARY. Build with 'cargo build --release'." >&2
  exit 1
fi

SCRATCH_ROOT="${TMPDIR:-/tmp}/faabp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SCRATCH_RESULTS="$SCRATCH_ROOT/results"
SCRATCH_LOGS="$SCRATCH_ROOT/logs"
mkdir -p "$SCRATCH_RESULTS" "$SCRATCH_LOGS" "$BASEFOLDER" "$LOGFOLDER"

cp -p "$BINARY" "$SCRATCH_ROOT/"
RUN_BINARY="$SCRATCH_ROOT/matan_spp"

# --- Physics/config ---
TOTALTIME=100000
DUMPFREQ=10000
MEANSFREQ=10
CPUNUMBER=1

N=10000
Phi=0.4
InvPe=0.0
Ov=10.0

# 26 inv_rot values × 3 replicas = 78 runs. Split each replica into two halves of 13.
INVROT_LIST=(0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36 0.38 0.40 0.42 0.44 0.46 0.48 0.50)
HALF_SIZE=13

# Map array index: 0→(replica=1, first half), 1→(replica=1, second half), 2→(replica=2, first half), ...
replica=$(( SLURM_ARRAY_TASK_ID / 2 ))   # 0,1,2
half=$(( SLURM_ARRAY_TASK_ID % 2 ))      # 0 or 1
RUN_IDX=$(( replica + 1 ))               # 1..3
START_INVROT=$(( half * HALF_SIZE ))
END_INVROT=$(( START_INVROT + HALF_SIZE ))  # exclusive

export N Phi InvPe Ov TOTALTIME DUMPFREQ MEANSFREQ CPUNUMBER RUN_BINARY SCRATCH_RESULTS SCRATCH_LOGS RUN_IDX START_INVROT END_INVROT
INVROT_CSV=$(printf '%s,' "${INVROT_LIST[@]}"); INVROT_CSV=${INVROT_CSV%,}
export INVROT_CSV

# Launch 13 tasks in one step. each task picks one inv_rot from the chosen half.
srun --ntasks=13 --cpus-per-task=1 --cpu-bind=cores --kill-on-bad-exit=1 \
bash -lc '
  set -euo pipefail
  IFS=, read -r -a INVROT_ARR <<< "$INVROT_CSV"

  idx=$SLURM_PROCID
  param_idx=$(( START_INVROT + idx ))
  if (( param_idx >= END_INVROT )); then exit 0; fi

  invrotpeclet=${INVROT_ARR[$param_idx]}
  printf -v InvPeRot "%.4f" "$invrotpeclet"
  printf -v run_label "%03d" "$RUN_IDX"

  run_results_dir="${SCRATCH_RESULTS}/N${N}/TotalTime${TOTALTIME}/dt1e-2/InvRotPe${InvPeRot}/overtake${Ov}/Run${run_label}"
  run_logs_dir="${SCRATCH_LOGS}/N${N}/TotalTime${TOTALTIME}/dt1e-2/InvRotPe${InvPeRot}/overtake${Ov}/Run${run_label}"
  mkdir -p "$run_results_dir" "$run_logs_dir"

  "$RUN_BINARY" \
    "$N" "$Phi" "$InvPe" "$InvPeRot" "$Ov" 5.50 \
    "$TOTALTIME" "$DUMPFREQ" "$MEANSFREQ" "$CPUNUMBER" "$run_results_dir" \
    > "$run_logs_dir/output.log" 2>&1
'

# Persist outputs
rsync -a "$SCRATCH_RESULTS/" "$BASEFOLDER/"
rsync -a "$SCRATCH_LOGS/" "$LOGFOLDER/"