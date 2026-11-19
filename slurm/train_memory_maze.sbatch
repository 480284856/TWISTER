#!/bin/bash
# Train TWISTER on Memory Maze (one GPU, 32 parallel env workers).
# Submit from the repository root:
#   sbatch slurm/train_memory_maze.sbatch                                   # 15x15
#   sbatch --export=ALL,MAZE=9x9 slurm/train_memory_maze.sbatch             # other maze sizes: 9x9, 11x11, 13x13
#   sbatch --export=ALL,OVERRIDE_CONFIG='{"epochs": 400}' slurm/train_memory_maze.sbatch
#   sbatch slurm/train_memory_maze.sbatch --wandb                           # extra arguments are passed to main.py
#
# The job always resumes from the last checkpoint (--load_last, model + replay buffer). Before the time limit,
# Slurm sends USR1 and the job requeues itself, so a multi week run continues across jobs. Work since the last
# checkpoint (saved every epoch) is lost on requeue: more epochs (OVERRIDE_CONFIG {"epochs": ...}) = more frequent
# checkpoints, the total number of env steps (100M) stays the same.
#
# Resources (Julia 2, Uni Wuerzburg, https://doku.hpc.uni-wuerzburg.de/partitions/):
#   partition  standard: L40 / L40s (48 GB), 16 threads per GPU, max 1 day per job -> requeued every day
#              (gpu_computervision: max 2 days, but jobs can be preempted by the computervision group)
#   CPUs       16 (per GPU limit) for 32 env worker processes + the trainer: env workers share cores (rendering runs on the GPU)
#   memory     replay buffer of 10M frames (64x64x3 uint8 = 12 KB each) ~ 115 GB RAM, + env workers / trainer
#   GPU hours  ~25 days of training: if the monthly GPU hour budget is used up (QOSGrpGRESMinutes), use --qos=normal
# Check before submitting: sbatch --test-only slurm/train_memory_maze.sbatch
#
#SBATCH --job-name=twister-mm
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
##SBATCH --qos=normal               # if the monthly GPU hour budget is used up (QOSGrpGRESMinutes error)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f main.py ] || { echo "submit from the TWISTER repository root (main.py not found in $(pwd))"; exit 1; }

MAZE="${MAZE:-9x9}"
RUN_NAME="${RUN_NAME:-memory_maze}"
EMPTY_CONFIG="{}"
export override_config="${OVERRIDE_CONFIG:-$EMPTY_CONFIG}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Trainer CPU threads = allocated CPUs (the node has many more cores than the job), env workers use 1 thread each
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

source .venv/bin/activate
echo "$(date) host $(hostname) job ${SLURM_JOB_ID:-local} restart ${SLURM_RESTART_COUNT:-0}: memory_maze-$MAZE, run $RUN_NAME, override_config $override_config"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Time limit: stop training and requeue the job (resumes from the last checkpoint)
requeue() {
    echo "$(date) time limit reached, requeue job ${SLURM_JOB_ID}"
    kill -TERM "$PID" 2>/dev/null || true
    wait "$PID" || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue USR1

env_name="memory_maze-$MAZE" run_name="$RUN_NAME" python3 main.py --load_last "$@" &
PID=$!
wait "$PID"
