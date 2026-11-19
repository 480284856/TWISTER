#!/bin/bash
# Verify the uv environment on a GPU node: EGL rendering + Memory Maze tests (real env, 32 parallel envs, CPU train steps).
# Submit from the repository root, after `bash install_uv.sh` on the login node:
#   sbatch slurm/verify_memory_maze.sbatch
#
# Julia 2 (Uni Wuerzburg, https://doku.hpc.uni-wuerzburg.de/partitions/): GPU jobs need a GPU partition (standard: L40 / L40s,
# 16 threads per GPU, max 1 day). Without -p the job goes to small_cpu (no GPUs): "QOSMaxGRESPerJob".
# Check before submitting: sbatch --test-only slurm/verify_memory_maze.sbatch
#
#SBATCH --job-name=twister-mm-verify
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
##SBATCH --qos=normal               # if the monthly GPU hour budget is used up (QOSGrpGRESMinutes error)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f main.py ] || { echo "submit from the TWISTER repository root (main.py not found in $(pwd))"; exit 1; }

# Envs of the parallel env speed test = allocated CPUs (at most 32)
export TEST_NUM_ENVS="${TEST_NUM_ENVS:-$(( ${SLURM_CPUS_PER_TASK:-32} < 32 ? ${SLURM_CPUS_PER_TASK:-32} : 32 ))}"

bash install_uv.sh --verify-only --test
