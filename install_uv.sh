#!/usr/bin/env bash
# Install the TWISTER + Memory Maze environment with uv.
#
# Target: Linux GPU cluster (CUDA wheels, headless EGL rendering). Also works on macOS for CPU testing.
# On a Slurm cluster, install on the login node (no GPU needed), then verify on a GPU node:
#
#   bash install_uv.sh                          # login node: install, check imports (rendering check skipped without GPU)
#   sbatch slurm/verify_memory_maze.sbatch      # GPU node: EGL rendering + Memory Maze tests
#
# Usage:
#   bash install_uv.sh [--test] [--verify-only] [--no-verify]
#     --test           also run the Memory Maze tests (real env, parallel envs, CPU train steps), needs rendering
#     --verify-only    do not install, only verify an existing venv (e.g. inside a GPU job)
#     --no-verify      only install
#
# Options (environment variables):
#   VENV_DIR=/path/to/venv             # default: <repo>/.venv
#   PYTHON_VERSION=3.11                # 3.9 - 3.12 (labmaze, a memory-maze dependency, has no wheels for 3.13+)
#   CUDA=cu121                         # PyTorch wheel index: cu118 / cu121 / cu124 / cpu (ignored on macOS)
#   TORCH_VERSION=2.5.1 TORCHVISION_VERSION=0.20.1
#   TEST_NUM_ENVS=32                   # number of envs for the parallel env speed test
#
# After installation:
#   source <repo>/.venv/bin/activate
#   env_name=memory_maze-15x15 run_name=memory_maze python3 main.py      (or sbatch slurm/train_memory_maze.sbatch)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA="${CUDA:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TEST_NUM_ENVS="${TEST_NUM_ENVS:-32}"
RUN_TESTS=0
INSTALL=1
VERIFY=1

for arg in "$@"; do
    case "$arg" in
        --test) RUN_TESTS=1 ;;
        --verify-only) INSTALL=0 ;;
        --no-verify) VERIFY=0 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg (see --help)"; exit 1 ;;
    esac
done

step() { echo; echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }

OS="$(uname -s)"

###############################################################################
# Checks
###############################################################################

step "Checks"
command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
echo "uv: $(uv --version)"
echo "repository: $REPO_DIR"
echo "venv: $VENV_DIR (python $PYTHON_VERSION)"
echo "host: $(hostname)${SLURM_JOB_ID:+, Slurm job $SLURM_JOB_ID}"

case "$PYTHON_VERSION" in
    3.9*|3.10*|3.11*|3.12*) ;;
    *) echo "PYTHON_VERSION=$PYTHON_VERSION not supported: labmaze (memory-maze dependency) has no wheels for it, use 3.9 - 3.12"; exit 1 ;;
esac

# GPU visible on this machine (Linux): needed for headless EGL rendering, not for installing the CUDA wheels
GPU_VISIBLE=0
if [ "$OS" = "Linux" ]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
        GPU_VISIBLE=1
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/GPU: /'
    else
        echo "GPU: none visible on $(hostname) (login node?) - fine for installing, EGL rendering is checked on a GPU node"
    fi
fi

###############################################################################
# Install
###############################################################################

if [ "$INSTALL" = "1" ]; then

    step "Virtual environment"
    if [ -x "$VENV_DIR/bin/python" ]; then
        echo "reusing existing venv $VENV_DIR ($("$VENV_DIR/bin/python" --version))"
    else
        uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    fi

    step "PyTorch $TORCH_VERSION / torchvision $TORCHVISION_VERSION"
    if [ "$OS" = "Darwin" ]; then
        uv pip install --python "$VENV_DIR/bin/python" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
    else
        uv pip install --python "$VENV_DIR/bin/python" --index-url "https://download.pytorch.org/whl/$CUDA" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
    fi

    # gym is required: memory_maze/__init__.py re-raises the ImportError when gym is missing.
    # Atari (gym 0.19 + atari-py) is not installed: nnet.envs.atari is optional.
    step "Memory Maze and training dependencies"
    uv pip install --python "$VENV_DIR/bin/python" memory-maze gym dm_control tqdm wandb av tensorboard
    # gym prints a deprecation notice from the gym-notices package at every import (every env worker): gym skips it without the package
    uv pip uninstall --python "$VENV_DIR/bin/python" gym-notices 2>/dev/null || true
fi

PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "no venv at $VENV_DIR, run without --verify-only first"; exit 1; }

###############################################################################
# Verification
###############################################################################

if [ "$VERIFY" = "1" ]; then

    # Linux without GPU (login node): EGL cannot render (libEGL may not even be installed),
    # only check the Python packages with rendering disabled
    CHECK_RENDERING=1
    if [ "$OS" = "Linux" ] && [ "$GPU_VISIBLE" = "0" ] && [ "${MUJOCO_GL:-egl}" = "egl" ]; then
        CHECK_RENDERING=0
        export MUJOCO_GL=disable
    elif [ -z "${MUJOCO_GL:-}" ]; then
        if [ "$OS" = "Darwin" ]; then export MUJOCO_GL=glfw; else export MUJOCO_GL=egl; fi
    fi

    step "Verification (MUJOCO_GL=$MUJOCO_GL)"
    cd "$REPO_DIR"
    CHECK_RENDERING="$CHECK_RENDERING" "$PY" -W ignore - <<'EOF'
import os
import time
import torch
import nnet

print("torch {} (CUDA {}), GPU available: {}{}".format(torch.__version__, torch.version.cuda, torch.cuda.is_available(), ", " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))
print("envs: memory_maze ok, dm_control {}, atari {}".format(
    "ok" if not isinstance(nnet.envs.dm_control, nnet.envs.MissingEnvPackage) else "not available",
    "ok" if not isinstance(nnet.envs.atari, nnet.envs.MissingEnvPackage) else "not available (optional)"))
import memory_maze  # noqa: F401  (imports dm_control with the selected MUJOCO_GL backend)
print("memory_maze package ok")

if os.environ["CHECK_RENDERING"] == "1":
    from nnet.envs.memory_maze import MemoryMazeEnv
    try:
        env = MemoryMazeEnv(maze_size=15, seed=0)
        obs = env.reset()
    except Exception:
        print("Memory Maze rendering FAILED with MUJOCO_GL={}. On Linux GPU nodes EGL needs libEGL + the NVIDIA EGL vendor "
              "(libEGL_nvidia, /usr/share/glvnd/egl_vendor.d/10_nvidia.json); in Docker set NVIDIA_DRIVER_CAPABILITIES=all".format(os.environ.get("MUJOCO_GL")))
        raise
    assert obs.state.dtype == torch.uint8 and tuple(obs.state.shape) == (3, 64, 64)
    t0 = time.time()
    for t in range(200):
        obs = env.step(env.sample().argmax())
    print("Memory Maze 15x15: rendering ok, observation {} {}, {:.1f} ms / env step".format(tuple(obs.state.shape), obs.state.dtype, 1000 * (time.time() - t0) / 200))

    # OpenGL implementation used by MuJoCo: NVIDIA (GPU) or Mesa llvmpipe / softpipe (CPU software rendering, much slower)
    try:
        from dm_control import _render
        from OpenGL import GL
        context = _render.Renderer(max_width=64, max_height=64)
        with context.make_current() as ctx:
            vendor = ctx.call(GL.glGetString, GL.GL_VENDOR).decode()
            renderer = ctx.call(GL.glGetString, GL.GL_RENDERER).decode()
        context.free()
        print("OpenGL: vendor {}, renderer {}".format(vendor, renderer))
        if any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "software")):
            import glob
            vendor_files = sorted(glob.glob("/usr/share/glvnd/egl_vendor.d/*.json") + glob.glob("/etc/glvnd/egl_vendor.d/*.json"))
            print("WARNING: software rendering on the CPU (slow env steps). EGL vendor files found: {}. If an NVIDIA one exists, "
                  "export __EGL_VENDOR_LIBRARY_FILENAMES=<path to 10_nvidia.json>, otherwise ask the cluster admins for the NVIDIA EGL "
                  "library (libEGL_nvidia.so) on the GPU nodes.".format(vendor_files or "none"))
    except Exception as error:
        print("OpenGL vendor / renderer: could not query ({!r})".format(error))
else:
    print("Memory Maze rendering: skipped (no GPU visible for EGL), run on a GPU node: sbatch slurm/verify_memory_maze.sbatch")
EOF

    if [ "$RUN_TESTS" = "1" ]; then
        if [ "$CHECK_RENDERING" = "1" ]; then
            step "Tests"
            "$PY" -W ignore tests/test_memory_maze_real.py
            "$PY" -W ignore tests/test_memory_maze_fast_targets.py
            "$PY" -W ignore tests/test_parallel_envs.py "$TEST_NUM_ENVS"
        else
            warn "--test skipped: no GPU visible for EGL rendering, run on a GPU node: sbatch slurm/verify_memory_maze.sbatch"
        fi
    fi
fi

step "Done"
echo "Activate:  source $VENV_DIR/bin/activate"
echo "Verify:    sbatch slurm/verify_memory_maze.sbatch    (GPU node: rendering + tests)"
echo "Train:     sbatch slurm/train_memory_maze.sbatch     (or: env_name=memory_maze-15x15 run_name=memory_maze python3 main.py)"
