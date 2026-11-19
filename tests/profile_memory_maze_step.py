"""Where does a Memory Maze env step spend its time: physics, camera rendering, the rest (task / observables / Python).

Run from the repository root (GPU node on Linux: MUJOCO_GL=egl is the default):
    python tests/profile_memory_maze_step.py [maze_size] [steps] [fast]
    fast: use the vectorized target activation check of nnet.envs.memory_maze (as in training)
"""

import os
import sys
import time
import cProfile
import pstats

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else "egl"

maze_size = int(sys.argv[1]) if len(sys.argv) > 1 else 15
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200

import numpy as np  # noqa: E402
from memory_maze import tasks  # noqa: E402

fast = len(sys.argv) > 3 and sys.argv[3] == "fast"
if fast:
    # Same patch as MemoryMazeEnv(fast_target_check=True), imported without nnet/__init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("memory_maze_env", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nnet", "envs", "memory_maze", "memory_maze_env.py"))
    sys.modules.setdefault("nnet", type(sys)("nnet"))
    sys.modules.setdefault("nnet.structs", type(sys)("nnet.structs")).AttrDict = dict
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_target_sphere()
print("target activation check: {}".format("vectorized (patched)" if fast else "original dm_control loop"))

# Threads / CPUs seen by this process
affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
print("host {}, os.cpu_count {}, CPUs usable (affinity) {}, SLURM_CPUS_PER_TASK {}, OMP_NUM_THREADS {}, MKL_NUM_THREADS {}, OPENBLAS_NUM_THREADS {}".format(
    os.uname().nodename, os.cpu_count(), affinity, os.environ.get("SLURM_CPUS_PER_TASK"), os.environ.get("OMP_NUM_THREADS"),
    os.environ.get("MKL_NUM_THREADS"), os.environ.get("OPENBLAS_NUM_THREADS")))
try:
    import threadpoolctl
    for pool in threadpoolctl.threadpool_info():
        print("thread pool: {} {} threads".format(pool.get("internal_api"), pool.get("num_threads")))
except ImportError:
    pass

env = getattr(tasks, "memory_maze_{}x{}".format(maze_size, maze_size))(seed=0, image_only_obs=True)

# Underlying composer environment (physics, control timestep)
base = env
while not hasattr(base, "physics") and hasattr(base, "_env"):
    base = base._env
while not hasattr(base, "physics") and hasattr(base, "env"):
    base = base.env
# The maze (and the physics) is rebuilt at every reset: always use base.physics after a reset
env.reset()
physics = base.physics
n_sub = int(round(base.control_timestep() / physics.timestep()))
print("MUJOCO_GL {}, control timestep {} s, physics timestep {} s -> {} physics steps per env step, ngeom {}, nbody {}".format(
    os.environ["MUJOCO_GL"], base.control_timestep(), physics.timestep(), n_sub, physics.model.ngeom, physics.model.nbody))
rng = np.random.default_rng(0)

# 1. Full env step
t0 = time.perf_counter()
for _ in range(steps):
    timestep = env.step(int(rng.integers(6)))
    if timestep.last():
        env.reset()
t_step = (time.perf_counter() - t0) / steps

# 2. Physics only (same number of physics steps as one env step)
env.reset()
physics = base.physics
t0 = time.perf_counter()
for _ in range(steps):
    for _ in range(n_sub):
        physics.step()
t_physics = (time.perf_counter() - t0) / steps

# 3. Camera rendering only (64x64 egocentric camera)
cameras = [physics.model.id2name(i, "camera") for i in range(physics.model.ncam)]
camera = next((name for name in cameras if "egocentric" in name), 0)
print("cameras {}, rendering {}".format(cameras, camera))
t0 = time.perf_counter()
for _ in range(steps):
    physics.render(height=64, width=64, camera_id=camera)
t_render = (time.perf_counter() - t0) / steps

print("\nper env step: total {:.1f} ms = physics {:.1f} ms + rendering {:.1f} ms + rest {:.1f} ms".format(
    1000 * t_step, 1000 * t_physics, 1000 * t_render, 1000 * (t_step - t_physics - t_render)))
print("per physics step: {:.3f} ms".format(1000 * t_physics / n_sub))

# 4. Hotspots of the full env step
profiler = cProfile.Profile()
profiler.enable()
for _ in range(min(steps, 100)):
    timestep = env.step(int(rng.integers(6)))
    if timestep.last():
        env.reset()
profiler.disable()
print("\nTop functions by own time (100 env steps):")
pstats.Stats(profiler).sort_stats("tottime").print_stats(12)
