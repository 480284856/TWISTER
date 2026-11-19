"""ParallelBatchEnv test with the real memory_maze package.

    A. Equivalence: same seeds and actions -> identical observations, rewards, is_first / is_last as BatchEnv,
       across episode ends and resets (envs[i].reset(), eager reset in the workers)
    B. Speed: batched env step time, sequential BatchEnv vs ParallelBatchEnv (Memory Maze 15x15)
    C. TWISTER memory_maze-9x9 with parallel_envs=True: prefill + train steps

Run from the repository root:
    python tests/test_parallel_envs.py [num_envs_speed]
"""

import os
import sys
import time
import functools

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import test_memory_maze_real as real  # noqa: E402  (nnet stubs, also run in the spawned workers via this module)

import torch  # noqa: E402
from nnet.envs import wrappers  # noqa: E402
from nnet.envs.memory_maze import MemoryMazeEnv  # noqa: E402


def make_batch(parallel, maze_size, seeds, time_limit):
    fns = [functools.partial(wrappers.make_env, MemoryMazeEnv, {"maze_size": maze_size, "seed": seed}, time_limit=time_limit) for seed in seeds]
    return wrappers.ParallelBatchEnv(fns) if parallel else wrappers.BatchEnv([fn() for fn in fns])


def rollout(env, actions):
    # Same loop as TWISTER.env_step: batched steps, single env resets on is_last
    trace = [env.reset()]
    for a in actions:
        obs = env.step(a)
        trace.append(obs)
        for i in range(env.num_envs):
            if obs.is_last[i]:
                trace.append(("reset", i, env.envs[i].reset()))
    return trace


def test_equivalence(maze_size=9, num_envs=4, steps=1100):
    # Different time limits per run are not possible in one BatchEnv: use the env episode length (1000) -> one reset in the run
    seeds = list(range(num_envs))
    actions = [torch.randint(0, 6, (num_envs,), generator=torch.Generator().manual_seed(t)) for t in range(steps)]
    traces = {}
    for parallel in (False, True):
        env = make_batch(parallel, maze_size, seeds, time_limit=MemoryMazeEnv.EPISODE_LENGTHS[maze_size])
        traces[parallel] = rollout(env, actions)
        if parallel:
            env.close()

    def flat(trace):
        out = []
        for item in trace:
            obs = item[2] if isinstance(item, tuple) else item
            out.append(torch.cat([obs.state.flatten().float(), obs.reward.flatten(), obs.is_first.flatten(), obs.is_last.flatten(), obs.done.flatten()]))
        return out

    a, b = flat(traces[False]), flat(traces[True])
    resets = sum(isinstance(item, tuple) for item in traces[True])
    ok = len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b)) and resets == num_envs
    real.report(ok, "A. parallel == sequential", "{} envs x {} steps, {} resets, all observations / rewards / flags identical: {}".format(num_envs, steps, resets, ok))


def test_speed(maze_size=15, num_envs=8, steps=40):
    seeds = list(range(num_envs))
    times = {}
    for parallel in (False, True):
        env = make_batch(parallel, maze_size, seeds, time_limit=None)
        env.reset()
        env.step(torch.zeros(num_envs, dtype=torch.long))
        t0 = time.time()
        for t in range(steps):
            env.step(torch.randint(0, 6, (num_envs,)))
        times[parallel] = (time.time() - t0) / steps
        if parallel:
            env.close()
    real.report(times[True] < times[False], "B. speed {}x{}, {} envs".format(maze_size, maze_size, num_envs),
                "batched step: sequential {:.0f} ms, parallel {:.0f} ms ({:.1f}x), {} CPU cores".format(1000 * times[False], 1000 * times[True], times[False] / times[True], os.cpu_count()))


def test_twister(num_envs=4):
    import tempfile
    from nnet.models.twister import TWISTER
    from nnet.datasets.episode_replay_buffer import EpisodeReplayBuffer

    torch.manual_seed(0)
    model = TWISTER(env_name="memory_maze-9x9", override_config={
        "model_size": "S", "batch_size": 1, "num_envs": num_envs, "eval_episodes": 0, "precision": "float32", "epoch_length": 2, "pre_fill_steps": 1,
        "att_query_chunk_size": 128, "cnn_checkpoint_chunk_size": 256, "imag_start_chunk_size": 256,
    })
    c = model.config
    parallel = isinstance(model.env, wrappers.ParallelBatchEnv)
    buffer = EpisodeReplayBuffer(batch_size=c.batch_size, root=tempfile.mkdtemp(), buffer_capacity=c.buffer_capacity, epoch_length=c.epoch_length, sample_length=c.L, save_trajectories=False)
    model.compile()
    model.set_replay_buffer(buffer)
    t0 = time.time()
    model.on_train_begin()
    prefill = time.time() - t0
    model.train()
    batch = buffer.collate_fn([buffer[i] for i in range(c.batch_size)])
    losses, _, _ = model.train_step(inputs=batch["inputs"], targets=batch["targets"], precision=torch.float32, grad_scaler=torch.cuda.amp.GradScaler(enabled=False), accumulated_steps=1, acc_step=0, eval_training=False)
    finite = all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values())
    info = buffer.get_infos()
    model.env.close()
    real.report(parallel and finite and info["num_episodes"] == num_envs, "C. TWISTER parallel_envs",
                "ParallelBatchEnv: {}, prefill {:.0f}s -> {} complete episodes ({} envs), train step losses finite: {}".format(parallel, prefill, info["num_episodes"], num_envs, finite))


if __name__ == "__main__":
    test_equivalence()
    test_speed(num_envs=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    test_twister()
    print("\n{} / {} passed".format(sum(real.RESULTS), len(real.RESULTS)))
    sys.exit(0 if all(real.RESULTS) else 1)
