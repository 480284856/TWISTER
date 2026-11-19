"""Memory Maze test with the real memory_maze package (pip install memory-maze gym; macOS: MUJOCO_GL=glfw is set by the env).

    A. Real 15x15 episode: length, observations, rewards, is_last / done, env step time
    B. End to end on CPU with real 9x9 episodes: env steps -> EpisodeReplayBuffer (complete 1001 step episodes)
       -> TWISTER train steps. Small "S" model and batch_size 1 to fit a CPU smoke test, all memory options on.

Run from the repository root:
    python tests/test_memory_maze_real.py
"""

import os
import sys
import types
import time
import tempfile
import importlib
import importlib.machinery

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stub(name, path=None, **attrs):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=path is not None)
    if path is not None:
        module.__path__ = [path]
    module.__dict__.update(attrs)
    sys.modules[name] = module


def stub_if_missing(name, **attrs):
    try:
        importlib.import_module(name)
    except ImportError:
        stub(name, **attrs)


# nnet without nnet/__init__.py, nnet.envs without the dmc / atari envs (only wrappers + memory_maze)
stub("nnet", path=os.path.join(ROOT, "nnet"))
stub("nnet.envs", path=os.path.join(ROOT, "nnet", "envs"))
stub_if_missing("torchvision", transforms=types.SimpleNamespace(RandomResizedCrop=lambda **kwargs: (lambda x: x)), io=types.SimpleNamespace(write_video=None))
stub_if_missing("torch.utils.tensorboard", SummaryWriter=object)
stub_if_missing("tqdm", tqdm=lambda x, **kwargs: x)
stub_if_missing("wandb")

torch.distributions.Distribution.set_default_validate_args(False)

import nnet.envs.wrappers  # noqa: E402,F401
import nnet.envs.memory_maze  # noqa: E402,F401
from nnet.envs.memory_maze import MemoryMazeEnv  # noqa: E402
from nnet.models.twister import TWISTER  # noqa: E402
from nnet.datasets.episode_replay_buffer import EpisodeReplayBuffer  # noqa: E402

RESULTS = []


def report(ok, name, detail):
    RESULTS.append(bool(ok))
    print("{} {:<30s} {}".format("OK  " if ok else "FAIL", name, detail), flush=True)


def test_real_episode(maze_size=15):
    env = MemoryMazeEnv(maze_size=maze_size, seed=0)
    obs = env.reset()
    ok = obs.state.dtype == torch.uint8 and tuple(obs.state.shape) == (3, 64, 64) and obs.is_first == 1
    steps, total_reward, t0 = 0, 0.0, time.time()
    while True:
        obs = env.step(env.sample().argmax())
        steps += 1
        total_reward += obs.reward.item()
        ok &= obs.done == 0 and obs.is_first == 0
        if obs.is_last == 1:
            break
    dt = (time.time() - t0) / steps
    pixels_vary = obs.state.float().std().item() > 1.0
    report(ok and steps == env.episode_length == 4000 and pixels_vary, "A. real episode {}x{}".format(maze_size, maze_size),
           "{} steps (is_last=1, done=0), random policy reward {:.0f}, {:.1f} ms / env step".format(steps, total_reward, 1000 * dt))
    return dt


def test_end_to_end(train_steps=2):
    torch.manual_seed(0)
    model = TWISTER(env_name="memory_maze-9x9", override_config={
        "model_size": "S", "batch_size": 1, "num_envs": 1, "eval_episodes": 0, "precision": "float32", "epoch_length": 2, "pre_fill_steps": 1,
        "att_query_chunk_size": 128, "cnn_checkpoint_chunk_size": 256, "contrastive_chunk_size": 512, "imag_start_chunk_size": 256,
    })
    c = model.config
    buffer = EpisodeReplayBuffer(batch_size=c.batch_size, root=tempfile.mkdtemp(), buffer_capacity=c.buffer_capacity, epoch_length=c.epoch_length, sample_length=c.L, save_trajectories=False)
    model.compile()
    model.set_replay_buffer(buffer)

    t0 = time.time()
    model.on_train_begin()
    prefill_time = time.time() - t0

    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    model.train()
    finite, aligned, step_times = True, True, []
    for _ in range(train_steps):
        batch = buffer.collate_fn([buffer[i] for i in range(c.batch_size)])
        inputs = batch["inputs"]
        states, is_firsts = inputs[0], inputs[4]
        aligned &= states.shape[1] == c.L == 1001 and bool((is_firsts[:, 0] == 1).all()) and bool((is_firsts[:, 1:] == 0).all())
        t0 = time.time()
        losses, _, _ = model.train_step(inputs=inputs, targets=batch["targets"], precision=torch.float32, grad_scaler=grad_scaler, accumulated_steps=1, acc_step=0, eval_training=False)
        step_times.append(time.time() - t0)
        finite &= all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values())

    info = buffer.get_infos()
    report(finite and aligned and info["num_sample_episodes"] >= 1, "B. end to end real 9x9 (CPU)",
           "prefill {:.0f}s ({} complete episodes), samples = full 1001 step episodes: {}, {} train steps, losses finite: {} ({:.0f}s / step incl. env steps), world model loss {:.1f}, actor {:.3f}, critic {:.3f}".format(
               prefill_time, info["num_episodes"], aligned, train_steps, finite, sum(step_times) / len(step_times),
               float(losses["world_model_loss"]), float(losses["actor_model_loss"]), float(losses["critic_model_loss"])))


if __name__ == "__main__":
    test_real_episode()
    test_end_to_end()
    print("\n{} / {} passed".format(sum(RESULTS), len(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
