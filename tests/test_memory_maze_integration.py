"""Memory Maze integration test with a fake memory_maze package (no MuJoCo needed).

    A. MemoryMazeEnv: observations, discrete actions, episode end at the fixed episode length (is_last=1, done=0)
    B. TWISTER memory_maze-<n>x<n> configuration for the 4 maze sizes (TT-MBRL Table 15b / 17, full episode context)
    C. End to end: env steps -> EpisodeReplayBuffer -> TWISTER train steps (world model, actor, critic) with all
       memory options enabled, replay ratio kept at env_step_period

Run from the repository root:
    python tests/test_memory_maze_integration.py
"""

import os
import sys
import types
import tempfile
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
    return module


# nnet without nnet/__init__.py, nnet.envs without dm_control / gym (only wrappers + memory_maze are imported)
stub("nnet", path=os.path.join(ROOT, "nnet"))
stub("nnet.envs", path=os.path.join(ROOT, "nnet", "envs"))
stub("torchvision", transforms=types.SimpleNamespace(RandomResizedCrop=lambda **kwargs: (lambda x: x)), io=types.SimpleNamespace(write_video=None))
stub("torch.utils.tensorboard", SummaryWriter=object)
stub("tqdm", tqdm=lambda x, **kwargs: x)
stub("wandb")


###############################################################################
# Fake memory_maze package (dm_env interface of memory_maze.tasks)
###############################################################################

FAKE_EPISODE_LENGTH = {}  # maze_size -> override of the episode length (for a fast end to end test)
CREATED = []


class FakeArray:
    # numpy like observation (process_obs calls .copy())
    def __init__(self, tensor):
        self.tensor = tensor

    def copy(self):
        return self.tensor.clone()


class FakeTimeStep:
    def __init__(self, first, last, observation, reward, discount):
        self._first, self._last = first, last
        self.observation, self.reward, self.discount = observation, reward, discount

    def first(self):
        return self._first

    def last(self):
        return self._last


class FakeMazeEnv:

    def __init__(self, maze_size, time_limit_steps, seed=None, camera_resolution=64, **kwargs):
        self.maze_size, self.length, self.res = maze_size, time_limit_steps, camera_resolution
        self.generator = torch.Generator().manual_seed(seed or 0)
        CREATED.append(self)

    def action_spec(self):
        return types.SimpleNamespace(num_values=6)

    def obs(self):
        return {"image": FakeArray(torch.randint(0, 256, (self.res, self.res, 3), dtype=torch.uint8, generator=self.generator))}

    def reset(self):
        self.t = 0
        return FakeTimeStep(True, False, self.obs(), None, None)

    def step(self, action):
        assert isinstance(action, int) and 0 <= action < 6
        self.t += 1
        last = self.t >= self.length
        return FakeTimeStep(False, last, self.obs(), float(torch.rand(1, generator=self.generator) < 0.05), 1.0)


def make_task(maze_size):
    def task(seed=None, camera_resolution=64, **kwargs):
        length = FAKE_EPISODE_LENGTH.get(maze_size, {9: 1000, 11: 2000, 13: 3000, 15: 4000}[maze_size])
        return FakeMazeEnv(maze_size, length, seed=seed, camera_resolution=camera_resolution, **kwargs)
    return task


stub("memory_maze", path="")
stub("memory_maze.tasks", **{"memory_maze_{}x{}".format(n, n): make_task(n) for n in (9, 11, 13, 15)})
sys.modules["memory_maze"].tasks = sys.modules["memory_maze.tasks"]

torch.distributions.Distribution.set_default_validate_args(False)

import nnet.envs.wrappers  # noqa: E402,F401
import nnet.envs.memory_maze  # noqa: E402,F401
from nnet.envs.memory_maze import MemoryMazeEnv  # noqa: E402
from nnet.models.twister import TWISTER  # noqa: E402
from nnet.datasets.episode_replay_buffer import EpisodeReplayBuffer  # noqa: E402

RESULTS = []


def report(ok, name, detail):
    RESULTS.append(bool(ok))
    print("{} {:<34s} {}".format("OK  " if ok else "FAIL", name, detail))


###############################################################################
# A. Env
###############################################################################

def test_env():
    env = MemoryMazeEnv(maze_size=9)
    obs = env.reset()
    ok = obs.state.dtype == torch.uint8 and tuple(obs.state.shape) == (3, 64, 64) and obs.is_first == 1 and obs.is_last == 0
    ok &= env.num_actions == 6 and env.sample().shape == (6,) and env.sample().sum() == 1
    steps = 0
    while True:
        obs = env.step(torch.tensor(steps % 6))
        steps += 1
        ok &= obs.done == 0 and obs.is_first == 0
        if obs.is_last == 1:
            break
        assert steps < 5000
    report(ok and steps == 1000, "A. MemoryMazeEnv 9x9", "episode ends after {} steps (is_last=1, done=0), obs uint8 (3, 64, 64), 6 actions".format(steps))


###############################################################################
# B. Configuration per maze size
###############################################################################

def test_configs():
    for n, E in ((9, 1000), (11, 2000), (13, 3000), (15, 4000)):
        model = TWISTER(env_name="memory_maze-{}x{}".format(n, n), override_config={"num_envs": 1, "eval_episodes": 0, "parallel_envs": False})
        c = model.config
        total_env_steps = c.epochs * c.epoch_length * c.batch_size * c.L / c.env_step_period
        rssm_att = model.rssm.transformer.blocks[0].self_att_module.attention
        long_ctx = c.L >= 3000
        ok = (c.L == E + 1 and c.att_context_left == E + 1 and c.batch_size == 4 and c.replay_buffer_mode == "episode"
              and c.time_limit == E and c.env_step_period == 64 and c.buffer_capacity == int(10e6) and c.policy_discrete
              and c.precision == torch.float16 and c.att_max_pos_encoding == max(2048, E + 2)
              and abs(total_env_steps / 100e6 - 1) < 1e-3
              and len(model.rssm.transformer.blocks) == 5 and rssm_att.num_heads == 10 and model.rssm.hidden_size == 640
              and model.encoder_network.dim_cnn == 48 and rssm_att.rel_pos_enc.max_len == c.att_max_pos_encoding
              and (c.att_query_chunk_size == 256 if long_ctx else c.att_query_chunk_size is None)
              and (c.cnn_checkpoint_chunk_size == 1024 if long_ctx else c.cnn_checkpoint_chunk_size is None))
        num_params = sum(p.numel() for p in model.parameters())
        report(ok, "B. config memory_maze-{}x{}".format(n, n),
               "L=C={}, B={}, epochs x epoch_length = {} x {} -> {:.1f}M env steps, max_pos {}, chunks {}/{}, {:.1f}M params".format(
                   c.L, c.batch_size, c.epochs, c.epoch_length, total_env_steps / 1e6, c.att_max_pos_encoding, c.att_query_chunk_size, c.cnn_checkpoint_chunk_size, num_params / 1e6))
        del model


###############################################################################
# C. End to end
###############################################################################

def test_end_to_end(E=16, num_envs=2, train_steps=6):

    FAKE_EPISODE_LENGTH[9] = E
    torch.manual_seed(0)
    model = TWISTER(env_name="memory_maze-9x9", override_config={
        "L": E + 1, "att_context_left": E + 1, "time_limit": E, "time_limit_eval": E, "num_envs": num_envs, "eval_episodes": 0,
        "precision": "float32", "epoch_length": 10, "pre_fill_steps": 1, "parallel_envs": False,  # env steps are counted on the in-process fake envs (parallel envs: tests/test_parallel_envs.py)
        # all memory options on, small chunks
        "att_query_chunk_size": 6, "cnn_checkpoint_chunk_size": 16, "contrastive_chunk_size": 20, "imag_start_chunk_size": 5,
    })
    c = model.config

    # Same construction as configs/twister.py
    replay_buffer_class = {"window": EpisodeReplayBuffer.__bases__[0], "episode": EpisodeReplayBuffer}[c.replay_buffer_mode]
    buffer = replay_buffer_class(batch_size=c.batch_size, root=tempfile.mkdtemp(), buffer_capacity=c.buffer_capacity, epoch_length=c.epoch_length, sample_length=c.L, save_trajectories=False)
    model.compile()  # creates model_step, before set_replay_buffer like configs/twister.py
    model.set_replay_buffer(buffer)
    model.on_train_begin()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    model.train()

    steps_before = sum(env.t for env in CREATED[-num_envs:]) + model.episodes.item() * E
    finite, aligned = True, True
    for _ in range(train_steps):
        batch = buffer.collate_fn([buffer[i] for i in range(c.batch_size)])
        inputs = batch["inputs"]
        aligned &= bool((inputs[4][:, 0] == 1).all()) and bool((inputs[4][:, 1:] == 0).all()) and inputs[0].shape[1] == c.L
        losses, _, _ = model.train_step(inputs=inputs, targets=batch["targets"], precision=torch.float32, grad_scaler=grad_scaler, accumulated_steps=1, acc_step=0, eval_training=False)
        finite &= all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values())
    env_steps = sum(env.t for env in CREATED[-num_envs:]) + model.episodes.item() * E - steps_before
    expected = train_steps * c.batch_size * c.L / c.env_step_period

    ok = finite and aligned and abs(env_steps - expected) < num_envs and buffer.can_sample()
    report(ok, "C. end to end (fake env)",
           "{} train steps, losses finite: {}, samples = full episodes: {}, env steps {} (expected {:.1f}), losses: {}".format(
               train_steps, finite, aligned, env_steps, expected, ", ".join("{} {:.3f}".format(k, float(v)) for k, v in losses.items() if "loss" in k)))


if __name__ == "__main__":
    test_env()
    test_configs()
    test_end_to_end()
    print("\n{} / {} passed".format(sum(RESULTS), len(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
