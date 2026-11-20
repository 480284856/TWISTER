"""Memory Maze 2D test with a fake memory_maze package (global observables, no MuJoCo needed).

    A. MemoryMaze2dEnv: observation layout, walls, rewards, new target, BFS expert reaches every target
    B. TWISTER memory_maze_2d-<n>x<n> configuration (same protocol as memory_maze-<n>x<n>, 5 actions)
    C. End to end: env steps -> EpisodeReplayBuffer -> TWISTER train steps

Run from the repository root:
    python tests/test_memory_maze_2d.py
"""

import os
import sys
import types
import tempfile
import importlib.machinery

import numpy as np
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


stub("nnet", path=os.path.join(ROOT, "nnet"))
stub("nnet.envs", path=os.path.join(ROOT, "nnet", "envs"))
stub("torch.utils.tensorboard", SummaryWriter=object)
stub("tqdm", tqdm=lambda x, **kwargs: x)
stub("wandb")


###############################################################################
# Fake memory_maze: random perfect maze (DFS) on odd cells, global observables
###############################################################################

class FakeTimeStep:
    def __init__(self, observation):
        self.observation = observation


class FakeMazeEnv:

    def __init__(self, maze_size, seed=None, **kwargs):
        assert kwargs.get("global_observables") and not kwargs.get("image_only_obs")
        self.n = maze_size
        self.rng = np.random.RandomState(seed or 0)
        self.num_targets = {9: 3, 11: 4, 13: 5, 15: 6}[maze_size]

    def reset(self):
        n, rng = self.n, self.rng
        layout = np.zeros((n, n), dtype=np.float64)
        cells = [(x, y) for y in range(0, n, 2) for x in range(0, n, 2)]
        start = cells[rng.randint(len(cells))]
        stack, seen = [start], {start}
        layout[start[1], start[0]] = 1
        while stack:
            x, y = stack[-1]
            nbrs = [(x + dx, y + dy) for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)) if 0 <= x + dx < n and 0 <= y + dy < n and (x + dx, y + dy) not in seen]
            if not nbrs:
                stack.pop()
                continue
            nx, ny = nbrs[rng.randint(len(nbrs))]
            layout[(y + ny) // 2, (x + nx) // 2] = 1
            layout[ny, nx] = 1
            seen.add((nx, ny))
            stack.append((nx, ny))
        free = [c for c in cells]
        pick = rng.choice(len(free), self.num_targets + 1, replace=False)
        agent = free[pick[0]]
        targets = [free[i] for i in pick[1:]]
        self.layout = layout
        # continuous grid coordinates (cell centers + 0.5), as memory_maze
        obs = {
            "maze_layout": layout,
            "agent_pos": np.array(agent, dtype=np.float64) + 0.5,
            "targets_pos": np.array(targets, dtype=np.float64) + 0.5,
            "image": np.zeros((16, 16, 3), dtype=np.uint8),
        }
        return FakeTimeStep(obs)


stub("memory_maze", path="")
stub("memory_maze.tasks", **{"memory_maze_{}x{}".format(n, n): (lambda n: lambda seed=None, **kw: FakeMazeEnv(n, seed=seed, **kw))(n) for n in (9, 11, 13, 15)})
sys.modules["memory_maze"].tasks = sys.modules["memory_maze.tasks"]

torch.distributions.Distribution.set_default_validate_args(False)

import nnet.envs.wrappers  # noqa: E402,F401
import nnet.envs.memory_maze  # noqa: E402,F401
from nnet.envs.memory_maze import MemoryMaze2dEnv  # noqa: E402

RESULTS = []


def report(ok, name, detail):
    RESULTS.append(bool(ok))
    print("{} {:<34s} {}".format("OK  " if ok else "FAIL", name, detail))


###############################################################################
# A. Env
###############################################################################

def test_env():
    env = MemoryMaze2dEnv(maze_size=9, img_size=(5, 5), seed=1)
    obs = env.reset()
    ok = obs.state.dtype == torch.uint8 and tuple(obs.state.shape) == (3, 5, 5) and obs.is_first == 1
    color = torch.tensor(env.COLORS[env.target_id], dtype=torch.uint8)
    ok &= bool((obs.state[:, 0, :] == color[:, None]).all()) and bool((obs.state[:, :, -1] == color[:, None]).all())
    # center = agent cell (free, white)
    ok &= bool((obs.state[:, 2, 2] == 255).all())
    # 3x3 view matches the layout around the agent
    x, y = env.pos
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            inside = 0 <= x + dx < 9 and 0 <= y + dy < 9
            free = inside and env.env.layout[y + dy, x + dx] == 1
            ok &= bool(obs.state[:, 2 + dy, 2 + dx].any()) == bool(free)
    report(ok, "A. obs layout", "5x5 view: target color border, 3x3 cells around the agent")

    # Walls block, rewards with the BFS expert
    env = MemoryMaze2dEnv(maze_size=9, seed=2)
    rewards, wall_ok, targets_changed = 0, True, True
    obs = env.reset()
    assert tuple(obs.state.shape) == (3, 64, 64)
    for t in range(1000):
        # wall bump keeps position
        for a in (1, 2, 3, 4):
            dx, dy = env.MOVES[a]
            if not env.is_free(env.pos[0] + dx, env.pos[1] + dy):
                pos = list(env.pos)
                o = env.step(torch.tensor(a))
                wall_ok &= env.pos == pos and o.reward == 0
                break
        prev_target = env.target_id
        o = env.step(env.best().argmax())
        if o.reward == 1:
            rewards += 1
            targets_changed &= env.target_id != prev_target
        assert o.done == 0 and o.is_last == 0
    report(wall_ok and targets_changed and rewards > 20, "A. walls / rewards / expert", "{} targets reached by the BFS expert in 1000 steps".format(rewards))

    # random policy, noop
    env.reset()
    pos = list(env.pos)
    env.step(torch.tensor(0))
    report(env.pos == pos and env.sample().shape == (5,), "A. noop / sample", "5 actions")


###############################################################################
# B. Config + C. End to end
###############################################################################

def test_twister():
    from nnet.models.twister import TWISTER
    from nnet.datasets.episode_replay_buffer import EpisodeReplayBuffer

    model = TWISTER(env_name="memory_maze_2d-9x9", override_config={"num_envs": 1, "eval_episodes": 0, "parallel_envs": False})
    c = model.config
    ok = c.env_class is MemoryMaze2dEnv and c.L == 1001 and c.att_context_left == 1001 and c.replay_buffer_mode == "episode" and c.time_limit == 1000 and model.env.num_actions == 5 and c.policy_discrete
    report(ok, "B. config memory_maze_2d-9x9", "L=C={}, B={}, {} actions".format(c.L, c.batch_size, model.env.num_actions))
    del model

    E, num_envs = 16, 2
    MemoryMaze2dEnv.EPISODE_LENGTHS = {**MemoryMaze2dEnv.EPISODE_LENGTHS, 9: E}
    torch.manual_seed(0)
    model = TWISTER(env_name="memory_maze_2d-9x9", override_config={
        "num_envs": num_envs, "eval_episodes": 0, "precision": "float32", "epoch_length": 10, "pre_fill_steps": 1, "parallel_envs": False,
    })
    c = model.config
    buffer = EpisodeReplayBuffer(batch_size=c.batch_size, root=tempfile.mkdtemp(), buffer_capacity=c.buffer_capacity, epoch_length=c.epoch_length, sample_length=c.L, save_trajectories=False)
    model.compile()
    model.set_replay_buffer(buffer)
    model.on_train_begin()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    model.train()
    finite = True
    for _ in range(3):
        batch = buffer.collate_fn([buffer[i] for i in range(c.batch_size)])
        losses, _, _ = model.train_step(inputs=batch["inputs"], targets=batch["targets"], precision=torch.float32, grad_scaler=grad_scaler, accumulated_steps=1, acc_step=0, eval_training=False)
        finite &= all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values())
    report(finite and c.L == E + 1, "C. end to end memory_maze_2d", "3 train steps, L={}, finite losses".format(c.L))


if __name__ == "__main__":
    test_env()
    test_twister()
    print("{}/{} passed".format(sum(RESULTS), len(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
