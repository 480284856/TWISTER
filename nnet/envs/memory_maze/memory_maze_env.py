# Memory Maze environment (Pasukonis et al., 2023, https://github.com/jurgisp/memory-maze)
#
# 3D mazes of size 9x9, 11x11, 13x13 and 15x15, 64x64 RGB first person observations, 6 discrete actions,
# fixed episode lengths of 1000, 2000, 3000 and 4000 steps (4 Hz control, time limits of 250, 500, 750 and 1000 s).
# Episodes only end with the time limit: is_last = 1 and done = 0 (no terminal state, the value is bootstrapped).

# PyTorch
import torch
import torchvision

# NeuralNets
from nnet.structs import AttrDict

# Other
import os
import sys
import random
import logging
import datetime


# memory_maze resets the agent with a velocity although it has no free joint: dm_control logs this warning at every
# episode reset (in every env worker). Harmless, filtered out.
class _NoFreeJointVelocityFilter(logging.Filter):
    def filter(self, record):
        return "Cannot set velocity on Entity with no free joint" not in record.getMessage()


logging.getLogger("absl").addFilter(_NoFreeJointVelocityFilter())


def _fast_update_activation(self, physics):

    """Vectorized dm_control TargetSphere._update_activation (same result)

    The original loops in Python over physics.data.contact (one Python object per contact) for every target
    after every physics substep: 6 targets x 50 substeps per Memory Maze step, ~2/3 of the env step time.
    Here the contact geom pairs are read as one (ncon, 2) array from MuJoCo.

    Activated iff a contact involves the target geom and, if specific_collision_geom_ids is set, one of those geoms.
    """

    if self._activated:
        return
    data = physics.data.ptr
    if data.ncon == 0:
        return
    contact = data.contact
    geom = contact.geom[:data.ncon] if hasattr(contact, "geom") else np.stack([contact.geom1, contact.geom2], axis=-1)[:data.ncon]
    hit = (geom == self._geom_id).any(axis=-1)
    if self._specific_collision_geom_ids:
        hit &= np.isin(geom, list(self._specific_collision_geom_ids)).any(axis=-1)
    if hit.any():
        self._activated = True
        physics.bind(self._material).rgba[-1] = 0


def patch_target_sphere():

    """Use the vectorized target activation check in this process (idempotent)"""

    global np
    try:
        import numpy as np
        from dm_control.locomotion.props import target_sphere
    except ImportError:
        return
    if target_sphere.TargetSphere._update_activation is not _fast_update_activation:
        target_sphere.TargetSphere._original_update_activation = target_sphere.TargetSphere._update_activation
        target_sphere.TargetSphere._update_activation = _fast_update_activation


class MemoryMazeEnv:

    # Episode length (agent steps) per maze size
    EPISODE_LENGTHS = {9: 1000, 11: 2000, 13: 3000, 15: 4000}

    def obs_space(self):

        return (["image", (3, self.img_size[0], self.img_size[1]), torch.uint8],)

    def __init__(
            self,
            maze_size=15,
            img_size=(64, 64),
            history_frames=1,
            episode_saving_path=None,
            action_repeat=1,
            seed=None,
            task_kwargs={"image_only_obs": True},
            fast_target_check=True
        ):

        assert maze_size in self.EPISODE_LENGTHS, "maze_size must be one of {}".format(list(self.EPISODE_LENGTHS))
        assert history_frames == 1, "history_frames > 1 not supported"
        assert action_repeat == 1, "Memory Maze episode lengths are defined in agent steps (4 Hz control), action_repeat must be 1"

        # memory_maze defaults to MUJOCO_GL=egl (headless GPU rendering on Linux), not available on macOS
        if sys.platform == "darwin" and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "glfw"

        # Imported here so that the other environments do not require memory_maze
        from memory_maze import tasks

        # Vectorized target touch check, same results, ~3x faster env steps
        if fast_target_check:
            patch_target_sphere()

        # Env
        self.maze_size = maze_size
        self.img_size = tuple(img_size)
        self.history_frames = history_frames
        self.action_repeat = action_repeat
        self.episode_length = self.EPISODE_LENGTHS[maze_size]
        self.task_kwargs = dict(task_kwargs)
        self.env = getattr(tasks, "memory_maze_{}x{}".format(maze_size, maze_size))(
            seed=seed if seed is not None else random.randint(0, 2**31 - 1),
            camera_resolution=self.img_size[0],
            **task_kwargs
        )
        self.num_actions = int(self.env.action_spec().num_values)

        # episode_saving_path
        self.episode_saving_path = episode_saving_path
        if self.episode_saving_path is not None:
            os.makedirs(self.episode_saving_path, exist_ok=True)

        # FPS (4 Hz control)
        self.fps = 4

    def sample(self):

        return torch.nn.functional.one_hot(torch.randint(low=0, high=self.num_actions, size=()), num_classes=self.num_actions).type(torch.float32)

    def process_obs(self, observation):

        # dm_env observation: image array (image_only_obs=True) or dict with "image" (H, W, 3) uint8
        image = observation["image"] if isinstance(observation, dict) else observation
        image = torch.as_tensor(image.copy()).permute(2, 0, 1)
        assert image.dtype == torch.uint8 and tuple(image.shape) == (3,) + self.img_size, "unexpected observation {} {}".format(image.dtype, tuple(image.shape))

        return image

    def reset(self):

        # Reset Env
        timestep = self.env.reset()
        state = self.process_obs(timestep.observation)

        # Episode Score / videos
        self.episode_score = 0.0
        if self.episode_saving_path is not None:
            self.episode_video = [state.permute(1, 2, 0)]

        return AttrDict(
            state=state,
            reward=torch.tensor(0.0, dtype=torch.float32),
            done=torch.tensor(False, dtype=torch.float32),
            is_first=torch.tensor(True, dtype=torch.float32),
            is_last=torch.tensor(False, dtype=torch.float32)
        )

    def step(self, action):

        # Discrete action index
        timestep = self.env.step(int(action.item()) if torch.is_tensor(action) else int(action))
        state = self.process_obs(timestep.observation)
        reward = float(timestep.reward or 0.0)
        is_last = bool(timestep.last())

        # Time limit only: no terminal state (discount stays 1)
        done = is_last and timestep.discount is not None and float(timestep.discount) == 0.0

        # Update Episode Score / videos
        self.episode_score += reward
        if self.episode_saving_path is not None:
            self.episode_video.append(state.permute(1, 2, 0))
            if is_last:
                date_time_score = str(datetime.datetime.now()).replace(" ", "_") + "_" + str(self.episode_score)
                torchvision.io.write_video(filename=os.path.join(self.episode_saving_path, "{}.mp4".format(date_time_score)), video_array=torch.stack(self.episode_video, dim=0), fps=self.fps, video_codec="libx264")

        return AttrDict(
            state=state,
            reward=torch.tensor(reward, dtype=torch.float32),
            done=torch.tensor(done, dtype=torch.float32),
            is_first=torch.tensor(False, dtype=torch.float32),
            is_last=torch.tensor(is_last, dtype=torch.float32)
        )

    def set_seed(self, seed):
        # dm_env tasks are seeded at construction, re create the env with the new seed
        from memory_maze import tasks
        self.env = getattr(tasks, "memory_maze_{}x{}".format(self.maze_size, self.maze_size))(seed=seed, camera_resolution=self.img_size[0], **self.task_kwargs)

    def random_seed(self):
        self.set_seed(random.randint(0, 2**31 - 1))
