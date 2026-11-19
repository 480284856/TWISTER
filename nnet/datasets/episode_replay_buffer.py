# Episode aligned replay buffer.
#
# ReplayBuffer stores one window of sample_length steps for every environment step (window ending at that step),
# sampled uniformly. Windows start at arbitrary episode offsets, so most training tokens only see a truncated
# part of their episode history, and each window holds sample_length references (10M steps x L=4000 is not storable).
#
# EpisodeReplayBuffer stores every finished episode once (stacked tensors) and samples windows that start at the
# first step of an episode: samples[0] is the reset step (is_first=1). With sample_length equal to the stored episode
# length (Memory Maze: episode length + 1 reset step), every sample is exactly one complete episode and the history
# visible to each training token matches evaluation.

# PyTorch
import torch

# NeuralNets
from nnet import datasets

# Other
import os
import glob
import random


class EpisodeReplayBuffer(datasets.ReplayBuffer):

    """Replay buffer of complete episodes, samples windows of sample_length steps starting at episode starts

    Episodes are finalized when the next reset step (is_first=1) of the same env stream is appended.
    Episodes shorter than sample_length are stored (they count towards capacity) but never sampled.
    Episodes longer than sample_length are sampled from their start (steps after sample_length are not used).
    buffer_capacity is a number of environment steps (stored frames), like ReplayBuffer.
    """

    # Index of is_first in the appended samples (state, action, reward, done, is_first, model_step)
    IS_FIRST_INDEX = 4

    def __init__(self, *args, buffer_name="EpisodeReplayBuffer", prune_saved_episodes=True, **kwargs):
        super(EpisodeReplayBuffer, self).__init__(*args, buffer_name=buffer_name, **kwargs)

        # Delete saved episode files once all their episodes have been evicted from the buffer
        # (otherwise the buffer directory keeps every episode ever collected: ~1.2 TB for 100M Memory Maze steps).
        # Older checkpoints (keep_last_k) may then reference deleted episodes: resume from the last checkpoint.
        self.prune_saved_episodes = prune_saved_episodes

        # Episode ids that can be sampled (length >= sample_length)
        self.sample_ids = []
        self.num_short_episodes = 0

    def get_infos(self):
        return {
            "traj_index": self.traj_index,
            "num_steps": self.num_steps,
            "num_episodes": len(self.ram_buffer),
            "num_sample_episodes": len(self.sample_ids),
            "num_short_episodes": self.num_short_episodes,
        }

    def can_sample(self):
        return len(self.sample_ids) > 0

    @staticmethod
    def episode_length(episode):
        return episode[0].shape[0]

    def load(self, buffer_keys):
        super(EpisodeReplayBuffer, self).load(buffer_keys)
        self.sample_ids = [key for key, episode in self.ram_buffer.items() if self.episode_length(episode) >= self.sample_length]

    def save(self):

        # Save finished episodes not saved yet (some may already have been evicted)
        if self.save_trajectories:
            save_trajs = {traj_id: self.ram_buffer[traj_id] for traj_id in range(self.last_save_traj_index, self.traj_index) if traj_id in self.ram_buffer}
            torch.save(save_trajs, os.path.join(self.buffer_dir, "{}.torch".format(self.traj_index)))
            self.last_save_traj_index = self.traj_index.item()

            # File <N>.torch holds episode ids < N: obsolete once N <= oldest episode id still in the buffer
            if self.prune_saved_episodes:
                oldest_id = next(iter(self.ram_buffer), self.traj_index.item())
                for path in glob.glob(os.path.join(self.buffer_dir, "*.torch")):
                    name = os.path.basename(path)[:-len(".torch")]
                    if name.isdigit() and int(name) <= oldest_id:
                        os.remove(path)

    def enforce_capacity(self):

        # Pop oldest episodes while stored steps > capacity
        while self.num_steps > self.buffer_capacity and len(self.ram_buffer) > 0:
            oldest_episode_id, oldest_episode = self.ram_buffer.popitem(last=False)
            self.num_steps -= self.episode_length(oldest_episode)
            if self.sample_ids and self.sample_ids[0] == oldest_episode_id:
                self.sample_ids.pop(0)

    def finalize_episode(self, stream):

        # Stack elements (elt, T, ...)
        episode = [torch.stack([step[elt] for step in stream], dim=0) for elt in range(len(stream[0]))]
        length = self.episode_length(episode)

        # Add to ram buffer
        episode_id = self.traj_index.item()
        self.ram_buffer[episode_id] = episode
        if length >= self.sample_length:
            self.sample_ids.append(episode_id)
        else:
            self.num_short_episodes += 1

        # Update Index
        self.traj_index += 1
        self.num_steps += length

        # Enforce num_steps <= buffer_capacity
        self.enforce_capacity()

    def append_step(self, sample, sample_id):

        # None sample
        if sample is None:
            return

        # Init Stream
        if sample_id not in self.streams:
            self.streams[sample_id] = []
        stream = self.streams[sample_id]

        # Reset step: the previous episode of this env is finished
        is_first = bool(sample[self.IS_FIRST_INDEX])
        if is_first and len(stream) > 0:
            self.finalize_episode(stream)
            stream = self.streams[sample_id] = []

        # Steps before the first reset of a stream (not episode aligned) are dropped
        if len(stream) == 0 and not is_first:
            return self.get_infos()

        # Update Stream
        stream.append([s.clone() for s in sample])

        return self.get_infos()

    def sample(self):

        # Select Episode, window starting at the episode reset step (elt, sample_length, ...)
        episode = self.ram_buffer[random.choice(self.sample_ids)]

        return [elt[:self.sample_length] for elt in episode]
