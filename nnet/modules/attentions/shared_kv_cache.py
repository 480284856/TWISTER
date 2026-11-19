# Shared Key / Value cache for batched imagination.
#
# The original TWISTER implementation starts one imagination rollout from every
# position t of every training sequence (B' = B * L start points) and gives each
# of them its own copy of the last C key / value states, i.e. a (B*L, C, D) tensor
# per layer built with torch.stack. All of those windows are slices of the same
# (B, L, D) tensors produced by the world model observe step.
#
# SharedKVCache keeps the observe keys / values once, shape (B, L, D), and only
# stores the keys / values of the imagined tokens per start point. The attention
# layer (RelPosMultiHeadSelfAttention.forward_shared_kv) recovers the per-start
# window with a band mask, reproducing the original sliding-window and episode
# boundary masking exactly.

# PyTorch
import torch


class SharedKVCache:

    """Per layer key / value cache for imagination rollouts started at every position of a batch of sequences

    Args:
        shared_k: observe keys split in heads, contiguous, shape (B, H, L, d), detached
        shared_v: observe values split in heads, contiguous, shape (B, H, L, d), detached
        ep_start: (B, L) long, first observe position each start point may attend to
            because of episode boundaries (see episode_start_from_is_firsts)
        context: attention left context C
        step: index h of the next imagined token (= number of imagined tokens so far)
        private_k: keys of the last min(h, C) imagined tokens, list of (B*L, 1, D) tensors
        private_v: values of the last min(h, C) imagined tokens, list of (B*L, 1, D) tensors
        start_chunk_size: attention computed by chunks of start points t (None = all at once),
            each chunk only reads the observe keys it can attend to (band of width C)

    Start point (b, t) at imagination step h (0-based) attends to:
        - observe positions s with max(t+1-C+h, ep_start[b, t]) <= s <= t
        - its imagined tokens max(0, h-C) .. h (h being the current token)
    which is the same key set as the original implementation (C cached keys + current token).
    """

    def __init__(self, shared_k, shared_v, ep_start, context, step=0, private_k=(), private_v=(), start_chunk_size=None):

        assert shared_k.dim() == 4 and shared_k.shape == shared_v.shape
        assert ep_start.shape == (shared_k.shape[0], shared_k.shape[2])
        assert len(private_k) == len(private_v) == min(step, context)

        self.shared_k = shared_k
        self.shared_v = shared_v
        self.ep_start = ep_start
        self.context = context
        self.step = step
        self.private_k = list(private_k)
        self.private_v = list(private_v)
        self.start_chunk_size = start_chunk_size

    @property
    def batch_size(self):
        return self.shared_k.shape[0]

    @property
    def num_heads(self):
        return self.shared_k.shape[1]

    @property
    def seq_len(self):
        return self.shared_k.shape[2]

    def append(self, k, v):

        """Return the cache for the next step with the current imagined token appended

        Keys / values are detached like the original hidden states (detach_hidden=True)
        """

        private_k = self.private_k + [k.detach()]
        private_v = self.private_v + [v.detach()]

        # Only the last C imagined tokens remain visible to the next token
        return SharedKVCache(
            shared_k=self.shared_k,
            shared_v=self.shared_v,
            ep_start=self.ep_start,
            context=self.context,
            step=self.step + 1,
            private_k=private_k[-self.context:],
            private_v=private_v[-self.context:],
            start_chunk_size=self.start_chunk_size,
        )


def build_shared_kv_caches(hidden, is_firsts, num_heads, context, start_chunk_size=None):

    """Build one SharedKVCache per transformer block from the observe hidden states

    Args:
        hidden: list of (K, V) per block, each of shape (B, L, D) (observe with prev_state=None)
        is_firsts: (B, L)
        num_heads: attention heads H
        context: attention left context C
        start_chunk_size: chunk size over start points (None = all at once)

    Return:
        list of SharedKVCache
    """

    ep_start = episode_start_from_is_firsts(is_firsts)

    def to_heads(x):
        B, L, D = x.shape
        return x.detach().reshape(B, L, num_heads, D // num_heads).transpose(1, 2).contiguous()

    return [SharedKVCache(to_heads(k), to_heads(v), ep_start, context, start_chunk_size=start_chunk_size) for k, v in hidden]


def episode_start_from_is_firsts(is_firsts):

    """First observe position visible to an imagination started at each position

    Reproduces the original TWISTER imagination masking (is_firsts_hidden built in
    WorldModel.forward + return_is_firsts_mask): an observe key at position s is
    visible to a rollout started at t iff no is_first flag lies in [s, t].

    Args:
        is_firsts: (B, L) float or bool, 1 = first step of an episode

    Return:
        ep_start: (B, L) long
    """

    batch_size, seq_len = is_firsts.shape
    positions = torch.arange(seq_len, device=is_firsts.device).unsqueeze(0).expand(batch_size, seq_len)

    # Last is_first position <= t, -1 if none
    last_first = torch.where(is_firsts.bool(), positions, torch.full_like(positions, -1))
    last_first = torch.cummax(last_first, dim=1).values

    return last_first + 1


def is_shared_kv_hidden(hidden):

    """True if hidden is a list of SharedKVCache (one per transformer block)"""

    return isinstance(hidden, (list, tuple)) and len(hidden) > 0 and isinstance(hidden[0], SharedKVCache)
