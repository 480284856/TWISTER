"""Equivalence and memory test: shared KV cache imagination vs original TWISTER imagination.

Run from the repository root:
    python tests/test_shared_kv_imagination.py
"""

import os
import sys
import types
import copy

import torch
import torch.nn as nn

# Import nnet sub packages without nnet/__init__.py (avoids gym / dm_control / torchvision imports)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nnet_pkg = types.ModuleType("nnet")
nnet_pkg.__path__ = [os.path.join(ROOT, "nnet")]
sys.modules["nnet"] = nnet_pkg

from nnet.modules.twister.tssm import TSSM  # noqa: E402


class Policy(nn.Module):

    """Deterministic differentiable policy, returns an object with rsample() like the TWISTER policy dists"""

    def __init__(self, feat_size, num_actions):
        super().__init__()
        self.linear = nn.Linear(feat_size, num_actions)

    def forward(self, feat):
        out = torch.tanh(self.linear(feat))
        return types.SimpleNamespace(rsample=lambda: out)


def make_inputs(B, L, num_actions, stoch, discrete, firsts):
    idx = torch.randint(0, discrete, (B, L, stoch))
    states = {"stoch": nn.functional.one_hot(idx, discrete).float(), "logits": torch.randn(B, L, stoch, discrete)}
    actions = torch.randn(B, L, num_actions).clamp(-1, 1)
    is_firsts = torch.zeros(B, L)
    for b, ts in firsts.items():
        for t in ts:
            is_firsts[b, t] = 1.0
    return states, actions, is_firsts


def run_imagination(tssm, policy, posts, is_firsts, H, shared_kv, seed):

    tssm.zero_grad(set_to_none=True)
    policy.zero_grad(set_to_none=True)

    prev_state, imag_firsts, imag_firsts_hidden = tssm.imagination_start(posts, is_firsts, shared_kv=shared_kv)

    torch.manual_seed(seed)
    img = tssm.imagine(p_net=policy, prev_state=prev_state, img_steps=H, is_firsts=imag_firsts, is_firsts_hidden=imag_firsts_hidden)

    torch.manual_seed(seed + 1)
    loss = (img["deter"] * torch.randn_like(img["deter"])).sum() + (img["logits"] * torch.randn_like(img["logits"])).sum() + (img["action"] * torch.randn_like(img["action"])).sum()
    loss.backward()

    grads = {"tssm." + n: p.grad.clone() for n, p in tssm.named_parameters() if p.grad is not None}
    grads.update({"policy." + n: p.grad.clone() for n, p in policy.named_parameters() if p.grad is not None})
    return {k: v.detach() for k, v in img.items()}, grads


def max_rel_err(a, b):
    # Floor on the denominator: some gradients are analytically zero (e.g. pos_layer.bias,
    # softmax gradients sum to zero over keys) and only contain ~1e-16 rounding noise
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-3)).item()


def check_equivalence(L, C, H, firsts, B=3, hidden=32, heads=4, blocks=2, num_actions=3, stoch=4, discrete=4, tol=1e-10):

    torch.manual_seed(0)
    tssm = TSSM(num_actions=num_actions, stoch_size=stoch, discrete=discrete, hidden_size=hidden, num_blocks=blocks, num_heads=heads, drop_rate=0.0, att_context_left=C, ff_ratio=2).double()
    policy = Policy(stoch * discrete + hidden, num_actions).double()
    tssm.eval()

    states, actions, is_firsts = make_inputs(B, L, num_actions, stoch, discrete, firsts)
    states = {k: v.double() for k, v in states.items()}
    actions, is_firsts = actions.double(), is_firsts.double()

    with torch.no_grad():
        posts, _ = tssm.observe(states=states, prev_actions=actions.clone(), is_firsts=is_firsts)

    img_ref, grads_ref = run_imagination(tssm, policy, copy.copy(posts), is_firsts, H, shared_kv=False, seed=1)
    img_new, grads_new = run_imagination(tssm, policy, copy.copy(posts), is_firsts, H, shared_kv=True, seed=1)

    errs = {"img." + k: max_rel_err(img_new[k], img_ref[k]) for k in img_ref}
    assert grads_ref.keys() == grads_new.keys(), set(grads_ref) ^ set(grads_new)
    errs.update({"grad." + k: max_rel_err(grads_new[k], grads_ref[k]) for k in grads_ref})

    worst = max(errs, key=errs.get)
    status = "OK  " if errs[worst] < tol else "FAIL"
    print("{} L={:3d} C={:3d} H={:2d} firsts={}  max rel err {:.2e} ({})".format(status, L, C, H, firsts, errs[worst], worst))
    return errs[worst] < tol


def saved_bytes(fn):

    """Bytes of unique storages saved for backward during fn()"""

    storages = {}

    def pack(t):
        s = t.untyped_storage()
        storages[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return sum(storages.values()), out


def hidden_bytes(hidden):
    if hasattr(hidden[0], "shared_k"):
        return sum(c.shared_k.nbytes + c.shared_v.nbytes for c in hidden)
    return sum(k.nbytes + v.nbytes for k, v in hidden)


def measure_memory(L, C, H=15, B=4, hidden=128, heads=4, blocks=4, num_actions=3, stoch=8, discrete=8):

    torch.manual_seed(0)
    tssm = TSSM(num_actions=num_actions, stoch_size=stoch, discrete=discrete, hidden_size=hidden, num_blocks=blocks, num_heads=heads, drop_rate=0.0, att_context_left=C, ff_ratio=2)
    policy = Policy(stoch * discrete + hidden, num_actions)
    tssm.eval()
    states, actions, is_firsts = make_inputs(B, L, num_actions, stoch, discrete, {0: [L // 3]})
    with torch.no_grad():
        posts, _ = tssm.observe(states=states, prev_actions=actions.clone(), is_firsts=is_firsts)

    res = {}
    for shared in (False, True):
        prev_state, f, fh = tssm.imagination_start(copy.copy(posts), is_firsts, shared_kv=shared)
        start = hidden_bytes(prev_state["hidden"])
        saved, _ = saved_bytes(lambda: tssm.imagine(p_net=policy, prev_state=prev_state, img_steps=H, is_firsts=f, is_firsts_hidden=fh))
        res[shared] = (start, saved)

    mb = lambda x: x / 2**20
    print("B={} L={} C={} H={} D={} blocks={} (fp32)".format(B, L, C, H, hidden, blocks))
    print("  {:<10s} {:>16s} {:>22s}".format("", "start KV cache", "saved for backward"))
    for shared, name in ((False, "original"), (True, "shared")):
        print("  {:<10s} {:>13.1f} MB {:>19.1f} MB".format(name, mb(res[shared][0]), mb(res[shared][1])))


if __name__ == "__main__":

    torch.set_printoptions(precision=10)

    cases = [
        # (L, C, H, is_firsts per batch element)
        (12, 5, 7, {0: [0, 6], 1: [3, 4], 2: []}),     # C < H: imagined tokens slide out of the window
        (10, 10, 4, {0: [0], 1: [5, 9], 2: [1]}),      # C == L
        (9, 3, 6, {0: [2, 3, 8], 1: [], 2: [0, 4]}),   # small C, consecutive firsts
        (16, 8, 15, {0: [], 1: [7], 2: [15]}),         # default TWISTER C / H ratio, first at last position
        (6, 1, 3, {0: [3], 1: [], 2: [5]}),            # C == 1
    ]
    ok = all([check_equivalence(*case) for case in cases])

    print()
    measure_memory(L=64, C=8)
    measure_memory(L=128, C=128)

    sys.exit(0 if ok else 1)
