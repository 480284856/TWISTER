"""Equivalence, dropout and memory test: query chunked + checkpointed observe attention vs full attention.

Run from the repository root:
    python tests/test_query_chunk_attention.py
"""

import os
import sys
import types

import torch
import torch.nn as nn

# Import nnet sub packages without nnet/__init__.py (avoids gym / dm_control / torchvision imports)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nnet_pkg = types.ModuleType("nnet")
nnet_pkg.__path__ = [os.path.join(ROOT, "nnet")]
sys.modules["nnet"] = nnet_pkg

from nnet.modules.twister.tssm import TSSM  # noqa: E402


def attentions(tssm):
    return [block.self_att_module.attention for block in tssm.transformer.blocks]


def set_chunks(tssm, chunk_size, checkpoint=True):
    for att in attentions(tssm):
        att.query_chunk_size = chunk_size
        att.query_chunk_checkpoint = checkpoint


def make_inputs(B, L, num_actions, stoch, discrete, firsts, dtype):
    idx = torch.randint(0, discrete, (B, L, stoch))
    states = {"stoch": nn.functional.one_hot(idx, discrete).to(dtype), "logits": torch.randn(B, L, stoch, discrete, dtype=dtype)}
    actions = torch.randn(B, L, num_actions, dtype=dtype).clamp(-1, 1)
    is_firsts = torch.zeros(B, L, dtype=dtype)
    for b, ts in firsts.items():
        for t in ts:
            is_firsts[b, t] = 1.0
    return states, actions, is_firsts


def observe_and_backward(tssm, states, actions, is_firsts, seed):

    """Observe forward + backward of a random projection of priors / posts, returns outputs and grads"""

    tssm.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    posts, priors = tssm.observe(states=states, prev_actions=actions.clone(), is_firsts=is_firsts)

    g = torch.Generator().manual_seed(seed + 1)
    loss = sum((x * torch.randn(x.shape, generator=g, dtype=x.dtype)).sum() for x in (priors["deter"], priors["logits"], priors["stoch"]))
    loss.backward()

    outs = {"deter": priors["deter"].detach(), "logits": priors["logits"].detach(), "stoch": priors["stoch"].detach()}
    outs.update({"hidden.{}.{}".format(b, kv): blk[kv].detach() for b, blk in enumerate(posts["hidden"]) for kv in (0, 1)})
    grads = {n: p.grad.clone() for n, p in tssm.named_parameters() if p.grad is not None}
    return outs, grads


def max_rel_err(a, b):
    # Floor on the denominator: some gradients are analytically zero (e.g. pos_layer.bias) and only hold rounding noise
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-3)).item()


def compare(ref, new):
    (outs_ref, grads_ref), (outs_new, grads_new) = ref, new
    assert grads_ref.keys() == grads_new.keys()
    errs = {"out." + k: max_rel_err(outs_new[k], outs_ref[k]) for k in outs_ref}
    errs.update({"grad." + k: max_rel_err(grads_new[k], grads_ref[k]) for k in grads_ref})
    worst = max(errs, key=errs.get)
    return errs[worst], worst


def build(C, drop_rate, dtype, hidden=32, heads=4, blocks=2, num_actions=3, stoch=4, discrete=4):
    torch.manual_seed(0)
    tssm = TSSM(num_actions=num_actions, stoch_size=stoch, discrete=discrete, hidden_size=hidden, num_blocks=blocks, num_heads=heads, drop_rate=drop_rate, att_context_left=C, ff_ratio=2).to(dtype)
    return tssm


def check_equivalence(L, C, chunk, firsts, B=3, tol=1e-10):

    """Dropout off, float64: chunked (+ checkpoint) must match full attention"""

    tssm = build(C, drop_rate=0.0, dtype=torch.float64)
    states, actions, is_firsts = make_inputs(B, L, 3, 4, 4, firsts, torch.float64)

    set_chunks(tssm, None)
    ref = observe_and_backward(tssm, states, actions, is_firsts, seed=1)

    ok = True
    for checkpoint in (False, True):
        set_chunks(tssm, chunk, checkpoint)
        err, worst = compare(ref, observe_and_backward(tssm, states, actions, is_firsts, seed=1))
        status = "OK  " if err < tol else "FAIL"
        ok &= err < tol
        print("{} equivalence     L={:3d} C={:3d} chunk={:3d} checkpoint={:<5} firsts={}  max rel err {:.2e} ({})".format(status, L, C, chunk, str(checkpoint), firsts, err, worst))
    return ok


def check_dropout_checkpoint(L, C, chunk, firsts, B=3, tol=1e-10):

    """Dropout on: checkpointed chunks must reproduce the forward dropout masks in backward (same grads as non checkpointed chunks)"""

    tssm = build(C, drop_rate=0.3, dtype=torch.float64)
    tssm.train()
    states, actions, is_firsts = make_inputs(B, L, 3, 4, 4, firsts, torch.float64)

    set_chunks(tssm, chunk, checkpoint=False)
    ref = observe_and_backward(tssm, states, actions, is_firsts, seed=2)
    set_chunks(tssm, chunk, checkpoint=True)
    err, worst = compare(ref, observe_and_backward(tssm, states, actions, is_firsts, seed=2))

    # Sanity: dropout is really active (a different seed gives different outputs)
    other = observe_and_backward(tssm, states, actions, is_firsts, seed=3)
    active = compare(ref, other)[0] > 1e-3

    ok = err < tol and active
    print("{} dropout+ckpt    L={:3d} C={:3d} chunk={:3d} drop=0.3  max rel err {:.2e} ({}), dropout active: {}".format("OK  " if ok else "FAIL", L, C, chunk, err, worst, active))
    return ok


def check_dropout_statistics(L=48, C=48, chunk=16, n=200, B=2):

    """Dropout on: chunked and full attention outputs have the same distribution (mean / std over seeds)"""

    tssm = build(C, drop_rate=0.3, dtype=torch.float64)
    tssm.train()
    states, actions, is_firsts = make_inputs(B, L, 3, 4, 4, {0: [20]}, torch.float64)

    samples = {}
    with torch.no_grad():
        for chunk_size in (None, chunk):
            set_chunks(tssm, chunk_size)
            outs = []
            for seed in range(n):
                # Independent seeds for the two settings, so that the samples are independent
                torch.manual_seed((1000 if chunk_size is None else 5000) + seed)
                _, priors = tssm.observe(states=states, prev_actions=actions.clone(), is_firsts=is_firsts)
                outs.append(priors["deter"])
            samples[chunk_size] = torch.stack(outs)

    full, chunked = samples[None], samples[chunk]
    mean_diff = (full.mean(0) - chunked.mean(0)).abs()
    se = (full.var(0) / n + chunked.var(0) / n).sqrt().clamp(min=1e-12)
    z = (mean_diff / se)
    std_ratio = (chunked.std(0).mean() / full.std(0).mean()).item()
    # Per element z-scores of the mean difference should look standard normal
    ok = 0.6 < z.mean().item() < 1.0 and abs(std_ratio - 1) < 0.05
    print("{} dropout stats   L={:3d} C={:3d} chunk={:3d} {} seeds: mean |z| of mean diff {:.2f} (~0.80 expected), std ratio chunked/full {:.3f}".format("OK  " if ok else "FAIL", L, C, chunk, n, z.mean().item(), std_ratio))
    return ok


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


def measure_memory(L, C, chunk, B=4, hidden=128, heads=4, blocks=4):

    tssm = build(C, drop_rate=0.1, dtype=torch.float32, hidden=hidden, heads=heads, blocks=blocks, stoch=8, discrete=8)
    tssm.train()
    states, actions, is_firsts = make_inputs(B, L, 3, 8, 8, {0: [L // 3]}, torch.float32)

    res = {}
    for chunk_size in (None, chunk):
        set_chunks(tssm, chunk_size)
        res[chunk_size], _ = saved_bytes(lambda: tssm.observe(states=states, prev_actions=actions.clone(), is_firsts=is_firsts))

    mb = lambda x: x / 2**20
    print("B={} L={} C={} D={} heads={} blocks={} fp32, observe saved for backward: full {:.1f} MB, chunk={} {:.1f} MB".format(B, L, C, hidden, heads, blocks, mb(res[None]), chunk, mb(res[chunk])))


if __name__ == "__main__":

    cases = [
        # (L, C, chunk, is_firsts per batch element)
        (12, 12, 5, {0: [0, 6], 1: [3, 4], 2: []}),      # C == L, chunk does not divide L
        (16, 5, 4, {0: [7], 1: [], 2: [15]}),            # sliding window C < L
        (20, 20, 1, {0: [2, 3], 1: [19], 2: [0]}),       # chunk of a single query
        (9, 3, 8, {0: [], 1: [4], 2: [1, 8]}),           # one full chunk + remainder
    ]
    ok = all([check_equivalence(*case) for case in cases])
    ok &= all([check_dropout_checkpoint(*case) for case in cases])
    ok &= check_dropout_statistics()

    print()
    measure_memory(L=512, C=512, chunk=64)
    measure_memory(L=1024, C=1024, chunk=128)

    sys.exit(0 if ok else 1)
