"""float16 overflow test for RelPosMultiHeadSelfAttention (float32 attention scores)

1. Same outputs as the previous implementation (git HEAD) in float64, full and query chunked attention.
2. float16, normal weights: old and new outputs agree (no overflow, new scores only more precise).
3. With large query / key weights, float16 Q.K^T overflows: the previous implementation returns NaN,
   the float32 score implementation stays finite.

Run from the repository root (CPU is enough):
    python tests/test_attention_fp16_overflow.py [git revision of the previous implementation, default HEAD]
"""

import os
import sys
import types
import subprocess

import torch

# Import nnet sub packages without nnet/__init__.py (avoids gym / dm_control / torchvision imports)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nnet_pkg = types.ModuleType("nnet")
nnet_pkg.__path__ = [os.path.join(ROOT, "nnet")]
sys.modules["nnet"] = nnet_pkg

from nnet.modules.attentions.rel_pos_multi_head_self_attention import RelPosMultiHeadSelfAttention as New  # noqa: E402

PATH = "nnet/modules/attentions/rel_pos_multi_head_self_attention.py"


def load_old(rev):
    src = subprocess.check_output(["git", "show", "{}:{}".format(rev, PATH)], cwd=ROOT, text=True)
    mod = types.ModuleType("old_rel_pos_attention")
    exec(compile(src, "old_" + os.path.basename(PATH), "exec"), mod.__dict__)
    return mod.RelPosMultiHeadSelfAttention


def build(cls, D=64, H=4, T=64, chunk=None):
    torch.manual_seed(0)
    m = cls(dim_model=D, num_heads=H, attn_drop_rate=0.0, max_pos_encoding=T, causal=True, query_chunk_size=chunk)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
    return m.eval()


def causal_mask(T):
    return torch.ones(T, T, dtype=torch.bool).tril().view(1, 1, T, T)


def run(m, x, mask):
    with torch.no_grad():
        return m(x, mask=mask).x


def check_equivalence(Old, B=2, T=64, D=64):
    for chunk in [None, 16]:
        old, new = build(Old, D=D, T=T, chunk=chunk).double(), build(New, D=D, T=T, chunk=chunk).double()
        new.load_state_dict(old.state_dict())
        x = torch.randn(B, T, D, dtype=torch.float64)
        err = (run(old, x, causal_mask(T)) - run(new, x, causal_mask(T))).abs().max().item()
        print("float64 old vs new, query_chunk_size={}: max abs diff {:.3e}".format(chunk, err))
        assert err < 1e-12


def half_pair(Old, T, D, chunk, gain):
    old = build(Old, D=D, T=T, chunk=chunk)
    with torch.no_grad():
        old.query_layer.weight.mul_(gain)
        old.key_layer.weight.mul_(gain)
    new = build(New, D=D, T=T, chunk=chunk)
    new.load_state_dict(old.state_dict())
    return old.half(), new.half()


def max_score(m, x):
    # Largest unscaled per head content score |(q + u).k| in float64, float16 max is 65504
    m = m.double()
    with torch.no_grad():
        B, T, _ = x.shape
        q = (m.query_layer(x.double()) + m.u).view(B, T, m.num_heads, -1)
        k = m.key_layer(x.double()).view(B, T, m.num_heads, -1)
        return torch.einsum("bthd,bshd->bhts", q, k).abs().max().item()


def check_fp16(Old, B=2, T=64, D=64):
    for chunk in [None, 16]:
        old, new = half_pair(Old, T, D, chunk, gain=1.0)
        x = torch.randn(B, T, D).half()
        y_old, y_new = run(old, x, causal_mask(T)), run(new, x, causal_mask(T))
        rel = ((y_new - y_old).float().abs().max() / y_old.float().abs().max()).item()
        print("float16 normal weights, query_chunk_size={}: old vs new rel diff {:.2e}".format(chunk, rel))
        assert torch.isfinite(y_new).all() and rel < 1e-2


def check_overflow(Old, B=2, T=64, D=64, gain=150.0):
    for chunk in [None, 16]:
        old, new = half_pair(Old, T, D, chunk, gain)
        x = torch.randn(B, T, D).half()
        s = max_score(half_pair(Old, T, D, chunk, gain)[1], x)
        y_old, y_new = run(old, x, causal_mask(T)), run(new, x, causal_mask(T))
        print("float16 large weights, query_chunk_size={}: max |q.k| {:.0f}, old finite {}, new finite {}".format(
            chunk, s, bool(torch.isfinite(y_old).all()), bool(torch.isfinite(y_new).all())))
        assert s > 65504, "test weights do not reach the float16 range"
        assert not torch.isfinite(y_old).all(), "old implementation did not overflow"
        assert torch.isfinite(y_new).all()


if __name__ == "__main__":
    Old = load_old(sys.argv[1] if len(sys.argv) > 1 else "HEAD")
    check_equivalence(Old)
    check_fp16(Old)
    check_overflow(Old)
    print("ok")
