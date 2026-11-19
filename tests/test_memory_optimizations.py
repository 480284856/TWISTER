"""Tests for the long context (Memory Maze) changes:

    1. EpisodeReplayBuffer: complete episodes, samples start at the episode reset step, capacity, save / load
    2. num_env_steps_to_run: fractional env steps carried over, exact replay ratio
    3. att_max_pos_encoding: L, C > 2048 (fails with the original 2048 table)
    4. actor imagination without autograd graph under actor_grad == "reinforce": same loss and policy gradients
    5. CNN encoder / decoder checkpointed by chunks of frames: same outputs and gradients, less saved memory
    8. InfoNCE by chunks of rows: same values, gradients and accuracy as the original implementation
    9. shared kv imagination by chunks of start points: same outputs as without chunks / the original copy

Run from the repository root:
    python tests/test_memory_optimizations.py
"""

import os
import sys
import types
import copy
import tempfile
import importlib.machinery

import torch
import torch.nn as nn

# Import nnet sub packages without nnet/__init__.py and stub the dependencies that are only needed
# for environments / logging (gym, dm_control, torchvision transforms, tensorboard, tqdm, wandb)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
nnet_pkg = types.ModuleType("nnet")
nnet_pkg.__path__ = [os.path.join(ROOT, "nnet")]
sys.modules["nnet"] = nnet_pkg


def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)  # torch._dynamo calls find_spec on some modules
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


stub("torchvision", transforms=types.SimpleNamespace(RandomResizedCrop=lambda **kwargs: None))
stub("torch.utils.tensorboard", SummaryWriter=object)
stub("tqdm", tqdm=lambda x, **kwargs: x)
stub("wandb")
stub("nnet.envs")

torch.distributions.Distribution.set_default_validate_args(False)

from nnet.datasets.episode_replay_buffer import EpisodeReplayBuffer  # noqa: E402
from nnet.models.twister import TWISTER, num_env_steps_to_run  # noqa: E402
from nnet.modules import twister as tn  # noqa: E402
from nnet.modules.twister.chunked import info_nce_loss  # noqa: E402
from nnet.structs import AttrDict  # noqa: E402
import test_shared_kv_imagination as shared_kv_test  # noqa: E402

RESULTS = []


def report(ok, name, detail):
    RESULTS.append(ok)
    print("{} {:<44s} {}".format("OK  " if ok else "FAIL", name, detail))


def max_rel_err(a, b):
    # Floor on the denominator: some gradients are analytically zero and only hold rounding noise
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-3)).item()


def saved_bytes(fn):
    storages = {}

    def pack(t):
        s = t.untyped_storage()
        storages[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return sum(storages.values()), out


###############################################################################
# 1. EpisodeReplayBuffer
###############################################################################

def step(t, is_first, episode):
    # (state, action, reward, done, is_first, model_step), state encodes (episode, t)
    return [torch.tensor([float(episode), float(t)]), torch.zeros(3), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(float(is_first)), torch.tensor(0)]


def test_episode_replay_buffer():

    E = 7          # env steps per episode
    L = E + 1      # stored episode = reset step + E steps
    root = tempfile.mkdtemp()
    buffer = EpisodeReplayBuffer(batch_size=2, root=root, buffer_capacity=5 * L, epoch_length=1, sample_length=L)

    # Two envs, env 1 starts mid-episode (steps before its first reset must be dropped)
    episode_ids = {0: 0, 1: 100}
    for t in range(3):
        buffer.append_step(step(t + 4, False, -1), 1)
    for env_i in (0, 1):
        buffer.append_step(step(0, True, episode_ids[env_i]), env_i)

    lengths = {0: E, 1: E}
    num_rounds = 6
    for r in range(num_rounds):
        for t in range(1, E + 1):
            for env_i in (0, 1):
                if t <= lengths[env_i]:
                    buffer.append_step(step(t, False, episode_ids[env_i]), env_i)
        # Reset both envs (one short episode on env 1 in round 2)
        for env_i in (0, 1):
            episode_ids[env_i] += 1
            buffer.append_step(step(0, True, episode_ids[env_i]), env_i)
        lengths[1] = 3 if r == 1 else E
        # Checkpoint mid run: its episodes are all evicted by the end, the file must be pruned at the next save
        if r == 0:
            buffer.save()
            first_save = os.path.join(buffer.buffer_dir, "{}.torch".format(int(buffer.traj_index)))
            ok_first_save = os.path.exists(first_save)

    infos = buffer.get_infos()
    ok_capacity = infos["num_steps"] <= 5 * L and int(infos["num_steps"]) == sum(ep[0].shape[0] for ep in buffer.ram_buffer.values())
    ok_short = infos["num_short_episodes"] == 1 and all(buffer.ram_buffer[i][0].shape[0] == L for i in buffer.sample_ids)

    # Samples: one complete episode, starting at its reset step
    ok_samples = True
    for _ in range(200):
        sample = buffer.sample()
        states, is_firsts = sample[0], sample[4]
        ok_samples &= states.shape[0] == L and is_firsts[0] == 1 and is_firsts[1:].sum() == 0
        ok_samples &= bool((states[:, 0] == states[0, 0]).all()) and bool((states[:, 1] == torch.arange(L).float()).all())
        ok_samples &= states[0, 0] >= 0  # never the partial stream of env 1

    # Save / load roundtrip
    buffer.save()
    state_dict = buffer.state_dict()
    loaded = EpisodeReplayBuffer(batch_size=2, root=root, buffer_capacity=5 * L, epoch_length=1, sample_length=L)
    loaded.load_state_dict(copy.deepcopy(state_dict))
    ok_load = sorted(loaded.ram_buffer) == sorted(buffer.ram_buffer) and loaded.sample_ids == buffer.sample_ids and int(loaded.num_steps) == int(buffer.num_steps)
    ok_prune = ok_first_save and not os.path.exists(first_save)

    report(ok_capacity and ok_short and ok_samples and ok_load and ok_prune, "1. EpisodeReplayBuffer",
           "episodes kept {}, steps {} / capacity {}, short {}, samples aligned {}, save/load {}, evicted save file pruned {}".format(
               len(buffer.ram_buffer), int(infos["num_steps"]), 5 * L, infos["num_short_episodes"], ok_samples, ok_load, ok_prune))


###############################################################################
# 2. Fractional env steps
###############################################################################

def test_env_step_balance():

    ok = True
    details = []
    for name, B, L, period, num_envs in (("9x9", 4, 1001, 64, 32), ("11x11", 4, 2001, 64, 32), ("13x13", 4, 3001, 64, 32), ("15x15", 4, 4001, 64, 32), ("15x15 L=4000", 4, 4000, 64, 32)):
        num = (B * L) / (period * num_envs)
        balance, calls, updates = 0.0, 0, 10000
        for _ in range(updates):
            n, balance = num_env_steps_to_run(balance, num)
            calls += n
        replay_ratio = updates * B * L / (calls * num_envs)
        truncated_ratio = B * L / (int(num) * num_envs)
        ok &= abs(replay_ratio - period) < 0.01 and 0 <= balance < 1
        details.append("{} {:.4f} (int(): {:.1f})".format(name, replay_ratio, truncated_ratio))
    report(ok, "2. env step balance, replay ratio", ", ".join(details))


###############################################################################
# 3. max_pos_encoding > 2048
###############################################################################

def test_max_pos_encoding(L=2300, C=2300):

    torch.manual_seed(0)
    kwargs = dict(num_actions=3, stoch_size=2, discrete=2, hidden_size=16, num_blocks=1, num_heads=2, drop_rate=0.0, att_context_left=C, ff_ratio=1)
    states = {"stoch": nn.functional.one_hot(torch.randint(0, 2, (1, L, 2)), 2).double(), "logits": torch.randn(1, L, 2, 2).double()}
    actions, is_firsts = torch.randn(1, L, 3).double(), torch.zeros(1, L).double()
    is_firsts[0, 1000] = 1

    # Original table size fails
    try:
        tn.TSSM(**kwargs).double().observe(states=states, prev_actions=actions.clone(), is_firsts=is_firsts)
        failed_2048 = False
    except Exception:
        failed_2048 = True

    # Larger table: full vs query chunked observe agree, shared kv imagination with C > 2048 runs
    tssm = tn.TSSM(**kwargs, max_pos_encoding=C + 1).double().eval()
    with torch.no_grad():
        torch.manual_seed(1)
        posts_full, priors_full = tssm.observe(states=states, prev_actions=actions.clone(), is_firsts=is_firsts)
        for block in tssm.transformer.blocks:
            block.self_att_module.attention.query_chunk_size = 512
        torch.manual_seed(1)
        posts, priors = tssm.observe(states=states, prev_actions=actions.clone(), is_firsts=is_firsts)
        err = max_rel_err(priors["deter"], priors_full["deter"])

        policy = shared_kv_test.Policy(2 * 2 + 16, 3).double()
        tssm.imag_start_chunk_size = 256
        prev_state, f, fh = tssm.imagination_start(copy.copy(posts), is_firsts, shared_kv=True)
        img = tssm.imagine(p_net=policy, prev_state=prev_state, img_steps=3, is_firsts=f, is_firsts_hidden=fh)
        finite = bool(torch.isfinite(img["deter"]).all())

    report(failed_2048 and err < 1e-10 and finite, "3. max_pos_encoding, L = C = {}".format(L),
           "max_pos=2048 fails: {}, chunked vs full observe rel err {:.1e}, imagination finite: {}".format(failed_2048, err, finite))


###############################################################################
# 4. Actor imagination without autograd graph (reinforce)
###############################################################################

def build_actor(no_grad, C=16, hidden=32, A=4, stoch=4, discrete=4, H=5):

    torch.manual_seed(0)
    feat = stoch * discrete + hidden
    outer = TWISTER.__new__(TWISTER)
    TWISTER.__bases__[0].__init__(outer, name="test")
    outer.config = AttrDict(H=H, target_value_reg=True, actor_grad="reinforce", imag_no_grad_reinforce=no_grad, gamma=0.997, lambda_td=0.95,
                            return_norm_decay=0.99, return_norm_limit=1.0, return_norm_perc_low=0.05, return_norm_perc_high=0.95, eta_entropy=3e-4)
    outer.rssm = tn.TSSM(num_actions=A, stoch_size=stoch, discrete=discrete, hidden_size=hidden, num_blocks=2, num_heads=4, drop_rate=0.1, att_context_left=C, ff_ratio=2).double()
    outer.policy_network = tn.PolicyNetwork(num_actions=A, hidden_size=hidden, feat_size=feat, num_mlp_layers=2, discrete=True).double()
    outer.value_network = tn.ValueNetwork(hidden_size=hidden, feat_size=feat, num_mlp_layers=2).double()
    outer.reward_network = tn.RewardNetwork(hidden_size=hidden, feat_size=feat, num_mlp_layers=2).double()
    outer.continue_network = tn.ContinueNetwork(hidden_size=hidden, feat_size=feat, num_mlp_layers=2).double()
    outer.v_target = copy.deepcopy(outer.value_network)
    outer.register_buffer("perc_low", torch.tensor(0.0, dtype=torch.float64))
    outer.register_buffer("perc_high", torch.tensor(0.0, dtype=torch.float64))
    actor = TWISTER.ActorModel(outer)

    # Training step state: rssm in eval mode, frozen world model / critic, policy trainable
    outer.rssm.eval()
    for net in (outer.rssm, outer.value_network, outer.reward_network, outer.continue_network):
        net.requires_grad_(False)
    return outer, actor


def test_actor_no_grad(B=2, L=16):

    results = {}
    for no_grad in (False, True):
        outer, actor = build_actor(no_grad)
        states, actions, is_firsts = shared_kv_test.make_inputs(B, L, 4, 4, 4, {0: [5], 1: [0, 11]})
        states = {k: v.double() for k, v in states.items()}
        is_firsts = is_firsts.double()
        with torch.no_grad():
            posts, _ = outer.rssm.observe(states=states, prev_actions=actions.double().clone(), is_firsts=is_firsts)
        outer.detached_posts, outer.detached_is_firsts, outer.detached_is_firsts_hidden = outer.rssm.imagination_start(posts, is_firsts, shared_kv=True)

        torch.manual_seed(1)
        actor.reset_losses()
        saved, _ = saved_bytes(lambda: actor.forward((None, None, None, torch.zeros(B, L, dtype=torch.float64), is_firsts, None)))
        loss = actor.added_losses["actor"]["loss"]
        outer.policy_network.zero_grad()
        loss.backward()
        grads = {n: p.grad.clone() for n, p in outer.policy_network.named_parameters()}
        results[no_grad] = (loss.detach(), grads, saved, outer.detached_feats)

    (loss_g, grads_g, saved_g, feats_g), (loss_n, grads_n, saved_n, feats_n) = results[False], results[True]
    err = max([max_rel_err(grads_n[k], grads_g[k]) for k in grads_g] + [max_rel_err(loss_n, loss_g), max_rel_err(feats_n, feats_g)])
    report(err < 1e-12, "4. actor imagination no_grad (reinforce)",
           "loss / feats / policy grads max rel err {:.1e}, saved for backward {:.2f} MB -> {:.2f} MB".format(err, saved_g / 2**20, saved_n / 2**20))


###############################################################################
# 5. CNN checkpoint by chunks of frames
###############################################################################

def test_cnn_checkpoint(B=2, L=24, chunk=10):

    norm = {"class": "LayerNorm", "params": {"eps": 1e-3}}  # no float32 conversion: the test runs in float64
    out = {}
    for chunk_size in (None, chunk):
        torch.manual_seed(0)
        enc = tn.EncoderNetwork(dim_input_cnn=3, dim_cnn=8, cnn_norm=norm, stoch_size=4, discrete=4, checkpoint_chunk_size=chunk_size).double()
        dec = tn.DecoderNetwork(dim_output_cnn=3, feat_size=16, dim_cnn=8, cnn_norm=norm, checkpoint_chunk_size=chunk_size).double()
        torch.manual_seed(1)
        x = torch.rand(B, L, 3, 64, 64, dtype=torch.float64) - 0.5

        def fn():
            torch.manual_seed(2)
            latent = enc(x)
            rec = dec(latent["stoch"].flatten(-2, -1))
            return -rec.log_prob(x).mean() + (latent["logits"] ** 2).mean(), latent, rec.mode()

        saved, (loss, latent, rec) = saved_bytes(fn)
        loss.backward()
        grads = {"enc." + n: p.grad.clone() for n, p in enc.named_parameters()}
        grads.update({"dec." + n: p.grad.clone() for n, p in dec.named_parameters()})
        out[chunk_size] = (latent["logits"].detach(), rec.detach(), grads, saved)

    (lg_f, rec_f, g_f, s_f), (lg_c, rec_c, g_c, s_c) = out[None], out[chunk]
    err = max([max_rel_err(lg_c, lg_f), max_rel_err(rec_c, rec_f)] + [max_rel_err(g_c[k], g_f[k]) for k in g_f])
    report(err < 1e-10, "5. CNN checkpoint by frame chunks",
           "outputs / grads max rel err {:.1e}, saved for backward {:.1f} MB -> {:.1f} MB ({} frames, chunk {})".format(err, s_f / 2**20, s_c / 2**20, B * L, chunk))


###############################################################################
# 8. InfoNCE by chunks of rows
###############################################################################

def original_contrastive_loss(features_x, features_y):
    # Verbatim TWISTER.compute_contrastive_loss before the change
    features_x = features_x.flatten(start_dim=0, end_dim=1)
    features_y = features_y.flatten(start_dim=0, end_dim=1)
    features = features_x.matmul(features_y.transpose(0, 1))
    features_pos = torch.diag(features)
    features_all = torch.logsumexp(features, dim=-1)
    info_nce_loss = features_pos - features_all
    acc_con = torch.mean(torch.where(features.argmax(dim=-1).cpu() == torch.arange(0, features.shape[0]), 1.0, 0.0))
    return info_nce_loss, acc_con


def test_info_nce(B=3, L=50, D=16, chunk=37):

    torch.manual_seed(0)
    x0, y0 = torch.randn(B, L, D, dtype=torch.float64), torch.randn(B, L, D, dtype=torch.float64)
    y0[:, ::3] = x0[:, ::3] + 0.1 * y0[:, ::3]  # some correct matches for a non trivial accuracy
    res = {}
    for name, fn in (("orig", original_contrastive_loss), ("full", lambda x, y: info_nce_loss(x, y)), ("chunk", lambda x, y: info_nce_loss(x, y, chunk_size=chunk))):
        x, y = x0.clone().requires_grad_(), y0.clone().requires_grad_()
        saved, (nce, acc) = saved_bytes(lambda: fn(x, y))
        (-(nce * torch.linspace(0.5, 1.5, nce.numel(), dtype=torch.float64)).mean()).backward()
        res[name] = (nce.detach(), acc, x.grad, y.grad, saved)

    ref = res["orig"]
    err = max(max_rel_err(res[k][i], ref[i]) for k in ("full", "chunk") for i in (0, 2, 3))
    acc_ok = all(res[k][1].item() == ref[1].item() for k in ("full", "chunk"))
    report(err < 1e-12 and acc_ok, "8. InfoNCE by row chunks",
           "values / grads max rel err {:.1e}, acc {:.3f} equal: {}, saved {:.0f} KB -> {:.0f} KB (B'={}, chunk {})".format(err, ref[1].item(), acc_ok, ref[4] / 1024, res["chunk"][4] / 1024, B * L, chunk))


###############################################################################
# 9. Shared kv imagination by chunks of start points
###############################################################################

def test_imagination_start_chunks():

    ok = True
    details = []
    for L, C, H, chunk, firsts in ((12, 5, 7, 5, {0: [0, 6], 1: [3, 4], 2: []}), (16, 8, 15, 3, {0: [], 1: [7], 2: [15]}), (20, 20, 4, 7, {0: [2], 1: [19], 2: [0, 9]}), (9, 2, 6, 1, {0: [3], 1: [], 2: [8]})):
        torch.manual_seed(0)
        tssm = tn.TSSM(num_actions=3, stoch_size=4, discrete=4, hidden_size=32, num_blocks=2, num_heads=4, drop_rate=0.0, att_context_left=C, ff_ratio=2).double().eval()
        policy = shared_kv_test.Policy(48, 3).double()
        states, actions, is_firsts = shared_kv_test.make_inputs(3, L, 3, 4, 4, firsts)
        states = {k: v.double() for k, v in states.items()}
        is_firsts = is_firsts.double()
        with torch.no_grad():
            posts, _ = tssm.observe(states=states, prev_actions=actions.double().clone(), is_firsts=is_firsts)

        img_ref, grads_ref = shared_kv_test.run_imagination(tssm, policy, copy.copy(posts), is_firsts, H, shared_kv=False, seed=1)
        tssm.imag_start_chunk_size = chunk
        img_new, grads_new = shared_kv_test.run_imagination(tssm, policy, copy.copy(posts), is_firsts, H, shared_kv=True, seed=1)
        tssm.imag_start_chunk_size = None
        err = max([max_rel_err(img_new[k], img_ref[k]) for k in img_ref] + [max_rel_err(grads_new[k], grads_ref[k]) for k in grads_ref])
        ok &= err < 1e-10
        details.append("L={} C={} chunk={}: {:.1e}".format(L, C, chunk, err))
    report(ok, "9. imagination start chunks vs original", ", ".join(details))


if __name__ == "__main__":
    test_episode_replay_buffer()
    test_env_step_balance()
    test_max_pos_encoding()
    test_actor_no_grad()
    test_cnn_checkpoint()
    test_info_nce()
    test_imagination_start_chunks()
    print("\n{} / {} passed".format(sum(RESULTS), len(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
