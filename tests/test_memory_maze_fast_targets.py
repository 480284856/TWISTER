"""Vectorized TargetSphere activation (nnet/envs/memory_maze/memory_maze_env.py) vs the original dm_control loop.

    A. Same activation decision as the original loop for every target after every physics substep of long rollouts
    B. Same trajectories: observations, rewards, is_last identical with the original and the patched check
    C. Env step time, original vs patched

Run from the repository root (needs memory_maze):
    python tests/test_memory_maze_fast_targets.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import test_memory_maze_real as real  # noqa: E402  (nnet stubs)

import torch  # noqa: E402
from nnet.envs.memory_maze import memory_maze_env  # noqa: E402
from nnet.envs.memory_maze import MemoryMazeEnv  # noqa: E402

from dm_control.locomotion.props import target_sphere  # noqa: E402

ORIGINAL = target_sphere.TargetSphere._update_activation
FAST = memory_maze_env._fast_update_activation


def use(method):
    target_sphere.TargetSphere._update_activation = method


def original_decision(target, physics):
    # Activation condition of the original dm_control loop, without side effects
    for contact in physics.data.contact:
        if target._specific_collision_geom_ids:
            specific = contact.geom1 in target._specific_collision_geom_ids or contact.geom2 in target._specific_collision_geom_ids
        else:
            specific = True
        if specific and target._geom_id in (contact.geom1, contact.geom2):
            return True
    return False


def fast_decision(target, physics):
    # Run the vectorized method on a copy of the activation state
    activated, material = target._activated, target._material
    target._activated = False
    rgba = physics.bind(material).rgba.copy()
    FAST(target, physics)
    decision = target._activated
    target._activated = activated
    physics.bind(material).rgba[:] = rgba
    return decision


def rollout(maze_size, seed, steps, compare_decisions=False):
    env = MemoryMazeEnv(maze_size=maze_size, seed=seed, fast_target_check=False)
    stats = {"checks": 0, "positives": 0, "mismatches": 0}

    if compare_decisions:
        original_after_substep = target_sphere.TargetSphere.after_substep

        def after_substep(self, physics, random_state):
            if not self._activated:
                a, b = original_decision(self, physics), fast_decision(self, physics)
                stats["checks"] += 1
                stats["positives"] += a
                stats["mismatches"] += a != b
            original_after_substep(self, physics, random_state)

        target_sphere.TargetSphere.after_substep = after_substep

    generator = torch.Generator().manual_seed(seed)
    trace = [env.reset()]
    for t in range(steps):
        obs = env.step(torch.randint(0, 6, (), generator=generator))
        trace.append(obs)
        if obs.is_last:
            trace.append(env.reset())

    if compare_decisions:
        target_sphere.TargetSphere.after_substep = original_after_substep
    return trace, stats


def test_decisions(maze_size=9, steps=3000):
    use(ORIGINAL)
    _, stats = rollout(maze_size, seed=1, steps=steps, compare_decisions=True)
    ok = stats["mismatches"] == 0 and stats["checks"] > 0
    real.report(ok, "A. activation decisions", "{} target checks over {} steps ({}x{}), {} activations, {} mismatches".format(
        stats["checks"], steps, maze_size, maze_size, stats["positives"], stats["mismatches"]))


def test_trajectories(maze_size=9, steps=3000):
    use(ORIGINAL)
    trace_original, stats = rollout(maze_size, seed=1, steps=steps, compare_decisions=True)  # seed 1: 5 target activations in 3000 steps
    use(FAST)
    trace_fast, _ = rollout(maze_size, seed=1, steps=steps)
    use(ORIGINAL)

    def flat(trace):
        return [torch.cat([o.state.flatten().float(), o.reward.flatten(), o.is_first.flatten(), o.is_last.flatten(), o.done.flatten()]) for o in trace]

    same = len(trace_original) == len(trace_fast) and all(torch.equal(a, b) for a, b in zip(flat(trace_original), flat(trace_fast)))
    rewards = sum(o.reward.item() for o in trace_original)
    real.report(same, "B. trajectories identical", "{} steps ({}x{}, episode resets included), {} target activations, total reward {:.0f}: identical {}".format(steps, maze_size, maze_size, stats["positives"], rewards, same))


def test_speed(maze_size=15, steps=300):
    times = {}
    for name, method in (("original", ORIGINAL), ("vectorized", FAST)):
        use(method)
        env = MemoryMazeEnv(maze_size=maze_size, seed=0, fast_target_check=False)
        env.reset()
        t0 = time.perf_counter()
        for t in range(steps):
            env.step(env.sample().argmax())
        times[name] = (time.perf_counter() - t0) / steps
    use(ORIGINAL)
    real.report(times["vectorized"] < times["original"], "C. env step time {}x{}".format(maze_size, maze_size),
                "original {:.1f} ms, vectorized {:.1f} ms ({:.1f}x)".format(1000 * times["original"], 1000 * times["vectorized"], times["original"] / times["vectorized"]))


if __name__ == "__main__":
    test_decisions()
    test_trajectories()
    test_speed()
    print("\n{} / {} passed".format(sum(real.RESULTS), len(real.RESULTS)))
    sys.exit(0 if all(real.RESULTS) else 1)
