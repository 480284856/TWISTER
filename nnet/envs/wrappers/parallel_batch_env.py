# Batch of environments stepped in parallel subprocesses.
#
# BatchEnv steps its environments one after the other in the training process. With slow simulators
# (Memory Maze: ~10-15 ms per step on CPU) and many environments, env stepping dominates the training time.
# ParallelBatchEnv runs every environment in its own process (spawn start method: MuJoCo / OpenGL are not fork safe)
# and steps them all at once, with the same interface and outputs as BatchEnv.

# PyTorch
import torch

# NeuralNets
from nnet import structs

# Other
import atexit
import pickle
import traceback
import multiprocessing


# Plain pickle over the pipe: Connection.send uses ForkingPickler, which moves torch tensors through shared memory
# (one file descriptor / shm segment per tensor, too many for small per step tensors)
def _send(conn, obj):
    conn.send_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def _recv(conn):
    return pickle.loads(conn.recv_bytes())


def make_env(env_class, env_params, time_limit=None, reset_on_exception=True):

    """Picklable env factory: ResetOnException(TimeLimit(env_class(**env_params)))"""

    from nnet.envs import wrappers

    env = env_class(**env_params)
    if time_limit is not None:
        env = wrappers.TimeLimit(env, time_limit=time_limit)
    if reset_on_exception:
        env = wrappers.ResetOnException(env)
    return env


def _worker(conn, env_fn, eager_reset):

    # One thread per worker: many workers share the CPU
    torch.set_num_threads(1)

    try:
        env = env_fn()
        _send(conn, ("ok", {
            "num_actions": env.num_actions,
            "action_repeat": env.action_repeat,
            "obs_space": env.obs_space(),
            "clip_low": getattr(env, "clip_low", None),
            "clip_high": getattr(env, "clip_high", None),
            "fps": getattr(env, "fps", None),
        }))
    except Exception:
        _send(conn, ("error", traceback.format_exc()))
        conn.close()
        return

    # Reset prepared right after the last step of an episode, returned by the next reset call
    pending_reset = None

    # The trainer process is gone (killed, requeued, Ctrl-C): exit quietly
    trainer_gone = (EOFError, BrokenPipeError, ConnectionResetError, KeyboardInterrupt)

    while True:
        try:
            command, data = _recv(conn)
        except trainer_gone:
            break

        try:
            if command == "step":
                obs = env.step(data)
                _send(conn, ("ok", obs))
                # Episode finished: the env can only be reset now, start it while the trainer processes the step
                if eager_reset and bool(obs.is_last):
                    pending_reset = env.reset()

            elif command == "reset":
                obs = pending_reset if pending_reset is not None else env.reset()
                pending_reset = None
                _send(conn, ("ok", obs))

            elif command == "sample":
                _send(conn, ("ok", env.sample()))

            elif command == "close":
                _send(conn, ("ok", None))
                break

            else:
                _send(conn, ("error", "unknown command {}".format(command)))

        except trainer_gone:
            break

        except Exception:
            try:
                _send(conn, ("error", traceback.format_exc()))
            except trainer_gone:
                break

    conn.close()


class _EnvProxy:

    """Single env handle (BatchEnv.envs[i]), commands are sent to the worker process"""

    def __init__(self, batch_env, index):
        self.batch_env = batch_env
        self.index = index

    def reset(self):
        return self.batch_env._call(self.index, "reset")

    def step(self, action):
        return self.batch_env._call(self.index, "step", action)

    def sample(self):
        return self.batch_env._call(self.index, "sample")

    def obs_space(self):
        return self.batch_env.obs_space


class ParallelBatchEnv:

    """Same interface as BatchEnv, each env runs in its own subprocess

    Args:
        env_fns: list of picklable callables creating the envs (e.g. functools.partial(make_env, ...))
        eager_reset: reset an env in its worker as soon as a step returns is_last (the next reset call returns it),
            so that resetting envs one after the other in the trainer does not wait for each reset
        start_method: multiprocessing start method, spawn (default) is required for MuJoCo / OpenGL envs
    """

    def __init__(self, env_fns, eager_reset=True, start_method="spawn"):

        context = multiprocessing.get_context(start_method)
        self.conns, self.processes = [], []
        for env_fn in env_fns:
            parent_conn, child_conn = context.Pipe()
            process = context.Process(target=_worker, args=(child_conn, env_fn, eager_reset), daemon=True)
            process.start()
            child_conn.close()
            self.conns.append(parent_conn)
            self.processes.append(process)
        self.closed = False
        atexit.register(self.close)

        # Env properties (from the workers)
        properties = [self._receive(i) for i in range(len(self.conns))]
        self.num_envs = len(self.conns)
        self.num_actions = properties[0]["num_actions"]
        self.action_repeat = properties[0]["action_repeat"]
        self.obs_space = properties[0]["obs_space"]
        self.num_states = len(self.obs_space)
        for key in ("clip_low", "clip_high", "fps"):
            if properties[0][key] is not None:
                setattr(self, key, properties[0][key])

        # BatchEnv.envs[i] handles
        self.envs = [_EnvProxy(self, i) for i in range(self.num_envs)]

    def _receive(self, index):
        try:
            status, data = _recv(self.conns[index])
        except EOFError:
            raise RuntimeError("env worker {} died (exit code {})".format(index, self.processes[index].exitcode))
        if status != "ok":
            raise RuntimeError("env worker {} failed:\n{}".format(index, data))
        return data

    def _call(self, index, command, data=None):
        _send(self.conns[index], (command, data))
        return self._receive(index)

    def _call_all(self, command, datas=None):
        # Send all commands first so that the workers run in parallel
        for i, conn in enumerate(self.conns):
            _send(conn, (command, None if datas is None else datas[i]))
        return [self._receive(i) for i in range(self.num_envs)]

    @staticmethod
    def _stack(outputs):
        batch_obs = structs.AttrDict()
        for key in outputs[0].keys():
            batch_obs[key] = torch.stack([obs[key] for obs in outputs], dim=0)
        return batch_obs

    def sample(self):
        return torch.stack(self._call_all("sample"), dim=0)

    def step(self, actions):
        return self._stack(self._call_all("step", [actions[i] for i in range(self.num_envs)]))

    def reset(self, env_i=None):

        # Reset Single env
        if env_i is not None:
            return self._call(env_i, "reset")

        return self._stack(self._call_all("reset"))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for conn in self.conns:
            try:
                _send(conn, ("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
