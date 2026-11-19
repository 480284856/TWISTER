import nnet
import os
import json

# Extract params from filename
env_name = os.environ["env_name"]
print("TWISTER selected env_name: {}".format(env_name))

# Override Config
override_config = os.environ.get("override_config", {})
if isinstance(override_config, str):
    override_config = json.loads(override_config)
print("override_config:", override_config)

# Model
model = nnet.models.TWISTER(env_name=env_name, override_config=override_config)
model.compile()

# Training
precision = model.config.precision
grad_init_scale = model.config.grad_init_scale
epochs = model.config.epochs
epoch_length = model.config.epoch_length

# Callback Path
if os.environ.get("run_name", False):
    callback_path = "callbacks/{}/{}".format(os.environ["run_name"], env_name)
else:
    callback_path = "callbacks/{}".format(env_name)

# Replay Buffer
# window: original TWISTER buffer (one window per env step), episode: complete episodes, windows start at episode starts
replay_buffer_class = {"window": nnet.datasets.ReplayBuffer, "episode": nnet.datasets.EpisodeReplayBuffer}[model.config.replay_buffer_mode]
training_dataset = replay_buffer_class(
    batch_size=model.config.batch_size,
    root=callback_path,
    buffer_capacity=model.config.buffer_capacity,
    epoch_length=epoch_length,
    sample_length=model.config.L
)
model.set_replay_buffer(training_dataset)

# Evaluation Dataset
evaluation_dataset = nnet.datasets.VoidDataset(num_steps=model.config.eval_episodes)
