# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Solve dm_control bug: EGL headless rendering (Linux), GLFW on macOS (no EGL), unless MUJOCO_GL is already set
import os
import sys
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

# PyTorch
import torch

# Functions
import functions

# Other
import os
import argparse
import importlib
import warnings

# Disable Warnings
warnings.filterwarnings("ignore")

def main(args):

    ###############################################################################
    # Init
    ###############################################################################

    # Print Mode
    print("Mode: {}".format(args.mode))

    # Load Config
    # Load the objects defined in a .py file.
    args.config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))

    # Load Model
    model = functions.load_model(args)

    # Load Dataset
    # dataset_train: replay buffer, inserting trajectories while training
    # dataset_eval: fake dataloader, outputs fixed number of episodes (nnet.models.twister.__init__:self.config.eval_episodes)
    dataset_train, dataset_eval = functions.load_datasets(args)

    ###############################################################################
    # Modes
    ###############################################################################

    # Training
    if args.mode == "training":

        model.fit(
            dataset_train=dataset_train, 
            epochs=getattr(args.config, "epochs", 1000),                                                            # nnet.models.twister.__init__:self.config.epochs
            dataset_eval=dataset_eval, 
            initial_epoch=int(args.checkpoint.split("_")[2]) if args.checkpoint != None else 0,                     # checkpoints_epoch_48_step_19182.ckpt
            callback_path=args.config.callback_path,
            precision=getattr(args.config, "precision", torch.float32),                                             # nnet.models.twister.__init__:self.config.epochs
            accumulated_steps=getattr(args.config, "accumulated_steps", 1),                                         # gradient accumulation
            eval_period_step=getattr(args.config, "eval_period_step", args.eval_period_step),                       # how many steps of parameter updates between evaluations, default: None (defined by `eval_period_step` at the end of this file)
                                                                                                                    # Related call stacks:
                                                                                                                    #   main.py
                                                                                                                    #   └─ model.fit(...)
                                                                                                                    #      └─ model.py → Model.fit()              进入训练循环
                                                                                                                    #         ├─ twister.py → TWISTER.train_step()  每个训练批次
                                                                                                                    #         │  ├─ model.py → Model.train_step()  世界模型
                                                                                                                    #         │  ├─ model.py → Model.train_step()  Actor
                                                                                                                    #         │  └─ model.py → Model.train_step()  Critic
                                                                                                                    #         └─ model.py → Model.on_step_end()   梯度累积完成后，检查是否触发评估

            eval_period_epoch=getattr(args.config, "eval_period_epoch", args.eval_period_epoch),                    # how many epochs between evaluations, default: 5 (defined by `eval_period_epoch` at the end of this file)
            saving_period_epoch=getattr(args.config, "saving_period_epoch", args.saving_period_epoch),              # how many epochs between model saving, default: 1 (defined by `saving_period_epoch` at the end of this file)
            log_figure_period_step=getattr(args.config, "log_figure_period_step", args.log_figure_period_step),     # how many steps of parameter updates between figure(TT-MBRL Fig 26) logging, default: None (defined by `log_figure_period_step` at the end of this file)
            log_figure_period_epoch=getattr(args.config, "log_figure_period_epoch", args.log_figure_period_epoch),  # log figure after every n epochs, default: 1 (defined by `log_figure_period_epoch` at the end of this file)
            step_log_period=args.step_log_period,                                                                   # log metrics(like loss, reward, etc.) after every n steps, default: 100 (defined by `step_log_period` at the end of this file)
            grad_init_scale=getattr(args.config, "grad_init_scale", 65536.0),                                       # multiply the initial gradient by this factor, and after backpropagation, divide the gradient by this factor after the optimizer step
            detect_anomaly=getattr(args.config, "detect_anomaly", args.detect_anomaly),                             # locate the problematic operation in the backward pass(only in debug mode).
            recompute_metrics=getattr(args.config, "recompute_metrics", False),                                     # not relevant to this project
            wandb_logging=args.wandb,
            verbose_progress_bar=args.verbose_progress_bar,
            keep_last_k=args.keep_last_k                                                                            # keep the last k checkpoints, default: 3 (defined by `keep_last_k` at the end of this file)
        )

    # Evaluation
    elif args.mode == "evaluation":

        model._evaluate(
            dataset_eval, 
            writer=None,
            recompute_metrics=getattr(args.config, "recompute_metrics", False),
            verbose_progress_bar=args.verbose_progress_bar,
        )

    # Pass
    elif args.mode == "pass":
        pass

if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",          type=str,   default="configs/twister.py",                                       help="Python configuration file containing model hyperparameters")
    parser.add_argument("-m", "--mode",                 type=str,   default="training", choices=["training", "evaluation", "pass"],     help="Mode: training, validation-clean, test-clean, eval_time-dev-clean, ...")
    parser.add_argument("-i", "--checkpoint",           type=str,   default=None,                                                       help="Load model from checkpoint name")
    parser.add_argument("--cpu",                        action="store_true",                                                            help="Load model on cpu")
    parser.add_argument("--load_last",                  action="store_true",                                                            help="Load last model checkpoint")
    parser.add_argument("--wandb",                      action="store_true",                                                            help="Initialize wandb logging")
    parser.add_argument("--verbose_progress_bar",       type=int,   default=1,                                                          help="Verbose level of progress bar display")

    # Training
    parser.add_argument("--saving_period_epoch",        type=int,   default=1,                                                          help="Model saving every 'n' epochs")
    parser.add_argument("--log_figure_period_step",     type=int,   default=None,                                                       help="Log figure every 'n' steps")
    parser.add_argument("--log_figure_period_epoch",    type=int,   default=1,                                                          help="Log figure every 'n' epochs")
    parser.add_argument("--step_log_period",            type=int,   default=100,                                                        help="Training step log period")
    parser.add_argument("--keep_last_k",                type=int,   default=3,                                                          help="Keep last k checkpoints")

    # Eval
    parser.add_argument("--eval_period_epoch",          type=int,   default=5,                                                          help="Model evaluation every 'n' epochs")
    parser.add_argument("--eval_period_step",           type=int,   default=None,                                                       help="Model evaluation every 'n' steps")

    # Info
    parser.add_argument("--show_dict",                  action="store_true",                                                            help="Show model dict summary")
    parser.add_argument("--show_modules",               action="store_true",                                                            help="Show model named modules")
    
    # Debug
    parser.add_argument("--detect_anomaly",             action="store_true",                                                            help="Enable or disable the autograd anomaly detection")
    
    # Parse Args
    args = parser.parse_args()

    # Run main
    main(args)
