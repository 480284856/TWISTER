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

# Other
import os
import sys
import importlib

# MuJoCo rendering backend, chosen by dm_control when it is first imported (here, before memory_maze):
# EGL for headless GPU rendering on Linux (memory_maze default), GLFW on macOS (no EGL). Set MUJOCO_GL to override.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else "egl"


class MissingEnvPackage:

    """Placeholder for an env sub package whose dependencies are not installed

    Importing nnet does not require the dependencies of every environment (dm_control for dmc,
    gym 0.19 + atari-py for atari100k, memory-maze for memory_maze). The import error is raised
    when the missing environment is actually used.
    """

    def __init__(self, name, error):
        self._name = name
        self._error = error

    def __getattr__(self, attr):
        raise ImportError("nnet.envs.{} is not available, its dependencies failed to import: {!r}".format(self._name, self._error)) from self._error


def _optional_import(name):
    try:
        return importlib.import_module("." + name, __name__)
    except Exception as error: # ImportError, or e.g. a rendering backend that fails to load
        return MissingEnvPackage(name, error)


# Sub Packages
from . import wrappers
dm_control = _optional_import("dm_control")
atari = _optional_import("atari")
from . import memory_maze # memory_maze itself is only imported when a MemoryMazeEnv is created
