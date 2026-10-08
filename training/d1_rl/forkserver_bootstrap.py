"""Preload read-mostly libraries in the single-threaded forkserver.

Workers inherit their pages instead of importing a separate Torch runtime each.
No tensors, model, physics world, CUDA context or background task is created.
"""
import gc
import mujoco  # noqa: F401
import numpy  # noqa: F401
import stable_baselines3  # noqa: F401
import torch

torch.set_num_threads(1)
gc.collect()
gc.freeze()
