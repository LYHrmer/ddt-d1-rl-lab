"""Session-wide guard: nothing under tests/ may build a MuJoCo model or advance physics.

The claim "these tests need no robot model and run zero physics steps" used to
be a sentence in a docstring. This file makes it executable, in two layers,
because a runtime patch alone has a measured blind spot: code that did
`from mujoco import mj_step` before the patch holds the old reference and
sails past it. So the runtime layer is paired with a static scan
(tests/test_no_model_no_physics_guard.py) that does not depend on import order.

Runtime layer. At collection time, before any test module is imported, the
model-building and physics-advancing entry points are replaced with functions
that raise. A test that reaches one fails on the spot, naming the entry point.
`MjModel()` and `MjData()` already refuse direct construction, so only the
documented constructors and the stepping functions need replacing.

This guard is deliberately a hard failure, not a skip: the point is that a
future change which quietly starts integrating physics under tests/ cannot
pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The names live in a side-effect-free module so the static scan can use them
# without importing this file, whose import installs the runtime patch.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _forbidden import MODEL_CONSTRUCTORS, PHYSICS_FUNCTIONS  # noqa: E402


class ForbiddenInZeroPhysicsTests(AssertionError):
    """Raised when a zero-model, zero-physics test touches the simulator."""


def _refuse(entry_point: str):
    def refuse(*args, **kwargs):
        raise ForbiddenInZeroPhysicsTests(
            f'{entry_point} was called under tests/. These tests promise no robot model '
            'and zero physics steps; move this case to the full static checks in tools04/ '
            'which run on the workstation, or remove the call.')
    return refuse


def install() -> dict:
    """Replace the entry points and return what was replaced, for the guard test."""
    try:
        import mujoco
    except ImportError:
        return {}  # nothing to guard where MuJoCo is not even installed
    replaced = {}
    for name in PHYSICS_FUNCTIONS:
        if hasattr(mujoco, name):
            replaced[f'mujoco.{name}'] = getattr(mujoco, name)
            setattr(mujoco, name, _refuse(f'mujoco.{name}'))
    for name in MODEL_CONSTRUCTORS:
        if hasattr(mujoco.MjModel, name):
            replaced[f'mujoco.MjModel.{name}'] = getattr(mujoco.MjModel, name)
            setattr(mujoco.MjModel, name, staticmethod(_refuse(f'mujoco.MjModel.{name}')))
    return replaced


INSTALLED = install()
