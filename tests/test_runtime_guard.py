"""Exercises the runtime half of the zero-model, zero-physics guard.

tests/conftest.py replaces the MuJoCo model constructors and stepping functions
with functions that raise. These tests prove that replacement happened and that
it names the entry point it blocked. The static half, which does not depend on
this layer, is in test_static_guard.py.
"""
from __future__ import annotations

import pytest

import conftest as guard

# --- the runtime layer, exercised --------------------------------------------

mujoco = pytest.importorskip('mujoco')


def test_runtime_guard_installed_every_entry_point():
    expected = {f'mujoco.{n}' for n in guard.PHYSICS_FUNCTIONS if hasattr(mujoco, n)}
    expected |= {f'mujoco.MjModel.{n}' for n in guard.MODEL_CONSTRUCTORS
                 if hasattr(mujoco.MjModel, n)}
    assert expected and set(guard.INSTALLED) == expected


@pytest.mark.parametrize('name', [n for n in guard.PHYSICS_FUNCTIONS])
def test_runtime_guard_blocks_physics_functions(name):
    if not hasattr(mujoco, name):
        pytest.skip(f'{name} absent in this MuJoCo build')
    with pytest.raises(guard.ForbiddenInZeroPhysicsTests, match=name):
        getattr(mujoco, name)(None, None)


@pytest.mark.parametrize('name', guard.MODEL_CONSTRUCTORS)
def test_runtime_guard_blocks_model_constructors(name):
    if not hasattr(mujoco.MjModel, name):
        pytest.skip(f'{name} absent in this MuJoCo build')
    with pytest.raises(guard.ForbiddenInZeroPhysicsTests, match=name):
        getattr(mujoco.MjModel, name)('anything')


def test_runtime_guard_leaves_read_only_helpers_alone():
    assert isinstance(mujoco.mj_versionString(), str)


def test_the_guard_failure_is_an_assertion_error_so_it_cannot_be_swallowed_as_a_skip():
    assert issubclass(guard.ForbiddenInZeroPhysicsTests, AssertionError)


def test_runtime_layer_alone_has_a_blind_spot_for_early_bound_references():
    """The reason a static layer exists, pinned as a fact rather than a claim.

    Code that bound the original function before the patch was installed is not
    stopped by it. This test takes the originals that install() saved, advances
    one physics step through them, and asserts that simulated time really moved.
    That is the measured evidence behind test_static_guard.py: if this test ever
    starts failing because the runtime guard became airtight, the static layer
    can be reconsidered; until then both layers are needed.

    The originals are used here only to build one trivial two-body model inside
    this single test, which is the one place the guard's own blind spot is
    demonstrated; nothing else under tests/ may do this.
    """
    original_step = guard.INSTALLED.get('mujoco.mj_step')
    original_make = guard.INSTALLED.get('mujoco.MjModel.from_xml_string')
    if original_step is None or original_make is None:
        pytest.skip('MuJoCo build lacks the entry points this demonstration needs')

    class EarlyHolder:  # stands in for third-party code imported before the patch
        step = staticmethod(original_step)
        make = staticmethod(original_make)

    model = EarlyHolder.make('<mujoco><worldbody><geom size="1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    before = data.time
    EarlyHolder.step(model, data)
    assert data.time > before, 'the early-bound reference should have advanced physics'
