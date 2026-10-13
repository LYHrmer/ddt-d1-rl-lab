"""Unit tests for the frozen D1 actor interface and the contract 04 reward terms.

Every case here is pure arithmetic: no MuJoCo model is loaded, no environment is
built and no physics step runs, so this suite works from a clean clone without
the robot assets that the repository does not redistribute. It is the executable
form of the interface properties the documents assert, so a silent change to
observation layout, torque mapping or reward shape fails here rather than
surfacing a thousand native steps later.

Run with:
    PYTHONPATH=training python3 -m pytest tests -q
"""
from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))

from d1_rl.interface import (  # noqa: E402
    ACTION_DIM,
    ACTION_SCALE,
    ACTOR_DIM,
    DEFAULT_Q,
    HISTORY_LENGTH,
    KD,
    KP,
    LEG_INDICES,
    OBS_DIM,
    TORQUE_LIMIT,
    WHEEL_INDICES,
    action_to_torque,
    build_observation,
    clip_action,
    inverse_rotate,
    pack_actor_input,
)

IDENTITY_QUAT = np.array([1., 0., 0., 0.])


# --- interface constants the schema and the C++ runtime both depend on -------

def test_dimensions_match_the_published_interface():
    assert (OBS_DIM, HISTORY_LENGTH, ACTION_DIM) == (57, 10, 16)
    assert ACTOR_DIM == OBS_DIM*(HISTORY_LENGTH+1) == 627


def test_wheels_are_the_foot_joints_and_carry_the_only_velocity_targets():
    # runtime03/schema03d.json: *_foot_joint has kp 0, kd .5, scale 5, limit 12.
    assert tuple(WHEEL_INDICES) == (3, 7, 11, 15)
    assert set(LEG_INDICES).isdisjoint(WHEEL_INDICES)
    assert len(LEG_INDICES)+len(WHEEL_INDICES) == ACTION_DIM
    assert np.all(KP[WHEEL_INDICES] == 0.)
    assert np.all(KP[list(LEG_INDICES)] == 60.)
    assert np.all(KD[WHEEL_INDICES] == .5)
    assert np.all(ACTION_SCALE[WHEEL_INDICES] == 5.)
    assert np.all(TORQUE_LIMIT[WHEEL_INDICES] == 12.)
    assert np.all(TORQUE_LIMIT[list(LEG_INDICES)] == 80.)


# --- clip_action -------------------------------------------------------------

def test_clip_action_clips_and_returns_float32():
    out = clip_action(np.array([2., -2.]+[0.]*14))
    assert out.dtype == np.float32
    assert out[0] == 1. and out[1] == -1.


@pytest.mark.parametrize('bad', [np.zeros(15), np.zeros(17), np.full(16, np.nan),
                                 np.full(16, np.inf)])
def test_clip_action_rejects_bad_shape_or_non_finite(bad):
    with pytest.raises(ValueError):
        clip_action(bad)


# --- inverse_rotate ---------------------------------------------------------

def test_inverse_rotate_identity_is_a_no_op():
    v = np.array([1., 2., 3.])
    assert np.allclose(inverse_rotate(IDENTITY_QUAT, v), v, atol=0, rtol=0)


def test_inverse_rotate_undoes_a_known_yaw():
    # Body yawed +90 deg: a world +x vector reads as body -y.
    half = math.pi/4
    quat = np.array([math.cos(half), 0., 0., math.sin(half)])
    assert np.allclose(inverse_rotate(quat, np.array([1., 0., 0.])),
                       np.array([0., -1., 0.]), atol=1e-12)


def test_inverse_rotate_preserves_length_and_tolerates_unnormalised_input():
    quat = np.array([2., 0., 0., 2.])  # same rotation, norm 2 sqrt 2
    v = np.array([.3, -.7, 1.1])
    assert np.isclose(np.linalg.norm(inverse_rotate(quat, v)), np.linalg.norm(v))


def test_inverse_rotate_rejects_a_zero_quaternion():
    with pytest.raises(ValueError):
        inverse_rotate(np.zeros(4), np.array([1., 0., 0.]))


# --- build_observation ------------------------------------------------------

def _observation(**overrides):
    kwargs = {'gyro_body': np.zeros(3), 'quat_wxyz': IDENTITY_QUAT,
              'command_vx_vy_wz': np.zeros(3), 'joint_q': DEFAULT_Q.copy(),
              'joint_dq': np.zeros(16), 'last_clipped_action': np.zeros(16),
              'previous_filtered_gyro': np.zeros(3)}
    kwargs.update(overrides)
    return build_observation(**kwargs)


def test_observation_layout_and_scales():
    gyro = np.array([.4, -.8, 1.2])
    command = np.array([.3, -.2, .1])
    dq = np.arange(16, dtype=np.float64)/10.
    last = np.linspace(-1., 1., 16)
    frame, filtered = _observation(gyro_body=gyro, command_vx_vy_wz=command,
                                   joint_dq=dq, last_clipped_action=last)
    assert frame.shape == (OBS_DIM,) and frame.dtype == np.float32
    # 3 filtered gyro, 3 body gravity, 3 command, 16 joint offset, 16 dq, 16 action.
    assert np.allclose(filtered, .97*gyro, atol=1e-12)
    assert np.allclose(frame[0:3], filtered*.25, atol=1e-6)
    assert np.allclose(frame[3:6], np.array([0., 0., -1.]), atol=1e-6)
    assert np.allclose(frame[6:9], command*np.array([2., 2., .25]), atol=1e-6)
    assert np.allclose(frame[25:41], dq*.05, atol=1e-6)
    assert np.allclose(frame[41:57], last, atol=1e-6)


def test_observation_zeroes_the_wheel_joint_offsets():
    # Wheels spin without limit, so their absolute angle must not reach the actor.
    q = DEFAULT_Q.copy()
    q[list(WHEEL_INDICES)] = [11., -7., 3., 99.]
    q[1] += .5
    frame, _ = _observation(joint_q=q)
    offsets = frame[9:25]
    assert np.all(offsets[list(WHEEL_INDICES)] == 0.)
    assert np.isclose(offsets[1], .5, atol=1e-6)


def test_observation_clips_the_command_before_scaling():
    frame, _ = _observation(command_vx_vy_wz=np.array([5., -5., 5.]))
    assert np.allclose(frame[6:9], np.array([2., -2., .25]), atol=1e-6)


def test_observation_carries_no_base_position_or_linear_velocity():
    """Pin the deployable inputs; no base position, linear velocity or terrain."""
    assert tuple(inspect.signature(build_observation).parameters) == (
        'gyro_body', 'quat_wxyz', 'command_vx_vy_wz', 'joint_q', 'joint_dq',
        'last_clipped_action', 'previous_filtered_gyro',
    )
    for unavailable in ('base_position', 'base_linear_velocity', 'terrain'):
        with pytest.raises(TypeError, match=unavailable):
            _observation(**{unavailable: np.zeros(3)})


def test_observation_gyro_filter_is_a_first_order_blend():
    gyro, previous = np.array([1., 0., 0.]), np.array([0., 1., 0.])
    _, filtered = _observation(gyro_body=gyro, previous_filtered_gyro=previous)
    assert np.allclose(filtered, .97*gyro+.03*previous, atol=1e-12)


@pytest.mark.parametrize('overrides', [
    {'gyro_body': np.zeros(2)}, {'command_vx_vy_wz': np.zeros(4)},
    {'joint_q': np.zeros(15)}, {'joint_dq': np.zeros(17)},
    {'last_clipped_action': np.zeros(15)},
    {'joint_dq': np.full(16, np.nan)},
])
def test_observation_rejects_bad_components(overrides):
    with pytest.raises(ValueError):
        _observation(**overrides)


# --- pack_actor_input -------------------------------------------------------

def test_pack_actor_input_puts_the_current_frame_first():
    current = np.full(OBS_DIM, 7., dtype=np.float32)
    history = np.arange(HISTORY_LENGTH*OBS_DIM, dtype=np.float32).reshape(
        HISTORY_LENGTH, OBS_DIM)
    packed = pack_actor_input(current, history)
    assert packed.shape == (ACTOR_DIM,) and packed.dtype == np.float32
    assert np.array_equal(packed[:OBS_DIM], current)
    # oldest to newest, so row 0 lands immediately after the current frame
    assert np.array_equal(packed[OBS_DIM:2*OBS_DIM], history[0])
    assert np.array_equal(packed[-OBS_DIM:], history[-1])


@pytest.mark.parametrize('current,history', [
    (np.zeros(OBS_DIM-1), np.zeros((HISTORY_LENGTH, OBS_DIM))),
    (np.zeros(OBS_DIM), np.zeros((HISTORY_LENGTH-1, OBS_DIM))),
    (np.full(OBS_DIM, np.nan), np.zeros((HISTORY_LENGTH, OBS_DIM))),
])
def test_pack_actor_input_rejects_bad_shapes_or_non_finite(current, history):
    with pytest.raises(ValueError):
        pack_actor_input(current, history)


# --- action_to_torque -------------------------------------------------------

def test_zero_action_at_the_default_pose_produces_no_torque():
    torque, saturation = action_to_torque(np.zeros(16), DEFAULT_Q.copy(), np.zeros(16))
    assert np.allclose(torque, 0., atol=0, rtol=0)
    assert not saturation.any()


def test_wheel_torque_is_pure_damping_towards_a_velocity_target():
    """tau = kd (scale a - omega), the relation the C++ trace audit recovered."""
    action = np.zeros(16)
    action[WHEEL_INDICES[0]] = .4           # 0.4 x 5 = 2 rad/s target
    dq = np.zeros(16)
    dq[WHEEL_INDICES[0]] = .5
    torque, _ = action_to_torque(action, DEFAULT_Q.copy(), dq)
    assert np.isclose(torque[WHEEL_INDICES[0]], .5*(2.-.5))
    # A wheel action must not leak into any leg channel.
    assert np.allclose(torque[list(LEG_INDICES)], 0., atol=0, rtol=0)


def test_wheel_position_error_is_ignored_because_kp_is_zero():
    q = DEFAULT_Q.copy()
    q[WHEEL_INDICES[1]] += 12.3
    torque, _ = action_to_torque(np.zeros(16), q, np.zeros(16))
    assert torque[WHEEL_INDICES[1]] == 0.


def test_leg_torque_is_position_pd_about_the_default_pose():
    action = np.zeros(16)
    action[1] = .5                           # 0.5 x 0.25 = 0.125 rad offset
    q = DEFAULT_Q.copy()
    dq = np.zeros(16)
    dq[1] = .2
    torque, _ = action_to_torque(action, q, dq)
    expected = KP[1]*(ACTION_SCALE[1]*.5)+KD[1]*(0.-.2)
    assert np.isclose(torque[1], expected)


def test_torque_is_clipped_and_saturation_is_reported():
    action = np.zeros(16)
    action[1] = 1.                           # 0.25 rad error x kp 60 = 15 N m
    q = DEFAULT_Q.copy()
    q[2] -= 10.                              # drives the calf channel far past 80
    torque, saturation = action_to_torque(action, q, np.zeros(16))
    assert np.all(np.abs(torque) <= TORQUE_LIMIT+1e-12)
    assert saturation[2] and torque[2] == TORQUE_LIMIT[2]
    assert not saturation[1]


def test_action_to_torque_accepts_a_frozen_variant_scale():
    scale = ACTION_SCALE.copy()
    scale[WHEEL_INDICES] = 1.
    action = np.zeros(16)
    action[WHEEL_INDICES[0]] = 1.
    torque, _ = action_to_torque(action, DEFAULT_Q.copy(), np.zeros(16), scale=scale)
    assert np.isclose(torque[WHEEL_INDICES[0]], .5*1.)


@pytest.mark.parametrize('kwargs', [
    {'clipped_executed_action': np.zeros(15)},
    {'joint_q': np.zeros(15)},
    {'joint_dq': np.zeros(15)},
    {'clipped_executed_action': np.full(16, 1.5)},   # must already be clipped
    {'joint_q': np.full(16, np.nan)},
])
def test_action_to_torque_rejects_bad_inputs(kwargs):
    base = {'clipped_executed_action': np.zeros(16), 'joint_q': DEFAULT_Q.copy(),
            'joint_dq': np.zeros(16)}
    base.update(kwargs)
    with pytest.raises(ValueError):
        action_to_torque(**base)


def test_action_to_torque_rejects_a_negative_scale():
    scale = ACTION_SCALE.copy()
    scale[0] = -1.
    with pytest.raises(ValueError):
        action_to_torque(np.zeros(16), DEFAULT_Q.copy(), np.zeros(16), scale=scale)
