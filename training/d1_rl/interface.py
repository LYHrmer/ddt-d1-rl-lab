"""Pure functions for the frozen D1 actor observation and P_V torque contract.

The input order follows evidence/interface_audit_03/golden_run_01/schema.json.
The new policy clips actions to [-1, 1] before both last_actions and delay FIFO.
No MuJoCo simulation or third-party policy is used in this module.
"""

from __future__ import annotations

import numpy as np


JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint",
)
WHEEL_INDICES = np.array([3, 7, 11, 15], dtype=np.int64)
LEG_INDICES = np.array([i for i in range(16) if i not in (3, 7, 11, 15)], dtype=np.int64)
DEFAULT_Q = np.tile(np.array([0.0, 0.8, -1.5, 0.0], dtype=np.float64), 4)
KP = np.tile(np.array([60.0, 60.0, 60.0, 0.0], dtype=np.float64), 4)
KD = np.tile(np.array([1.5, 1.5, 1.5, 0.5], dtype=np.float64), 4)
ACTION_SCALE = np.tile(np.array([0.25, 0.25, 0.25, 5.0], dtype=np.float64), 4)
TORQUE_LIMIT = np.tile(np.array([80.0, 80.0, 80.0, 12.0], dtype=np.float64), 4)
OBS_DIM = 57
HISTORY_LENGTH = 10
ACTOR_DIM = OBS_DIM * (HISTORY_LENGTH + 1)
ACTION_DIM = 16


def clip_action(action: np.ndarray) -> np.ndarray:
    """Validate policy output and return a clipped float32 action."""
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (ACTION_DIM,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Expected finite action with shape ({ACTION_DIM},)")
    return np.clip(value, -1.0, 1.0).astype(np.float32, copy=False)


def inverse_rotate(quat_wxyz: np.ndarray, vector_world: np.ndarray) -> np.ndarray:
    """Rotate a world vector into the body frame using a body-to-world quaternion."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vector_world, dtype=np.float64)
    if q.shape != (4,) or v.shape != (3,) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
        raise ValueError("Expected finite wxyz quaternion and 3D vector")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    q = q / norm
    w, xyz = q[0], q[1:]
    # R(q)^T v = v - 2w(q_xyz x v) + 2(q_xyz x (q_xyz x v)).
    cross = np.cross(xyz, v)
    return v - 2.0 * w * cross + 2.0 * np.cross(xyz, cross)


def build_observation(
    gyro_body: np.ndarray,
    quat_wxyz: np.ndarray,
    command_vx_vy_wz: np.ndarray,
    joint_q: np.ndarray,
    joint_dq: np.ndarray,
    last_clipped_action: np.ndarray,
    previous_filtered_gyro: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one 57D actor frame and the updated body-gyro filter state."""
    gyro = np.asarray(gyro_body, dtype=np.float64)
    previous = np.asarray(previous_filtered_gyro, dtype=np.float64)
    command = np.asarray(command_vx_vy_wz, dtype=np.float64)
    q = np.asarray(joint_q, dtype=np.float64)
    dq = np.asarray(joint_dq, dtype=np.float64)
    last = np.asarray(last_clipped_action, dtype=np.float64)
    if (gyro.shape != (3,) or previous.shape != (3,) or command.shape != (3,) or
            q.shape != (16,) or dq.shape != (16,) or last.shape != (16,)):
        raise ValueError("Incorrect D1 observation component shape")
    if not all(np.all(np.isfinite(item)) for item in (gyro, previous, command, q, dq, last)):
        raise ValueError("Non-finite D1 observation component")
    filtered = 0.97 * gyro + 0.03 * previous
    gravity_body = inverse_rotate(quat_wxyz, np.array([0.0, 0.0, -1.0]))
    joint_offset = q - DEFAULT_Q
    joint_offset[WHEEL_INDICES] = 0.0
    frame = np.concatenate((filtered * 0.25, gravity_body,
                            np.clip(command, -1.0, 1.0) * np.array([2.0, 2.0, 0.25]),
                            joint_offset, dq * 0.05, last)).astype(np.float32)
    assert frame.shape == (OBS_DIM,)
    if not np.all(np.isfinite(frame)):
        raise FloatingPointError("Non-finite D1 actor observation")
    return frame, filtered


def pack_actor_input(current: np.ndarray, previous_history: np.ndarray) -> np.ndarray:
    """Flatten current frame followed by ten previous frames, oldest to newest."""
    current = np.asarray(current, dtype=np.float32)
    history = np.asarray(previous_history, dtype=np.float32)
    if current.shape != (OBS_DIM,) or history.shape != (HISTORY_LENGTH, OBS_DIM):
        raise ValueError("Actor input requires (57,) current and (10,57) history")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(history)):
        raise ValueError("Non-finite actor input")
    return np.concatenate((current, history.reshape(-1))).astype(np.float32, copy=False)


def action_to_torque(
    clipped_executed_action: np.ndarray, joint_q: np.ndarray, joint_dq: np.ndarray,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """P_V command; default keeps pilot03, optional scale selects a frozen variant."""
    action = np.asarray(clipped_executed_action, dtype=np.float64)
    q = np.asarray(joint_q, dtype=np.float64)
    dq = np.asarray(joint_dq, dtype=np.float64)
    action_scale = ACTION_SCALE if scale is None else np.asarray(scale, dtype=np.float64)
    if action.shape != (16,) or q.shape != (16,) or dq.shape != (16,):
        raise ValueError("P_V inputs must each have shape (16,)")
    if action_scale.shape != (16,) or np.any(action_scale < 0):
        raise ValueError("P_V action scale must have 16 nonnegative entries")
    if not all(np.all(np.isfinite(x)) for x in (action, q, dq, action_scale)):
        raise ValueError("Non-finite P_V input")
    if np.any(np.abs(action) > 1.0 + 1e-7):
        raise ValueError("P_V action must already be clipped to [-1,1]")
    target_q = DEFAULT_Q + action_scale * action
    target_dq = np.zeros(16, dtype=np.float64)
    target_dq[WHEEL_INDICES] = action_scale[WHEEL_INDICES] * action[WHEEL_INDICES]
    raw = KP * (target_q - q) + KD * (target_dq - dq)
    saturation = np.abs(raw) >= TORQUE_LIMIT
    return np.clip(raw, -TORQUE_LIMIT, TORQUE_LIMIT), saturation
