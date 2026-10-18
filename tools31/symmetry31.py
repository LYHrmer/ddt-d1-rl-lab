"""Lateral group projection for D1's current57 + ten previous57 actor."""
import numpy as np

ORDER = np.array([4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11])
SIGN = np.tile(np.array([-1,1,1,1], dtype=np.float32), 4)


def mirror_action(value):
    action = np.asarray(value)
    if action.shape != (16,) or action.dtype != np.float32 or not np.isfinite(action).all():
        raise ValueError('finite float32 interleaved action16 required')
    return action[ORDER]*SIGN


def mirror_observation(value):
    obs = np.asarray(value)
    if obs.shape != (627,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
        raise ValueError('finite float32 actor627 required')
    frames = obs.reshape(11,57)
    result = frames.copy()
    result[:, [0,2,4,7,8]] *= -1  # gyro axial X/Z, gravity Y, command vy/yaw
    for start in (9,25,41):  # joint offset, velocity and issued action, all interleaved
        result[:, start:start+16] = frames[:, start+ORDER]*SIGN
    return result.reshape(627)


def combine(original, reflected):
    a, b = np.asarray(original), np.asarray(reflected)
    # Both components must be legal original network outputs before averaging.
    for value in (a,b):
        mirror_action(value)
        if np.any(abs(value)>1):
            raise ValueError('unclipped actor action')
    return ((a.astype(np.float64)+mirror_action(b).astype(np.float64))*.5).astype(np.float32)
