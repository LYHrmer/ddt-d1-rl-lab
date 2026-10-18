"""Fixed yaw command feedforward; physical command/score remain unchanged."""
import numpy as np

GAIN = np.float32(1.5)
YAW_COLUMNS = np.arange(11)*57+8


def compensate(observation):
    obs = np.asarray(observation)
    if obs.shape != (627,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
        raise ValueError('finite actor627 float32 required')
    result = obs.copy()
    # Original encoder clips physical yaw to +/-1 and scales by .25.
    result[YAW_COLUMNS] = np.clip(obs[YAW_COLUMNS]*GAIN, -.25, .25)
    return result
