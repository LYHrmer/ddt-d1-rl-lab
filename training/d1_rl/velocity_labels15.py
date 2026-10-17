"""Add a training-only body-velocity target at the existing D1 sensor boundary.

Wrap the scalar Gymnasium environment before SB3's vector/autoreset layer.
``frame_vel`` must be the D1 world-frame velocity in m/s; the target is body
velocity in m/s, without actor normalization. Reading cached sensors does not
refresh sensors, construct a model, reset, or advance physics.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .interface import ACTOR_DIM, inverse_rotate

VELOCITY_LABEL_CONTRACT = {
    "actor": {"shape": [ACTOR_DIM], "dtype": "float32", "source": "unchanged base observation"},
    "velocity_target": {"shape": [3], "dtype": "float32", "units": "m/s", "frame": "body"},
    "source_velocity": {"sensor": "frame_vel", "units": "m/s", "frame": "world"},
    "source_quaternion": {"sensor": "trunk_quat", "order": "wxyz_body_to_world"},
    "time": "same cached sensor boundary as the returned base reset/step observation",
}


def _actor_copy(actor: np.ndarray) -> np.ndarray:
    value = np.asarray(actor)
    if value.shape != (ACTOR_DIM,) or value.dtype != np.float32 or not np.all(np.isfinite(value)):
        raise ValueError(f"Expected finite float32 actor observation with shape ({ACTOR_DIM},)")
    return value.copy()


def label_observation(
    actor: np.ndarray, quat: np.ndarray, velocity_world: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pure array labeling: wxyz body-to-world quaternion, world velocity in m/s.

    Actor values are copied exactly, and never concatenated with the target.
    Shape/finite checks cannot infer physical units: callers must supply the
    declared D1 sensor units, rather than scaled actor features or cm/s.
    """
    actor_copy = _actor_copy(actor)
    velocity_body = inverse_rotate(quat, velocity_world)
    with np.errstate(over="ignore", invalid="ignore"):
        target = np.asarray(velocity_body, dtype=np.float32)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Body velocity target is not representable as finite float32 m/s")
    return {"actor": actor_copy, "velocity_target": target.copy()}


class VelocityLabelWrapper(gym.Wrapper):
    """Keep D1 transitions intact and attach only their same-boundary label.

    This wrapper expects a non-autoresetting base environment. Its placement
    below DummyVecEnv preserves terminal targets before the outer vector layer
    resets. A base terminal_observation must describe the returned terminal
    actor frame; differing frames are refused because their sensor target is
    not available from the current cache.
    """

    velocity_target_units = "m/s"
    velocity_target_frame = "body"
    velocity_label_contract = VELOCITY_LABEL_CONTRACT

    def __init__(self, env: gym.Env):
        super().__init__(env)
        actor_space = env.observation_space
        if (not isinstance(actor_space, gym.spaces.Box) or actor_space.shape != (ACTOR_DIM,)
                or actor_space.dtype != np.float32):
            raise ValueError(f"Base observation space must be float32 Box({ACTOR_DIM},)")
        if not callable(getattr(env.unwrapped, "_sensor", None)):
            raise TypeError("Base D1 environment must expose cached _sensor(name)")
        self.observation_space = gym.spaces.Dict({
            "actor": actor_space,
            "velocity_target": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

    def _label(self, actor):
        base = self.env.unwrapped
        # D1 reset/step already called _observe; never call it a second time.
        return label_observation(actor, base._sensor("trunk_quat"), base._sensor("frame_vel"))

    def reset(self, **kwargs):
        actor, info = self.env.reset(**kwargs)
        return self._label(actor), info

    def step(self, action):
        actor, reward, terminated, truncated, info = self.env.step(action)
        labelled = self._label(actor)
        if (terminated or truncated) and "terminal_observation" in info:
            terminal_actor = _actor_copy(info["terminal_observation"])
            if not np.array_equal(terminal_actor, labelled["actor"]):
                raise ValueError("Base terminal actor differs from returned actor; cached label would be misaligned")
            info = dict(info)
            info["terminal_observation"] = {
                "actor": terminal_actor,
                "velocity_target": labelled["velocity_target"].copy(),
            }
        return labelled, reward, terminated, truncated, info
