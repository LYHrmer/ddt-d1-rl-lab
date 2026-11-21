"""Fresh MuJoCo experience for the D1 cold-start actor. One environment per episode."""
from __future__ import annotations

from operator import index

import numpy as np

from tools94.actor94 import predict
from training.d1_rl.interface import LEG_INDICES, WHEEL_INDICES, clip_action


# Endpoints include the reset. Interval row k links endpoint k to endpoint k+1.
ENDPOINTS = {
    "state45": (np.float64, (45,)), "obs643": (np.float32, (643,)),
    "filtered3": (np.float64, (3,)), "time_s": (np.float64, ()),
    "sensor_capture_tick": (np.int64, ()), "sensor_delivery_tick": (np.int64, ()),
    "ray_due": (bool, ()), "ray_raw16": (np.float32, (16,)),
    "pending_capture_tick": (np.int64, ()), "pending_ray16": (np.float32, (16,)),
}
INTERVALS = {
    "raw16": (np.float32, (16,)), "exploration16": (np.float32, (16,)),
    "issued16": (np.float32, (16,)), "applied16": (np.float32, (16,)),
    "support4": (bool, (4,)), "fallen": (bool, ()), "illegal": (bool, ()),
    "terminated": (bool, ()), "truncated": (bool, ()), "reward": (np.float64, ()),
    "command3": (np.float64, (3,)), "wheel_saturation_count": (np.int64, ()),
}


class NativeEpisodeError(RuntimeError):
    """A stopped collection, with its valid prefix and counters including attempted work."""

    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace


def _open_env(reset_seed, ticks):
    from tools89.course89 import make_env

    return make_env("up", reset_seed, curriculum=True, slope_start_x_m=1.10,
                    perception=True, max_transitions=ticks)


def _reset_uphill(env, reset_seed):
    _, first = env.reset(seed=reset_seed)
    obs, second = env.reset()
    if (first["episode_index"] != 0 or first["slope_start"] or second["episode_index"] != 1
            or not second["slope_start"] or second["start_x_m"] != 1.10 or second["delay_ticks"] != 1):
        raise ValueError("expected the second fresh reset on the 4-degree uphill with a one-tick FIFO")
    if (env.data.time != 0 or env.free_qpos != 0 or env.free_dof != 0
            or not np.array_equal(env.qpos_ids, np.arange(7, 23))
            or not np.array_equal(env.dof_ids, np.arange(6, 22))
            or not np.array_equal(env.action_scales, np.tile([0.5, 0.5, 0.5, 5.0], 4))):
        raise ValueError("D1 reset, state order or servo action scales differ")
    if any(env.counters()[k] for k in ("policy_transitions", "pd_updates", "native_steps")):
        raise ValueError("a native integration or policy decision occurred before the first sample")
    return obs


def _row(layout, **values):
    row = {}
    for name, (dtype, shape) in layout.items():
        value = np.asarray(values[name], dtype=dtype)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid {name} in native trace")
        row[name] = value.copy()
    return row


def _endpoint(env, obs, tick):
    pending_tick = -1
    pending_packet = np.zeros(16, dtype=np.float32)
    if env._pending is not None:
        pending_tick, pending_packet = env._pending
    due = pending_tick == tick
    if not np.array_equal(obs[627:], env.last_raw_features):
        raise ValueError("policy observation differs from the delivered range packet")
    return _row(ENDPOINTS, state45=np.concatenate((env.data.qpos, env.data.qvel)), obs643=obs,
                filtered3=env._filtered_gyro, time_s=env.data.time,
                sensor_capture_tick=env.sensor_capture_tick, sensor_delivery_tick=env.sensor_delivery_tick,
                ray_due=due, ray_raw16=pending_packet if due else np.zeros(16, dtype=np.float32),
                pending_capture_tick=pending_tick, pending_ray16=pending_packet)


def _columns(rows, layout):
    return {key: np.stack([row[key] for row in rows]) if rows else np.empty((0, *shape), dtype=dtype)
            for key, (dtype, shape) in layout.items()}


def _draw_exploration(rng):
    noise = np.empty(16, dtype=np.float32)
    noise[LEG_INDICES] = rng.uniform(-0.06, 0.06, size=len(LEG_INDICES))
    common = rng.uniform(-0.10, 0.10)
    noise[WHEEL_INDICES] = common + rng.uniform(-0.04, 0.04, size=len(WHEEL_INDICES))
    return noise


def collect_episode(actor, reset_seed, *, ticks=50, explore_seed=None, wheel_offset_rad_s=0.0,
                    progress=None) -> dict[str, np.ndarray]:
    """Collect a finite native prefix, including its terminal transition when one occurs.

    ``progress(trace)`` can persist the reset and each completed interval. A raised
    ``NativeEpisodeError`` carries every committed row and the simulator's actual
    work counters, even if a native integration failed partway through a tick.
    ``sensor_capture_tick`` is the capture time of the *delivered* packet. The
    pending fields carry a captured packet waiting for its two-tick delivery.
    """
    ticks, reset_seed = index(ticks), index(reset_seed)
    if ticks <= 0 or reset_seed < 0 or not np.isfinite(wheel_offset_rad_s):
        raise ValueError("a positive horizon, nonnegative reset seed and finite wheel offset are required")
    if explore_seed is not None:
        explore_seed = index(explore_seed)
        if explore_seed < 0:
            raise ValueError("exploration seed must be nonnegative")
    rng = np.random.default_rng(explore_seed) if explore_seed is not None else None
    endpoints, intervals = [], []
    env = None

    def snapshot():
        counts = env.counters() if env is not None else dict(policy_transitions=0, pd_updates=0, native_steps=0)
        return {**_columns(endpoints, ENDPOINTS), **_columns(intervals, INTERVALS),
                "reset_seed": np.int64(reset_seed), "requested_ticks": np.int64(ticks),
                "explore_seed": np.int64(explore_seed if explore_seed is not None else -1),
                "wheel_offset_rad_s": np.float64(wheel_offset_rad_s),
                "completed_transitions": np.int64(len(intervals)),
                "policy_steps": np.int64(counts["policy_transitions"]),
                "servo_steps": np.int64(counts["pd_updates"]), "native_steps": np.int64(counts["native_steps"])}

    try:
        env = _open_env(reset_seed, ticks)
        try:
            obs = _reset_uphill(env, reset_seed)
            endpoints.append(_endpoint(env, obs, 0))
            if progress is not None:
                progress(snapshot())
            noise = np.zeros(16, dtype=np.float32)
            for tick in range(ticks):
                raw = predict(actor, obs)
                if rng is not None and tick % 5 == 0:
                    noise = _draw_exploration(rng)
                issued = raw + noise
                issued[WHEEL_INDICES] += np.float32(wheel_offset_rad_s / 5.0)
                issued = clip_action(issued)
                next_obs, reward, terminated, truncated, info = env.step(issued)
                step = _row(INTERVALS, raw16=raw, exploration16=noise, issued16=env._last_policy_action,
                            applied16=info["applied_action"], support4=info["loaded_wheels_past_toe"],
                            fallen=info["fallen"], illegal=info["illegal_body_contact"], terminated=terminated,
                            truncated=truncated, reward=reward, command3=info["command_during_interval"],
                            wheel_saturation_count=info["wheel_saturation_count"])
                endpoint = _endpoint(env, next_obs, tick + 1)
                if (not np.array_equal(step["issued16"], issued) or not np.array_equal(next_obs[41:57], issued)
                        or not np.array_equal(step["applied16"], endpoints[-1]["obs643"][41:57])):
                    raise ValueError("issued, observed and applied actions violate the one-tick FIFO")
                intervals.append(step)
                endpoints.append(endpoint)
                obs = next_obs
                if progress is not None:
                    progress(snapshot())
                if terminated or truncated:
                    break
            counts = env.counters()
            n = len(intervals)
            if (counts["policy_transitions"], counts["pd_updates"], counts["native_steps"]) != (n, 5*n, 10*n):
                raise ValueError("completed native, servo and policy clocks disagree")
        finally:
            env.close()
    except Exception as exc:
        raise NativeEpisodeError(f"{type(exc).__name__}: {exc}", snapshot()) from exc
    return snapshot()
