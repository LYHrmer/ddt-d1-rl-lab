"""Score only native trajectory measurements, independent of the actor and simulator."""
from __future__ import annotations

import numpy as np


def score(trace):
    t = np.asarray(trace["time_s"])
    n = len(t) - 1
    if not 1 <= n <= 400 or not np.allclose(t, np.arange(n + 1) / 50., atol=1e-9, rtol=0.):
        raise ValueError("invalid policy-boundary clock")
    shape = {"qpos": (n+1, 23), "qvel": (n+1, 22), "wheels_actual": (n+1, 4),
             "wheel_target": (n, 4), "wheel_saturated": (n, 4), "fallen": (n,), "illegal": (n,)}
    for name, want in shape.items():
        a = np.asarray(trace[name])
        if a.shape != want or not np.isfinite(a).all():
            raise ValueError(f"invalid {name} data")
    q, dq = trace["qpos"], trace["qvel"]
    w, x, y, z = q[0, 3:7]
    yaw0 = np.arctan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z))
    direction = np.array([np.cos(yaw0), np.sin(yaw0)])
    xy = q[:, :2] - q[0, :2]
    along = xy @ direction
    lateral = xy @ np.array([-direction[1], direction[0]])
    speeds = np.linalg.norm(dq[:, :2], axis=1)
    displacement = float(along[275] - along[175]) if n >= 275 else None
    stop_speed = float(speeds[375:].max()) if n == 400 else None
    failed = []
    if n != 400:
        failed.append("incomplete_horizon")
    if displacement is None or not 3. <= displacement <= 3.15:
        failed.append("cruise_distance")
    if stop_speed is None or stop_speed > .05:
        failed.append("stop_speed")
    if np.any(trace["fallen"]):
        failed.append("fall")
    if np.any(trace["illegal"]):
        failed.append("illegal_contact")
    half_seconds = ([float((along[k+25]-along[k])/.5) for k in range(175, 275, 25)]
                    if displacement is not None else [])
    # Distance traveled on the floor, allowing for sideways motion or overshoot.
    braking_path = None
    if n == 400 and stop_speed <= .05:
        last_fast = np.flatnonzero(speeds[275:] > .05)
        stop_tick = 275 if not len(last_fast) else 276 + int(last_fast[-1])
        braking_path = float(np.linalg.norm(np.diff(xy[275:stop_tick+1], axis=0), axis=1).sum())
    return {"ticks": n, "passed": not failed, "failed": failed, "cruise_distance_m": displacement,
            "cruise_speed_m_s": None if displacement is None else displacement / 2.,
            "cruise_half_second_means_m_s": half_seconds, "peak_horizontal_speed_m_s": float(speeds.max()),
            "stop_tail_max_m_s": stop_speed, "braking_path_m": braking_path,
            "max_abs_lateral_m": float(np.abs(lateral).max()),
            "target_wheels_rad_s": trace["wheel_target"][175:275].mean(axis=0).tolist() if n >= 275 else None,
            "actual_wheels_rad_s": trace["wheels_actual"][176:276].mean(axis=0).tolist() if n >= 275 else None,
            "wheel_saturated_ticks": np.count_nonzero(trace["wheel_saturated"], axis=0).tolist()}
