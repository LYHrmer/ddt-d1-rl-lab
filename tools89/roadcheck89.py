"""Measure the same road cross-section at a grid edge and between grid edges, without integration."""
import numpy as np
import mujoco

from tools87.terrain87 import ground_height, make_env
from tools89.common89 import BASE, registered_plan, write


def run():
    plan = registered_plan()
    seed = plan["preflight"]["reset_seeds"][0]
    rows = []
    for terrain in ("flat", "up", "down"):
        env = make_env(terrain, seed, max_transitions=1)
        try:
            env.reset(seed=seed)
            for gid in sorted(env.wheel_geoms):
                x = float(env.data.geom_xpos[gid, 0])
                for y in (1.2, 1.217, 1.237, 1.263, .817):
                    pos = np.array([x, y, 1.0])
                    direction = np.array([0., 0., -1.])
                    geom = np.array([-1], np.int32)
                    distance = mujoco.mj_ray(env.model, env.data, pos, direction, None, True, -1, geom)
                    exact = float(ground_height(terrain, x))
                    rows.append({"terrain": terrain, "wheel_x_m": x, "ray_y_m": y,
                                 "hit_floor": int(geom[0]) == env.floor_geom, "expected_z_m": exact,
                                 "hit_z_m": 1.0 - distance, "error_m": abs(1.0 - distance - exact)})
            if env.counters()["native_steps"] != 0:
                raise ValueError("road measurement integrated physics")
        finally:
            env.close()
    write(BASE / "roadcheck_01.json", {"rows": rows, "native_steps": 0})
    for y in sorted({s["ray_y_m"] for s in rows}):
        print(y, max(s["error_m"] for s in rows if s["ray_y_m"] == y), flush=True)
