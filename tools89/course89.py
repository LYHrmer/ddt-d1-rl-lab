"""D1 training starts on the original road, plus online wheel-support diagnostics."""
from __future__ import annotations

import hashlib
import math

import mujoco
import numpy as np

from tools87.run87 import PLAN as PLAN87, ROOT, evidence, read, sha
from tools87.terrain87 import (ACTOR_DIM, D1Terrain87, GRID_COLS, GRID_ROWS, HALF_WIDTH_X_M,
                               HALF_WIDTH_Y_M, RAMP_LENGTH_M, RAMP_START_X_M, SLOPE_DEG,
                               TERRAINS, ground_height, load_model)
from training.d1_rl.env import FOOT_BODIES


def slope_start_for_episode(terrain_mode, seed, episode):
    """Use the second, fourth, ... uphill reset; terrain cycling is identical to 87."""
    if terrain_mode == "up":
        uphill_index = episode
    elif terrain_mode == "cycle":
        first = (TERRAINS.index("up") - seed) % len(TERRAINS)
        uphill_index, remainder = divmod(episode - first, len(TERRAINS))
        if uphill_index < 0 or remainder:
            return False
    else:
        return False
    return uphill_index % 2 == 1


def place_on_uphill(pose, world_velocity, start_x):
    """Rigidly map the original flat reset to the inclined plane at the chosen base x."""
    angle = math.radians(SLOPE_DEG["up"])
    rotation = np.array([math.cos(angle / 2), 0., -math.sin(angle / 2), 0.])
    # Give the base the same perpendicular height above the new plane as above the old one.
    pose[2] = float(ground_height("up", start_x)) + pose[2] / math.cos(angle)
    pose[0] = start_x
    mujoco.mju_mulQuat(pose[3:7], rotation, pose[3:7].copy())
    mujoco.mju_rotVecQuat(world_velocity, world_velocity.copy(), rotation)
    # MuJoCo expresses free-joint angular velocity in the body frame.


def wheel_geom_slots(env):
    """Map each colliding wheel cylinder to the action-interface order FL, FR, RL, RR."""
    bodies = {env.model.body(name).id: i for i, name in enumerate(FOOT_BODIES)}
    slots = {g: bodies[int(env.model.geom_bodyid[g])] for g in env.wheel_geoms}
    if len(slots) != 4 or set(slots.values()) != set(range(4)):
        raise ValueError("one colliding wheel geometry per foot is required")
    return slots


def loaded_wheels_past_toe(env, slots):
    """True for each uphill wheel with a floor contact at/past the toe bearing >1 N."""
    loaded = np.zeros(4, dtype=bool)
    if env.current_terrain != "up":
        return loaded
    force = np.empty(6)
    for index, contact in enumerate(env.data.contact[:env.data.ncon]):
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if env.floor_geom not in (g1, g2) or contact.pos[0] < RAMP_START_X_M:
            continue
        wheel = slots.get(g2 if g1 == env.floor_geom else g1)
        if wheel is None or loaded[wheel]:
            continue
        mujoco.mj_contactForce(env.model, env.data, index, force)
        loaded[wheel] = force[0] > 1.0
    return loaded


class D1Course89(D1Terrain87):
    def __init__(self, terrain, seed, *, curriculum=False, slope_start_x_m=1.10,
                 perception=True, max_transitions=None):
        if type(curriculum) is not bool:
            raise ValueError("curriculum must be a boolean")
        if not RAMP_START_X_M < slope_start_x_m < RAMP_START_X_M + RAMP_LENGTH_M:
            raise ValueError("the slope start must lie within the original uphill segment")
        self.curriculum = curriculum
        self.slope_start_x_m = float(slope_start_x_m)
        self.slope_start = False
        self._placement_pending = False
        super().__init__(terrain, seed, perception=perception, max_transitions=max_transitions)
        self._wheel_slots = wheel_geom_slots(self)

    def _observe(self):
        # The frozen reset has now drawn the joints/yaw and set the original 2 mm clearance.
        # Transform before the first sensor frame, its ten history copies and the tick-zero raycast.
        if self._placement_pending:
            place_on_uphill(self.data.qpos[self.free_qpos:self.free_qpos + 7],
                            self.data.qvel[self.free_dof:self.free_dof + 3], self.slope_start_x_m)
            self._placement_pending = False
        return super()._observe()

    def reset(self, *, seed=None, options=None):
        base_seed = self.base_seed if seed is None else int(seed)
        episode = self._episode_index if seed is None else 0
        self.slope_start = self.curriculum and slope_start_for_episode(self.terrain_mode, base_seed, episode)
        self._placement_pending = self.slope_start
        obs, info = super().reset(seed=seed, options=options)
        if self._placement_pending or (self.slope_start and self.current_terrain != "up"):
            raise RuntimeError("unexpected terrain or first observation in the uphill reset")
        info.update(slope_start=self.slope_start, start_x_m=float(self.data.qpos[self.free_qpos]))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(loaded_wheels_past_toe=loaded_wheels_past_toe(self, self._wheel_slots),
                    base_origin_past_toe=bool(self.data.qpos[self.free_qpos] > RAMP_START_X_M),
                    slope_start=self.slope_start, start_x_m=self.slope_start_x_m if self.slope_start else 0.)
        return obs, reward, terminated, truncated, info


def make_env(terrain, seed, **kwargs):
    return D1Course89(terrain, seed, **kwargs)


def _require(condition, label):
    if not condition:
        raise ValueError(label)


def road_cross_sections(env, x):
    """Use the identical across-track profile at two points away from triangle edges."""
    dx = 2 * HALF_WIDTH_X_M / (GRID_COLS - 1)
    dy = 2 * HALF_WIDTH_Y_M / (GRID_ROWS - 1)
    fraction_x = ((x - RAMP_START_X_M + HALF_WIDTH_X_M) / dx) % 1.
    # Each cell has one of two diagonal directions. Stay away from both.
    fraction_y = max((.25, .5, .75), key=lambda v: min(v, 1-v, abs(v-fraction_x), abs(v-(1-fraction_x))))
    inside = 1.2 + fraction_y * dy
    expected = float(ground_height(env.current_terrain, x))
    samples = []
    for y in (1.2, inside, -inside):
        geom = np.array([-1], np.int32)
        depth = mujoco.mj_ray(env.model, env.data, np.array([x, y, 1.]), np.array([0., 0., -1.]),
                              None, True, -1, geom)
        height = 1. - depth if depth >= 0 else None
        error = abs(height - expected) if height is not None else None
        matches = int(geom[0]) == env.floor_geom and error is not None and error < 2e-6
        samples.append({"y_m": float(y), "height_m": height, "abs_error_m": error, "floor_within_tolerance": matches})
    _require(all(s["floor_within_tolerance"] for s in samples[1:]),
             f"reset native road mismatch {env.current_terrain=} {x=} {expected=} {samples=}")
    return {"x_m": float(x), "expected_height_m": expected, "original_row_edge": samples[0],
            "two_cell_interiors": samples[1:]}


def _measure_reset(env, obs, info, shifted):
    angle = math.radians(SLOPE_DEG["up"]) if shifted else 0.
    normal = np.array([-math.sin(angle), 0., math.cos(angle)])
    origin = np.array([RAMP_START_X_M if shifted else 0., 0., 0.])
    gaps, bounds_x, roads = [], [], []
    for gid in sorted(env.wheel_geoms, key=wheel_geom_slots(env).get):
        center = env.data.geom_xpos[gid]
        local = env.data.geom_xmat[gid].reshape(3, 3).T @ normal
        radius, half_width = env.model.geom_size[gid, :2]
        extent = radius * np.linalg.norm(local[:2]) + half_width * abs(local[2])
        gaps.append(float(normal @ (center - origin) - extent))
        bounds_x.append(center[0] + np.array([-1., 1.]) * env.model.geom_rbound[gid])
        roads.append(road_cross_sections(env, center[0]))
    frames = obs[:ACTOR_DIM].reshape(11, 57)
    _require(np.array_equal(frames, np.broadcast_to(env._current_obs, frames.shape))
             and not np.any(obs[ACTOR_DIM:]) and not np.any(env._last_policy_action)
             and not np.any(env.data.qvel) and not np.any(env._filtered_gyro)
             and info["delay_ticks"] == len(env._delay_fifo) == 1 and not np.any(env._delay_fifo)
             and env._pending[0] == 0 and np.array_equal(env._pending[1], env._measure())
             and env.sensor_capture_tick == env.sensor_delivery_tick == -1, "reset history, FIFO or sensor clock")
    return {"state": env.get_initial_state(), "obs": obs, "pending": env._pending[1].copy(),
            "rot": env.data.xmat[env.base_body].reshape(3, 3).copy(),
            "gaps": np.array(gaps), "bounds_x": np.array(bounds_x), "road_sections": roads,
            "max_height_error_m": max(s["abs_error_m"] for road in roads for s in road["two_cell_interiors"])}


def _reset_samples(terrain, seeds, role, slope_start_x_m):
    if role == "parent87":
        env = D1Terrain87(terrain, seeds[0], max_transitions=1)
    else:
        env = make_env(terrain, seeds[0], curriculum=role == "mixed", slope_start_x_m=slope_start_x_m,
                       max_transitions=1)
    rows = {}
    try:
        for seed in seeds:
            for episode, reset_seed in enumerate((seed, None)):
                obs, info = env.reset(seed=reset_seed)
                shifted = role == "mixed" and terrain == "up" and episode == 1
                _require(info.get("slope_start", False) == shifted
                         and env.data.qpos[env.free_qpos] == (slope_start_x_m if shifted else 0.),
                         "wrong reset position or schedule")
                rows[seed, episode] = _measure_reset(env, obs, info, shifted)
        _require(all(env.counters()[k] == 0 for k in ("policy_transitions", "pd_updates", "native_steps")),
                 "reset audit advanced physics")
    finally:
        env.close()
    return rows


def _archive_rear_bounds():
    plan = read(PLAN87)
    directory = ROOT / "evidence/full_body87/training_01"
    receipts = {r["model"]: r for r in evidence(directory, "training")["models"]}
    model = load_model("up")
    data = mujoco.MjData(model)
    wheel_ids = []
    for name in FOOT_BODIES:
        body = model.body(name).id
        geoms = [g for g in range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
                 if model.geom_contype[g] and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
        _require(len(geoms) == 1, "archive wheel geometry")
        wheel_ids.extend(geoms)
    rbound = model.geom_rbound[wheel_ids]
    rows = []
    for seed in plan["training"]["seeds"]:
        name = f"range_{seed}"
        folder = directory / name
        rep = read(folder / "report.json")
        _require(sha(folder / "report.json") == receipts[name]["report_sha256"] and rep["status"] == "COMPLETE"
                 and sha(folder / "policy.zip") == rep["policy_sha256"], "incomplete archive or changed checkpoint")
        with np.load(folder / "train_trace.npz", allow_pickle=False) as trace:
            qpos, terrain = (trace[key] for key in ("qpos", "terrain"))
        for key, array in (("qpos", qpos), ("terrain", terrain)):
            _require(hashlib.sha256(array.tobytes()).hexdigest() == rep["trace"]["array_sha256"][key],
                     name + " " + key)
        poses = qpos[terrain == TERRAINS.index("up")]
        upper = np.empty((len(poses), len(wheel_ids)))
        for i, q in enumerate(poses):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            upper[i] = data.geom_xpos[wheel_ids, 0] + rbound
        possible = upper >= RAMP_START_X_M
        rows.append({"seed": seed, "uphill_ticks": len(poses), "base_origin_past_toe_ticks":
                     int((poses[:, 0] > RAMP_START_X_M).sum()), "max_wheel_x_upper_bound_m": upper.max(axis=0).tolist(),
                     "max_rear_wheel_x_upper_bound_m": float(upper[:, 2:].max()),
                     "both_axles_could_reach_toe_ticks": int((possible[:, :2].any(1) & possible[:, 2:].any(1)).sum()),
                     "all_four_could_reach_toe_ticks": int(possible.all(1).sum()),
                     "training_report_sha256": receipts[name]["report_sha256"]})
    return {"method": "World x of each native wheel cylinder center plus its bounding-sphere radius. "
                      "Below-toe upper bounds rule out a loaded contact beyond the toe at that stored endpoint.",
            "wheel_order": list(FOOT_BODIES), "sphere_radii_m": rbound.tolist(), "seeds": rows,
            "uphill_ticks": sum(row["uphill_ticks"] for row in rows),
            "all_four_could_reach_toe_ticks": sum(row["all_four_could_reach_toe_ticks"] for row in rows)}


def zero_preflight(seeds=(89101, 89102, 89103), slope_start_x_m=1.10):
    """Compare original and mixed reset states and bound 87 exposure with no integration."""
    seeds = tuple(seeds)
    archive = _archive_rear_bounds()
    matched, on_slope = 0, []
    # Record original edge observations as well as the two interior rays used by the gate.
    sections = []
    angle = math.radians(SLOPE_DEG["up"])
    tilt = np.array([[math.cos(angle), 0., -math.sin(angle)], [0., 1., 0.],
                     [math.sin(angle), 0., math.cos(angle)]])
    for terrain in TERRAINS:
        reference = _reset_samples(terrain, seeds, "parent87", slope_start_x_m)
        for (seed, episode), row in reference.items():
            sections.append({"seed": seed, "terrain": terrain, "uphill_curriculum": False, "episode": episode,
                             "wheels": row["road_sections"]})
        for role in ("original", "mixed"):
            samples = _reset_samples(terrain, seeds, role, slope_start_x_m)
            for (seed, episode), value in samples.items():
                parent = reference[seed, episode]
                if role == "mixed" and terrain == "up" and episode == 1:
                    sections.append({"seed": seed, "terrain": terrain, "uphill_curriculum": True, "episode": episode,
                                     "wheels": value["road_sections"]})
                    gap_error = float(np.max(np.abs(value["gaps"] - parent["gaps"])))
                    bounds = value["bounds_x"]
                    _require(gap_error < 1e-12 and abs(min(value["gaps"]) - .002) < 1e-12
                             and np.allclose(value["rot"], tilt @ parent["rot"], atol=1e-12, rtol=0)
                             and np.array_equal(value["state"]["qpos"][7:], parent["state"]["qpos"][7:])
                             and bounds.min() > RAMP_START_X_M and bounds.max() < RAMP_START_X_M + RAMP_LENGTH_M,
                             "rigid slope-normal reset or wheel footprint")
                    on_slope.append({"seed": seed, "base_x_m": value["state"]["qpos"][0],
                                     "base_z_m": value["state"]["qpos"][2], "per_wheel_normal_clearance_m": value["gaps"].tolist(),
                                     "max_clearance_difference_m": gap_error, "lowest_x_wheel_bound_m": float(bounds.min()),
                                     "max_native_road_height_error_m": value["max_height_error_m"]})
                else:
                    _require(value["state"] == parent["state"] and np.array_equal(value["obs"], parent["obs"])
                             and np.array_equal(value["pending"], parent["pending"]), "original reset differs from 87")
                    matched += 1
    reference_failures = []
    for sample in sections:
        for wheel, row in zip(FOOT_BODIES, sample["wheels"], strict=True):
            if not sample["uphill_curriculum"] and not row["original_row_edge"]["floor_within_tolerance"]:
                reference_failures.append({"seed": sample["seed"], "terrain": sample["terrain"],
                                           "episode": sample["episode"], "wheel": wheel, **row})
    return {"archive87": archive, "original_resets_matching87": matched, "mixed_uphill_starts": on_slope,
            "native_road_probes": {"tolerance_m": 2e-6, "unique_reset_states": sections,
                                  "interior_sample_count": 8 * len(sections),
                                  "original_reference_row_edge_failures": len(reference_failures),
                                  "original_reference_edge_first_failure": next(iter(reference_failures), None),
                                  "original_reference_edge_max_error_m": max(
                                      (w["original_row_edge"]["abs_error_m"] for s in sections if not s["uphill_curriculum"]
                                       for w in s["wheels"] if w["original_row_edge"]["abs_error_m"] is not None), default=None),
                                  "max_interior_error_m": max(v["abs_error_m"] for s in sections for w in s["wheels"]
                                                              for v in w["two_cell_interiors"])},
            "policy_transitions": 0, "pd_updates": 0, "native_steps": 0}
