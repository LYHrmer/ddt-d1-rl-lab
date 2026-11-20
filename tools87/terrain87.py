"""Native D1 plant and delayed first-return range measurements for experiment 87."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

from training.d1_rl.delay_env18 import D1Delay18Env
from training.d1_rl.interface import ACTOR_DIM, DEFAULT_Q, LEG_INDICES, WHEEL_INDICES
from training.d1_rl.task03d import DEFAULT_CONFIG_03D, ROOT


TERRAINS = ("flat", "up", "down")
SLOPE_DEG = {"flat": 0.0, "up": 4.0, "down": -4.0}
RAMP_START_X_M = 0.60
RAMP_LENGTH_M = 2.0
HALF_WIDTH_X_M = 4.0
HALF_WIDTH_Y_M = 1.8
GRID_ROWS, GRID_COLS = 37, 81
FIELD_BASE_Z_M, FIELD_SCALE_Z_M = -0.16, 0.32
POLICY_HZ, CAPTURE_HZ, SENSOR_DELAY_TICKS = 50, 6, 2
EPISODE_TICKS = 300
COMMAND_M_S = 0.20
MIN_RANGE_M, MAX_RANGE_M = 0.14, 1.45
TERRAIN_DIM = 16
MOUNTS_M = np.array([[0.355, 0.190, 0.055], [0.355, -0.190, 0.055]])
DEPRESSION = np.deg2rad([82.0, 56.0, 43.0, 33.0])
DIRECTIONS = np.column_stack((np.cos(DEPRESSION), np.zeros(4), -np.sin(DEPRESSION)))


def ground_height(terrain, x):
    """World z of the road. Only scene generation, diagnostics and rendering use it."""
    if terrain not in SLOPE_DEG:
        raise ValueError(f"unknown physical terrain {terrain!r}")
    ramp_x = np.clip(np.asarray(x) - RAMP_START_X_M, 0.0, RAMP_LENGTH_M)
    return ramp_x * math.tan(math.radians(SLOPE_DEG[terrain]))


def _write_scene(directory):
    # The legacy loader verifies an adjacent robot.xml. Keep that check and
    # the robot's ../meshes/ resolution in a private temporary scene directory.
    config = json.loads(DEFAULT_CONFIG_03D.read_text())
    original = ROOT / config["scene"]
    original_xml = original.read_bytes()
    if hashlib.sha256(original_xml).hexdigest() != config["scene_sha256"]:
        raise ValueError("source plane scene changed")
    robot = original.parent / "robot.xml"
    if hashlib.sha256(robot.read_bytes()).hexdigest() != config["robot_xml_sha256"]:
        raise ValueError("source D1 robot changed")
    directory = Path(directory)
    scene_dir = directory / "mujoco"
    scene_dir.mkdir()
    (scene_dir / "robot.xml").symlink_to(robot)
    (directory / "meshes").symlink_to(original.parent.parent / "meshes", target_is_directory=True)
    root = ET.fromstring(original_xml)
    asset = root.find("asset")
    floor = root.find("./worldbody/geom[@name='floor']")
    if asset is None or floor is None or floor.get("type") != "plane":
        raise ValueError("source scene must contain the original world floor")
    ET.SubElement(asset, "hfield", name="road87", nrow=str(GRID_ROWS), ncol=str(GRID_COLS),
                  size=f"{HALF_WIDTH_X_M} {HALF_WIDTH_Y_M} {FIELD_SCALE_Z_M} 0.02")
    del floor.attrib["size"]
    floor.set("type", "hfield")
    floor.set("hfield", "road87")
    floor.set("pos", f"{RAMP_START_X_M} 0 {FIELD_BASE_Z_M}")
    path = scene_dir / "scene87.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _set_road(model, terrain):
    xs = RAMP_START_X_M + np.linspace(-HALF_WIDTH_X_M, HALF_WIDTH_X_M, GRID_COLS)
    normalized = (ground_height(terrain, xs) - FIELD_BASE_Z_M) / FIELD_SCALE_Z_M
    if (model.nhfield != 1 or model.hfield_nrow[0] != GRID_ROWS
            or model.hfield_ncol[0] != GRID_COLS or not np.all((normalized >= 0) & (normalized <= 1))):
        raise ValueError("road height-field geometry differs from the 87 configuration")
    model.hfield_data.reshape(GRID_ROWS, GRID_COLS)[:] = normalized


def load_model(terrain):
    """Build the visualization model with the identical D1 robot and physical floor."""
    with TemporaryDirectory(prefix="d1-terrain87-view-") as directory:
        model = mujoco.MjModel.from_xml_path(str(_write_scene(directory)))
    _set_road(model, terrain)
    return model


def capture_due(tick):
    """First 50 Hz boundary on or after each event of a 6 Hz acquisition clock."""
    return tick == 0 or tick * CAPTURE_HZ // POLICY_HZ != (tick - 1) * CAPTURE_HZ // POLICY_HZ


def range_features(ranges_m, occluded_by_robot):
    """Pack eight first returns followed by their validity flags in left/right beam order."""
    ranges = np.asarray(ranges_m, dtype=np.float64).reshape(8)
    self_hits = np.asarray(occluded_by_robot, dtype=bool).reshape(8)
    valid = np.isfinite(ranges) & (ranges >= MIN_RANGE_M) & (ranges <= MAX_RANGE_M) & ~self_hits
    normalized = np.where(valid, ranges / MAX_RANGE_M, 0.0)
    return np.concatenate((normalized, valid)).astype(np.float32)


class D1Terrain87(D1Delay18Env):
    """One native D1, one 16D PD command per tick, 627 proprio + 16 range inputs."""

    def __init__(self, terrain, seed, *, perception=True, max_transitions=None):
        if terrain not in (*TERRAINS, "cycle"):
            raise ValueError("terrain must be flat, up, down, or cycle")
        if type(perception) is not bool:
            raise ValueError("perception must be a boolean mask setting")
        self.terrain_mode = terrain
        self.perception = perception
        self._workspace = TemporaryDirectory(prefix="d1-terrain87-")
        try:
            scene = _write_scene(self._workspace.name)
            super().__init__(
                config=DEFAULT_CONFIG_03D, arm="B", seed=seed,
                eval_delay_ticks=1, scene=scene, max_transitions=max_transitions,
                eval_command_sequence=[{"start_s": 0.0, "vx": 0.0, "wz": 0.0},
                                       {"start_s": 1.0, "vx": COMMAND_M_S, "wz": 0.0}],
            )
        except BaseException:
            self._workspace.cleanup()
            raise
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(ACTOR_DIM + TERRAIN_DIM,),
                                                dtype=np.float32)
        self.base_body = self.model.body("base_link").id
        self.floor_geom = self.model.geom("floor").id
        front_bodies = {self.model.body(name).id for name in ("FL_foot", "FR_foot")}
        self.front_geoms = sorted(g for g in self.wheel_geoms if self.model.geom_bodyid[g] in front_bodies)
        if len(self.front_geoms) != 2:
            raise ValueError("expected exactly two colliding front wheels")
        self.last_raw_features = np.zeros(TERRAIN_DIM, dtype=np.float32)
        self._masked = np.zeros_like(self.last_raw_features)
        self.current_terrain = TERRAINS[0]
        self.sensor_capture_tick = self.sensor_delivery_tick = -1
        self._pending = None

    def _measure(self):
        rotation = self.data.xmat[self.base_body].reshape(3, 3)
        origins = MOUNTS_M @ rotation.T + self.data.xpos[self.base_body]
        directions = DIRECTIONS @ rotation.T
        distances = np.empty((2, 4))
        self_hits = np.zeros((2, 4), dtype=bool)
        first = np.empty(1, dtype=np.int32)
        for lane, origin in enumerate(origins):
            for beam, direction in enumerate(directions):
                distance = mujoco.mj_ray(
                    self.model, self.data, origin, direction, None, True, -1, first,
                )
                # Predict self returns using only the known articulated robot.
                # The world floor is static, and the robot is the only moving
                # body. Do not consult the simulator's semantic ID for a hit.
                self_distance = mujoco.mj_ray(
                    self.model, self.data, origin, direction, None, False, -1, first,
                )
                distances[lane, beam] = distance
                self_hits[lane, beam] = self_distance >= 0 and abs(self_distance - distance) <= 1e-8
        return range_features(distances, self_hits)

    def _update_measurement(self, tick):
        if capture_due(tick):
            self._pending = (tick, self._measure())
        if self._pending is not None and self._pending[0] + SENSOR_DELAY_TICKS == tick:
            self.sensor_capture_tick, self.last_raw_features = self._pending
            self.sensor_delivery_tick = tick
            self._pending = None

    def _with_ranges(self, original):
        return np.concatenate((original, self.last_raw_features if self.perception else self._masked))

    def reset(self, *, seed=None, options=None):
        base = self.base_seed if seed is None else int(seed)
        episode = self._episode_index if seed is None else 0
        self.current_terrain = (TERRAINS[(base + episode) % len(TERRAINS)]
                                if self.terrain_mode == "cycle" else self.terrain_mode)
        _set_road(self.model, self.current_terrain)
        self._pending = None
        self.sensor_capture_tick = self.sensor_delivery_tick = -1
        self.last_raw_features = np.zeros(TERRAIN_DIM, dtype=np.float32)
        obs, info = super().reset(seed=seed, options=options)
        self._update_measurement(0)
        return self._with_ranges(obs), info

    def current_observation(self):
        return self._with_ranges(super().current_observation())

    def front_contact_x_m(self):
        """Farthest front-wheel support point with >1 N at this observation boundary."""
        farthest = None
        force = np.empty(6)
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if not ((g1 == self.floor_geom and g2 in self.front_geoms)
                    or (g2 == self.floor_geom and g1 in self.front_geoms)):
                continue
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if force[0] > 1.0:
                x = float(contact.pos[0])
                farthest = x if farthest is None else max(x, farthest)
        return farthest

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._update_measurement(self._episode_steps)
        observation = self._with_ranges(obs)
        truncated = bool(truncated or (self._episode_steps >= EPISODE_TICKS and not terminated))
        self._episode_done = bool(terminated or truncated)
        info["TimeLimit.truncated"] = truncated
        if self._episode_done:
            info["terminal_observation"] = observation.copy()
        info["wheel_saturation_count"] = int(np.count_nonzero(self._last_saturation[WHEEL_INDICES]))
        info["front_axle_x_m"] = float(self.data.geom_xpos[self.front_geoms, 0].mean())
        info["front_contact_x_m"] = self.front_contact_x_m()
        target = self.action_scales * self._applied_action
        info["leg_target_rad"] = (DEFAULT_Q[LEG_INDICES] + target[LEG_INDICES]).tolist()
        info["wheel_target_rad_s"] = target[WHEEL_INDICES].tolist()
        info["last_torque_nm"] = self._last_tau.tolist()
        return observation, reward, bool(terminated), truncated, info

    def get_counts(self):
        return self.counters()

    def close(self):
        super().close()
        self._workspace.cleanup()


def make_env(terrain, seed, *, perception=True, max_transitions=None):
    return D1Terrain87(terrain, seed, perception=perception, max_transitions=max_transitions)
