"""Gymnasium environment for PPO on the original pure D1 MuJoCo robot.

Model loading and reset use mj_forward only. Physics integration happens solely
inside step(), where each transition is exactly 10 native MuJoCo steps.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .interface import (
    ACTION_DIM, ACTION_SCALE, ACTOR_DIM, DEFAULT_Q, HISTORY_LENGTH, JOINT_NAMES,
    KD, KP, OBS_DIM, TORQUE_LIMIT,
    action_to_torque, build_observation, clip_action, inverse_rotate,
    pack_actor_input,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "training/configs/pilot03.json"
FOOT_BODIES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


class D1Env(gym.Env):
    """50 Hz pure-D1 PPO environment with explicit P_V and delay timing.

    ``arm`` is B (0-tick), R (episode-uniform 0/1/2-tick), or M (1-tick).
    ``eval_delay_ticks`` overrides it with a fixed 0/1/2/3-tick delay.
    ``set_command(vx, wz)`` supports externally scheduled evaluation commands.
    Each environment instance has independent reset, command, and delay RNGs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: str | Path | dict[str, Any] | None = None,
        *,
        arm: str = "B",
        seed: int = 0,
        rank: int = 0,
        max_transitions: int | None = None,
        eval_delay_ticks: int | None = None,
        eval_command_sequence: list[dict[str, float]] | None = None,
        scene: str | Path | None = None,
    ):
        super().__init__()
        if mujoco.__version__ != "3.3.0":
            raise RuntimeError(f"D1Env requires MuJoCo 3.3.0, got {mujoco.__version__}")
        if arm not in ("B", "R", "M"):
            raise ValueError("arm must be B, R, or M")
        self.config = (json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8")) if config is None
                       else json.loads(_path(config).read_text(encoding="utf-8"))
                       if isinstance(config, (str, Path)) else dict(config))
        self._validate_config()
        self.arm, self.rank, self.base_seed = arm, int(rank), int(seed)
        if self.rank < 0 or self.base_seed < 0:
            raise ValueError("seed and rank must be nonnegative")
        self._episode_index = 0
        limit = self.config["max_transitions"] if max_transitions is None else max_transitions
        self.max_transitions = None if limit is None else int(limit)
        if self.max_transitions is not None and self.max_transitions < 1:
            raise ValueError("max_transitions must be positive")
        self.eval_delay_ticks = eval_delay_ticks
        if eval_delay_ticks is not None and eval_delay_ticks not in self.config["delay"]["eval_allowed_fixed_ticks"]:
            raise ValueError("eval_delay_ticks must be one of 0, 1, 2, 3")
        self.eval_command_sequence = self._validate_eval_sequence(eval_command_sequence)

        self.scene_path = _path(scene if scene is not None else self.config["scene"]).resolve()
        if scene is None and hashlib.sha256(self.scene_path.read_bytes()).hexdigest() != self.config["scene_sha256"]:
            raise ValueError("Default training scene SHA256 mismatch")
        robot_path = self.scene_path.parent / "robot.xml"
        if hashlib.sha256(robot_path.read_bytes()).hexdigest() != self.config["robot_xml_sha256"]:
            raise ValueError("Included D1 robot.xml SHA256 mismatch")
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        if not math.isclose(self.model.opt.timestep, self.config["native_dt_s"], abs_tol=1e-12):
            raise ValueError("Scene timestep differs from pilot03.json")
        self.data = mujoco.MjData(self.model)
        self._counts = {"native_steps": 0, "pd_updates": 0, "policy_transitions": 0,
                        "resets": 0, "forward_calls": 0}
        self._resolve_model_indices()
        self.nominal_base_height_m = self._measure_nominal_base_height()

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(ACTOR_DIM,), dtype=np.float32)
        self._reset_rng = None
        self._command_rng = None
        self._delay_rng = None
        self._current_obs = None
        self._history = np.zeros((HISTORY_LENGTH, OBS_DIM), dtype=np.float32)
        self._filtered_gyro = np.zeros(3, dtype=np.float64)
        self._last_policy_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._applied_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._last_tau = np.zeros(ACTION_DIM, dtype=np.float64)
        self._last_saturation = np.zeros(ACTION_DIM, dtype=bool)
        self._command = np.zeros(3, dtype=np.float64)
        self._manual_command = False
        self._command_phase = -1
        self._episode_steps = 0
        self._episode_done = True
        self._delay_ticks = 0
        self._delay_fifo: deque[np.ndarray] = deque()

    def _validate_config(self):
        config = self.config
        schema_path = _path(config["interface_schema"])
        if hashlib.sha256(schema_path.read_bytes()).hexdigest() != config["interface_schema_sha256"]:
            raise ValueError("Interface schema SHA256 mismatch")
        selected_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        interface_id = config.get("interface_id")
        if interface_id is None:
            if config["version"] != "d1-pure-ppo-pilot-03" or "action_scales" in config:
                raise ValueError("Legacy pilot03 must use its original audited action scales")
            schema = selected_schema
            self.action_scales = ACTION_SCALE.copy()
        elif interface_id == "d1-selftrained-pilot03b-v2":
            if (config["version"] != "d1-pure-ppo-pilot-03b" or
                    selected_schema["interface_id"] != interface_id or
                    selected_schema["schema_version"] != 2):
                raise ValueError("pilot03b interface ID or version mismatch")
            audited = selected_schema["audited_observation_schema"]
            if (audited["path"] != config["audited_interface_schema"] or
                    audited["sha256"] != config["audited_interface_schema_sha256"]):
                raise ValueError("pilot03b does not reference the frozen golden observation schema")
            audited_path = _path(audited["path"])
            if hashlib.sha256(audited_path.read_bytes()).hexdigest() != audited["sha256"]:
                raise ValueError("Audited observation schema SHA256 mismatch")
            schema = json.loads(audited_path.read_text(encoding="utf-8"))
            expected_scale = np.tile(np.array([0.5, 0.5, 0.5, 5.0]), 4)
            self.action_scales = np.asarray(config["action_scales"], dtype=np.float64)
            vectors = (("joint_order", selected_schema["joint_order"], JOINT_NAMES),
                       ("wheel_indices", selected_schema["wheel_indices"], [3, 7, 11, 15]),
                       ("default_joint_angles", selected_schema["default_joint_angles"], DEFAULT_Q),
                       ("joint_kp", selected_schema["joint_kp"], KP),
                       ("joint_kd", selected_schema["joint_kd"], KD),
                       ("torque_limits", selected_schema["torque_limits"], TORQUE_LIMIT),
                       ("schema_action_scales", selected_schema["action_scales"], expected_scale),
                       ("config_action_scales", self.action_scales, expected_scale))
            for name, actual, expected in vectors:
                if list(actual) != list(expected):
                    raise ValueError(f"pilot03b {name} differs from the frozen interface")
            obs03b = selected_schema["observation"]
            expected_slices = {
                "body_gyro_filtered_scaled": [0, 3],
                "projected_gravity_body": [3, 6],
                "command_scaled": [6, 9],
                "joint_position_offset_wheels_zero": [9, 25],
                "joint_velocity_scaled": [25, 41],
                "last_clipped_issued_action": [41, 57],
            }
            expected_scales = {"body_gyro": 0.25, "commands_vx_vy_wz": [2.0, 2.0, 0.25],
                               "joint_position": 1.0, "joint_velocity": 0.05}
            expected_delay = {"FIFO_contents": "whole clipped 16D actions",
                              "reset_value": "zero actions", "training_B_ticks": 0,
                              "training_R_ticks": [0, 1, 2], "training_M_ticks": 1,
                              "evaluation_ticks": [0, 1, 2, 3]}
            if config["delay"] != {"B_fixed_ticks": 0, "R_uniform_ticks": [0, 1, 2],
                                   "M_fixed_ticks": 1,
                                   "eval_allowed_fixed_ticks": [0, 1, 2, 3]}:
                raise ValueError("pilot03b delay arm contract mismatch")
            if (selected_schema["robot_xml"] !=
                    "src/ddt_ros2_control/urdfs/d1_description/mujoco/robot.xml" or
                    selected_schema["robot_xml_sha256"] != config["robot_xml_sha256"] or
                    selected_schema["action_clip"] != [-1.0, 1.0] or
                    selected_schema["control_type"] != "P_V" or
                    selected_schema["force_application"] !=
                    "qfrc_applied; ctrl zero; free-base applied force zero" or
                    (obs03b["current_dim"], obs03b["history_frames"], obs03b["actor_dim"]) !=
                    (OBS_DIM, HISTORY_LENGTH, ACTOR_DIM) or
                    obs03b["slices"] != expected_slices or
                    obs03b["scales"] != expected_scales or
                    obs03b["quaternion_order"] != "wxyz_body_to_world" or
                    obs03b["history_order"] != "oldest_to_newest_previous_only" or
                    obs03b["last_action_semantics"] != "clipped issued policy action before execution delay" or
                    obs03b["gyro_filter_new"] != 0.97 or obs03b["gyro_filter_previous"] != 0.03 or
                    obs03b["sensor_refresh"] != "mj_forward once at each policy observation boundary" or
                    selected_schema["timing"] != {"native_dt_s": 0.002, "pd_native_steps": 2,
                                                  "policy_native_steps": 10,
                                                  "pd_zero_order_hold": True} or
                    selected_schema["delay"] != expected_delay):
                raise ValueError("pilot03b observation/timing contract mismatch")
        else:
            raise ValueError(f"Unknown D1 training interface ID: {interface_id}")
        if (tuple(schema["joint_order"]) != JOINT_NAMES or schema["wheel_indices"] != [3, 7, 11, 15]
                or schema["yaml_rl_flat_lab"]["num_obs"] != OBS_DIM
                or schema["yaml_rl_flat_lab"]["history_len"] != HISTORY_LENGTH
                or schema["yaml_rl_flat_lab"]["num_actions"] != ACTION_DIM):
            raise ValueError("Interface constants disagree with audited schema")
        old = schema["yaml_rl_flat_lab"]
        for field, expected in (("default_joint_angles", DEFAULT_Q), ("joint_kp", KP),
                                ("joint_kd", KD), ("action_scales", ACTION_SCALE)):
            if not np.allclose(old[field], expected, rtol=0, atol=1e-12):
                raise ValueError(f"{field} disagrees with audited P_V interface")
        if not np.allclose(schema["effort_limits_from_expanded_urdf"], TORQUE_LIMIT,
                           rtol=0, atol=1e-12):
            raise ValueError("Torque limits disagree with audited expanded URDF")
        obs = config["actor_observation"]
        if (obs["current_dim"], obs["history_frames"], obs["total_dim"]) != (OBS_DIM, HISTORY_LENGTH, ACTOR_DIM):
            raise ValueError("Actor observation dimensions must be 57+10x57=627")
        if config["action_clip"] != [-1.0, 1.0]:
            raise ValueError("Action execution clip must be [-1,1]")
        if config["pd_native_steps"] != 2 or config["policy_native_steps"] != 10:
            raise ValueError("D1 pilot requires 2 native steps per PD and 10 per policy tick")
        period = config["native_dt_s"] * config["policy_native_steps"]
        if not math.isclose(config["native_dt_s"], 0.002, abs_tol=1e-12):
            raise ValueError("D1 pilot requires 0.002 s native timestep")
        if not math.isclose(config["reward"]["multiply_total_by_policy_dt_s"], period,
                            abs_tol=1e-12):
            raise ValueError("Reward integration period must equal the policy period")
        self.episode_ticks = round(config["episode_seconds"] / period)
        self.first_zero_ticks = round(config["commands"]["first_zero_seconds"] / period)
        self.command_hold_ticks = round(config["commands"]["resample_seconds"] / period)
        if (not math.isclose(self.episode_ticks * period, config["episode_seconds"], abs_tol=1e-9)
                or not math.isclose(self.first_zero_ticks * period,
                                    config["commands"]["first_zero_seconds"], abs_tol=1e-9)
                or not math.isclose(self.command_hold_ticks * period,
                                    config["commands"]["resample_seconds"], abs_tol=1e-9)):
            raise ValueError("Episode and command times must be integer policy ticks")

    @staticmethod
    def _validate_eval_sequence(sequence):
        if sequence is None:
            return None
        if not sequence or sequence[0]["start_s"] != 0:
            raise ValueError("eval_command_sequence must start at t=0")
        previous = -math.inf
        result = []
        for segment in sequence:
            start = float(segment["start_s"])
            vx, wz = float(segment["vx"]), float(segment["wz"])
            if start <= previous or not all(map(math.isfinite, (start, vx, wz))):
                raise ValueError("Evaluation command sequence must have increasing finite start times")
            if abs(vx) > 0.3 or abs(wz) > 0.3:
                raise ValueError("Evaluation commands exceed frozen ±0.3 limits")
            result.append((start, vx, wz))
            previous = start
        return result

    def _resolve_model_indices(self):
        self.joint_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                   for name in JOINT_NAMES], dtype=np.int64)
        if np.any(self.joint_ids < 0):
            raise ValueError("Scene is missing an audited D1 joint")
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids].copy()
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids].copy()
        free = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if free < 0 or self.model.jnt_type[free] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("D1 floating base joint not found")
        self.free_qpos = int(self.model.jnt_qposadr[free])
        self.free_dof = int(self.model.jnt_dofadr[free])
        self._sensors = {}
        for name in ("trunk_gyro", "trunk_quat", "frame_vel"):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise ValueError(f"Scene is missing sensor {name}")
            self._sensors[name] = (int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid]))
        self.wheel_geoms = set()
        for name in FOOT_BODIES:
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body < 0:
                raise ValueError(f"Scene is missing wheel body {name}")
            for gid in range(self.model.body_geomadr[body],
                             self.model.body_geomadr[body] + self.model.body_geomnum[body]):
                if (self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
                        and self.model.geom_contype[gid] != 0):
                    self.wheel_geoms.add(int(gid))
        if len(self.wheel_geoms) != 4:
            raise ValueError("Expected exactly four colliding wheel cylinders")

    def _sensor(self, name):
        address, size = self._sensors[name]
        return self.data.sensordata[address:address + size].copy()

    def _place_static_pose(self, joint_q, yaw):
        self.data.qpos[self.free_qpos:self.free_qpos + 3] = (0, 0, 0)
        self.data.qpos[self.free_qpos + 3:self.free_qpos + 7] = (
            math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.data.qpos[self.qpos_ids] = joint_q
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        self.data.qfrc_applied[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._counts["forward_calls"] += 1

    def _wheel_lowest_z(self):
        bottoms = []
        for gid in self.wheel_geoms:
            rotation = self.data.geom_xmat[gid].reshape(3, 3)
            radius, half_length = self.model.geom_size[gid, :2]
            vertical_half_extent = (radius * np.linalg.norm(rotation[2, :2]) +
                                    half_length * abs(rotation[2, 2]))
            bottoms.append(float(self.data.geom_xpos[gid, 2] - vertical_half_extent))
        return min(bottoms)

    def _measure_nominal_base_height(self):
        mujoco.mj_resetData(self.model, self.data)
        self._place_static_pose(DEFAULT_Q, 0.0)
        height = -self._wheel_lowest_z()
        if not 0.25 < height < 0.8:
            raise ValueError(f"Implausible static wheel-ground base height: {height:.4f}m")
        return height

    def _episode_rng(self, stream):
        return np.random.default_rng(np.random.SeedSequence(
            [self.base_seed, self.rank, self._episode_index, stream]))

    def _pick_delay(self):
        if self.eval_delay_ticks is not None:
            return int(self.eval_delay_ticks)
        delay = self.config["delay"]
        if self.arm == "B":
            return int(delay["B_fixed_ticks"])
        if self.arm == "M":
            return int(delay["M_fixed_ticks"])
        return int(self._delay_rng.choice(delay["R_uniform_ticks"]))

    def _update_command(self):
        if self._manual_command:
            return
        elapsed = self._episode_steps * self.config["native_dt_s"] * self.config["policy_native_steps"]
        if self.eval_command_sequence is not None:
            segment = self.eval_command_sequence[0]
            for candidate in self.eval_command_sequence:
                if candidate[0] <= elapsed + 1e-12:
                    segment = candidate
                else:
                    break
            self._command[:] = (segment[1], 0, segment[2])
            return
        if self._episode_steps < self.first_zero_ticks:
            self._command[:] = 0
            return
        phase = (self._episode_steps - self.first_zero_ticks) // self.command_hold_ticks
        if phase != self._command_phase:
            self._command_phase = phase
            command_config = self.config["commands"]
            if self._command_rng.random() < command_config["zero_probability"]:
                self._command[:] = 0
            else:
                self._command[:] = (self._command_rng.uniform(*command_config["vx_uniform_m_s"]),
                                    0.0, self._command_rng.uniform(*command_config["wz_uniform_rad_s"]))

    def _observe(self):
        # Refresh all sensors from the same current qpos/qvel state before reading.
        mujoco.mj_forward(self.model, self.data)
        self._counts["forward_calls"] += 1
        frame, self._filtered_gyro = build_observation(
            self._sensor("trunk_gyro"), self._sensor("trunk_quat"), self._command,
            self.data.qpos[self.qpos_ids], self.data.qvel[self.dof_ids],
            self._last_policy_action, self._filtered_gyro)
        self._current_obs = frame
        return frame

    def _actor_observation(self):
        return pack_actor_input(self._current_obs, self._history)

    def current_observation(self):
        """Current 627D actor input, useful after an external set_command()."""
        if self._current_obs is None or self._episode_done:
            raise gym.error.ResetNeeded("Call reset() before requesting an observation")
        return self._actor_observation().copy()

    def get_initial_state(self):
        """JSON-ready policy-boundary state for exact cross-language playback."""
        if self._current_obs is None or self._episode_done:
            raise gym.error.ResetNeeded("Call reset() before capturing initial state")
        return {
            "format": "d1-ppo-policy-boundary-state-v1",
            "interface_id": self.config.get("interface_id", "d1-selftrained-pilot03-v1"),
            "scene_xml_sha256": hashlib.sha256(self.scene_path.read_bytes()).hexdigest(),
            "robot_xml_sha256": self.config["robot_xml_sha256"],
            "interface_schema_sha256": self.config["interface_schema_sha256"],
            "audited_interface_schema_sha256": self.config.get(
                "audited_interface_schema_sha256", self.config["interface_schema_sha256"]),
            "action_scales": self.action_scales.tolist(),
            "nominal_base_height_m": self.nominal_base_height_m,
            "qpos": self.data.qpos.tolist(), "qvel": self.data.qvel.tolist(),
            "qacc_warmstart": self.data.qacc_warmstart.tolist(),
            "ctrl": self.data.ctrl.tolist(), "qfrc_applied": self.data.qfrc_applied.tolist(),
            "simulation_time_s": float(self.data.time),
            "command_vx_vy_wz": self._command.tolist(),
            "delay_ticks": self._delay_ticks,
            "delay_fifo_oldest_to_newest": [item.tolist() for item in self._delay_fifo],
            "last_clipped_policy_action": self._last_policy_action.tolist(),
            "executed_action": self._applied_action.tolist(),
            "filtered_gyro_body": self._filtered_gyro.tolist(),
            "current_observation_57": self._current_obs.tolist(),
            "history_oldest_to_newest_10x57": self._history.tolist(),
            "episode_policy_tick": self._episode_steps,
        }

    def set_command(self, vx: float, wz: float):
        """Override sampled commands until reset; vy is fixed to zero."""
        vx, wz = float(vx), float(wz)
        if not math.isfinite(vx) or not math.isfinite(wz) or abs(vx) > 0.3 or abs(wz) > 0.3:
            raise ValueError("Command must be finite with |vx|, |wz| <= 0.3")
        self._manual_command = True
        self._command[:] = (vx, 0.0, wz)
        if self._current_obs is not None:
            # Command changes do not trigger a second gyro-filter update.
            self._current_obs[6:9] = np.array([2 * vx, 0.0, 0.25 * wz], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            if seed < 0:
                raise ValueError("seed must be nonnegative")
            self.base_seed = int(seed)
            self._episode_index = 0
        self._reset_rng = self._episode_rng(0)
        self._command_rng = self._episode_rng(1)
        self._delay_rng = self._episode_rng(2)
        self._episode_index += 1
        self._episode_steps = 0
        self._episode_done = False
        self._command_phase = -1
        self._manual_command = False
        self._command[:] = 0
        self._filtered_gyro[:] = 0
        self._last_policy_action[:] = 0
        self._applied_action[:] = 0
        self._last_tau[:] = 0
        self._last_saturation[:] = False
        self._delay_ticks = self._pick_delay()
        self._delay_fifo = deque(np.zeros(ACTION_DIM, dtype=np.float32)
                                 for _ in range(self._delay_ticks))
        mujoco.mj_resetData(self.model, self.data)
        joint_noise = np.zeros(ACTION_DIM, dtype=np.float64)
        leg_indices = [i for i in range(ACTION_DIM) if i not in (3, 7, 11, 15)]
        joint_noise[leg_indices] = self._reset_rng.uniform(
            *self.config["reset"]["leg_joint_perturbation_uniform_rad"], size=len(leg_indices))
        yaw = float(self._reset_rng.uniform(*self.config["reset"]["yaw_perturbation_uniform_rad"]))
        self._place_static_pose(DEFAULT_Q + joint_noise, yaw)
        self.data.qpos[self.free_qpos + 2] = (self.config["reset"]["wheel_ground_clearance_m"]
                                              - self._wheel_lowest_z())
        self._update_command()
        self._observe()
        self._history[:] = self._current_obs
        self._counts["resets"] += 1
        return self._actor_observation(), {"delay_ticks": self._delay_ticks,
                                           "nominal_base_height_m": self.nominal_base_height_m,
                                           "reset_height_m": float(self.data.qpos[self.free_qpos + 2]),
                                           "episode_index": self._episode_index - 1}

    def _delayed_action(self, clipped):
        if self._delay_ticks == 0:
            return clipped.copy()
        self._delay_fifo.append(clipped.copy())
        return self._delay_fifo.popleft()

    def _contact_diagnostics(self):
        illegal = False
        self_contacts = 0
        threshold = self.config["termination"]["illegal_support_contact_normal_force_above_n"]
        force = np.zeros(6, dtype=np.float64)
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            body_a, body_b = int(self.model.geom_bodyid[a]), int(self.model.geom_bodyid[b])
            if body_a != 0 and body_b != 0:
                self_contacts += 1
                continue
            if body_a == 0 and body_b != 0:
                world, robot = a, b
            elif body_b == 0 and body_a != 0:
                world, robot = b, a
            else:
                continue
            if (robot in self.wheel_geoms or self.model.geom_contype[world] == 0
                    or self.model.geom_type[world] not in (
                        mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_BOX,
                        mujoco.mjtGeom.mjGEOM_HFIELD)):
                continue
            force[:] = 0
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if force[0] > threshold:
                illegal = True
        return illegal, self_contacts

    def _check_finite_state(self):
        if not (np.all(np.isfinite(self.data.qpos)) and
                np.all(np.isfinite(self.data.qvel)) and
                np.all(np.isfinite(self.data.qacc))):
            self._episode_done = True
            raise FloatingPointError("Non-finite MuJoCo state; abort this development run")
        if any(item.number > 0 for item in self.data.warning):
            self._episode_done = True
            raise FloatingPointError("MuJoCo emitted a numerical warning; abort this development run")

    def _reward(self, interval_command, previous_action, mean_tau_squared, illegal, self_contacts):
        quat = self._sensor("trunk_quat")
        body_velocity = inverse_rotate(quat, self._sensor("frame_vel"))
        gyro = self._sensor("trunk_gyro")
        gravity = inverse_rotate(quat, np.array([0.0, 0.0, -1.0]))
        qw, qx, qy, qz = map(float, quat)
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = math.asin(float(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)))
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        height = float(self.data.qpos[self.free_qpos + 2])
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        r = self.config["reward"]
        raw = {
            "vx_tracking": math.exp(-((body_velocity[0] - interval_command[0]) ** 2) /
                                    r["vx_tracking_error_scale_m2_s2"]),
            "wz_tracking": math.exp(-((gyro[2] - interval_command[2]) ** 2) /
                                    r["wz_tracking_error_scale_rad2_s2"]),
            "alive": 1.0,
            "body_vy_squared": float(body_velocity[1] ** 2),
            "gravity_xy_squared": float(np.dot(gravity[:2], gravity[:2])),
            "height_error_squared": (height - self.nominal_base_height_m) ** 2,
            "action_rate_mean_squared": float(np.mean(np.square(self._last_policy_action - previous_action))),
            "torque_sum_squared": mean_tau_squared,
            "illegal_body_contact": float(illegal),
        }
        weights = {name: r[name + "_weight"] for name in raw}
        contributions = {name: raw[name] * weights[name] for name in raw}
        policy_dt = self.config["reward"]["multiply_total_by_policy_dt_s"]
        reward = float(sum(contributions.values()) * policy_dt)
        if not math.isfinite(reward):
            self._episode_done = True
            raise FloatingPointError("Non-finite PPO reward; abort this development run")
        fallen = (height < self.nominal_base_height_m -
                  self.config["termination"]["base_height_below_nominal_by_m"] or
                  tilt > self.config["termination"]["tilt_above_rad"])
        metrics = {"body_vx_m_s": float(body_velocity[0]),
                   "body_vy_m_s": float(body_velocity[1]),
                   "body_horizontal_speed_m_s": float(np.linalg.norm(body_velocity[:2])),
                   "body_wz_rad_s": float(gyro[2]),
                   "base_x_m": float(self.data.qpos[self.free_qpos]),
                   "base_y_m": float(self.data.qpos[self.free_qpos + 1]),
                   "base_height_m": height,
                   "base_roll_rad": roll, "base_pitch_rad": pitch, "base_yaw_rad": yaw,
                   "tilt_rad": tilt,
                   "fallen": bool(fallen), "illegal_body_contact": bool(illegal),
                   "self_contact_count": int(self_contacts),
                   "tau_saturation": bool(np.any(self._last_saturation)),
                   "tau_saturation_count": int(np.count_nonzero(self._last_saturation)),
                   "tau_saturation_fraction": float(np.mean(self._last_saturation)),
                   "applied_action": self._applied_action.tolist()}
        scaled_contributions = {name: value * policy_dt for name, value in contributions.items()}
        return reward, {"raw": raw, "weights": weights,
                        "contributions_per_second": contributions,
                        "contributions_per_tick": scaled_contributions,
                        "policy_dt_s": policy_dt}, metrics

    def step(self, action):
        if self._episode_done:
            raise gym.error.ResetNeeded("Call reset() before step() or after episode end")
        if self.max_transitions is not None and self._counts["policy_transitions"] >= self.max_transitions:
            raise RuntimeError("Hard max_transitions reached before integration")
        self._check_finite_state()
        clipped = clip_action(action)
        interval_command = self._command.copy()
        previous_action = self._last_policy_action.copy()
        # The old C++ history input excludes the current observation at inference.
        # Insert it now, after the policy action for this tick has been supplied.
        self._history[:-1] = self._history[1:].copy()
        self._history[-1] = self._current_obs
        self._last_policy_action[:] = clipped
        self._applied_action[:] = self._delayed_action(clipped)
        native_steps = self.config["policy_native_steps"]
        pd_period = self.config["pd_native_steps"]
        tau_squared_sum = 0.0
        illegal_during_interval = False
        self_contacts_max = 0
        saturation_any = np.zeros(ACTION_DIM, dtype=bool)
        # Count attempted work before integration so failed numerical steps
        # still consume the hard budget rather than vanishing from counters.
        self._counts["policy_transitions"] += 1
        for native_tick in range(native_steps):
            if native_tick % pd_period == 0:
                torque, saturation = action_to_torque(
                    self._applied_action, self.data.qpos[self.qpos_ids],
                    self.data.qvel[self.dof_ids], scale=self.action_scales)
                self.data.qfrc_applied[:] = 0
                self.data.qfrc_applied[self.dof_ids] = torque
                self.data.ctrl[:] = 0
                self._last_tau[:] = torque
                self._last_saturation[:] = saturation
                saturation_any |= saturation
                self._counts["pd_updates"] += 1
            tau_squared_sum += float(np.sum(np.square(self._last_tau)))
            self._counts["native_steps"] += 1
            try:
                mujoco.mj_step(self.model, self.data)
            except Exception:
                self._episode_done = True
                raise
            self._check_finite_state()
            illegal_now, self_contacts_now = self._contact_diagnostics()
            illegal_during_interval |= illegal_now
            self_contacts_max = max(self_contacts_max, self_contacts_now)
        self._episode_steps += 1
        self._update_command()
        self._observe()
        self._check_finite_state()
        illegal_now, self_contacts_now = self._contact_diagnostics()
        illegal_during_interval |= illegal_now
        self_contacts_max = max(self_contacts_max, self_contacts_now)
        self._last_saturation[:] = saturation_any
        reward, reward_terms, metrics = self._reward(
            interval_command, previous_action, tau_squared_sum / native_steps,
            illegal_during_interval, self_contacts_max)
        terminated = metrics["fallen"] or metrics["illegal_body_contact"]
        truncated = self._episode_steps >= self.episode_ticks and not terminated
        self._episode_done = bool(terminated or truncated)
        observation = self._actor_observation()
        info = {"reward_terms": reward_terms, **metrics,
                "command_during_interval": interval_command.tolist(),
                "next_command": self._command.tolist(),
                "delay_ticks": self._delay_ticks, "episode_policy_tick": self._episode_steps,
                "TimeLimit.truncated": bool(truncated)}
        if self._episode_done:
            info["terminal_observation"] = observation.copy()
        return observation, reward, bool(terminated), bool(truncated), info

    def counters(self):
        """Cumulative work across all resets; no implicit disk logging."""
        return dict(self._counts)

    def close(self):
        self._episode_done = True
