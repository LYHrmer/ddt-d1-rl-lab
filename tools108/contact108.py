"""Paired native D1 step contacts, observed at the original 500 Hz solver and 50 Hz actor clocks."""
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read, write
from training.d1_rl.interface import DEFAULT_Q, JOINT_NAMES


STEP_NAME = "step108"
KNEES = [JOINT_NAMES.index("FL_calf_joint"), JOINT_NAMES.index("FR_calf_joint")]
SCALE_RAD = 0.5
DT = 0.002
SUBSTEPS = 10
WHEEL_ORDER = ("FL", "FR", "RL", "RR")
CHECKPOINT_SHA = "9663c97d27ad307fb71ca2efbadf0fa3e1fa7c9e87b8b0e40d8a05f22ab8b4c0"


def add_step(worldbody, geometry):
    x, length, width, height = (float(geometry[k]) for k in ("front_x_m", "length_m", "half_width_m", "height_m"))
    require(all(np.isfinite([x, length, width, height])) and min(x, length, width, height) > 0.,
            "invalid step geometry")
    require(worldbody.find(f"geom[@name='{STEP_NAME}']") is None, "duplicate step")
    # The lower face is flush with the flat road. This static world geom keeps native collision filters.
    ET.SubElement(worldbody, "geom", name=STEP_NAME, type="box", pos=f"{x + length/2:.12g} 0 {height/2:.12g}",
                  size=f"{length/2:.12g} {width:.12g} {height/2:.12g}", friction="1.0 0.005 0.0001",
                  contype="1", conaffinity="1", rgba="0.62 0.53 0.31 1")


def build_contact_scene(directory, geometry):
    from tools87.terrain87 import _write_scene

    base = _write_scene(directory)
    tree = ET.parse(base)
    add_step(tree.find("worldbody"), geometry)
    destination = base.with_name("step108.xml")
    with destination.open("xb") as stream:
        tree.write(stream, encoding="utf-8", xml_declaration=True)
    return destination


def load_contact_model(geometry):
    """A copy of the episode model for rendering archived poses."""
    import mujoco
    from tools87.terrain87 import _set_road

    with TemporaryDirectory(prefix="d1-step108-view-") as directory:
        model = mujoco.MjModel.from_xml_path(str(build_contact_scene(directory, geometry)))
    _set_road(model, "flat")
    return model


def intervention(nominal, arm, tick, trigger_tick, spec):
    """Offset only the two front knees before the existing action limiter and delayed actuator FIFO."""
    nominal = np.asarray(nominal, dtype=np.float32)
    require(nominal.shape == (16,) and np.isfinite(nominal).all() and (np.abs(nominal) <= 1.).all()
            and arm in ("original", "flexion"), "invalid actor output or intervention")
    active = (arm == "flexion" and trigger_tick is not None and trigger_tick <= tick < trigger_tick + spec["pulse_ticks"])
    angle = float(spec["pulse_rad"]) if active else 0.
    issued = nominal.copy()
    issued[KNEES] = np.clip(issued[KNEES] + angle/SCALE_RAD, -1., 1.)
    return issued, angle


def first_front_contact(step_forces, threshold):
    forces = np.asarray(step_forces, dtype=np.float64)
    require(forces.ndim == 2 and forces.shape[1] == 4 and np.isfinite(forces).all()
            and (forces >= 0.).all() and np.isfinite(threshold) and threshold > 0., "invalid wheel forces")
    hits = np.flatnonzero((forces[:, :2] >= threshold).any(axis=1))
    return int(hits[0]) if len(hits) else None


def contact_channels(records, slots, floor, step):
    """Aggregate positive normal forces; retain individual step contacts for face/crest diagnostics."""
    wheels = np.zeros(4)
    floor_wheels = np.zeros(4)
    vectors = np.zeros((4, 3))
    pos_weighted = np.zeros((4, 3))
    step_other = floor_other = 0.
    raw = []
    for g1, g2, force, normal, position in records:
        if force <= 0. or (g1 not in (floor, step) and g2 not in (floor, step)):
            continue
        terrain_first = g1 in (floor, step)
        terrain, geom = (g1, g2) if terrain_first else (g2, g1)
        if geom in (floor, step):
            continue
        wheel = slots.get(geom, -1)
        if terrain == floor:
            if wheel >= 0:
                floor_wheels[wheel] += force
            else:
                floor_other = max(floor_other, force)
        else:
            direction = np.asarray(normal) * (1. if terrain_first else -1.)
            raw.append([wheel, geom, force, *position, *direction])
            if wheel >= 0:
                wheels[wheel] += force
                vectors[wheel] += force * direction
                pos_weighted[wheel] += force * np.asarray(position)
            else:
                step_other = max(step_other, force)
    positions = np.full((4, 3), np.nan)
    np.divide(pos_weighted, wheels[:, None], out=positions, where=wheels[:, None] > 0.)
    return dict(step_force_n=wheels, ground_force_n=floor_wheels, step_normal_vector_n=vectors,
                step_contact_pos_m=positions, nonwheel_step_max_n=step_other, nonwheel_floor_max_n=floor_other), raw


def _environment(seed, spec):
    import gymnasium as gym
    import mujoco
    from tools87.terrain87 import D1Terrain87, TERRAIN_DIM
    from tools89.course89 import wheel_geom_slots
    from training.d1_rl.delay_env18 import D1Delay18Env
    from training.d1_rl.interface import ACTOR_DIM
    from training.d1_rl.task03d import DEFAULT_CONFIG_03D

    class StepEnvironment(D1Terrain87):
        def _contact_diagnostics(self):
            original = super()._contact_diagnostics()
            index = self._counts["native_steps"] - 1
            if self.substeps is None or index < len(self.substeps):
                return original
            require(index == len(self.substeps), "missed native contact interval")
            records = []
            force = np.empty(6)
            for n, item in enumerate(self.data.contact[:self.data.ncon]):
                a, b = int(item.geom1), int(item.geom2)
                if a not in (self.floor_geom, self.step_geom) and b not in (self.floor_geom, self.step_geom):
                    continue
                mujoco.mj_contactForce(self.model, self.data, n, force)
                records.append((a, b, float(force[0]), item.frame[:3].copy(), item.pos.copy()))
            channels, contacts = contact_channels(records, self.wheel_slots, self.floor_geom, self.step_geom)
            channels["illegal_substep"] = bool(original[0])
            channels["self_contacts_substep"] = int(original[1])
            self.substeps.append(channels)
            self.step_contacts.extend([[index, *row] for row in contacts])
            return original

    # D1Terrain87 builds its own scene in its constructor. Initialize its native parent with
    # the identical config, delays and course; inherit its reset, stepping and sensing unmodified.
    env = StepEnvironment.__new__(StepEnvironment)
    env.terrain_mode, env.perception = "flat", True
    env.substeps, env.step_contacts = None, []
    env._workspace = TemporaryDirectory(prefix="d1-contact108-")
    try:
        scene = build_contact_scene(env._workspace.name, spec["geometry"])
        D1Delay18Env.__init__(env, config=DEFAULT_CONFIG_03D, arm="B", seed=seed, eval_delay_ticks=1,
                             scene=scene, max_transitions=spec["ticks"],
                             eval_command_sequence=[{"start_s": 0., "vx": 0., "wz": 0.},
                                                    {"start_s": 1., "vx": .20, "wz": 0.}])
        env.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(ACTOR_DIM + TERRAIN_DIM,), dtype=np.float32)
        env.base_body = env.model.body("base_link").id
        env.floor_geom, env.step_geom = env.model.geom("floor").id, env.model.geom(STEP_NAME).id
        env.label_data = mujoco.MjData(env.model)
        env.label_kinematics_calls = 0
        env.wheel_slots = wheel_geom_slots(env)
        env.ordered_wheels = [next(g for g, i in env.wheel_slots.items() if i == w) for w in range(4)]
        env.front_geoms = env.ordered_wheels[:2]
        env.last_raw_features = np.zeros(TERRAIN_DIM, dtype=np.float32)
        env._masked = np.zeros_like(env.last_raw_features)
        env.current_terrain, env._pending = "flat", None
        env.sensor_capture_tick = env.sensor_delivery_tick = -1
        require(np.array_equal(env.action_scales[KNEES], [SCALE_RAD, SCALE_RAD]) and env.model.geom_bodyid[env.step_geom] == 0
                and env.model.geom_contype[env.step_geom] and env.model.geom_conaffinity[env.step_geom]
                and env.config["policy_native_steps"] == SUBSTEPS and env.config["pd_native_steps"] == 2
                and np.isclose(env.model.opt.timestep, DT), "changed native D1 action, clock or collision contract")
        return env
    except BaseException:
        env._workspace.cleanup()
        raise


def _append(rows, values):
    for key, value in values.items():
        rows.setdefault(key, []).append(value)


def _boundary(env, obs):
    import mujoco

    # mj_step's contact solution belongs to the integrated interval. Read decision-boundary
    # geometry from its exact qpos without refreshing or modifying the live solver state.
    env.label_data.qpos[:] = env.data.qpos
    mujoco.mj_kinematics(env.model, env.label_data)
    env.label_kinematics_calls += 1
    centers = env.label_data.geom_xpos[env.ordered_wheels].copy()
    return dict(time_s=float(env.data.time), qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                actor_observation=np.asarray(obs, dtype=np.float32).copy(),
                front_axle_x_m=float(centers[:2, 0].mean()), wheel_xyz_m=centers,
                sensor_capture_tick=int(env.sensor_capture_tick), sensor_delivery_tick=int(env.sensor_delivery_tick))


def _save_trace(folder, rows, env, report, complete):
    if not rows and (env is None or not env.substeps):
        return
    subs = env.substeps or []
    data = {k: np.asarray(v) for k, v in rows.items()}
    data.update({k: np.asarray([r[k] for r in subs]) for k in (subs[0] if subs else ())})
    data["step_contacts"] = np.asarray(env.step_contacts, dtype=float).reshape(-1, 10)
    for k in ("trigger_tick", "first_contact_substep"):
        data[k] = -1 if report[k] is None else report[k]
    name = "trace.npz" if complete else "partial_trace.npz"
    path = folder / name
    arrays(path, data)
    report.update(trace_file=name, trace_sha256=digest(path))


def run_contact(folder, seed, arm, spec, predict):
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    report = dict(status="PARTIAL", seed=int(seed), arm=arm, checkpoint_sha256=spec["checkpoint_sha256"],
                  geometry=spec["geometry"], ticks_completed=0, full_horizon=False, trigger_tick=None,
                  first_contact_substep=None, terminal_reason="error", trace_file=None, trace_sha256=None)
    rows, env = {}, None
    try:
        env = _environment(int(seed), spec)
        obs, initial = env.reset(seed=int(seed))
        require(initial["delay_ticks"] == 1 and obs.shape == (643,) and env.data.time == 0., "unexpected reset")
        env.substeps = []
        _append(rows, _boundary(env, obs))
        report["initial_qpos_sha256"] = _array_sha(env.data.qpos)
        for tick in range(spec["ticks"]):
            require(np.array_equal(obs[627:], env.last_raw_features), "policy received a stale range packet")
            nominal = predict(obs[:627], obs[627:])
            issued, requested = intervention(nominal, arm, tick, report["trigger_tick"], spec)
            obs, reward, terminated, truncated, info = env.step(issued)
            require(len(env.substeps) == (tick + 1) * SUBSTEPS, "native/policy contact clock mismatch")
            _append(rows, _boundary(env, obs))
            _append(rows, dict(nominal_action=nominal.copy(), issued_action=issued.copy(),
                               applied_action=env._applied_action.copy(), pulse_target_rad=requested,
                               actual_issue_offset_rad=(issued[KNEES] - nominal[KNEES]) * SCALE_RAD,
                               applied_knee_target_rad=DEFAULT_Q[KNEES] + env._applied_action[KNEES] * SCALE_RAD,
                               leg_target_rad=np.array(info["leg_target_rad"]), wheel_target_rad_s=np.array(info["wheel_target_rad_s"]),
                               last_torque_nm=np.array(info["last_torque_nm"]), reward=float(reward),
                               body_vx_m_s=float(info["body_vx_m_s"]), roll_rad=float(info["base_roll_rad"]),
                               pitch_rad=float(info["base_pitch_rad"]), tilt_rad=float(info["tilt_rad"]),
                               fallen=bool(info["fallen"]), illegal_contact=bool(info["illegal_body_contact"]),
                               terminated=bool(terminated), truncated=bool(truncated)))
            report["ticks_completed"] = tick + 1
            if report["trigger_tick"] is None:
                local = first_front_contact([v["step_force_n"] for v in env.substeps[-SUBSTEPS:]], spec["trigger_n"])
                if local is not None:
                    report["first_contact_substep"] = tick * SUBSTEPS + local
                    report["trigger_tick"] = tick + 1
            if terminated or truncated:
                report["terminal_reason"] = ("fallen" if info["fallen"] else "illegal_contact" if info["illegal_body_contact"]
                                             else "native_time_limit")
                break
        else:
            report["terminal_reason"] = "horizon"
        report["full_horizon"] = report["ticks_completed"] == spec["ticks"] and not any(rows["terminated"])
        report["first_contact_end_s"] = (None if report["first_contact_substep"] is None else
                                         (report["first_contact_substep"] + 1) * DT)
        report["post32_available"] = (report["trigger_tick"] is not None and
                                      report["trigger_tick"] + 32 <= report["ticks_completed"])
        report["status"] = "COMPLETE" if report["full_horizon"] else "PARTIAL"
    except BaseException as err:
        report["error"] = f"{type(err).__name__}: {err}"
        raise
    finally:
        try:
            if env is not None:
                report["counts"] = env.counters()
                report["label_kinematics_calls"] = env.label_kinematics_calls
                _save_trace(folder, rows, env, report, report["status"] == "COMPLETE")
        finally:
            write(folder / "report.json", report)
            if env is not None:
                env.close()
    return report


def _array_sha(value):
    from hashlib import sha256
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def collect(episodes_dir, spec):
    """Exclusively record a registered three-reset matrix. The caller owns serial resource admission."""
    from tools89.train89 import load_predictor89

    folder = Path(episodes_dir)
    folder.mkdir(exist_ok=False)
    checkpoint = ROOT / spec["checkpoint"]
    require(spec["arms"] == ["original", "flexion"] and len(set(spec["reset_seeds"])) == len(spec["reset_seeds"]) == 3
            and spec["ticks"] == 200 and spec["pulse_ticks"] == 16 and spec["pulse_rad"] == -.08
            and spec["trigger_n"] == 1. and checkpoint.name == "policy.zip" and not checkpoint.is_symlink()
            and digest(checkpoint) == spec["checkpoint_sha256"] == CHECKPOINT_SHA, "unregistered input or budget")
    receipt = dict(status="PARTIAL", reset_seeds=spec["reset_seeds"], arms=spec["arms"],
                   checkpoint_sha256=CHECKPOINT_SHA, episodes=[])
    try:
        predict = load_predictor89(checkpoint.parent)
        for seed in spec["reset_seeds"]:
            for arm in spec["arms"]:
                name = f"seed{seed}_{arm}"
                item = run_contact(folder / name, seed, arm, spec, predict)
                receipt["episodes"].append(dict(name=name, **item))
        if all(item["status"] == "COMPLETE" for item in receipt["episodes"]):
            receipt["status"] = "COMPLETE"
    finally:
        write(folder / "index.json", receipt)
    return receipt
