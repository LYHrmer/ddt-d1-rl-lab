"""Causal, measured-input replay of the 180 registered WBC targets (no integration)."""
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from training.d1_rl.interface import DEFAULT_Q, JOINT_NAMES, WHEEL_INDICES


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/experiments90/stand_wbc.json"
PHASES = ("flat", "slope_stand", "front_contact")
DT = 0.020
SUPPORT_GAP_M = 0.025
MAX_EXTENSION_M = 0.15
MAX_LATERAL_M = 0.18
UP = np.array([0., 0., 1.])


def _elapsed(start):
    return (perf_counter_ns() - start) / 1e6


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value)}")


def _write(path, value):
    with path.open("x") as file:
        _line(file, value)


def _upward(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("degenerate footprint normal")
    return vectors * np.where(vectors[..., 2:3] >= 0, 1., -1.) / norms


def footprint_normals(bottoms):
    """Use the encoded front/rear span on this side and the left/right span on this axle."""
    fore = bottoms[[0, 1, 0, 1]] - bottoms[[2, 3, 2, 3]]
    left = bottoms[[0, 0, 2, 2]] - bottoms[[1, 1, 3, 3]]
    return _upward(np.cross(fore, left))


def rim_points(centers, axes, radius, half_width, normals):
    axial_dot = (axes * normals).sum(axis=1)
    radial = normals - axial_dot[:, None] * axes
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    return centers - radius[:, None]*radial - (half_width*np.sign(axial_dot))[:, None]*axes


class TerrainPatches:
    """Keep measured per-lane patches in an arbitrary episode-local odometry frame."""

    def __init__(self):
        self.packets = []
        self.lanes = [None, None]

    def add(self, points, valid, tick):
        new = [[], []]
        for lane in (0, 1):
            ids = np.flatnonzero(valid[lane])
            for i, j in zip(ids[:-1], ids[1:]):
                # A valid return in the other lane supplies an independent lateral tangent.
                mates = [b for b in (i, j) if valid[1 - lane, b]]
                if not mates:
                    continue
                b = mates[0]
                a, end = points[lane, i], points[lane, j]
                along = end - a
                normal = np.cross(along, points[1 - lane, b] - points[lane, b])
                if np.linalg.norm(along[:2]) < .02 or abs(normal[2]) < 1e-6:
                    continue
                normal = _upward(normal)
                if normal[2] < .5:
                    continue
                new[lane].append(np.r_[a, along[:2] / np.linalg.norm(along[:2]),
                                        np.linalg.norm(along[:2]), normal, tick])
        self.packets.append(new)
        for lane in (0, 1):
            patches = [p for packet in self.packets for p in packet[lane]]
            self.lanes[lane] = np.asarray(patches) if patches else None

    def at_wheel(self, lane, position):
        patches = self.lanes[lane]
        if patches is None:
            return None
        diff = position[:2] - patches[:, :2]
        along = np.sum(diff * patches[:, 3:5], axis=1)
        lateral = np.abs(diff[:, 0]*patches[:, 4] - diff[:, 1]*patches[:, 3])
        outside = np.maximum.reduce([-along, along - patches[:, 5], np.zeros(len(along))])
        ids = np.flatnonzero((outside <= MAX_EXTENSION_M) & (lateral <= MAX_LATERAL_M))
        if not ids.size:
            return None
        # Prefer a real spatial bracket, then the smallest extension/closest patch center.
        chosen = ids[np.lexsort((-patches[ids, 9], np.abs(along[ids] - .5*patches[ids, 5]),
                                 lateral[ids], outside[ids]))[0]]
        p = patches[chosen]
        source = "range_bracket" if outside[chosen] <= 1e-8 else "range_extension"
        return p[:3], p[6:9], source, int(p[9]), float(outside[chosen])


def _wheel_geoms(model, mj):
    ids = [model.joint(JOINT_NAMES[i]).id for i in WHEEL_INDICES]
    wheels = model.jnt_bodyid[ids]
    geoms = []
    for body in wheels:
        matching = np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0) &
                                  (model.geom_type == mj.mjtGeom.mjGEOM_CYLINDER))
        if matching.size != 1:
            raise ValueError("D1 needs exactly one collision cylinder on each wheel body")
        geoms.append(matching.item())
    return wheels, np.asarray(geoms)


class SensorEstimator:
    """The only online inputs are the actual current actor frame and delayed, held returns."""

    def __init__(self, model):
        import mujoco
        from scipy.spatial.transform import Rotation
        from tools87.terrain87 import (DIRECTIONS, MAX_RANGE_M, MOUNTS_M, SENSOR_DELAY_TICKS, capture_due)

        self.mj, self.model, self.rotation = mujoco, model, Rotation
        self.directions, self.mounts, self.range_max = DIRECTIONS, MOUNTS_M, MAX_RANGE_M
        self.delay, self.capture_due = SENSOR_DELAY_TICKS, capture_due
        self.data = mujoco.MjData(model)
        joints = [model.joint(name).id for name in JOINT_NAMES]
        self.qadr, self.dadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.bodies, self.geoms = _wheel_geoms(model, mujoco)
        self.radius, self.half_width = model.geom_size[self.geoms, :2].T
        self.jac = np.empty((3, model.nv))
        self.patches = TerrainPatches()
        self.last_tick = -1
        self.capture_pose = None
        self.last_delivery = None
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.yaw = 0.
        self.wheel_angles = np.zeros(4)
        self.previous_rates = np.zeros(4)
        self.support = np.ones(4, dtype=bool)
        self.normals = np.tile(UP, (4, 1))

    def update(self, tick, proprio, visible):
        from tools90.wbc90 import EstimatedState90

        if tick != self.last_tick + 1:
            raise ValueError("estimator requires uninterrupted chronological episode measurements")
        self.last_tick = tick
        frame, visible = np.asarray(proprio, dtype=float)[:57], np.asarray(visible, dtype=float)
        if (frame.shape != (57,) or visible.shape != (16,) or not np.isfinite(frame).all()
                or not np.isfinite(visible).all() or not np.isin(visible[8:], [0, 1]).all()
                or np.any((visible[:8] < 0) | (visible[:8] > 1))):
            raise ValueError("invalid saved measurements")
        gyro, gravity = frame[:3] / .25, frame[3:6]
        if abs(np.linalg.norm(gravity) - 1.) > 1e-4:
            raise ValueError("invalid projected gravity")
        roll = np.arctan2(-gravity[1], -gravity[2])
        pitch = np.arcsin(np.clip(gravity[0], -1, 1))
        if abs(np.cos(pitch)) < .05:
            raise ValueError("gravity/gyro heading chart is singular")
        if tick:
            self.yaw += DT * (np.sin(roll)*gyro[1] + np.cos(roll)*gyro[2]) / np.cos(pitch)
        rotation = self.rotation.from_euler("xyz", [roll, pitch, self.yaw])
        quat = np.roll(rotation.as_quat(), 1)
        joint_q, joint_dq = DEFAULT_Q + frame[9:25], frame[25:41] / .05
        if tick:
            self.wheel_angles += .5*DT*(joint_dq[WHEEL_INDICES] + self.previous_rates)
        joint_q[WHEEL_INDICES] = self.wheel_angles
        self.previous_rates = joint_dq[WHEEL_INDICES].copy()

        data, model = self.data, self.model
        data.qpos[:3] = 0.
        data.qpos[3:7], data.qpos[self.qadr] = quat, joint_q
        data.qvel[:3], data.qvel[3:6], data.qvel[self.dadr] = 0., gyro, joint_dq
        self.mj.mj_kinematics(model, data)
        self.mj.mj_comPos(model, data)
        centers = data.geom_xpos[self.geoms].copy()
        axes = data.geom_xmat[self.geoms].reshape(4, 3, 3)[:, :, 2]
        points = rim_points(centers, axes, self.radius, self.half_width, self.normals)
        odom_wheels = np.flatnonzero(self.support)
        velocities = []
        for i in odom_wheels:
            self.mj.mj_jac(model, data, self.jac, None, points[i], self.bodies[i])
            velocities.append(-self.jac @ data.qvel)
        velocity = np.median(velocities, axis=0) if velocities else self.velocity.copy()
        if tick:
            self.position += .5*DT*(self.velocity + velocity)
        else:
            self.position[2] = -np.mean(points[:, 2])
        self.velocity = velocity
        centers += self.position
        points += self.position
        if self.capture_due(tick):
            self.capture_pose = (tick, self.position.copy(), rotation.as_matrix())
        valid = visible[8:].astype(bool).reshape(2, 4)
        if self.capture_pose and tick == self.capture_pose[0] + self.delay:
            captured, origin, attitude = self.capture_pose
            ranges = (visible[:8]*self.range_max).reshape(2, 4)
            body_points = self.mounts[:, None, :] + ranges[:, :, None]*self.directions
            self.patches.add(origin + body_points @ attitude.T, valid, captured)
            self.capture_pose = None
            self.last_delivery = captured

        horizontal = rim_points(centers, axes, self.radius, self.half_width, np.tile(UP, (4, 1)))
        normals = footprint_normals(horizontal)
        # A fitted footprint provides a clearly labelled prior for wheels outside the sensor map.
        center = horizontal.mean(axis=0)
        _, _, vh = np.linalg.svd(horizontal - center, full_matrices=False)
        footprint_plane = _upward(vh[-1])
        anchors = np.tile(center, (4, 1))
        gap_normals = np.tile(footprint_plane, (4, 1))
        sources = ["kinematic_prior"] * 4
        source_ticks, extensions = [None] * 4, [None] * 4
        for i in range(4):
            patch = self.patches.at_wheel(i % 2, points[i])
            if patch is not None:
                anchors[i], normals[i], sources[i], source_ticks[i], extensions[i] = patch
                gap_normals[i] = normals[i]
        points = rim_points(centers, axes, self.radius, self.half_width, gap_normals)
        gaps = ((points - anchors)*gap_normals).sum(axis=1)
        self.support = gaps <= SUPPORT_GAP_M
        self.normals = normals.copy()
        state = EstimatedState90(quat, self.position.copy(), velocity.copy(), gyro, joint_q,
                                 joint_dq, self.support.copy(), normals)
        evidence = {"normal_sources": sources, "patch_capture_ticks": source_ticks,
                    "patch_extension_m": extensions, "normal_gap_m": gaps,
                    "valid_delayed_beams": int(valid.sum()), "latest_delivered_capture_tick": self.last_delivery,
                    "range_bracketed_wheel_count": sources.count("range_bracket"),
                    "support_flags_are_kinematic_estimates": True,
                    "odometry_support_wheels": odom_wheels, "odometry_held": not len(velocities),
                    "estimated_body_vx_m_s": float(rotation.inv().apply(velocity)[0])}
        return state, frame[6:9] / np.array([2., 2., .25]), evidence


def selected_windows(source, arrays):
    """Validate registration against phase labels; only the row identifiers cross into replay."""
    ticks = arrays["episode_tick"]
    starts = np.flatnonzero(ticks == 1)
    terrain, shifted, past = [arrays[k] for k in ("terrain", "slope_start", "loaded_wheels_past_toe")]
    command = arrays["proprio627"][:, 6] / 2.
    out = {}
    for phase in PHASES:
        selected = [i for a, b in source[phase] for i in range(a, b + 1)]
        if len(set(selected)) != 20 or selected != sorted(selected):
            raise ValueError(f"invalid {phase} selection")
        start = int(starts[np.searchsorted(starts, selected[0], side="right") - 1])
        if not np.array_equal(ticks[start:selected[-1]+1], np.arange(1, selected[-1]-start+2)):
            raise ValueError(f"crossed an episode boundary in {phase}")
        k = np.asarray(selected) - start
        if phase == "flat":
            valid = (terrain[start] == 0 and not shifted[start] and np.array_equal(k, np.arange(40, 60))
                     and np.all(command[selected[:10]] == 0.) and np.all(command[selected[10:]] > 0.))
        elif phase == "slope_stand":
            valid = (terrain[start] == 1 and shifted[start]
                     and np.array_equal(k, np.r_[0:10, 40:50]) and np.all(command[selected] == 0.))
        else:
            first = selected[0]
            hits = past[start:first, :2].any(axis=1) & ~past[start:first, 2:].any(axis=1)
            valid = (terrain[start] == 1 and not shifted[start] and np.all(np.diff(k) == 1)
                     and np.all(command[selected] > 0.) and hits[-1] and not hits[:-1].any())
        if not valid:
            raise ValueError(f"registered {phase} semantics disagree with archive")
        out[phase] = (start, selected)
    return out


def _source(source, spec, stage, plan89):
    from tools87.ppo87 import load_predictor
    from tools89.common89 import parent_folder

    path = ROOT / spec["source_pattern"].format(seed=source["seed"])
    receipt_path = path.parent / "report.json"
    receipt = json.loads(receipt_path.read_bytes())
    entries = [r for r in stage["models"] if r["model"] == path.parent.name]
    if (len(entries) != 1 or _hash(receipt_path) != entries[0]["report_sha256"]
            or receipt["status"] != "COMPLETE" or receipt["arm"] != "mixed"
            or receipt["parent_seed"] != source["parent"] or receipt["training_seed"] != source["seed"]
            or receipt["trace"]["completed_recorded_ticks"] != 4096 or receipt["hyperparameters"]["n_steps"] != 4096
            or path.is_symlink() or _hash(path) != source["trace_sha256"]):
        raise ValueError(f"unqualified archive at {path}")
    keys = ("proprio627", "visible_range", "issued", "applied", "episode_tick", "terrain", "slope_start",
            "loaded_wheels_past_toe", "qpos", "qvel", "last_torque_nm", "body_vx_m_s")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in keys}
    for key, arr in arrays.items():
        if len(arr) != 4096 or sha256(arr.tobytes()).hexdigest() != receipt["trace"]["array_sha256"][key]:
            raise ValueError(f"changed {key} at {path}")
    ids = np.flatnonzero(arrays["episode_tick"] > 1)
    if not np.array_equal(arrays["proprio627"][ids, 41:57], arrays["issued"][ids-1]):
        raise ValueError("historical action in actor observations differs from actual issued action")
    folder = parent_folder(source["parent"], plan89)
    if _hash(folder / "policy.zip") != receipt["parent_policy_sha256"]:
        raise ValueError("archived parent does not match nominal policy")
    predict = load_predictor(folder)
    return arrays, selected_windows(source, arrays), predict, receipt["parent_policy_sha256"]


def _rejected(reason):
    return dict(accepted=False, status="rejected", reason=reason, action=None, servo_torque_nm=None,
                solver_status=None, violations=None)


def _line(stream, value):
    # A rejected solver can still carry diagnostic +/-infinity. Keep the frame, the signed
    # failure value and a standards-compliant JSON stream when a backend reports one.
    encoded = json.dumps(value, default=_json, separators=(",", ":"), allow_nan=True)
    if any(token in encoded for token in ("NaN", "Infinity")):
        encoded = json.dumps(json.loads(encoded, parse_constant=lambda s: f"nonfinite({s})"),
                             allow_nan=False, separators=(",", ":"))
    stream.write(encoded + "\n")
    stream.flush()


def _test_frame(estimator, controller, predict, tick, proprio, ranges, history_error):
    """Time exactly one contiguous estimate, nominal-model assembly and enabled QP interval."""
    from tools90.wbc90 import passthrough_action

    start = perf_counter_ns()
    nominal = None
    prediction_error = None
    try:
        nominal = predict(proprio, ranges)
    except Exception as exc:
        prediction_error = f"parent prediction: {type(exc).__name__}: {exc}"
    ppo_ms = _elapsed(start)
    state = command = evidence = None
    start = perf_counter_ns()
    if history_error is None:
        try:
            state, command, evidence = estimator.update(tick, proprio, ranges)
        except Exception as exc:
            history_error = f"estimator: {type(exc).__name__}: {exc}"
    estimate_ms = _elapsed(start)
    attempted = state is not None and nominal is not None
    try:
        enabled = controller.correct(state, nominal, command) if attempted else _rejected(history_error or prediction_error)
    except Exception as exc:
        enabled = _rejected(f"controller exception: {type(exc).__name__}: {exc}")
    elapsed_ms = _elapsed(start)
    disabled = None
    identical = False
    if nominal is not None:
        original = np.asarray(nominal)
        direct = passthrough_action(original)
        identical = original.dtype == direct.dtype and direct.tobytes() == original.tobytes()
        if state is not None:
            try:
                disabled = controller.correct(state, original, command, enabled=False)
                restored = np.asarray(disabled["action"], dtype=original.dtype)
                identical = identical and restored.shape == original.shape and restored.tobytes() == original.tobytes()
            except Exception as exc:
                disabled = _rejected(f"bypass exception: {type(exc).__name__}: {exc}")
                identical = False
        else:
            disabled = dict(status="action_only_missing_state", action=direct.tolist(), servo_torque_nm=None)
    return {"ppo_ms": ppo_ms, "estimate_ms": estimate_ms, "elapsed_ms": elapsed_ms,
            "solver_attempted": attempted, "command": command, "nominal_action": nominal,
            "estimated_state": asdict(state) if state is not None else None,
            "estimate_evidence": evidence, "enabled": enabled, "bypass": disabled,
            "bypass_bitwise_equal": identical}, history_error


class TruthAudit:
    """Post-hoc cold same-boundary ground contact and speed scoring, never passed into WBC."""

    def __init__(self, terrain):
        import mujoco
        from tools87.terrain87 import load_model

        self.mj, self.model = mujoco, load_model(terrain)
        self.data = mujoco.MjData(self.model)
        _, geoms = _wheel_geoms(self.model, mujoco)
        self.ids = {g: i for i, g in enumerate(geoms)}
        self.floor = self.model.geom("floor").id
        self.dadr = self.model.jnt_dofadr[[self.model.joint(name).id for name in JOINT_NAMES]]

    def score(self, sample, estimated, evidence):
        data, mj = self.data, self.mj
        mj.mj_resetData(self.model, data)
        data.time = sample["tick"] * DT
        data.qpos[:], data.qvel[:] = sample["qpos"], sample["qvel"]
        data.qfrc_applied[self.dadr] = sample["last_torque_nm"]
        mj.mj_forward(self.model, data)
        support = np.zeros(4, dtype=bool)
        forces = np.zeros(4)
        normals = [None] * 4
        past = np.zeros(4, dtype=bool)
        from tools87.terrain87 import RAMP_START_X_M

        force = np.empty(6)
        for j, contact in enumerate(data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            if a == self.floor and b in self.ids:
                idx, n = self.ids[b], contact.frame[:3]
            elif b == self.floor and a in self.ids:
                idx, n = self.ids[a], -contact.frame[:3]
            else:
                continue
            mj.mj_contactForce(self.model, data, j, force)
            if force[0] <= 1.:
                continue
            support[idx] = True
            if sample["terrain"] == "up" and contact.pos[0] >= RAMP_START_X_M:
                past[idx] = True
            if force[0] > forces[idx]:
                forces[idx], normals[idx] = force[0], n.copy()
        angles = [None] * 4
        if estimated:
            # The sensor estimator's yaw origin is arbitrary. Compare normals at the same
            # heading, using each attitude to express its normal in its own base frame.
            from scipy.spatial.transform import Rotation
            predicted_rotation = Rotation.from_quat(np.roll(estimated["base_quat_wxyz"], -1)).as_matrix()
            actual_rotation = data.xmat[self.model.body("base_link").id].reshape(3, 3)
            for i, normal in enumerate(normals):
                if normal is not None:
                    cosine = (normal @ actual_rotation) @ (np.asarray(estimated["normals_world"][i]) @ predicted_rotation)
                    angles[i] = float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))
        true_vx = sample["body_vx_m_s"]
        return dict(cold_support=support, cold_peak_force_n=forces, cold_normals_world=normals,
                    normal_error_deg=angles, archived_body_vx_m_s=true_vx,
                    body_vx_error_m_s=None if evidence is None else evidence["estimated_body_vx_m_s"]-true_vx,
                    archived_loaded_past_toe=sample["loaded_wheels_past_toe"], cold_loaded_past_toe=past,
                    cold_archived_past_toe_equal=bool(np.array_equal(past, sample["loaded_wheels_past_toe"])))


def _diagnostic_summary(rows):
    support_counts = Counter()
    angles, errors = [], []
    geometry = Counter()
    for row in rows:
        estimate, evidence = row["estimated_state"], row["estimate_evidence"]
        if evidence:
            geometry.update(evidence["normal_sources"])
        score = row.get("score")
        if estimate and score:
            for pred, truth in zip(estimate["support"], score["cold_support"]):
                support_counts[(bool(pred), bool(truth))] += 1
            angles.extend(a for a in score["normal_error_deg"] if a is not None)
            errors.append(score["body_vx_error_m_s"])
    return dict(normal_sources_per_wheel=dict(geometry), fully_range_bracketed_frames=sum(
                    r["estimate_evidence"] is not None and
                    r["estimate_evidence"]["normal_sources"].count("range_bracket") == 4 for r in rows),
                missing_delayed_range_frames=sum(r["estimate_evidence"] is not None and
                    r["estimate_evidence"]["valid_delayed_beams"] == 0 for r in rows),
                cold_scored_frames=sum(bool(r.get("score")) for r in rows),
                cold_vs_archived_past_toe_disagreements=sum(r.get("score") is not None and
                    not r["score"]["cold_archived_past_toe_equal"] for r in rows),
                support_true_positive=support_counts[True, True], support_true_negative=support_counts[False, False],
                support_false_positive=support_counts[True, False], support_false_negative=support_counts[False, True],
                normal_angle_mean_deg=float(np.mean(angles)) if angles else None,
                scored_wheel_normals=len(angles), body_vx_rmse_m_s=float(np.sqrt(np.mean(np.square(errors)))) if errors else None)


def _summary(rows, spec):
    residuals = [r["enabled"]["violations"]["max"] for r in rows if r["enabled"].get("violations") is not None]
    ids = ("seed", "parent_seed", "phase", "source_row", "episode_tick")
    elapsed = np.asarray([r["elapsed_ms"] for r in rows])
    failures = [{**{k: r[k] for k in ids}, "reason": r["enabled"]["reason"]}
                for r in rows if not r["enabled"]["accepted"]]
    over = [{**{k: r[k] for k in ids}, "elapsed_ms": r["elapsed_ms"]}
            for r in rows if r["elapsed_ms"] > spec["deadline_ms"]]
    n, accepted = len(rows), len(rows)-len(failures)
    off_ok = sum(r["bypass_bitwise_equal"] for r in rows)
    worst = max(residuals, default=None)
    return dict(attempted=n, solver_attempted=sum(r["solver_attempted"] for r in rows), accepted=accepted,
                bypass_bitwise_equal=off_ok, max_hard_residual=worst, numerical_residual_rows=len(residuals),
                mechanical_pass=bool(n == spec["max_frames"] == accepted == off_ok == len(residuals)
                                     and worst is not None and worst <= spec["hard_tolerance"]),
                timing_pass=n == spec["max_frames"] and not over,
                latency_ms={key: float(np.percentile(elapsed, p)) for key, p in
                            (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))} if n else {},
                deadline_ms=spec["deadline_ms"], deadline_violations=over, failures=failures,
                solver_status_counts=dict(Counter(r["enabled"].get("solver_status") or "no_status" for r in rows)),
                diagnostic=_diagnostic_summary(rows))


def run(out: Path):
    out = Path(out)
    out.mkdir(exist_ok=False, parents=True)
    rows, score_inputs = [], []
    spec = json.loads(PLAN.read_bytes())["shadow"]
    report = {"status": "PARTIAL", "protocol_sha256": _hash(PLAN), "setup_ms": {},
              "measured_native_steps": 0, "native_steps": 0, "policy_updates": 0,
              "timed_boundary": "continuous estimator.update through enabled controller.correct return, "
                                "including nominal kinematics, dynamics, cold QP, independent residual audit; "
                                "policy inference, bypass, warmup, I/O and later truth scoring excluded",
              "truth_boundary": "Prior archived endpoint; native cold forward only, force >1 N. "
                                "Normal angles compared in the respective body frames to align arbitrary odometry yaw.",
              "support_gap_m": SUPPORT_GAP_M, "max_range_extension_m": MAX_EXTENSION_M,
              "max_lane_lateral_m": MAX_LATERAL_M, "sources": []}
    start = perf_counter_ns()
    try:
        from tools87.terrain87 import load_model
        from tools89.common89 import BASE, complete_stage, registered_plan
        from tools90.wbc90 import WBC90

        stage = complete_stage("training")
        plan89 = registered_plan()
        if _hash(BASE / "training_01/report.json") != spec["source_stage_sha256"]:
            raise ValueError("registered source stage changed")
        model = load_model("flat")
        controller = WBC90(model)
        report["setup_ms"]["imports_receipts_model_and_controller"] = _elapsed(start)
        with (out / "frames.jsonl").open("x") as frames:
            for source in spec["sources"]:
                start = perf_counter_ns()
                arrays, windows, predict, policy_sha = _source(source, spec, stage, plan89)
                source_record = dict(seed=source["seed"], parent_seed=source["parent"], parent_policy_sha256=policy_sha,
                                     trace_sha256=source["trace_sha256"], source_verify_unpack_and_predictor_ms=_elapsed(start))
                report["sources"].append(source_record)
                for phase, (first, indices) in windows.items():
                    start = perf_counter_ns()
                    estimator = SensorEstimator(model)
                    source_record.setdefault("estimator_setup_ms", {})[phase] = _elapsed(start)
                    selected = set(indices)
                    history_error = None
                    for i in range(first, indices[-1] + 1):
                        tick = i - first
                        prop, ranges = arrays["proprio627"][i], arrays["visible_range"][i]
                        if i not in selected:
                            if history_error is None:
                                try:
                                    estimator.update(tick, prop, ranges)
                                except Exception as exc:
                                    history_error = f"causal warmup row {i}: {type(exc).__name__}: {exc}"
                            continue
                        result, history_error = _test_frame(estimator, controller, predict, tick, prop, ranges, history_error)
                        row = dict(seed=source["seed"], parent_seed=source["parent"], phase=phase, source_row=i,
                                   episode_tick=tick, episode_start_row=first, proprio627_sha256=sha256(prop.tobytes()).hexdigest(),
                                   current_proprio57=prop[:57], visible_range=ranges, archived_issued=arrays["issued"][i],
                                   archived_applied=arrays["applied"][i], **result)
                        _line(frames, row)
                        rows.append(row)
                        # Archive qpos/qvel/torque are post-step. Match pre-issue i with post-step i-1 in this episode.
                        sample = None
                        if tick:
                            sample = {key: arrays[key][i-1] for key in
                                      ("qpos", "qvel", "last_torque_nm", "body_vx_m_s", "loaded_wheels_past_toe")}
                            sample.update(tick=tick, terrain="flat" if phase == "flat" else "up")
                        score_inputs.append(sample)
        start = perf_counter_ns()
        auditors = {terrain: TruthAudit(terrain) for terrain in ("flat", "up")}
        report["setup_ms"]["posthoc_truth_models"] = _elapsed(start)
        with (out / "truth.jsonl").open("x") as stream:
            for row, sample in zip(rows, score_inputs, strict=True):
                error, score = None, None
                if sample is not None:
                    try:
                        score = auditors[sample["terrain"]].score(sample, row["estimated_state"], row["estimate_evidence"])
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                else:
                    error = "reset has no preceding archived endpoint"
                row["score"] = score
                _line(stream, {**{k: row[k] for k in ("seed", "parent_seed", "source_row", "episode_tick", "phase")},
                               "score": score, "unavailable_reason": error})
        report.update(_summary(rows, spec), status="COMPLETE", frames_sha256=_hash(out / "frames.jsonl"),
                      truth_sha256=_hash(out / "truth.jsonl"),
                      phase_counts={phase: sum(r["phase"] == phase for r in rows) for phase in PHASES},
                      per_phase={phase: {"accepted": sum(r["enabled"]["accepted"] for r in rows if r["phase"] == phase),
                                         **_diagnostic_summary([r for r in rows if r["phase"] == phase])} for phase in PHASES})
        if report["attempted"] != spec["max_frames"]:
            raise RuntimeError("incomplete attempted panel")
    except BaseException as exc:
        report.update(status="PARTIAL", completed_frame_records=len(rows), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _write(out / "report.json", report)
    return report
