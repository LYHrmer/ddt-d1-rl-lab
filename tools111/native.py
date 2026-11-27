"""Run a single 50 Hz whole-body actor with a delayed sensor on the registered native course."""
from tempfile import TemporaryDirectory

import mujoco as mj
import numpy as np

from tools87.terrain87 import capture_due
from tools94.common94 import arrays, digest, write
from tools110.course import CFG, TICKS, command
from tools110.native import Plant, actor_frame, torque
from tools111.control import Preview, WheelOdometry, body_to_level, yaw_from_quat
from tools111.course import ground, install_heightfield, make_scene, score, step_bounds
from tools111.sensing import LegGeometry, Lidar
from training.d1_rl.interface import WHEEL_INDICES


def contacts(plant, wheel_ids, platform, bounds):
    exit_load, top_load = np.zeros(4, bool), np.zeros(4, bool)
    front_touch = False
    force = np.empty(6)
    for i, c in enumerate(plant.data.contact[:plant.data.ncon]):
        a, b = int(c.geom1), int(c.geom2)
        if a in wheel_ids:
            wheel, road = wheel_ids[a], b
        elif b in wheel_ids:
            wheel, road = wheel_ids[b], a
        else:
            continue
        if plant.model.geom_bodyid[road] != 0:
            continue
        mj.mj_contactForce(plant.model, plant.data, i, force)
        if force[0] <= 1.:
            continue
        x, y, z = c.pos
        exit_load[wheel] |= x > 7. and abs(y) <= 1.
        if road == platform:
            front_touch |= wheel < 2
            top_load[wheel] |= (bounds[0]+.002 < x < bounds[1]-.002 and abs(y) <= 1.
                                and abs(z-.03) < .002 and abs(c.frame[2]) > .95)
    return exit_load, top_load, front_touch


def snapshot(rows, plant, wheels):
    data, model = plant.data, plant.model
    rows["time"].append(float(data.time))
    rows["qpos"].append(data.qpos.copy())
    rows["qvel"].append(data.qvel.copy())
    rows["xy"].append(data.qpos[:2].copy())
    rows["vxy"].append(data.qvel[:2].copy())
    centers = data.geom_xpos[wheels].copy()
    rows["wheels_xy"].append(centers[:, :2].copy())
    base = model.body("base_link").id
    relative = (centers-data.xpos[base]) @ data.xmat[base].reshape(3, 3)
    rows["measured_relative_z"].append(relative[:, 2].copy())
    coeff = data.geom_xmat[wheels[:2], :3]
    radius, half = model.geom_size[wheels[:2], :2].T
    extent = radius*np.hypot(coeff[:, 0], coeff[:, 1])+half*np.abs(coeff[:, 2])+model.geom_margin[wheels[:2]]
    rows["front_extent_x"].append(float(np.max(centers[:2, 0]+extent)))


def drive(plant, session, rows, counts, layout, arm, gain):
    model, data = plant.model, plant.data
    geometry, lidar = LegGeometry(model), Lidar(model)
    wheels = geometry.wheels
    mapping = {int(geom): i for i, geom in enumerate(wheels)}
    bounds = step_bounds(layout) if layout != "flat" else (0., 0.)
    box = model.geom("platform111").id if layout != "flat" else -1
    reference = yaw_from_quat(plant._sensor("trunk_quat"))
    odom, preview = WheelOdometry(reference), Preview(arm, gain)
    pending = history = None
    gyro = np.zeros(3)
    previous_raw = np.zeros(16, np.float32)
    previous_issued = previous_raw.copy()
    previous_lift = np.zeros(4)
    snapshot(rows, plant, wheels)
    for k in range(TICKS):
        quat = plant._sensor("trunk_quat")
        q, dq = data.qpos[plant.qpos_ids], data.qvel[plant.dof_ids]
        odom.update(yaw_from_quat(quat), dq[WHEEL_INDICES])
        wheel_xy = (geometry.centers(q) @ body_to_level(quat))[:, :2]
        if capture_due(k):
            pending, distances, valid, shadow = lidar.capture(k, data, quat, odom)
            counts["packets"] += 1
            counts["ray_queries"] = lidar.ray_calls
            for key, value in (("capture_tick", k), ("range_m", distances), ("range_valid", valid), ("self_hit", shadow)):
                rows[key].append(value)
        if pending is not None and k == pending.tick+2:
            preview.observe(k, pending, odom, wheel_xy)
            pending = None
        cmd = command(k)
        vx, wz, lift = preview.targets(k, cmd, reference, odom, wheel_xy)
        frame, gyro = actor_frame(plant, vx, previous_raw, gyro)
        frame[8] = .25*wz
        if history is None:
            history = np.tile(frame, (10, 1))
        raw = session.run(["nn_output"], {"nn_input0": frame[None, :], "nn_input1": history[None, :, :]})[0][0]
        counts["decisions"] += 1
        if raw.shape != (16,) or not np.isfinite(raw).all():
            raise ValueError("invalid manufacturer actor output")
        issued, issued_fk = geometry.raised_targets(raw, lift)
        history[:-1], history[-1] = history[1:].copy(), frame
        exit_load, top_load, saturation = (np.zeros(4, bool) for _ in range(3))
        illegal = front = False
        for sub in range(10):
            if sub % 2 == 0:
                tau, sat = torque(previous_issued, data.qpos[plant.qpos_ids], data.qvel[plant.dof_ids])
                data.qfrc_applied[:] = 0.
                data.qfrc_applied[plant.dof_ids] = tau
                data.ctrl[:] = 0.
                if sub == 0:
                    first_torque = data.qfrc_applied[plant.dof_ids].copy()
                counts["servo"] += 1
                saturation |= sat[WHEEL_INDICES]
            mj.mj_step(model, data)
            counts["physics"] += 1
            plant._check_finite_state()
            illegal |= plant._contact_diagnostics()[0]
            after, above, touched = contacts(plant, mapping, box, bounds)
            exit_load |= after
            top_load |= above
            front |= touched
        mj.mj_forward(model, data)
        plant._check_finite_state()
        illegal |= plant._contact_diagnostics()[0]
        snapshot(rows, plant, wheels)
        _, qx, qy, _ = data.qpos[3:7]
        clearance = data.qpos[2]-ground(layout, *data.qpos[:2])
        fallen = (clearance < plant.nominal_base_height_m-.18
                  or np.arccos(np.clip(1.-2.*(qx*qx+qy*qy), -1., 1.)) > .6)
        values = dict(base_cmd=gain*cmd, actor_vx=vx, actor_wz=wz, actor_frame_command=frame[6:9].copy(),
                      imu_yaw=odom.yaw, wheel_query_xy=wheel_xy,
                      imu_quat=quat.copy(), encoder_wheel_rad_s=dq[WHEEL_INDICES].copy(), first_torque_nm=first_torque,
                      odom_travel=odom.travel, capture_used=preview.last_packet, coverage=preview.coverage.copy(),
                      lift_demand=lift, raw=raw, issued=issued, applied=previous_issued, issued_lift_fk=issued_fk,
                      applied_lift_fk=previous_lift, wheel_saturation=saturation, exit_contact=exit_load,
                      top_contact=top_load, front_platform_touch=front, fallen=fallen, illegal=illegal)
        for key, value in values.items():
            rows[key].append(value)
        previous_raw, previous_issued, previous_lift = raw, issued, issued_fk
        if fallen or illegal:
            break


def episode(out, seed, layout, arm, gain, session):
    out.mkdir(exist_ok=False)
    settings = dict(seed=seed, layout=layout, arm=arm, gain=gain)
    write(out / "reservation.json", settings)
    keys = ("time xy vxy qpos qvel wheels_xy measured_relative_z front_extent_x capture_tick range_m range_valid self_hit first_torque_nm "
            "base_cmd actor_vx actor_wz actor_frame_command imu_yaw wheel_query_xy imu_quat encoder_wheel_rad_s odom_travel capture_used coverage lift_demand raw issued applied "
            "issued_lift_fk applied_lift_fk wheel_saturation exit_contact top_contact front_platform_touch fallen illegal")
    rows = {k: [] for k in keys.split()}
    counts = dict(decisions=0, servo=0, physics=0, packets=0, ray_queries=0)
    try:
        with TemporaryDirectory(prefix="d1-compound111-") as temp:
            plant = Plant(config=CFG, scene=make_scene(temp, layout), seed=seed, arm="B", eval_delay_ticks=1,
                          eval_command_sequence=[{"start_s": 0., "vx": 0., "wz": 0.}], max_transitions=TICKS)
            try:
                install_heightfield(plant.model, layout)
                plant.reset(seed=seed)
                drive(plant, session, rows, counts, layout, arm, gain)
            finally:
                plant.close()
        trace = {k: np.asarray(v) for k, v in rows.items()}
        motion, effects = score(trace, layout)
        n = motion["ticks"]
        assert counts["decisions"] == n and counts["servo"] == 5*n and counts["physics"] == 10*n
        assert counts["ray_queries"] == 1040*counts["packets"] <= 74880
        arrays(out / "trace.npz", trace)
        result = dict(status="COMPLETE", **settings, counts=counts, motion=motion, effects=effects,
                      trace_sha256=digest(out / "trace.npz"))
        write(out / "report.json", result)
        return result
    except BaseException as error:
        arrays(out / "partial.npz", {k: np.asarray(v) for k, v in rows.items()})
        write(out / "partial.json", dict(status="PARTIAL", error=repr(error), counts=counts))
        raise
