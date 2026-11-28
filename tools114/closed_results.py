"""Independently check the observed parking intervention and re-score the native traces."""
from pathlib import Path

import numpy as np

from tools91.launch91 import digest, read, require
from tools110.course import TICKS, command
from tools111.control import SPEED_GAIN
from tools111.kinematics import rotation, vendor_targets
from tools113.audit import _commands
from tools113.results import compare_tree
from tools114.common import BLIND, DEV, baseline
from tools114.replay import RESET, STOP_LIMIT, TAIL


WHEELS = slice(3, None, 4)


def _first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def parking_stop(qpos, qvel):
    """Use 26 velocity endpoints for the maximum and 25 displacements for signed travel."""
    if len(qpos) != TICKS + 1:
        return None
    rotations = rotation(qpos[TAIL:, 3:7])
    yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
    velocity = qvel[TAIL:, :2]
    speeds = np.linalg.norm(velocity, axis=1)
    peak = int(np.argmax(speeds))
    body_speed = np.sum(velocity * forward, axis=1)
    travel = np.sum(np.diff(qpos[TAIL:, :2], axis=0) * forward[:-1])
    return dict(peak_m_s=float(speeds[peak]), peak_tick=TAIL+peak,
                peak_world_vxy_m_s=velocity[peak].tolist(), peak_forward_m_s=float(body_speed[peak]),
                mean_forward_m_s=float(body_speed[:-1].mean()), forward_displacement_mm=float(1000.*travel),
                endpoints_above_limit=int((speeds > STOP_LIMIT).sum()))


def score_native(z, nominal_height):
    """Reconstruct the original 3–6 m and parking gates from their stored native samples."""
    q, v = z["qpos"], z["qvel"]
    n = len(q)-1
    require(1 <= n <= TICKS and q.shape == (n+1,23) and v.shape == (n+1,22)
            and np.isfinite(q).all() and np.isfinite(v).all(), "invalid native states")
    np.testing.assert_allclose(z["time"], np.arange(n+1)/50., rtol=0., atol=1e-9)
    np.testing.assert_array_equal(z["xy"],q[:,:2])
    np.testing.assert_array_equal(z["vxy"],v[:,:2])
    np.testing.assert_allclose(np.linalg.norm(q[:,3:7],axis=1), 1., rtol=0., atol=1e-9)
    wheels = z["wheels_xy"]
    require(wheels.shape == (n+1,4,2) and np.isfinite(wheels).all(), "invalid wheel positions")
    for name, shape in (("exit_contact",(n,4)), ("top_contact",(n,4)), ("fallen",(n,)), ("illegal",(n,))):
        require(z[name].shape == shape and z[name].dtype == np.bool_, "invalid native safety flags: "+name)
    require(.25 < nominal_height < .8, "invalid nominal body height")
    tilt = np.arccos(np.clip(1.-2.*(q[1:,4]**2 + q[1:,5]**2), -1., 1.))
    np.testing.assert_array_equal(z["fallen"],(q[1:,2] < nominal_height-.18) | (tilt > .6))
    terminated = z["fallen"] | z["illegal"]
    require(not terminated[:-1].any() and (n == TICKS or terminated[-1]), "unexplained native termination")

    gates = []
    for x in (3., 4., 5., 6.):
        k = _first((q[:-1,0] < x) & (q[1:,0] >= x))
        if k is None:
            break
        fraction = (x-q[k,0]) / (q[k+1,0]-q[k,0])
        t0, t1 = z["time"][k:k+2]
        gates.append(float(t0+fraction*(t1-t0)))
    intervals = np.diff(gates)
    speed, sections = None, []
    if len(intervals) == 3 and np.all(intervals > 0.):
        sections = (1./intervals).tolist()
        speed = 3./(gates[-1]-gates[0])
    corridor = (wheels[:,:,0] >= 2.) & (wheels[:,:,0] <= 7.)
    lane = float(np.abs(wheels[:,:,1][corridor]).max()) if corridor.any() else None
    exits = np.any(z["exit_contact"], axis=0)
    tops = np.any(z["top_contact"], axis=0)
    stop = parking_stop(q,v)
    checks = dict(horizon=n == TICKS, forward_speed=speed is not None and 1.50 <= speed <= 1.575,
                  one_metre_speeds=len(sections) == 3 and all(1.425 <= x <= 1.575 for x in sections),
                  lane=lane is not None and lane <= 1., all_wheels_exited=exits.all() and (wheels[-1,:,0]>7.).all(),
                  step_top_support=True, stopped=stop is not None and stop["peak_m_s"] <= STOP_LIMIT,
                  fall_free=not z["fallen"].any(), legal_support=not z["illegal"].any())
    failed = [key for key,value in checks.items() if not value]
    motion = dict(passed=not failed, failed=failed, ticks=n, gate_times_s=gates, speed_m_s=speed,
                  sections_m_s=sections,max_lane_y_m=lane,exit_contact=exits.tolist(),
                  step_top_contact=tops.tolist(), stop_tail_max_m_s=stop["peak_m_s"] if stop else None)
    return motion, stop


def check_targets(z):
    raw, issued, applied = (z[k] for k in ("raw","issued","applied"))
    n = len(raw)
    for action in (raw,issued,applied):
        require(action.shape == (n,16) and action.dtype == np.float32 and np.isfinite(action).all(),
                "invalid vendor action")
    ticks = np.arange(n)
    driver = np.array([command(k) for k in ticks])
    np.testing.assert_array_equal(z["driver_command_xyz"],np.column_stack((driver,np.zeros((n,2)))))
    active = ticks >= RESET
    np.testing.assert_array_equal(z["parking_active"],active)
    np.testing.assert_array_equal(z["lift_demand"],np.zeros((n,4)))
    np.testing.assert_array_equal(z["issued_lift_fk"],np.zeros((n,4)))
    np.testing.assert_array_equal(z["applied_lift_fk"],np.zeros((n,4)))
    np.testing.assert_array_equal(z["unprojected_issued"],raw)
    wheels = raw[:,WHEELS].astype(float)
    common = np.where(active,wheels.mean(axis=1),0.)
    expected = raw.copy()
    expected[:,WHEELS] = wheels - common[:,None]
    np.testing.assert_array_equal(z["wheel_common_removed_rad_s"],5.*common)
    np.testing.assert_array_equal(issued,expected)
    zeros = np.zeros((1,16),np.float32)
    previous = np.concatenate((zeros,raw[:-1]))
    np.testing.assert_array_equal(applied,np.concatenate((zeros,issued[:-1])))
    np.testing.assert_array_equal(z["actor_previous_raw"],previous)
    history = np.maximum(ticks[:,None] + np.arange(-10,0),0)
    np.testing.assert_array_equal(z["actor_history_previous_raw"],previous[history])
    first_issue = _first(np.any(raw != issued,axis=1))
    first_applied = _first(np.any(previous != applied,axis=1))
    require(first_applied == (None if first_issue is None or first_issue+1 == n else first_issue+1),
            "projection reached the servo at the wrong tick")
    physical = 5.*issued[:,WHEELS].astype(float)
    residual = float(np.max(np.abs(physical[active].mean(axis=1)))) if active.any() else None
    difference = physical-5.*wheels
    differential_error = float(np.ptp(difference,axis=1).max())
    return dict(first_active_tick=_first(active), first_changed_issue_tick=first_issue,
                first_changed_applied_tick=first_applied, active_intervals=int(active.sum()),
                policy_history_windows_checked=n, fifo_intervals_checked=n,
                max_active_common_abs_rad_s=residual,max_pairwise_wheel_error_rad_s=differential_error)


def _actuation_and_clock(z, row):
    n = len(z["raw"])
    captures = np.array([k for k in range(n) if not k or k*6//50 != (k-1)*6//50])
    np.testing.assert_array_equal(z["capture_tick"],captures)
    require(row["counts"] == dict(decisions=n,servo=n*5,physics=n*10,packets=len(captures),
                                  ray_queries=len(captures)*1040), "native budget or capture clock mismatch")
    require(n == len(z["time"])-1,"state and action clocks differ")
    encoders = z["qvel"][:-1,9::4]
    np.testing.assert_array_equal(z["encoder_wheel_rad_s"],encoders)
    measured, base, increments, actual = _commands(encoders)
    for name,expected in (("encoder_forward_m_s",measured),("base_cmd",base),("preview_actor_vx",base),
                          ("velocity_trim",increments),("actor_vx",actual)):
        np.testing.assert_allclose(z[name],expected,rtol=0.,atol=1e-12)
    # Heading-only preview. Onboard attitude gives the same heading as the native body endpoints.
    imu = z["imu_quat"]
    np.testing.assert_allclose(np.abs(np.sum(imu*z["qpos"][:-1,3:7],axis=1)),1.,rtol=0.,atol=1e-9)
    rot = rotation(imu)
    yaw = np.arctan2(rot[:,1,0],rot[:,0,0])
    np.testing.assert_allclose(z["imu_yaw"],yaw,rtol=0.,atol=1e-9)
    error = yaw[0]-yaw
    rate = np.where(z["driver_command_xyz"][:,0] > .01,
                    np.clip(2.5*np.arctan2(np.sin(error),np.cos(error)),-.35,.35),0.)
    np.testing.assert_allclose(z["actor_wz"],rate,rtol=0.,atol=1e-9)
    frame = np.column_stack((2.*actual,np.zeros(n),.25*z["actor_wz"])).astype(np.float32)
    np.testing.assert_array_equal(z["actor_frame_command"],frame)
    # The saved first torque is one of the five servo samples within each decision.
    q = z["qpos"][:-1,7:].copy()
    q[:,WHEELS] = 0.
    unconstrained = (vendor_targets(z["applied"])-q)*np.tile([100.,60.,60.,10.],4)
    unconstrained -= z["qvel"][:-1,6:]*np.tile([4.,1.5,1.5,.5],4)
    limit = np.tile([80.,80.,80.,12.],4)
    np.testing.assert_allclose(z["first_torque_nm"],np.clip(unconstrained,-limit,limit),rtol=0.,atol=2e-9)
    saturated = z["wheel_saturation"]
    require(saturated.shape == (n,4) and saturated.dtype == np.bool_
            and np.all(saturated | (abs(unconstrained[:,WHEELS]) < 12.)), "invalid wheel saturation flag")


def check_prefix(z, old, wiring):
    """State at the start of the first affected servo interval is still paired."""
    n = len(z["raw"])
    first_issue, first_apply = (wiring[k] for k in ("first_changed_issue_tick","first_changed_applied_tick"))
    applied = n if first_apply is None else first_apply
    observed_inputs = min(n,applied+1)
    for key in ("qpos","qvel"):
        np.testing.assert_array_equal(z[key][:applied+1],old[key][:applied+1],err_msg=key)
    for key in ("raw","actor_vx","actor_wz","actor_frame_command","encoder_wheel_rad_s","imu_quat"):
        np.testing.assert_array_equal(z[key][:observed_inputs],old[key][:observed_inputs],err_msg=key)
    issued = n if first_issue is None else first_issue
    np.testing.assert_array_equal(z["issued"][:issued],old["issued"][:issued])
    for key in ("applied","first_torque_nm"):
        np.testing.assert_array_equal(z[key][:applied],old[key][:applied],err_msg=key)
    return dict(equal_state_endpoints=applied+1,equal_raw_actor_outputs=observed_inputs,
                equal_issued_intervals=issued,equal_applied_intervals=applied)


def _missing(seed, directory, report):
    status = "PARTIAL" if directory.is_dir() and any(directory.iterdir()) else "NOT_STARTED"
    facts = {p.name:digest(p) for p in directory.glob("*") if p.is_file()}
    partial = directory / "partial_counts.json"
    issue = read(partial) if partial.is_file() else {}
    worker_file = directory / "worker_partial.json"
    worker = read(worker_file) if worker_file.is_file() else {}
    return dict(seed=seed,stage_status=status,complete=False,paired=seed in DEV,gate_pass=False,
                failed=[status.lower()],motion=None,stop=None,comparison=None,wiring=None,
                artifacts=facts,reported_status=report.get("status"),partial_counts=issue.get("counts"),
                error=issue.get("error",worker.get("error",report.get("error"))))


def assess(seed, directory):
    """Return one result from a sealed native stage. Incomplete workers cannot qualify."""
    require(seed in (*DEV,*BLIND),"unregistered native parking seed")
    directory = Path(directory)
    header = directory/"report.json"
    report = read(header) if header.is_file() else {}
    if report.get("status") != "COMPLETE" or report.get("result",{}).get("status") != "COMPLETE":
        return _missing(seed,directory,report)
    path = directory/"trace.npz"
    row = report["result"]
    ref_path, ref_row = baseline(seed if seed in DEV else DEV[0])
    paired = seed in DEV
    require(report["stage"] == f"native_{seed}" and report["files"] == {"trace.npz":row["trace_sha256"]}
            and report.get("baseline_sha256") == row["reference_sha256"]
            and row["seed"] == seed and row["cohort"] == ("development" if paired else "blind")
            and row["layout"] == "flat" and row["surface"] == "level_heightfield" and row["gain"] == SPEED_GAIN
            and row["arm"] == "heading_on" and row["trim"] == "brake_aware"
            and row["projection"] == "parked_wheel_common_zero" and row["effects"] == {}
            and row["reference_sha256"] == (ref_row["trace_sha256"] if paired else None)
            and all(row[k] == ref_row[k] for k in ("actor_sha256","actuator_config_sha256","mujoco_version"))
            and row["trace_sha256"] == digest(path), "incorrect contract or native source binding")
    require(not any((directory/name).exists() for name in ("partial.npz","partial_counts.json","worker_partial.json")),
            "complete stage cannot include a partial trajectory")
    with np.load(path,allow_pickle=False) as trace:
        z = dict(trace)
    motion, stop = score_native(z,row["nominal_base_height_m"])
    compare_tree(motion,row["motion"])
    _actuation_and_clock(z,row)
    wiring = check_targets(z)
    comparison = None
    if paired:
        with np.load(ref_path,allow_pickle=False) as reference:
            old = dict(reference)
        prefix = check_prefix(z,old,wiring)
        off, old_stop = score_native(old,row["nominal_base_height_m"])
        compare_tree(off,ref_row["motion"])
        comparison = dict(reference_sha256=ref_row["trace_sha256"],prefix=prefix,
                          reference_motion=off,reference_stop=old_stop,
                          delta_cruise_m_s=None if motion["speed_m_s"] is None else motion["speed_m_s"]-off["speed_m_s"],
                          delta_stop_peak_m_s=None if stop is None else stop["peak_m_s"]-old_stop["peak_m_s"],
                          delta_forward_displacement_mm=None if stop is None else
                          stop["forward_displacement_mm"]-old_stop["forward_displacement_mm"])
    complete = motion["ticks"] == TICKS
    return dict(seed=seed,stage_status="COMPLETE",complete=complete,paired=paired,
                gate_pass=complete and motion["passed"],failed=motion["failed"],motion=motion,stop=stop,
                wiring=wiring,comparison=comparison, counts=row["counts"],trace_sha256=row["trace_sha256"])


def development_passed(rows):
    """Release fresh starts only after all three distinct registered development cases qualify."""
    return (len(rows) == len(DEV) and {r["seed"] for r in rows} == set(DEV)
            and all(r["complete"] and r["gate_pass"] and r["paired"] for r in rows))


def verify_public(directory, seed, expected):
    """Recalculate one public result using NumPy and the sealed reference, without physics."""
    row = assess(seed,directory)
    compare_tree(row,expected)
    return row
