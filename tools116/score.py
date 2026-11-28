"""Recompute the matched compound-course measurements from the native trajectories."""
import math
from pathlib import Path

import numpy as np

from tools91.launch91 import ROOT, digest, read, require
from tools110.course import TICKS, command
from tools111.control import SPEED_GAIN
from tools111.course import SEEDS as COURSE_SEEDS, ground, score, step_bounds
from tools111.replay import check_episode
from tools114.closed_results import parking_stop


SEEDS = COURSE_SEEDS["development"]
INDEX = ROOT / "reports/lidar_compound111/native.json"
INDEX_SHA = "f0c78571880588577622f6130225ebbb59230dbd25dea9faa71bc0c0cacd3ea3"


def _source(seed):
    require(seed in SEEDS and digest(INDEX) == INDEX_SHA, "unregistered compound-route reference")
    case, = [c for c in read(INDEX)["cases"] if c["seed"] == seed and c["arm"] == "both" and c["layout"] == "development"]
    row = case["episode"]
    path = INDEX.parent / case["file"]
    parent = "INCOMPLETE" if seed == 111101 else "COMPLETE"
    require(case["case_status"] == row["status"] == "COMPLETE" and case["unit_status"] == parent
            and row["seed"] == seed and row["layout"] == "development" and row["arm"] == "both"
            and row["gain"] == SPEED_GAIN and row["motion"]["ticks"] == row["counts"]["decisions"] == TICKS
            and row["trace_sha256"] == digest(path), "unbound or incomplete individual round-111 case")
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["time"], np.arange(TICKS+1)/50., atol=1e-9, rtol=0.)
    return path, case


def reference(seed):
    """Return the completed individual episode, including 111101 in its interrupted multi-case unit."""
    path, case = _source(seed)
    return path, case["episode"]


def _first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def first_failed_section(motion):
    """Report the first original one-metre interval outside its band or never reached."""
    times = motion["gate_times_s"]
    for i in range(3):
        speed = 1./(times[i+1]-times[i]) if len(times) > i+1 and times[i+1] > times[i] else None
        if speed is None or not 1.425 <= speed <= 1.575:
            reason = "not_measured" if speed is None else ("too_slow" if speed < 1.425 else "too_fast")
            return dict(from_x_m=3+i, to_x_m=4+i, speed_m_s=speed, reason=reason)
    return None


def lidar_effects(z, increment):
    """Compare delivered actor inputs and delayed FK lift targets strictly before the first edge/contact."""
    edge, _ = step_bounds("development")
    extent = _first(z["front_extent_x"][1:] >= edge)
    touch = _first(z["front_platform_touch"])
    reached = [k for k in (extent, touch) if k is not None]
    end = min(reached) if reached else None
    start = max(0, end-25) if reached else None
    window = np.arange(start, end) if reached else np.array([], dtype=int)
    packet = z["capture_used"]
    delivered = (packet >= 0) & (packet + 2 <= np.arange(len(packet)))
    speed_ticks = window[delivered[window]]
    lifts = window[window > 0]
    lift_ticks = lifts[delivered[lifts-1]]  # This interval's servo uses the preceding issued lift.
    # The identical trim is held fixed at each measured tick. Cast both sides at
    # the actual 57-element float32 actor input before converting to metres/second.
    masked_frame = (2.*(z["base_cmd"] + increment)).astype(np.float32)
    delta = .5 * (masked_frame.astype(float) - z["actor_frame_command"][:, 0].astype(float))
    speed_tick = int(speed_ticks[np.argmax(delta[speed_ticks])]) if len(speed_ticks) else None
    speed = float(delta[speed_tick]) if speed_tick is not None else None
    lift_tick = side = height = None
    if len(lift_ticks):
        i, side = np.unravel_index(int(np.argmax(z["applied_lift_fk"][lift_ticks, :2])), (len(lift_ticks), 2))
        lift_tick, side = int(lift_ticks[i]), int(side)
        height = float(z["applied_lift_fk"][lift_tick, side])
    speed_pass = speed is not None and speed >= .020
    lift_pass = height is not None and height >= .005
    measured = None
    if len(window):
        measured = (z["measured_relative_z"][end,:2]-z["measured_relative_z"][start,:2]).tolist()
    return dict(first_front_extent_tick=extent, first_loaded_platform_touch_tick=touch,
                window_start_tick=start, end_tick_exclusive=end, speed_valid_ticks=len(speed_ticks),
                lift_valid_ticks=len(lift_ticks), speed_tick=speed_tick,
                speed_capture_tick=int(packet[speed_tick]) if speed_tick is not None else None,
                speed_reduction_m_s=speed, speed_active=speed_pass,
                lift_servo_tick=lift_tick, lift_issue_tick=lift_tick-1 if lift_tick is not None else None,
                lift_capture_tick=int(packet[lift_tick-1]) if lift_tick is not None else None,
                front_wheel=("FL","FR")[side] if side is not None else None, front_lift_m=height, lift_active=lift_pass,
                measured_front_relative_z_change_m=measured, passed=speed_pass and lift_pass)


def _observe(z):
    n = len(z["time"])-1
    captures = [k for k in range(n) if k == 0 or (6*k)//50 != (6*(k-1))//50]
    counts = dict(decisions=n, servo=5*n, physics=10*n, packets=len(captures), ray_queries=1040*len(captures))
    q, v = z["qpos"], z["qvel"]
    require(1 <= n <= TICKS and q.shape==(n+1,23) and v.shape==(n+1,22)
            and np.isfinite(q).all() and np.isfinite(v).all(), "invalid native endpoints")
    np.testing.assert_array_equal(q[:,:2], z["xy"])
    np.testing.assert_array_equal(v[:,:2], z["vxy"])
    np.testing.assert_allclose(np.linalg.norm(q[:,3:7],axis=1),1.,atol=1e-9,rtol=0.)
    for key,shape in (("fallen",(n,)),("illegal",(n,)),("front_platform_touch",(n,)),
                      ("exit_contact",(n,4)),("top_contact",(n,4))):
        require(z[key].dtype == bool and z[key].shape == shape,"invalid native flag "+key)
    ended = z["fallen"] | z["illegal"]
    require(not ended[:-1].any() and (n == TICKS or ended[-1]), "unexplained early or ignored terminal")
    for key, shape in (("raw",(n,16)),("issued",(n,16)),("applied",(n,16)),("first_torque_nm",(n,16)),
                       ("issued_lift_fk",(n,4)),("applied_lift_fk",(n,4)),("front_extent_x",(n+1,)),
                       ("measured_relative_z",(n+1,4)),("actor_frame_command",(n,3))):
        require(z[key].shape == shape and np.isfinite(z[key]).all(), "invalid native series "+key)
    return counts


def _trim(z):
    """Independent state trace of the round-113 trim, feeding back actual radar-shaped commands."""
    preview = z["preview_actor_vx"]
    n = len(z["raw"])
    for key in ("preview_actor_vx","actor_vx","velocity_trim","encoder_forward_m_s","trim_previous_actor_vx"):
        require(z[key].shape == (n,) and np.isfinite(z[key]).all(), "invalid compensation series "+key)
    requests = np.array([command(k) for k in range(n)])
    measured = .087 * np.median(z["encoder_wheel_rad_s"], axis=1)
    bias = cruise = last = inc = 0.
    held = 0
    weight = -math.expm1(-.02/.4)
    increments = np.empty(n)
    previous = np.empty(n)
    for k, cmd in enumerate(requests):
        previous[k] = last
        held = held + 1 if k and cmd == requests[k-1] else 0
        if held >= 10:
            bias += weight * (last-measured[k]-bias)
            aim = np.clip((1.025-SPEED_GAIN)*cmd+bias, 0., .18)
            cruise += np.clip(aim-cruise, -.0036, .0036)
        limit = .18*np.clip(cmd/1.5, 0., 1.) + max(0., -measured[k])
        inc += np.clip(min(cruise,limit)-inc, -.0036, .0036)
        increments[k] = inc
        last = preview[k] + inc
    for got, expected in ((z["velocity_trim"],increments), (z["actor_vx"],preview+increments),
                          (z["encoder_forward_m_s"],measured), (z["trim_previous_actor_vx"],previous)):
        np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0.)
    frame = z["actor_frame_command"]
    require(frame.dtype == np.float32, "invalid vendor command precision")
    np.testing.assert_array_equal(frame[:,0], (2.*z["actor_vx"]).astype(np.float32))
    feedback = np.vstack((np.zeros(16, dtype=np.float32), z["raw"][:-1]))
    history = np.maximum(np.arange(n)[:,None] + np.arange(-10,0), 0)
    np.testing.assert_array_equal(z["actor_previous_raw"],feedback)
    np.testing.assert_array_equal(z["actor_history_previous_raw"],feedback[history])
    np.testing.assert_array_equal(z["actor_history_frame_command"],frame[history])
    return increments


def _prefix(z, old):
    n = len(z["raw"])
    first_input = _first(z["actor_frame_command"][:,0] != (2.*z["preview_actor_vx"]).astype(np.float32))
    first_issue = _first(np.any(z["issued"] != old["issued"][:n], axis=1))
    first_applied = _first(np.any(z["applied"] != old["applied"][:n], axis=1))
    require(first_applied == (None if first_issue is None or first_issue+1 == n else first_issue+1)
            and (first_issue is None or (first_input is not None and first_issue >= first_input)),
            "an actuator or paired state changed before the admitted input could act")
    until = n if first_applied is None else first_applied
    for k in ("qpos","qvel","xy","vxy","wheels_xy","measured_relative_z","front_extent_x"):
        np.testing.assert_array_equal(z[k][:until+1],old[k][:until+1],err_msg=k)
    for k in ("applied","first_torque_nm","applied_lift_fk","fallen","illegal","top_contact","exit_contact"):
        np.testing.assert_array_equal(z[k][:until],old[k][:until],err_msg=k)
    inputs_end = min(until+1, n)
    for k in ("imu_quat","encoder_wheel_rad_s","wheel_query_xy","actor_wz","base_cmd","lift_demand","capture_used"):
        np.testing.assert_array_equal(z[k][:inputs_end],old[k][:inputs_end],err_msg=k)
    np.testing.assert_array_equal(z["preview_actor_vx"][:inputs_end],old["actor_vx"][:inputs_end])
    packets = int((z["capture_tick"] < inputs_end).sum())
    for k in ("capture_tick", "range_m", "range_valid", "self_hit"):
        np.testing.assert_array_equal(z[k][:packets],old[k][:packets],err_msg=k)
    before = n if first_input is None else first_input
    for k in ("actor_frame_command", "raw", "issued"):
        np.testing.assert_array_equal(z[k][:before],old[k][:before],err_msg=k)
    return dict(first_changed_input_tick=first_input, first_changed_issue_tick=first_issue,
                first_changed_applied_tick=first_applied, matched_state_endpoints=until+1,
                matched_first_servo_torques=until, matched_raw_before_input=before,
                candidate_history_windows_checked=n, baseline_actor_history_recorded=False)


def score_candidate(path, seed):
    path = Path(path)
    require(path.name == "trace.npz" and not any((path.parent/f).exists() for f in
            ("partial.npz","partial_counts.json","worker_partial.json","partial.json")), "partial native trajectory")
    source, case = _source(seed)
    row = case["episode"]
    with np.load(path,allow_pickle=False) as archive, np.load(source,allow_pickle=False) as baseline:
        z, old = dict(archive), dict(baseline)
    old_counts = _observe(old)
    require(row["counts"] == old_counts, "incorrect old native clock")
    reference_motion, reference_effects = check_episode(old,row)
    reference_lidar = lidar_effects(old, np.zeros(len(old["raw"])))
    counts = _observe(z)
    inc = _trim(z)
    height = float(z["nominal_base_height_m"])
    q = z["qpos"][1:]
    require(.25 < height < .8, "invalid nominal root height")
    tilt = np.arccos(np.clip(1.-2.*(q[:,4]**2+q[:,5]**2), -1., 1.))
    np.testing.assert_array_equal(z["fallen"], (q[:,2]-ground("development",q[:,0],q[:,1]) < height-.18) | (tilt>.6))
    frame = z["actor_frame_command"].copy()
    frame[:,0] = (2.*z["preview_actor_vx"]).astype(np.float32)
    preview = dict(z,actor_vx=z["preview_actor_vx"],actor_frame_command=frame)
    motion, effects = score(preview,"development")
    _, net_effects = score(z,"development")
    # Replay first returns, 40 ms packet latency, odometry and delayed PD targets
    # with the original checker. Only its network-wide VX field refers to the pre-trim input.
    check_episode(preview,dict(layout="development",arm="both",gain=SPEED_GAIN,counts=counts,motion=motion,effects=effects))
    lidar = lidar_effects(z,inc)
    wiring = _prefix(z,old)
    return dict(seed=seed, trace_sha256=digest(path), reference_sha256=row["trace_sha256"],
                reference_case_status=case["case_status"], reference_parent_unit=case["unit"],
                reference_parent_unit_status=case["unit_status"], complete_horizon=motion["ticks"] == TICKS,
                gate_pass=motion["passed"] and lidar["passed"], motion_pass=motion["passed"], lidar_pass=lidar["passed"],
                motion=motion, effects=effects, net_actor_effects=net_effects, lidar=lidar, counts=counts, wiring=wiring,
                stop=parking_stop(z["qpos"],z["qvel"]), first_failed_section=first_failed_section(motion),
                reference_motion=reference_motion, reference_effects=reference_effects, reference_lidar=reference_lidar,
                reference_stop=parking_stop(old["qpos"],old["qvel"]),
                reference_first_failed_section=first_failed_section(reference_motion))
