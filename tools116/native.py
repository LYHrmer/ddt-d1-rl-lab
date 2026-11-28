"""Run the round-111 compound route with encoder compensation after the lidar speed preview."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType

import numpy as np

from tools91.launch91 import require
from tools94.common94 import arrays, digest, write
from tools110.course import CFG, TICKS
from tools111.course import score
from tools116.control import PrevTrimActor
from tools116.score import reference


def with_preview_input(drive, actor):
    """Tap the original, unrounded preview at the frozen loop's frame-construction boundary."""
    frame = drive.__globals__["actor_frame"]

    def capture(plant, vx, previous_raw, gyro):
        actor.set_preview(vx)
        return frame(plant, vx, previous_raw, gyro)

    namespace = dict(drive.__globals__, actor_frame=capture)
    return FunctionType(drive.__code__, namespace, drive.__name__, drive.__defaults__, drive.__closure__)


def check_feedback(z):
    n = len(z["raw"])
    frame = z["actor_frame_command"]
    np.testing.assert_array_equal(frame[:,0],np.float32(2.*z["actor_vx"]))
    np.testing.assert_array_equal(z["actor_vx"],z["preview_actor_vx"]+z["velocity_trim"])
    np.testing.assert_array_equal(z["trim_previous_actor_vx"],np.r_[0.,z["actor_vx"][:-1]])
    np.testing.assert_array_equal(z["encoder_forward_m_s"],.087*np.median(z["encoder_wheel_rad_s"],axis=1))
    previous = np.vstack((np.zeros(16,np.float32), z["raw"][:-1]))
    history = np.maximum(0,np.arange(n)[:,None]+np.arange(-10,0))
    np.testing.assert_array_equal(z["actor_previous_raw"],previous)
    np.testing.assert_array_equal(z["actor_history_previous_raw"],previous[history])
    np.testing.assert_array_equal(z["actor_history_frame_command"],frame[history])
    # Only the original lidar leg FK is allowed between raw and issued in this arm.
    np.testing.assert_array_equal(z["issued"][:,3::4],z["raw"][:,3::4])
    np.testing.assert_array_equal(z["applied"],np.vstack((np.zeros(16),z["issued"][:-1])))
    np.testing.assert_array_equal(z["applied_lift_fk"],np.vstack((np.zeros(4),z["issued_lift_fk"][:-1])))


def matched_prefix(z, old):
    """Compare the closed loops up to the first effective change in the ONNX input."""
    n = len(z["raw"])
    preview_frame = np.float32(2.*z["preview_actor_vx"])
    changed = np.flatnonzero(z["actor_frame_command"][:,0] != preview_frame)
    first = int(changed[0]) if len(changed) else None
    unchanged = n if first is None else first
    # During the first changed decision the servo still applies the preceding action.
    applied = min(n,unchanged+1)
    measured = min(n,applied+1)
    for key in ("qpos","qvel","wheels_xy","measured_relative_z","front_extent_x"):
        np.testing.assert_array_equal(z[key][:applied+1],old[key][:applied+1],err_msg=key)
    for key in ("raw","issued","issued_lift_fk","actor_frame_command"):
        np.testing.assert_array_equal(z[key][:unchanged],old[key][:unchanged],err_msg=key)
    for key in ("applied","applied_lift_fk","first_torque_nm"):
        np.testing.assert_array_equal(z[key][:applied],old[key][:applied],err_msg=key)
    for key in ("base_cmd","actor_wz","imu_quat","encoder_wheel_rad_s","capture_used","coverage","lift_demand"):
        np.testing.assert_array_equal(z[key][:measured],old[key][:measured],err_msg=key)
    np.testing.assert_array_equal(z["preview_actor_vx"][:measured],old["actor_vx"][:measured])
    np.testing.assert_array_equal(z["actor_frame_command"][:measured,1:],old["actor_frame_command"][:measured,1:])
    packets = int(np.count_nonzero(z["capture_tick"] < measured))
    for key in ("capture_tick","range_m","range_valid","self_hit"):
        np.testing.assert_array_equal(z[key][:packets],old[key][:packets],err_msg=key)
    return dict(first_changed_actor_frame_tick=first, equal_issued_ticks=unchanged,
                equal_applied_ticks=applied, equal_state_endpoints=applied+1, equal_lidar_packets=packets)


def episode(out, seed):
    """Write one guarded, already reserved trial; return its measurements to the worker."""
    out = Path(out)
    if not out.is_dir():
        raise FileNotFoundError(f"reserved native directory is missing: {out}")
    if any((out/name).exists() for name in ("trace.npz","partial.npz","partial_counts.json")):
        raise FileExistsError(f"preserve the prior native attempt in {out}")
    rows = defaultdict(list)
    counts = dict(decisions=0,servo=0,physics=0,packets=0,ray_queries=0)
    actor, nominal_height = None, None
    try:
        path, old = reference(seed)
        from tools110.native import MODEL_SHA, YAML_SHA, Plant, load_policy, mj
        from tools111.course import install_heightfield, make_scene
        from tools111.native import drive
        from training.d1_rl.interface import WHEEL_INDICES

        layout, arm, gain = old["layout"], old["arm"], old["gain"]
        require(layout == "development" and arm == "both", "wrong paired compound course or lidar mode")
        session = load_policy()
        with TemporaryDirectory(prefix="d1-compound116-") as tmp:
            plant = Plant(config=CFG, scene=make_scene(tmp,layout), seed=seed, arm="B", eval_delay_ticks=1,
                          eval_command_sequence=[{"start_s":0.,"vx":0.,"wz":0.}], max_transitions=TICKS)
            try:
                install_heightfield(plant.model,layout)
                plant.reset(seed=seed)
                nominal_height = float(plant.nominal_base_height_m)
                with np.load(path,allow_pickle=False) as before:
                    np.testing.assert_array_equal(plant.data.qpos,before["qpos"][0])
                    np.testing.assert_array_equal(plant.data.qvel,before["qvel"][0])
                wheel_ids = np.asarray(plant.dof_ids)[WHEEL_INDICES]
                actor = PrevTrimActor(session, lambda: plant.data.qvel[wheel_ids], enabled=True)
                with_preview_input(drive,actor)(plant,actor,rows,counts,layout,arm,gain)
            finally:
                plant.close()
        trace = actor.trace(rows)
        trace["nominal_base_height_m"] = np.asarray(nominal_height)
        np.testing.assert_array_equal(trace["preview_actor_vx"],rows["actor_vx"])
        check_feedback(trace)
        with np.load(path,allow_pickle=False) as before:
            prefix = matched_prefix(trace,before)
        # The original effects formula subtracts its actor_vx argument from base_cmd.
        # Keep the two physical meanings explicit once the actor additionally receives trim.
        motion, net = score(trace,layout)
        _, lidar = score(dict(trace,actor_vx=trace["preview_actor_vx"]),layout)
        n = motion["ticks"]
        require(counts["decisions"] == n and counts["servo"] == 5*n and counts["physics"] == 10*n <= 6000
                and counts["packets"] == len(trace["capture_tick"])
                and counts["ray_queries"] == 1040*counts["packets"] <= 74880, "compound native clock mismatch")
        arrays(out/"trace.npz",trace)
        return dict(status="COMPLETE",seed=seed,layout=layout,arm=arm,gain=gain,trim="brake_aware_after_lidar",
                    counts=counts,motion=motion,effects=lidar,net_actor_effects=net,paired_prefix=prefix,
                    nominal_base_height_m=nominal_height,reference_sha256=old["trace_sha256"],
                    trace_sha256=digest(out/"trace.npz"),actor_sha256=MODEL_SHA,actuator_config_sha256=YAML_SHA,
                    mujoco_version=mj.__version__)
    except BaseException as exc:
        partial = actor.trace(rows) if actor is not None else {k:np.asarray(v) for k,v in rows.items()}
        if nominal_height is not None:
            partial["nominal_base_height_m"] = np.asarray(nominal_height)
        arrays(out/"partial.npz",partial)
        write(out/"partial_counts.json",dict(error=repr(exc),counts=counts))
        raise
