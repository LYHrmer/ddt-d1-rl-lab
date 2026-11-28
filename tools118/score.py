"""Score the frozen R117 controller at independent resets from its archived sensor and state records."""
import hashlib
import math
from pathlib import Path

import numpy as np

from tools91.launch91 import ROOT, digest, read, require
from tools110.course import CFG, TICKS
from tools111.control import SPEED_GAIN
from tools111.course import ground, score
from tools111.kinematics import rotation, vendor_targets
from tools111.replay import check_episode
from tools114.closed_results import parking_stop
from tools116.score import _observe, first_failed_section, lidar_effects
from tools116.wiring import ArchiveGeometry, _check_leg_targets
from tools117.score import _complete_file, _projected_wheels, _speed_signals


def _wheel_bottom(joints):
    """Lowest point of the four oriented wheel cylinders relative to the floating base."""
    chain = ArchiveGeometry()
    centers = chain.forward(joints)[0][0]
    orientation = np.broadcast_to(np.eye(3), (4, 3, 3))
    for j, angle in enumerate(joints.reshape(4, 4).T):
        half = .5 * angle
        hinge = np.column_stack((np.cos(half), np.zeros(4), np.zeros(4), np.sin(half)))
        orientation = orientation @ chain.rotations[:, j] @ rotation(hinge)
    # Radius and half-length of each collision cylinder in the audited R111 geometry.
    vertical_extent = .087 * np.linalg.norm(orientation[:, 2, :2], axis=1) + .020 * abs(orientation[:, 2, 2])
    return float(np.min(centers[:, 2] - vertical_extent))


def verify_reset(seed, qpos0, qvel0, nominal_base_height_m):
    """Reconstruct all 45 entries from the registered rank-0, episode-0 reset without loading MuJoCo."""
    require(isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) and seed >= 0,
            "expected a nonnegative integer seed")
    q, v = np.asarray(qpos0), np.asarray(qvel0)
    require(q.shape == (23,) and v.shape == (22,), "incorrect initial state dimensions")
    config = read(CFG)
    reset = config["reset"]
    schema = ROOT / config["interface_schema"]
    require(digest(schema) == config["interface_schema_sha256"], "changed reset interface")
    interface = read(schema)
    default_joints = np.array(interface["default_joint_angles"])
    legs = [i for i in range(16) if i not in interface["wheel_indices"]]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0, 0, 0]))
    noise = np.zeros(16)
    noise[legs] = rng.uniform(*reset["leg_joint_perturbation_uniform_rad"], size=len(legs))
    yaw = float(rng.uniform(*reset["yaw_perturbation_uniform_rad"]))
    pose = vendor_targets(np.zeros(16))
    # Preserve the floating-point order in D1Env.reset and the R110 Plant's vendor posture offset.
    joints = (default_joints + noise) + (pose - default_joints)
    expected_height = reset["wheel_ground_clearance_m"] - _wheel_bottom(joints)
    np.testing.assert_array_equal(q[:2], np.zeros(2))
    np.testing.assert_array_equal(q[3:7], [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
    np.testing.assert_array_equal(q[7:], joints)
    np.testing.assert_array_equal(v, np.zeros(22))
    # FK matrix products and MuJoCo's quaternion path can differ at double-precision roundoff.
    np.testing.assert_allclose(q[2], expected_height, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(nominal_base_height_m, -_wheel_bottom(pose), atol=1e-12, rtol=0.)
    fingerprint = hashlib.sha256(np.concatenate((q, v)).astype("<f8").tobytes()).hexdigest()
    return dict(seed=int(seed), rank=0, episode=0, stream=0, sampled_yaw_rad=yaw,
                initial_base_height_m=float(q[2]), height_reconstruction_error_m=float(abs(q[2] - expected_height)),
                fingerprint_sha256=fingerprint)


def score_trace(path, seed):
    """Read the original terrain and stop gates from one trace; a documented early terminal fails the horizon."""
    path = Path(path)
    _complete_file(path)
    with np.load(path, allow_pickle=False) as record:
        z = dict(record)
    counts = _observe(z)
    height = float(z["nominal_base_height_m"])
    reset = verify_reset(seed, z["qpos"][0], z["qvel"][0], height)
    sha = digest(path)
    receipt = path.parent / "report.json"
    if receipt.exists():
        header = read(receipt)
        result = header.get("result", {})
        require(header.get("status") == result.get("status") == "COMPLETE" and
                result.get("seed") == seed and result.get("trace_sha256") == sha, "unbound native receipt")

    increment, first_release = _speed_signals(z)
    first_projection = _projected_wheels(z, True)
    pose = z["qpos"][1:]
    tilt = np.arccos(np.clip(1.-2.*(pose[:, 4]**2 + pose[:, 5]**2), -1., 1.))
    clearance = pose[:, 2] - ground("development", pose[:, 0], pose[:, 1])
    np.testing.assert_array_equal(z["fallen"], (clearance < height-.18) | (tilt > .6))

    # Feed the R111 lidar-only verifier its original speed signal and the wheel values
    # already verified at the parking/leg boundary. Its torque check uses the executed FIFO target.
    frame = z["actor_frame_command"].copy()
    frame[:, 0] = (2. * z["preview_actor_vx"]).astype(np.float32)
    adapted = z["raw"].astype(float)
    adapted[:, 3::4] = z["issued"][:, 3::4]
    preview = dict(z, actor_vx=z["preview_actor_vx"], actor_frame_command=frame, raw=adapted)
    motion, effects = score(preview, "development")
    check_episode(preview, dict(layout="development", arm="both", gain=SPEED_GAIN,
                                counts=counts, motion=motion, effects=effects))
    geometry = _check_leg_targets(z, ArchiveGeometry())
    lidar = lidar_effects(z, increment)
    stop = parking_stop(z["qpos"], z["qvel"])
    if stop is not None:
        require(stop["peak_m_s"] == motion["stop_tail_max_m_s"] and
                ("stopped" in motion["failed"]) == (stop["endpoints_above_limit"] > 0),
                "inconsistent stop window")
    complete = motion["ticks"] == TICKS
    return dict(seed=int(seed), trace_sha256=sha, counts=counts, complete_horizon=complete,
                driving_pass=complete and lidar["passed"] and all(k == "stopped" for k in motion["failed"]),
                gate_pass=complete and motion["passed"] and lidar["passed"], motion=motion, effects=effects,
                lidar=lidar, stop=stop, stop_endpoints=26 if complete else 0,
                first_failed_section=first_failed_section(motion), geometry=geometry, reset=reset,
                first_release_input_tick=first_release, first_projected_issue_tick=first_projection)
