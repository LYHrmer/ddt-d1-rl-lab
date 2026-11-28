"""Read the four wheel requests from decision 525 and the first servo response at 526."""
import json

import numpy as np

from tools91.launch91 import ROOT, digest, read, require
from tools114.common import BLIND, DEV


ISSUE, APPLY = 525, 526
WHEELS = slice(3, None, 4)  # FL, FR, RL, RR in the fixed 16-output vendor interface.
ARCHIVES = {
    113: ("reports/speed_world113/results.json", "409a86c9f06fe9fce8df10cda0f0abf0c17b53574c41fae64abc4ef12d59b1dc"),
    114: ("reports/parking114/closed_loop.json", "8c7be8254f8ac3e6631e1cab39fa9c6140f3968f402f5697ba1c82d9c55d04b5"),
}


def rf_contrast_share(targets):
    """Squared projection onto FR versus the mean of FL/RL/RR, divided by all differential action energy."""
    wheels = np.asarray(targets, dtype=float)
    require(wheels.shape == (4,) and np.isfinite(wheels).all(), "four finite wheel targets required")
    residual = wheels - wheels.mean()
    norm_sq = float(residual @ residual)
    axis = np.array([-1., 3., -1., -1.]) / np.sqrt(12.)
    return float((residual @ axis)**2 / norm_sq) if norm_sq else None


def mode(omega, tau):
    require(np.isfinite(omega) and np.isfinite(tau), "finite encoder speed and physical torque required")
    power = omega * tau
    if power < 0.:
        return "braking"
    if power > 0.:
        return "driving_negative" if omega < 0 else "driving_positive"
    return "zero_power"


def _case(folder, stage, seed, projected):
    directory = folder / f"native_{seed}"
    path, report = directory / "trace.npz", directory / "report.json"
    require(stage["status"] == "COMPLETE" and digest(report) == stage["files"]["report.json"], "unsealed native worker")
    worker = read(report)
    run = worker["result"]
    trace_sha = digest(path)
    require(worker["status"] == run["status"] == "COMPLETE" and run["seed"] == seed and run["motion"]["ticks"] == 600
            and trace_sha == stage["files"]["trace.npz"] == worker["files"]["trace.npz"] == run["trace_sha256"],
            "wrong native trace or incomplete motion")
    keys = ("time", "qpos", "qvel", "imu_quat", "actor_frame_command", "encoder_wheel_rad_s", "first_torque_nm",
            "raw", "issued", "applied", "base_cmd")
    if projected:
        keys += ("driver_command_xyz", "parking_active", "actor_previous_raw", "actor_history_previous_raw",
                 "unprojected_issued", "wheel_common_removed_rad_s")
    # A compressed member is decoded once. Only the small 514..526 slice is retained.
    start = ISSUE - 11
    with np.load(path, allow_pickle=False) as archive:
        z = {k: archive[k][start:APPLY+1].copy() for k in keys}
    i, j = ISSUE - start, APPLY - start
    raw, issued, applied = (z[k] for k in ("raw", "issued", "applied"))
    np.testing.assert_allclose(z["time"], np.arange(start, APPLY+1) / 50., rtol=0., atol=1e-9)
    np.testing.assert_array_equal(applied[1:], issued[:-1])
    np.testing.assert_array_equal(raw[:i], issued[:i])
    np.testing.assert_array_equal(z["encoder_wheel_rad_s"], z["qvel"][:, 9::4])
    np.testing.assert_allclose(abs(np.sum(z["imu_quat"] * z["qpos"][:, 3:7], axis=1)), 1., rtol=0., atol=1e-9)
    require(z["base_cmd"][i-1] > 0 and not z["base_cmd"][i:].any(), "not the original stop edge")
    raw_wheels = raw[i, WHEELS].astype(float)
    if projected:
        require(run["projection"] == "parked_wheel_common_zero", "wrong parking controller")
        np.testing.assert_array_equal(z["driver_command_xyz"][i:], np.zeros((2, 3)))
        np.testing.assert_array_equal(z["parking_active"][i-1:], [False, True, True])
        np.testing.assert_array_equal(z["unprojected_issued"][i], raw[i])
        np.testing.assert_array_equal(z["actor_previous_raw"][i:j+1], raw[i-1:j])
        for k in (i, j):
            np.testing.assert_array_equal(z["actor_history_previous_raw"][k], raw[k-11:k-1])
        np.testing.assert_allclose(z["wheel_common_removed_rad_s"][i], 5.*raw_wheels.mean(), rtol=0., atol=1e-12)
        expected = raw[i].copy()
        expected[WHEELS] = raw_wheels - raw_wheels.mean()
        np.testing.assert_array_equal(issued[i], expected)
    else:
        np.testing.assert_array_equal(issued, raw)
    request = 5. * raw_wheels
    delivered = 5. * applied[j, WHEELS].astype(float)
    omega, tau = z["encoder_wheel_rad_s"][j], z["first_torque_nm"][j, WHEELS]
    np.testing.assert_allclose(tau, np.clip(.5 * (delivered - omega), -12., 12.), rtol=0., atol=2e-9)
    q = z["qpos"][i]
    tilt = np.arccos(np.clip(1.-2.*(q[4]**2+q[5]**2), -1., 1.))
    row = dict(seed=seed, trace_sha256=trace_sha, raw_wheel_action=raw_wheels.tolist(),
               raw_wheel_target_rad_s=request.tolist(), delivered_wheel_target_rad_s=delivered.tolist(),
               encoder_at_issue_rad_s=z["encoder_wheel_rad_s"][i].tolist(),
               delivery_change_from_raw_rad_s=(delivered-request).tolist(), wheel_encoder_rad_s=omega.tolist(),
               first_servo_torque_nm=tau.tolist(), first_servo_joint_power_w=(tau*omega).tolist(),
               first_servo_mode=[mode(w,t) for w,t in zip(omega,tau)], rf_contrast_share=rf_contrast_share(request),
               body_height_m_at_issue=float(q[2]), body_tilt_deg_at_issue=float(np.rad2deg(tilt)),
               actor_command_frame_at_issue=z["actor_frame_command"][i].tolist())
    aligned = {k: z[k][i:j+1] for k in ("qpos", "qvel", "imu_quat", "raw", "actor_frame_command", "encoder_wheel_rad_s")}
    return row, aligned


def audit():
    cases, paired_states = [], {}
    old = None
    for number, (name, sha) in ARCHIVES.items():
        path = ROOT / name
        require(digest(path) == sha, "public parent archive changed")
        stages = read(path)["stages"]
        for seed in ((DEV[0],) if number == 113 else (*DEV, *BLIND)):
            row, state = _case(path.parent, stages[f"native_{seed}"], seed, number == 114)
            if seed == DEV[0]:
                paired_states[number] = state
            if number == 113:
                old = row
            else:
                cases.append(row)
    for k in paired_states[113]:
        np.testing.assert_array_equal(paired_states[113][k], paired_states[114][k], err_msg=k)
    observed = np.array([r["encoder_at_issue_rad_s"] for r in cases])
    others = observed[:, [0, 2, 3]]
    return dict(issue_tick=ISSUE, applied_tick=APPLY, wheel_order=["FL", "FR", "RL", "RR"],
                archive_sha256={name: sha for name, sha in ARCHIVES.values()}, round114=cases, round113_original=old,
                matched_111011_equal_at_both_ticks=list(paired_states[113]),
                rf_braking_first_samples=sum(r["first_servo_mode"][1] == "braking" for r in cases),
                rf_fastest_at_issue=int((observed[:, 1] > others.max(axis=1)).sum()),
                rf_slowest_at_issue=int((observed[:, 1] < others.min(axis=1)).sum()),
                power_scope="Instantaneous joint power at the first of five servo samples, tau times encoder omega.",
                causal_scope="Body tilt and actor command are observed. Balance purpose and the causal contribution of "
                             "an individual actor input require independent interventions.",
                physics=0, actor_forwards=0, world_steps=0)


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
