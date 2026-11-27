"""Read the wheel commands, measured motion and delivered torque from the sealed 111 trajectories."""
import json

import numpy as np

from tools91.launch91 import digest, read
from tools111.results import PUBLIC


REPORT = PUBLIC / "wheel_response.json"
WHEELS = slice(3, None, 4)
RADIUS_M = .087
WHEEL_GAIN = .5


def sustained_from(flags, first_tick):
    """First sample of the final uninterrupted true interval after braking begins."""
    if not flags[-1]:
        return None
    false_ticks = np.flatnonzero(~flags[first_tick:])
    return first_tick + (int(false_ticks[-1]) + 1 if len(false_ticks) else 0)


def wheel_signals(z):
    action = z["applied"][:, WHEELS]
    np.testing.assert_array_equal(action[0], np.zeros(4))
    np.testing.assert_array_equal(action[1:], z["raw"][:-1, WHEELS])
    target = 5. * action.astype(float)
    measured = z["encoder_wheel_rad_s"]
    np.testing.assert_array_equal(measured, z["qvel"][:-1, 9::4])
    torque = z["first_torque_nm"][:, WHEELS]
    np.testing.assert_allclose(torque, np.clip(WHEEL_GAIN*(target-measured), -12., 12.), rtol=0., atol=2e-9)
    return target, measured, torque


def value(x):
    return round(float(x), 6)


def episode(z, motion):
    np.testing.assert_allclose(z["time"], np.arange(601)/50., rtol=0., atol=1e-9)
    target, measured, torque = wheel_signals(z)
    yaw = z["imu_yaw"]
    heading = np.column_stack((np.cos(yaw), np.sin(yaw)))
    body_v = np.sum(z["vxy"][:-1]*heading, axis=1)
    dx = np.sum(np.diff(z["xy"], axis=0)*heading, axis=1)
    cruise = slice(400, 450)  # 8.00 s to just before braking at 9.00 s, same 50 samples in every case.
    assert z["wheels_xy"][cruise, :, 0].min() > 7.
    np.testing.assert_array_equal(z["actor_vx"][cruise], z["base_cmd"][cruise])
    speed_cmd = z["actor_vx"][cruise]
    assert np.all(speed_cmd == speed_cmd[0])
    desired = RADIUS_M*target[cruise].mean()
    encoder = RADIUS_M*measured[cruise].mean()
    body = body_v[cruise].mean()
    post_course = dict(actor_command_m_s=value(speed_cmd[0]), mean_wheel_target_m_s=value(desired),
                       mean_wheel_encoder_m_s=value(encoder), body_forward_m_s=value(body),
                       target_minus_encoder_m_s=value(desired-encoder), encoder_minus_body_m_s=value(encoder-body),
                       saturated_wheel_ticks=int(z["wheel_saturation"][cruise].sum()))

    tail = slice(575, 600)  # 25 actions and 26 endpoint-inclusive body states between 11.50 and 12.00 s.
    assert not np.any(z["actor_vx"][tail]) and not np.any(z["applied_lift_fk"][tail])
    speeds = np.linalg.norm(z["vxy"][575:], axis=1)
    assert abs(speeds.max()-motion["stop_tail_max_m_s"]) < 1e-9
    measured_reverse = measured[tail] < 0.
    joint_displacement = z["qpos"][-1, 10::4]-z["qpos"][575, 10::4]
    zero_target_torque = np.clip(-WHEEL_GAIN*measured[tail], -12., 12.)
    stop = dict(signed_travel_mm=value(1000.*dx[tail].sum()), peak_speed_m_s=value(speeds.max()),
                body_states_above_0p05=int((speeds > .05).sum()), wheel_samples=int(target[tail].size),
                negative_targets_per_wheel=(target[tail] < 0.).sum(axis=0).tolist(),
                negative_encoders=int(measured_reverse.sum()), negative_joint_displacements=int((joint_displacement < 0.).sum()),
                negative_first_torques=int((torque[tail] < 0.).sum()),
                positive_first_wheel_power_samples=int((torque[tail]*measured[tail] > 0.).sum()),
                saturated_wheel_ticks=int(z["wheel_saturation"][tail].sum()),
                mean_first_torque_nm=value(torque[tail].mean()), mean_zero_target_torque_nm=value(zero_target_torque.mean()),
                positive_zero_target_torques=int((zero_target_torque > 0.).sum()),
                sustained_body_below_minus_0p01_tick=sustained_from(body_v < -.01, 450),
                sustained_four_negative_targets_tick=sustained_from((target < 0.).all(axis=1), 450))
    return dict(post_course=post_course, stop=stop)


def archive():
    manifest = read(PUBLIC/"native.json")
    cases = []
    for slot in manifest["cases"]:
        case = {key: slot[key] for key in ("seed", "arm", "case_status", "unit_status")}
        if slot["case_status"] == "COMPLETE":
            path = PUBLIC/slot["file"]
            sha = digest(path)
            assert sha == slot["episode"]["trace_sha256"]
            with np.load(path, allow_pickle=False) as z:
                case.update(trace_sha256=sha, **episode(z, slot["episode"]["motion"]))
        cases.append(case)
    return dict(source_manifest_sha256=digest(PUBLIC/"native.json"), cases=cases)


if __name__ == "__main__":
    result = archive()
    assert result == read(REPORT), "archived wheel-response measurements changed"
    print(json.dumps([r["stop"] for r in result["cases"] if r["arm"] == "both" and r["case_status"] == "COMPLETE"]))
