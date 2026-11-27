"""Reconstruct the issued commands from archived first returns, IMU and encoder signals."""
import argparse
from pathlib import Path

import numpy as np

from tools91.launch91 import ROOT, read
from tools100.sensor100 import _grid
from tools110.course import command
from tools111.control import Packet, Preview, WheelOdometry, axes, body_to_level, yaw_from_quat
from tools111.course import score


def check_episode(z, row):
    z = dict(z)
    motion, effects = score(z, row["layout"])
    assert row["motion"] == motion and row["effects"] == effects
    n, gain = motion["ticks"], row["gain"]
    captures = [k for k in range(n) if not k or 6*k//50 != 6*(k-1)//50]
    np.testing.assert_array_equal(z["capture_tick"], captures)
    count = len(captures)
    assert row["counts"] == dict(decisions=n, servo=5*n, physics=10*n, packets=count, ray_queries=1040*count)
    assert z["range_m"].shape == (count, 520)
    spec = read(ROOT/"docs/experiments100/pointcloud_density.json")["geometry"]
    beams = _grid(spec)[0]
    origins = np.repeat(spec["mounts_m"], len(beams), axis=0)
    directions = np.tile(beams, (2, 1))
    valid = (z["range_m"] >= spec["range_min_m"]) & (z["range_m"] <= spec["range_max_m"]) & ~z["self_hit"]
    np.testing.assert_array_equal(valid, z["range_valid"])
    np.testing.assert_array_equal(z["applied"], np.vstack((np.zeros(16), z["issued"][:-1])))
    np.testing.assert_array_equal(z["applied_lift_fk"], np.vstack((np.zeros(4), z["issued_lift_fk"][:-1])))
    np.testing.assert_array_equal(z["raw"][:, 3::4], z["issued"][:, 3::4])
    np.testing.assert_array_equal(z["encoder_wheel_rad_s"], z["qvel"][:-1, 9::4])
    np.testing.assert_allclose(abs(np.sum(z["imu_quat"]*z["qpos"][:-1, 3:7],axis=1)), 1., rtol=0, atol=1e-9)
    # This uses ground truth only to independently check the logged encoder-derived query location.
    for k, yaw in enumerate(z["imu_yaw"]):
        actual = (z["wheels_xy"][k]-z["xy"][k]) @ axes(yaw)
        np.testing.assert_allclose(z["wheel_query_xy"][k], actual, atol=1e-8, rtol=0.)
    # Compare to the manufacturer force law independently of the simulator or the live adapter.
    pose = np.array([.1,.8,-1.5,0., -.1,.8,-1.5,0., .1,1.,-1.5,0., -.1,1.,-1.5,0.])
    actual_q = z["qpos"][:-1, 7:].copy()
    actual_q[:, 3::4] = 0.
    torque = np.tile([100.,60.,60.,10.],4)*(pose+np.tile([.125,.25,.25,.25],4)*z["applied"]-actual_q)
    torque -= np.tile([4.,1.5,1.5,.5],4)*z["qvel"][:-1, 6:]
    limits = np.tile([80.,80.,80.,12.],4)
    np.testing.assert_allclose(np.clip(torque,-limits,limits), z["first_torque_nm"], atol=2e-9, rtol=0.)

    # Only onboard signals from here to the targets. Remove the terrain response on the same trajectory as a control.
    imu, wheel_speed, wheel_xy = (z[k] for k in ("imu_quat","encoder_wheel_rad_s","wheel_query_xy"))
    reference = yaw_from_quat(imu[0])
    odom, active = WheelOdometry(reference), Preview(row["arm"], gain)
    masked = Preview("neither" if active.heading else "heading_off", gain)
    packet = None
    idx = 0
    frames = []
    for k in range(n):
        yaw = yaw_from_quat(imu[k])
        odom.update(yaw, wheel_speed[k])
        if idx < count and k == captures[idx]:
            points = (origins + directions*z["range_m"][idx, :, None]) @ body_to_level(imu[k])
            points[~valid[idx]] = np.nan
            packet = Packet(k, points, odom.xy.copy(), yaw)
            idx += 1
        if packet is not None and k == packet.tick+2:
            active.observe(k, packet, odom, wheel_xy[k])
            packet = None
        cmd = command(k)
        vx, wz, lift = active.targets(k, cmd, reference, odom, wheel_xy[k])
        masked_vx, _, masked_lift = masked.targets(k, cmd, reference, odom, wheel_xy[k])
        assert masked_vx == gain*cmd and not any(masked_lift)
        assert active.last_packet == z["capture_used"][k]
        frames.append([vx,wz,*lift,masked_vx,odom.travel])
    wanted = np.asarray(frames)
    logged = np.column_stack((z["actor_vx"],z["actor_wz"],z["lift_demand"],z["base_cmd"],z["odom_travel"]))
    np.testing.assert_allclose(wanted, logged, atol=1e-9, rtol=0.)
    expected_frame = np.column_stack((2*wanted[:,0],np.zeros(n),.25*wanted[:,1])).astype(np.float32)
    np.testing.assert_array_equal(expected_frame, z["actor_frame_command"])
    return motion, effects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="native episode directory containing trace.npz and report.json")
    directory = parser.parse_args().directory
    record = read(directory/"report.json")
    from tools91.launch91 import digest
    assert record["status"] == "COMPLETE" and record["trace_sha256"] == digest(directory/"trace.npz")
    with np.load(directory/"trace.npz", allow_pickle=False) as trace:
        _, effects = check_episode(trace, record)
    print(record["seed"], record["arm"], record["motion"]["passed"], effects)
