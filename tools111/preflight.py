"""Static readback of the real sensor, geometry and leg targets; no integration or fitting."""
from tempfile import TemporaryDirectory

import mujoco as mj
import numpy as np

from tools110.native import POSE, SCALE
from tools111.control import WheelOdometry, Preview, body_to_level, yaw_from_quat
from tools111.course import ground, install_heightfield, make_scene, step_bounds
from tools111.sensing import LegGeometry, Lidar


def run():
    probes, checks, poses, rays = [], [], [], 0
    for layout in ("flat", "development", "blind"):
        with TemporaryDirectory(prefix="d1-static111-") as directory:
            model = mj.MjModel.from_xml_path(str(make_scene(directory, layout)))
        install_heightfield(model, layout)
        data = mj.MjData(model)
        data.qpos[2] = .44
        data.qpos[3:7] = (1., 0., 0., 0.)
        data.qpos[7:] = POSE
        mj.mj_forward(model, data)
        ids = np.empty(1, np.int32)
        locations = [2.68, 3.24, 3.72, 4.20, 4.48, 4.72, 6.20]
        if layout != "flat":
            start, end = step_bounds(layout)
            locations.extend((start-.007, start+.007, end-.007, end+.007))
        for x in locations:
            for y in (-.25, .25):
                hit = mj.mj_ray(model, data, np.array([x, y, 1.]), np.array([0., 0., -1.]), None, True, -1, ids)
                expected = float(ground(layout, x, y))
                assert hit >= 0. and abs(1.-hit-expected) <= .00065, (layout, x, y, 1.-hit, expected)
                probes.append([layout, x, y, 1.-hit, expected])
                rays += 1
        if layout == "blind":
            continue
        geometry, lidar = LegGeometry(model), Lidar(model)
        wheel_centers = geometry.centers(POSE)
        if layout == "flat":
            for val in (-.5, 0., .5):
                raw = val*np.tile([.1, -.3, .5, .8], 4)
                for lift in (np.array([.005, .008, .011, .012]), np.array([.012, 0., 0., 0.])):
                    before = geometry.centers(POSE+SCALE*raw)
                    issued, fk = geometry.raised_targets(raw, lift)
                    after = geometry.centers(POSE+SCALE*issued)
                    error = float(np.max(np.abs(fk-lift)))
                    horizontal = float(np.max(np.linalg.norm((after-before)[:, :2], axis=1)))
                    assert error < .0004 and horizontal < .0004
                    np.testing.assert_array_equal(issued[3::4], raw[3::4])
                    poses.append(dict(input_scale=val, lift=lift.tolist(), measured=fk.tolist(),
                                      max_vertical_error_m=error, max_horizontal_m=horizontal))
        # Flat at two pitches and a real 30 mm step ahead of the same stationary axle.
        origin_x = 0. if layout == "flat" else step_bounds(layout)[0]-.65
        for pitch in ((0., .06) if layout == "flat" else (0.,)):
            data.qpos[0], data.qpos[2] = origin_x, .44
            data.qpos[3:7] = (np.cos(pitch/2), 0., np.sin(pitch/2), 0.)
            mj.mj_forward(model, data)
            quat = data.sensor("trunk_quat").data.copy()
            wheel_xy = (wheel_centers @ body_to_level(quat))[:, :2]
            odom = WheelOdometry(yaw_from_quat(quat))
            packet, distances, valid = lidar.capture(0, data, quat, odom)
            rays += 1040
            # Compare batch raycasts to independent scalar first returns on both mounts.
            rot = data.xmat[lidar.base].reshape(3, 3)
            for i in np.arange(0, len(distances), 17):
                n = len(lidar.directions)
                origin = data.xpos[lidar.base] + rot @ lidar.mounts[i//n]
                direction = rot @ lidar.directions[i % n]
                scalar, robot = (mj.mj_ray(model, data, origin, direction, None, static, -1, ids)
                                 for static in (True, False))
                good = .14 <= scalar <= 1.45 and not (robot >= 0. and abs(robot-scalar) <= 1e-8)
                assert good == valid[i] and (not good or abs(distances[i]-scalar) < 1e-7)
                rays += 2
            c = Preview("both")
            c.observe(2, packet, odom, wheel_xy)
            for k in range(2, 9):
                speed, _, lift = c.targets(k, 1.5, 0., odom, wheel_xy)
            reduction = c.gain*1.5 - speed
            result = dict(layout=layout, pitch_rad=pitch, valid=int(valid.sum()), bins=c.coverage.tolist(),
                          speed_reduction_m_s=reduction, demand_m=lift.tolist())
            if layout == "flat":
                assert reduction == 0. and max(lift) == 0., result
            else:
                assert reduction >= .02 and min(lift[:2]) >= .005, result
            checks.append(result)
        assert data.time == 0.
    assert rays < 60000
    return dict(terrain_probes=probes, live_sensor=checks, kinematic_checks=poses,
                static_ray_queries=rays, physical_substeps=0)
