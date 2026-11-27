import unittest

import numpy as np

from tools111.control import Packet, Preview, SPEED_GAIN, WheelOdometry, body_to_level, heading_rate
from tools111.course import ground, score
from tools110.course import TICKS


WHEELS = np.array([[.17,.255],[.17,-.255],[-.278,.255],[-.278,-.255]])


def packet(raised):
    x = np.arange(.24, 1.08, .008)
    points = np.vstack([np.column_stack((x, np.full_like(x, y+d),
                        -.43+raised*((x >= .68) & (y > 0))))
                        for y in (.255, -.255) for d in (-.04, 0., .04)])
    return Packet(0, points, np.zeros(2), 0.)


def run_frame(arm, p):
    c, odom = Preview(arm), WheelOdometry(0.)
    c.observe(2, p, odom, WHEELS)
    for k in range(2, 9):
        vx, _, lift = c.targets(k, 1.5, 0., odom, WHEELS)
    return vx, lift


class LidarCompound(unittest.TestCase):
    def test_wheel_queries_and_returns_share_the_gravity_frame(self):
        pitch = .12
        level = body_to_level(np.array([np.cos(pitch/2), 0., np.sin(pitch/2), 0.]))
        wheel = np.array([.17,.255,-.35])
        expected_x = .17*np.cos(pitch)-.35*np.sin(pitch)
        self.assertAlmostEqual((wheel @ level)[0], expected_x)
        self.assertGreater(abs(wheel[0]-expected_x), .04)
        np.testing.assert_allclose(level @ level.T, np.eye(3), atol=1e-12)

    def test_isolated_channels_and_missing_returns(self):
        base = 1.5*SPEED_GAIN
        for p in (packet(0), Packet(0, np.full((520, 3), np.nan), np.zeros(2), 0.)):
            vx, lift = run_frame("both", p)
            self.assertEqual(vx, base)
            np.testing.assert_array_equal(lift, 0.)
        for arm in ("neither", "speed", "legs", "both"):
            vx, lift = run_frame(arm, packet(.03))
            self.assertAlmostEqual(base-vx, .045 if arm in ("speed", "both") else 0.)
            expected = .012 if arm in ("legs", "both") else 0.
            np.testing.assert_allclose(lift, [expected, 0., 0., 0.], atol=1e-10)

    def test_future_stale_yaw_and_rear_handoff(self):
        odom, c, p = WheelOdometry(0.), Preview("both"), packet(.03)
        for k in (0, 1, 3):
            with self.assertRaises(ValueError):
                c.observe(k, p, odom, WHEELS)
        c.observe(2, p, odom, WHEELS)
        with self.assertRaises(ValueError):
            c.observe(2, p, odom, WHEELS)
        odom.xy[0] = odom.travel = .82
        for k in range(3, 10):
            _, _, lift = c.targets(k, 1.5, 0., odom, WHEELS)
        self.assertEqual(lift[0], 0.)
        self.assertGreater(lift[2], .005)
        for k in range(47, 60):
            vx, _, lift = c.targets(k, 1.5, 0., odom, WHEELS)
        self.assertEqual(vx, 1.5*SPEED_GAIN)
        np.testing.assert_array_equal(lift, 0.)
        self.assertAlmostEqual(heading_rate(-np.pi+.02, np.pi-.02, True), .1)
        self.assertEqual(heading_rate(1., 0., False), 0.)
        np.testing.assert_allclose(p.aligned(odom)[:, 0], p.level_points[:, 0]-.82)
        odom.yaw = np.pi/2
        np.testing.assert_allclose(p.aligned(odom)[:, 0], p.level_points[:, 1], atol=1e-12)

    def test_true_geometry_and_before_contact_scoring(self):
        self.assertAlmostEqual(float(ground("development", 3.25, 0)), .75*np.tan(np.deg2rad(4)))
        self.assertEqual(float(ground("development", 5.15, 0)), .03)
        self.assertEqual(float(ground("blind", 5.15, 0)), 0.)
        y = np.array([.25, -.25])
        self.assertGreater(abs(np.diff(ground("development", 4.3, y))[0]), .0249)
        t = np.arange(TICKS+1)/50
        x = 1.53*t
        xy = np.column_stack((x, np.zeros_like(x)))
        z = dict(time=t, xy=xy, vxy=np.zeros_like(xy), wheels_xy=xy[:,None,:]+WHEELS,
                 exit_contact=np.ones((TICKS,4),bool), top_contact=np.ones((TICKS,4),bool),
                 fallen=np.zeros(TICKS,bool), illegal=np.zeros(TICKS,bool),
                 front_extent_x=x+.27, front_platform_touch=np.zeros(TICKS,bool),
                 actor_vx=np.ones(TICKS), base_cmd=np.ones(TICKS), applied_lift_fk=np.zeros((TICKS,4)),
                 capture_used=np.zeros(TICKS,int))
        motion, effects = score(z, "development")
        self.assertTrue(motion["passed"])
        k = effects["first_front_edge_tick"]
        z["actor_vx"][k:] -= .04
        z["applied_lift_fk"][k:, 0] = .012
        self.assertFalse(score(z, "development")[1]["speed_active"])
        self.assertFalse(score(z, "development")[1]["lift_active"])
        z["actor_vx"][k-5] -= .03
        z["applied_lift_fk"][k-3, 1] = .006
        self.assertTrue(score(z, "development")[1]["speed_active"])
        self.assertTrue(score(z, "development")[1]["lift_active"])


if __name__ == "__main__":
    unittest.main()
