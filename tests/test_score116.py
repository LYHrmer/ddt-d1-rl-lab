import unittest

import numpy as np

from tools110.course import command
from tools113.control import BrakeTrim
from tools116.score import SEEDS, _source, _trim, first_failed_section, lidar_effects


def route(n=35, crossing=30):
    extent = np.zeros(n+1)
    if crossing is not None:
        extent[crossing+1:] = 5.12
    return dict(front_extent_x=extent, front_platform_touch=np.zeros(n, bool), capture_used=np.zeros(n, int),
                base_cmd=np.ones(n), actor_frame_command=np.zeros((n,3),np.float32),
                applied_lift_fk=np.zeros((n,4)), measured_relative_z=np.zeros((n+1,4)))


class CompoundScoring(unittest.TestCase):
    def test_both_channels_are_observed_before_first_contact_with_same_speed_trim(self):
        z = route(crossing=32)
        z["front_platform_touch"][30] = True
        trim = np.full(35,.15)
        z["actor_frame_command"][:,0] = 2.*(z["base_cmd"]+trim)
        z["actor_frame_command"][13,0] = 2.*(1.+.15-.030)
        z["applied_lift_fk"][21,1] = .006
        r = lidar_effects(z,trim)
        self.assertTrue(r["passed"])
        self.assertEqual((r["first_front_extent_tick"],r["first_loaded_platform_touch_tick"]), (32,30))
        self.assertEqual((r["window_start_tick"],r["end_tick_exclusive"]), (5,30))
        self.assertEqual((r["speed_tick"],r["lift_issue_tick"],r["lift_servo_tick"],r["front_wheel"]), (13,20,21,"FR"))
        self.assertAlmostEqual(r["speed_reduction_m_s"],.03,places=6)
        self.assertLess(z["base_cmd"][13] - z["actor_frame_command"][13,0]/2., 0.)

    def test_close_threshold_and_late_lift_do_not_pass(self):
        z = route()
        z["actor_frame_command"][:,0] = 2.
        z["actor_frame_command"][10,0] = 2.*(1.-.019999)
        z["applied_lift_fk"][20,0] = .0049999
        z["applied_lift_fk"][30,0] = .012
        r = lidar_effects(z,np.zeros(35))
        self.assertFalse(r["speed_active"] or r["lift_active"])
        self.assertEqual(r["lift_servo_tick"],20)
        z["actor_frame_command"][10,0] = 2.*(1.-.020001)
        z["applied_lift_fk"][20,0] = .005
        self.assertTrue(lidar_effects(z,np.zeros(35))["passed"])

    def test_lift_requires_a_packet_when_issued_and_an_edge_event(self):
        z = route()
        z["actor_frame_command"][:,0] = 2.*.95
        z["capture_used"][:20] = -1
        z["applied_lift_fk"][20,0] = .012
        self.assertFalse(lidar_effects(z,np.zeros(35))["lift_active"])
        z["front_extent_x"][:] = 0.
        r = lidar_effects(z,np.zeros(35))
        self.assertFalse(r["passed"])
        self.assertIsNone(r["speed_reduction_m_s"])
        self.assertIsNone(r["front_lift_m"])

    def test_completed_episode_from_interrupted_legacy_unit_is_identified(self):
        rows = [_source(seed)[1] for seed in SEEDS]
        self.assertEqual([r["unit_status"] for r in rows],["INCOMPLETE","COMPLETE","COMPLETE"])
        self.assertTrue(all(r["case_status"]=="COMPLETE" for r in rows))
        self.assertEqual([first_failed_section(r["episode"]["motion"])["from_x_m"] for r in rows], [4,3,4])

    def test_independent_compensation_replay_keeps_the_previous_radar_shaped_request(self):
        path, _ = _source(SEEDS[0])
        with np.load(path, allow_pickle=False) as f:
            z = {k: f[k] for k in ("raw", "encoder_wheel_rad_s", "actor_frame_command", "actor_vx")}
        z["preview_actor_vx"] = z["actor_vx"]
        control = BrakeTrim()
        last = 0.
        previous, inc, actual = [], [], []
        for k, (vx, wheels) in enumerate(zip(z["preview_actor_vx"], z["encoder_wheel_rad_s"])):
            previous.append(last)
            inc.append(control.update(command(k), last, wheels))
            last = vx + inc[-1]
            actual.append(last)
        z.update(actor_vx=np.array(actual), velocity_trim=np.array(inc), trim_previous_actor_vx=np.array(previous),
                 encoder_forward_m_s=.087*np.median(z["encoder_wheel_rad_s"],axis=1))
        z["actor_frame_command"][:,0] = 2.*z["actor_vx"]
        z["actor_previous_raw"] = np.vstack((np.zeros(16,np.float32), z["raw"][:-1]))
        h = np.maximum(np.arange(len(actual))[:,None] + np.arange(-10,0), 0)
        z["actor_history_frame_command"] = z["actor_frame_command"][h]
        z["actor_history_previous_raw"] = z["actor_previous_raw"][h]
        np.testing.assert_allclose(_trim(z), inc, atol=1e-12, rtol=0.)
        z["trim_previous_actor_vx"][200] += .01
        with self.assertRaises(AssertionError):
            _trim(z)
