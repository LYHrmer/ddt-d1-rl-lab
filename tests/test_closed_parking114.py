import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tools110.course import TICKS, command
from tools114.closed_results import assess, check_prefix, check_targets, development_passed, score_native
from tools114.common import DEV
from tools114.replay import RESET, TAIL


def flat_trace(n=TICKS):
    t = np.arange(n+1)/50.
    q = np.zeros((n+1,23))
    q[:,2:4] = [.44,1.]
    v = np.zeros((n+1,22))
    v[np.arange(n+1) < 450,0] = 1.52
    q[1:,0] = np.cumsum(v[:-1,0]) / 50.
    wheel = np.repeat(q[:,None,:2],4,axis=1)
    wheel[:,:,0] += [.2,.2,-.2,-.2]
    wheel[:,:,1] += [.2,-.2,.2,-.2]
    return dict(time=t,qpos=q,qvel=v,xy=q[:,:2],vxy=v[:,:2],wheels_xy=wheel,
                exit_contact=wheel[1:,:,0]>7.,top_contact=np.zeros((n,4),bool),
                fallen=np.zeros(n,bool),illegal=np.zeros(n,bool))


def actions(n):
    raw = np.broadcast_to(np.arange(16,dtype=np.float32)/8.-1.,(n,16)).copy()
    issued = raw.copy()
    mean = raw[:,3::4].astype(float).mean(axis=1)
    mean[:RESET] = 0.
    issued[:,3::4] = raw[:,3::4]-mean[:,None]
    zero = np.zeros((1,16),np.float32)
    prev = np.vstack((zero,raw[:-1]))
    return dict(raw=raw,issued=issued,applied=np.vstack((zero,issued[:-1])),
                driver_command_xyz=np.column_stack(([command(k) for k in range(n)],np.zeros((n,2)))),
                parking_active=np.arange(n)>=RESET,lift_demand=np.zeros((n,4)),
                issued_lift_fk=np.zeros((n,4)),applied_lift_fk=np.zeros((n,4)),
                unprojected_issued=raw.copy(),wheel_common_removed_rad_s=5.*mean,
                actor_previous_raw=prev,actor_history_previous_raw=prev[np.maximum(
                    np.arange(n)[:,None]+np.arange(-10,0),0)])


class NativeParkingAudit(unittest.TestCase):
    def test_independent_gate_uses_inclusive_vector_speed_and_signed_travel(self):
        z = flat_trace()
        z["qpos"][TAIL:,0] -= .0004*np.arange(26)
        z["qvel"][TAIL:,0] = -.02
        motion,stop = score_native(z,.44)
        self.assertTrue(motion["passed"])
        np.testing.assert_allclose(motion["sections_m_s"],1.52)
        self.assertAlmostEqual(stop["forward_displacement_mm"],-10.)
        z["qvel"][-1,:2] = [0.,.05]
        self.assertTrue(score_native(z,.44)[0]["passed"])
        z["qvel"][-1,:2] = [.04,.04]
        motion,stop = score_native(z,.44)
        self.assertEqual(motion["failed"],["stopped"])
        self.assertEqual(stop["peak_tick"],600)
        self.assertEqual(stop["peak_world_vxy_m_s"],[.04,.04])
        self.assertEqual(stop["endpoints_above_limit"],1)

    def test_early_result_and_false_fall_report(self):
        z = flat_trace(590)
        z["illegal"][-1] = True
        motion,stop = score_native(z,.44)
        self.assertIsNone(stop)
        self.assertEqual(motion["failed"],["horizon","stopped","legal_support"])
        z["qpos"][-1,2] = .20
        with self.assertRaises(AssertionError):
            score_native(z,.44)

    def test_target_projection_does_not_feed_back_to_actor_history(self):
        n = RESET + 5
        z = actions(n)
        a = check_targets(z)
        self.assertEqual(a["first_changed_issue_tick"],RESET)
        self.assertEqual(a["first_changed_applied_tick"],RESET+1)
        self.assertEqual(a["max_active_common_abs_rad_s"],0.)
        z["actor_history_previous_raw"][-1,-1] = z["issued"][-3]
        with self.assertRaises(AssertionError):
            check_targets(z)

    def test_paired_prefix_extends_through_start_of_first_changed_application(self):
        n = RESET + 5
        z = dict(**flat_trace(n),**actions(n))
        for key in ("actor_vx","actor_wz","actor_frame_command","encoder_wheel_rad_s","imu_quat","first_torque_nm"):
            z[key] = np.zeros(n)
        old = {key:arr.copy() for key,arr in z.items()}
        wiring = check_targets(z)
        z["qpos"][RESET+2,2] += .01
        older = check_prefix(z,old,wiring)
        self.assertEqual(older["equal_state_endpoints"],RESET+2)
        z["qvel"][RESET+1,0] += .01
        with self.assertRaises(AssertionError):
            check_prefix(z,old,wiring)

    def test_partial_and_duplicate_seeds_cannot_open_blind_gate(self):
        rows = [dict(seed=s,complete=True,gate_pass=True,paired=True) for s in DEV]
        self.assertTrue(development_passed(rows))
        self.assertFalse(development_passed([rows[0],rows[0],rows[0]]))
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            self.assertEqual(assess(DEV[-1],p)["stage_status"],"NOT_STARTED")
            (p/"worker_partial.json").write_text(json.dumps({"error":"execution interrupted"}))
            partial=assess(DEV[-1],p)
            self.assertEqual(partial["stage_status"],"PARTIAL")
            self.assertEqual(partial["error"],"execution interrupted")
            self.assertFalse(development_passed(rows[:-1]+[partial]))
