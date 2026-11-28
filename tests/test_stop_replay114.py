import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from tools110.course import command
from tools111.control import SPEED_GAIN
from tools113.model import ACTIONS, STATES, TICKS
from tools114.replay import RESET, TAIL, _check_transitions, _row, _signals, fixed_suffix, stopping


class StopReplay(unittest.TestCase):
    def test_suffix_uses_exact_original_ticks_and_applied_actions(self):
        ticks = []

        def step(tick, state, target):
            ticks.append(tick)
            out = state.copy()
            out[3] += target[3]
            return out

        state = np.zeros(STATES)
        state[3] = .1
        actions = np.zeros((3, ACTIONS))
        actions[:,3] = [1., -2., 4.]
        destination = np.full((4,STATES),np.nan)
        path = fixed_suffix(SimpleNamespace(step=step), RESET, state, actions, destination)
        self.assertIs(path,destination)
        self.assertEqual(ticks, [525,526,527])
        np.testing.assert_allclose(path[:,3],[.1,1.1,-.9,3.1])
        self.assertEqual(state[3], .1)

    def test_peak_includes_last_endpoint_excludes_tick_before_tail(self):
        path = np.zeros((TICKS-RESET+1, STATES))
        path[TAIL-RESET-1,3] = 9.
        path[TAIL-RESET:,3] = -.03
        path[-1,3:5] = [0.,.06]
        result = stopping(path)
        self.assertEqual(result["peak_tick"], 600)
        self.assertEqual(result["endpoints_above_limit"], 1)
        self.assertFalse(result["passes_stop"])
        self.assertAlmostEqual(result["mean_forward_m_s"], -.03)
        path[-1,4] = .05
        self.assertTrue(stopping(path)["passes_stop"])

    def test_factorial_order_preserves_false_pass_and_interaction(self):
        def block(peak):
            arr = np.zeros((TICKS-RESET+1, STATES))
            arr[:,3] = peak
            return arr

        values = {"sealed_sealed": .01, "native_sealed": .02, "sealed_native": .06, "native_native": .09}
        paths = {key:block(value) for key,value in values.items()}
        row = _row(1,paths,block(.07),block(.01),{}, {})
        self.assertTrue(row["arms"]["native_sealed"]["false_pass"])
        self.assertFalse(row["arms"]["sealed_native"]["false_pass"])
        self.assertAlmostEqual(row["arms"]["native_native"]["peak_error_m_s"], .02)
        self.assertAlmostEqual(row["delta_peak_m_s"]["state_with_native_targets"], .03)
        self.assertAlmostEqual(row["delta_peak_m_s"]["targets_with_native_state"], .07)
        self.assertAlmostEqual(row["delta_peak_m_s"]["interaction"], .02)

    def test_audit_detects_tampered_transition_without_new_rollout(self):
        world = SimpleNamespace(reference_state=np.zeros((TICKS+1,STATES)),
                                reference_target=np.zeros((TICKS,ACTIONS)),
                                state_scale=np.ones(STATES),target_scale=np.ones(ACTIONS),
                                weight=np.zeros((STATES+ACTIONS,STATES)),bias=np.zeros(STATES),
                                step=Mock(side_effect=AssertionError("unexpected rollout")))
        world.weight[STATES+3,3] = 1.
        path = np.zeros((TICKS-RESET+1,STATES))
        targets = np.zeros((TICKS-RESET,ACTIONS))
        _check_transitions(world,path,targets)
        targets[-1,3] = 1.
        with self.assertRaises(AssertionError):
            _check_transitions(world,path,targets)
        world.step.assert_not_called()

    def test_signal_check_distinguishes_driver_and_actor_at_stop(self):
        drive = SPEED_GAIN*np.array([command(k) for k in range(TICKS)])
        actor = drive.copy()
        actor[RESET:] = -.1
        raw = np.zeros((TICKS,ACTIONS))
        raw[RESET-1:,3::4] = -.2
        applied = np.vstack((np.zeros(ACTIONS),raw[:-1]))
        v = np.zeros((TICKS+1,22))
        v[RESET:,9::4] = -1.2
        torque = np.zeros_like(raw)
        torque[:,3::4] = .5*(5.*applied[:,3::4]-v[:-1,9::4])
        z = dict(base_cmd=drive, actor_vx=actor, time=np.arange(TICKS+1)/50., raw=raw,
                 applied=applied, encoder_wheel_rad_s=v[:-1,9::4], qvel=v, first_torque_nm=torque,
                 wheel_saturation=np.zeros((TICKS,4),dtype=bool))
        signals = _signals(z)
        self.assertEqual((signals["first_stop_request_tick"],signals["stop_request_m_s"]),(525,0.))
        self.assertEqual(signals["actor_vx_at_stop_request_m_s"],-.1)
        self.assertEqual(signals["negative_targets_per_wheel"],[25]*4)
        self.assertEqual(signals["negative_encoders_per_wheel"],[25]*4)
        self.assertEqual(signals["negative_first_torques_per_wheel"],[0]*4)
        self.assertEqual(signals["saturated_wheel_intervals_per_wheel"],[0]*4)
