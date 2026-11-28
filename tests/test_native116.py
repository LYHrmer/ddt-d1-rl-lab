from collections import defaultdict
import unittest

import numpy as np


def actor_frame(plant, vx, last, gyro):
    return np.float32(2.*vx)


def sample_drive(requests):
    return [actor_frame(None,v,None,None) for v in requests]


def replay_inputs(z, actor, rows):
    """Feed saved vendor outputs back into the input/history FIFO, without ONNX or a plant."""
    history = None
    raw = np.zeros(16,np.float32)
    for k,vx in enumerate(z["actor_vx"]):
        frame = np.zeros(57,np.float32)
        frame[6] = actor_frame(None,vx,raw,None)
        frame[7:9] = z["actor_frame_command"][k,1:]
        frame[41:57] = raw
        if history is None:
            history = np.tile(frame,(10,1))
        raw = actor.run(["recorded"],{"nn_input0":frame[None,:],"nn_input1":history[None,:,:]})[0][0]
        rows["actor_frame_command"].append(frame[6:9].copy())
        history[:-1],history[-1] = history[1:].copy(),frame


class ArchivedVendor:
    def __init__(self, actions):
        self.actions = iter(actions)

    def run(self, outputs, inputs):
        return [next(self.actions)[None,:]]


class TestCompoundDrive(unittest.TestCase):
    def test_driver_taps_exact_lidar_speed_without_mutating_the_frozen_loop(self):
        from tools116.native import with_preview_input

        class Recorder:
            def __init__(self):
                self.seen = []

            def set_preview(self, speed):
                self.seen.append(speed)

        actor = Recorder()
        speeds = [.97792348*1.5-.045, .97792348*1.5-.009]
        bound = with_preview_input(sample_drive,actor)
        self.assertEqual(bound(speeds),sample_drive(speeds))
        self.assertEqual(actor.seen,speeds)
        self.assertNotEqual(float(np.float32(speeds[0])),actor.seen[0])

    def test_archived_leg_fk_delayed_actions_and_lidar_inputs_survive_the_encoder_trim(self):
        from tools116.control import PrevTrimActor
        from tools116.native import check_feedback, matched_prefix, with_preview_input
        from tools116.score import reference

        path, old_row = reference(111101)
        self.assertEqual(old_row["motion"]["ticks"],600)
        with np.load(path,allow_pickle=False) as file:
            z = dict(file)
        # This archived case actually reaches the step, applies lidar slowdown and lifts its front legs.
        self.assertGreater(np.max(z["base_cmd"]-z["actor_vx"]),.04)
        self.assertGreater(np.max(z["applied_lift_fk"][:,:2]),.01)
        for enabled in (False,True):
            rows = defaultdict(list)
            wheel = z["encoder_wheel_rad_s"]
            actor = PrevTrimActor(ArchivedVendor(z["raw"]),lambda:wheel[len(rows["actor_frame_command"])],enabled)
            with_preview_input(replay_inputs,actor)(z,actor,rows)
            trace = actor.trace(dict(z,**rows))
            check_feedback(trace)
            prefix = matched_prefix(trace,z)
            np.testing.assert_array_equal(trace["preview_actor_vx"],z["actor_vx"])
            np.testing.assert_array_equal(trace["actor_frame_command"][:,1:],z["actor_frame_command"][:,1:])
            np.testing.assert_array_equal(trace["issued_lift_fk"],z["issued_lift_fk"])
            if enabled:
                tick = prefix["first_changed_actor_frame_tick"]
                self.assertIsNotNone(tick)
                self.assertEqual(prefix["equal_state_endpoints"],tick+2)
                disturbed = dict(trace,first_torque_nm=trace["first_torque_nm"].copy())
                disturbed["first_torque_nm"][tick,7] += .25
                with self.assertRaises(AssertionError):
                    matched_prefix(disturbed,z)
            else:
                np.testing.assert_array_equal(trace["actor_frame_command"],z["actor_frame_command"])
                self.assertIsNone(prefix["first_changed_actor_frame_tick"])
                self.assertEqual(prefix["equal_state_endpoints"],601)
