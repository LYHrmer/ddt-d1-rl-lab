import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tools115 import policy_probe as p


class LinearStub:
    """Synthetic 11-frame policy reading only the four wheel channels being probed."""
    def __init__(self):
        self.calls = 0

    def run(self, names, inputs):
        self.calls += 1
        current,history = inputs["nn_input0"],inputs["nn_input1"]
        np.testing.assert_equal(current.shape,(1,57))
        np.testing.assert_equal(history.shape,(1,10,57))
        f = np.vstack((history[0],current))
        output = np.zeros((1,16),np.float32)
        output[0,p.WHEEL_INDICES] = (f[:,p.ENCODER_COLUMNS]+f[:,p.FEEDBACK_COLUMNS]).mean(axis=0)
        return [output]


def source(seed):
    frames = np.zeros((11,57),np.float32)
    frames[:,p.ENCODER_COLUMNS] = [-1.,1.,0.,0.]
    frames[:,p.FEEDBACK_COLUMNS] = [-1.,1.,0.,0.]
    raw = np.zeros(16,np.float32)
    raw[p.WHEEL_INDICES] = [-2.,2.,0.,0.]
    return p.Source(seed,frames,raw,str(seed))


class FixedPolicyProbe(unittest.TestCase):
    def test_input_rebuild_uses_filtered_local_gyro_and_real_raw_history(self):
        n = 14
        q = np.zeros((n+1,23))
        q[:,3]=1.
        q[:,7:]=p.POSE
        dq = np.zeros((n+1,22))
        dq[0,3]=1.
        dq[:,9::4] = [20.,2.,-3.,-4.]
        raw = (np.arange(n*16).reshape(n,16)/8.).astype(np.float32)
        previous = np.vstack((np.zeros((1,16),np.float32),raw[:-1]))
        history = previous[np.maximum(np.arange(n)[:,None]+np.arange(-10,0),0)]
        trace = dict(qpos=q,qvel=dq,imu_quat=q[:-1,3:7],raw=raw,actor_frame_command=np.zeros((n,3),np.float32),
                     actor_vx=np.zeros(n),actor_wz=np.zeros(n),encoder_wheel_rad_s=dq[:-1,9::4],
                     actor_previous_raw=previous,actor_history_previous_raw=history)
        frames = p.rebuild_inputs(trace,11)
        self.assertEqual(frames.dtype,np.float32)
        self.assertEqual(frames[0,0],np.float32(.25*.97*.03))
        np.testing.assert_array_equal(frames[:,3:6],np.tile([0.,0.,-1.],(11,1)))
        np.testing.assert_array_equal(frames[:,9:25],np.zeros((11,16)))
        np.testing.assert_array_equal(frames[-1,p.ENCODER_COLUMNS],np.float32([1.,.1,-.15,-.2]))
        np.testing.assert_array_equal(frames[-1,41:],raw[10])
        trace["actor_history_previous_raw"][11,-1,0] += 1.
        with self.assertRaises(AssertionError):
            p.rebuild_inputs(trace,11)

    def test_each_mean_is_per_frame_and_leaves_all_other_slots_untouched(self):
        frames = np.arange(11*57,dtype=np.float32).reshape(11,57)
        for arm,slots in zip(p.ARMS[1:],(p.ENCODER_COLUMNS,p.FEEDBACK_COLUMNS)):
            altered = p.altered(frames,arm)
            keep = np.ones(57,dtype=bool)
            keep[slots] = False
            np.testing.assert_array_equal(frames[:,keep],altered[:,keep])
            expected = frames[:,slots].mean(axis=1,keepdims=True)
            np.testing.assert_array_equal(altered[:,slots],np.broadcast_to(expected,(11,4)))
        np.testing.assert_array_equal(p.altered(frames,"original"),frames)

    def test_wheel_parity_checks_physical_units(self):
        registered = np.zeros(16,np.float32)
        observed = registered.copy()
        observed[7]=3e-6
        error=p._parity(observed,registered)
        self.assertLess(error["max_raw_error"],1e-5)
        self.assertGreater(error["max_wheel_target_error_rad_s"],1e-5)
        self.assertFalse(error["all_outputs_match"])

    def test_budget_gate_and_zero_forward_archive_verifier(self):
        for mismatch in (False,True):
            stub = LinearStub()

            def original(seed):
                s = source(seed)
                if mismatch and seed == p.SEEDS[-1]:
                    s.raw[7] += .01
                return s

            with self.subTest(mismatch=mismatch), TemporaryDirectory() as name, \
                    patch.object(p,"prepare",side_effect=original), patch.object(p,"open_actor",return_value=stub), \
                    patch.object(p,"_check_imu_contract",side_effect=p._expected_imu_contract) as xml_check:
                directory = Path(name)
                result=p.run_probe(directory)
                expected = 6 if mismatch else 18
                self.assertEqual(result["actor_forwards"],expected)
                self.assertEqual(result["baseline_parity"],not mismatch)
                if not mismatch:
                    self.assertEqual(result["counts_reaching_reduction"],{k:6 for k in p.ARMS[1:]})
                with patch.object(p,"_check_imu_contract",side_effect=FileNotFoundError("no vendor XML in public checkout")):
                    receipt=p.verify(directory,result)
                xml_check.assert_called_once_with()
                self.assertEqual(receipt["new_forwards"],0)
                self.assertEqual(stub.calls,expected)
                # A fresh checksum of an altered convention is insufficient.
                file = directory/p.IMU_RECEIPT
                misbound = json.loads(file.read_text())
                misbound["gyro_qvel_columns"] = [0,1,2]
                file.write_text(json.dumps(misbound))
                with self.assertRaises(ValueError):
                    p.verify(directory,dict(result,imu_contract_sha256=p.digest(file)))
                with self.assertRaises(ValueError):
                    p.run_probe(directory)

    def test_interrupted_forward_retains_attempt_and_completed_rows(self):
        stub = LinearStub()
        run = stub.run

        def interrupt(names,inputs):
            if stub.calls == 2:
                raise RuntimeError("synthetic interruption")
            return run(names,inputs)

        with TemporaryDirectory() as name, patch.object(p,"prepare",side_effect=source), \
                patch.object(p,"open_actor",return_value=stub), patch.object(stub,"run",side_effect=interrupt), \
                patch.object(p,"_check_imu_contract",side_effect=p._expected_imu_contract):
            folder = Path(name)
            with self.assertRaises(RuntimeError):
                p.run_probe(folder)
            partial=np.load(folder/"partial_policy_outputs.npy")
            self.assertTrue(np.isfinite(partial[:2,0]).all())
            self.assertTrue(np.isnan(partial[2:,0]).all())
            self.assertEqual(len((folder/p.JOURNAL).read_text().splitlines()),5)
            self.assertFalse((folder/p.BUNDLE).exists())
            self.assertTrue((folder/p.IMU_RECEIPT).is_file())

    def test_local_xml_hash_sensor_binding_and_pose_are_checked(self):
        with TemporaryDirectory() as name:
            root = Path(name)
            cfg = root/"pilot.json"
            robot = root/"robot.xml"
            for attribute in ("", 'pos="0 0 .3"', 'quat="0 1 0 0"'):
                xml = ('<mujoco><worldbody><body name="base_link">'
                       '<joint name="floating_base_joint" type="free"/>'
                       f'<site name="trunk_imu" {attribute}/></body></worldbody>'
                       '<sensor><gyro name="trunk_gyro" site="trunk_imu"/>'
                       '<framequat name="trunk_quat" objtype="site" objname="trunk_imu"/></sensor></mujoco>')
                robot.write_text(xml)
                sha = p.digest(robot)
                cfg.write_text(json.dumps({"scene":"scene.xml","robot_xml_sha256":sha}))
                with self.subTest(attribute=attribute), patch.object(p,"ROOT",root), patch.object(p,"CFG",cfg), \
                        patch.object(p,"CFG_SHA",p.digest(cfg)), patch.object(p,"ROBOT_SHA",sha):
                    if attribute:
                        with self.assertRaises(ValueError):
                            p._check_imu_contract()
                    else:
                        self.assertEqual(p._check_imu_contract(),p._expected_imu_contract())
                    robot.write_text(xml+" ")
                    with self.assertRaisesRegex(ValueError,"unregistered D1 IMU geometry"):
                        p._check_imu_contract()

    def test_local_xml_failure_prevents_policy_load(self):
        with TemporaryDirectory() as name, patch.object(p,"_check_imu_contract",side_effect=FileNotFoundError), \
                patch.object(p,"open_actor") as load:
            with self.assertRaises(FileNotFoundError):
                p.run_probe(Path(name))
            load.assert_not_called()
            self.assertEqual(list(Path(name).iterdir()), [])
