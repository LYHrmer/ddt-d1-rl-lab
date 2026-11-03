"""Standard-library driver of actual native adapter82 calls; no model/ROS/physics."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NativeAdapter82(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++')
        if compiler is None:
            raise unittest.SkipTest('A C++17 compiler is required for native checks')
        cls.work = tempfile.TemporaryDirectory(prefix='d1-adapter82-')
        cls.addClassCleanup(cls.work.cleanup)
        cls.executables = {}
        for name, source in (('tests', 'tests/native/adapter82_test.cpp'),
                             ('probe', 'runtime82/probe82.cpp')):
            output = Path(cls.work.name) / name
            subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-pedantic', '-I', str(ROOT), str(ROOT / source), '-o', str(output)],
                           check=True, capture_output=True, text=True, timeout=30)
            cls.executables[name] = output

    def run_native(self, name, *args):
        result = subprocess.run([str(self.executables[name]), *args], check=True,
                                capture_output=True, text=True, timeout=10)
        return json.loads(result.stdout)

    def test_native_contract(self):
        result = self.run_native('tests')
        print('adapter82 native receipt: ' + json.dumps(result, sort_keys=True))
        self.assertEqual(result['status'], 'PASS')
        self.assertGreater(result['assertions'], 100)
        self.assertEqual(result['physics_steps'], 0)
        self.assertEqual(result['model_calls'], 0)

    def test_nominal_probe_is_explicitly_synthetic_and_maps_25(self):
        result = self.run_native('probe', '--scenario', 'nominal')
        self.assertTrue(result['synthetic_actions'])
        self.assertFalse(result['ros_connected'])
        self.assertTrue(result['authorized'])
        self.assertEqual(len(result['raw_actor627']), 627)
        self.assertEqual(result['target_q16'][0], .125)
        self.assertEqual(result['target_dq16'][3], 5.)
        self.assertEqual(result['effort16'][0], 7.5)
        self.assertEqual(result['effort16'][3], 2.5)

    def test_fault_probe_never_exposes_position_targets(self):
        for scenario in ('late', 'stale_command', 'reset'):
            with self.subTest(scenario=scenario):
                result = self.run_native('probe', '--scenario', scenario)
                self.assertFalse(result['authorized'])
                self.assertFalse(result['has_policy_targets'])
                self.assertEqual(result['effort16'], [0] * 16)
                self.assertNotIn('target_q16', result)

    def test_legacy_failure_evidence_is_not_a_replayed_unsafe_clamp(self):
        # Static evidence only: do not execute std::clamp with reversed bounds.
        source = ROOT / 'src/ddt_ros2_control/controller/rl_controller/src/fsm'
        if not (source / 'FSM.cpp').exists() or not (source / 'FSMState_RL.cpp').exists():
            self.skipTest('Vendor ROS sources are local audit inputs, not shipped in this repository')
        fsm = (source / 'FSM.cpp').read_text()
        self.assertIn('std::clamp(\n      _data->low_cmd->tau_cmd[i], '
                      '_data->params->torque_limit[i], -_data->params->torque_limit[i]);', fsm)
        old = (source / 'FSMState_RL.cpp').read_text()
        enter = old.split('void FSMState_RL::enter()', 1)[1].split('void FSMState_RL::run()', 1)[0]
        self.assertNotIn('action_vec_.setZero', enter)
        evidence = (ROOT / 'docs/d1_interface_audit_03.md').read_text()
        self.assertIn('action_vec_', evidence)
        self.assertIn('22', evidence)

    def test_existing_03_native_receipt_reports_hot_reentry_failure(self):
        path = ROOT / 'evidence/interface_audit_03/golden_run_01/report.json'
        if not path.exists():
            self.skipTest('Local archived 03 evidence is not shipped in the source repository')
        evidence = json.loads(path.read_text())
        self.assertEqual(evidence['fixtures'], 22)
        self.assertEqual(evidence['actual_native_onnx_calls'], 19)
        self.assertTrue(evidence['warm_reentry_preserves_stale_action'])
        self.assertTrue(evidence['warm_reentry_history_last_actions_zero'])
        self.assertTrue(evidence['source_hashes_unchanged'])
        self.assertEqual(evidence['physics_steps'], 0)


if __name__ == '__main__':
    unittest.main()
