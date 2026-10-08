#!/usr/bin/env python3
"""Compare the new native runtime to the independent Python training interface.

No model is loaded, no ROS node starts and no native physics step is performed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path('/home/lyh/ddt_ros2_ws')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--schema', type=Path, default=ROOT/'runtime03/schema_pilot03.json')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    schema = json.loads(args.schema.read_text())
    path = ROOT/'training/d1_rl/interface.py'
    spec = importlib.util.spec_from_file_location('d1_pure_interface03', path)
    interface = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interface)
    executable = ROOT/'runtime03/build/d1_runtime03'
    rng = np.random.default_rng(91303)
    results = []
    for delay in (0, 1, 2, 3):
        cases = []
        for tick in range(32):
            quat = rng.normal(size=4)
            quat /= np.linalg.norm(quat)
            state = {'gyro': rng.uniform(-3, 3, 3).tolist(), 'quat_wxyz': quat.tolist(),
                     'command': rng.uniform(-2, 2, 3).tolist(),
                     'q': (interface.DEFAULT_Q + rng.uniform(-.3, .3, 16)).tolist(),
                     'dq': rng.uniform(-60, 60, 16).tolist()}
            state['q'][3::4] = rng.uniform(-100, 100, 4).tolist()
            op = 'reset' if tick in (0, 17) else 'observe'
            cases.append({**state, 'op': op, 'label': f'{tick:02}_{op}'})
            cases.append({**state, 'op': 'action', 'label': f'{tick:02}_action',
                          'action': rng.uniform(-3, 3, 16).astype(np.float32).tolist()})
        fixture = {'delay_ticks': delay, 'cases': cases}
        fixture_path = args.output/f'fixture_delay_{delay}.json'
        fixture_path.write_text(json.dumps(fixture, indent=2)+'\n')
        native_output = args.output/f'cpp_delay_{delay}'
        env = dict(os.environ)
        env['LD_LIBRARY_PATH'] = '/opt/ros/humble/lib:/home/lyh/ddt_ros2_ws/install/mujoco/lib'
        native = subprocess.run(['rtk', 'proxy', str(executable), '--offline-golden',
                                 str(fixture_path), '--schema', str(args.schema), '--output', str(native_output)],
                                text=True, capture_output=True, env=env, timeout=30)
        (args.output/f'cpp_delay_{delay}.stdout').write_text(native.stdout)
        (args.output/f'cpp_delay_{delay}.stderr').write_text(native.stderr)
        if native.returncode:
            raise RuntimeError(native.stderr+native.stdout)
        actual = json.loads((native_output/'report.json').read_text())['cases']
        last = np.zeros(16, np.float32)
        executed = np.zeros(16, np.float32)
        filtered = np.zeros(3)
        history = np.zeros((10, 57), np.float32)
        fifo = []
        max_errors = {key: 0. for key in ('obs', 'history', 'filtered_gyro',
                                         'last_clipped_action', 'executed_action', 'torque')}
        for item, row in zip(cases, actual):
            if item['op'] in ('reset', 'observe'):
                if item['op'] == 'reset':
                    last.fill(0)
                    executed.fill(0)
                    filtered.fill(0)
                    fifo = [np.zeros(16, np.float32) for _ in range(delay)]
                obs, filtered = interface.build_observation(
                    item['gyro'], item['quat_wxyz'], item['command'], item['q'], item['dq'],
                    last, filtered)
                if item['op'] == 'reset':
                    history = np.tile(obs, (10, 1))
            else:
                last = interface.clip_action(item['action'])
                fifo.append(last.copy())
                executed = fifo.pop(0)
                history = np.vstack((history[1:], obs))
            torque, saturation = interface.action_to_torque(executed, item['q'], item['dq'],
                                                            scale=schema['action_scales'])
            expected = {'obs': obs, 'history': history.reshape(-1), 'filtered_gyro': filtered,
                        'last_clipped_action': last, 'executed_action': executed, 'torque': torque}
            for key, value in expected.items():
                err = float(np.max(np.abs(np.array(row[key], dtype=np.float64)-value)))
                max_errors[key] = max(max_errors[key], err)
                if err > 1e-5:
                    raise AssertionError((delay, item['label'], key, err))
            if not np.array_equal(row['saturation'], saturation):
                raise AssertionError((delay, item['label'], 'saturation'))
        results.append({'delay_ticks': delay, 'cases': len(cases), 'max_absolute_errors': max_errors,
                        'all_absolute_errors_le_1e_5': True, 'saturation_masks_match': True})
    report = {'status': 'PASS', 'physics_steps': 0, 'cases': sum(x['cases'] for x in results),
              'absolute_tolerance': 1e-5, 'results': results,
              'python_interface_sha256': digest(path), 'cpp_binary_sha256': digest(executable),
              'cpp_header_sha256': digest(ROOT/'runtime03/include/runtime03/interface.hpp'),
              'runtime_schema_sha256': digest(args.schema), 'interface_id': schema['interface_id'],
              'fixture_scope': 'new-contract observation/history, clipped issued action, delays 0..3, warm resets, PD force limits; no dynamics or ONNX execution'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
