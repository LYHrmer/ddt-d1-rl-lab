#!/usr/bin/env python3
"""Exercise real CLI output ownership and schema rejection without integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/lyh/ddt_ros2_ws')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    executable = ROOT / 'runtime03/build/d1_runtime03'
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = '/opt/ros/humble/lib:/home/lyh/ddt_ros2_ws/install/mujoco/lib'

    def invoke(label, options):
        result = subprocess.run(['rtk', 'proxy', str(executable), *options],
                                env=env, text=True, capture_output=True, timeout=30)
        (args.output / f'{label}.stdout').write_text(result.stdout)
        (args.output / f'{label}.stderr').write_text(result.stderr)
        return result

    successful = args.output / 'component_selftest'
    assert invoke('selftest', ['--self-test', '--output', str(successful)]).returncode == 0
    first = successful / 'report.json'
    before = digest(first)
    assert invoke('existing_output', ['--self-test', '--output', str(successful)]).returncode != 0
    assert digest(first) == before, 'Existing report was overwritten'
    rejected = args.output / 'wrong_schema_hash'
    assert invoke('wrong_schema_hash', ['--validate-only', '--expected-schema-sha256',
                                       '0' * 64, '--output', str(rejected)]).returncode != 0
    failure = json.loads((rejected / 'report.json').read_text())
    assert failure['native_steps_attempted'] == 0
    assert failure['native_steps_completed'] == 0
    assert 'schema SHA256 mismatch' in failure['error'], failure['error']
    result = {
        'status': 'PASS', 'physics_steps': 0, 'runtime_binary_sha256': digest(executable),
        'component_selftest': json.loads(first.read_text())['checks'],
        'existing_output_rejected_without_overwrite': True,
        'existing_report_sha256_before_and_after': before,
        'wrong_schema_hash_rejected_before_model_load': True,
        'scope': 'Production CLI and pure component paths only; no mjModel, ROS node or integration',
    }
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
