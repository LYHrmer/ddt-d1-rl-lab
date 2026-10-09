#!/usr/bin/env python3
"""Zero-physics checks of version routing and the real 03b print-only CLI."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from deployment_variants import ROOT, VARIANTS, select_deployment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protected = [ROOT/'runtime03/build/d1_runtime03', ROOT/'runtime03/schema_pilot03b.json',
                 ROOT/'runtime03/schema_pilot03.json', ROOT/'runtime03/src/main.cpp',
                 ROOT/'runtime03/include/runtime03/interface.hpp',
                 ROOT/'training/configs/pilot03b.json', ROOT/'training/configs/pilot03c.json',
                 ROOT/'training/configs/interface03b.json']
    before = {str(path): sha(path) for path in protected}
    original = json.loads((ROOT/'evidence/career_contract_03/pilot03b_export_01/metadata.json').read_text())
    config_path, schema_path, config, schema = select_deployment(original)
    assert config['version'] == 'd1-pure-ppo-pilot-03b'
    assert schema_path == ROOT/'runtime03/schema_pilot03b.json'
    # This dictionary tests metadata routing only. It is not an export directory,
    # contains no ONNX policy, and is never supplied to a simulator or planner.
    synthetic = {'config_version': 'd1-pure-ppo-pilot-03c',
                 'interface_id': 'd1-selftrained-pilot03b-v2',
                 'sha256': {'config': VARIANTS['d1-pure-ppo-pilot-03c']['config_sha256']}}
    _, new_schema_path, new_config, new_schema = select_deployment(synthetic)
    assert new_schema_path == ROOT/'runtime03/schema03c.json'
    assert new_config['version'] == 'd1-pure-ppo-pilot-03c'
    differences = {key for key in schema if schema[key] != new_schema[key]}
    assert differences == {'training_config', 'training_config_sha256'}
    failures = []
    for label, change in [('missing_version', {'config_version': None}),
                          ('unknown_version', {'config_version': 'unqualified-deployment-version'}),
                          ('wrong_config_hash', {'sha256': {'config': '0'*64}}),
                          ('wrong_interface', {'interface_id': 'wrong'})]:
        changed = copy.deepcopy(synthetic)
        changed.update(change)
        try:
            select_deployment(changed)
        except ValueError:
            failures.append(label)
        else:
            raise AssertionError(f'Missing rejection: {label}')
    target = (args.output/'must_not_be_created').resolve()
    command = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(ROOT/'tools/run_selftrained03.py'),
               '--export', str(ROOT/'evidence/career_contract_03/pilot03b_export_01'),
               '--output', str(target), '--print-command']
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    (args.output/'03b_cli_print.stdout').write_text(result.stdout)
    (args.output/'03b_cli_print.stderr').write_text(result.stderr)
    assert result.returncode == 0 and not target.exists(), result.stdout+result.stderr
    assert str(ROOT/'runtime03/schema_pilot03b.json') in result.stdout
    help_result = subprocess.run(['rtk', 'proxy', '/usr/bin/python3', '-B',
                                  str(ROOT/'runtime03/prepare_pilot_deployment03.py'), '--help'],
                                 text=True, capture_output=True, timeout=30)
    assert help_result.returncode == 0
    after = {str(path): sha(path) for path in protected}
    assert before == after
    report = {'status': 'PASS', 'physics_steps': 0,
              '03b_actual_export_print_command_passed_without_native_launch': True,
              '03c_synthetic_metadata_routing_only': True,
              '03c_runtime_schema_sha256': sha(new_schema_path),
              '03c_schema_only_changed_fields': sorted(differences),
              'invalid_metadata_rejections': failures,
              'preparer_help_passed_no_plan_or_trial_created': True,
              'protected_sources_and_binary_unchanged': before,
              'tool_sha256': {str(p): sha(p) for p in [ROOT/'tools/run_selftrained03.py',
                  ROOT/'runtime03/deployment_variants.py', ROOT/'runtime03/prepare_pilot_deployment03.py']},
              'limitation': 'No 03c policy export existed for this test; real 03c ONNX validation and rollout remain root-owned later gates.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
