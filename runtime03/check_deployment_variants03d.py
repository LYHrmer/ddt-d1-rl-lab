#!/usr/bin/env python3
"""Check 03d identity routing and prior real-export print CLIs, without physics."""
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
    protected = [ROOT/'runtime03/build/d1_runtime03', ROOT/'runtime03/src/main.cpp',
                 ROOT/'runtime03/include/runtime03/interface.hpp', ROOT/'runtime03/schema_pilot03.json',
                 ROOT/'runtime03/schema_pilot03b.json', ROOT/'runtime03/schema03c.json',
                 ROOT/'training/d1_rl/env.py', ROOT/'training/d1_rl/interface.py',
                 ROOT/'training/configs/interface03b.json', ROOT/'training/configs/pilot03b.json',
                 ROOT/'training/configs/pilot03c.json', ROOT/'training/configs/pilot03d.json']
    before = {str(path): sha(path) for path in protected}
    prior = json.loads((ROOT/'runtime03/schema03c.json').read_text())
    version = 'd1-pure-ppo-pilot-03d'
    # In-memory routing input only: this is not a 03d export or inference test.
    synthetic = {'config_version': version, 'interface_id': 'd1-selftrained-pilot03b-v2',
                 'sha256': {'config': VARIANTS[version]['config_sha256']}}
    config_path, schema_path, config, schema = select_deployment(synthetic)
    assert config_path == ROOT/'training/configs/pilot03d.json'
    assert schema_path == ROOT/'runtime03/schema03d.json'
    assert config['version'] == version
    differences = {key for key in prior if prior[key] != schema[key]}
    assert set(prior) == set(schema)
    assert differences == {'training_config', 'training_config_sha256'}
    failures = []
    for name, changes in [('missing_version', {'config_version': None}),
                          ('unknown_version', {'config_version': 'unqualified-deployment-version'}),
                          ('wrong_config_hash', {'sha256': {'config': '0'*64}}),
                          ('03c_hash_with_03d_version', {'sha256': {'config': VARIANTS['d1-pure-ppo-pilot-03c']['config_sha256']}}),
                          ('wrong_interface', {'interface_id': 'not-qualified'})]:
        metadata = copy.deepcopy(synthetic)
        metadata.update(changes)
        try:
            select_deployment(metadata)
        except ValueError:
            failures.append(name)
        else:
            raise AssertionError(f'Expected rejection: {name}')
    real_exports = []
    for variant, expected_schema in [('03b', 'schema_pilot03b.json'), ('03c', 'schema03c.json')]:
        export = ROOT/f'evidence/career_contract_03/pilot{variant}_export_01'
        target = (args.output/f'{variant}_must_not_be_created').resolve()
        command = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(ROOT/'tools/run_selftrained03.py'),
                   '--export', str(export), '--output', str(target), '--print-command']
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
        (args.output/f'{variant}_cli_print.stdout').write_text(result.stdout)
        (args.output/f'{variant}_cli_print.stderr').write_text(result.stderr)
        assert result.returncode == 0 and not target.exists(), result.stdout+result.stderr
        assert str(ROOT/'runtime03'/expected_schema) in result.stdout
        real_exports.append({'variant': variant, 'metadata_sha256': sha(export/'metadata.json'),
                             'printed_only': True, 'native_process_launched': False})
    for path in [ROOT/'runtime03/prepare_pilot_deployment03.py', ROOT/'tools/run_selftrained03.py']:
        result = subprocess.run(['rtk', 'proxy', '/usr/bin/python3', '-B', str(path), '--help'],
                                text=True, capture_output=True, timeout=30)
        assert result.returncode == 0 and '03d' in result.stdout
    after = {str(path): sha(path) for path in protected}
    assert before == after
    report = {'status': 'PASS', 'native_steps': 0, 'policy_inference_calls': 0,
              '03d_check_kind': 'synthetic in-memory metadata routing, not an exported policy qualification',
              '03d_config_sha256': sha(config_path), '03d_runtime_schema_sha256': sha(schema_path),
              '03d_schema_only_changed_fields': sorted(differences),
              'metadata_rejection_cases': failures, 'actual_prior_export_print_cli_checks': real_exports,
              'launcher_and_preparer_help_passed': True,
              'protected_sources_and_binary_sha256_before': before,
              'protected_sources_and_binary_sha256_after': after,
              'protected_sources_and_binary_unchanged': True,
              'tool_sha256': {str(path): sha(path) for path in [ROOT/'tools/run_selftrained03.py',
                  ROOT/'runtime03/deployment_variants.py', ROOT/'runtime03/prepare_pilot_deployment03.py',
                  Path(__file__).resolve()]},
              'limitation': 'Real 03d policy export, C++ ORT golden inference, physical comparison and deployment remain root-owned later gates.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('status', 'native_steps', 'policy_inference_calls',
          '03d_config_sha256', '03d_runtime_schema_sha256', 'metadata_rejection_cases')}, indent=2))


if __name__ == '__main__':
    main()
