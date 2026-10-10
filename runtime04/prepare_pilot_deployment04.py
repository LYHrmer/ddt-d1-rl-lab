#!/usr/bin/env python3
"""Prepare the three 04a independent C++ replay trials; execute no trial.

Differences from runtime03/prepare_pilot_deployment03.py, which is referenced
verbatim by the 03 receipts and therefore not edited:

  * selection goes through runtime04/deployment_variants04.py so 04a resolves;
  * all three qualification tasks get their own replay trial, because the 04a
    hypothesis is about stop displacement on every task and 03d only scored
    forward and reverse in C++;
  * no live watchdog slot. Per docs/career_execution_contract_04a.md the live
    trial is registered separately, after the flat three-task gate and the
    replay trials pass, and must launch through runtime04/launch_env04.sh.

The planner integrates nothing. Each native trial still has to be released and
counted individually.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'runtime04'))

from deployment_variants04 import select_deployment  # noqa: E402

EXPECTED_BINARY = '62134e7c94fafaefcdee9f8325191b8f621cb3eb51f9ed2a167d68198469975d'
TASKS = ('forward_stop', 'reverse_stop', 'turn_stop')
SECONDS = 12
MAX_NATIVE_PER_TRIAL = 6000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--onnx-golden-report', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path,
                        default=ROOT/'runtime03/fresh_initial_state03b_seed7001_warmstart.json')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    export, initial, output = args.export.resolve(), args.initial_state.resolve(), args.output.resolve()
    policy = export/'policy.onnx'
    metadata, golden = load(export/'metadata.json'), load(args.onnx_golden_report)
    config_path, schema_path, config, schema = select_deployment(metadata)
    binary = ROOT/'runtime03/build/d1_runtime03'
    state = load(initial)
    model = ROOT/config['scene']

    if sha(binary) != EXPECTED_BINARY:
        raise ValueError('Frozen native runtime changed; requalify before a new plan')
    if metadata['config_version'] not in ('d1-pure-ppo-pilot-04a', 'd1-pure-ppo-pilot-04b'):
        raise ValueError('This planner is only for the contract 04 exports')
    if metadata['interface_id'] != 'd1-selftrained-pilot03b-v2' or metadata['training_kind'] != 'pilot':
        raise ValueError('Plan requires a self-trained contract 04 pilot export on the frozen interface')
    if (metadata['teacher_policy_used'] is not False or metadata['random_initialization'] is not True or
            metadata['action_scales'] != schema['action_scales']):
        raise ValueError('Expected random-initialized, no-teacher PPO with the frozen 03b action scales')
    for key, path in [('policy_onnx', policy), ('config', config_path), ('scene_xml', model),
                      ('interface_schema', Path(schema['training_interface_schema'])),
                      ('robot_xml', Path(schema['robot_xml']))]:
        if metadata['sha256'][key] != sha(path):
            raise ValueError(f'Export identity mismatch: {key}')
    if (golden['status'] != 'PASS' or golden['policy_sha256'] != sha(policy) or
            golden['onnxruntime_version'] != '1.10.0' or golden['onnx_inference_calls'] != 64 or
            golden['max_absolute_error'] > 1e-5):
        raise ValueError('Actual C++ ORT 1.10 golden gate is not satisfied')
    if (state['scene_xml_sha256'] != sha(model) or
            state['interface_schema_sha256'] != sha(Path(schema['training_interface_schema'])) or
            state['episode_policy_tick'] != 0 or state['simulation_time_s'] != 0 or
            len(state['qacc_warmstart']) != 22 or state['delay_ticks'] != 0):
        raise ValueError('Initial state must be the frozen fresh reset with explicit warmstart')

    output.mkdir(parents=True, exist_ok=False)
    hashes = {'model': sha(model), 'policy': sha(policy), 'schema': sha(schema_path),
              'runtime': sha(binary)}
    env = ['rtk', 'proxy', 'env', 'ROS_DOMAIN_ID=84', 'ROS_LOCALHOST_ONLY=1',
           'LD_LIBRARY_PATH=/opt/ros/humble/lib:/home/lyh/ddt_ros2_ws/install/mujoco/lib',
           'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1']
    common = [str(binary), '--schema', str(schema_path), '--model', str(model),
              '--policy', str(policy),
              '--expected-model-sha256', hashes['model'],
              '--expected-policy-sha256', hashes['policy'],
              '--expected-schema-sha256', hashes['schema'],
              '--initial-state', str(initial), '--seed', '7001', '--delay-ticks', '0',
              '--seconds', str(SECONDS), '--max-native', str(MAX_NATIVE_PER_TRIAL)]
    expected_args = sum(([f'--expected-{key}-sha256', value] for key, value in hashes.items()), [])

    trials, commands = [], []
    for index, name in enumerate(TASKS, 1):
        run = output/f'trial{index:02}_{name}'
        sequence = ROOT/'runtime03/sequences03b'/(name+'.json')
        native = env+common+['--mode', 'replay', '--sequence', str(sequence), '--output', str(run)]
        scorer = ['rtk', 'proxy', '/usr/bin/python3', '-B',
                  str(ROOT/'tools/evaluate_runtime03_trace.py'), '--run', str(run),
                  '--sequence', name, '--schema', str(schema_path), *expected_args,
                  '--output', str(output/f'score{index:02}_{name}')]
        trials.append({'trial': index, 'name': name, 'seconds': SECONDS,
                       'max_native': MAX_NATIVE_PER_TRIAL, 'mode': 'replay',
                       'output': str(run), 'native_command': native,
                       'native_shell': shlex.join(native), 'sequence_sha256': sha(sequence)})
        commands.extend([{'kind': name+'_native_trial', 'argv': native,
                          'shell': shlex.join(native)},
                         {'kind': name+'_offline_scoring', 'argv': scorer,
                          'shell': shlex.join(scorer)}])

    manifest = {
        'status': 'PLANNED_NOT_EXECUTED',
        'physics_steps_executed_by_planner': 0,
        'contract': 'docs/career_execution_contract_04a.md',
        'config_version': metadata['config_version'],
        'training_config': str(config_path), 'runtime_schema': str(schema_path),
        'max_trials': len(TASKS),
        'aggregate_max_native_steps': MAX_NATIVE_PER_TRIAL*len(TASKS),
        'trials': trials,
        'policy_metadata_sha256': sha(export/'metadata.json'),
        'cxx_ort_golden_receipt_sha256': sha(args.onnx_golden_report),
        'initial_state_sha256': sha(initial), 'identities': hashes,
        'live_watchdog_in_this_plan': False,
        'live_watchdog_note': ('Registered separately after the flat three-task gate; it must '
                               'launch through runtime04/launch_env04.sh, which loads the full '
                               'ROS environment instead of hand-writing PYTHONPATH as the 03 '
                               'planner did. Import preflight passing is not watchdog validation.'),
        'release_condition': ('Root reviews the Python pilot evaluation and releases each native '
                              'run. This manifest alone launches and authorises nothing.'),
        'early_failure_rule': ('Every started native trial consumes its slot and its actual '
                               'native steps, even if the robot falls early. No automatic retries.'),
        'scoring_gate_unchanged': {'stop_displacement_max_m': .05,
                                   'last_second_speed_max_m_s': .02,
                                   'last_second_abs_wz_max_rad_s': .03,
                                   'peak_abs_roll_or_pitch_max_rad': .20},
    }
    (output/'plan.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    (output/'commands_for_review.txt').write_text(
        '\n\n'.join(item['kind']+'\n'+item['shell'] for item in commands)+'\n')
    print(json.dumps({'status': manifest['status'], 'max_trials': len(TASKS),
                      'max_native_total': manifest['aggregate_max_native_steps'],
                      'output': str(output), 'physics_steps': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
