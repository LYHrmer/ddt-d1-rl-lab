#!/usr/bin/env python3
"""Zero-integration acceptance for contract 05M before any physics is spent.

Builds the real 04b environment exactly as the probe will, resets it, and checks the
things the probe copied or assumed against the real code: the command schedule
(by driving the environment's own _update_command, no stepping), the wheel action
layout, the ONNX signature, the frozen inputs, and that not one native step ran.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training'))
sys.path.insert(0, str(ROOT/'tools05'))

import probe_stop_factorisation05 as probe  # noqa: E402

CONFIG = ROOT/'training/configs/pilot04b.json'
ONNX = ROOT/'evidence/career_contract_04/pilot04b_export_01/policy.onnx'


def schedule_of(env, ticks):
    """The command the environment itself applies on each decision tick."""
    out = []
    for tick in range(ticks):
        env._episode_steps = tick
        env._update_command()
        out.append((float(env._command[0]), float(env._command[2])))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    checks = []

    def record(name, ok, detail):
        checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})

    try:
        probe.check_frozen_inputs(probe.FROZEN_INPUTS)
        record('frozen_inputs_match', True, {'inputs': len(probe.FROZEN_INPUTS)})
    except ValueError as exc:
        record('frozen_inputs_match', False, str(exc))

    from d1_rl.factory04b import environment_class
    from d1_rl.interface import ACTION_DIM, JOINT_NAMES, WHEEL_INDICES

    record('wheel_indices_are_the_four_foot_joints',
           list(probe.WHEEL_ACTION_INDEX) == WHEEL_INDICES.tolist()
           and all(JOINT_NAMES[i].endswith('_foot_joint') for i in WHEEL_INDICES)
           and ACTION_DIM == 16,
           {'indices': WHEEL_INDICES.tolist(), 'names': [JOINT_NAMES[i] for i in WHEEL_INDICES]})

    with np.load(ROOT/probe.FROZEN_COMPARISON) as frozen:
        frozen_command = np.asarray(frozen['command_during_interval'])
        frozen_trace = {k: frozen[k] for k in frozen.files}

    evaluator_source = (ROOT/'tools/evaluate_d1_ppo.py').read_text()
    found = re.search(r'^EPISODE_TICKS\s*=\s*(\d+)\s*$', evaluator_source, re.MULTILINE)
    record('evaluator_loop_length_equals_the_probe_loop_length',
           bool(found) and int(found.group(1)) == probe.EPISODE_TICKS,
           {'evaluator_EPISODE_TICKS': int(found.group(1)) if found else None,
            'probe_EPISODE_TICKS': probe.EPISODE_TICKS})

    counters = {}
    for name, spec in probe.RUNS.items():
        env = environment_class(CONFIG)(config=CONFIG, arm='B', seed=probe.SEED, rank=0,
                                        max_transitions=probe.EPISODE_TICKS, eval_delay_ticks=0,
                                        eval_command_sequence=spec['sequence'])
        try:
            env.reset(seed=probe.SEED)
            real = schedule_of(env, spec['ticks'])
            copied = [probe.command_at_tick(spec['sequence'], k) for k in range(spec['ticks'])]
            mismatches = [k for k, (a, b) in enumerate(zip(real, copied)) if a != b]
            record(f'schedule_{name}_matches_environment', not mismatches,
                   {'ticks': spec['ticks'], 'first_mismatches': mismatches[:5]})
            if name == 'A':
                same = [bool(np.allclose(frozen_command[k][[0, 2]], real[k], atol=0))
                        for k in range(spec['ticks'])]
                record('schedule_A_matches_frozen_evaluation_trace', all(same),
                       {'ticks': spec['ticks'], 'mismatches': [k for k, ok in enumerate(same) if not ok][:5]})
                record('environment_does_not_truncate_before_the_probe_loop_ends',
                       env.episode_ticks >= probe.EPISODE_TICKS and env.eval_delay_ticks == 0,
                       {'environment_episode_ticks': env.episode_ticks,
                        'probe_loop_ticks': probe.EPISODE_TICKS, 'eval_delay_ticks': env.eval_delay_ticks})
            counters[name] = env.counters()
        finally:
            env.close()
    record('no_native_integration_during_checks',
           all(c['native_steps'] == 0 and c['policy_transitions'] == 0 for c in counters.values()),
           counters)

    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ONNX), sess_options=options, providers=['CPUExecutionProvider'])
    signature = ([(x.name, x.shape, x.type) for x in session.get_inputs()],
                 [(x.name, x.shape, x.type) for x in session.get_outputs()])
    record('onnx_signature_is_obs_history_to_actions',
           signature == ([('obs', [1, 57], 'tensor(float)'), ('history', [1, 10, 57], 'tensor(float)')],
                         [('actions', [1, 16], 'tensor(float)')]), signature)

    self_fidelity = probe.fidelity(frozen_trace, frozen_trace)
    record('fidelity_gate_accepts_the_frozen_trace_against_itself', self_fidelity['passed'], self_fidelity)
    record('cut_tick_matches_the_frozen_trace',
           probe.cut_tick_of(probe.RUNS['A']['sequence'], probe.EPISODE_TICKS) == probe.CUT_TICK
           == int(np.flatnonzero(np.any(frozen_command[:-1] != 0, axis=1)
                                 & ~np.any(frozen_command[1:] != 0, axis=1))[-1])+1, {})
    record('budget_adds_up',
           sum(probe.planned_native(k) for k in probe.RUNS) == probe.POOL_CAP_NATIVE,
           {'planned': {k: probe.planned_native(k) for k in probe.RUNS}, 'cap': probe.POOL_CAP_NATIVE})

    failed = [c['check'] for c in checks if c['status'] == 'FAIL']
    report = {'format': 'd1-contract05m-static-checks-v1',
              'status': 'PASS' if not failed else 'FAIL',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'native_integration_steps_executed': 0, 'transitions_consumed': 0,
              'training_started': False,
              'tool_sha256': probe.sha(__file__), 'probe_sha256': probe.sha(probe.__file__),
              'failed_checks': failed, 'checks': checks}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'failed_checks': failed, 'checks': len(checks),
                      'native_integration_steps_executed': 0,
                      'output': str(args.output/'report.json')}, indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
