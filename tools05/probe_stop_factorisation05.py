#!/usr/bin/env python3
"""Contract 05M: split the post-cut decay time constant into its factors.

Replays the frozen forward_stop course for the exported 04b policy through the
same environment code path as the evaluator, and from the cut tick onward forces
the four wheel actions to zero while the legs keep following the policy. If tau
collapses, the policy's own wheel action is the lever; if it does not, the lever
is somewhere else. See docs/career_execution_contract_05m.md.

Pure helpers sit at the top and import no simulator; the run code imports the
environment and onnxruntime lazily, so the helpers can be tested with no model
and no physics. Real runs consume native steps and need a release receipt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training'))
sys.path.insert(0, str(ROOT/'tools04'))

SEED = 7001
EPISODE_TICKS = 600
CUT_TICK = 400
NATIVE_PER_TICK = 10
WHEEL_ACTION_INDEX = (3, 7, 11, 15)
COMMAND_LIMIT = .3

# Thresholds and tolerances are frozen by the release receipt; see contract section 5.
P1_TOLERANCE = 1e-6
P2_SUPPORTED_AT_OR_BELOW_S = .29
P2_REFUTED_AT_OR_ABOVE_S = .41
P3_DISPLACEMENT_BELOW_M = .05
P4_RATIO_RANGE = (.75, 1.25)
P5_RATIO_RANGE = (.8, 1.2)
P4_ENTRY_RATIO_EXCLUDED = (.8, 1.25)     # B/A entry speeds this close cannot test linearity
TAU_POLICY_ARM_S = .5804338711860841     # analysis receipt, 04b forward, policy wheel actions
POOL_CAP_NATIVE = 18020

FROZEN_COMPARISON = ('evidence/career_contract_04/pilot04b_python_eval_01/'
                     'forward_stop_delay0_seed7001.npz')
FROZEN_INPUTS = {
    'training/configs/pilot04b.json':
        '9d855063fb654f1d9bf0ae4d392f7f70294c328c5e944ef79f8e93902fb82e0a',
    'evidence/career_contract_04/pilot04b_export_01/policy.onnx':
        'f8328e286a4d566422f32d85f8cb58b680d27b442b42d085dd2f51c98a4b5c8b',
    FROZEN_COMPARISON:
        '173b0905597b48f56ad0468d76e2fc55762e6b3e4f35017ed8aafd95e4860169',
    'tools/evaluate_d1_ppo.py':
        '47e5b3dd2a806b958c1be5d24a46a54848e6a4981401a07f40b0be3a4107f121',
    'training/d1_rl/factory04b.py':
        '7513d7b4db65a28368b47f39cdd15fea0951d22fe668c30d91fc23edf699f808',
    'tools04/analyze_cut_to_zero04.py':
        '7e3ac5048ff190b83add88ae89588742b5588663403311888512427542d31b69',
}
TRACE_KEYS = ('reward', 'command_during_interval', 'body_vx_m_s', 'body_vy_m_s',
              'body_wz_rad_s', 'base_x_m', 'base_y_m', 'base_height_m', 'base_roll_rad',
              'base_pitch_rad', 'base_yaw_rad', 'tilt_rad', 'fallen',
              'illegal_body_contact', 'tau_saturation_count', 'issued_action', 'applied_action')


def build_sequence(drive_vx, drive_start_s=2., cut_s=8.):
    if abs(drive_vx) > COMMAND_LIMIT:
        raise ValueError('drive command exceeds the frozen +-0.3 limit')
    return [{'start_s': 0., 'vx': 0., 'wz': 0.},
            {'start_s': float(drive_start_s), 'vx': float(drive_vx), 'wz': 0.},
            {'start_s': float(cut_s), 'vx': 0., 'wz': 0.}]


def command_at_tick(sequence, tick, tick_s=.02):
    """The command the environment applies on a decision tick (same rule as env._update_command)."""
    elapsed = tick*tick_s
    chosen = sequence[0]
    for candidate in sequence:
        if candidate['start_s'] <= elapsed+1e-12:
            chosen = candidate
        else:
            break
    return chosen['vx'], chosen['wz']


def cut_tick_of(sequence, ticks, tick_s=.02):
    """First tick at which a moving command has just become zero; None when there is none."""
    previous = command_at_tick(sequence, 0, tick_s)
    for tick in range(1, ticks):
        current = command_at_tick(sequence, tick, tick_s)
        if previous != (0., 0.) and current == (0., 0.):
            return tick
        previous = current
    return None


RUNS = {
    'smoke': {'ticks': 2, 'cut_tick': 1, 'zero_wheels': True,
              'sequence': [{'start_s': 0., 'vx': .2, 'wz': 0.}, {'start_s': .02, 'vx': 0., 'wz': 0.}],
              'purpose': 'receipt-writing path only; no physical conclusion'},
    'A': {'ticks': EPISODE_TICKS, 'cut_tick': CUT_TICK, 'zero_wheels': True,
          'sequence': build_sequence(.2), 'frozen_comparison': FROZEN_COMPARISON,
          'purpose': 'main measurement: wheel actions forced to zero after the cut'},
    'B': {'ticks': EPISODE_TICKS, 'cut_tick': CUT_TICK, 'zero_wheels': True,
          'sequence': build_sequence(.3),
          'purpose': 'linearity: same arm at the higher drive command'},
    'C': {'ticks': EPISODE_TICKS, 'cut_tick': CUT_TICK, 'zero_wheels': False,
          'sequence': build_sequence(.3),
          'purpose': 'matched control for B: the policy keeps its own wheel actions'},
}


def planned_native(name):
    return RUNS[name]['ticks']*NATIVE_PER_TICK


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def force_wheels_zero(action, tick, cut_tick, enabled):
    """Executed action: wheel entries zero from the cut tick on, everything else untouched."""
    out = np.array(action, dtype=np.float32, copy=True)
    if enabled and tick >= cut_tick:
        out[list(WHEEL_ACTION_INDEX)] = 0.
    return out


def check_frozen_inputs(frozen, root=ROOT):
    """frozen maps a path (relative to root or absolute) to the SHA-256 it must have."""
    for path, expected in frozen.items():
        resolved = Path(path) if Path(path).is_absolute() else Path(root)/path
        if not resolved.is_file():
            raise ValueError(f'frozen input is missing: {path}')
        if sha(resolved) != expected:
            raise ValueError(f'frozen input changed: {path}')


def fidelity(new, frozen, cut_tick=CUT_TICK, tolerance=P1_TOLERANCE):
    """P1: the replay must equal the frozen evaluation trace on every tick before the cut."""
    result = {'ticks_compared': cut_tick, 'tolerance': tolerance}
    worst = {}
    for key in ('body_vx_m_s', 'base_x_m', 'base_y_m'):
        a, b = np.asarray(new[key], dtype=float), np.asarray(frozen[key], dtype=float)
        if len(a) < cut_tick or len(b) < cut_tick:
            result.update(passed=False, reason=f'{key} shorter than the cut tick')
            return result
        worst[key] = float(np.abs(a[:cut_tick]-b[:cut_tick]).max())
    result['max_abs_difference'] = worst
    result['passed'] = bool(all(v <= tolerance for v in worst.values()))
    return result


def classify_p2(tau_s):
    if tau_s is None:
        return 'UNMEASURABLE'
    if tau_s <= P2_SUPPORTED_AT_OR_BELOW_S:
        return 'SUPPORTED'
    if tau_s >= P2_REFUTED_AT_OR_ABOVE_S:
        return 'REFUTED'
    return 'UNCERTAIN'


def _tau(analysis):
    return (analysis or {}).get('decay_time_constant_s', {}).get('ticks_10_to_50')


def _in(value, bounds):
    return value is not None and bounds[0] <= value <= bounds[1]


def entry_speed_ratio(a, b):
    """Entry speed of run B over run A, or None when either is missing."""
    va, vb = (a or {}).get('entry_speed_m_s'), (b or {}).get('entry_speed_m_s')
    return vb/va if va and vb is not None else None


def p4_precondition(a, b):
    """P4 asks whether tau depends on speed, so it needs two clearly different entry speeds."""
    ratio = entry_speed_ratio(a, b)
    if ratio is None:
        return {'entry_speed_ratio_B_over_A': None, 'valid': False, 'reason': 'entry speed missing'}
    low, high = P4_ENTRY_RATIO_EXCLUDED
    return {'entry_speed_ratio_B_over_A': ratio, 'valid': not (low <= ratio <= high),
            'reason': None if not (low <= ratio <= high) else
            f'entry speeds differ by less than the frozen margin [{low}, {high}]'}


def judge(a, b=None, c=None, fidelity_passed=True):
    """Verdicts for P1..P5 from the per-run analyses. Missing inputs are NOT_EVALUABLE."""
    out = {'P1': 'MET' if fidelity_passed else 'NOT_MET'}
    if not fidelity_passed:
        out.update({k: 'NOT_EVALUABLE' for k in ('P2', 'P3', 'P4', 'P5')})
        return out
    out['P2'] = classify_p2(_tau(a))
    if a is None or a.get('ticks_after_cut_available', 0) < EPISODE_TICKS-CUT_TICK:
        out['P3'] = 'NOT_MET' if a is not None else 'NOT_EVALUABLE'
    else:
        out['P3'] = ('MET' if a['final_euclidean_displacement_m'] < P3_DISPLACEMENT_BELOW_M
                     else 'NOT_MET')
    tau_a, tau_b = _tau(a), _tau(b)
    if tau_a and tau_b and p4_precondition(a, b)['valid']:
        out['P4'] = 'MET' if _in(tau_b/tau_a, P4_RATIO_RANGE) else 'NOT_MET'
    else:
        out['P4'] = 'NOT_EVALUABLE'
    ratios = [(r or {}).get('first_order_stop', {}).get('predicted_over_path') for r in (a, b)]
    if all(r is not None for r in ratios):
        out['P5'] = 'MET' if all(_in(r, P5_RATIO_RANGE) for r in ratios) else 'NOT_MET'
    else:
        out['P5'] = 'NOT_EVALUABLE'
    return out


def interpretation(p2, p3):
    if p2 == 'SUPPORTED' and p3 == 'MET':
        return 'wheel action is the lever and zeroing it is enough to pass the displacement gate'
    if p2 == 'SUPPORTED' and p3 == 'NOT_MET':
        return 'wheel action is a lever but not sufficient; entry speed or active braking also matter'
    if p2 == 'REFUTED':
        return 'wheel action is not the main lever; look at leg and body dynamics'
    return 'inconclusive; no mechanism conclusion and no basis for a training contract'


def require_passed(report_path, what):
    """Order gate: a later run needs an earlier receipt that exists and passed."""
    path = Path(report_path)
    if not path.is_file():
        raise SystemExit(f'{what}: required receipt is missing: {path}')
    report = json.loads(path.read_text())
    if report.get('status') != 'COMPLETE':
        raise SystemExit(f'{what}: required receipt did not complete')
    if what == 'A' and report.get('run') != 'smoke':
        raise SystemExit('A requires the smoke receipt')
    if what in ('B', 'C') and (report.get('run') != 'A' or not report.get('fidelity', {}).get('passed')):
        raise SystemExit(f'{what} requires a run A receipt whose fidelity gate (P1) passed')
    return report


def verify_release(release_path, expected=None, root=ROOT):
    """The run only starts if the release receipt matches the inputs and this very tool."""
    expected = FROZEN_INPUTS if expected is None else expected
    release = json.loads(Path(release_path).read_text())
    for path, digest in expected.items():
        if release['frozen_inputs'].get(path) != digest:
            raise SystemExit(f'release receipt does not freeze the expected input: {path}')
    check_frozen_inputs(release['frozen_inputs'], root)
    if release['tool_sha256'] != sha(__file__):
        raise SystemExit('this tool differs from the one frozen in the release receipt')
    return release


def assemble_trace(trace, policy_action, positions):
    """Arrays in the layout analyze_cut_to_zero04.analyse_trace reads.

    `issued_action` is what the probe handed to env.step, as in the evaluator, and the
    analyser reads the wheel action from it; `applied_action` is what the environment
    says it applied; what the policy wanted is kept apart.
    """
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    arrays['policy_action_raw'] = np.asarray(policy_action)
    arrays['boundary_xy_m'] = np.asarray(positions)
    arrays['policy_interval_start_s'] = np.arange(len(trace['reward']))*.02
    return arrays


def override_effect(arrays, cut_tick, zero_wheels, tolerance=1e-6):
    """Did the override actually reach the robot, judged from what the environment recorded?"""
    wheels = list(WHEEL_ACTION_INDEX)
    n = len(arrays['applied_action'])
    if n <= cut_tick:
        return {'checked_ticks': 0, 'effective': False, 'reason': 'no tick at or after the cut'}
    applied = np.asarray(arrays['applied_action'], dtype=float)[cut_tick:, wheels]
    wanted = np.asarray(arrays['policy_action_raw'], dtype=float)[cut_tick:, wheels]
    if zero_wheels:
        zero = bool(np.all(np.abs(applied) <= tolerance))
        wanted_nonzero = bool(np.any(np.abs(wanted) > tolerance))
        return {'checked_ticks': int(len(applied)), 'applied_wheels_zero_after_cut': zero,
                'policy_wanted_nonzero_after_cut': wanted_nonzero,
                'effective': zero and wanted_nonzero}
    untouched = bool(np.all(np.abs(applied-wanted) <= tolerance))
    return {'checked_ticks': int(len(applied)), 'applied_wheels_match_policy': untouched,
            'effective': untouched}


def execute_run(name, onnx, config_path, factory, output, native_budget, seed=SEED):
    spec = RUNS[name]
    planned = planned_native(name)
    if planned > native_budget:
        raise SystemExit(f'planned {planned} native steps exceeds the {native_budget} remaining')
    output.mkdir(parents=True, exist_ok=False)     # before any physics, so a crash leaves a trace

    import importlib
    import onnxruntime as ort
    from d1_rl.interface import ACTOR_DIM, OBS_DIM
    environment_class = importlib.import_module(factory).environment_class

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx), sess_options=options,
                                   providers=['CPUExecutionProvider'])

    def predict(observation):
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (ACTOR_DIM,) or not np.all(np.isfinite(vector)):
            raise ValueError('malformed actor observation')
        result = session.run(['actions'], {'obs': vector[:OBS_DIM].reshape(1, OBS_DIM),
                                           'history': vector[OBS_DIM:].reshape(1, 10, OBS_DIM)})[0]
        return np.clip(result[0], -1., 1.).astype(np.float32)

    config = Path(config_path).resolve()
    env = environment_class(config)(config=config, arm='B', seed=seed, rank=0,
                                    max_transitions=EPISODE_TICKS, eval_delay_ticks=0,
                                    eval_command_sequence=spec['sequence'])
    trace = {key: [] for key in TRACE_KEYS}
    policy_action = []
    positions = []
    status, error, counts, reason = 'COMPLETE', None, None, 'HORIZON'
    try:
        observation, _ = env.reset(seed=seed)
        positions.append([float(env.data.qpos[env.free_qpos]), float(env.data.qpos[env.free_qpos+1])])
        for tick in range(spec['ticks']):
            raw = predict(observation)
            executed = force_wheels_zero(raw, tick, spec['cut_tick'], spec['zero_wheels'])
            observation, reward, terminated, truncated, info = env.step(executed)
            item = {'reward': reward, 'issued_action': executed, **info}
            for key in TRACE_KEYS:
                trace[key].append(item[key])
            policy_action.append(raw)
            positions.append([float(info['base_x_m']), float(info['base_y_m'])])
            if terminated or truncated:
                reason = 'TERMINATED' if terminated else 'TRUNCATED'
                break
    except BaseException as exc:                    # keep the partial trace and the counters
        status, error = 'CRASHED', f'{type(exc).__name__}: {exc}'
    finally:
        try:
            counts = env.counters()
        finally:
            env.close()

    arrays = assemble_trace(trace, policy_action, positions)
    trace_path = output/'trace.npz'
    with trace_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)

    report = {'format': 'd1-contract05m-stop-factorisation-run-v1', 'run': name,
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'spec': {k: spec[k] for k in ('ticks', 'cut_tick', 'zero_wheels', 'sequence', 'purpose')},
              'seed': seed, 'status': status, 'error': error, 'done_reason': reason,
              'planned_native_steps': planned, 'actual_counts': counts,
              'onnx_sha256': sha(onnx), 'config_sha256': sha(config),
              'trace': trace_path.name, 'trace_sha256': sha(trace_path),
              'tool_sha256': sha(__file__)}
    if counts is not None and counts['native_steps'] != NATIVE_PER_TICK*counts['policy_transitions']:
        report['status'] = 'COUNTERS_INCONSISTENT'
    if status == 'COMPLETE':
        report['override'] = override_effect(arrays, spec['cut_tick'], spec['zero_wheels'])
        if not report['override']['effective'] and report['status'] == 'COMPLETE':
            report['status'] = 'OVERRIDE_NOT_EFFECTIVE'
    if status == 'COMPLETE' and spec.get('frozen_comparison'):
        with np.load(ROOT/spec['frozen_comparison']) as frozen:
            report['fidelity'] = fidelity(arrays, {k: frozen[k] for k in frozen.files})
    if status == 'COMPLETE' and name != 'smoke':
        from analyze_cut_to_zero04 import analyse_trace
        report['analysis'] = analyse_trace(trace_path, None)
    (output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('name', choices=sorted(RUNS))
    run.add_argument('--release', type=Path, required=True)
    run.add_argument('--onnx', type=Path, required=True)
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--factory', default='d1_rl.factory04b')
    run.add_argument('--native-budget', type=int, required=True, help='remaining 05M pool')
    run.add_argument('--smoke-receipt', type=Path, help='required for A')
    run.add_argument('--a-receipt', type=Path, help='required for B and C')
    run.add_argument('--output', type=Path, required=True)
    jud = sub.add_parser('judge')
    jud.add_argument('--a', type=Path, required=True)
    jud.add_argument('--b', type=Path)
    jud.add_argument('--c', type=Path)
    jud.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    if args.command == 'run':
        release = verify_release(args.release)
        if release.get('onnx_sha256') != sha(args.onnx) or release.get('config_sha256') != sha(args.config):
            raise SystemExit('policy or config differs from the release receipt')
        if args.name == 'A':
            require_passed(args.smoke_receipt or '', 'A')
        if args.name in ('B', 'C'):
            require_passed(args.a_receipt or '', args.name)
        report = execute_run(args.name, args.onnx, args.config, args.factory,
                             args.output, args.native_budget)
        print(json.dumps({'run': args.name, 'status': report['status'],
                          'native_steps': (report['actual_counts'] or {}).get('native_steps'),
                          'fidelity': report.get('fidelity', {}).get('passed')}, indent=2))
        return 0 if report['status'] == 'COMPLETE' else 1

    reports = {k: json.loads((getattr(args, k)/'report.json').read_text())
               for k in ('a', 'b', 'c') if getattr(args, k)}
    analyses = {k: v.get('analysis') for k, v in reports.items()}
    verdicts = judge(analyses['a'], analyses.get('b'), analyses.get('c'),
                     fidelity_passed=bool(reports['a'].get('fidelity', {}).get('passed')))
    tau_c = _tau(analyses.get('c'))
    verdict = {'format': 'd1-contract05m-verdict-v1',
               'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'predictions': verdicts,
               'interpretation': interpretation(verdicts['P2'], verdicts['P3']),
               'p4_precondition': p4_precondition(analyses['a'], analyses.get('b')),
               'entry_speed_m_s': {k: (analyses.get(k) or {}).get('entry_speed_m_s') for k in ('a', 'b', 'c')},
               'tau_s': {'A_forced_zero_cmd0p2': _tau(analyses['a']), 'policy_arm_cmd0p2': TAU_POLICY_ARM_S,
                         'B_forced_zero_cmd0p3': _tau(analyses.get('b')),
                         'C_policy_arm_cmd0p3': tau_c},
               'native_steps_used': {k: (v['actual_counts'] or {}).get('native_steps')
                                     for k, v in reports.items()},
               'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'verdict.json').write_text(json.dumps(verdict, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
