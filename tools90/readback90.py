"""Verify saved targets, clock boundaries, residuals, and all 180 identities without stepping or solving."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tools90.launch90 import BASE, PLAN, PLAN_SHA256, ROOT, sha
from training.d1_rl.interface import DEFAULT_Q, KD, KP, TORQUE_LIMIT, WHEEL_INDICES, inverse_rotate


def check(ok, label):
    if not ok:
        raise ValueError(label)


def read(path):
    return json.loads(path.read_bytes())


def stand_readback(folder, report, plan):
    from tools87.ppo87 import load_predictor
    from tools89.common89 import parent_folder, registered_plan

    prior = registered_plan()
    predictors = {seed: load_predictor(parent_folder(seed, prior)) for seed in plan['parent_seeds']}
    expected_ids = [(seed, arm) for seed in plan['reset_seeds'] for arm in plan['arms']]
    check([(c['reset_seed'], c['arm']) for c in report['cases']] == expected_ids, 'six registered stand cases')
    cut, horizon = plan['stand_ticks'], plan['ticks_per_episode']
    endpoints, rows_read = {}, 0
    for case in report['cases']:
        trace = folder / case['trace']
        check(sha(trace) == case['trace_sha256'], 'stand trace digest')
        with np.load(trace, allow_pickle=False) as a:
            initial = hashlib.sha256(b''.join(a[key].tobytes() for key in ('initial_qpos','initial_qvel','initial_obs')))
            check(initial.hexdigest() == case['initial_state_sha256'], 'initial pose, velocity, observation')
            n = len(a['issued'])
            check(n == case['ticks'] and case['pd_updates'] == 5*n and case['native_steps'] == 10*n, 'actual clocks')
            check(np.allclose(a['time_s'], np.arange(1, n+1)*.02, atol=1e-9, rtol=0), 'physical time')
            obs, raw, issued, applied = (a[k] for k in ('observation', 'policy_raw', 'issued', 'applied'))
            policy = predictors[case['parent_seed']]
            replay = np.stack([policy(o[:627], o[627:]) for o in obs])
            check(replay.tobytes() == raw.tobytes(), 'frozen policy action bytes')
            expected = raw.copy()
            wheels = raw[:, WHEEL_INDICES].astype(np.float64) * 5.
            if case['arm'] == 'common':
                for tick, u in enumerate(wheels[:cut]):
                    shift = min(max(0., -sum(u)/4.), 5.-max(u))
                    if shift:
                        expected[tick, WHEEL_INDICES] = (u+shift)/5.
            check(np.array_equal(issued, expected) and np.array_equal(applied[1:], issued[:-1]) and
                  np.all(applied[0] == 0), 'all issued targets and actual FIFO')
            check(np.array_equal(applied[:, WHEEL_INDICES] * np.full(4,5.), a['applied_wheel_target_rad_s']), 'PD target')
            commands = a['command']
            check(np.all(commands[:cut] == 0) and np.allclose(commands[cut:, 0], .2) and
                  np.all(commands[:, 1:] == 0), 'command window')
            x = int(a['free_qpos_index'])
            start = float(a['initial_qpos'][x])
            positions = a['qpos'][:, x]
            clean = n == horizon and not a['fallen'].any() and not a['illegal'].any()
            check(clean == case['clean'], 'stand full clean')
            if n >= cut:
                stop = float(positions[cut-1])
                check(max(0., start-stop) == case['rollback_m'], 'stand rollback')
                if n == horizon:
                    check(positions[-1] - stop == case['signed_moving_dx_m'], 'forward-command signed displacement')
            endpoints[(case['reset_seed'], case['arm'])] = (case['rollback_m'], clean, initial.hexdigest())
            rows_read += n
    independent_passes = []
    for seed, pair in zip(plan['reset_seeds'], report['pairs'], strict=True):
        (a, a_clean, a_hash), (b, b_clean, b_hash) = (endpoints[(seed, arm)] for arm in plan['arms'])
        reduction = None if a is None or b is None else a-b
        passed = a_clean and b_clean and reduction is not None and reduction >= plan['min_rollback_reduction_m']
        check(a_hash == b_hash and reduction == pair['rollback_reduction_m'] and passed == pair['passed'], 'matched pair gate')
        independent_passes.append(passed)
    check(all(independent_passes) == report['passed'], 'stand gate')
    return {'replayed_policy_frames':rows_read, 'replayed_action_components':rows_read*16, 'checked_pairs':len(independent_passes)}


def expected_rows(plan):
    return {(s['seed'], s['parent'], phase, row) for s in plan['sources'] for phase in ('flat','slope_stand','front_contact')
            for start, end in s[phase] for row in range(start,end+1)}


def verify_frames(frames, plan):
    """Re-score the exact projected servo map and hard residual receipt independently of solver status."""
    seen, rejected, overdue = set(), Counter(), []
    bypass_count = 0
    rows = []
    scale = np.tile([.5,.5,.5,5.], 4)
    gain = KP * scale
    gain[WHEEL_INDICES] = KD[WHEEL_INDICES] * scale[WHEEL_INDICES]
    for f in frames:
        seed, phase, i = f['seed'], f['phase'], f['source_row']
        identity = seed, f['parent_seed'], phase, i
        check(identity not in seen, 'duplicate frame identity')
        seen.add(identity)
        nominal = np.asarray(f['nominal_action'], dtype=np.float32)
        bypass = np.asarray((f['bypass'] or {}).get('action'), dtype=np.float32)
        identical = nominal.shape == (16,) and nominal.shape == bypass.shape and nominal.tobytes() == bypass.tobytes()
        check(identical == f['bypass_bitwise_equal'], 'bypass byte comparison')
        bypass_count += int(identical)
        state, attempt = f['estimated_state'], f['enabled']
        elapsed = f['elapsed_ms']
        check(np.isfinite(elapsed) and 0 <= f['estimate_ms'] <= elapsed and attempt.get('solve_ms', 0) <= elapsed+.002,
              'per-frame timing boundary')
        maximum = None
        if attempt.get('violations'):
            physical = dict(attempt['violations'])
            maximum = physical.pop('max')
            check(all(np.isfinite(list(physical.values()))) and max(physical.values()) == maximum, 'hard residual summary')
        if attempt['accepted']:
            q, dq = np.asarray(state['joint_q']), np.asarray(state['joint_dq'])
            action = np.asarray(attempt['action'], dtype=np.float32)
            check(action.shape == (16,) and np.isfinite(action).all() and np.abs(action).max() <= 1., 'returned servo action')
            bias = KP * (DEFAULT_Q-q) - KD*dq
            lo = np.clip(bias-gain, -TORQUE_LIMIT, TORQUE_LIMIT)
            hi = np.clip(bias+gain, -TORQUE_LIMIT, TORQUE_LIMIT)
            torque = np.clip(bias+gain*action.astype(np.float64), -TORQUE_LIMIT, TORQUE_LIMIT)
            check(np.allclose(torque, attempt['servo_torque_nm'], atol=1e-11, rtol=0) and
                  np.allclose(lo, attempt['reachable_lo_nm'], atol=1e-11, rtol=0) and
                  np.allclose(hi, attempt['reachable_hi_nm'], atol=1e-11, rtol=0), 'independent actuator image')
            check(maximum is not None and maximum <= plan['hard_tolerance'] and
                  np.max(np.abs(torque-attempt['qp_torque_nm'])) <= plan['hard_tolerance'], 'physical acceptance')
        else:
            check(attempt['action'] is None and bool(attempt['reason']), 'rejection must withhold targets')
            rejected[attempt['reason']] += 1
        if elapsed > plan['deadline_ms']:
            overdue.append({'seed':seed, 'phase':phase, 'source_row':i, 'elapsed_ms':elapsed})
        rows.append({'seed':seed, 'parent':f['parent_seed'], 'phase':phase, 'source_row':i, 'accepted':attempt['accepted'],
                     'solver_status':attempt['solver_status'], 'iterations':attempt.get('iterations'), 'reason':attempt['reason'],
                     'estimate_ms':f['estimate_ms'], 'elapsed_ms':elapsed, 'max_hard_residual':maximum,
                     'support':state['support'] if state else None,
                     'normal_sources':(f['estimate_evidence'] or {}).get('normal_sources')})
    check(seen == expected_rows(plan) and len(seen) == plan['max_frames'], 'exact registered identities')
    times = np.asarray([r['elapsed_ms'] for r in rows])
    return {'attempted':len(rows),'accepted':len(rows)-sum(rejected.values()),'bypass_bitwise_equal':bypass_count,
            'mechanical_pass':not rejected and bypass_count == len(rows),'timing_pass':not overdue,'deadline_violations':overdue,
            'rejections':dict(rejected),'elapsed_ms':dict(zip(('p50','p95','p99','max'), np.percentile(times,[50,95,99,100]).tolist())),
            'rows':rows}


def resource_receipt(stage):
    directory = BASE / f'guard_{stage}_01'
    guard = read(directory/'report.json')
    samples = [json.loads(line) for line in (directory/'resources.jsonl').read_text().splitlines()]
    started = [s for s in samples if s['stage'] == 'startup'][-1]
    check(guard['status'] == 'COMPLETE' and guard['owned_cgroup_children_empty'] and guard['returncode'] == 0,
          f'{stage} pressure guard')
    check(started['effective_available_bytes'] >= 4 << 30 and guard['minimum_effective_bytes'] >= 2 << 30 and
          not any(guard['final_memory_events'].values()), f'{stage} resource limits')
    return {k:guard[k] for k in ('elapsed_s','minimum_effective_bytes','peak_group_bytes','cpu_affinity','final_memory_events')} | {
        'guard_sha256':sha(directory/'report.json'),'admitted_effective_bytes':started['effective_available_bytes']}


def archived_truth_attitudes(frames, spec):
    """Get the floating base attitude at the pre-decision boundary, for scoring only."""
    poses = {}
    for source in spec['sources']:
        path = ROOT / spec['source_pattern'].format(seed=source['seed'])
        check(sha(path) == source['trace_sha256'], 'archived orientation source')
        with np.load(path, allow_pickle=False) as trace:
            poses[source['seed']] = trace['qpos'][:,3:7]
    return [poses[f['seed']][f['source_row']-1].tolist() if f['episode_tick'] else None for f in frames]


def verify_support_scores(frames, truth, attitudes):
    counts = Counter()
    unavailable = Counter()
    normal_errors, independent_errors, speed_errors = [], [], []
    for f, label, quat in zip(frames, truth, attitudes, strict=True):
        check(all(f[key] == label[key] for key in ('seed','parent_seed','phase','source_row','episode_tick')), 'truth join')
        score = label['score']
        if score is None:
            unavailable[label['unavailable_reason']] += 1
            continue
        if f['estimated_state'] is None:
            unavailable['no estimator state'] += 1
            continue
        estimate = f['estimated_state']
        for wheel, (claimed, loaded, normal) in enumerate(zip(estimate['support'], score['cold_support'],
                                                             score['cold_normals_world'], strict=True)):
            counts[bool(claimed), bool(loaded)] += 1
            if normal is not None:
                observed = inverse_rotate(quat, normal)
                estimated = inverse_rotate(estimate['base_quat_wxyz'], estimate['normals_world'][wheel])
                cosine = np.dot(observed, estimated)
                recorded = score['normal_error_deg'][wheel]
                check(abs(cosine-np.cos(np.deg2rad(recorded))) < 1e-13, 'normal comparison in body coordinates')
                independent_errors.append(float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(observed, estimated)), cosine))))
                normal_errors.append(recorded)
        speed_errors.append(f['estimate_evidence']['estimated_body_vx_m_s']-score['archived_body_vx_m_s'])
    return {'support_true_positive':counts[True,True], 'support_true_negative':counts[False,False],
            'support_false_positive':counts[True,False], 'support_false_negative':counts[False,True],
            'scored_frames':len(speed_errors),'unscored_frames':dict(unavailable),'scored_wheel_normals':len(normal_errors),
            'normal_angle_mean_deg':float(np.mean(normal_errors)) if normal_errors else None,
            'independent_body_normal_angle_mean_deg':float(np.mean(independent_errors)) if independent_errors else None,
            'body_vx_rmse_m_s':float(np.sqrt(np.mean(np.square(speed_errors)))) if speed_errors else None}


def publish_stream(path, compressed_path):
    with compressed_path.open('xb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:
            compressed.write(path.read_bytes())


def run(out: Path):
    import mujoco
    import osqp
    import scipy
    from tools90.plot90 import plot_stand
    from tools90.mechanics_check90 import check_solutions

    out.mkdir()
    plan = read(PLAN)
    stand_dir, shadow_dir = BASE/'stand_01', BASE/'shadow_01'
    stand, shadow = read(stand_dir/'report.json'), read(shadow_dir/'report.json')
    check(sha(PLAN) == PLAN_SHA256 and stand['protocol_sha256'] == shadow['protocol_sha256'] == PLAN_SHA256,
          'immutable preregistration')
    stand_checked = stand_readback(stand_dir, stand, plan['stand'])
    frame_file = shadow_dir/'frames.jsonl'
    truth_file = shadow_dir/'truth.jsonl'
    check(sha(frame_file) == shadow['frames_sha256'] and sha(truth_file) == shadow['truth_sha256'], 'frame and score hashes')
    frames = [json.loads(line) for line in frame_file.read_text().splitlines()]
    truth = [json.loads(line) for line in truth_file.read_text().splitlines()]
    wbc_checked = verify_frames(frames, plan['shadow'])
    for key in ('attempted','accepted','bypass_bitwise_equal','mechanical_pass','timing_pass'):
        check(wbc_checked[key] == shadow[key], f'shadow reported {key}')
    attitudes = archived_truth_attitudes(frames, plan['shadow'])
    support = verify_support_scores(frames, truth, attitudes)
    for key, value in support.items():
        if key in shadow['diagnostic']:
            recorded = shadow['diagnostic'][key]
            match = recorded is None if value is None else np.isclose(value, recorded, rtol=0, atol=1e-10)
            check(match, f'support diagnostic {key}')
    mechanics = check_solutions(frames)
    check(mechanics['accepted_rows'] == shadow['accepted'] and
          mechanics['numeric_rows'] + mechanics['skipped_non_numeric_rows'] == plan['shadow']['max_frames'],
          'independent mechanics denominator')
    physical = mechanics['accepted_within_tolerance']
    publish_stream(frame_file, out/'stand_wbc90_frames.jsonl.gz')
    publish_stream(truth_file, out/'stand_wbc90_truth.jsonl.gz')
    plot_stand(stand_dir, stand['cases'], out/'stand90.png')
    result = {'protocol_sha256':PLAN_SHA256,'resources':{s:resource_receipt(s) for s in ('stand','shadow')},
              'source_sha256':{name:sha(ROOT/'tools90'/name) for name in
                               ('stand90.py','shadow90.py','wbc90.py','readback90.py','mechanics_check90.py','plot90.py')},
              'numeric_environment':{'mujoco':mujoco.__version__,'osqp':osqp.__version__,
                                     'numpy':np.__version__,'scipy':scipy.__version__},
              'independent_mechanical_gate':{'passing_frames':physical,'registered_frames':len(frames),
                  'passed':physical == len(frames) == wbc_checked['bypass_bitwise_equal']},
              'stand':stand,'stand_readback':stand_checked,'shadow':shadow,'shadow_readback':wbc_checked,
              'support_readback':support,'mechanics_readback':mechanics,
              'truth_base_quats_wxyz':attitudes,
              'stand_sha256':sha(stand_dir/'report.json'),'shadow_sha256':sha(shadow_dir/'report.json'),
              'frames_sha256':sha(frame_file),'frames_gzip_sha256':sha(out/'stand_wbc90_frames.jsonl.gz'),
              'truth_sha256':sha(truth_file),'truth_gzip_sha256':sha(out/'stand_wbc90_truth.jsonl.gz'),
              'plot_sha256':sha(out/'stand90.png'),'native_steps_during_readback':0,'qp_solves_during_readback':0}
    with (out/'stand_wbc90.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
