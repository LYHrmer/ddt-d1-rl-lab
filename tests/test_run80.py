"""Accounting and fail-closed stage transitions, without robot physics."""
import json

import pytest

from tools80 import run80


def points():
    return [{'task': task, 'seed': seed, 'tick': tick, 'snapshot': object()}
            for task, seeds in run80.DEV_SEEDS.items() for seed in seeds
            for tick in (run80.TURN_TICKS if task == 'turn_stop' else run80.OLD_TICKS)]


def test_sampling_changes_only_six_turn_points_and_shares_exact_live_objects():
    source = points()
    legacy, steady = [run80.select_points(source, arm) for arm in run80.ARMS]
    assert len(legacy) == len(steady) == 27
    common = set(map(id, legacy)) & set(map(id, steady))
    assert len(common) == 21
    assert all(id(p) in common for p in legacy if p['task'] != 'turn_stop' or p['tick'] == 410)
    assert {p['tick'] for p in steady if p['task'] == 'turn_stop'} == {175, 325, 410}
    assert all(any(p is original for original in source) for p in legacy + steady)


def test_missing_duplicate_or_changed_task_point_never_fills_a_training_grid():
    for source in (points()[:-1], points()[:-1] + [points()[0]],
                   points()[:-1] + [{'task': 'forward_stop', 'seed': 80109, 'tick': 410}]):
        with pytest.raises(ValueError):
            run80.select_points(source, 'steady')


def test_short_screen_failure_never_calls_holdout_or_selects_passing_seed():
    report = {'screen': {'status': 'COMPLETE', 'enter_full_evaluation': False,
                        'by_model_seed': [{'passed': True}, {'passed': False}, {'passed': True}]},
              'baseline': [], 'candidates': []}
    def forbidden():
        pytest.fail('holdout callback invoked after failed whole screen')
    assert run80.conditional_evaluation(report, forbidden) == 9240
    assert report['evaluation_status'] == 'NOT_RUN' and report['outcome'] == 'SCREEN_REJECTED'
    assert not report['sampling_passed']


def test_successful_screen_requires_complete_full_comparison_and_preserves_negative_result():
    report = {'screen': {'status': 'COMPLETE', 'enter_full_evaluation': True}}
    calls = []
    def complete():
        calls.append(1)
        report['comparison'] = {'status': 'COMPLETE', 'sampling_passed': False}
    assert run80.conditional_evaluation(report, complete) == 47040 and calls == [1]
    assert report['evaluation_status'] == 'COMPLETE' and not report['sampling_passed']
    report['comparison']['status'] = 'INCOMPLETE'
    with pytest.raises(ValueError):
        run80.conditional_evaluation(report, lambda: None)


def test_partial_screen_and_already_touched_holdout_rejected():
    for report in [
        {'screen': {'status': 'INCOMPLETE', 'enter_full_evaluation': True}},
        {'screen': {'status': 'COMPLETE', 'enter_full_evaluation': False}, 'baseline': [1], 'candidates': []},
    ]:
        with pytest.raises(ValueError):
            run80.conditional_evaluation(report, lambda: pytest.fail('should not evaluate'))


def test_updates_recover_completed_prefix_without_promoting_partial(tmp_path):
    for name, status, n, attempted in [('actor_steady_1', 'COMPLETE', 1, 1),
                                      ('actor_steady_2', 'INCOMPLETE', 0, 1)]:
        p = tmp_path / 'seed64001' / name
        p.mkdir(parents=True)
        (p / 'receipt.json').write_text(json.dumps({'status': status, 'updates_completed': n,
                                                   'updates_attempted': attempted}))
    result = run80.recover_updates(tmp_path)
    assert result['policy_network_updates'] == 1 and result['model_updates'] == 0
    assert result['receipts'][1]['status'] == 'INCOMPLETE'
    (tmp_path / 'seed64001/actor_steady_2/receipt.json').write_text(json.dumps(
        {'status': 'COMPLETE', 'updates_completed': True, 'updates_attempted': 1}))
    with pytest.raises(ValueError):
        run80.recover_updates(tmp_path)


@pytest.mark.parametrize('tamper', ('plan', 'artifact'))
def test_late_verification_revokes_screen_and_sampling_passes(tmp_path, monkeypatch, tamper):
    monkeypatch.setattr(run80, 'verify', lambda _plan: None)
    plan_path, artifact = tmp_path / 'plan.json', tmp_path / 'policy.pt'
    plan_path.write_text('{}')
    artifact.write_bytes(b'fixture')
    plan = {'_sha256': run80.sha(plan_path), 'artifact_cap_bytes': 128 << 20}
    report = {'status': 'COMPLETE', 'raw_artifacts_sha256': {'policy.pt': run80.sha(artifact)},
              'screen': {'status': 'COMPLETE', 'passed': True, 'enter_full_evaluation': True},
              'comparison': {'status': 'COMPLETE', 'sampling_passed': True,
                             'sampling': {'passed': True}, 'control_gates': [{'passed': True}]}}
    (plan_path if tamper == 'plan' else artifact).write_bytes(b'tampered')
    with pytest.raises(ValueError):
        run80.verify_final(report, plan, plan_path, tmp_path)
    assert report['provisional_screen_before_revocation']['passed']
    assert report['provisional_comparison_before_revocation']['sampling_passed']
    assert not report['screen']['enter_full_evaluation'] and not report['comparison']['sampling_passed']
    assert not report['comparison']['control_gates'][0]['passed'] and not report['global_candidate_accepted']


@pytest.mark.parametrize('kind', ('development', 'predictions', 'admission'))
def test_final_manifest_cannot_hide_mutation_after_original_seal(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(run80, 'verify', lambda _plan: None)
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text('{}')
    member = tmp_path / 'input.json'
    member.write_text('original')
    original_sha = run80.sha(member)
    report = {'status': 'COMPLETE'}
    if kind == 'admission':
        report['admission'] = {'path': str(member), 'sha256': original_sha}
    else:
        name = 'development_sealed_before_training.json' if kind == 'development' else 'predictions_sealed_before_physics.json'
        seal = tmp_path / name
        payload = {'artifact_sha256': {'input.json': original_sha}} if kind == 'development' else [
            {'trace_path': 'input.json', 'trace_sha256': original_sha}]
        seal.write_text(json.dumps(payload))
        report['development_seal_sha256' if kind == 'development' else 'prediction_seal_sha256'] = run80.sha(seal)
    member.write_text('changed after seal')
    report['raw_artifacts_sha256'] = {'input.json': run80.sha(member)}
    with pytest.raises(ValueError, match='changed'):
        run80.verify_final(report, {'_sha256': run80.sha(plan_path), 'artifact_cap_bytes': 128 << 20}, plan_path, tmp_path)


def test_holdout_exception_keeps_pending_episode_and_full_prefix(tmp_path, monkeypatch):
    np = pytest.importorskip('numpy')
    pytest.importorskip('torch')
    from types import SimpleNamespace

    from tools52 import control52
    from tools80 import policy80

    actor = SimpleNamespace(state_dict=dict)
    actors = {(s, a): actor for s in run80.SEEDS for a in run80.ARMS}
    env = SimpleNamespace(counters=lambda: {'policy_transitions': 9242, 'native_steps': 92420, 'pd_updates': 46210},
                          _validate_eval_sequence=lambda x: x, data=SimpleNamespace(qpos=[1., 2.]), free_qpos=0)
    monkeypatch.setattr(policy80, 'mapping_sha256', lambda _state: 'fixture')
    monkeypatch.setattr(control52, 'real_action', lambda *_args: np.zeros(16))
    def episode(_env, predict, *_args):
        predict(None)
        raise RuntimeError('synthetic holdout interruption')
    saves = []
    def save_prefix(path, initial_xy):
        saves.append(initial_xy)
        with path.open('xb') as f:
            np.savez_compressed(f, body_wz_rad_s=[.1, .2])
        return 2
    report = {'baseline': [], 'candidates': []}
    helpers = SimpleNamespace(SEQUENCES={t: [] for t in run80.TASKS}, run_episode=episode)
    with pytest.raises(RuntimeError, match='holdout interruption'):
        run80.evaluate(env, actor, actors, helpers, tmp_path, report, lambda *_args: None,
                       lambda _phase: None, save_prefix)
    assert saves == [[1., 2.]] and len(report['baseline']) == 1
    row = report['baseline'][0]
    assert row['status'] == 'INCOMPLETE' and row['partial_trace_rows'] == 2
    assert (tmp_path / row['partial_trace_path']).is_file()
    assert len(list((tmp_path / 'evaluation').glob('*.pending.json'))) == 1


@pytest.mark.parametrize('fail_on_third', (False, True))
def test_physical_branch_uses_raw_yaw_and_keeps_prefix_and_actual_incomplete_counts(tmp_path, monkeypatch, fail_on_third):
    np = pytest.importorskip('numpy')
    pytest.importorskip('torch')
    from tools52 import adapter52, control52
    from tools55 import probe_local55

    class Env:
        def __init__(self):
            self.n, self.native, self.pd = 0, 0, 0
            self._command = np.array([0., 0., .2])

        def counters(self):
            return {'policy_transitions': self.n, 'native_steps': self.native, 'pd_updates': self.pd}

        def step(self, action):
            self.n += 1
            if fail_on_third and self.n == 3:
                self.native += 5
                self.pd += 2
                raise RuntimeError('synthetic native-step failure')
            self.native += 10
            self.pd += 5
            return np.zeros(627), 1., False, False, {'body_wz_rad_s': .4,
                'command_during_interval': self._command, 'applied_action': action,
                'fallen': False, 'illegal_body_contact': False}

    env = Env()
    snapshot = object()
    restores = []
    monkeypatch.setattr(adapter52, 'restore', lambda _env, _shaper, snap: restores.append(snap))
    monkeypatch.setattr(control52, 'real_action', lambda *_args: np.zeros(16, dtype=np.float32))
    monkeypatch.setattr(control52, 'physical', lambda _env: np.zeros(45))
    monkeypatch.setattr(probe_local55, 'cost64', lambda *_args: (.25, [.25] * 32))
    start = {'snapshot': snapshot, 'commands': np.tile(env._command, (33, 1))}
    receipt, path = {'status': 'INCOMPLETE', 'steps': 0}, tmp_path / 'branch.npz'
    if fail_on_third:
        with pytest.raises(RuntimeError, match='native-step'):
            run80.physical_branch(env, None, start, path, receipt)
        assert receipt['status'] == 'INCOMPLETE' and receipt['steps'] == 2
        assert receipt['actual_counts'] == {'policy_transitions': 3, 'native_steps': 25, 'pd_updates': 12}
    else:
        run80.physical_branch(env, None, start, path, receipt)
        assert receipt['status'] == 'COMPLETE' and receipt['steps'] == 32
        assert receipt['yaw_mae'] == pytest.approx(.2) and receipt['cost'] == .25
        assert receipt['native_steps'] == 320 and receipt['pd_updates'] == 160
    assert restores == [snapshot]
    with np.load(path, allow_pickle=False) as saved:
        assert len(saved['states']) == (2 if fail_on_third else 32)
        assert all(saved['body_wz'] == .4) and all(saved['observations'][:, 2] == 0.)
    assert json.loads(path.with_suffix('.json').read_text())['status'] == receipt['status']
