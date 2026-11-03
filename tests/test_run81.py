"""Fail-closed81 stage transitions and sealed prefix integrity, no robot assets."""
import json

import pytest

from tools81 import run81


def test_registered_budget_and_disjoint_roles():
    assert run81.SCREEN_STEPS == 1800 + 1800 + (300 + 72)*32
    assert run81.FULL_STEPS == run81.SCREEN_STEPS + 63*600 == 53304
    assert run81.FULL_STEPS + 224 == 53528
    assert run81.MODEL_CALLS == 6*6*9*32 + 387*32 + 3*6*32 == 23328
    sets = [set(s) for s in (run81.D_SEEDS, run81.V_SEEDS, run81.EVAL_SEEDS)]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])


def test_missing_duplicate_or_wrong_task_point_does_not_fill_grid():
    points = [{'seed': s, 'tick': t, 'task': 'turn_stop'} for s in run81.D_SEEDS for t in run81.STEADY_TICKS]
    assert run81.validate_points(points, run81.D_SEEDS, run81.STEADY_TICKS) == points
    for bad in (points[:-1], points[:-1] + [points[0]], points[:-1] + [dict(points[-1], task='forward_stop')]):
        with pytest.raises(ValueError):
            run81.validate_points(bad, run81.D_SEEDS, run81.STEADY_TICKS)


def test_screen_rejection_never_calls_holdout_and_cannot_select_seed():
    report = {'screen': {'status': 'COMPLETE', 'enter_full_evaluation': False}, 'baseline': [], 'candidates': []}
    assert run81.conditional_evaluation(report, lambda: pytest.fail('holdout touched')) == 15504
    assert report['outcome'] == 'SCREEN_REJECTED' and report['evaluation_status'] == 'NOT_RUN'
    assert report['objective_passed'] is False


def test_full_negative_result_is_complete_and_partial_never_accepted():
    report = {'screen': {'status': 'COMPLETE', 'enter_full_evaluation': True}}
    def complete():
        report['comparison'] = {'status': 'COMPLETE', 'objective_passed': False}
    assert run81.conditional_evaluation(report, complete) == 53304
    assert report['evaluation_status'] == 'COMPLETE' and not report['objective_passed']
    report['comparison']['status'] = 'INCOMPLETE'
    with pytest.raises(ValueError):
        run81.conditional_evaluation(report, lambda: None)


@pytest.mark.parametrize('kind', ('partial_screen', 'touched_holdout'))
def test_invalid_stage_state_rejected(kind):
    report = {'screen': {'status': 'COMPLETE', 'enter_full_evaluation': False}, 'baseline': [], 'candidates': []}
    if kind == 'partial_screen':
        report['screen']['status'] = 'INCOMPLETE'
    else:
        report['baseline'] = [1]
    with pytest.raises(ValueError):
        run81.conditional_evaluation(report, lambda: pytest.fail('holdout called'))


def test_seal_is_exclusive_and_late_hash_rewrite_cannot_hide_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(run81, 'verify', lambda _plan: None)
    plan_path, member = tmp_path / 'plan.json', tmp_path / 'member'
    plan_path.write_text('{}')
    member.write_text('original')
    report = {'status': 'COMPLETE', 'screen': {'status': 'COMPLETE', 'passed': True, 'enter_full_evaluation': True}}
    run81.seal(tmp_path, 'sealed.json', [member], report)
    with pytest.raises(FileExistsError):
        run81.seal(tmp_path, 'sealed.json', [member], report)
    member.write_text('modified')
    report['raw_artifacts_sha256'] = {'member': run81.sha(member)}
    with pytest.raises(ValueError, match='sealed member changed'):
        run81.verify_final(report, {'_sha256': run81.sha(plan_path), 'artifact_cap_bytes': 160 << 20}, plan_path, tmp_path)
    assert report['screen']['status'] == 'INCOMPLETE' and not report['screen']['enter_full_evaluation']
    assert report['provisional_screen_before_revocation']['passed']
    assert not report['global_candidate_accepted']


def test_incomplete_update_count_is_retained_without_promoting_policy(tmp_path):
    path = tmp_path / 'seed64001/actor_Q6'
    path.mkdir(parents=True)
    (path / 'receipt.json').write_text(json.dumps({'status': 'INCOMPLETE', 'updates_completed': 1, 'updates_attempted': 1}))
    result = run81.recover_updates(tmp_path)
    assert result['policy_network_updates'] == 1 and result['receipts'][0]['status'] == 'INCOMPLETE'
    (path / 'receipt.json').write_text(json.dumps({'status': 'COMPLETE', 'updates_completed': True, 'updates_attempted': 1}))
    with pytest.raises(ValueError):
        run81.recover_updates(tmp_path)


@pytest.mark.parametrize('fail_on_third', (False, True))
def test_physical_branch_captures_real_first_input_and_preserves_partial(tmp_path, monkeypatch, fail_on_third):
    np = pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace

    from tools52 import adapter52, control52
    from tools55 import probe_local55
    from tools81 import diagnostics81

    class Env:
        def __init__(self):
            self.n = self.native = self.pd = 0
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
    actor = SimpleNamespace(encoder=torch.nn.Identity())
    restores = []
    monkeypatch.setattr(adapter52, 'restore', lambda _env, _shaper, snap: restores.append(snap))
    def action(_env, _shaper, actor, _theta):
        actor.encoder(torch.full((1, 627), 2., dtype=torch.float32))
        return np.zeros(16, dtype=np.float32)
    monkeypatch.setattr(control52, 'real_action', action)
    monkeypatch.setattr(control52, 'physical', lambda _env: np.zeros(45))
    monkeypatch.setattr(probe_local55, 'cost64', lambda *_args: (.25, [.25] * 32))
    monkeypatch.setattr(diagnostics81, 'physical_cost_parts', lambda *_args: np.full((32, 6), .25/6))
    snapshot = object()
    point = {'snapshot': snapshot, 'commands': np.tile(env._command, (33, 1))}
    receipt, path = {'status': 'INCOMPLETE', 'steps': 0}, tmp_path / 'trace.npz'
    if fail_on_third:
        with pytest.raises(RuntimeError, match='native-step'):
            run81.physical_branch(env, actor, point, path, receipt)
        assert receipt['status'] == 'INCOMPLETE' and receipt['steps'] == 2
        assert receipt['actual_counts'] == {'policy_transitions': 3, 'native_steps': 25, 'pd_updates': 12}
    else:
        run81.physical_branch(env, actor, point, path, receipt)
        assert receipt['status'] == 'COMPLETE' and receipt['yaw_mae'] == pytest.approx(.2)
        assert receipt['prefix_metrics']['32']['cost'] == .25
    assert restores == [snapshot] and not actor.encoder._forward_pre_hooks
    with np.load(path, allow_pickle=False) as data:
        assert np.array_equal(data['first_actor_input'], np.full(627, 2., dtype=np.float32))
        assert len(data['states']) == (2 if fail_on_third else 32)
    assert json.loads(path.with_suffix('.json').read_text())['status'] == receipt['status']
