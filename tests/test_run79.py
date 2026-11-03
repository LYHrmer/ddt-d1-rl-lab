"""No numerical imports: fail-closed accounting of new79 executions."""
import json

import pytest

from tools79 import run79
from tools79.run79 import recover_updates, validate_counts


def test_physics_restore_cannot_refund_native_or_pd():
    validate_counts({'policy_transitions': 224, 'native_steps': 2240, 'pd_updates': 1120}, 224)
    with pytest.raises(ValueError):
        validate_counts({'policy_transitions': 224, 'native_steps': 1600, 'pd_updates': 1120}, 224)
    with pytest.raises(ValueError):
        validate_counts({'policy_transitions': 223, 'native_steps': 2230, 'pd_updates': 1115}, 224)


def test_partial_training_is_counted_without_promoting_its_status(tmp_path):
    values = [('actor_first', 'COMPLETE', 1, 1), ('refit_refresh', 'INCOMPLETE', 7, 8),
              ('actor_refresh', 'INCOMPLETE', 0, 0)]
    for name, status, completed, attempted in values:
        path = tmp_path / 'seed64001' / name
        path.mkdir(parents=True)
        (path / 'receipt.json').write_text(json.dumps({'status': status, 'updates_completed': completed,
                                                     'updates_attempted': attempted}))
    result = recover_updates(tmp_path)
    assert result['policy_network_updates'] == 1 and result['model_updates'] == 7
    assert [r['status'] for r in result['receipts']].count('INCOMPLETE') == 2


def test_update_receipts_reject_overrun_and_fabricated_work(tmp_path):
    path = tmp_path / 'seed64001' / 'actor_first'
    path.mkdir(parents=True)
    receipt = path / 'receipt.json'
    for completed, attempted in [(2, 2), (1, 0), (-1, 0), (True, 1)]:
        receipt.write_text(json.dumps({'status': 'COMPLETE', 'updates_completed': completed,
                                       'updates_attempted': attempted}))
        with pytest.raises(ValueError):
            recover_updates(tmp_path)


@pytest.mark.parametrize('tamper', ('plan', 'artifact'))
def test_late_verification_revokes_pass_but_preserves_original_evidence(tmp_path, monkeypatch, tamper):
    monkeypatch.setattr(run79, 'verify', lambda _plan: None)
    plan_file = tmp_path / 'plan.json'
    plan_file.write_text('{}')
    plan = {'_sha256': run79.sha(plan_file)}
    artifact = tmp_path / 'policy.pt'
    artifact.write_bytes(b'original fixture')
    report = {'status': 'COMPLETE', 'raw_artifacts_sha256': {'policy.pt': run79.sha(artifact)},
              'comparison': {'status': 'COMPLETE', 'feedback': {'passed': True},
                             'control_gates': [{'passed': True}]}}
    (plan_file if tamper == 'plan' else artifact).write_text('changed')
    with pytest.raises(ValueError):
        run79.verify_final(report, plan, plan_file, tmp_path)
    assert report['provisional_comparison_before_revocation']['feedback']['passed']
    assert report['comparison']['status'] == 'INCOMPLETE'
    assert not report['comparison']['feedback']['passed']
    assert not report['comparison']['control_gates'][0]['passed']
    assert not report['global_candidate_accepted']
