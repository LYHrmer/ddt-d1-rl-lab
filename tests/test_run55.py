"""Selection isolation and monotonic work accounting for the55 runner."""
import copy
import json

import pytest

from tools55.run55 import audit_work, seal_predictions


def test_probe_predictions_cannot_change_sealed_development_theta():
    proposal = {'theta': .01, 'development_mean_gradients': [-.2, -.1, -.3],
                'physical_probe_results_seen': False, 'test_gradients': []}
    before = copy.deepcopy(proposal)
    rows = [{'test_gradients': [{'gradient': 1e9}]} for _ in range(3)]
    sealed = seal_predictions(proposal, rows)
    assert sealed['theta'] == .01
    assert sealed['development_mean_gradients'] == before['development_mean_gradients']
    assert not sealed['probe_perturbation_results_seen']
    assert proposal == before
    with pytest.raises(ValueError, match='three models'):
        seal_predictions(proposal, rows[:2])


def record(tick, phase):
    return {'phase': phase, 'counts': {'policy_transitions': tick, 'native_steps': tick * 10,
                                      'pd_updates': tick * 5}}


def test_work_audit_retains_phase_counts(tmp_path):
    path = tmp_path / 'steps.jsonl'
    rows = [record(1, 'fresh_baseline_collection'), record(2, 'physical_probes'), record(3, 'physical_probes')]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert audit_work(path) == {'policy_steps': 3, 'by_phase': {'fresh_baseline_collection': 1, 'physical_probes': 2}}


@pytest.mark.parametrize('bad', [record(1, 'physical_probes'), record(3, 'physical_probes'),
    {'phase': 'physical_probes', 'counts': {'policy_transitions': 2, 'native_steps': 10, 'pd_updates': 10}}])
def test_work_audit_rejects_refunded_missing_or_inconsistent_steps(tmp_path, bad):
    path = tmp_path / 'steps.jsonl'
    path.write_text(json.dumps(record(1, 'fresh_baseline_collection')) + '\n' + json.dumps(bad) + '\n')
    with pytest.raises(ValueError, match='nonmonotonic'):
        audit_work(path)
