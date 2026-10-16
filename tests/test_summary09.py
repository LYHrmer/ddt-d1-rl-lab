"""An incomplete training consumes budget but provides no scientific verdict. No physics."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from _workstation import needs_workstation_evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools09'))

import build_summary09 as summary  # noqa: E402
import verify_scale09 as verify  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def receipts(tmp_path):
    ev = tmp_path/'evidence/career_contract_09'
    config = tmp_path/'config.json'
    write(config, {'frozen': True})
    release = {'frozen_inputs': {'config.json': summary.sha(config)},
               'plans': {'scale': {'steps': 8388608}}, 'contract': 'contract09.md',
               'single_change': 'entropy coefficient 0.005 -> 0',
               'budget': {'cap': {'transitions': 8421376, 'native': 84231760},
                          'project_cumulative_before': {'transitions': 13664256, 'native': 137162750}}}
    write(ev/summary.RECEIPTS[0], release)
    def trainer(n, status):
        counts = {'policy_transitions': n, 'native_steps': n*10}
        return {'status': status, 'trained_transitions': n, 'actual_counts': counts,
                'per_worker_counts': [dict(counts)], 'inputs_unchanged': True,
                'finished_utc': '2026-10-16T06:16:09+00:00', 'min_available_ram_bytes': 3307237376}
    for prefix, n, status in [('preflight', 32768, 'COMPLETE'),
                              ('train', 4807168, 'RAM_RESERVE_BELOW_3_GIB')]:
        result = trainer(n, status)
        write(ev/f'scale09_{prefix}_01/result.json', result)
        launch = {'trainer_result': result, 'release_sha256': summary.sha(ev/summary.RECEIPTS[0]),
                  'kind': 'scale' if prefix == 'train' else 'preflight', 'equivalence': 'PASS',
                  'trainer_returncode': 0 if prefix == 'preflight' else 1,
                  'interrupted_by_launcher': False, 'wall_s': 3679., 'rollouts_compared': 1}
        write(ev/f'scale09_{prefix}_01_launcher/report.json', launch)
    write(ev/summary.RECEIPTS[5], {'inputs_sha256': {str(config): summary.sha(config)},
                                 'planned_transitions': 8388608, 'hyperparameters': {'ent_coef': 0},
                                 'kind': 'scale', 'seed': 101})
    write(ev/summary.RECEIPTS[6], {'predictions': verify.judge(None, None, {}, launch),
                                 'qualification': {'overall': 'NOT_EVALUATED', 'tasks_passed': None},
                                 'inputs': {'launcher_report_sha256': summary.sha(ev/summary.RECEIPTS[3])},
                                 'interpretation': 'training did not complete'})
    (ev/'scale09_train_01/partial_policy.zip').write_bytes(b'partial: never load this')
    write(tmp_path/'reports/contract08_summary.json',
          {'project_cumulative': release['budget']['project_cumulative_before']})
    return tmp_path, ev


def test_failed_training_steps_are_counted_without_qualifying_the_partial_policy(receipts):
    root, ev = receipts
    before = {p: p.read_bytes() for p in ev.rglob('*') if p.is_file()}
    out = summary.build(root)
    assert out['status'] == 'INCOMPLETE'
    assert out['budget']['used_transitions'] == 4839936
    assert out['budget']['used_native'] == 48399360
    assert out['project_cumulative']['transitions'] == 18504192
    assert out['project_cumulative']['native'] == 185562110
    assert out['qualification']['tasks_passed'] is None
    assert all(out['pre_registered_verdict'][q] == 'NOT_EVALUABLE' for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'))
    assert out['physics_steps_executed_by_this_tool'] == 0
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize('change, message', [
    ('frozen', 'frozen input changed'), ('result', 'launcher/trainer result mismatch'),
    ('qualification', 'must not have predictions'), ('qualification-count', 'must not have predictions'),
    ('final', 'unexpected final policy'),
    ('post-training', 'unexpected post-training artifacts'), ('previous', 'cumulative total differs'),
])
def test_inconsistent_or_post_training_artifacts_are_rejected(receipts, change, message):
    root, ev = receipts
    if change == 'frozen':
        write(root/'config.json', {})
    elif change == 'result':
        path = ev/summary.RECEIPTS[4]
        data = summary.read(path)
        data['trained_transitions'] -= 8
        write(path, data)
    elif change == 'qualification':
        path = ev/summary.RECEIPTS[6]
        data = summary.read(path)
        data['qualification']['overall'] = 'PASS'
        write(path, data)
    elif change == 'qualification-count':
        path = ev/summary.RECEIPTS[6]
        data = summary.read(path)
        data['qualification']['tasks_passed'] = 3
        write(path, data)
    elif change == 'final':
        (ev/'scale09_train_01/policy.zip').touch()
    elif change == 'post-training':
        (ev/'scale09_python_eval_01').mkdir()
    elif change == 'previous':
        write(root/'reports/contract08_summary.json', {'project_cumulative': {'transitions': 0, 'native': 0}})
    with pytest.raises(ValueError, match=message):
        summary.build(root)


def test_worker_disagreement_cannot_be_hidden_by_a_matching_trainer_and_launcher(receipts):
    _, ev = receipts
    result = summary.read(ev/summary.RECEIPTS[4])
    wrong = copy.deepcopy(result)
    wrong['per_worker_counts'][0]['native_steps'] -= 10
    with pytest.raises(ValueError, match='worker counters disagree'):
        summary.counters(wrong)


@needs_workstation_evidence
def test_published_closeout_matches_the_original_receipts():
    published = ROOT/'reports/contract09_summary.json'
    if not published.is_file() or not (ROOT/'evidence/career_contract_09/scale09_verdict_01/verdict.json').is_file():
        pytest.skip('contract 09 closeout receipts not on this machine')
    expected, actual = summary.read(published), summary.build()
    expected.pop('prepared_utc')
    actual.pop('prepared_utc')
    assert expected == actual
