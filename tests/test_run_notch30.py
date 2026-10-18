"""Synthetic actor/trace checks; no robot model or physics required."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools30 import run_notch30 as run  # noqa: E402


def original():
    calls = []
    def predict(obs):
        calls.append(obs)
        return np.full(16, obs, dtype=np.float32)
    return SimpleNamespace(predict=predict, make_env=lambda *args: args, backend={}, calls=calls)


def recorded():
    rt = run.FilteredRuntime(original())
    rt.make_env('turn_stop', [], 3, 1001)
    for x in [.5, -.8, .7, .9, -.4, -.1]:
        rt.predict(x)
    arrays = rt.arrays()
    applied = np.zeros_like(arrays['issued'])
    applied[3:] = arrays['issued'][:-3]
    return rt, arrays, applied


def test_one_inference_per_tick_exact_float32_issue_and_constant_reset():
    rt, arrays, applied = recorded()
    assert len(rt.original.calls) == len(arrays['raw']) == rt.filter.calls == 6
    assert run.verify_filter(arrays, arrays['issued'], applied, 3) == rt.filter.receipt()
    assert arrays['issued'].dtype == np.float32 and arrays['linear'].dtype == np.float64
    with pytest.raises(ValueError, match='not verified'):
        rt.make_env('forward_stop', [], 0, 2001)
    rt.rows.clear()
    rt.make_env('forward_stop', [], 0, 2001)
    assert np.array_equal(rt.predict(.25), np.full(16, .25, dtype=np.float32))
    assert rt.filter.calls == 1


def test_filter_does_not_depend_on_case_seed_or_delay():
    a, b = run.FilteredRuntime(original()), run.FilteredRuntime(original())
    a.make_env('turn_stop', [], 3, 1001)
    b.make_env('reverse_stop', [], 0, 2003)
    for value in [.5, -.7, .3, -.1]:
        assert np.array_equal(a.predict(value), b.predict(value))


@pytest.mark.parametrize('field', ['raw', 'linear', 'issued', 'physical_issue', 'physical_apply'])
def test_saved_replay_rejects_corrupted_signals(field):
    _, arrays, applied = recorded()
    issued = arrays['issued'].copy()
    target = issued if field == 'physical_issue' else applied if field == 'physical_apply' else arrays[field]
    target[2, 7] += .01
    with pytest.raises(ValueError):
        run.verify_filter(arrays, issued, applied, 3)


@pytest.mark.parametrize('delay', [0, 1, 2, 3])
def test_fifo_replay_including_initial_zeros(delay):
    _, arrays, _ = recorded()
    applied = np.zeros_like(arrays['issued'])
    applied[:] = arrays['issued'] if delay == 0 else np.concatenate(
        [np.zeros((delay, 16), dtype=np.float32), arrays['issued'][:-delay]])
    run.verify_filter(arrays, arrays['issued'], applied, delay)


def complete(stage='formal'):
    return {'status': 'COMPLETE', 'cleanup_complete': True, 'actual_counts': run.COUNTS(21600),
            'known_actual_counts': run.COUNTS(21600), 'all_36_reference_pass': True,
            'cases': [{'sequence': name, 'delay_ticks': delay, 'reset_seed': seed,
                       'reference_thresholds_met': True, 'failure': False,
                       'done_reason': 'HORIZON_12S', 'policy_transitions': 600}
                      for name, delay, seed in run.matrix(stage)]}


def test_holdout_requires_exact_original_matrix_and_full_success():
    good = complete()
    assert run.eligible(good)
    for key, value in [('reference_thresholds_met', False), ('failure', True),
                       ('policy_transitions', 599), ('done_reason', 'TERMINATED'), ('reset_seed', 2001)]:
        bad = copy.deepcopy(good)
        bad['cases'][0][key] = value
        assert not run.eligible(bad)
    bad = copy.deepcopy(good)
    bad['cases'][1] = copy.deepcopy(bad['cases'][0])
    assert not run.eligible(bad)


@pytest.mark.parametrize('outcome', ['pass', 'task_failure', 'interrupted'])
def test_execution_gates_holdout_and_preserves_partial_accounting(tmp_path, monkeypatch, outcome):
    plan = tmp_path/'plan.json'
    plan.write_text('{}')
    monkeypatch.setattr(run, 'PLAN', plan)
    monkeypatch.setattr(run, 'OUTPUT', tmp_path/'execution')
    monkeypatch.setattr(run, 'load_plan', lambda: ({'inputs': {}, 'source_identity': {}}, {}))
    monkeypatch.setattr(run, 'source_identity', lambda *args: {})
    monkeypatch.setattr(run, 'inputs', lambda: ({}, {}))
    monkeypatch.setattr(run.os, 'getpriority', lambda *args: 10)
    clock = iter(range(0, 10000, 11))
    monkeypatch.setattr(run.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(run.time, 'sleep', lambda _: None)
    monkeypatch.setattr(run, 'snapshot_memory', lambda _: {'effective_available_bytes': 8 << 30, 'errors': []})
    stages = []
    def evaluate(stage, bundle, parent, output, report, poll, **kwargs):
        stages.append(stage)
        if outcome == 'interrupted':
            report.update(actual_counts=None, known_actual_counts=run.COUNTS(7), cleanup_complete=False)
            raise RuntimeError('synthetic interrupted counters unknown')
        report.update(complete(stage))
        if outcome == 'task_failure':
            report['cases'][-1]['reference_thresholds_met'] = False
            report['all_36_reference_pass'] = False
    monkeypatch.setattr(run, 'evaluate', evaluate)
    result = run.execute()
    assert stages == (['formal', 'holdout'] if outcome == 'pass' else ['formal'])
    assert result['python_original_and_holdout_qualified'] is (outcome == 'pass')
    if outcome == 'interrupted':
        assert result['status'] == 'INCOMPLETE' and result['actual_counts'] is None
        assert result['known_actual_counts'] == run.COUNTS(7) and result['cleanup_complete'] is False
    else:
        assert result['status'] == 'COMPLETE'
        assert result['actual_counts'] == run.COUNTS(43200 if outcome == 'pass' else 21600)
    with pytest.raises(FileExistsError):
        run.execute()
