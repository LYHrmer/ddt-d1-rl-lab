"""Admission tests; synthetic inputs only, no weights, physics or training."""
import copy
import json

import numpy as np
import pytest

from tools71 import run71


def source():
    return {'status': 'COMPLETE', 'verification_after': 'COMPLETE', 'completed_model_fits': 6,
            'model_updates': 6000, 'models': [
                {'seed': s, 'arm': a, 'status': 'COMPLETE', 'fit': {'status': 'COMPLETE',
                 'updates': 1000, 'updates_attempted': 1000, 'checkpoint_selection': 'final_only',
                 'eligible_for_evaluation': True}} for s in run71.SEEDS for a in run71.ARMS]}


def test_complete_all_six_required():
    run71.complete_source(source())
    missing = source()
    missing['models'].pop()
    with pytest.raises(ValueError, match='all six'):
        run71.complete_source(missing)


@pytest.mark.parametrize('field,value', [('status', 'PARTIAL'), ('updates', 999),
                                       ('updates_attempted', 1001), ('eligible_for_evaluation', False),
                                       ('checkpoint_selection', 'best')])
def test_partial_or_selected_checkpoint_rejected(field, value):
    record = source()
    record['models'][3]['fit'][field] = value
    with pytest.raises(ValueError, match='partial'):
        run71.complete_source(record)


def test_contract_and_budget_fixed():
    good = {'schema': 'd1-training-gradient71-v1', 'decision_spec': copy.deepcopy(run71.SPEC),
            'cpu_seconds': 120, 'wall_seconds': 180, 'source_sha256': {}}
    for key, value in [('cpu_seconds', 121), ('wall_seconds', 181)]:
        changed = good | {key: value}
        with pytest.raises(ValueError, match='budget'):
            run71.verify71(changed)
    good['decision_spec']['state_windows'] = 64
    with pytest.raises(ValueError, match='contract'):
        run71.verify71(good)


def test_binding_set_cannot_drop_inputs():
    plan = {'schema': 'd1-training-gradient71-v1', 'decision_spec': run71.SPEC,
            'cpu_seconds': 120, 'wall_seconds': 180, 'source_sha256': {}}
    with pytest.raises(ValueError, match='binding set'):
        run71.verify71(plan)
    assert len([p for p in run71.required_inputs() if p.endswith('final_state_dict.pt')]) == 6
    assert len([p for p in run71.required_inputs() if '/branch_' in p]) == 48


def test_zero_is_original_action_not_all_zero_and_repeated_branch_must_match(tmp_path, monkeypatch):
    monkeypatch.setattr(run71, 'ROOT', tmp_path)
    labels = {'s': np.zeros((96, 45)), 'a0': np.ones((96, 16)) * .125,
              'source_id': np.repeat([58, 62], 48)}
    for directory in run71.ZERO_SOURCES.values():
        (tmp_path / directory).mkdir(parents=True)
        for point in range(12):
            for branch in (0, 1):
                np.savez(tmp_path / directory / f'branch_{point:02d}_{branch:02d}.npz',
                         s=np.zeros((1, 45)), applied_action=np.ones((1, 16)) * .125,
                         target=np.ones((1, 45)) * point)
    targets = run71.read_zero_targets(labels)
    assert targets.shape == (24, 45) and np.array_equal(targets[:, 28], np.tile(np.arange(12), 2))
    directory = next(iter(run71.ZERO_SOURCES.values()))
    np.savez(tmp_path / directory / 'branch_00_01.npz', s=np.zeros((1, 45)),
             applied_action=np.ones((1, 16)) * .125, target=np.ones((1, 45)))
    with pytest.raises(ValueError, match='repeated zero'):
        run71.read_zero_targets(labels)


def test_spec_serializes_without_nonfinite():
    assert json.loads(json.dumps(run71.SPEC, allow_nan=False)) == run71.SPEC
