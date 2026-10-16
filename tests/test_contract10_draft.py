"""The proposed next budget is concrete, but a draft cannot start training."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools10'))
import supervise_scale10 as supervisor  # noqa: E402

DRAFT = ROOT/'training/configs/contract10_supervisor_draft.json'


def test_draft_budget_and_equivalence_are_the_declared_new_experiment():
    plan = json.loads(DRAFT.read_text())
    assert plan['status'] == 'DRAFT_NOT_AUTHORISED'
    assert plan['proposed_budget']['authorised'] is False
    assert plan['seed'] == 101 and plan['envs'] == 8
    assert plan['entry'] == 'tools09/train_d1_ppo09.py'
    assert plan['plans']['scale']['steps'] == 8388608
    assert plan['plans']['scale']['compare_rollouts'] == 1173
    assert plan['plans']['preflight']['compare_rollouts'] == 8
    assert plan['resources']['start_min_available_ram_bytes'] == 7*1024**3
    assert sum(row['steps'] for row in plan['plans'].values()) == plan['proposed_budget']['transitions']
    assert plan['proposed_budget']['transitions']*10+18000 == plan['proposed_budget']['native']
    assert all(row['output'].startswith('evidence/career_contract_10/') for row in plan['plans'].values())


def test_draft_is_rejected_before_any_resource_read_or_spawn(monkeypatch):
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **k: pytest.fail('draft must not spawn'))
    def forbidden():
        pytest.fail('draft must be rejected before resource admission')
    with pytest.raises(SystemExit, match='frozen release'):
        supervisor.run('preflight', DRAFT, memory_reader=forbidden)
