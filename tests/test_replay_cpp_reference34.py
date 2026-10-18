import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools34 import replay_cpp_reference34 as replay  # noqa: E402


def make_execution(tmp_path, *, qualified=True):
    execution = tmp_path / 'execution'
    (execution / 'formal').mkdir(parents=True)
    raw = np.arange(2 * 627, dtype=np.float32).reshape(2, 627) / 1000
    issued = np.full((2, 16), .25, np.float32)
    sidecar = execution / 'formal' / 'a_cpp_oracle.npz'
    np.savez_compressed(sidecar, raw_actor627=raw, cpp_issued16=issued)
    case = {'trace_file': 'a.npz', 'cpp_reference34': {
        'sidecar': sidecar.name, 'sha256': replay.sha(sidecar),
        'transport': {'identities': {'binary_sha256': 'b' * 64, 'model_sha256': 'c' * 64, 'source_sha256': 'd' * 64}}}}
    report = {'status': 'COMPLETE', 'error': None, 'cpp_original_and_holdout_qualified': qualified,
              'automatic_model_replacement': False, 'cpp_build': {'binary': '/x/actor', 'binary_sha256': 'b' * 64},
              'stages': {'formal': {'cases': [case]}, 'holdout': {'cases': []}}}
    (execution / 'report.json').write_text(json.dumps(report))
    return execution, report, case


def test_unqualified_execution_is_refused(tmp_path):
    execution, _, _ = make_execution(tmp_path, qualified=False)
    with pytest.raises(ValueError):
        replay.load_execution(execution)


def test_select_filters_by_episode_and_limit(tmp_path):
    _, report, _ = make_execution(tmp_path)
    assert [s for _, s, _ in replay.select(report, ['formal'], ['a'], None)] == ['a']
    with pytest.raises(ValueError):
        replay.select(report, ['formal'], ['missing'], None)


class FakeActor:
    outputs = None

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def start(self):
        return self

    def predict(self, row):
        value = self.outputs[self.calls]
        self.calls += 1
        return value

    def close(self):
        return {'status': 'CLOSED', 'cleanup_complete': True, 'requests_completed': self.calls,
                'inference_returns': self.calls}


@pytest.mark.parametrize('perturb', [0., 1e-7])
def test_replay_requires_bitwise_equal_actions(tmp_path, monkeypatch, perturb):
    execution, report, case = make_execution(tmp_path)
    model = tmp_path / 'policy.onnx'
    model.write_bytes(b'm')
    monkeypatch.setattr(replay, 'sha', lambda path: 'c' * 64 if Path(path) == model else replay.transport.sha(path))
    FakeActor.outputs = [np.full(16, .25 + perturb, np.float32)] * 2
    monkeypatch.setattr(replay, 'CppReferenceActor', FakeActor)
    result = replay.replay_case(execution, tmp_path / 'out', report, 'formal', 'a', case, model)
    assert result['requests'] == 2 and result['cleanup_complete']
    assert result['actions_bitwise_equal_to_saved'] is (perturb == 0.)
    assert (result['mismatched_ticks'] == 0) is (perturb == 0.)


def test_changed_sidecar_is_refused(tmp_path, monkeypatch):
    execution, report, case = make_execution(tmp_path)
    case['cpp_reference34']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        replay.replay_case(execution, tmp_path / 'out', report, 'formal', 'a', case, tmp_path / 'policy.onnx')
