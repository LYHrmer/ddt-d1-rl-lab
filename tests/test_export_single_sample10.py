"""Batch-one parity must retain the original numerical gate. No Torch or physics."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools10'))
import export_single_sample10 as entry  # noqa: E402


class BatchRoundingModel:
    def __init__(self, invalid=False):
        self.batches = []
        self.invalid = invalid

    def predict(self, inputs, deterministic=True):
        self.batches.append(len(inputs))
        value = 2e-6 if len(inputs) > 1 else 0.
        return np.full((len(inputs), 16), np.nan if self.invalid else value), None


def test_batch_rounding_is_reported_but_single_actions_reach_the_unchanged_gate():
    model = BatchRoundingModel()
    predictor = entry.SingleSamplePredictor(model)
    actions, state = predictor.predict(np.zeros((64, 627)))
    assert model.batches == [64]+[1]*64
    assert np.array_equal(actions, np.zeros((64, 16))) and state is None
    assert predictor.diagnostics['batch_vs_single_max_abs_error'] == 2e-6
    assert predictor.diagnostics['batch_diagnostic_used_for_gate'] is False


def test_non_finite_predictions_are_not_accepted():
    with pytest.raises(ValueError, match='non-finite'):
        entry.SingleSamplePredictor(BatchRoundingModel(invalid=True)).predict(np.zeros((2, 627)))


def test_a_changed_upstream_is_rejected_before_importing_its_dependencies(tmp_path):
    path = tmp_path/entry.UPSTREAM
    path.parent.mkdir()
    path.write_text('raise AssertionError("must not import changed exporter")')
    with pytest.raises(ValueError, match='SHA256 changed'):
        entry.load_exporter(tmp_path)


def fake_exporter(output, model, *, fail=False):
    def golden(actor, predictor, path):
        actions, _ = predictor.predict(np.zeros((64, 627)))
        assert np.max(np.abs(actions)) <= 1e-6
        if fail:
            raise ValueError('ONNX parity failed: Torch/ORT=2e-5')
        return [{'torch_actions': [0.]*16, 'python_ort_actions': [0.]*16}], 0., 0.
    base = SimpleNamespace(golden_parity=golden)
    def main():
        output.mkdir()
        base.golden_parity(None, model, output/'policy.onnx')
        (output/'metadata.json').write_text('{"original": true}\n')
    base.main = main
    return base


def test_the_wrapper_records_its_identity_without_rewriting_upstream_metadata(tmp_path):
    checkpoint, config, output = tmp_path/'policy.zip', tmp_path/'config.json', tmp_path/'export'
    checkpoint.write_bytes(b'checkpoint')
    config.write_text('{}')
    model = BatchRoundingModel()
    base = fake_exporter(output, model)
    original, argv = base.golden_parity, sys.argv
    result = entry.export(checkpoint, config, output, exporter=base)
    assert base.golden_parity is original and sys.argv is argv
    assert (output/'metadata.json').read_text() == '{"original": true}\n'
    assert result['upstream_metadata_sha256'] == entry.sha(output/'metadata.json')
    assert result['absolute_limits_unchanged'] == {'sb3_vs_torch': 1e-6, 'torch_vs_ort': 1e-5}
    assert result['prediction_path']['batch_size_used_for_gate'] == 1
    assert result['physics_steps_executed'] == 0
    assert json.loads((output/'single_sample_export.json').read_text()) == result
    with pytest.raises(FileExistsError):
        entry.export(checkpoint, config, output, exporter=base)


def test_a_real_single_sample_mismatch_still_fails_and_restores_the_exporter(tmp_path):
    checkpoint, config, output = tmp_path/'policy.zip', tmp_path/'config.json', tmp_path/'export'
    checkpoint.write_bytes(b'checkpoint')
    config.write_text('{}')
    base = fake_exporter(output, BatchRoundingModel(), fail=True)
    original, argv = base.golden_parity, sys.argv
    with pytest.raises(ValueError, match='ONNX parity failed'):
        entry.export(checkpoint, config, output, exporter=base)
    assert base.golden_parity is original and sys.argv is argv
    assert not (output/'single_sample_export.json').exists()
