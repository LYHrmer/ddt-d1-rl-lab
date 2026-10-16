"""Published numerical deployment checks must bind the same artifacts. No physics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from _workstation import needs_workstation_evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools10'))
import summarize_export10 as summary  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def receipts(tmp_path):
    export = tmp_path/'export'
    export.mkdir()
    (export/'policy.onnx').write_bytes(b'network')
    (export/'golden_fixtures.json').write_text('{}')
    metadata = {'sha256': {'policy_onnx': summary.sha(export/'policy.onnx'), 'checkpoint': 'original-weight'},
                'golden_parity': {'sb3_predict_vs_torch_max_abs_error': 0.,
                                  'torch_vs_python_ort_max_abs_error': 1.3e-6,
                                  'fixtures': 64, 'python_ort_version': '1.23.2'}}
    write(export/'metadata.json', metadata)
    sidecar = {'status': 'PASS', 'upstream_metadata_sha256': summary.sha(export/'metadata.json'),
               'checkpoint_sha256': 'original-weight', 'entry_sha256': 'new-entry',
               'upstream_exporter_sha256': 'frozen-entry', 'physics_steps_executed': 0,
               'absolute_limits_unchanged': {'sb3_vs_torch': 1e-6, 'torch_vs_ort': 1e-5},
               'prediction_path': {'batch_size_used_for_gate': 1}}
    write(export/'single_sample_export.json', sidecar)
    cpp = tmp_path/'cpp.json'
    write(cpp, {'status': 'PASS', 'policy_sha256': metadata['sha256']['policy_onnx'],
                'fixture_sha256': summary.sha(export/'golden_fixtures.json'),
                'max_absolute_error': 1.3e-6, 'absolute_tolerance': 1e-5,
                'onnx_inference_calls': 64, 'comparisons': [{'passed': True}]*64,
                'physics_steps': 0, 'native_steps_attempted': 0, 'native_steps_completed': 0,
                'runtime_binary_sha256': 'frozen-runtime', 'onnxruntime_version': '1.10.0'})
    old = tmp_path/'old.json'
    write(old, {'export_gate': {'status': 'FAIL'}, 'qualification': {'tasks_passed': 1, 'overall': 'FAIL'}})
    return export, cpp, old


def test_new_numerical_pass_does_not_rewrite_historical_failure(receipts):
    out = summary.build(*receipts)
    assert out['status'] == 'PASS'
    assert out['historical_contract08_export_gate'] == 'FAIL'
    assert out['historical_contract08_qualification']['overall'] == 'FAIL'
    assert out['new_native_steps'] == out['new_training_transitions'] == 0


@pytest.mark.parametrize('field, value, message', [
    ('policy_sha256', 'different', 'same successful artifacts'),
    ('onnx_inference_calls', 1, '64-fixture'),
    ('max_absolute_error', 2e-5, 'numerical gates'),
    ('max_absolute_error', float('nan'), 'numerical gates'),
    ('native_steps_attempted', 1, 'zero physics'),
])
def test_inconsistent_cpp_receipts_cannot_be_published_as_success(receipts, field, value, message):
    _, cpp, _ = receipts
    damaged = summary.read(cpp)
    damaged[field] = value
    write(cpp, damaged)
    with pytest.raises(ValueError, match=message):
        summary.build(*receipts)


@needs_workstation_evidence
def test_published_export_result_matches_real_network_and_cpp_receipts():
    ev = ROOT/'evidence/engineering10'
    if not (ev/'cxx_single_sample_golden_01/report.json').is_file():
        pytest.skip('network-only engineering receipts not on this machine')
    assert summary.read(ROOT/'reports/export_engineering10_summary.json') == summary.build(
        ev/'export_single_sample_01', ev/'cxx_single_sample_golden_01/report.json',
        ROOT/'reports/contract08_summary.json')
