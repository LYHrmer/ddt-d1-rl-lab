#!/usr/bin/env python3
"""Publish the new numerical export check without rewriting contract 08's verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def build(export_dir, cpp_report, previous):
    export_dir, cpp_report, previous = map(Path, (export_dir, cpp_report, previous))
    metadata, sidecar, cpp, old = (read(export_dir/'metadata.json'),
                                   read(export_dir/'single_sample_export.json'),
                                   read(cpp_report), read(previous))
    if (sidecar['status'] != 'PASS' or cpp['status'] != 'PASS'
            or sidecar['upstream_metadata_sha256'] != sha(export_dir/'metadata.json')
            or cpp['policy_sha256'] != metadata['sha256']['policy_onnx']
            or cpp['policy_sha256'] != sha(export_dir/'policy.onnx')
            or cpp['fixture_sha256'] != sha(export_dir/'golden_fixtures.json')
            or sidecar['checkpoint_sha256'] != metadata['sha256']['checkpoint']):
        raise ValueError('export, metadata and C++ receipts are not the same successful artifacts')
    parity = metadata['golden_parity']
    sb3, python_ort, cpp_ort = (parity['sb3_predict_vs_torch_max_abs_error'],
                               parity['torch_vs_python_ort_max_abs_error'], cpp['max_absolute_error'])
    if (not all(math.isfinite(value) for value in (sb3, python_ort, cpp_ort))
            or sb3 > 1e-6 or python_ort > 1e-5 or cpp_ort > 1e-5
            or cpp['absolute_tolerance'] != 1e-5
            or sidecar['absolute_limits_unchanged'] != {'sb3_vs_torch': 1e-6, 'torch_vs_ort': 1e-5}
            or sidecar['prediction_path']['batch_size_used_for_gate'] != 1
            or parity['fixtures'] != 64 or cpp['onnx_inference_calls'] != 64
            or len(cpp['comparisons']) != 64 or not all(row['passed'] is True for row in cpp['comparisons'])):
        raise ValueError('the numerical gates or 64-fixture scope do not match')
    if (sidecar['physics_steps_executed'] != 0 or cpp['physics_steps'] != 0
            or cpp['native_steps_attempted'] != 0 or cpp['native_steps_completed'] != 0):
        raise ValueError('this engineering check must execute zero physics')
    return {'format': 'd1-export-engineering10-summary-v1', 'status': 'PASS',
            'scope': 'batch-one numerical deployment check; no new qualification',
            'checkpoint_sha256': sidecar['checkpoint_sha256'], 'onnx_sha256': cpp['policy_sha256'],
            'entry_sha256': sidecar['entry_sha256'],
            'upstream_exporter_sha256': sidecar['upstream_exporter_sha256'],
            'cpp_binary_sha256': cpp['runtime_binary_sha256'],
            'fixtures': 64, 'sb3_single_vs_torch_max_abs_error': sb3,
            'torch_vs_python_ort_max_abs_error': python_ort,
            'torch_vs_cpp_ort_max_abs_error': cpp_ort,
            'python_ort_version': parity['python_ort_version'], 'cpp_ort_version': cpp['onnxruntime_version'],
            'batch_diagnostic': sidecar['prediction_path'],
            'absolute_limits': sidecar['absolute_limits_unchanged'],
            'historical_contract08_export_gate': old['export_gate']['status'],
            'historical_contract08_qualification': old['qualification'],
            'new_training_transitions': 0, 'new_native_steps': 0,
            'receipt_sha256': {'metadata': sha(export_dir/'metadata.json'),
                               'new_entry_sidecar': sha(export_dir/'single_sample_export.json'),
                               'cpp_report': sha(cpp_report), 'contract08_summary': sha(previous)},
            'tool_sha256': sha(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--cpp-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build(args.export_dir, args.cpp_report, ROOT/'reports/contract08_summary.json')
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'fixtures': report['fixtures'],
                      'cpp_error': report['torch_vs_cpp_ort_max_abs_error'], 'new_native_steps': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
