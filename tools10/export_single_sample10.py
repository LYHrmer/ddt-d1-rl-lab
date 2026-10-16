#!/usr/bin/env python3
"""Export with the deployed batch-one SB3 path, preserving both frozen tolerances.

The frozen exporter compares batch-64 SB3 with batch-one Torch. This new entry
reuses it unchanged and supplies single-sample SB3 predictions instead. Batch
rounding remains a reported diagnostic. A sidecar binds the new entry to the
unchanged upstream metadata; old export failures are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'tools/export_d1_ppo.py'
UPSTREAM_SHA256 = '1563d6a96dbd044d844b14a54ab273c754eb8272c0d8ec6ccda774182c6cb27a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_exporter(root=ROOT):
    path = Path(root)/UPSTREAM
    if sha(path) != UPSTREAM_SHA256:
        raise ValueError('frozen upstream exporter SHA256 changed')
    spec = importlib.util.spec_from_file_location('_frozen_exporter10', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SingleSamplePredictor:
    def __init__(self, model):
        self.model = model
        self.diagnostics = {}

    def predict(self, inputs, deterministic=True):
        if deterministic is not True:
            raise ValueError('export parity requires deterministic prediction')
        inputs = np.asarray(inputs)
        if inputs.ndim != 2 or not len(inputs) or not np.isfinite(inputs).all():
            raise ValueError('invalid export fixtures')
        batch, _ = self.model.predict(inputs, deterministic=True)
        single = np.concatenate([self.model.predict(row[None], deterministic=True)[0] for row in inputs])
        if (single.shape != (len(inputs), 16) or np.asarray(batch).shape != single.shape
                or not np.isfinite(single).all() or not np.isfinite(batch).all()):
            raise ValueError('invalid or non-finite SB3 actions')
        self.diagnostics = {'batch_size_used_for_gate': 1, 'batch_size_diagnostic': len(inputs),
                            'batch_vs_single_max_abs_error': float(np.abs(batch-single).max()),
                            'batch_diagnostic_used_for_gate': False}
        return single, None


def export(checkpoint, config, output, exporter=None):
    checkpoint, config, output = map(lambda p: Path(p).resolve(), (checkpoint, config, output))
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    base = load_exporter() if exporter is None else exporter
    original, argv = base.golden_parity, sys.argv
    diagnostics = {}

    def parity(actor, model, onnx_path):
        predictor = SingleSamplePredictor(model)
        fixtures, torch_ort, sb3_torch = original(actor, predictor, onnx_path)
        if not np.isfinite([torch_ort, sb3_torch]).all():
            raise ValueError('non-finite export parity error')
        for fixture in fixtures:
            if not all(np.isfinite(fixture[key]).all() for key in ('torch_actions', 'python_ort_actions')):
                raise ValueError('non-finite export fixture actions')
        diagnostics.update(predictor.diagnostics)
        return fixtures, torch_ort, sb3_torch

    try:
        base.golden_parity = parity
        sys.argv = [str(ROOT/UPSTREAM), '--checkpoint', str(checkpoint),
                    '--config', str(config), '--output', str(output)]
        base.main()
    finally:
        base.golden_parity, sys.argv = original, argv
    metadata = output/'metadata.json'
    receipt = {'format': 'd1-single-sample-export-v1', 'status': 'PASS',
               'scope': 'numerical export validation only; no control qualification',
               'physics_steps_executed': 0, 'checkpoint_sha256': sha(checkpoint),
               'upstream_exporter_sha256': UPSTREAM_SHA256, 'entry_sha256': sha(__file__),
               'upstream_metadata_sha256': sha(metadata), 'prediction_path': diagnostics,
               'absolute_limits_unchanged': {'sb3_vs_torch': 1e-6, 'torch_vs_ort': 1e-5}}
    with (output/'single_sample_export.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = export(args.checkpoint, args.config, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
