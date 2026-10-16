#!/usr/bin/env python3
"""Why the frozen exporter refused the contract 08 policy: a zero-physics diagnosis.

The frozen exporter compares the SB3 deterministic action with its own Torch actor on
64 fixed random fixtures and refuses above an absolute 1e-6. float32 rounding grows with
the size of the numbers involved, so this receipt puts the observed difference beside
the output magnitude and the largest weight, for the 8x policy and for the frozen 03d
policy that passed. It loads policies and runs their networks; it builds no environment
and executes no physics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'training'))

SB3_TORCH_LIMIT = 1e-6
TORCH_ORT_LIMIT = 1e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarise(differences, raw_actions, limit=SB3_TORCH_LIMIT):
    """Plain numbers from the two arrays: how big the error is, against how big the output is."""
    differences, raw_actions = np.abs(np.asarray(differences)), np.abs(np.asarray(raw_actions))
    peak = float(raw_actions.max())
    eps = float(np.finfo(np.float32).eps)
    return {'max_abs_difference': float(differences.max()), 'elements': int(differences.size),
            'elements_above_limit': int((differences > limit).sum()), 'limit': limit,
            'max_abs_unclipped_action': peak, 'float32_eps_times_peak': eps*peak,
            'difference_over_eps_times_peak': float(differences.max())/(eps*peak) if peak else None}


def measure(checkpoint):
    import torch
    from stable_baselines3 import PPO
    import export_d1_ppo as exporter
    from d1_rl.interface import HISTORY_LENGTH, OBS_DIM
    model = PPO.load(str(checkpoint), device='cpu')
    actor = exporter.DeterministicActor(model.policy).eval()
    rng = np.random.default_rng(exporter.FIXTURE_SEED)
    obs = rng.normal(0, .5, size=(exporter.FIXTURE_COUNT, OBS_DIM)).astype(np.float32)
    history = rng.normal(0, .5, size=(exporter.FIXTURE_COUNT, HISTORY_LENGTH, OBS_DIM)).astype(np.float32)
    obs[0] = 0
    history[0] = 0
    flat = np.concatenate((obs, history.reshape(exporter.FIXTURE_COUNT, -1)), axis=1)
    sb3, _ = model.predict(flat, deterministic=True)
    with torch.no_grad():
        torch_actions = np.concatenate([actor(torch.from_numpy(obs[i:i+1]), torch.from_numpy(history[i:i+1])).numpy()
                                        for i in range(exporter.FIXTURE_COUNT)])
        raw = model.policy.get_distribution(model.policy.obs_to_tensor(flat)[0]).distribution.mean.numpy()
        weight = max(float(p.detach().abs().max()) for p in model.policy.parameters())
    out = summarise(sb3-torch_actions, raw)
    out['max_abs_weight'] = weight
    out['checkpoint_sha256'] = sha(checkpoint)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', action='append', default=[], metavar='LABEL=PATH')
    parser.add_argument('--export-error', required=True, help='the exporter error line, verbatim')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    results = {label: measure(Path(path)) for label, path in (item.split('=', 1) for item in args.checkpoint)}
    report = {'format': 'd1-contract08-export-diagnosis-v1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physics_steps_executed': 0, 'exporter_sha256': sha(ROOT/'tools/export_d1_ppo.py'),
              'exporter_limits': {'sb3_vs_torch': SB3_TORCH_LIMIT, 'torch_vs_python_ort': TORCH_ORT_LIMIT},
              'export_error_verbatim': args.export_error, 'policies': results, 'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: {m: v[m] for m in ('max_abs_difference', 'elements_above_limit', 'max_abs_unclipped_action',
                                             'difference_over_eps_times_peak', 'max_abs_weight')}
                      for k, v in results.items()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
