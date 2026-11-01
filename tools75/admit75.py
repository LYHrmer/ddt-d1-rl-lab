"""Read-only75 input/interface admission before six fits; no model/actor/physics."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import signal
import time
from pathlib import Path

from tools75 import run75 as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    report = {'status': 'INCOMPLETE', 'failure': None, 'model_updates': 0,
              'model_calls': 0, 'actor_calls': 0, 'physics_steps': 0, 'consumed53_inputs': {}}
    plan = None
    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU):
        signal.signal(signum, interrupted)
    try:
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = run.sha(args.plan)
        run.verify75(plan)
        original = json.loads((run.ROOT / run.PLAN53).read_text())
        run.verify(original)
        summary = json.loads((run.ROOT / run.SUMMARY53).read_text())
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        run.require(report['versions'] == original['versions'], 'package versions differ before numeric import')
        from tools64 import fit64
        from tools75 import fit75 as fit
        fit.torch.set_num_threads(1)
        fit.torch.set_num_interop_threads(1)
        report['torch_runtime_version'] = fit.torch.__version__
        labels = fit.load_labels75(run.ROOT / run.LABEL75)
        sources = fit64.load_label_sources(run.ROOT / run.LABEL58, run.ROOT / run.LABEL60, run.ROOT / run.LABEL62)
        legacy = fit64.compose_labels(sources, 'active')
        run.require(all(fit.np.array_equal(labels[key], value) for key, value in legacy.items()), '68 labels changed')
        previous, controls = run.prior69()
        run.require(report['versions'] == previous['versions']
                    and report['torch_runtime_version'] == previous['torch_runtime_version'], '69 runtime differs')
        path = run.bound53('execution_01/train_transitions.npz', summary['raw_manifest'], report['consumed53_inputs'])
        with fit.np.load(path, allow_pickle=False) as raw:
            train = fit.frozen.validate_data({key: raw[key].copy() for key in fit.paired.DATA_KEYS})
        windows = fit.paired.common_window_indices(train).astype('<i8', copy=False)
        run.require(len(train['s']) == 7200 and windows.shape == (6828, 32), 'full original training grid required')
        hashes = {'training_data_sha256': fit.paired.mapping_sha256(train), **fit.label_hashes(labels),
                  'common_windows_sha256': fit.paired.array_sha256(windows),
                  'planned_sampling_rows_sha256': fit.paired.array_sha256(
                      fit.paired.sampled_window_rows(6828).numpy().astype('<i8', copy=False))}
        for row in previous['models']:
            if row['arm'] == 'active':
                run.require(all(row['fit'][key] == value for key, value in hashes.items()
                                if key not in ('world_vz_labels_sha256', 'combined75_labels_sha256')), '69 fit input differs')
        points, targets, costs, grids, first, heads, world = run.historical75(original, summary, report['consumed53_inputs'])
        run.require(set(points) == set(targets) == set(run.ROLES[:-1])
                    and set(costs) == set(run.COST_ROLES[:-1])
                    and set(grids) == set(first) == set(heads) == set(world) == set(run.FIRST_ROLES[:-1]), 'historical role missing')
        report.update(input_hashes=hashes, label_rows=96, states=24, body_channels=2,
                      control_reference_mapping_sha256=controls, historical_point_roles=list(points),
                      historical_first_head_roles=list(heads), old69_references_exact=True,
                      yaw_coefficients=run.YAW_COEFFICIENT, body_coefficients=run.BODY_COEFFICIENT, world_vz_coefficients=run.WORLD_COEFFICIENT,
                      world_vz_reference_roles=list(world),
                      control_reproduction_requires_formal_fit=True, status='COMPLETE')
    except BaseException as exc:  # noqa: BLE001 - preserve failed admission without retry.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if plan is not None:
                run.verify75(plan)
                run.require(run.sha(args.plan) == report['plan_sha256'], 'plan changed during admission')
            report['verification_after'] = 'COMPLETE' if plan is not None else 'NOT_RUN'
            if 'versions' in report:
                report['versions_after'] = {name: importlib.metadata.version(name) for name in report['versions']}
                run.require(report['versions_after'] == report['versions'], 'versions changed')
        except Exception as exc:  # noqa: BLE001 - retain initial and final verification failures.
            report.update(status='INCOMPLETE', verification_after='FAILED', verification_failure=f'{type(exc).__name__}: {exc}')
        report['cpu_seconds'] = time.process_time() - started
        run.write_json(args.output / 'report.json', report)
        print(json.dumps(report, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
