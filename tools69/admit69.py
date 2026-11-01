"""Read-only69 label/interface admission; no model, actor, fit or physics calls."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from tools69 import run69


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    report = {'status': 'INCOMPLETE', 'failure': None, 'model_updates': 0,
              'model_calls': 0, 'actor_calls': 0, 'physics_steps': 0}
    plan = None
    try:
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = run69.sha(args.plan)
        run69.verify69(plan)
        original = json.loads((run69.ROOT / run69.PLAN53).read_text())
        run69.verify(original)
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        run69.require(report['versions'] == original['versions'], 'package versions differ before numeric import')
        from tools64 import fit64
        from tools69 import fit69

        fit69.torch.set_num_threads(1)
        fit69.torch.set_num_interop_threads(1)
        labels = fit69.load_labels68(run69.ROOT / run69.LABEL68)
        sources = fit64.load_label_sources(run69.ROOT / run69.LABEL58, run69.ROOT / run69.LABEL60,
                                          run69.ROOT / run69.LABEL62)
        legacy = fit64.compose_labels(sources, 'active')
        run69.require(all(fit69.np.array_equal(labels[key], value) for key, value in legacy.items()),
                      '68 state/yaw arrays differ from64 active')
        hashes = fit69.label_hashes(labels)
        prior = json.loads((run69.ROOT / run69.OLD64 / 'report.json').read_text())
        run69.require(prior['status'] == prior['verification_after'] == 'COMPLETE', 'complete64 control identity required')
        controls = [row for row in prior['models'] if row['arm'] == 'active']
        run69.require([row['seed'] for row in controls] == list(run69.MODEL_SEEDS)
                      and all(row['fit']['status'] == 'COMPLETE' for row in controls), 'three64 active controls required')
        for row in controls:
            for key in ('labels_sha256', 'common58_labels_sha256', 'second_labels_sha256'):
                run69.require(row['fit'][key] == hashes[key], 'label mapping differs from old64: ' + key)
        report.update(label_hashes=hashes, label_rows=96, states=24, body_channels=2,
                      scalar_body_targets=192, qualified_state_channels=48,
                      control_reference_mapping_sha256={str(row['seed']): row['fit']['final_model_state_sha256']
                                                        for row in controls},
                      fixed_lambda=run69.FIXED_LAMBDA, body_coefficients=fit69.BODY_COEFFICIENT,
                      control_reproduction_requires_formal_fit=True,
                      old64_unknown_reference_remains_formal=True)
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - retain failed admission without retry.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if plan is not None:
                run69.verify69(plan)
                run69.require(run69.sha(args.plan) == report['plan_sha256'], 'plan changed during admission')
            report['verification_after'] = 'COMPLETE' if plan is not None else 'NOT_RUN'
        except Exception as exc:  # noqa: BLE001 - do not hide the initial failure.
            report.update(status='INCOMPLETE', verification_after='FAILED',
                          verification_failure=f'{type(exc).__name__}: {exc}')
        report['cpu_seconds'] = time.process_time() - started
        run69.write_json(args.output / 'report.json', report)
        print(json.dumps(report, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
