"""Read-only64 data/interface admission: zero fits, model updates and physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools64 import fit64, run64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'INCOMPLETE', 'failure': None, 'model_updates': 0, 'physics_steps': 0}
    try:
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = run64.sha(args.plan)
        run64.verify64(plan)
        fit64.torch.set_num_threads(1)
        fit64.torch.set_num_interop_threads(1)
        sources = fit64.load_label_sources(run64.ROOT / run64.LABEL58, run64.ROOT / run64.LABEL60,
                                          run64.ROOT / run64.LABEL62)
        report['sources'] = {key: {'rows': len(value['target']),
                                   'mapping_sha256': fit64.paired.mapping_sha256(value),
                                   'positive_rows': int((value['target'] > 0).sum()),
                                   'negative_rows': int((value['target'] < 0).sum())}
                             for key, value in sources.items()}
        report['arms'] = {arm: {'rows': len(labels['target']),
                                'mapping_sha256': fit64.paired.mapping_sha256(labels)}
                          for arm in fit64.ARMS for labels in (fit64.compose_labels(sources, arm),)}
        run64.verify64(plan)
        fit64.require(run64.sha(args.plan) == report['plan_sha256'], 'plan changed during admission')
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - preserve failed admission without retry.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        run64.write_json(args.output / 'report.json', report)
        print(json.dumps(report, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
