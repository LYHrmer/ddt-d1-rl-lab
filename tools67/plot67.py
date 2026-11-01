"""Render all64 models/points at registered H1/H32 from completed66/67 JSON only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODELS = tuple((seed, arm) for seed in (64001, 64002, 64003) for arm in ('control', 'active'))
POINTS = tuple((seed, tick) for seed in (64301, 64302, 64303) for tick in (110, 260, 410))
KINDS = ('free', 'cost_rebased', 'local')
STATES = (
    ('match', '+', '#b8dfd4', 'Direction match'),
    ('mismatch', 'x', '#efb2aa', 'Direction mismatch'),
    ('physical_unknown', 'P', '#dedede', 'Physical reference unresolved'),
    ('direction_unknown', 'D', '#f2d38b', 'Model direction unresolved'),
    ('numeric_unknown', 'N', '#c9b4e7', 'Numerical self-check unresolved'),
    ('actual_unknown', 'A', '#b6d7ef', 'Actual cost tangent unresolved'),
)
LOCAL = dict(zip(('LOCAL_DIRECTION_MATCH', 'LOCAL_DIRECTION_MISMATCH', 'REFERENCE_UNRESOLVED',
                  'MODEL_DIRECTION_UNRESOLVED', 'NUMERIC_UNRESOLVED', 'ACTUAL_COST_TANGENT_UNRESOLVED'),
                 (row[0] for row in STATES), strict=True))


def free_state(row):
    """Show numerical failure separately; never color ineligible values as errors."""
    if not row['autograd_agrees']:
        return 'numeric_unknown'
    if not row['physical_reference_resolved']:
        return 'physical_unknown'
    if not row['direction_identifiable']:
        return 'direction_unknown'
    assert type(row['valid_direction_match']) is bool, 'eligible direction must have a boolean result'
    return 'match' if row['valid_direction_match'] else 'mismatch'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report66', type=Path, required=True)
    parser.add_argument('--report67', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {'report66': args.report66, 'report67': args.report67, 'plot_source': Path(__file__)}
    contents = {name: path.read_bytes() for name, path in paths.items()}
    digests = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    reports = {name: json.loads(contents[name]) for name in ('report66', 'report67')}
    by_model = {}
    for name, report in reports.items():
        assert report['status'] == report['verification_after'] == 'COMPLETE' and report['diagnostics_complete']
        assert report['completed_model_points'] == 54 and len(report['models']) == 6
        by_model[name] = {(row['seed'], row['arm']): row for row in report['models']}
        assert set(by_model[name]) == set(MODELS)
        for model in report['models']:
            assert model['status'] == 'COMPLETE'
            assert [(point['seed'], point['tick']) for point in model['points']] == list(POINTS)
            assert all(point['status'] == 'COMPLETE' for point in model['points'])
    labels, matrices = [], {}
    for horizon in (1, 32):
        matrix = []
        for identity in MODELS:
            for kind in KINDS:
                states = []
                for point in range(9):
                    if kind == 'local':
                        row = by_model['report67'][identity]['points'][point]['cost_projection']['prefixes'][horizon - 1]
                        assert row['horizon'] == horizon
                        states.append(LOCAL[row['total']['status']])
                    else:
                        row = by_model['report66'][identity]['points'][point][kind]['prefixes'][horizon - 1]
                        assert row['horizon'] == horizon
                        states.append(free_state(row))
                matrix.append(states)
                if horizon == 1:
                    labels.append(f'{identity[0]} {identity[1]} / {kind.replace("_", "-")}')
        matrices[str(horizon)] = matrix
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'INCOMPLETE', 'input_sha256': digests, 'models': 6, 'points': 9,
               'horizons': [1, 32], 'row_labels': labels, 'point_order': POINTS, 'states': matrices,
               'classification_priority_free_rebased': ['numeric self-check', 'physical reference', 'model direction', 'match'],
               'model_inferences': 0, 'model_updates': 0, 'new_physics_steps': 0, 'is_control_gain_claim': False}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        lookup = {row[0]: index for index, row in enumerate(STATES)}
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.hashsalt': 'joint-response67-all-models'}):
            fig, axes = plt.subplots(1, 2, figsize=(17, 10.5), sharey=True)
            for axis, horizon in zip(axes, (1, 32), strict=True):
                matrix = matrices[str(horizon)]
                axis.imshow([[lookup[state] for state in row] for row in matrix],
                            cmap=ListedColormap([row[2] for row in STATES]), vmin=0, vmax=5, aspect='auto')
                for y, row in enumerate(matrix):
                    for x, state in enumerate(row):
                        axis.text(x, y, STATES[lookup[state]][1], ha='center', va='center', color='#17212b', fontsize=10)
                axis.set_title(f'H = {horizon} policy step' + ('s' if horizon > 1 else ''), fontsize=12, pad=12)
                axis.set_xticks(range(9), [f'{seed}\n{tick}' for seed, tick in POINTS])
                axis.set_yticks(range(18), labels)
                axis.set_xlabel('Physical reset seed / snapshot tick', labelpad=10)
                axis.set_xticks([value - .5 for value in range(10)], minor=True)
                axis.set_yticks([value - .5 for value in range(19)], minor=True)
                axis.grid(which='minor', color='white', linewidth=.8)
                axis.tick_params(which='both', length=0)
                for edge in (2.5, 5.5, 8.5, 11.5, 14.5):
                    axis.axhline(edge, color='#43515c', linewidth=.8)
                for spine in axis.spines.values():
                    spine.set_visible(False)
            fig.suptitle('Frozen models: free, cost-rebased and local-oracle diagnostics', x=.045, ha='left', fontsize=16, y=.975)
            fig.text(.045, .937, 'Registered H1/H32 views; all 9 points and all 6 models. All 32 prefixes remain in machine-readable evidence.',
                     color='#43515c', fontsize=10)
            fig.legend(handles=[Patch(facecolor=color, label=f'{symbol}  {label}') for _, symbol, color, label in STATES],
                       loc='lower center', bbox_to_anchor=(.56, .05), ncol=3, frameon=False, fontsize=9)
            fig.text(.045, .025, 'Offline diagnostic comparisons only: local uses observed trajectories. No policy update or control-performance claim.',
                     color='#43515c', fontsize=9)
            fig.subplots_adjust(left=.205, right=.985, top=.88, bottom=.19, wspace=.08)
            for suffix in ('png', 'svg'):
                fig.savefig(args.output / f'joint_response67_all_models.{suffix}', dpi=160, facecolor='white',
                            metadata={'Creator': 'tools67/plot67.py'} if suffix == 'svg' else None)
            plt.close(fig)
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digests[name] for name, path in paths.items())
        receipt['files'] = {path.name: {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                            for path in sorted(args.output.iterdir()) if path.suffix in ('.png', '.svg')}
        assert len(receipt['files']) == 2 and all(row['bytes'] < 5_000_000 for row in receipt['files'].values())
        receipt['status'] = 'COMPLETE'
    except BaseException as exc:
        receipt['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        with (args.output / 'render_receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(receipt['files']), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
