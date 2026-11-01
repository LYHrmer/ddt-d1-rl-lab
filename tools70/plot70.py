"""Plot the complete70 all27 FD-MAE summary without model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SEEDS = (64001, 64002, 64003)
ARMS = ('control', 'active')
COMPONENTS = ('vx', 'yaw', 'lateral', 'tilt', 'height', 'action_smoothness')
EPS = (.00003125, .0000625, .000125, .00025)
COLORS = {'control': '#2166ac', 'active': '#d6601a'}
CLASSIFICATIONS = ('ALL_THREE_NUMERICALLY_LOWER', 'MIXED_OR_TIED',
                   'ALL_THREE_NUMERICALLY_LOWER', 'ALL_THREE_NUMERICALLY_HIGHER',
                   'ALL_THREE_NUMERICALLY_HIGHER', 'ALL_EQUAL')


def plot_data(report):
    """Validate complete coverage and read all six terms in the frozen order."""
    assert report['schema'] == 'd1-cost-decomposition70-result-v1'
    assert report['status'] == report['verification_after'] == 'COMPLETE'
    assert report['failure'] is None and report['summary']['status'] == 'COMPLETE'
    assert (report['completed_model_points'], report['completed_intervals'],
            report['completed_component_pairs']) == (162, 648, 3888)
    assert tuple(report['components']) == COMPONENTS
    cases = {(role, seed, tick)
             for role, seeds in (('old56', (56301, 56302, 56303)),
                                 ('old64', (64301, 64302, 64303)),
                                 ('fresh', (69301, 69302, 69303)))
             for seed in seeds for tick in (110, 260, 410)}
    expected = {(seed, arm, *case) for seed in SEEDS for arm in ARMS for case in cases}
    assert len(report['points']) == 162
    assert {(p['model_seed'], p['arm'], p['role'], p['seed'], p['tick'])
            for p in report['points']} == expected
    for point in report['points']:
        assert point['status'] == 'COMPLETE' and len(point['eps']) == 4
        assert tuple(row['epsilon_true'] for row in point['eps']) == EPS
        for row in point['eps']:
            assert row['status'] == 'COMPLETE' and row['epsilon_model'] == .001
            assert all(len(row[key]) == 6 for key in ('model_fd', 'true_fd', 'difference_fd'))
    rows = [row for row in report['summary']['metric_rows'] if row['role'] == 'all27']
    indexed = {(row['model_seed'], row['arm'], row['component']): row for row in rows}
    assert len(rows) == 36 and set(indexed) == {
        (seed, arm, component) for seed in SEEDS for arm in ARMS for component in COMPONENTS}
    values = {str(seed): {arm: [] for arm in ARMS} for seed in SEEDS}
    for (seed, arm, _component), row in indexed.items():
        assert (row['points'], row['intervals']) == (27, 108)
        assert math.isfinite(row['fd_absolute_error']) and row['fd_absolute_error'] > 0
    paired = [row for row in report['summary']['paired_component_FD_MAE'] if row['role'] == 'all27']
    assert len(paired) == 6 and {row['component'] for row in paired} == set(COMPONENTS)
    paired = {row['component']: row for row in paired}
    for component, classification in zip(COMPONENTS, CLASSIFICATIONS, strict=True):
        row = paired[component]
        assert row['classification'] == classification
        pairs = {pair['model_seed']: pair for pair in row['pairs']}
        assert len(row['pairs']) == 3 and set(pairs) == set(SEEDS)
        for seed in SEEDS:
            for arm in ARMS:
                value = indexed[seed, arm, component]['fd_absolute_error']
                assert value == pairs[seed][arm + '_fd_MAE']
                values[str(seed)][arm].append(value)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, script = args.report.read_bytes(), Path(__file__).read_bytes()
    values = plot_data(json.loads(source))
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'INCOMPLETE', 'source_report_sha256': hashlib.sha256(source).hexdigest(),
               'plot_source_sha256': hashlib.sha256(script).hexdigest(), 'model_seeds': list(SEEDS),
               'component_order': COMPONENTS, 'arms': ARMS, 'colors': COLORS, 'fd_MAE': values,
               'role': 'all27', 'states_per_model': 27, 'true_epsilon_grid': EPS,
               'model_epsilon': .001, 'fd_intervals_differ': True, 'y_scale': 'log',
               'model_points': 162, 'intervals': 648, 'component_pairs': 3888,
               'model_inferences': 0, 'physics_steps': 0, 'model_updates': 0,
               'is_controller_performance': False, 'qualification_changed': False,
               'gate_pass_claimed': False, 'selected_seed': None}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'svg.hashsalt': 'cost-decomposition70-all27-fixed-order'}):
            fig, axes = plt.subplots(1, 3, figsize=(15, 6.3), sharey=True)
            for axis, seed in zip(axes, SEEDS, strict=True):
                pair = values[str(seed)]
                for index in range(6):
                    axis.plot([index - .12, index + .12],
                              [pair['control'][index], pair['active'][index]],
                              color='#aab2b9', linewidth=1.1, zorder=2)
                for arm, offset, marker, text_offset in (
                        ('control', -.12, 'o', 10), ('active', .12, 's', -16)):
                    axis.scatter([index + offset for index in range(6)], pair[arm],
                                 color=COLORS[arm], marker=marker, s=37, zorder=3)
                    for index, value in enumerate(pair[arm]):
                        axis.annotate(f'{value:.2e}', (index + offset, value),
                                      xytext=(0, text_offset), textcoords='offset points',
                                      ha='center', fontsize=8, color=COLORS[arm])
                axis.set_title(f'Paired model seed {seed}', loc='left', fontsize=12, pad=14)
                axis.set_yscale('log')
                axis.set_ylim(1e-8, 2e-3)
                axis.set_xlim(-.55, 5.55)
                axis.set_xticks(range(6), ('vx', 'yaw', 'lateral', 'tilt', 'height', 'action\nsmoothness'))
                axis.set_xlabel('Cost component (fixed order)')
                axis.grid(axis='y', which='major', alpha=.23, zorder=0)
            axes[0].set_ylabel('Mean absolute FD error (log scale; lower is better)')
            fig.suptitle('H1 cost-component error across all 27 states', x=.065, ha='left', y=.975, fontsize=17)
            fig.text(.065, .925, 'vx / lateral decrease; tilt / height increase; yaw is mixed; action smoothness is unchanged.',
                     fontsize=11, color='#43515c')
            fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=COLORS['control'], label='Control: state + yaw'),
                                Line2D([], [], marker='s', linestyle='', color=COLORS['active'], label='Active: state + yaw + body')],
                       loc='upper center', bbox_to_anchor=(.53, .89), ncol=2, frameon=False)
            fig.subplots_adjust(left=.075, right=.98, top=.77, bottom=.28, wspace=.20)
            fig.text(.065, .16, 'Each point averages 27 states x 4 physical epsilon values equally. All 6 terms and all 3 paired seeds are shown.',
                     fontsize=9, color='#43515c')
            fig.text(.065, .115, 'Different FD intervals: model +/-0.001; physical +/-{3.125e-5, 6.25e-5, 1.25e-4, 2.5e-4}.',
                     fontsize=9, color='#43515c')
            fig.text(.065, .07, 'Finite-interval model diagnostics only: not controller performance, statistical significance, or a qualification-gate pass.',
                     fontsize=9, color='#43515c')
            for suffix in ('png', 'svg'):
                fig.savefig(args.output / f'cost_decomposition70_all27.{suffix}', dpi=170, facecolor='white',
                            metadata={'Creator': 'tools70/plot70.py'} if suffix == 'svg' else None)
            plt.close(fig)
        assert args.report.read_bytes() == source and Path(__file__).read_bytes() == script
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
