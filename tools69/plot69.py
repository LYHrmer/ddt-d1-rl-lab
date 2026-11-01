"""Render the complete69 paired diagnostics from JSON; no model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SEEDS = (64001, 64002, 64003)
ARMS = ('control', 'active')
COLORS = {'control': '#2166ac', 'active': '#d6601a'}
METRICS = (
    ('body_normalized_rmse', 'Body response normalized RMSE', 'Lower is better', 1000., None),
    ('cost_valid_matches', 'H32 cost direction', 'Higher is better', 1., 45),
    ('body_valid_matches', 'Body response direction', 'Higher is better', 1., 54),
    ('cost_h1_valid_matches', 'H1 cost direction', 'Higher is better', 1., 27),
)


def plot_data(report):
    """Keep all fixed seeds; count metrics already include source qualification."""
    assert report['status'] == report['verification_after'] == 'COMPLETE'
    assert report['completed_model_fits'] == 6 and len(report['models']) == 6
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    assert {(row['seed'], row['arm']) for row in report['models']} == expected
    assert all(row['status'] == row['fit']['status'] == 'COMPLETE' for row in report['models'])
    comparison = report['comparison']
    assert comparison['status'] == 'COMPLETE' and not comparison['eligibility_pending_final_audit']
    assert (comparison['cost_points'], comparison['body_channels'], comparison['cost_h1_points']) == (45, 54, 27)
    models = {(row['seed'], row['arm']): row for row in comparison['models']}
    assert len(comparison['models']) == 6 and set(models) == expected
    values = {}
    for metric, _title, _direction, _scale, denominator in METRICS:
        values[metric] = {arm: [models[seed, arm][metric] for seed in SEEDS] for arm in ARMS}
        for series in values[metric].values():
            assert all(math.isfinite(value) and value >= 0 for value in series)
            if denominator is not None:
                assert all(type(value) is int and value <= denominator for value in series)
    assert all(row['first_step_valid_matches'] == 26 for row in models.values())
    assert all(sum(role['resolved_count'] for role in row['cost'].values()) == 44 for row in models.values())
    paired = {row['seed']: row for row in comparison['paired']}
    assert len(comparison['paired']) == 3 and set(paired) == set(SEEDS)
    for seed in SEEDS:
        for arm in ARMS:
            assert models[seed, arm]['body_normalized_rmse'] == paired[seed][arm + '_body_normalized_rmse']
            assert models[seed, arm]['cost_valid_matches'] == paired[seed][arm + '_cost_valid_matches_of45']
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.report.read_bytes()
    script = Path(__file__).read_bytes()
    report = json.loads(source)
    values = plot_data(report)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'INCOMPLETE', 'source_report_sha256': hashlib.sha256(source).hexdigest(),
               'plot_source_sha256': hashlib.sha256(script).hexdigest(), 'model_seeds': list(SEEDS),
               'arms': list(ARMS), 'colors': COLORS, 'unscaled_values': values,
               'rmse_display_multiplier': 1000., 'all_y_axes_start_at_zero': True,
               'yaw_valid_each_model': '26/27', 'h32_formal_denominator': 45, 'h32_resolved_references': 44,
               'h32_unknown_references': 1, 'qualification_changed': False,
               'source_target_status': report['comparison']['target_status'],
               'is_controller_performance': False, 'selected_seed': None,
               'model_inferences': 0, 'model_updates': 0, 'new_physics_steps': 0}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import MaxNLocator
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'svg.hashsalt': 'body-jvp69-fixed-three-pairs'}):
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            for axis, (metric, title, direction, scale, denominator) in zip(axes.flat, METRICS, strict=True):
                for arm, offset in (('control', -.18), ('active', .18)):
                    heights = [value * scale for value in values[metric][arm]]
                    bars = axis.bar([index + offset for index in range(3)], heights, width=.32,
                                    color=COLORS[arm], zorder=3)
                    labels = [f'{value:.4f}' if denominator is None else f'{int(value)}/{denominator}' for value in heights]
                    axis.bar_label(bars, labels=labels, padding=5, fontsize=9, color='#202a33')
                axis.set_title(f'{title}\n{direction}', loc='left', fontsize=12, pad=11)
                axis.set_xticks(range(3), [str(seed) for seed in SEEDS])
                axis.set_xlabel('Paired model seed')
                axis.grid(axis='y', alpha=.18, zorder=0)
                axis.set_axisbelow(True)
                if denominator is None:
                    maximum = max(value * scale for arm in ARMS for value in values[metric][arm])
                    axis.set_ylim(0, maximum * 1.22 if maximum else 1.)
                    axis.set_ylabel('Normalized RMSE (x 1e-3)')
                    axis.yaxis.set_major_locator(MaxNLocator(5))
                else:
                    axis.set_ylim(0, denominator * 1.16)
                    axis.set_yticks(list(range(0, denominator + 1, 9)))
                    axis.set_ylabel('Valid direction matches')
                    axis.axhline(denominator, color='#87929b', linewidth=.7, linestyle=':', zorder=1)
            fig.suptitle('Body-response objective: paired model diagnostics', x=.07, ha='left', fontsize=17, y=.975)
            fig.text(.07, .935, 'All three fixed seeds. The active objective adds body-vx/vy response loss; state and yaw terms stay fixed.',
                     fontsize=10, color='#43515c')
            fig.legend(handles=[Patch(facecolor=COLORS['control'], label='Control: state + yaw'),
                                Patch(facecolor=COLORS['active'], label='Active: state + yaw + body')],
                       loc='upper center', bbox_to_anchor=(.53, .914), ncol=2, frameon=False, fontsize=11)
            fig.subplots_adjust(left=.09, right=.975, top=.82, bottom=.20, hspace=.63, wspace=.30)
            fig.text(.07, .125, 'Yaw direction: 26/27 for every model. H32 retains the formal 45-point denominator, including 1 unresolved reference.',
                     fontsize=9, color='#43515c')
            fig.text(.07, .091, 'Direction, magnitude and numerical qualification rules are unchanged. Overall target: '
                     + report['comparison']['target_status'] + '.', fontsize=9, color='#43515c')
            fig.text(.07, .057, 'These are model diagnostics, not controller performance. No seed was selected.', fontsize=9, color='#43515c')
            for suffix in ('png', 'svg'):
                fig.savefig(args.output / f'body_jvp69_paired_summary.{suffix}', dpi=170, facecolor='white',
                            metadata={'Creator': 'tools69/plot69.py'} if suffix == 'svg' else None)
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
