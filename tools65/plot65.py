"""Plot every frozen65 fresh point from its completed JSON report; no inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.report.read_bytes()
    report = json.loads(source)
    assert report['status'] == report['verification_after'] == 'COMPLETE'
    assert report['completed_model_points'] == 54 and report['completed_physical_points'] == 9
    args.output.mkdir(parents=True, exist_ok=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {'control': '#2166ac', 'active': '#d6601a'}
    styles = {64001: '-', 64002: '--', 64003: ':'}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'fixed-cost-prefix65'}):
        fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), sharex=True)
        for index, physical in enumerate(report['physical']):
            row = (110, 260, 410).index(physical['tick'])
            col = (64301, 64302, 64303).index(physical['seed'])
            axis = axes[row, col]
            prefixes = physical['prefixes']
            horizon = [point['horizon'] for point in prefixes]
            mean = [point['reference_gradient'] for point in prefixes]
            lower = [min(value['central64'] for value in point['gradients']) for point in prefixes]
            upper = [max(value['central64'] for value in point['gradients']) for point in prefixes]
            axis.fill_between(horizon, lower, upper, color='#bdbdbd', alpha=.45, linewidth=0)
            axis.plot(horizon, mean, color='#252525', linewidth=1.8, zorder=5)
            unknown = [point for point in prefixes if not point['resolved']]
            axis.scatter([point['horizon'] for point in unknown],
                         [point['reference_gradient'] for point in unknown],
                         color='#a50f15', marker='x', s=27, linewidths=1.1, zorder=6)
            for model in report['models']:
                point = model['points'][index]
                assert (point['seed'], point['tick']) == (physical['seed'], physical['tick'])
                assert point['all_prefix_total_selfchecks']
                axis.plot(horizon, [p['gradient'] for p in point['prefixes']],
                          color=colors[model['arm']], linestyle=styles[model['seed']], linewidth=1.1, alpha=.85)
            axis.axhline(0., color='#777777', linewidth=.6)
            axis.set_title(f"Reset {physical['seed']} / tick {physical['tick']}", loc='left', fontsize=10)
            axis.set_xlim(1, 32)
            axis.set_xticks([1, 8, 16, 24, 32])
            axis.grid(axis='y', alpha=.17)
            axis.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
            if row == 2:
                axis.set_xlabel('Rollout length (policy steps)')
            if col == 0:
                axis.set_ylabel('d(mean cost) / d(theta)')
        handles = [Line2D([], [], color='#252525', lw=1.8, label='Physical: mean of all four scales'),
                   Line2D([], [], color='#a50f15', marker='x', ls='', label='Physical reference unresolved'),
                   Line2D([], [], color=colors['control'], label='Control: labels 58 + 60'),
                   Line2D([], [], color=colors['active'], label='Active: labels 58 + 62')]
        handles += [Line2D([], [], color='#555555', ls=style, label=f'Model seed {seed}')
                    for seed, style in styles.items()]
        fig.suptitle('Frozen world models: cost gradient across rollout length', fontsize=16, x=.065, ha='left')
        fig.text(.065, .935, 'All 9 fresh states and all 6 models. Independent y scales; gray bands show scale ranges, not confidence intervals.',
                 fontsize=9, color='#444444')
        fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8.5,
                   bbox_to_anchor=(.5, .005))
        fig.tight_layout(rect=(0, .075, 1, .925), h_pad=2.0, w_pad=2.)
        for suffix in ('png', 'svg'):
            fig.savefig(args.output / f'cost_prefix65_all_fresh.{suffix}', dpi=170,
                        facecolor='white', metadata={'Creator': 'tools65/plot65.py'} if suffix == 'svg' else None)
        plt.close(fig)
    assert hashlib.sha256(args.report.read_bytes()).hexdigest() == hashlib.sha256(source).hexdigest()
    manifest = {path.name: {'bytes': path.stat().st_size,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(args.output.iterdir()) if path.is_file()}
    assert all(item['bytes'] < 5_000_000 for item in manifest.values())
    with (args.output / 'render_receipt.json').open('x') as stream:
        json.dump({'status': 'COMPLETE', 'source_report_sha256': hashlib.sha256(source).hexdigest(),
                   'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'models': 6, 'physical_points': 9, 'prefixes_per_point': 32,
                   'model_updates': 0, 'new_physics_steps': 0, 'model_inferences': 0,
                   'files': manifest}, stream, indent=2)
        stream.write('\n')
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
