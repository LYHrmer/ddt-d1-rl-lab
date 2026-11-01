"""Plot the complete fixed55B/55C grids from saved JSON; zero simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.report.read_text())
    runs = [data['runs'][name] for name in ('stability_01', 'local_01')]
    if any(run['status'] != 'COMPLETE' for run in runs):
        raise ValueError('Both complete, unmodified diagnostic grids are required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout='constrained')
    for ax, seed in zip(axes, (55301, 55303), strict=True):
        values = {}
        for run in runs:
            point, = [row for row in run['points'] if row['seed'] == seed and row['tick'] == 110]
            for row in point['gradients']:
                previous = values.get(row['epsilon'])
                if previous is not None and any(previous[key] != row[key]
                    for key in ('central64', 'forward64', 'backward64')):
                    raise ValueError('Overlapping independently replayed values differ')
                values[row['epsilon']] = row
        ordered = [values[key] for key in sorted(values)]
        ax.axvspan(.00003125, .00025, color='#d5e8dd', alpha=.7,
                   label='Fixed 55C local grid')
        ax.axhline(0, color='#77838f', linewidth=.8)
        for key, label, style, color in (
            ('central64', 'Central', 'o-', '#173e68'),
            ('forward64', 'Forward', '.--', '#b34830'),
            ('backward64', 'Backward', '.--', '#6e5790'),
        ):
            ax.plot([row['epsilon'] for row in ordered], [row[key] for row in ordered],
                    style, label=label, color=color, linewidth=1.3, markersize=4)
        ax.set_xscale('log')
        ax.set_xlabel('Perturbation size, epsilon')
        ax.set_ylabel('Finite difference of mean 32-step cost')
        ax.set_title(f'Seed {seed}, tick 110', loc='left', weight='bold')
        ax.grid(axis='y', color='#dce2e8', linewidth=.6)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useMathText=True)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('Physical gradient references depend on perturbation scale',
                 x=.01, ha='left', fontsize=14, weight='bold')
    fig.supxlabel('All preregistered scales retained. Independent diagnostics; original 55 verdict unchanged.',
                  fontsize=9, color='#485566')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive open prevents replacing any previously rendered evidence.
    with args.output.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=160, facecolor='white')
    plt.close(fig)
    print(json.dumps({'output': str(args.output), 'bytes': args.output.stat().st_size,
                      'new_physics_steps': 0, 'new_model_calls': 0}))


if __name__ == '__main__':
    main()
