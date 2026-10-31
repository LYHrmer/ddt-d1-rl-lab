"""Read saved collection52 arrays; no actor, physics or fitting is imported."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    source = args.input
    cases = json.loads((source / 'collection.json').read_text())
    rows, hashes = [], {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    colors = {'train': '#347a9c', 'development': '#bc5c27', 'probe': '#764ca5'}
    indices = {key: 0 for key in colors}
    for order, case in enumerate(cases):
        role, seed = case['role'], case['seed']
        index = indices[role]
        indices[role] += 1
        amplitude = (.1, .2, .3)[index % 3]
        path = source / f'collection_{role}_{seed}.npz'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            n = len(data['s'])
            assert n == case['steps'] == len(data['target'])
            assert np.isfinite(data['s']).all() and np.isfinite(data['target']).all()
            assert np.array_equal(data['tick'], np.arange(n))
            assert data['done'][-1] and not data['done'][:-1].any()
            assert np.array_equal(data['target'][:-1], data['s'][1:])
            q = data['target'][:, 3:7]
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
            tilt = np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))
            row = dict(case, amplitude_rad_s=amplitude, vx_command_m_s=0. if index % 4 < 2 else .1,
                       action_noise_sigma=.02 if role == 'train' else 0.,
                       duration_s=n * .02, peak_tilt_rad=float(tilt.max()),
                       terminal_tilt_rad=float(tilt[-1]), terminal_height_m=float(data['target'][-1, 2]),
                       terminal_tilt_threshold_exceeded=bool(tilt[-1] > .6))
            rows.append(row)
            if case['terminated']:
                axes[1].plot(np.arange(1, n + 1) * .02, tilt, lw=1.3,
                             label=f'{role} {seed}', color=colors[role], alpha=.8)
        axes[0].barh(order, n * .02, color='#b64a41' if case['terminated'] else '#509178', height=.7)
        axes[0].text(n * .02 + .08, order, f'{amplitude:.1f}', va='center', fontsize=7)
    axes[0].set_yticks(range(len(rows)), [f'{r["role"][:3]} {r["seed"]}' for r in rows], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].axvline(8, ls='--', color='#555555', lw=.8)
    axes[0].set(xlim=(0, 9), xlabel='Recorded duration (s); labels = |yaw command| (rad/s)',
                title='Collection gate: 13 complete / 5 terminated')
    axes[1].axhline(.6, ls='--', color='#b64a41', label='Original tilt limit 0.6 rad')
    axes[1].set(xlabel='Time (s)', ylabel='Body tilt (rad)', title='Saved states of all five terminated trials')
    axes[1].legend(fontsize=7, loc='upper left')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(args.output / 'collection52.svg', metadata={'Date': None})
    plt.close(fig)
    result = {'schema': 'model-gradient52-collection-readback-v1', 'rows': rows,
              'source_npz_sha256': hashes, 'complete': sum(r['complete_8s_trial'] for r in rows),
              'terminated': sum(r['terminated'] for r in rows), 'transitions': sum(r['steps'] for r in rows),
              'amplitude_groups': {str(a): {'trials': sum(r['amplitude_rad_s'] == a for r in rows),
                  'terminated': sum(r['terminated'] and r['amplitude_rad_s'] == a for r in rows)} for a in (.1, .2, .3)},
              'new_physics_steps': 0, 'model_fits': 0,
              'interpretation': 'Post-hoc association only; command amplitude, translation, timing and noise were not isolated. No model-gradient evidence.'}
    with (args.output / 'readback.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('complete', 'terminated', 'transitions', 'amplitude_groups')}))


if __name__ == '__main__':
    main()
