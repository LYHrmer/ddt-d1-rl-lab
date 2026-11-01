"""Read back53 work, gradients and post-hoc diagnostics without running a model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools53.run_pilot53 import require, sha, verify, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = ROOT / 'evidence/model_gradient53'
    work = ROOT.parent / 'world_model53_work_20261101'
    plan_path = ROOT / 'docs/experiments53/model_gradient53_preregistered.json'
    plan = json.loads(plan_path.read_text())
    verify(plan)
    formal = json.loads((evidence / 'execution_01/report.json').read_text())
    admission = json.loads((evidence / 'admission_01/report.json').read_text())
    diagnostic = json.loads((evidence / 'gradient_diagnostic_01/report.json').read_text())
    proposal = json.loads((evidence / 'execution_01/proposal_before_physics.json').read_text())
    require(formal['status'] == admission['status'] == diagnostic['status'] == 'COMPLETE',
            'this readback expects all three stages complete')
    require(formal['plan_sha256'] == admission['plan_sha256'] == sha(plan_path), 'plan binding differs')
    require(formal['sealed_proposal_sha256'] == sha(evidence / 'execution_01/proposal_before_physics.json'),
            'sealed proposal differs')
    previous = 0
    with (evidence / 'execution_01/all_physics_steps.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row['counts']['policy_transitions'] == previous + 1, 'work log is not monotonic')
            previous += 1
            require(row['counts']['native_steps'] == previous * 10
                    and row['counts']['pd_updates'] == previous * 5, 'work ratios differ')
            require(not any(row[k] for k in ('fallen', 'illegal_body_contact', 'terminated', 'truncated')),
                    'physical failure in complete pilot')
    require(previous == formal['counts']['policy_transitions'] == 12528, 'physical log total differs')
    model_rows = []
    for seed, gate in zip(formal['seeds'], formal['direction_gate']['models'], strict=True):
        raw = valid = 0
        for predicted, actual in zip(seed['test_gradients'], formal['physical_probes'], strict=True):
            same = predicted['gradient'] * actual['gradient'] > 0
            raw += same
            valid += same and predicted['autograd_agrees'] and predicted['direction_identifiable'] and actual['identifiable']
        require(valid == gate['matches'], 'independent gradient recount differs')
        with (evidence / f'execution_01/model_{seed["seed"]}/updates.jsonl').open() as stream:
            updates = [json.loads(line)['update'] for line in stream]
        require(updates == list(range(1, 1001)), 'model update journal incomplete')
        model_rows.append({'seed': seed['seed'], 'updates': len(updates),
            'raw_sign_matches': int(raw), 'valid_direction_matches': int(valid),
            'probe_autograd_passes': sum(g['autograd_agrees'] for g in seed['test_gradients']),
            'development_autograd_passes': sum(g['autograd_agrees'] for g in seed['development_gradients']),
            'individual_gate_passed': gate['passed'], 'fit_cpu_seconds': seed['fit']['cpu_seconds'],
            'probe_yaw_rmse_rad_s': {h: seed['diagnostics']['probe'][h]['state_rmse45'][28] for h in ('1', '8')}})
    manifest = {}
    for name in ('admission_01', 'execution_01', 'gradient_diagnostic_01'):
        for path in sorted((evidence / name).rglob('*')):
            if path.is_file():
                manifest[str(path.relative_to(evidence))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    receipt = {'schema': 'd1-model-gradient53-readback-v1', 'status': 'COMPLETE',
               'experiment_status': formal['status'], 'candidate_accepted': formal['candidate_accepted'],
               'plan_sha256': sha(plan_path), 'sealed_identity_count': len(plan['identities']),
               'all_frozen_identities_unchanged': True, 'models': model_rows,
               'total_physics_counts': {key: formal['counts'][key] + admission['counts'][key]
                                       for key in ('policy_transitions', 'native_steps', 'pd_updates')},
               'formal': formal, 'admission': admission, 'sealed_proposal': proposal,
               'post_hoc_diagnostic': diagnostic, 'raw_manifest': manifest,
               'raw_bytes': sum(x['bytes'] for x in manifest.values()),
               'supervisors': {name: json.loads((work / name / 'report.json').read_text())
                              for name in ('checks01', 'checks02', 'admission01', 'execution01', 'gradientdiag01')},
               'readback_new_physics_steps': 0, 'readback_new_model_calls': 0}
    write_json(args.output / 'summary.json', receipt)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cells = np.zeros((4, 9))
    values = [[p['gradient'] for p in formal['physical_probes']]]
    for i, model in enumerate(formal['seeds'], start=1):
        values.append([p['gradient'] for p in model['test_gradients']])
        for j, (pred, actual) in enumerate(zip(model['test_gradients'], formal['physical_probes'], strict=True)):
            cells[i, j] = (2 if not pred['autograd_agrees'] or not pred['direction_identifiable']
                           else 0 if pred['gradient'] * actual['gradient'] > 0 else 1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True,
                             gridspec_kw={'height_ratios': [1.3, 1]})
    axes[0].imshow(cells, vmin=0, vmax=2, aspect='auto',
                   cmap=ListedColormap(['#356b8b', '#efb568', '#d7dce0']))
    for i in range(4):
        for j in range(9):
            axes[0].text(j, i, f'{values[i][j]:.3g}', ha='center', va='center', fontsize=9,
                         color='white' if cells[i, j] == 0 else '#1f2933')
    axes[0].set_xticks(range(9), [f'{p["seed"]}\n{p["tick"] * .02:.1f}s' for p in formal['physical_probes']], fontsize=8)
    axes[0].set_yticks(range(4), ['MuJoCo FD', 'Model 53001', 'Model 53002', 'Model 53003'])
    axes[0].set_title('Local cost gradients at all nine probe states; no controller update accepted')
    axes[0].legend(handles=[Patch(color='#356b8b', label='Reference / same direction'),
                           Patch(color='#efb568', label='Opposite direction'),
                           Patch(color='#d7dce0', label='Model numerical check failed')],
                   loc='upper center', bbox_to_anchor=(.5, -.19), ncol=3, fontsize=8, frameon=False)
    for model in diagnostic['models']:
        counts = [sum(next(c for c in p['checks'] if c['epsilon'] == eps)['original_tolerance_formula_passes']
                      for p in model['points']) for eps in diagnostic['epsilon_grid']]
        axes[1].plot(diagnostic['epsilon_grid'], counts, marker='o', label=f'Model {model["seed"]}', alpha=.8)
    axes[1].axvline(.001, color='#555555', ls='--', lw=1, label='Registered epsilon')
    axes[1].set_xscale('log')
    axes[1].set_xticks(diagnostic['epsilon_grid'], [str(e) for e in diagnostic['epsilon_grid']])
    axes[1].set_yticks([16, 17, 18])
    axes[1].set(xlabel='Finite-difference epsilon (post-hoc diagnostic only)',
                ylabel='Checks passing / 18 points', ylim=(15.6, 18.4),
                title='Step-size sensitivity; the original 0.001 decision is unchanged')
    axes[1].legend(ncol=4, fontsize=8, loc='lower right')
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].grid(alpha=.15)
    fig.savefig(args.output / 'gradients53.svg', metadata={'Date': None})
    fig.savefig(args.output / 'gradients53.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'models': model_rows, 'total_physics_counts': receipt['total_physics_counts'],
                      'raw_bytes': receipt['raw_bytes']}, indent=2))


if __name__ == '__main__':
    main()
