"""Plot measured horizontal speed and two fixed-action replays from the public archive."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import digest, read, require
from tools111.results import install
from tools114.common import DEV, PUBLIC, ROOT, baseline
from tools114.replay import BUNDLE, RESET, TAIL, TICKS, STOP_LIMIT


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = PUBLIC / 'replay'
    report = read(folder / 'report.json')['model_replay']
    require(digest(folder / BUNDLE) == report['bundle_sha256'], 'wrong stopping suffixes')
    plt.rcParams.update({'font.family': ['Noto Sans CJK JP', 'DejaVu Sans'], 'font.size': 11})
    fig, axes = plt.subplots(3, 1, figsize=(9.1, 6.9), sharex=True, sharey=True)
    time = np.arange(RESET, TICKS + 1) / 50.
    maximum = STOP_LIMIT
    with np.load(folder / BUNDLE, allow_pickle=False) as data:
        for ax, seed in zip(axes, DEV):
            reference, _ = baseline(seed)
            with np.load(reference, allow_pickle=False) as trace:
                measured = np.linalg.norm(trace['vxy'][RESET:], axis=1)
            original = np.linalg.norm(data[f'{seed}_sealed_sealed'][:, 3:5], axis=1)
            fixed = np.linalg.norm(data[f'{seed}_native_native'][:, 3:5], axis=1)
            maximum = max(maximum, measured.max(), original.max(), fixed.max())
            ax.axvspan(TAIL / 50., TICKS / 50., color='#dce8ec', alpha=.65, linewidth=0)
            ax.plot(time, measured, color='#142f3c', linewidth=2.25, label='原生仿真实测')
            ax.plot(time, original, color='#787092', linewidth=1.5, linestyle='--', label='原封存预测')
            ax.plot(time, fixed, color='#b14d17', linewidth=1.85, label='实测起点与动作的回放预测')
            ax.axhline(STOP_LIMIT, color='#3a776d', linewidth=1.2, linestyle=':', label='停车门槛 0.050 m/s')
            ax.set_title(f'起点 {seed}', loc='left', fontsize=10)
            ax.grid(axis='y', color='#d2dadb', linewidth=.55)
            ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set(xlim=(time[0], time[-1]), ylim=(0, maximum * 1.08), xlabel='仿真时间（秒）')
    axes[-1].set_xticks(np.arange(10.5, 12.01, .25))
    fig.supylabel('机身水平速率（m/s）', x=.035, fontsize=12)
    fig.suptitle('D1 驾驶请求归零后的车速', x=.11, y=.98, ha='left', fontsize=18)
    fig.text(.11, .925, '浅色区域为停车验收窗口（11.50～12.00 秒）。三条曲线均取自第 113 轮控制器。', fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.10, .905),
               ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.11, right=.976, bottom=.085, top=.785, hspace=.44)
    dest = ROOT / 'docs/assets/parking114_replay.png'
    with TemporaryDirectory() as temp:
        file = Path(temp) / dest.name
        fig.savefig(file, format='png', dpi=140, facecolor='white')
        install(file, dest, digest(file))
    plt.close(fig)
    return dest


if __name__ == '__main__':
    print(render())
