"""Plot the paired uphill wheel-demand probe from its completed interval traces."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools87.readback87 import digest, load


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/experiments88/uphill_wheel_probe.json"
ARMS = ("original", "wheel_plus")


def trailing_mean(values, width=5):
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    if len(values) >= width:
        result[width-1:] = np.convolve(values, np.full(width, 1/width), mode="valid")
    return result


def series(root):
    plan, receipt = load(PLAN), load(root / "report.json")
    course_path = ROOT / plan["parent_protocol"]
    if (receipt["status"] != "COMPLETE" or receipt["protocol_sha256"] != digest(PLAN)
            or digest(course_path) != plan["parent_protocol_sha256"]):
        raise ValueError("a complete probe with the registered protocol is required")
    course = load(course_path)["course"]
    times = np.arange(1, course["ticks"]+1)/course["policy_hz"]
    cases = {(c["arm"], c["seed"]): c for c in receipt["cases"]}
    expected = {(arm, seed) for arm in ARMS for seed in plan["reset_seeds"]}
    if len(receipt["cases"]) != len(expected) or set(cases) != expected:
        raise ValueError("expected the registered three seed-matched pairs")
    arrays = {}
    for arm in ARMS:
        group = np.full((2, len(plan["reset_seeds"]), len(times)), np.nan)
        for j, seed in enumerate(plan["reset_seeds"]):
            case = cases[arm, seed]
            folder = root / f"{arm}_{seed}"
            path = folder / "trace.npz"
            if case != load(folder / "metrics.json") or digest(path) != case["trace_sha256"]:
                raise ValueError(f"{folder}: source identity mismatch")
            with np.load(path, allow_pickle=False) as z:
                t, v, x = (z[k][1:] for k in ("time_s", "body_vx_m_s", "front_contact_x_m"))
            n = case["ticks"]
            if (not 0 < n <= len(times) or v.shape != (n,) or x.shape != (n,) or t.shape != (n,)
                    or not np.allclose(t, times[:n], atol=1e-8, rtol=0)
                    or not np.isfinite(v).all() or np.isinf(x).any()):
                raise ValueError(f"{folder}: invalid time or signal trace")
            group[0, j, :n] = trailing_mean(v)
            group[1, j, :n] = x - course["toe_x_m"]
        arrays[arm] = group
    return times, arrays, plan, course


def plot(root, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.lines import Line2D

    t, groups, plan, course = series(root)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback") if f in fonts), None)
    if font:
        plt.rcParams["font.family"] = [font, "DejaVu Sans"]
    title, speed, position, clock, demand, mark, toe, issue, scoring, original, plus, note = (
        ("D1 在 4° 上坡时的轮速指令对照", "机身前向速度 (m/s)", "最前承重前轮接地点\n距坡脚 (m)", "仿真时间 (s)",
         "速度指令", "距离门槛", "坡脚", "发出轮速增量", "评分时间窗", "原始全身策略", "四轮目标",
         "线为 3 个配对种子的均值，色带为最小值至最大值。速度仅为显示取最近 5 拍均值，接触距离用原始值。")
        if font else
        ("D1 wheel-speed demand on the 4° uphill", "Body forward speed (m/s)", "Foremost loaded front\ncontact vs. ramp toe (m)",
         "Simulation time (s)", "Command", "Course mark", "Toe", "Wheel offset issued", "Scoring window",
         "Original whole-body policy", "All wheel targets",
         "Lines show 3 paired-seed means, bands the full range. Speed uses a trailing 5-tick mean for display only. Contact is raw.")
    )
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8.1, 5.8))
    fig.subplots_adjust(left=.14, right=.97, bottom=.16, top=.79, hspace=.37)
    fig.suptitle(title, x=.14, y=.98, ha="left", size=16, weight="bold")
    colors = ("#22598a", "#b14d1e")
    styles = ("solid", (0, (4, 2)))
    labels = [original, f"{plus} +{plan['arm_delta_rad_s']['wheel_plus']:.2f} rad/s"]
    tracks = [Line2D([], [], color=color, ls=style, lw=2, label=label)
              for color, style, label in zip(colors, styles, labels)]
    fig.legend(handles=tracks, loc="upper left", bbox_to_anchor=(.13, .92), ncol=2, frameon=False)
    start, end = np.asarray(plan["primary"]["interval_end_ticks_exclusive_inclusive"]) / course["policy_hz"]
    offset = plan["issue_start_s"]
    for i, ax in enumerate(axs):
        ax.axvspan(start, end, color="#eef2ef", zorder=-3)
        ax.grid(axis="y", color="#d8deda", lw=.65, zorder=-2)
        ax.axvline(offset, color="#687671", ls=":", lw=1.2, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#88928f")
        for arm, color, style in zip(ARMS, colors, styles):
            rows = groups[arm][i]
            # Propagate missing ticks: the seed cohort stays fixed for every point shown.
            ax.fill_between(t, rows.min(axis=0), rows.max(axis=0), color=color, alpha=.17, lw=0)
            ax.plot(t, rows.mean(axis=0), color=color, ls=style, lw=1.8)
        ax.set_axisbelow(True)
    axs[0].text(offset-.05, 1.02, f"{issue}  {offset:.2f} s", transform=axs[0].get_xaxis_transform(),
                ha="right", va="bottom", color="#41514b", fontsize=9)
    axs[0].text((start+end)/2, 1.02, f"{scoring}  {start:g}–{end:g} s", transform=axs[0].get_xaxis_transform(),
                ha="center", va="bottom", color="#41514b", fontsize=9)
    target = course["command_forward_m_s"]
    initial = course["stand_ticks"] / course["policy_hz"]
    axs[0].step([0, initial, t[-1]], [0, target, target], where="post", color="#5c645f", ls=":", lw=1.1, zorder=1)
    axs[0].annotate(f"{demand}  {target:.2f}", (.11, target), xytext=(0, 3), textcoords="offset points", fontsize=8)
    threshold = plan["independent_course_mark_m_past_toe"]
    for height, label in ((0., toe), (threshold, f"{mark}  +{threshold:.2f} m")):
        axs[1].axhline(height, color="#5c645f", ls=":", lw=1, zorder=0)
        axs[1].annotate(label, (.10, height), xytext=(0, 3), textcoords="offset points", fontsize=8)
    axs[0].set_ylabel(speed)
    axs[1].set_ylabel(position)
    axs[1].set_xlabel(clock)
    axs[1].set_xlim(0, t[-1])
    fig.text(.14, .033, note, fontsize=8.3, color="#344039")
    try:
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=160, facecolor="white",
                        metadata={"Description": f"Input report SHA256 {digest(root / 'report.json')}"})
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.root, args.output)
