"""Plot the measured world-x trajectories from the six qualified stand attempts."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np


def _curve(folder, case, ticks):
    raw = (folder / case["trace"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != case["trace_sha256"]:
        raise ValueError(f"changed stand trace: {case['trace']}")
    if not case["clean"] or case["status"] != "COMPLETE" or case["ticks"] != ticks:
        raise ValueError("full clean stand trajectories required")
    with np.load(BytesIO(raw), allow_pickle=False) as data:
        axis = int(data["free_qpos_index"])
        origin = float(data["initial_qpos"][axis])
        x = data["qpos"][:, axis]
        time = data["time_s"]
    if (x.shape != (ticks,) or not np.isfinite(x).all() or not np.isfinite(origin)
            or time.shape != (ticks,) or not np.allclose(time, .02 * np.arange(1, ticks + 1), rtol=0., atol=1e-9)):
        raise ValueError("incomplete physical time axis or world-x trajectory")
    return origin, np.r_[0., time], np.r_[0., (x - origin) * 100.]


def plot_stand(folder, cases, output_png):
    """Write an exclusive PNG using the original stage report's six case records."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    folder = Path(folder)
    output_png = Path(output_png)
    protocol = Path(__file__).resolve().parents[1] / "docs/experiments90/stand_wbc.json"
    plan = json.loads(protocol.read_bytes())["stand"]
    pairs = list(zip(plan["parent_seeds"], plan["reset_seeds"], strict=True))
    by_key = {(c["parent_seed"], c["reset_seed"], c["arm"]): c for c in cases}
    expected = {(parent, seed, arm) for parent, seed in pairs for arm in ("original", "common")}
    if len(cases) != len(expected) or set(by_key) != expected or len(pairs) != 3:
        raise ValueError("all three registered pairs must appear exactly once")
    curves = {k: _curve(folder, c, plan["ticks_per_episode"]) for k, c in by_key.items()}

    font_dir = Path("/usr/share/fonts/opentype/noto")
    regular = FontProperties(fname=font_dir / "NotoSansCJK-Regular.ttc")
    bold = FontProperties(fname=font_dir / "NotoSansCJK-Bold.ttc")
    figure = Figure(figsize=(13, 5.6), dpi=145, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3, sharey=True)
    figure.subplots_adjust(left=.083, right=.981, top=.665, bottom=.19, wspace=.12)
    ink, muted = "#182B3F", "#485C6C"
    styles = {"original": {"color": "#274B70", "linestyle": (0, (3.2, 2.2)), "linewidth": 2.1},
              "common": {"color": "#B44729", "linestyle": "-", "linewidth": 2.5}}

    threshold_mm = plan["min_rollback_reduction_m"] * 1000.
    figure.text(.083, .957, f"4° 上坡驻车   每组首秒少退至少 {threshold_mm:g} mm 才达标",
                fontproperties=bold, size=20, color=ink, va="top")
    figure.text(.083, .854, "0 至 1 s 的指令速度为零，虚线处切换至前进 0.20 m/s。负位移表示机身向坡下后退。",
                fontproperties=regular, size=10.5, color=muted, va="baseline")
    handles = [Line2D([], [], **styles["original"]), Line2D([], [], **styles["common"])]
    figure.legend(handles, ["原感知 PPO 策略", "同一策略，驻车时修正四轮共同目标"],
                  loc="upper right", bbox_to_anchor=(.984, .946), ncol=1, frameon=False,
                  handlelength=3, labelspacing=.27, prop=regular.copy(), labelcolor=ink)

    low = 5 * math.floor(min(float(x.min()) for _, _, x in curves.values()) / 5)
    high = 5 * math.ceil(max(float(x.max()) for _, _, x in curves.values()) / 5)
    stop = plan["stand_ticks"]
    for panel, (parent, seed) in zip(axes, pairs, strict=True):
        base, t, original = curves[(parent, seed, "original")]
        other_base, other_t, common = curves[(parent, seed, "common")]
        if base != other_base or not np.array_equal(t, other_t):
            raise ValueError("paired reset position or physical times differ")
        switch = float(t[stop])
        reduction_mm = 10. * (max(0., -original[stop]) - max(0., -common[stop]))
        label = f"首秒少退 {reduction_mm:.2f} mm" if reduction_mm >= 0 else f"首秒多退 {-reduction_mm:.2f} mm"
        panel.set_title(f"策略种子 {parent}  /  重置 {seed}", fontproperties=regular, fontsize=10.5,
                        loc="left", color=muted, pad=34)
        panel.text(0., 1.055, label, transform=panel.transAxes, fontproperties=bold, fontsize=14, color=ink)

        panel.axvspan(0., switch, facecolor="#F0F4F7", zorder=0)
        panel.grid(axis="y", linewidth=.65, color="#D5DDE3")
        panel.axvline(switch, color="#687A87", linestyle=(0, (2, 2.5)), linewidth=1.05)
        panel.plot(t, common, **styles["common"], zorder=3)
        panel.plot(t, original, **styles["original"], zorder=4)
        for y, arm, marker in ((common[stop], "common", "s"), (original[stop], "original", "o")):
            panel.plot(switch, y, marker=marker, markersize=4.7, color=styles[arm]["color"],
                       markeredgecolor="white", markeredgewidth=.65, zorder=5)
        panel.set_xlim(float(t[0]), float(t[-1]))
        panel.set_ylim(low - 1., high + 1.)
        panel.set_xticks([0., .5, 1., 1.5, 2.], ["0", "0.5", "1", "1.5", "2"])
        panel.set_yticks(np.arange(low, high + 1, 5))
        panel.tick_params(axis="both", colors=muted, labelsize=9, length=3)
        for edge in ("top", "right"):
            panel.spines[edge].set_visible(False)
        for edge in ("left", "bottom"):
            panel.spines[edge].set_color("#9AABB7")
    axes[0].set_ylabel("机身相对起点的世界 x 位移 (cm)", fontproperties=regular, fontsize=11, color=ink, labelpad=10)
    figure.text(.532, .088, "物理时间 (s)", ha="center", va="center", fontproperties=regular, size=11, color=ink)
    with output_png.open("xb") as stream:
        figure.savefig(stream, format="png", facecolor="white")
    return output_png
