"""Plot the first false 30 mm fill among the archived masked exits in round 106."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from tools94.common94 import ROOT, digest, read
from tools100.sensor100 import _grid
from tools104.motion104 import capture_query, heading, measured_motion, reference_motion, rotation
from tools105.data105 import MEASUREMENTS
from tools105.surface105 import _covers_query, _hull
from tools106.footprint106 import rear_returns


PUBLIC = ROOT / "reports/history106"
TOP, GROUND, INK, MUTED = "#A54E00", "#006D95", "#203343", "#4F6170"
MEASURED, REFERENCE = "#A02F65", "#35569A"


def archived_case():
    sealed = read(PUBLIC / "seal.json")
    for name in ("report.json", "reference101.npz", "measured.npz"):
        if digest(PUBLIC / name) != sealed["files"][name]:
            raise ValueError(f"changed round-106 artifact: {name}")
    report = read(PUBLIC / "report.json")
    entry = next(e for e in report["old_masked_exit_footprints"]
                 if e["history_mm"] is not None and abs(e["history_mm"]) > 10)
    i = entry["row_id"]
    if i != 798 or entry["oracle_mm"] is not None:
        raise ValueError("expected the archived left-wheel false fill followed by a reference abstention")
    with np.load(ROOT / "reports/jeplo101/train.npz", allow_pickle=False) as archive:
        saved = {k: archive[k][i-1:i+1] for k in ("depth_m", "valid", "capture_body_to_level", "world_query_xy",
                 "query_xy", "series_id", "capture_tick", "delivery_tick", "source_seed", "episode", "layout",
                 "visibility", "side", "row_id", "target_mm")}
    ticks = saved["capture_tick"]
    if (not np.array_equal(saved["row_id"], [i-1, i]) or saved["series_id"][0] != saved["series_id"][1]
            or not np.all(saved["layout"] == entry["layout"]) or not np.all(saved["visibility"] == "masked")
            or not np.all(saved["side"] == "left") or not np.array_equal(saved["target_mm"], [30., 0.])
            or not np.array_equal(ticks, [entry["previous_tick"], entry["tick"]])):
        raise ValueError("incorrect physical case, wheel or scan clocks")

    episode = [entry["source_seed"], entry["episode"]]
    with np.load(PUBLIC / "reference101.npz", allow_pickle=False) as poses:
        ep, = np.flatnonzero((poses["episode_keys"] == episode).all(axis=1))
        frames = np.searchsorted(poses["capture_ticks"], ticks)
        np.testing.assert_array_equal(poses["capture_ticks"][frames], ticks)
        capture = poses["reference_poses"][ep, frames, 0]

    plan = read(ROOT / "docs/experiments101/predictive_perception.json")["data"]
    fixture, = [item for item in plan["layouts"]["train"] if item["id"] + "_left" == entry["layout"]]
    geometry = read(ROOT / plan["sensor_base"])["geometry"]
    rays, _, _ = _grid(geometry)
    mounts = np.repeat(geometry["mounts_m"], len(rays), axis=0)
    rays = np.tile(rays, (2, 1))
    clouds = []
    for n, pose in enumerate(capture):
        body = mounts + saved["depth_m"][n, :, None].astype(float) * rays
        xyz = body[saved["valid"][n]] @ saved["capture_body_to_level"][n]
        xyz[:, :2] = xyz[:, :2] @ rotation(heading(pose)).T
        clouds.append(xyz + pose[:3])

    # Both estimates retain the measured 40 ms capture-to-delivery correction.
    # Only the older-to-newer capture transform differs in the two history reads.
    delivery = saved["delivery_tick"][-1]
    with np.load(ROOT / "reports/surface105/windows.npz", allow_pickle=False) as windows:
        row, = np.flatnonzero((windows["keys"] == [*episode, ticks[-1], delivery]).all(axis=1))
        correction = measured_motion(*(windows[k][row] for k in MEASUREMENTS))
    query = capture_query(saved["query_xy"][-1, :1], correction)
    with np.load(PUBLIC / "measured.npz", allow_pickle=False) as tape:
        np.testing.assert_array_equal(tape["episode_keys"][ep], episode)
        elapsed = (tape["sample_ticks"] >= ticks[0]) & (tape["sample_ticks"] <= ticks[-1])
        np.testing.assert_array_equal(tape["sample_ticks"][elapsed], np.arange(ticks[0], ticks[-1]+1))
        measured = measured_motion(*(tape[k][ep, elapsed] for k in MEASUREMENTS))
    estimates = np.vstack([capture_query(query, motion) for motion in (measured, reference_motion(capture))])
    estimates = estimates @ rotation(heading(capture[0])).T + capture[0, :2]
    actual = saved["world_query_xy"][-1, 0]
    np.testing.assert_allclose(1000*np.linalg.norm(estimates[0] - estimates[1]),
                               entry["measured_reference_query_offset_mm"], atol=1e-7, rtol=0)

    footprints = [rear_returns(xyz, actual, fixture, "left")[0] for xyz in clouds]
    for key, xy in zip(("previous", "current", "union"), [*footprints, np.vstack(footprints)], strict=True):
        if entry[key] != {"local_points": len(xy), "encloses_query": bool(_covers_query(xy))}:
            raise ValueError(f"footprint differs from archived {key}")
    hull = _hull(np.vstack(footprints)) + actual
    return entry, saved, fixture, geometry, clouds, 1000*hull, 1000*np.vstack([estimates, actual])


def draw(output):
    entry, saved, fixture, geometry, clouds, hull, locations = archived_case()
    high = 1000 * geometry["box_height_m"]
    center = 1000 * np.array([fixture["box_center_x_m"], fixture["box_abs_y_m"]])
    half = 1000 * np.array([fixture["box_half_x_m"], fixture["box_half_y_m"]])
    rear = center[0] + half[0]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP",
                         "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(11.4, 6.3), facecolor="white")
    fig.text(.065, .949, "第 798 包为什么把平地读成了 30 毫米台阶？", size=19, weight="bold")
    gap_ms = 20 * np.diff(saved["capture_tick"]).item()
    fig.text(.065, .893, f"D1 触阶前遮挡回放 · 前轴前方 {geometry['lookahead_m']:.2f} m 的左轮迹 · "
             f"两帧采集相隔 {gap_ms} ms", color=MUTED, size=11)
    axes = [fig.add_axes([.073, .219, .444, .525]), fig.add_axes([.600, .219, .335, .525])]
    zoom = (802, 864, 156, 208)
    for ax, view in zip(axes, ((678, 981, 102, 317), zoom), strict=True):
        ax.set_aspect("equal")
        ax.set(xlim=view[:2], ylim=view[2:], xlabel="世界 x（mm）", ylabel="世界 y（mm）")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#AFB9BF")
        ax.tick_params(length=3, labelsize=8.5)
        ax.add_patch(Rectangle(center-half, *(2*half), facecolor="#F8EFE2", edgecolor="#AA977C", lw=.9, zorder=0))
        ax.plot([rear, rear], [center[1]-half[1], center[1]+half[1]], color="#4D4337", lw=1.35, zorder=1)
        ax.fill(*hull.T, facecolor="#DFEFF0", alpha=.78, edgecolor="none", zorder=0)
        ax.plot(*np.vstack([hull, hull[0]]).T, color="#247780", lw=1.1, linestyle=(0, (4, 2)), zorder=2)
        # Show in-lane horizontal first returns. Marker shape identifies capture time, color identifies height.
        for frame, xyz in enumerate(clouds):
            p = xyz*1000
            lane = abs(p[:, 1] - center[1]) <= half[1]+.001
            for z, color in ((high, TOP), (0., GROUND)):
                xy = p[lane & (abs(p[:, 2]-z) <= .001), :2]
                if frame == 0:
                    ax.scatter(*xy.T, s=18, marker="o", facecolors="none", edgecolors=color, lw=.85, zorder=3)
                else:
                    ax.scatter(*xy.T, s=23, marker="+", c=color, lw=1., zorder=3)
    broad, detail = axes
    broad.set_title("两帧真实回波的俯视分布", loc="left", weight="bold", fontsize=12, pad=7)
    detail.set_title(f"台面后缘附近放大　│　后缘 x = {rear:.0f} mm", loc="left", weight="bold", fontsize=12, pad=7)
    broad.set(xticks=[700, 750, 800, 850, 900, 950], yticks=[125, 175, 225, 275])
    detail.set(xticks=[810, 825, 840, 855], yticks=[160, 175, 190, 205])
    broad.text(center[0], center[1]+half[1]-7, f"台面 {2*half[0]:.0f} × {2*half[1]:.0f} mm",
               color=TOP, ha="center", va="top", fontsize=9)
    broad.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1]-zoom[0], zoom[3]-zoom[2],
                              fill=False, edgecolor="#505D66", lw=1.25, ls=(0, (2, 2)), zorder=5))
    ticks = saved["capture_tick"]
    legend = [Line2D([], [], ls="", marker="o", mfc="none", mec=INK, label=f"上帧 · 第 {ticks[0]} 拍"),
              Line2D([], [], ls="", marker="+", color=INK, ms=8, label=f"本帧 · 第 {ticks[1]} 拍"),
              Patch(facecolor=TOP, label=f"台面回波（{high:.0f} mm）"), Patch(facecolor=GROUND, label="地面回波（0 mm）"),
              Patch(facecolor="#DFEFF0", edgecolor="#247780", linestyle="--", label="两帧低面回波的局部凸包")]
    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(.061,.836), frameon=False, ncol=5,
               columnspacing=1.6, handlelength=1.35, handletextpad=.38, fontsize=9)

    colors = (MEASURED, REFERENCE, INK)
    for ax in axes:
        for xy, color, marker, size in zip(locations, colors, ("s", "D", "*"), (59, 60, 133), strict=True):
            ax.scatter(*xy, s=size, marker=marker, c=color, edgecolors="white", linewidths=1.1, zorder=7)
    detail.annotate("", xy=locations[1], xytext=locations[0], zorder=6,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.15, shrinkA=3, shrinkB=4))
    labels = ("轮速 / IMU 对齐", "仅跨帧用真值", "真实轮前查询")
    for xy, color, label, text_pos in zip(locations, colors, labels, [(803, 203.5), (837, 199), (839, 161)], strict=True):
        detail.annotate(label, xy=xy, xytext=text_pos, ha="left", va="center", fontsize=8.8, weight="bold",
                        color=color, zorder=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.7, alpha=.94),
                        arrowprops=dict(arrowstyle="-", color=color, lw=.8, shrinkB=4))
    fig.text(.065,.111, f"第 {saved['delivery_tick'][-1]} 拍的查询在后缘外 {entry['query_past_rear_mm']:.2f} mm。"
             f"150 mm 半径内有 {entry['previous']['local_points']} + {entry['current']['local_points']} 个后缘地面回波，两帧合起来仍未围住查询点。",
             fontsize=10.25, weight="bold")
    fig.text(.065,.054, "姿态与台阶真值仅用于画图和事后对照。两种对齐都保留本包 40 ms 的测量补偿，"
             "只把跨帧位姿换成真值时，读数回到未知。", fontsize=9.2, color=MUTED)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Left wheel XY in world mm: measured {locations[0]}, cross-frame reference {locations[1]}, delivery {locations[2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/terrain_history106.png")
    draw(parser.parse_args().output)
