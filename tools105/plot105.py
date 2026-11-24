"""Illustrate the two archived round-104 edge cases with the frozen round-105 readers."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from tools94.common94 import ROOT, read
from tools100.score100 import RADIUS_M, TIE_ATOL_M, estimate_delta_mm
from tools101.common101 import protocol
from tools104.motion104 import PUBLIC as MOTION104, heading, rotation
from tools105.data105 import inputs
from tools105.surface105 import surface_heights_m


BLUE, ORANGE, GREEN, INK, MUTED = "#317b9c", "#b64e26", "#127456", "#263743", "#526470"
HALF_WIDTH_MM = 40
CASES = ((218, "进入台面", "左轮前方"), (127, "离开台面", "右轮前方"))


def _panel(ax, row_id, title, wheel_label, side, points, queries, pose, edge_x, top_mm):
    points = points[np.isfinite(points).all(axis=1)]
    baseline = estimate_delta_mm(points, queries)
    heights = surface_heights_m(points, queries)
    distance = np.linalg.norm(points[:, :2] - queries[side], axis=1)
    nearest = (distance <= RADIUS_M) & (distance - distance.min() <= TIE_ATOL_M)
    if baseline is None or nearest.sum() != 1:
        raise ValueError("expected one selected echo in each archived example")

    # The readers finish in the capture frame. The archived pose only places the
    # measured points and the fixture on common world axes for the illustration.
    orientation = rotation(heading(pose)).T
    xy_mm = 1000 * ((points[:, :2] - queries[side]) @ orientation)
    z_mm = 1000 * (points[:, 2] + pose[2])
    old_height = z_mm[nearest].item()
    new_height = None if heights[side] is None else 1000 * (heights[side] + pose[2])
    if not np.isclose(old_height, (1 if side == 0 else -1) * baseline, atol=0.001, rtol=0):
        raise ValueError("the opposite wheel must be on the reference ground in these examples")

    query_world = queries[side] @ orientation + pose[:2]
    edge_mm = 1000 * (edge_x - query_world[0])
    left_mm, right_mm = (0., top_mm) if row_id == 218 else (top_mm, 0.)
    span = [-50., edge_mm, edge_mm, 50.]
    profile = [left_mm, left_mm, right_mm, right_mm]
    ax.fill_between(span, -5., profile, color="#e9eff1", zorder=0)
    ax.plot(span, profile, color=MUTED, lw=1.15, zorder=1)
    ax.axvline(0, color="#76868b", linestyle=(0, (3, 3)), linewidth=1., zorder=2)
    ax.text(0., 37., "轮前查询点", ha="center", fontsize=8.5, color=MUTED)

    visible = (np.abs(xy_mm[:, 1]) <= HALF_WIDTH_MM) & (np.abs(xy_mm[:, 0]) <= 50.)
    if not visible[nearest].item():
        raise ValueError("the selected echo lies outside the displayed cross section")
    ax.scatter(xy_mm[visible, 0], z_mm[visible], s=21, color=BLUE, zorder=3)
    ax.scatter(xy_mm[nearest, 0], z_mm[nearest], s=121, facecolor="none", edgecolor=ORANGE,
               linewidth=1.9, zorder=4)
    if new_height is not None:
        ax.scatter(0., new_height, s=64, marker="D", facecolor=GREEN, edgecolor="white", linewidth=.8, zorder=5)

    ax.text(0., 1.33, f"{title}　·　{wheel_label}（包 {row_id}）", transform=ax.transAxes,
            fontsize=13.5, color=INK, weight="bold")
    ax.text(0., 1.13, f"旧最近点  {old_height:.2f} mm", transform=ax.transAxes, fontsize=11.5, color=ORANGE)
    label = "未知" if new_height is None else f"{new_height:.2f} mm"
    ax.text(.53, 1.13, f"新支撑面  {label}", transform=ax.transAxes, fontsize=11.5, color=GREEN)
    ax.set(xlim=(-50., 50.), ylim=(-4., 42.), xticks=[-40, -20, 0, 20, 40], yticks=[0, 10, 20, 30])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b6c3c7")
    ax.tick_params(labelsize=9, length=3, color="#94a5aa", labelcolor=MUTED)


def draw(path):
    rows, clouds, queries = inputs("design")
    data = protocol()["data"]
    fixture, = data["layouts"]["holdout"]
    top_mm = 1000 * read(ROOT / data["sensor_base"])["geometry"]["box_height_m"]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP",
                         "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(10., 4.95), sharey=True)
    fig.subplots_adjust(left=.079, right=.966, bottom=.295, top=.615, wspace=.20)
    fig.text(.079, .91, "轮前点云怎样读出台阶高度？", color=INK, fontsize=20, weight="bold")
    fig.text(.079, .841, "两段 D1 触阶前的仿真存档回放。每帧的两种读法使用同一包遮挡回波和同一轮前位置。",
             fontsize=10.5, color=MUTED)
    with np.load(MOTION104 / "windows.npz", allow_pickle=False) as windows:
        lookup = {tuple(key): i for i, key in enumerate(windows["keys"])}
        poses = windows["reference_poses"]
        for ax, (i, title, wheel_label) in zip(axes, CASES, strict=True):
            side = 0 if i == 218 else 1
            if (rows["row_id"][i] != i or rows["visibility"][i] != "masked"
                    or rows["side"][i] != ("left", "right")[side]):
                raise ValueError("unexpected design packet or wheel side")
            key = tuple(rows[k][i] for k in ("episode", "capture_tick", "delivery_tick"))
            capture, delivery = poses[lookup[key]]
            if delivery[0] <= capture[0]:
                raise ValueError("the archived forward motion must point toward positive world X")
            edge = fixture["box_center_x_m"] + (-1 if i == 218 else 1) * fixture["box_half_x_m"]
            _panel(ax, i, title, wheel_label, side, clouds[i], queries[i], capture, edge, top_mm)
    axes[0].set_ylabel("离地高度（mm）", fontsize=10, color=INK)
    fig.text(.52, .206, "距轮前查询点的距离（mm）　→ 行进方向", ha="center", color=MUTED, fontsize=10)
    handles = [Line2D([], [], color=MUTED, label="仿真地形（核对用）"),
               Line2D([], [], linestyle="none", marker="o", color=BLUE, label="收到的回波"),
               Line2D([], [], linestyle="none", marker="o", markerfacecolor="none", markeredgewidth=1.8,
                      markeredgecolor=ORANGE, markersize=9, label="旧法选中的回波"),
               Line2D([], [], linestyle="none", marker="D", color=GREEN, markersize=7, label="新法给出的高度")]
    fig.legend(handles=handles, ncol=4, loc="center", bbox_to_anchor=(.52, .121), frameon=False, fontsize=9.5)
    fig.text(.079, .040, f"截面取轮迹左右各 {HALF_WIDTH_MM} mm 的回波。左图的最近点打在竖面低处，右图的查询点落在顶面和地面回波的间隙。",
             fontsize=10, color=MUTED)
    with path.open("xb") as f:
        fig.savefig(f, format="png", dpi=135, pil_kwargs={"optimize": True})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    draw(parser.parse_args().output)
