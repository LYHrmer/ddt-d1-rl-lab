"""Plot the measured returns for both remaining masked transition failures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools94.common94 import ROOT, read
from tools101.common101 import protocol
from tools102.mask102 import level_points
from tools104.motion104 import PACKETS, PUBLIC, heading, rotation


def draw(output):
    failures = [r for r in read(PUBLIC / "returns.json")["failures"] if r["switched"]]
    failures.sort(key=lambda r: r["truth_mm"] == 0)
    if len(failures) != 2:
        raise ValueError("both remaining transition failures are required")
    data = protocol()["data"]
    geom = read(ROOT / data["sensor_base"])["geometry"]
    fixture, = data["layouts"]["holdout"]
    front = fixture["box_center_x_m"] - fixture["box_half_x_m"]
    rear = fixture["box_center_x_m"] + fixture["box_half_x_m"]
    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    with np.load(PACKETS / "holdout.npz") as p, np.load(PUBLIC / "windows.npz") as w:
        lookup = {tuple(k): j for j, k in enumerate(w["keys"])}
        ids = [row["row_id"] for row in failures]
        local = level_points(p["geometry_depth_m64"][ids], p["valid"][ids], p["capture_body_to_level"][ids], geom)
        for ax, row, pts in zip(axes, failures, local, strict=True):
            i = row["row_id"]
            pose = w["reference_poses"][lookup[(p["episode"][i], p["capture_tick"][i], p["delivery_tick"][i])], 0]
            world = pts[np.isfinite(pts).all(axis=1)].copy()
            world[:, :2] = world[:, :2] @ rotation(heading(pose)).T + pose[:2]
            world[:, 2] += pose[2]
            query = p["world_query_xy"][i, 0 if row["side"] == "left" else 1]
            cloud = world * 1000
            cloud[:, :2] -= 1000 * query
            picked = cloud[np.argmin(np.linalg.norm(cloud[:, :2], axis=1))]
            edge = 1000 * ((front if row["truth_mm"] else rear) - query[0])
            high = geom["box_height_m"] * 1000
            zs = [0, 0, high, high] if row["truth_mm"] else [high, high, 0, 0]
            ax.plot([-52, edge, edge, 52], zs, lw=1.1, color="#52616c", label="已知地形剖面")
            visible = (np.abs(cloud[:, 1]) <= 25) & (np.abs(cloud[:, 0]) < 52)
            ax.scatter(cloud[visible, 0], cloud[visible, 2], s=19, color="#5885aa", alpha=0.8, label="已有回波")
            ax.scatter(picked[0], picked[2], s=141, facecolor="none", edgecolor="#c44434",
                       linewidth=2.2, zorder=4, label="算法选中的回波")
            ax.scatter(0., abs(row["truth_mm"]), marker="D", s=65, facecolor="#20724c",
                       edgecolor="white", linewidth=0.8, zorder=5, label="查询点与真实高度")
            ax.set(title="进入台面" if row["truth_mm"] else "离开台面", xlim=(-52, 52), ylim=(-6, 38),
                   xlabel="相对轮前查询点的前后距离（mm）", yticks=[0, 10, 20, 30])
            ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("离地高度（mm）")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    with output.open("xb") as f:
        fig.savefig(f, format="png", dpi=145, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    draw(parser.parse_args().output)
