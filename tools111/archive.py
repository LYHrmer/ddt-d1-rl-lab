"""Replay public pre-contact lidar and flat odometry without a robot, physics steps or training."""
import argparse
import json

import numpy as np

from tools91.launch91 import ROOT, digest, read, write
from tools111.control import Packet, Preview, SPEED_GAIN, WheelOdometry, yaw_from_quat


PUBLIC = ROOT/"reports/lidar_compound111"


def cloud_audit():
    file = "reports/pointcloud100_clouds.npz"
    sha = digest(ROOT/file)
    assert sha == read(ROOT/"reports/pointcloud100_manifest.json")["files"][file]
    old = read(ROOT/"reports/pointcloud100.json")
    truth = {(r["seed"], r["layout"], r["capture_tick"], r["visibility"]): r["truth_mm"] for r in old["rows"]}
    rows = []
    with np.load(ROOT/file, allow_pickle=False) as z:
        all_points, queries, keep = z["points"], z["query_xy"], z["angle_keep"]
        for i, (seed, layout, tick) in enumerate(zip(z["cloud_seed"], z["cloud_layout"], z["capture_tick"])):
            # The archive supplies a delayed front-axle midpoint query with a 350 mm forward offset.
            front = queries[i]-[.35, 0.]
            wheels = np.vstack((front, front-[.447, 0.]))
            for visibility in ("clear", "masked"):
                points = all_points[i] if visibility == "clear" else all_points[i][keep]
                c, odom = Preview("both"), WheelOdometry(0.)
                c.observe(2, Packet(0, points, np.zeros(2), 0.), odom, wheels)
                for k in range(2, 9):
                    vx, _, lift = c.targets(k, 1.5, 0., odom, wheels)
                rows.append(dict(seed=int(seed), layout=str(layout), tick=int(tick), visibility=visibility,
                                 query_truth_mm=truth[int(seed), str(layout), int(tick), visibility],
                                 speed_reduction_m_s=round(SPEED_GAIN*1.5-vx, 8),
                                 left_target_mm=round(1000*lift[0], 5), right_target_mm=round(1000*lift[1], 5)))
    summary = []
    for seed in sorted({r["seed"] for r in rows}):
        for view in ("clear", "masked"):
            group = [r for r in rows if r["seed"] == seed and r["visibility"] == view]
            flat = [r for r in group if r["layout"] == "flat"]
            elevated = [r for r in group if abs(r["query_truth_mm"]) > 10.]
            correct = 0
            for r in elevated:
                left, right = r["left_target_mm"], r["right_target_mm"]
                lift, other = (left, right) if r["query_truth_mm"] > 0 else (right, left)
                correct += int(lift >= 5 and other < 5 and r["speed_reduction_m_s"] >= .02)
            flat_active = sum(r["speed_reduction_m_s"] > 1e-7 or
                              max(r["left_target_mm"], r["right_target_mm"]) > 1e-4 for r in flat)
            summary.append(dict(seed=seed, view=view, frames=len(group), flat=len(flat),
                                flat_active=int(flat_active), raised_query=len(elevated), correct_speed_and_side=correct))
    assert len(rows) == 216 and len(summary) == 6
    return dict(source_sha256=sha, rows=rows, summary=summary)


def odom_audit():
    parent = ROOT/"reports/speed110"
    rows = []
    for source in read(parent/"results.json")["episodes"]:
        assert digest(parent/source["file"]) == source["trace_sha256"]
        with np.load(parent/source["file"], allow_pickle=False) as z:
            q, v = z["qpos"], z["qvel"]
            odom = WheelOdometry(yaw_from_quat(q[0,3:7]))
            estimated = []
            for pose, velocity in zip(q,v):
                odom.update(yaw_from_quat(pose[3:7]), velocity[[9,13,17,21]])
                estimated.append(odom.xy.copy())
            error = np.asarray(estimated) - q[:, :2]
            middle = (q[:,0]>=3.) & (q[:,0]<=6.)
            capture = np.flatnonzero(middle[:-2] & middle[2:])
            delay_error = np.linalg.norm(error[capture+2]-error[capture], axis=1)
            uncorrected = np.linalg.norm(q[capture+2,:2]-q[capture,:2],axis=1)
            rows.append(dict(seed=source["seed"], trace_sha256=source["trace_sha256"], windows=len(capture),
                             max_position_error_mm=round(1000*float(np.linalg.norm(error[middle],axis=1).max()), 5),
                             median_uncompensated_40ms_mm=round(1000*float(np.median(uncorrected)), 5),
                             max_compensated_40ms_error_mm=round(1000*float(delay_error.max()), 5)))
    return rows


def result():
    return dict(cloud=cloud_audit(), odometry=odom_audit(), fresh_rays=0, physical_substeps=0, updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    measured = result()
    output = PUBLIC/"archive.json"
    if args.record:
        json.dumps(measured, allow_nan=False)
        PUBLIC.mkdir(exist_ok=True, parents=True)
        write(output, measured)
    assert measured == read(output)
    print(json.dumps(dict(cloud=measured["cloud"]["summary"], odometry=measured["odometry"]), indent=2))
