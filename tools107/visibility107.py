"""Paired optical-line diagnostic on the old eight rear-edge failure examples."""
import numpy as np

from tools77.geometry77 import _rotate
from tools94.common94 import ROOT, digest, read
from tools100.common100 import protocol as sensor_protocol
from tools101.score101 import require
from tools105 import score105
from tools106.history106 import oracle_capture_delivery
from tools106.sightline106 import box_sightline, mount_origins
from tools107.mount107 import mount_positions


PLAN = ROOT / "docs/experiments107/installation.json"
PUBLIC = ROOT / "reports/mount107"


def direction_in_view(target, origin, quaternion, spec):
    delta = _rotate(quaternion, target - origin, inverse=True)
    distance = float(np.linalg.norm(delta))
    azimuth = float(np.rad2deg(np.arctan2(delta[1], delta[0])))
    depression = float(np.rad2deg(np.arctan2(-delta[2], np.hypot(*delta[:2]))))
    inside = (spec["range_min_m"] <= distance <= spec["range_max_m"]
              and min(spec["azimuth_deg"]) <= azimuth <= max(spec["azimuth_deg"])
              and min(spec["depression_deg"]) <= depression <= max(spec["depression_deg"]))
    low, high = spec["mask_depression_deg"]
    shutter = abs(azimuth) <= spec["mask_abs_azimuth_max_deg"] and low <= depression <= high
    return dict(range_m=round(distance, 8), azimuth_deg=round(azimuth, 6), depression_deg=round(depression, 6),
                in_angular_range=bool(inside), in_masked_angular_range=bool(inside and not shutter))


def evaluate():
    design = read(PLAN)["design"]
    archive = ROOT / design["public_sightlines"]
    old = read(archive)
    require([d["row_id"] for d in old["details"]] == design["known_masked_exits"], "registered roster changed")
    parent = score105.load_predictions(score105.PUBLIC / "predictions.npz")
    rows = {k: v[parent["split"] == "validation"] for k, v in parent.items()}
    poses = oracle_capture_delivery(rows, ROOT / "reports/history106/reference101.npz")
    spec = sensor_protocol()["geometry"]
    cases = []
    for case in old["details"]:
        i = case["row_id"]
        target = np.asarray(case["ground_query_world_xyz_m"])
        results = {}
        for arm in ("original", "forward"):
            lines = []
            for label, j in (("previous", i - 1), ("current", i)):
                require(rows["series_id"][j] == rows["series_id"][i], "scan crosses acquisition stream")
                pose = poses[j, 0]
                for side, origin in zip(("left", "right"), mount_origins(pose, mount_positions(arm)), strict=True):
                    state, length = box_sightline(origin, target, case["box_lower_xyz_m"], case["box_upper_xyz_m"])
                    lines.append(dict(capture=label, mount=side, classification=state, inside_mm=round(1000*length, 6),
                                      **direction_in_view(target, origin, pose[3:], spec)))
            results[arm] = lines
        require([v["classification"] for v in results["original"]] == [v["classification"] for v in case["lines"]],
                "original placement does not reproduce old sightlines")
        cases.append(dict(row_id=i, source_seed=case["source_seed"], lines=results))

    def clear(case, arm):
        return any(v["classification"] == "clear" for v in case["lines"][arm])

    blocked = [c for c in cases if all(v["classification"] == "blocked" for v in c["lines"]["original"])]
    require(len(blocked) == 6, "unexpected original blocked roster")
    opened = [c["row_id"] for c in blocked if clear(c, "forward")]
    counts = {}
    for arm in ("original", "forward"):
        lines = [v for c in cases for v in c["lines"][arm]]
        counts[arm] = {state: sum(v["classification"] == state for v in lines) for state in ("clear", "tangent", "blocked")}
        counts[arm]["clear_in_masked_angular_range"] = sum(
            v["classification"] == "clear" and v["in_masked_angular_range"] for v in lines)
    return dict(status="COMPLETE", passed=bool(opened), opened_row_ids=opened,
                old_four_way_blocked_ids=[c["row_id"] for c in blocked], counts=counts, cases=cases,
                parent_sha256=digest(archive), native_decisions=0, sensor_rays=0)
