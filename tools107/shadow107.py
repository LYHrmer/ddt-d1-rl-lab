"""Post-score inverse geometry for a ground query just beyond the fixture's rear edge."""
import argparse

import numpy as np

from tools77.geometry77 import _rotate
from tools94.common94 import ROOT, digest, read, write
from tools101.score101 import require
from tools105 import score105
from tools106.history106 import oracle_capture_delivery
from tools106.sightline106 import box_sightline, mount_origins
from tools107.mount107 import mount_positions
from tools107.visibility107 import PLAN, PUBLIC


def exit_from_shadow_m(origin, direction, target, lower, upper):
    """First translation tangent to the box. Target is on ground, beyond its rear and inside its width."""
    x, y, z = target
    rear, ymax, top = upper
    _, ymin, floor = lower
    require(z == floor == 0 and x > rear and ymin < y < ymax, "expected an in-lane ground query beyond the rear")
    sx, sy, sz = origin
    ux, uy, uz = direction
    # At x=rear the fraction along the line from target to source is (x-rear)/(x-sx).
    # Multiply each rear-face test by the positive denominator. Each margin becomes affine in a slide.
    h, dx = x-rear, x-sx
    margins = (
        (rear-sx, -ux, "rear_plane"),
        (sz, uz, "ground"),
        (top*dx-h*sz, -top*ux-h*uz, "top"),
        ((y-ymin)*dx+h*(sy-y), -(y-ymin)*ux+h*uy, "lower_y"),
        ((ymax-y)*dx-h*(sy-y), -(ymax-y)*ux-h*uy, "upper_y"),
    )
    require(all(value > 0 for value, _, _ in margins), "the origin must start strictly inside the shadow")
    distance, face = min((-value/rate, name) for value, rate, name in margins if rate < 0)
    # Check the analytic boundary against the general 3-D segment/box clip on both sides of it.
    for delta, expected in ((-1e-6, "blocked"), (1e-6, "clear")):
        pos = origin + (distance + delta)*direction
        require(box_sightline(pos, target, lower, upper)[0] == expected, "incorrect shadow boundary")
    return float(distance), face


def explanation():
    source = PUBLIC / "installation/sightlines.json"
    sight = read(source)
    old_cases = {v["row_id"]: v for v in read(ROOT / read(PLAN)["design"]["public_sightlines"])["details"]}
    rows = score105.load_predictions(score105.PUBLIC / "predictions.npz")
    rows = {k: v[rows["split"] == "validation"] for k, v in rows.items()}
    poses = oracle_capture_delivery(rows, ROOT / "reports/history106/reference101.npz")
    original = mount_positions("original")
    tested = float(mount_positions("forward")[0, 0] - original[0, 0])
    result = []
    for i in sight["old_four_way_blocked_ids"]:
        if i in sight["opened_row_ids"]:
            continue
        case = old_cases[i]
        target, low, high = (np.array(case[k]) for k in ("ground_query_world_xyz_m","box_lower_xyz_m","box_upper_xyz_m"))
        boundaries = []
        for frame, j in (("previous", i-1), ("current", i)):
            pose = poses[j, 0]
            direction = np.array(_rotate(pose[3:], [1., 0., 0.], inverse=False))
            for side, s in zip(("left", "right"), mount_origins(pose, original), strict=True):
                distance, face = exit_from_shadow_m(s, direction, target, low, high)
                boundaries.append((distance, frame, side, face))
        distance, frame, side, face = min(boundaries)
        require(distance >= tested, "computed boundary conflicts with tested mount")
        result.append(dict(row_id=i, total_slide_mm=round(distance*1000, 3), extra_past_forward_mm=round((distance-tested)*1000, 3),
                           capture=frame, mount=side, limiting_edge=face))
    return dict(interpretation="Post-score analytic lower bounds, with the old source rotation and a fixed ground query. "
                "A tangent only touches the boundary. A usable sampled return also needs an available unobstructed sensor beam.",
                tested_slide_mm=round(tested*1000, 3), results=result, source_sha256=digest(source), new_sensor_rays=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("publish", "check"))
    action = parser.parse_args().action
    path = PUBLIC / "shadow_bounds.json"
    result = explanation()
    if action == "publish":
        write(path, result)
    else:
        require(result == read(path), "the post-score shadow bounds changed")
