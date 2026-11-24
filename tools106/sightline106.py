"""Post-score visibility of the eight original masked exit queries with respect to their box.

For each affected wheel, join its delivered world XY at z=0 to each of the two
physical mounts at the preceding and current capture. A positive length in the
open, nominal 3-D box is blocked. Contact with only a face, edge, or endpoint
is tangent. A segment disjoint from the closed box is clear. Use the fixture's
optical surface without collision margin inflation. Field of view, discrete
beams, the angular shutter, other geometry and robot self-hits are outside this
calculation. A clear ideal line therefore cannot establish an observed return.
"""
from __future__ import annotations

from argparse import ArgumentParser
from fractions import Fraction
from math import dist, isfinite

import numpy as np

from tools77.geometry77 import _rotate
from tools91.launch91 import require
from tools94.common94 import ROOT, digest, read, write
from tools100.common100 import protocol as sensor_protocol
from tools101.common101 import protocol as fixture_protocol
from tools104.motion104 import heading, rotation
from tools105 import score105
from tools106.history106 import oracle_capture_delivery


PUBLIC = ROOT / "reports/history106"
OUTPUT = PUBLIC / "sightlines.json"
STATES = ("blocked", "clear", "tangent")


def box_sightline(origin, target, lower, upper):
    """Return optical class and length in metres strictly inside the open box.

    Clip the finite segment using exact fractions of the supplied finite binary
    coordinates so a single shared boundary has no positive intersection length.
    """
    vectors = [tuple(map(float, v)) for v in (origin, target, lower, upper)]
    require(all(len(v) == 3 and all(isfinite(x) for x in v) for v in vectors), "expected finite 3-D coordinates")
    origin, target, lower, upper = vectors
    require(origin != target and all(lo < hi for lo, hi in zip(lower, upper)), "degenerate segment or box")
    entry, leave = Fraction(0), Fraction(1)
    follows_face = False
    for a, b, lo, hi in zip(origin, target, lower, upper, strict=True):
        a, b, lo, hi = map(Fraction, (a, b, lo, hi))
        direction = b - a
        if direction == 0:
            if a < lo or a > hi:
                return "clear", 0.
            follows_face |= a == lo or a == hi
            continue
        t0, t1 = sorted(((lo - a) / direction, (hi - a) / direction))
        entry, leave = max(entry, t0), min(leave, t1)
        if entry > leave:
            return "clear", 0.
    if follows_face or entry == leave:
        return "tangent", 0.
    return "blocked", float(leave - entry) * dist(origin, target)


def mount_origins(pose, mounts):
    """Rotate both recorded body-frame lever arms into the full capture pose."""
    return np.array([pose[:3] + _rotate(pose[3:], mount, inverse=False) for mount in mounts])


def _counts(lines):
    return {state: sum(line["classification"] == state for line in lines) for state in STATES}


def sightlines():
    """Recompute the post-score explanation using public poses, queries and fixture dimensions."""
    registered = ROOT / "docs/experiments106/history.json"
    predictions = score105.PUBLIC / "predictions.npz"
    reference = PUBLIC / "reference101.npz"
    for p in (predictions, reference):
        seal = read(p.with_name("seal.json"))
        require(seal["status"] == "COMPLETE" and seal["files"][p.name] == digest(p), "unsealed sightline source")
    all_rows = score105.with_truth(score105.load_predictions(predictions))
    rows = {k: values[all_rows["split"] == "validation"] for k, values in all_rows.items()}
    series, truth = rows["series_id"], rows["target_mm"]
    consecutive = np.r_[False, series[1:] == series[:-1]]
    ids = np.flatnonzero(consecutive & (truth == 0) & (np.roll(truth, 1) != 0)
                         & (rows["visibility"] == "masked") & np.isnan(rows["candidate_mm"]))
    require(ids.tolist() == read(registered)["known_exit_row_ids"] and len(ids) == 8,
            "the original eight masked round-105 exits changed")
    poses = oracle_capture_delivery(rows, reference)
    packet_path = score105.PACKETS["validation"]
    with np.load(packet_path, allow_pickle=False) as archive:
        world, local, levels = (archive[k] for k in ("world_query_xy", "query_xy", "capture_body_to_level"))
    sensor = sensor_protocol()["geometry"]
    fixtures = {spec["id"]: spec for spec in fixture_protocol()["data"]["layouts"]["train"]}
    mounts = np.asarray(sensor["mounts_m"])
    height = sensor["box_height_m"]
    details = []
    for i in ids:
        fixture_name, side = str(rows["layout"][i]).rsplit("_", 1)
        fixture = fixtures[fixture_name]
        require(rows["side"][i] == side and side in ("left", "right"), "fixture mirror and wheel disagree")
        wheel = 0 if side == "left" else 1
        cx = fixture["box_center_x_m"]
        cy = fixture["box_abs_y_m"] * (1 if side == "left" else -1)
        hx, hy = fixture["box_half_x_m"], fixture["box_half_y_m"]
        lower = (cx - hx, cy - hy, 0.)
        upper = (cx + hx, cy + hy, height)
        target = np.r_[world[i, wheel], 0.]
        previous_target = world[i - 1, wheel]
        require(lower[0] <= previous_target[0] <= upper[0] and lower[1] <= previous_target[1] <= upper[1]
                and target[0] > upper[0] and lower[1] <= target[1] <= upper[1],
                "selected wheel did not exit behind its own fixture")
        delivery = poses[i, 1]
        np.testing.assert_allclose(local[i] @ rotation(heading(delivery)).T + delivery[:2],
                                   world[i], atol=1e-10, rtol=0)
        lines = []
        for when, j in (("previous", i-1), ("current", i)):
            capture = poses[j, 0]
            origins = mount_origins(capture, mounts)
            leveled = mounts @ levels[j]
            leveled[:, :2] = leveled[:, :2] @ rotation(heading(capture)).T
            np.testing.assert_allclose(origins, leveled + capture[:3], atol=1e-10, rtol=0)
            for mount, origin in zip(("left", "right"), origins, strict=True):
                state, length = box_sightline(origin, target, lower, upper)
                lines.append(dict(capture=when, capture_tick=int(rows["capture_tick"][j]), mount=mount,
                                  origin_world_xyz_m=origin.round(10).tolist(), classification=state,
                                  inside_length_mm=round(length * 1000., 6)))
        details.append(dict(row_id=int(i), source_seed=int(rows["source_seed"][i]), episode=int(rows["episode"][i]),
                            layout=str(rows["layout"][i]), wheel=side, delivery_tick=int(rows["delivery_tick"][i]),
                            ground_query_world_xyz_m=target.round(10).tolist(), box_lower_xyz_m=np.round(lower, 10).tolist(),
                            box_upper_xyz_m=np.round(upper, 10).tolist(), lines=lines))
    lines = [line for row in details for line in row["lines"]]
    require(len(lines) == 32, "incomplete ideal sightline roster")
    inputs = (registered, ROOT / "docs/experiments100/pointcloud_density.json",
              ROOT / "docs/experiments101/predictive_perception.json", packet_path, predictions, reference)
    return dict(scope="Post-score geometric explanation of the eight original masked round-105 exits.",
                definition="At both capture times use the physical mount XYZ and the same delivered affected-wheel XY at "
                           "world z=0. Blocked requires positive segment length strictly inside the nominal optical box. "
                           "Tangent touches only its boundary. Clear is disjoint from the closed box.",
                coverage="Only the original 30 mm fixture is tested. Robot self-occlusion, angular shutter, "
                         "discrete sensor beams, field of view and all other geometry are outside this calculation. "
                         "Clear does not establish a measured return. The collision margin is not part of the optical box.",
                sources_sha256={str(p.relative_to(ROOT)): digest(p) for p in inputs},
                rows=len(details), ideal_segments=len(lines), native_decisions=0, new_sensor_rays=0, model_updates=0,
                counts=_counts(lines),
                by_capture={when: _counts([line for line in lines if line["capture"] == when])
                            for when in ("previous", "current")}, details=details)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("publish", "check"))
    args = parser.parse_args()
    if args.action == "publish" and OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = sightlines()
    if args.action == "publish":
        write(OUTPUT, result)
    else:
        require(read(OUTPUT) == result, "published sightline evidence differs from the sealed inputs")
    print(f"{result['rows']} original exits, {result['ideal_segments']} ideal segments, {result['counts']}")


if __name__ == "__main__":
    main()
