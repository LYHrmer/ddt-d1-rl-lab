"""Post-hoc position-only localization of D1-94's recorded illegal-contact stops."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from tools94.common94 import ROOT


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _parts(bodies):
    names = [name.split("_", 1) for name in bodies if name.startswith(("FL_", "FR_", "RL_", "RR_"))]
    sides = {leg[1] for leg, _ in names}
    paired = []
    for axle, label in (("F", "front"), ("R", "rear")):
        left = {part for leg, part in names if leg == axle + "L"}
        right = {part for leg, part in names if leg == axle + "R"}
        paired.extend(f"{label}_{part}" for part in sorted(left & right))
    return sides == {"L", "R"}, paired


def _boundary(mj, model, data, floor, wheels, state, tick, flags):
    data.qpos[:] = state[:model.nq]
    mj.mj_fwdPosition(model, data)
    _require(data.time == 0., "position reconstruction advanced native time")
    contacts = []
    for i, c in enumerate(data.contact[:data.ncon]):
        a, b = int(c.geom1), int(c.geom2)
        if floor not in (a, b):
            continue
        robot = b if a == floor else a
        body = int(model.geom_bodyid[robot])
        if robot in wheels or not body:
            continue
        contacts.append({"contact_index": i, "body": model.body(body).name,
                         "geom_id": robot, "geom_name": model.geom(robot).name or None,
                         "signed_distance_m": float(c.dist), "world_position_m": c.pos.tolist()})
    bodies = sorted({c["body"] for c in contacts if c["signed_distance_m"] <= 0.})
    bilateral, paired = _parts(bodies)
    return {"tick": tick, "time_s": tick / 50., "archived_interval_ending_here": (
            {k: bool(v[tick-1]) for k, v in flags.items()} if tick else None),
            "floor_nonwheel_candidates": contacts, "touching_or_penetrating_bodies": bodies,
            "left_and_right": bilateral, "paired_axle_parts": paired}


def run(native_rows):
    """Verify all native sources, then query only the two endpoints around each illegal stop."""
    import mujoco
    from tools87.terrain87 import load_model
    from tools88.contact88 import geometry

    model = load_model("up")
    data = mujoco.MjData(model)
    floor, wheels, _ = geometry(model, mujoco)
    kinds = {"fit", "calibration", "evaluation"}
    counts = {k: {"episodes": 0, "early_stops": 0, "illegal_episodes": 0} for k in sorted(kinds)}
    phases = {k: {"body_episode_counts": Counter(), "bilateral_episode_count": 0,
                  "paired_axle_part_episode_counts": Counter(), "any_nonwheel_touch": 0}
              for k in ("previous", "terminal")}
    seen, cases, sources = set(), [], []
    configured_types = {mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_HFIELD}
    supports = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0 and model.geom_contype[g] and
                model.geom_type[g] in configured_types}
    _require(supports == {floor} and model.nq == 23 and model.nv == 22, "unexpected D1 road or coordinates")
    for row in native_rows:
        kind, seed, reset = row["kind"], int(row["seed"]), int(row["reset"])
        path = (ROOT / Path(row["path"])).resolve()
        _require(kind in kinds and path not in seen and path.is_relative_to(ROOT), "invalid or repeated native source")
        seen.add(path)
        with path.open("rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
            _require(sha == row["sha256"], f"native trace SHA mismatch: {path.name}")
            f.seek(0)
            with np.load(f, allow_pickle=False) as z:
                flags = {k: z[k] for k in ("illegal", "fallen", "terminated", "truncated")}
                n, requested = int(z["completed_transitions"]), int(z["requested_ticks"])
                offset = float(z["wheel_offset_rad_s"])
                q, times = z["state45"], z["time_s"]
                _require(int(z["reset_seed"]) == reset and offset == float(row["offset_rad_s"]),
                         "source seed or wheel offset differs")
        _require(0 < n <= requested and q.shape == (n + 1, 45) and np.isfinite(q).all()
                 and times.shape == (n + 1,) and np.allclose(times, np.arange(n+1)/50., atol=1e-9, rtol=0.),
                 "invalid native trace length, state, or time")
        _require(all(v.shape == (n,) and v.dtype == bool for v in flags.values()) and
                 np.array_equal(flags["terminated"], flags["illegal"] | flags["fallen"]) and
                 not (flags["terminated"][:-1] | flags["truncated"][:-1]).any(), "archived end flags disagree")
        source = {"kind": kind, "seed": seed, "reset": reset, "offset_rad_s": offset, "sha256": sha,
                  "path": path.relative_to(ROOT).as_posix(), "ticks": n, "requested_ticks": requested,
                  "early_stop": n < requested, "final_archived_flags": {k: bool(v[-1]) for k, v in flags.items()}}
        sources.append(source)
        group = counts[kind]
        group["episodes"] += 1
        group["early_stops"] += int(n < requested)
        if not flags["illegal"][-1]:
            continue
        group["illegal_episodes"] += 1
        case = dict(source)
        for phase, tick in (("previous", n - 1), ("terminal", n)):
            endpoint = _boundary(mujoco, model, data, floor, wheels, q[tick], tick, flags)
            case[phase] = endpoint
            totals = phases[phase]
            totals["body_episode_counts"].update(endpoint["touching_or_penetrating_bodies"])
            totals["any_nonwheel_touch"] += bool(endpoint["touching_or_penetrating_bodies"])
            totals["bilateral_episode_count"] += int(endpoint["left_and_right"])
            totals["paired_axle_part_episode_counts"].update(endpoint["paired_axle_parts"])
        cases.append(case)
    for item in phases.values():
        for key in ("body_episode_counts", "paired_axle_part_episode_counts"):
            item[key] = dict(sorted(item[key].items()))
    return {"method": "post-hoc MuJoCo mj_fwdPosition of archived qpos at the last two 20 ms boundaries",
            "counting": "One count per episode and named body per phase, with signed geometric distance <= 0 m. "
                        "Positive-distance contacts inside a margin remain in the candidate list.",
            "interpretation": "The recorded illegal flag is an interval OR over contact normal force >1 N "
                              "for a non-wheel body on the support surface. Endpoint geometry reports proximity "
                              "and penetration only. It cannot identify the first force-threshold crossing.",
            "scene": "D1 87 up, 4 degrees", "mujoco_version": mujoco.__version__,
            "physics_integrations": 0, "policy_calls": 0, "force_queries": 0, "position_queries": 2 * len(cases),
            "by_kind": counts, "flagged_episodes": len(cases), "phase_counts": phases,
            "sources": sources, "cases": cases}


def archived_rows():
    from tools94.common94 import SEEDS, read, seed_dir
    for seed in SEEDS:
        result = read(seed_dir(seed) / "report.json")
        for r in result["rounds"]:
            for source in r["fit"]:
                yield dict(kind="fit", seed=seed, offset_rad_s=0., **source)
            for cell in r["calibration"]:
                for key, source in cell["native"].items():
                    yield dict(kind="calibration", seed=seed, reset=cell["reset"], offset_rad_s=int(key)/100., **source)
        for row in result["evaluation"]:
            for arm in ("initial", "final"):
                yield dict(kind="evaluation", seed=seed, reset=row["reset"], offset_rad_s=0., **row[arm])


if __name__ == "__main__":
    import argparse
    import gzip
    import io
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(archived_rows())
    with args.output.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    print(json.dumps({k: result[k] for k in ("position_queries", "flagged_episodes", "phase_counts")}), flush=True)
