"""Package guarded native trajectories or independently recalculate the published 1.5 m/s verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from tools109.score109 import score


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/"reports/speed109"
FILM = ROOT/"docs/assets/speed109.mp4"
DEV = (109011, 109023, 109037)
BLIND = (109101, 109103, 109107)
SCALES = {"actor5": 5., "actor24": 24., "reference24": 24.}
WHEELS = [3, 7, 11, 15]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_new(src, dst):
    with src.open("rb") as reader, dst.open("xb") as writer:
        shutil.copyfileobj(reader, writer)


def verified_rows(stage, pack):
    result = []
    for row in pack["episodes"]:
        path = PUBLIC/row["file"]
        assert sha(path) == row["trace_sha256"]
        with np.load(path, allow_pickle=False) as z:
            s = score(z)
            assert s == row["score"] and row["status"] == "COMPLETE"
            n, arm = s["ticks"], row["arm"]
            assert row["counts"]["policy_transitions"] == n and row["counts"]["native_steps"] == 10*n
            assert row["counts"]["pd_updates"] == 5*n and row["counts"]["resets"] == 1
            q, dq = z["qpos"], z["qvel"]
            reset = hashlib.sha256(q[0].tobytes()+dq[0].tobytes()).hexdigest()
            assert reset == row["initial_state_sha256"]
            ticks = np.arange(n).reshape(-1, 1)
            history = np.concatenate((ticks, np.maximum(0, ticks + np.arange(-10, 0))), axis=1)
            cmd = np.interp(history, [0, 50, 150, 275, 350, 400], [0., 0., 1.5, 1.5, 0., 0.])
            np.testing.assert_array_equal(z["command_history"], (2.*cmd).astype(np.float32))
            np.testing.assert_array_equal(z["command"], cmd[:, 0])
            wanted = z["nominal"].copy()
            if arm == "reference24":
                speed = 1.02*cmd[:, 0]/.087
                target = speed + (.1*speed+.2*np.sign(speed))/.5
                wanted[:, WHEELS] = (target / 24.)[:, None]
            np.testing.assert_array_equal(z["issued"], wanted)
            expected = np.vstack((np.zeros(4), wanted[:-1, WHEELS]))*SCALES[arm]
            np.testing.assert_array_equal(z["wheel_target"], expected)
            qw, qx, qy, qz = q[0, 3:7]
            yaw = np.arctan2(2.*(qw*qz+qx*qy), 1.-2.*(qy*qy+qz*qz))
            direction = np.array([np.cos(yaw), np.sin(yaw)])
            forward = dq[:, :2] @ direction
            at_speed = (np.diff(q[:, :2], axis=0) @ direction) * 50. >= 1.5
            changes = np.flatnonzero(np.diff(np.r_[False, at_speed, False].astype(int))).reshape(-1, 2)
            longest = max((end-start for start, end in changes), default=0)/50.
            result.append(dict(stage=stage, seed=row["seed"], arm=arm, ticks=n, passed=s["passed"],
                               peak_forward_m_s=float(forward.max()), peak_horizontal_m_s=s["peak_horizontal_speed_m_s"],
                               longest_contiguous_forward_at_least_1p5_s=float(longest),
                               last_interval_command_m_s=float(cmd[-1, 0]), last_base_height_m=float(q[-1, 2]),
                               height_change_since_1s_m=float(q[-1, 2]-q[50, 2]) if n >= 50 else None,
                               final_mean_knee_rad=float(q[-1, [9, 13, 17, 21]].mean()),
                               wheel_saturation_ticks=sum(s["wheel_saturated_ticks"]), failed=s["failed"]))
    return result


def verify():
    proof = json.loads((PUBLIC/"results.json").read_text())
    assert sha(ROOT/"docs/speed109_proposal.md") == proof["proposal_sha256"]
    for name, expected in proof["sources"].items():
        assert sha(ROOT/name) == expected, name
    all_rows = []
    for stage, pack in proof["stages"].items():
        all_rows.extend(verified_rows(stage, pack))
        resource = pack["resources"]
        assert resource["startup_effective_bytes"] >= 4<<30 and resource["minimum_effective_bytes"] >= 2<<30
        assert len(resource["cpu_affinity"]) == 1 and resource["owned_cgroup_children_empty"]
    for stage, arms in [("actors", ("actor5", "actor24")), ("reference", ("reference24",))]:
        assert [(r["seed"], r["arm"]) for r in all_rows if r["stage"] == stage] == [(s,a) for s in DEV for a in arms]
    for seed in DEV:
        receipts = [r for s in ("actors", "reference") for r in proof["stages"][s]["episodes"] if r["seed"] == seed]
        assert len(receipts) == 3 and len({r["initial_state_sha256"] for r in receipts}) == 1
    choice = next((a for a in ("actor24", "reference24")
                   if all(r["passed"] for r in all_rows if r["stage"] != "blind" and r["arm"] == a)), None)
    held = [(r["seed"], r["arm"]) for r in all_rows if r["stage"] == "blind"]
    assert held == ([] if choice is None else [(s, choice) for s in BLIND])
    assert proof["blind_candidate"] == choice and proof["policy_updates"] == proof["world_updates"] == 0
    film = proof["film"]
    assert sha(FILM) == film["sha256"] and FILM.stat().st_size == film["bytes"]
    for clip in film["clips"]:
        source, = [r for r in proof["stages"][clip["stage"]]["episodes"]
                   if r["seed"] == clip["seed"] and r["arm"] == clip["arm"]]
        assert source["trace_sha256"] == clip["source_trace_sha256"] and source["score"]["ticks"] == clip["end_tick"]
    full = sum(r["ticks"] for r in all_rows)
    return {"trials": len(all_rows), "decisions": full, "physics_substeps": 10*full,
            "blind_candidate": choice, "blind_trials": len(held), "rows": all_rows}


def publish():
    from tools109.run109 import PROPOSAL_SHA, completed, select_candidate, sources
    from tools109.cruise109 import CHECKPOINT_SHA

    choice = select_candidate()
    stages = ["actors", "reference"] + (["blind"] if choice else [])
    video_dir, film, video_usage = completed("video")
    PUBLIC.mkdir(exist_ok=False)
    (PUBLIC/"trajectories").mkdir()
    data = {}
    for stage in stages:
        root, report, usage = completed(stage)
        episodes = []
        for row in report["episodes"]:
            tag = str(row["seed"])+"_"+row["arm"]
            name = "trajectories/"+tag+".npz"
            copy_new(root/tag/"trace.npz", PUBLIC/name)
            episodes.append({**row, "file": name})
        data[stage] = {"resources": usage, "episodes": episodes}
    copy_new(video_dir/film["filename"], FILM)
    proof = dict(proposal_sha256=PROPOSAL_SHA, checkpoint_sha256=CHECKPOINT_SHA, sources=sources(), stages=data,
                 blind_candidate=choice, policy_updates=0, world_updates=0, film_resources=video_usage,
                 film={key: film[key] for key in ("filename", "sha256", "bytes", "clips", "frames", "stream")})
    with (PUBLIC/"results.json").open("x") as f:
        json.dump(proof, f, indent=2, allow_nan=False)
        f.write("\n")
    verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.publish:
        publish()
    print(json.dumps(verify(), indent=2, allow_nan=False))
