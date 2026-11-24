"""Publish guarded speed110 records and recalculate the qualification from the raw poses."""
import argparse
import json

import numpy as np

from tools91.launch91 import ROOT, digest, read, write
from tools109.readback109 import copy_new
from tools110.course import SEEDS, assess, command, next_gain


PUBLIC = ROOT / "reports/speed110"
FILM = ROOT / "docs/assets/speed110.mp4"


def publish():
    from tools110.run import BASE, PLAN_SHA, complete

    flat_dir, flat, usage = complete("flat")
    _, preflight, pre_usage = complete("preflight")
    video_dir, video, film_usage = complete("video")
    public_traces = PUBLIC / "trajectories"
    public_traces.mkdir(parents=True, exist_ok=False)
    episodes = []
    for row in flat["episodes"]:
        name = f"flat_{row['seed']}"
        file = f"trajectories/{name}.npz"
        copy_new(flat_dir / name / "trace.npz", PUBLIC / file)
        episodes.append({**row, "file": file})
    copy_new(video_dir / video["filename"], FILM)
    write(PUBLIC / "results.json", dict(
        plan=PLAN_SHA, sources=flat["sources"], episodes=episodes, resources=usage,
        preflight=preflight, preflight_resources=pre_usage,
        preflight_partial=read(BASE / "preflight_01/partial.json"),
        preflight_partial_guard=read(BASE / "guard_preflight_01/report.json"),
        video=video, video_resources=film_usage, calibration_trials=0, terrain_trials=0,
        blind_trials=0, policy_updates=0, world_model_updates=0))


def verify():
    proof = read(PUBLIC / "results.json")
    assert digest(ROOT / "docs/speed110_proposal.md") == proof["plan"]
    for path, expected in proof["sources"].items():
        assert expected == digest(ROOT / path), path
    for key in ("resources", "preflight_resources", "video_resources"):
        usage = proof[key]
        assert usage["startup_effective_bytes"] >= 4 << 30 and usage["minimum_effective_bytes"] >= 2 << 30
        assert len(usage["cpu_affinity"]) == 1 and usage["owned_cgroup_children_empty"]
    initial, preflight = proof["preflight_partial"], proof["preflight"]
    assert initial["status"] == "PARTIAL" and proof["preflight_partial_guard"]["status"] == "INCOMPLETE"
    assert preflight["status"] == "COMPLETE" and preflight["physics_steps"] == 0 and preflight["max_torque_error_nm"] == 0
    assert set(preflight["terrain_rays"]) == {"flat", "ridge", "ripple", "step"}
    for rays in preflight["terrain_rays"].values():
        assert len(rays) == 14 and all(abs(measured - expected) <= .0005 for _, _, measured, expected in rays)
    rows = proof["episodes"]
    assert tuple(row["seed"] for row in rows) == SEEDS["flat"]
    for row in rows:
        path = PUBLIC / row["file"]
        assert row["status"] == "COMPLETE" and row["terrain"] == "flat" and row["gain"] == 1.
        assert row["trace_sha256"] == digest(path)
        with np.load(path, allow_pickle=False) as trace:
            score = assess(trace, row["terrain"])
            assert score == row["score"] and row["initial_pose"] == trace["qpos"][0].tolist()
            n = score["ticks"]
            assert row["counts"] == dict(decisions=n, servo=5*n, physics=10*n, policy_forwards=n)
            np.testing.assert_array_equal(trace["cmd"], [command(k) for k in range(n)])
            np.testing.assert_array_equal(trace["actor_cmd"], trace["cmd"])
            np.testing.assert_array_equal(trace["applied"], np.vstack((np.zeros(16), trace["issued"][:-1])))
            np.testing.assert_array_equal(trace["qpos"][:, :2], trace["xy"])
            np.testing.assert_array_equal(trace["qvel"][:, :2], trace["vxy"])
    assert not any(r["score"]["passed"] for r in rows) and next_gain(rows) is None
    assert all(proof[k] == 0 for k in ("calibration_trials", "terrain_trials", "blind_trials", "policy_updates", "world_model_updates"))
    video = proof["video"]
    assert video["status"] == "COMPLETE" and video["physics_steps"] == 0
    assert digest(FILM) == video["sha256"] and FILM.stat().st_size == video["bytes"]
    for clip in video["clips"]:
        source, = (r for r in rows if r["seed"] == clip["seed"] and r["terrain"] == clip["terrain"])
        assert clip["source_sha256"] == source["trace_sha256"] and clip["end_tick"] == source["score"]["ticks"]
    return dict(episodes=len(rows), physics=sum(r["counts"]["physics"] for r in rows),
                passing=sum(r["score"]["passed"] for r in rows), terrain_trials=proof["terrain_trials"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    if parser.parse_args().publish:
        publish()
    print(json.dumps(verify()))
