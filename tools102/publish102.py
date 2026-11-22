"""Release the complete position comparison and its synchronized native pose film."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, digest, read, write
from tools100.publish100 import copy_verified
from tools101.common101 import PUBLIC as PARENT
from tools102.common102 import MOVIE, PLAN_SHA, PUBLIC, STAGES, protocol, sources
from tools102.replay102 import readback


def assets():
    return {"replay": {n: PUBLIC / n for n in ("packets.npz", "predictions.npz", "report.json")},
            "video": {"poses.npz": PUBLIC / "poses.npz", "video.json": PUBLIC / "video.json",
                      "driver.json": PUBLIC / "driver.json", "occlusion102.mp4": MOVIE}}


def verify_movie(plan):
    movie = read(PUBLIC / "video.json")
    spec = plan["video"]
    source_data = read(PARENT / "data.json")
    source, = (s for s in source_data["sources"] if s["source_seed"] == plan["source_seed"])
    trial, = (e for e in source["admitted_flat_episodes"] if e["episode"] == spec["episode"])
    ticks = np.array(plan["capture_ticks"])
    lo, hi = spec["tick_range"]
    require(movie["status"] == "COMPLETE" and movie["protocol_sha256"] == PLAN_SHA and movie["cover_frames"] == 0
            and movie["layout"] == spec["layout"] and movie["mask_side"] == spec["target_side"]
            and movie["native_source_seed"] == plan["source_seed"] and movie["native_episode"] == spec["episode"]
            and movie["native_trace_sha256"] == source["trace_sha256"]
            and movie["native_episode_qpos_sha256"] == trial["physical_episode_qpos_sha256"]
            and movie["fps"] == spec["fps"] and movie["playback_rate"] == spec["motion_speed"]
            and movie["video_sha256"] == digest(MOVIE) and movie["bytes"] == MOVIE.stat().st_size
            and 0 < movie["bytes"] <= spec["maximum_bytes"], "unexpected film, timing or original robot trajectory")
    for entry, path in (("poses_sha256", PUBLIC / "poses.npz"), ("packets_sha256", PUBLIC / "packets.npz"),
                        ("report_sha256", PUBLIC / "report.json"), ("holdout_sha256", PARENT / "holdout.npz"),
                        ("source_index_sha256", PARENT / "data.json")):
        require(movie[entry] == digest(path), f"film source differs: {entry}")
    require(all(movie[key] == 0 for key in ("world_updates", "perception_updates", "policy_updates",
                                             "policy_calls", "physics_substeps", "new_rays")), "unexpected video work")
    driver = read(PUBLIC / "driver.json")
    require(driver["renderer"] == movie["renderer"] and "llvmpipe" in driver["renderer"].lower(),
            "movie did not use the verified CPU device")
    with np.load(PUBLIC / "poses.npz", allow_pickle=False) as poses:
        require(int(poses["source_seed"]) == plan["source_seed"] and int(poses["episode"]) == spec["episode"]
                and poses["qpos"].shape == (hi - ticks[0] + 1, 23) and np.isfinite(poses["qpos"]).all()
                and np.array_equal(poses["tick"], np.arange(ticks[0], hi + 1))
                and np.array_equal(poses["shown_ticks"], np.arange(lo, hi + 1))
                and np.array_equal(poses["capture_ticks"], ticks)
                and np.array_equal(poses["delivery_ticks"], ticks + plan["delay_ticks"]),
                "film did not retain the registered native pose and packet clocks")
    with np.load(PUBLIC / "packets.npz", allow_pickle=False) as packets:
        selected = ((packets["episode"] == spec["episode"]) & (packets["layout"] == spec["layout"])
                    & (packets["mask_side"] == spec["target_side"]))
        rows = np.flatnonzero(selected)
        rows = rows[np.argsort(packets["capture_tick"][rows])]
        require(len(rows) == len(ticks) and np.array_equal(movie["cloud_ids"], packets["cloud_id"][rows])
                and np.array_equal(movie["active"], packets["active"][rows])
                and np.array_equal(movie["delivery_ticks"], packets["delivery_tick"][rows]), "film shows wrong packets")
    frames_per_tick = spec["fps"] / (movie["native_rate_hz"] * spec["motion_speed"])
    holds = {h["delivery_tick"]: h for h in movie["hold_events"]}
    for t in (*spec["focus_delivery_ticks"], spec["return_delivery_tick"]):
        require(holds[t]["frames"] >= 5 * spec["fps"], "focus packet is not held long enough")
    cumulative = 0
    for t, hold in sorted(holds.items()):
        require(hold["first_frame"] == (t - lo) * frames_per_tick + cumulative, "film pause is out of sync")
        cumulative += hold["frames"]
    require(movie["frames"] == (hi - lo + 1) * frames_per_tick + cumulative
            and movie["duration_s"] == movie["frames"] / spec["fps"]
            and int(movie["stream"]["nb_read_frames"]) == movie["frames"], "film readback clock differs")
    return {str(t): holds[t]["first_frame"] / spec["fps"] for t in (*spec["focus_delivery_ticks"], spec["return_delivery_tick"])}


def verify(replay_models=False):
    from tools102.mesa_video102 import video_sources

    plan = protocol()
    manifest = read(PUBLIC / "manifest.json")
    require(manifest["protocol_sha256"] == PLAN_SHA and set(manifest["stages"]) == set(STAGES),
            "incomplete publication")
    for stage, outputs in assets().items():
        record = manifest["stages"][stage]
        seal, limits = record["seal"], plan["resources"]
        expected_sources = video_sources() if stage == "video" else sources(stage)
        require(seal["protocol_sha256"] == PLAN_SHA and seal["stage"] == stage and seal["status"] == "COMPLETE"
                and seal["sources"] == expected_sources, "stage sources changed")
        for name, public in outputs.items():
            require(digest(public) == seal["files"][name], f"published {name} differs from the completed stage")
        res = record["resources"]
        require(res["startup_bytes"] >= limits["admission_min_bytes"]
                and res["minimum_running_bytes"] >= limits["runtime_min_bytes"]
                and len(res["cpu_affinity"]) == 1, "insufficient resource reserve")
    require(manifest["stages"]["video"]["seal"]["replay_seal_sha256"] == manifest["stages"]["replay"]["seal_sha256"],
            "video refers to a different replay")
    attempt = PUBLIC / "video_attempt01.json"
    failed = read(attempt)
    require(digest(attempt) == manifest["earlier_video_attempt"]["receipt_sha256"]
            and failed["status"] == "PARTIAL" and failed["frames"] == 0
            and "CPU rasterizer" in failed["error"] and failed["protocol_sha256"] == PLAN_SHA,
            "the earlier failed export is not accounted for")
    prior = manifest["stages"]["video"]["seal"]["first_attempt"]
    references = manifest["earlier_video_attempt"]
    require(prior["movie_receipt"] == references["receipt_sha256"]
            and prior["partial"] == references["worker_partial_sha256"] and prior["guard"] == references["guard_sha256"],
            "second export refers to a different first attempt")
    return dict(measurements=readback(PUBLIC, plan, replay_models=replay_models), focus_at_seconds=verify_movie(plan))


def publish(desktop=None):
    from tools102.launch102 import completed, folder, guard
    from tools102.mesa_video102 import completed_video

    plan = protocol()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol_sha256=PLAN_SHA, stages={})
    for stage, outputs in assets().items():
        output, resources = completed_video(plan) if stage == "video" else completed(stage, plan)
        for name, public in outputs.items():
            copy_verified(output / name, public)
        manifest["stages"][stage] = dict(seal=read(output / "seal.json"), seal_sha256=digest(output / "seal.json"),
                                         resources=resources)
    first_video = folder("video")
    manifest["earlier_video_attempt"] = dict(
        receipt_sha256=copy_verified(first_video / "video.json", PUBLIC / "video_attempt01.json"),
        worker_partial_sha256=digest(first_video / "partial.json"),
        guard_sha256=digest(guard("video") / "report.json"))
    path = PUBLIC / "manifest.json"
    if not path.exists():
        write(path, manifest)
    require(read(path) == manifest, "preserving a different existing publication")
    result = verify()
    if desktop:
        copy_verified(MOVIE, desktop)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--replay-models", action="store_true")
    parser.add_argument("--desktop", type=Path)
    args = parser.parse_args()
    result = verify(args.replay_models) if args.check or args.replay_models else publish(args.desktop)
    print(json.dumps(result))
