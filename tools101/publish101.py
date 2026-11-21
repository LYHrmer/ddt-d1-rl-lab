"""Publish immutable paired observations and checkpoints, and verify the public experiment."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, digest, read, write
from tools100.publish100 import copy_verified
from tools100.score100 import TOLERANCE_MM, grade
from tools101.common101 import MOVIE, PLAN_SHA, PUBLIC, STAGES, producer, protocol
from tools101.score101 import score_archives


def verify_data(plan):
    data = read(PUBLIC / "data.json")
    budget = plan["budget"]
    require(data["status"] == "COMPLETE" and data["parent_stage_sha256"] == plan["data"]["source_stage_sha256"]
            and data["counts"]["independent_clouds"] == budget["independent_clouds"]
            and data["counts"]["rays_per_cloud"] == budget["directions_per_cloud"], "acquisition record differs")
    require({str(s["source_seed"]): s["trace_sha256"] for s in data["sources"]} == plan["data"]["trace_sha256"],
            "native trace identity differs")
    for clock in ("physical_steps", "policy_updates", "world_updates", "perception_updates"):
        require(data["counts"][clock] == 0, "unexpected work during fixed-pose acquisition")
    identities, poses, fixtures = set(), set(), set()
    for split, receipt in data["splits"].items():
        path = PUBLIC / f"{split}.npz"
        roster = receipt["episode_roster"]
        ids = {(e["source_seed"], e["episode"]) for e in roster}
        fingerprints = {e["physical_episode_qpos_sha256"] for e in roster}
        shapes = {tuple(spec[k] for k in ("box_center_x_m", "box_half_x_m", "box_abs_y_m", "box_half_y_m"))
                  for spec in plan["data"]["layouts"][split]}
        require(receipt["sha256"] == digest(path) and receipt["rows"] == budget[f"{split}_rows"]
                and len(ids) == len(fingerprints) == budget[f"{split}_episodes"]
                and not (ids & identities or fingerprints & poses or shapes & fixtures), "holdout overlaps training")
        identities.update(ids)
        poses.update(fingerprints)
        fixtures.update(shapes)
        require(all(b["first_tick"] == 0 and b["last_tick"] == max(plan["capture_ticks"]) + plan["delay_ticks"]
                    and b["wheel_max_x_m"] < b["minimum_layout_leading_edge_x_m"] for b in receipt["prefix_bounds"])
                and len(receipt["prefix_bounds"]) == len(ids), "unverified precontact prefix")
        with np.load(path, allow_pickle=False) as packets:
            series, ticks = packets["series_id"], packets["capture_tick"]
            require(np.array_equal(np.unique(packets["source_seed"]), plan["data"]["source_seeds"][split])
                    and set(zip(packets["source_seed"].tolist(), packets["episode"].tolist())) == ids
                    and set(packets["layout"].tolist()) == set(receipt["layouts"])
                    and np.array_equal(packets["delivery_tick"], ticks + plan["delay_ticks"])
                    and len(np.unique(series)) == receipt["series"] and len(ticks) == receipt["rows"]
                    and all(np.array_equal(ticks[series == s], plan["capture_ticks"]) for s in np.unique(series)),
                    "missing, repeated or cross-episode packet")
    require(set(data["splits"]) == {"train", "holdout"}, "incomplete split roster")


def verify_updates(plan):
    fit_folder = PUBLIC / "fit"
    fit = read(fit_folder / "fit.json")
    events = [json.loads(line) for line in (fit_folder / "updates.jsonl").read_text().splitlines()]
    require(fit["status"] == "COMPLETE" and fit["train_sha256"] == digest(PUBLIC / "train.npz")
            and fit["seeds"] == plan["seeds"] and len(events) == fit["updates"] == plan["budget"]["perception_updates"],
            "fit is incomplete")
    for seed in plan["seeds"]:
        pair = fit["per_seed"][str(seed)]
        history = []
        for arm, loss_weight in (("baseline", 0), ("predictive", 1)):
            info = pair["arms"][arm]
            rows = [e for e in events if e["seed"] == seed and e["arm"] == arm]
            n = plan["fit"]["updates_per_arm"]
            require(info["updates"] == n and info["lambda_next"] == loss_weight
                    and info["sha256"] == digest(fit_folder / info["file"])
                    and [e["update"] for e in rows] == list(range(1, n + 1)), "missing update or changed checkpoint")
            history.append([e["batch_sha256"] for e in rows])
        require(history[0] == history[1] and pair["arms"]["baseline"]["first"] == pair["arms"]["predictive"]["first"],
                "paired optimization changed inputs or initialization")
    return digest(fit_folder / "fit.json")


def raised_failure_profile(holdout, predictions):
    """Descriptive follow-up after grading. Each seed sees the same raised, masked observations."""
    with np.load(holdout, allow_pickle=False) as data, np.load(predictions, allow_pickle=False) as estimates:
        selected = (data["visibility"] == "masked") & (np.abs(data["target_mm"]) > TOLERANCE_MM)
        truth = data["target_mm"][selected]
        result = {}
        for arm in ("baseline", "predictive"):
            by_seed = []
            for seed, values in zip(estimates["seeds"], estimates[f"{arm}_mm"], strict=True):
                values = values[selected]
                passed = [grade(t, float(e) if np.isfinite(e) else None) for t, e in zip(truth, values, strict=True)]
                failed = ~np.array(passed)
                lower = np.abs(truth) - TOLERANCE_MM
                by_seed.append(dict(seed=int(seed), raised_packets=len(truth), failed=int(failed.sum()),
                                    undershot_magnitude=int((failed & (np.abs(values) < lower)).sum()),
                                    missing=int(np.isnan(values).sum())))
            result[arm] = by_seed
    return result


def compare_geometric_views(holdout):
    """Compare clear and masked outputs on identical physical captures, after the registered test."""
    with np.load(holdout, allow_pickle=False) as data:
        cloud, view = data["cloud_id"], data["visibility"]
        geometry, valid = data["geometric_mm"], data["valid"]
    sorted_rows = {}
    for kind in ("clear", "masked"):
        rows = np.flatnonzero(view == kind)
        sorted_rows[kind] = rows[np.argsort(cloud[rows])]
    clear, masked = sorted_rows["clear"], sorted_rows["masked"]
    require(np.array_equal(cloud[clear], cloud[masked]), "physical captures do not match")
    return dict(captures=len(clear), clear_valid_counts=np.unique(valid[clear].sum(axis=1)).tolist(),
                masked_valid_counts=np.unique(valid[masked].sum(axis=1)).tolist(),
                identical_finite_estimates=int(np.sum(np.isfinite(geometry[clear]) & (geometry[clear] == geometry[masked]))))


def verify_public():
    plan = protocol()
    manifest = read(PUBLIC / "manifest.json")
    require(manifest["protocol_sha256"] == PLAN_SHA and set(manifest["stages"]) == set(STAGES),
            "publication does not bind the complete registered experiment")
    for filename, expected in manifest["files"].items():
        require(digest(ROOT / filename) == expected, f"published file changed: {filename}")
    for stage, record in manifest["stages"].items():
        require(record["seal"]["sources"] == producer(stage) and record["seal"]["protocol_sha256"] == PLAN_SHA
                and record["seal"]["status"] == "COMPLETE", f"{stage} producer or receipt differs")
        res, limits = record["resources"], plan["resources"]
        require(res["startup_bytes"] >= limits["admission_min_bytes"]
                and res["minimum_running_bytes"] >= limits["runtime_min_bytes"]
                and len(res["cpu_affinity"]) == 1, f"{stage} reserve or CPU assignment differs")
        for parent, checksum in record["seal"]["parents"].items():
            require(manifest["stages"][parent]["seal_sha256"] == checksum, "broken stage lineage")
    verify_data(plan)
    fit_hash = verify_updates(plan)
    holdout, predictions = PUBLIC / "holdout.npz", PUBLIC / "predictions.npz"
    report = read(PUBLIC / "report.json")
    bindings = dict(holdout_sha256=digest(holdout), fit_sha256=fit_hash, predictions_sha256=digest(predictions))
    require(report["status"] == "COMPLETE" and report["protocol_sha256"] == PLAN_SHA
            and bindings == read(PUBLIC / "forecast_seal.json") and all(report[k] == v for k, v in bindings.items()),
            "forecasts, fit or score are not from the same experiment")
    with np.load(predictions, allow_pickle=False) as grid:
        require(grid["fit_sha256"].item() == fit_hash, "different network used in holdout predictions")
    score = score_archives(holdout, predictions, plan["seeds"])
    require(report["score"] == score, "public rows do not reproduce the promotion decision")
    require(manifest["posthoc_raised_failures"] == raised_failure_profile(holdout, predictions),
            "post-hoc height failure diagnosis differs from archived estimates")
    latent = read(PUBLIC / "latent.json")
    count = (len(plan["capture_ticks"]) - 1) * plan["budget"]["holdout_episodes"] * 2
    for seed in plan["seeds"]:
        for view in plan["visibilities"]:
            cell = latent["per_seed"][str(seed)][view]
            require(cell["pairs"] == count and all(np.isfinite(x) and x >= 0 for x in cell.values()),
                    "incomplete or non-finite next-packet diagnostic")
    movie = read(PUBLIC / "video.json")
    require(movie["status"] == "COMPLETE" and movie["protocol_sha256"] == PLAN_SHA
            and movie["video_sha256"] == digest(MOVIE) and movie["report_sha256"] == digest(PUBLIC / "report.json")
            and movie["predictions_sha256"] == bindings["predictions_sha256"]
            and movie["holdout_sha256"] == bindings["holdout_sha256"]
            and movie["cover_frames"] == 0 and 0 < movie["bytes"] == MOVIE.stat().st_size <= 5_000_000
            and all(movie[k] == 0 for k in ("new_rays", "physics_substeps", "perception_updates",
                                            "world_updates", "policy_updates")),
            "movie or receipt does not match the measured comparisons")
    return score


def replay_weights():
    """Independently load the six public checkpoints and regenerate the outputs without updating a parameter."""
    import torch
    from tools101.fit101 import predict

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with TemporaryDirectory(prefix="d1-jeplo101-readback-") as tmp:
        output = predict(PUBLIC / "holdout.npz", PUBLIC / "fit", Path(tmp), protocol())
        with np.load(output, allow_pickle=False) as replay, np.load(PUBLIC / "predictions.npz", allow_pickle=False) as old:
            for arm in ("baseline_mm", "predictive_mm"):
                np.testing.assert_allclose(replay[arm], old[arm], rtol=0, atol=0.0002, equal_nan=True)
        latent = read(Path(tmp) / "latent.json")
        published = read(PUBLIC / "latent.json")
        for seed, rows in published["per_seed"].items():
            for view, metrics in rows.items():
                for metric, value in metrics.items():
                    np.testing.assert_allclose(latent["per_seed"][seed][view][metric], value, rtol=2e-5, atol=1e-7)


def publish(desktop=None):
    from tools101.launch101 import completed

    plan = protocol()
    stages = {s: completed(s, plan) for s in STAGES}
    data, fit, ev, video = (stages[s][0] for s in STAGES)
    (PUBLIC / "fit").mkdir(parents=True, exist_ok=True)
    names = ["fit.json", "updates.jsonl", *(f"seed_{s}_{a}.pt" for s in plan["seeds"] for a in ("baseline", "predictive"))]
    assets = [(data / n, PUBLIC / n) for n in ("train.npz", "holdout.npz", "data.json")]
    assets.extend((fit / n, PUBLIC / "fit" / n) for n in names)
    assets.extend((ev / n, PUBLIC / n) for n in ("predictions.npz", "report.json", "forecast_seal.json", "latent.json"))
    assets.extend(((video / "video.json", PUBLIC / "video.json"), (video / "jeplo101.mp4", MOVIE)))
    manifest = dict(protocol_sha256=PLAN_SHA,
                    files={str(dest.relative_to(ROOT)): copy_verified(src, dest) for src, dest in assets},
                    posthoc_raised_failures=raised_failure_profile(PUBLIC / "holdout.npz", PUBLIC / "predictions.npz"),
                    stages={s: dict(seal=read(folder / "seal.json"), seal_sha256=digest(folder / "seal.json"),
                                    resources=res) for s, (folder, res) in stages.items()})
    path = PUBLIC / "manifest.json"
    if not path.exists():
        write(path, manifest)
    require(read(path) == manifest, "preserving different publication")
    verify_public()
    if desktop:
        copy_verified(MOVIE, desktop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--replay-weights", action="store_true")
    parser.add_argument("--desktop", type=Path)
    args = parser.parse_args()
    if args.check or args.replay_weights:
        result = verify_public()
        if args.replay_weights:
            replay_weights()
        print(json.dumps({"passing_seeds": result["passing_seeds"], "holdout_packets": result["heldout_arrivals"],
                          "posthoc_geometric_views": compare_geometric_views(PUBLIC / "holdout.npz")}))
    else:
        publish(args.desktop)
