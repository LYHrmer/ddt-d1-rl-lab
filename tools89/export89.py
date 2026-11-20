"""Package measured D1 endpoints, training contact counts and resource receipts without integration."""
import argparse
import csv
import gzip
import hashlib
import io
from pathlib import Path

import numpy as np

from tools87.readback87 import check
from tools88.export88 import resources
from tools89.common89 import PLAN, ROOT, read, registered_plan, sha, write
from tools89.eval89 import checked_models, matrix


def guarded_resources(guard):
    data = resources(guard)
    check(data["supervisor_sha256"] == sha(ROOT.parent / "wheel-legged-control-lab/scripts/run_pressure83.py"),
          guard.name + " supervisor source")
    return {**data, "report_sha256": sha(guard / "report.json"), "samples_sha256": sha(guard / "resources.jsonl")}


def training_evidence(directory, header, plan, course):
    model_paths = checked_models(directory, plan)
    output = []
    for model in header["models"]:
        name, seed, arm = model["model"], model["train_seed"], model["arm"]
        folder = directory / name
        report = read(folder / "report.json")
        path = folder / "train_trace.npz"
        fields = ("terrain", "episode_tick", "loaded_wheels_past_toe", "command_vx_m_s", "slope_start",
                  "base_origin_past_toe", "terminated", "truncated", "start_x_m", "qpos", "body_vx_m_s")
        with np.load(path, allow_pickle=False) as archive:
            a = {k: archive[k] for k in fields}
        check(all(hashlib.sha256(v.tobytes()).hexdigest() == report["trace"]["array_sha256"][k] for k, v in a.items()),
              name + " compressed training source")
        n = len(a["terrain"])
        check(n == plan["training"]["transitions_per_model"] == report["counts"]["policy_transitions"]
              and report["counts"]["pd_updates"] == 5*n and report["counts"]["native_steps"] == 10*n
              and report["optimizer_steps"] == model["optimizer_steps"] > 0
              and a["loaded_wheels_past_toe"].shape == (n, 4) and a["loaded_wheels_past_toe"].dtype == bool,
              name + " budget and contacts")
        mask = a["loaded_wheels_past_toe"] @ np.array([1, 2, 4, 8])
        starts = np.flatnonzero(a["episode_tick"] == 1)
        check(len(starts) > 0 and starts[0] == 0 and np.array_equal(starts[1:],
              1 + np.flatnonzero((a["terminated"] | a["truncated"])[:-1])), name + " recorded resets")
        episodes = []
        for i, j in zip(starts, [*starts[1:], n], strict=True):
            t = a["episode_tick"][i:j]
            terrain = course["terrains"][int(a["terrain"][i])]
            moving = a["command_vx_m_s"][i:j] > 0
            check(np.array_equal(t, np.arange(1, j-i+1)) and np.array_equal(moving, t > course["stand_ticks"])
                  and np.all(a["slope_start"][i:j] == a["slope_start"][i])
                  and np.all(a["terrain"][i:j] == a["terrain"][i]), name + " within-episode labels")
            row = {"terrain": terrain, "ticks": int(j-i), "moving_ticks": int(moving.sum()),
                   "slope_start": bool(a["slope_start"][i]),
                   "terminated": bool(a["terminated"][j-1]), "truncated": bool(a["truncated"][j-1])}
            if row["slope_start"]:
                row["start_x_m"] = float(a["start_x_m"][i])
                end = i + course["stand_ticks"] - 1
                row["stand_endpoint"] = None
                if end < j:
                    tick = int(a["episode_tick"][end])
                    row["stand_endpoint"] = {"end_tick": tick, "time_s": tick / course["policy_hz"],
                        "command_m_s": float(a["command_vx_m_s"][end]), "base_x_m": float(a["qpos"][end, 0]),
                        "body_vx_m_s": float(a["body_vx_m_s"][end])}
            if terrain == "up":
                row["moving_wheel_mask_counts"] = np.bincount(mask[i:j][moving], minlength=16).tolist()
                row["moving_base_past_toe"] = int((a["base_origin_past_toe"][i:j] & moving).sum())
            episodes.append(row)
        interesting = ("parent_policy_sha256", "inheritance", "initial_probe_actions_sha256", "pre_rollout_torch_rng_sha256",
                       "range_projection_delta_l2", "leg_rows_changed", "wheel_rows_changed", "counts",
                       "reload_probe_actions_bitwise_equal", "max_probed_change_from_parent", "max_probed_sensor_action_change")
        check(model_paths[arm, seed][1] == model["policy_sha256"] and report["leg_rows_changed"] == 12
              and report["wheel_rows_changed"] == 4 and report["reload_probe_actions_bitwise_equal"], name + " PPO")
        stand_hashes = {k: report["trace"]["array_sha256"][k] for k in
                        ("episode_tick", "slope_start", "start_x_m", "qpos", "body_vx_m_s", "command_vx_m_s")}
        output.append({**model, **{k: report[k] for k in interesting}, "episodes": episodes,
                       "stand_endpoint_source_arrays_sha256": stand_hashes,
                       "train_trace_sha256": sha(path)})
    return {"models": output, "wheel_mask_bits_FL_FR_RL_RR": [1, 2, 4, 8]}


def evaluation_csv(path, directory, verified, original, plan):
    index = {c["case"]: c for c in original["cases"]}
    check(len(index) == len(verified) == len(matrix(plan)) == plan["evaluation"]["max_episodes"], "evaluation cases")
    expected = {f"{arm}_{seed}_{terrain}_{reset}" for arm, seed, terrain, reset in matrix(plan)}
    check(set(index) == expected == {r["case"] for r in verified}, "evaluation matrix")
    count = 0
    with (
        path.open("xb") as binary,
        gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8", newline="") as stream,
    ):
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("arm", "training_seed", "terrain", "reset_seed", "end_tick", "command_vx_m_s", "body_vx_m_s",
                         "loaded_front_past_toe_m", "fallen", "illegal_contact", "wheel_saturation_count", "wheel_slope_mask"))
        for row in verified:
            name = row["case"]
            source = directory / name / "trace.npz"
            case = index[name]
            check(read(source.with_name("metrics.json")) == case and row["trace_sha256"] == case["trace_sha256"] == sha(source)
                  and row["policy_sha256"] == case["policy_sha256"] and case["status"] == "COMPLETE", name + " source")
            with np.load(source, allow_pickle=False) as z:
                fields = ("command_vx_m_s", "body_vx_m_s", "front_contact_x_m", "fallen", "illegal_body_contact",
                          "wheel_saturation_count", "wheel_on_slope")
                cmd, vx, xs, falls, illegal, saturation, contact = (z[k][1:] for k in fields)
            check(len(vx) == row["ticks"] == case["ticks"] and np.isfinite(cmd).all() and np.isfinite(vx).all()
                  and not np.isinf(xs).any(), name + " observation boundaries")
            bits = contact @ np.array([1, 2, 4, 8])
            for i in range(len(vx)):
                x = float(xs[i] - plan["curriculum"]["toe_x_m"]) if np.isfinite(xs[i]) else ""
                writer.writerow((row["arm"], row["training_seed"], row["terrain"], row["seed"], i+1, float(cmd[i]),
                                 float(vx[i]), x, int(falls[i]), int(illegal[i]), int(saturation[i]), int(bits[i])))
            count += len(vx)
    return {"file": path.name, "rows": count, "bytes": path.stat().st_size, "sha256": sha(path)}


def archived_attempt(root, plan_sha):
    old = root / "preflight_01/report.json"
    if not old.exists():
        return []
    r = read(old)
    guard = root / "guard_preflight_01"
    g = read(guard / "report.json")
    # This specific failure occurred inside the zero-integration audit, before the episode loop.
    check(r["protocol_sha256"] == plan_sha and r["status"] == "INCOMPLETE" and not r["cases"] and not r["models"]
          and r["error"] == "ValueError: reset geometry differs from native terrain"
          and "zero_preflight(" in (guard / "stderr.log").read_text()
          and not list(old.parent.glob("*/trace.npz")) and not list(old.parent.glob("*/partial_trace.npz"))
          and g["status"] == "INCOMPLETE" and g["child_reaped"] and g["owned_cgroup_children_empty"],
          "unaccounted previous attempt")
    return [{"stage": "preflight", "status": r["status"], "error": r["error"], "native_steps": 0,
             "report_sha256": sha(old), "guard_sha256": sha(guard / "report.json"), "sources": r["sources"]}]


def add_observational_readbacks(result, root):
    paths = {"sensor": root / "sensor_01.json", "training_contacts": root / "training_contacts_01.json"}
    if not all(p.exists() for p in paths.values()):
        return
    audits = {key: read(p) for key, p in paths.items()}
    for key, audit in audits.items():
        result["source_receipts_sha256"][key] = sha(paths[key])
        if audit["status"] != "COMPLETE":
            result[key + "_readback"] = {"status": audit["status"], "error": audit.get("error")}
    if any(r["status"] != "COMPLETE" for r in audits.values()):
        return
    guard = root / "guard_sensor_01"
    result["resources"]["sensor"] = guarded_resources(guard)
    for key, audit in audits.items():
        module = "sensor89.py" if key == "sensor" else "training_contacts89.py"
        check(audit["protocol_sha256"] == result["protocol_sha256"] and audit["source_sha256"] == sha(ROOT / "tools89" / module)
              and audit["training_report_sha256"] == result["source_receipts_sha256"]["training"]
              and audit["optimizer_steps"] == audit["native_steps"] == 0, key + " observational audit")
    private88 = ROOT / "evidence/uphill88/probe_01"
    sensor = audits["sensor"]
    check(sensor["probe88_report_sha256"] == sha(private88 / "report.json"), "sensor source rollouts")
    for case in sensor["source_cases"]:
        folder = private88 / case["case"]
        check(sha(folder / "trace.npz") == case["trace_sha256"] and sha(folder / "metrics.json") == case["metrics_sha256"],
              case["case"] + " recorded sensor source")
    result["sensor_readback"] = {**sensor, "report_sha256": sha(paths["sensor"])}
    contacts = audits["training_contacts"]
    models = {m["model"]: m for m in result["training"]["models"]}
    check({m["model"] for m in contacts["models"]} == set(models) and len(contacts["models"]) == len(models),
          "training contact audit models")
    for entry in contacts["models"]:
        model = models[entry["model"]]
        check(entry["trace_sha256"] == model["train_trace_sha256"] and entry["train_report_sha256"] == model["report_sha256"]
              and entry["policy_sha256"] == model["policy_sha256"] and entry["archived_ticks"] == model["counts"]["policy_transitions"],
              entry["model"] + " training contact source")
    events = []
    for event in contacts["events"]:
        history = []
        for frame in event["history"]:
            summary = {k: frame[k] for k in ("rollout_row", "episode_tick", "time_s", "roll_deg", "pitch_deg",
                       "base_world_xyz_m", "body_vx_m_s", "command_m_s", "archived_interval_illegal",
                       "archived_fallen", "cold_illegal_at_endpoint")}
            summary["nonwheel_support_contacts"] = [
                {k: c[k] for k in ("body", "cold_normal_force_n", "above_threshold", "contact_world_xyz_m")}
                for c in frame["nonwheel_support_contacts"]]
            history.append(summary)
        events.append({**event, "history": history})
    result["training_contacts_readback"] = {**contacts, "events": events, "report_sha256": sha(paths["training_contacts"])}


def export(root, output):
    plan = registered_plan()
    plan_sha = sha(PLAN)
    parent = ROOT / plan["parent_protocol"]
    course = read(parent)["course"]
    sources = {s: root / f"{s}_01/report.json" for s in ("admission", "smoke", "training", "evaluate")}
    sources.update(readback=root / "readback_01.json", video=root / "uphill89.json")
    stages = {s: read(p) for s, p in sources.items()}
    for s, receipt in stages.items():
        check(receipt["status"] == "COMPLETE" and receipt["protocol_sha256"] == plan_sha, s + " unqualified input")
    guards = {s: guarded_resources(root / f"guard_{s}_01") for s in sources}
    originals = archived_attempt(root, plan_sha)
    roadcheck = root / "roadcheck_01.json"
    if roadcheck.exists():
        check(read(roadcheck)["native_steps"] == 0, "extra physics in the road query snapshot")
        guards["roadcheck"] = guarded_resources(root / "guard_roadcheck_01")
        sources["roadcheck"] = roadcheck
    geom = stages["admission"]["zero_preflight"]
    rays = geom["native_road_probes"]
    check(geom["native_steps"] == geom["policy_transitions"] == geom["pd_updates"] == 0
          and rays["max_interior_error_m"] <= rays["tolerance_m"], "zero-step geometry")
    admission = {k: v for k, v in geom.items() if k != "native_road_probes"}
    admission["native_road_probes"] = {k: v for k, v in rays.items() if k != "unique_reset_states"}
    smoke = stages["smoke"]
    sm_cases = smoke["cases"]
    check(smoke["preflight_pass"] and len(sm_cases) == plan["preflight"]["max_episodes"]
          and {(c["arm"], c["seed"]) for c in sm_cases} ==
          {(a, s) for a in plan["training"]["arms"] for s in plan["preflight"]["reset_seeds"]}, "smoke matrix")
    for c in sm_cases:
        case_dir = root / "smoke_01" / f"{c['arm']}_{c['seed']}"
        check(c["passed"] and c == read(case_dir / "metrics.json") and c["trace_sha256"] == sha(case_dir / "trace.npz"),
              f"{case_dir.name} smoke readback")
    training, evaluation, replay = (stages[s] for s in ("training", "evaluate", "readback"))
    check(replay["evaluation_report_sha256"] == sha(sources["evaluate"])
          and evaluation["training_report_sha256"] == sha(sources["training"])
          and replay["new_native_steps_in_readback"] == 0 and replay["counts"] == evaluation["counts"], "readback chain")
    verified_training = training_evidence(root / "training_01", training, plan, course)
    models = verified_training["models"]
    vid, video_path = stages["video"], root / "uphill89.mp4"
    check(vid["training_seed"] == plan["training"]["seeds"][0] and vid["reset_seed"] == plan["evaluation"]["reset_seeds"][0]
          and vid["terrain"] == "up" and vid["unmodified_policy_output"] and vid["policy_calls"] == vid["physics_steps"] == 0
          and sha(video_path) == vid["sha256"] and 0 < vid["bytes"] == video_path.stat().st_size <= 5_000_000
          and vid["frames_written"] == int(vid["stream"]["nb_read_frames"]) == 150 and vid["fps"] == 25
          and float(vid["stream"]["duration"]) == 6., "registered video")
    for arm in plan["training"]["arms"]:
        entry = vid["source_cases"][arm]
        name = f"{arm}_{vid['training_seed']}_up_{vid['reset_seed']}"
        check(entry["case"] == name and all(entry[k] == sha(root / "evaluate_01" / name / k)
              for k in ("trace.npz", "metrics.json")), "video source " + arm)

    counts = {"smoke_episodes": len(sm_cases), "training_models": len(models), "evaluation_episodes": len(replay["cases"]),
              "smoke_transitions": sum(c["ticks"] for c in sm_cases),
              "training_transitions": sum(m["counts"]["policy_transitions"] for m in models),
              "evaluation_transitions": replay["counts"]["policy_transitions"],
              "optimizer_steps": sum(m["optimizer_steps"] for m in models)}
    counts["policy_transitions"] = sum(counts[k] for k in ("smoke_transitions", "training_transitions", "evaluation_transitions"))
    for k in ("pd_updates", "native_steps"):
        counts[k] = sum(c[k] for c in sm_cases) + sum(m["counts"][k] for m in models) + replay["counts"][k]
    check(counts["training_transitions"] == plan["training"]["max_transitions"]
          and counts["smoke_transitions"] <= plan["preflight"]["max_transitions"]
          and counts["evaluation_transitions"] <= plan["evaluation"]["max_transitions"]
          and counts["policy_transitions"] <= plan["max_new_policy_transitions"]
          and counts["native_steps"] == 10*counts["policy_transitions"] <= plan["max_native_steps"]
          and counts["pd_updates"] == 5*counts["policy_transitions"], "registered resource budget")
    output.mkdir(exist_ok=False)
    measurement = evaluation_csv(output / "uphill_curriculum89_eval.csv.gz", root / "evaluate_01", replay["cases"], evaluation, plan)
    check(measurement["rows"] == counts["evaluation_transitions"], "public sample count")
    result = {"status": "COMPLETE", "protocol_sha256": plan_sha, "counts": counts, "prior_attempts": originals,
              "admission": admission, "smoke": {"cases": sm_cases, "preflight_pass": smoke["preflight_pass"]},
              "training": verified_training, "coverage_pairs": training["coverage_pairs"], "coverage_pass": training["coverage_pass"],
              "evaluation": {"cases": replay["cases"], "verdict": replay["verdict"]}, "measurements": measurement,
              "video": vid, "resources": guards, "source_receipts_sha256": {k: sha(p) for k, p in sources.items()}}
    result["passes_all_gates"] = bool(training["coverage_pass"] and all(replay["verdict"][k] for k in
                                      ("uphill_speed_pass", "uphill_stable_pass", "retention_pass")))
    add_observational_readbacks(result, root)
    write(output / "uphill_curriculum89.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.output)
