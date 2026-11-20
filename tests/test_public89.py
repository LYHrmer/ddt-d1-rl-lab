"""Recompute the registered D1 curriculum gates with the standard library and public files."""
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/experiments89/uphill_curriculum.json"
RESULT = ROOT / "reports/uphill_curriculum89.json"


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, abs_tol=2e-14, rel_tol=0))


def test_all_scoring_endpoints_and_untuned_distance_marks_reproduce_evaluation():
    plan, result = load(PROTOCOL), load(RESULT)
    assert result["status"] == "COMPLETE" and result["protocol_sha256"] == digest(PROTOCOL)
    course = load(ROOT / plan["parent_protocol"])["course"]
    assert digest(ROOT / plan["parent_protocol"]) == plan["parent_protocol_sha256"]
    train, ev = plan["training"], plan["evaluation"]
    sets = {(arm, seed, terrain, reset) for arm in train["arms"]
            for seed, matched in zip(train["seeds"], ev["reset_seeds"], strict=True) for terrain in ev["terrains"]
            for reset in (ev["reset_seeds"] if terrain == "up" else [matched])}
    cases = {(r["arm"], r["training_seed"], r["terrain"], r["seed"]): r for r in result["evaluation"]["cases"]}
    assert set(cases) == sets and len(sets) == len(result["evaluation"]["cases"]) == ev["max_episodes"]
    metadata = result["measurements"]
    path = ROOT / "reports/uphill_curriculum89_eval.csv.gz"
    assert metadata["file"] == path.name and metadata["sha256"] == digest(path)
    assert metadata["bytes"] == path.stat().st_size
    groups = defaultdict(list)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["arm"], int(row["training_seed"]), row["terrain"], int(row["reset_seed"])
            groups[key].append(row)
    assert set(groups) == sets
    assert sum(map(len, groups.values())) == metadata["rows"] == result["counts"]["evaluation_transitions"]
    a, b = ev["speed_interval_end_ticks"]
    warm, horizon = course["stand_ticks"], course["ticks"]
    target, mark = course["command_forward_m_s"], ev["course_mark_m_past_toe"]
    measured = {}
    for key, rows in groups.items():
        record = cases[key]
        n = len(rows)
        assert n == record["ticks"] and 0 < n <= horizon
        assert [int(r["end_tick"]) for r in rows] == list(range(1, n+1))
        speed = [float(r["body_vx_m_s"]) for r in rows]
        assert all(math.isfinite(v) for v in speed)
        assert all(math.isclose(float(r["command_vx_m_s"]), target if i >= warm else 0., abs_tol=1e-7)
                   for i, r in enumerate(rows))
        vx = mean(speed[a:b]) if n >= b else None
        squared = [(v - target)**2 for v in speed[warm:]]
        squared.extend([target**2] * (horizon-warm-len(squared)))
        rmse = math.sqrt(mean(squared))
        close(vx, record["score_vx_m_s"])
        close(rmse, record["fixed_speed_rmse_m_s"])
        distances = [float(r["loaded_front_past_toe_m"]) for r in rows if r["loaded_front_past_toe_m"]]
        assert all(math.isfinite(v) for v in distances)
        progress = max(distances, default=None)
        reached = progress is not None and progress >= mark
        assert progress == record["max_loaded_front_contact_past_toe_m"] and reached == record["reached_mark"]
        assert all(r[k] in ("0", "1") for r in rows for k in ("fallen", "illegal_contact"))
        fallen = any(r["fallen"] == "1" for r in rows)
        illegal = any(r["illegal_contact"] == "1" for r in rows)
        clean = n == horizon and not (fallen or illegal)
        assert (fallen, illegal, clean) == (record["fallen"], record["illegal_contact"], record["course_completed"])
        masks = [int(r["wheel_slope_mask"]) for r in rows]
        assert all(0 <= m < 16 and (key[2] == "up" or m == 0) for m in masks)
        assert sum(m == 15 for m in masks) == record["four_wheel_slope_ticks"]
        assert sum(bool(m & 12) for m in masks) == record["rear_wheel_slope_ticks"]
        assert sum(int(r["wheel_saturation_count"]) > 0 for r in rows) == record["wheel_saturated_ticks"]
        measured[key] = dict(vx=vx, rmse=rmse, fallen=fallen, illegal=illegal, clean=clean, reached=reached)

    verdict = result["evaluation"]["verdict"]
    passes = []
    assert [r["training_seed"] for r in verdict["uphill_paired_train_seed_means"]] == train["seeds"]
    for data in verdict["uphill_paired_train_seed_means"]:
        seed = data["training_seed"]
        pairs = [[measured[arm, seed, "up", reset] for reset in ev["reset_seeds"]] for arm in train["arms"]]
        means = [mean(r["vx"] for r in group) if all(r["vx"] is not None for r in group) else None for group in pairs]
        delta = means[1]-means[0] if None not in means else None
        clean = all(r["clean"] for group in pairs for r in group)
        passes.append(clean and delta is not None and delta >= ev["min_paired_uphill_delta_m_s"])
        for value, name in zip([*means, delta], ("original_vx_m_s", "mixed_vx_m_s", "paired_delta_vx_m_s"), strict=True):
            close(value, data[name])
        assert data["passes"] == passes[-1] and clean == data["all_six_clean"]
    assert all(passes) == verdict["uphill_speed_pass"]
    marks = {arm: sum(r["clean"] and r["reached"] for (role, _, terrain, _), r in measured.items()
                      if terrain == "up" and role == arm) for arm in train["arms"]}
    assert marks == verdict["uphill_clean_marks"]
    assert (marks["mixed"] == len(train["seeds"]) * len(ev["reset_seeds"])) == verdict["uphill_stable_pass"]
    kept = []
    for terrain in ("flat", "down"):
        record = verdict["pace_retention"][terrain]
        deltas, full = [], []
        for seed, reset, row in zip(train["seeds"], ev["reset_seeds"], record["paired"], strict=True):
            left, right = (measured[arm, seed, terrain, reset] for arm in train["arms"])
            deltas.append(right["rmse"]-left["rmse"])
            full.append(left["clean"] and right["clean"])
            assert (seed, reset, full[-1]) == (row["training_seed"], row["seed"], row["both_clean"])
            close(left["rmse"], row["original_rmse_m_s"])
            close(right["rmse"], row["mixed_rmse_m_s"])
            close(deltas[-1], row["delta_rmse_m_s"])
        close(mean(deltas), record["mean_delta_rmse_m_s"])
        kept.append(all(full) and mean(deltas) <= ev["max_mean_retention_rmse_increase_m_s"])
        assert kept[-1] == record["passes"]
    assert all(kept) == verdict["retention_pass"]
    assert result["passes_all_gates"] == (result["coverage_pass"] and all(passes)
                                           and verdict["uphill_stable_pass"] and all(kept))
    assert sum(r["clean"] for r in measured.values()) == verdict["total_clean_full_horizon"]
    assert sum(r["fallen"] for r in measured.values()) == verdict["total_falls"]
    assert sum(r["illegal"] for r in measured.values()) == verdict["total_illegal_contacts"]


def test_training_exposure_from_episode_histograms_and_total_native_budget():
    plan, result = load(PROTOCOL), load(RESULT)
    train = plan["training"]
    course = load(ROOT / plan["parent_protocol"])["course"]
    data = result["training"]["models"]
    by_key = {(m["arm"], m["train_seed"]): m for m in data}
    assert set(by_key) == {(arm, s) for arm in train["arms"] for s in train["seeds"]}
    assert len(data) == len(by_key) == train["max_models"]
    passes = []
    by_seed = {p["train_seed"]: p for p in result["coverage_pairs"]}
    assert set(by_seed) == set(train["seeds"]) and len(by_seed) == len(result["coverage_pairs"])
    for parent, seed in zip(train["parent_seeds"], train["seeds"], strict=True):
        pair = by_seed[seed]
        left, right = (by_key[a, seed] for a in train["arms"])
        assert all(left[k] == right[k] for k in ("parent_policy_sha256", "inheritance", "initial_probe_actions_sha256",
                                                  "pre_rollout_torch_rng_sha256"))
        inherited = left["inheritance"]
        assert inherited["parameter_tensors"] == inherited["adam_parameters"] == len(inherited["inherited_adam_steps"])
        assert min(inherited["inherited_adam_steps"].values()) > 0
        fractions = []
        for arm in train["arms"]:
            model = by_key[arm, seed]
            assert model["parent_seed"] == parent and model["parent_policy_sha256"] == plan["parent_policy_sha256"][str(parent)]
            assert model["optimizer_steps"] > 0 and model["leg_rows_changed"] == 12 and model["wheel_rows_changed"] == 4
            episodes = model["episodes"]
            anchors = model["stand_endpoint_source_arrays_sha256"]
            assert set(anchors) == {"episode_tick", "slope_start", "start_x_m", "qpos", "body_vx_m_s", "command_vx_m_s"}
            assert all(len(v) == 64 for v in anchors.values())
            assert sum(ep["ticks"] for ep in episodes) == train["transitions_per_model"] == model["counts"]["policy_transitions"]
            assert all(0 < ep["ticks"] <= course["ticks"] and ep["moving_ticks"] == max(0, ep["ticks"] - course["stand_ticks"])
                       for ep in episodes)
            all4 = both = denominator = up_ticks = past = up_index = prefix = 0
            before_shift = True
            for i, ep in enumerate(episodes):
                assert ep["terrain"] == course["terrains"][(seed+i) % len(course["terrains"])]
                assert ep["slope_start"] == (arm == "mixed" and ep["terrain"] == "up" and up_index % 2 == 1)
                if ep["slope_start"]:
                    assert ep["start_x_m"] == plan["curriculum"]["slope_start_x_m"]
                    bound = ep["stand_endpoint"]
                    if ep["ticks"] < course["stand_ticks"]:
                        assert bound is None
                    else:
                        assert bound["end_tick"] == course["stand_ticks"] and bound["time_s"] == course["stand_ticks"] / course["policy_hz"]
                        assert bound["command_m_s"] == 0. and math.isfinite(bound["body_vx_m_s"]) and math.isfinite(bound["base_x_m"])
                        assert bound["body_vx_m_s"] < 0. and bound["base_x_m"] < ep["start_x_m"]
                before_shift = before_shift and not ep["slope_start"]
                if before_shift:
                    prefix += ep["ticks"]
                if ep["terrain"] == "up":
                    histogram = ep["moving_wheel_mask_counts"]
                    assert len(histogram) == 16 and all(isinstance(v, int) and v >= 0 for v in histogram)
                    assert sum(histogram) == ep["moving_ticks"]
                    all4 += histogram[15]
                    both += sum(histogram[k] for k in range(16) if k & 3 and k & 12)
                    denominator += sum(histogram)
                    up_ticks += ep["ticks"]
                    past += ep["moving_base_past_toe"]
                    up_index += 1
            if arm == "mixed":
                assert prefix == pair["identical_prefix_ticks"]
            observed = pair[arm]
            assert (up_ticks, denominator, all4, both, past) == (
                observed["uphill_ticks"], observed["uphill_moving_ticks"], observed["all_four_on_slope_moving"],
                observed["both_axles_on_slope_moving"], observed["uphill_moving_base_past_toe"])
            assert sum(bool(e["terminated"]) for e in episodes) == observed["terminated_episodes"]
            fractions.append(all4/denominator)
            close(fractions[-1], observed["all_four_fraction"])
        delta = fractions[1]-fractions[0]
        close(delta, pair["fraction_increase"])
        passes.append(fractions[1] >= plan["coverage"]["minimum_mixed_fraction"]
                      and delta >= plan["coverage"]["minimum_paired_fraction_increase"])
        assert pair["coverage_pass"] == passes[-1]
    assert all(passes) == result["coverage_pass"]
    count = result["counts"]
    assert count["training_models"] == len(data) and count["evaluation_episodes"] == plan["evaluation"]["max_episodes"]
    smoke = result["smoke"]["cases"]
    assert count["smoke_episodes"] == len(smoke) == plan["preflight"]["max_episodes"]
    assert count["smoke_transitions"] == sum(r["ticks"] for r in smoke) <= plan["preflight"]["max_transitions"]
    assert count["training_transitions"] == sum(m["counts"]["policy_transitions"] for m in data) == train["max_transitions"]
    assert count["optimizer_steps"] == sum(m["optimizer_steps"] for m in data)
    assert count["policy_transitions"] == sum(count[k] for k in ("smoke_transitions", "training_transitions", "evaluation_transitions"))
    assert count["policy_transitions"] <= plan["max_new_policy_transitions"]
    assert count["native_steps"] == 10*count["policy_transitions"] <= plan["max_native_steps"]
    assert count["pd_updates"] == 5*count["policy_transitions"]


def test_published_video_matches_preregistered_pair_and_receipt():
    plan, result = load(PROTOCOL), load(RESULT)
    video, path = result["video"], ROOT / "docs/assets/uphill89.mp4"
    assert video["status"] == "COMPLETE" and video["sha256"] == digest(path)
    assert 0 < video["bytes"] == path.stat().st_size <= 5_000_000
    assert video["physics_steps"] == video["policy_calls"] == 0 and video["unmodified_policy_output"]
    assert video["fps"] == 25 and video["frames_written"] == 150 and float(video["stream"]["duration"]) == 6.
    seed, reset = plan["training"]["seeds"][0], plan["evaluation"]["reset_seeds"][0]
    assert (video["training_seed"], video["reset_seed"], video["terrain"]) == (seed, reset, "up")
    by_case = {r["case"]: r for r in result["evaluation"]["cases"]}
    for arm in plan["training"]["arms"]:
        entry = video["source_cases"][arm]
        name = f"{arm}_{seed}_up_{reset}"
        assert entry["case"] == name and entry["trace.npz"] == by_case[name]["trace_sha256"]


def test_optional_readbacks_use_the_declared_models_and_public_resource_bounds():
    plan, result = load(PROTOCOL), load(RESULT)
    for guard in result["resources"].values():
        assert guard["startup_effective_bytes"] >= plan["resources"]["minimum_effective_gib_at_start"] << 30
        assert guard["minimum_effective_bytes"] >= plan["resources"]["minimum_effective_gib_during_run"] << 30
        assert guard["owned_cgroup_children_empty"] and len(guard["cpu_affinity"]) == 1
        assert all(len(guard[k]) == 64 for k in ("supervisor_sha256", "report_sha256", "samples_sha256"))
    sources = result["source_receipts_sha256"]
    assert ("sensor" in sources) == ("training_contacts" in sources)
    if "sensor" not in sources:
        return
    sensor, contacts = result["sensor_readback"], result["training_contacts_readback"]
    assert "sensor" in result["resources"]
    for item, name, source in ((sensor, "sensor", "sensor89.py"), (contacts, "training_contacts", "training_contacts89.py")):
        assert item["status"] == "COMPLETE" and item["native_steps"] == item["optimizer_steps"] == 0
        assert item["protocol_sha256"] == result["protocol_sha256"] and item["training_report_sha256"] == sources["training"]
        assert item["report_sha256"] == sources[name] and item["source_sha256"] == digest(ROOT / "tools89" / source)
    models = {m["model"]: m for m in result["training"]["models"]}
    old = load(ROOT / "reports/uphill_diagnosis88.json")
    assert sensor["probe88_report_sha256"] == old["probe_report_sha256"]
    old_cases = {c["seed"]: c for c in old["cases"] if c["arm"] == "original"}
    assert set(old_cases) == {c["seed"] for c in sensor["source_cases"]}
    for ref in sensor["source_cases"]:
        assert ref["trace_sha256"] == old_cases[ref["seed"]]["trace_sha256"]
        assert ref["archived_range_values_matched"] == 4800 and 0 <= ref["window_decisions_with_distance_change"] <= 100
        assert 0 <= ref["window_decisions_with_validity_change"] <= 100
    assert [s["training_seed"] for s in sensor["training_seeds"]] == plan["training"]["seeds"]
    for s in sensor["training_seeds"]:
        for arm in plan["training"]["arms"]:
            assert s["model_sha256"][arm] == models[f"{arm}_{s['training_seed']}"]["policy_sha256"]
        comps = s["comparisons"]
        for stats in comps.values():
            close(mean(stats["per_wheel_signed_mean_rad_s"]), stats["signed_mean_rad_s"])
            close(mean(stats["reference_signed_means_rad_s"]), stats["signed_mean_rad_s"])
            assert abs(stats["signed_mean_rad_s"]) <= stats["mean_abs_rad_s"] + 2e-14
        close(comps["mixed_up_minus_flat"]["signed_mean_rad_s"] - comps["original_up_minus_flat"]["signed_mean_rad_s"],
              comps["curriculum_change_in_up_minus_flat"]["signed_mean_rad_s"])
    assert {m["model"] for m in contacts["models"]} == set(models)
    flagged = 0
    for rec in contacts["models"]:
        model = models[rec["model"]]
        assert rec["policy_sha256"] == model["policy_sha256"] and rec["trace_sha256"] == model["train_trace_sha256"]
        assert rec["train_report_sha256"] == model["report_sha256"] and rec["archived_ticks"] == model["counts"]["policy_transitions"]
        flagged += rec["illegal_contact_intervals"]
    assert len(contacts["events"]) == contacts["observed_flagged_intervals"] == flagged
    confirmed = prior = 0
    course = load(ROOT / plan["parent_protocol"])["course"]
    for evt in contacts["events"]:
        model, last = models[evt["model"]], evt["history"][-1]
        ep = model["episodes"][evt["episode_index"]]
        assert model["train_seed"] == evt["training_seed"] and ep["terrain"] == evt["terrain"] and ep["slope_start"] == evt["slope_start"]
        assert ep["terminated"] and ep["ticks"] == last["episode_tick"]
        assert sum(e["ticks"] for e in model["episodes"][:evt["episode_index"]+1])-1 == evt["terminal_rollout_row"] == last["rollout_row"]
        assert last["archived_interval_illegal"] and evt["end_time_s"] == last["time_s"]
        for frame in evt["history"]:
            position = frame["base_world_xyz_m"]
            assert len(position) == 3 and all(math.isfinite(x) for x in position)
            assert math.isfinite(frame["body_vx_m_s"])
            command = course["command_forward_m_s"] if frame["episode_tick"] > course["stand_ticks"] else 0.
            close(frame["command_m_s"], command)
            assert all(len(c["contact_world_xyz_m"]) == 3 and all(math.isfinite(x) for x in c["contact_world_xyz_m"])
                       for c in frame["nonwheel_support_contacts"])
            bodies = sorted({c["body"] for c in frame["nonwheel_support_contacts"] if c["cold_normal_force_n"] > evt["normal_force_threshold_n"]})
            assert bool(bodies) == frame["cold_illegal_at_endpoint"]
        assert bodies == evt["confirmed_bodies_at_endpoint"] and bool(bodies) == evt["threshold_at_endpoint_reconstructed"]
        confirmed += bool(bodies)
        prior += len(evt["history"]) - 1
    assert confirmed == contacts["cold_positive_endpoints"] and prior == contacts["checked_earlier_endpoints"]
