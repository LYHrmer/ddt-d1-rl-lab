"""Audit the public cold-start results using only published numbers and NumPy."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/cold_start94.json"
CURVES = ROOT / "reports/cold_start94_curves.npz"
VIDEO = ROOT / "docs/assets/cold_start94.mp4"
SEEDS = (94011, 94023, 94037)
CALIBRATION = ((948101, 948102, 948103), (948201, 948202, 948203))
EVALUATION = (949001, 949002, 949003)
OFFSETS = (-20, -10, 10, 20)
EVENTS = ("fallen", "illegal", "terminated", "truncated")
LIMITS = dict(native_decisions=9000, servo_updates=45000, physics_substeps=90000,
              world_updates=600, policy_updates=24, ppo_updates=0)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matches(actual, expected):
    if isinstance(expected, dict):
        assert set(expected) <= set(actual)
        for key, value in expected.items():
            matches(actual[key], value)
    elif expected is None or isinstance(expected, (bool, np.bool_)):
        assert actual is (None if expected is None else bool(expected))
    elif isinstance(expected, (float, np.floating)):
        assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), (actual, expected)
    else:
        assert actual == expected


def branch(curves, prefix, source, offset):
    sign = "m" if offset < 0 else "p"
    key = f"{prefix}_{source}_{sign}{abs(offset):02d}"
    x = curves[key].astype(np.float64)
    assert x.ndim == 1 and len(x) <= 51 and np.isfinite(x).all()
    if source == "native":
        assert len(x) >= 1 and x[0] == 1.10
    return x


def rollback(x):
    return max(0., 1000. * (x[0] - x[50])) if len(x) >= 51 else None


def short_return(x):
    return -math.fsum(abs(float(endpoint) - float(x[0])) for endpoint in x[1:33]) / 32.


def similar_slopes(a, b):
    return a*b > 0 and abs(a-b) <= max(abs(a), abs(b)) / 4


def clean(events, ticks, horizon):
    assert all(events[k] in (0, 1) for k in EVENTS)
    return ticks == horizon and all(events[k] == 0 for k in EVENTS)


def test_physical_differences_and_model_gates_from_public_endpoints():
    report = read(REPORT)
    assert digest(CURVES) == report["curves_sha256"]
    passed = []
    with np.load(CURVES, allow_pickle=False) as curves:
        for cell in report["calibration"]:
            prefix, score = cell["endpoints_prefix"], cell["score"]
            assert prefix == f"s{cell['seed']}_r{cell['round']}_c{cell['reset']}"
            truth, safety, real_mm, model_mm = {}, {}, {}, {}
            for k in OFFSETS:
                truth[k] = x = branch(curves, prefix, "native", k)
                y = branch(curves, prefix, "model", k)
                a, b = rollback(x), rollback(y)
                real_mm[k], model_mm[k] = a, b
                complete = safety[k] = clean(cell["native_events"][str(k)], len(x)-1, 50)
                matches(score["arms"][str(k)], dict(real_ticks=len(x)-1, model_ticks=len(y)-1,
                        complete_safe=complete, real_rollback_mm=a, model_rollback_mm=b,
                        model_error_mm=None if a is None or b is None else b-a))
            # A one-tick actuator delay makes the first native interval identical in all arms.
            for x in truth.values():
                n = min(2, len(truth[-20]), len(x))
                np.testing.assert_array_equal(x[:n], truth[-20][:n])
            derivative = score["model_derivative"]  # Published autodiff scalar at zero wheel offset.
            available = derivative is not None and math.isfinite(derivative) and all(
                real_mm[k] is not None and model_mm[k] is not None for k in OFFSETS)
            matches(score["available"], available)
            checks = {"all_arms_complete_safe": all(safety.values())}
            if available:
                slopes = []
                grades = []
                for radius in (10, 20):
                    slope = (short_return(truth[radius]) - short_return(truth[-radius])) / (radius * .02)
                    slopes.append(slope)
                    observed = real_mm[-radius] - real_mm[radius]
                    predicted = model_mm[-radius] - model_mm[radius]
                    error = predicted - observed
                    grade = dict(gradient_agrees=similar_slopes(derivative, slope), effect_sign=observed*predicted > 0,
                                 effect_within_3mm=abs(error) <= 3., beats_zero=abs(error) < abs(observed))
                    grades.append(grade)
                    matches(score["radii"][str(radius)], dict(native_gradient=slope, native_effect_mm=observed,
                            model_effect_mm=predicted, signed_effect_error_mm=error, **grade))
                checks.update(native_gradient_stable=similar_slopes(*slopes),
                              model_gradient_matches_native=all(g["gradient_agrees"] for g in grades),
                              all_arms_within_30mm=all(abs(real_mm[k] - model_mm[k]) <= 30. for k in OFFSETS))
                for name in ("effect_sign", "effect_within_3mm", "beats_zero"):
                    checks[name] = all(g[name] for g in grades)
            else:
                assert score["radii"] == {}
                checks["full_numeric_comparison"] = False
            matches(score["gates"], checks)
            assert set(score["gates"]) == set(checks)
            passed.append(all(checks.values()))
            matches(score["passed"], passed[-1])
    assert report["calibration_cells_passed"] == sum(passed)


def test_budget_training_sources_and_gated_rounds():
    report = read(REPORT)
    protocol = ROOT / "docs/experiments94/cold_start.json"
    assert read(protocol)["budget"] == LIMITS and digest(protocol) == report["protocol_sha256"]
    assert [s["seed"] for s in report["seeds"]] == list(SEEDS)
    assert len({s["initial"]["weights_sha256"] for s in report["seeds"] if s["initial"]}) == sum(
        s["initial"] is not None for s in report["seeds"])
    measured = Counter()
    total = Counter()
    cells = {(c["seed"], c["round"], c["reset"]):c for c in report["calibration"]}
    assert len(cells) == len(report["calibration"])
    examined = set()
    with np.load(CURVES, allow_pickle=False) as curves:
        for c in cells.values():
            measured[c["seed"]] += sum(len(branch(curves, c["endpoints_prefix"], "native", k))-1 for k in OFFSETS)
        for pair in report["evaluation"]:
            measured[pair["seed"]] += sum(len(curves[pair["endpoints_prefix"]+"_"+side])-1 for side in ("initial", "final"))
    for i, s in enumerate(report["seeds"]):
        seed, saved = s["seed"], s["totals"]
        assert set(saved) == set(LIMITS) and all(type(v) is int and v >= 0 for v in saved.values())
        total.update(saved)
        world = policy = 0
        rounds = s["rounds"]
        assert len(rounds) <= 2
        for j, group in enumerate(rounds):
            r = j + 1
            assert group["round"] == r
            predecessor = s["initial"] if j == 0 else rounds[0]["policy_checkpoint"]
            assert group["collector_weights_sha256"] == predecessor["weights_sha256"]
            fit = group["fit_episodes"]
            assert len(fit) <= 12
            assert [e["reset"] for e in fit] == [944000+1000*i+100*j+n for n in range(len(fit))]
            for ep in fit:
                assert 0 < ep["ticks"] <= 50
                matches(ep["full_clean_window"], clean(ep["native_events"], ep["ticks"], 50))
                measured[seed] += ep["ticks"]
            admitted = set()
            for reset in CALIBRATION[j]:
                key = (seed, r, reset)
                if key in cells:
                    examined.add(key)
                    if cells[key]["score"]["passed"]:
                        admitted.add(reset)
            if group.get("world_fit"):
                receipt = group["world_fit"]
                assert len(fit) == 12 and receipt["updates_completed"] == 100
                assert len(receipt["round_data"]) == r and receipt["rows_per_group_per_update"] == 32//r
                for batch, recorded_round in zip(receipt["round_data"], rounds[:r], strict=True):
                    episodes = recorded_round["fit_episodes"]
                    lengths = [e["ticks"] for e in episodes]
                    matches(batch, dict(seeds=[e["reset"] for e in episodes], episode_lengths=lengths,
                                        transitions=sum(lengths), windows=sum(max(0, n-7) for n in lengths)))
                if j == 1:
                    first = rounds[0]["world_fit"]
                    assert receipt["ridge_and_stats_sha256"] == first["ridge_and_stats_sha256"]
                    assert receipt["initial_model_sha256"] == first["final_model_sha256"]
                    assert receipt["round_data"][0]["sha256"] == first["round_data"][0]["sha256"]
                world += receipt["updates_completed"]
            if group["policy_update"]:
                assert admitted == set(CALIBRATION[j])
                policy += group["policy_update"]["updates_completed"]
            if group["status"] == "COMPLETE":
                assert group["policy_update"]["status"] == "COMPLETE"
                assert group["policy_update"]["updates_completed"] == 4
            if group["status"] in ("STOPPED_AT_CALIBRATION", "STOPPED_AT_POLICY_CHECK"):
                assert r == len(rounds) and s["status"] == group["status"]
                assert all((seed, r, reset) in cells for reset in CALIBRATION[j])
                if group["status"] == "STOPPED_AT_CALIBRATION":
                    assert admitted != set(CALIBRATION[j]) and group["policy_update"] is None
        if s["status"] == "PARTIAL":
            assert saved["native_decisions"] >= measured[seed] and saved["world_updates"] >= world
            assert saved["policy_updates"] >= policy
        else:
            assert s["status"] in ("EVALUATED", "STOPPED_AT_CALIBRATION", "STOPPED_AT_POLICY_CHECK")
            assert saved == dict(native_decisions=measured[seed], physics_substeps=10*measured[seed],
                                 servo_updates=5*measured[seed], world_updates=world, policy_updates=policy, ppo_updates=0)
            assert s["guard"]["startup_effective_bytes"] >= 4*2**30 and s["guard"]["runtime_min_effective_bytes"] >= 2*2**30
            assert s["guard"]["status"] == "COMPLETE" and s["guard"]["owned_cgroup_children_empty"]
            assert len(s["guard"]["cpu_affinity"]) == 1
        assert all(saved[k] <= LIMITS[k]//3 for k in LIMITS)
    assert examined == set(cells) and dict(total) == report["totals"]
    assert all(total[k] <= LIMITS[k] for k in LIMITS)


def test_independent_control_candidates_and_all_nine_pairs():
    report = read(REPORT)
    eligible = {s["seed"] for s in report["seeds"] if s["status"] == "EVALUATED"}
    for s in report["seeds"]:
        if s["seed"] in eligible:
            assert [r["status"] for r in s["rounds"]] == ["COMPLETE", "COMPLETE"]
    assert sorted((c["seed"], c["reset"]) for c in report["evaluation"]) == sorted(
        (seed, reset) for seed in eligible for reset in EVALUATION)
    passed = []
    with np.load(CURVES, allow_pickle=False) as curves:
        for pair in report["evaluation"]:
            prefix = f"s{pair['seed']}_eval{pair['reset']}"
            assert pair["endpoints_prefix"] == prefix
            x, y = (curves[prefix+"_"+side].astype(np.float64) for side in ("initial", "final"))
            np.testing.assert_array_equal(x[:2], y[:2])
            expected = []
            for side, values in (("initial", x), ("final", y)):
                assert values.ndim == 1 and 1 <= len(values) <= 101 and np.isfinite(values).all()
                assert values[0] == 1.10
                expected.append(clean(pair["native_events"][side], len(values)-1, 100))
            safe = all(expected)
            matches(pair["score"]["clean"], safe)
            qualifies = False
            if len(x) == len(y) == 101:
                peaks = []
                for side, values in (("initial", x), ("final", y)):
                    peaks.append(1000.*max(abs(values[:51] - values[0])))
                    matches(pair["score"][side], dict(rollback_mm=rollback(values), maximum_absolute_drift_mm=peaks[-1],
                            second_second_displacement_mm=1000.*(values[100]-values[50])))
                benefit = rollback(x)-rollback(y)
                matches(pair["score"]["rollback_reduction_mm"], benefit)
                qualifies = safe and benefit >= 30 and peaks[1] <= peaks[0]
            passed.append(bool(qualifies))
            matches(pair["score"]["passed"], qualifies)
    assert report["control_pairs_passed"] == sum(passed)
    matches(report["control_qualified"], len(passed) == 9 and all(passed))


def sample_count(mp4):
    # ISO BMFF video sample table. Reading the actual count ties the timeline to the MP4 bytes.
    data = memoryview(mp4)
    for requested in (b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsz"):
        at = 0
        while at < len(data):
            length, kind = struct.unpack_from(">I4s", data, at)
            header = 8
            if length == 1:
                length, = struct.unpack_from(">Q", data, at+8)
                header = 16
            if length == 0:
                length = len(data)-at
            assert header <= length <= len(data)-at
            if kind == requested:
                data = data[at+header:at+length]
                break
            at += length
        else:
            raise AssertionError(f"missing MP4 box {requested!r}")
    return struct.unpack_from(">I", data, 8)[0]


def test_video_begins_at_a_measured_pose_and_covers_every_selected_pair():
    report, video = read(REPORT), read(ROOT / "reports/cold_start94_video.json")
    movie = VIDEO.read_bytes()
    assert hashlib.sha256(movie).hexdigest() == video["sha256"] and len(movie) == video["bytes"] <= 5_000_000
    matches(video, dict(status="COMPLETE", cover_frames=0, fps=25, source_dt_s=.02, playback_speed=.5,
                        endpoint_hold_frames=8, new_physics_steps=0, policy_calls=0, model_steps=0, training_updates=0))
    expected = {}
    for seed in report["seeds"]:
        rows = [p for p in report["evaluation"] if p["seed"] == seed["seed"]]
        kind = "policy"
        if not rows:
            kind = "calibration"
            rows = [p for p in report["calibration"] if p["seed"] == seed["seed"] and p["round"] == len(seed["rounds"])]
        for p in rows:
            expected[p["seed"], p["reset"]] = (kind, p, seed["status"])
    assert video["pairs"] and [(p["seed"], p["reset"]) for p in video["pairs"]] == sorted(expected)
    next_frame = 0
    poses = 0
    with np.load(CURVES, allow_pickle=False) as curves:
        for p in video["pairs"]:
            kind, source, status = expected[p["seed"], p["reset"]]
            assert (p["kind"], p["status"], p["round"]) == (kind, status, source.get("round", 2))
            prefix = source["endpoints_prefix"]
            lengths = []
            for side, offset, group in (("left", -20, "initial"), ("right", 20, "final")):
                if kind == "calibration":
                    x = branch(curves, prefix, "native", offset)
                    trace_hash = source["source_hashes"]["native"][str(offset)]
                else:
                    x = curves[prefix+"_"+group].astype(np.float64)
                    trace_hash = source["source_hashes"][group]
                n = len(x)-1
                lengths.append(n)
                matches(p[side], dict(sha256=trace_hash, ticks=n, source_end_s=n*.02,
                                      final_displacement_mm=1000.*(x[-1]-x[0]), first_second_rollback_mm=rollback(x)))
            last = next_frame + max(lengths)
            assert (p["first_frame"], p["endpoint_frame"], p["last_frame"]) == (next_frame, last, last+8)
            next_frame, poses = last+9, poses + 2*(max(lengths)+1)
    assert video["rendered_poses"] == poses
    assert next_frame == video["frames"] == int(video["stream"]["nb_read_frames"]) == sample_count(movie)
    matches(video["duration_s"], next_frame/25.)


def test_posthoc_common_tick_and_scoreable_gradient_radii():
    report = read(REPORT)
    audit = read(ROOT / "reports/cold_start94_prefix.json")
    assert audit["status"] == "POST_HOC" and audit["curves_sha256"] == digest(CURVES)
    counts = Counter()
    video_pairs = []
    with np.load(CURVES, allow_pickle=False) as curves:
        for cell, saved in zip(report["calibration"], audit["cells"], strict=True):
            derivative = cell["score"]["model_derivative"]
            matches(saved, dict(seed=cell["seed"], reset=cell["reset"], model_derivative=derivative))
            for r, entry in zip((10, 20), saved["radii"], strict=True):
                low, high = (branch(curves, cell["endpoints_prefix"], "native", o) for o in (-r, r))
                slope = (short_return(high) - short_return(low))/(r*.02) if min(len(low), len(high)) > 32 else None
                comparable = slope is not None and derivative is not None
                sign = bool(derivative*slope > 0) if comparable else None
                close = bool(similar_slopes(derivative, slope)) if comparable else None
                matches(entry, dict(radius_rad_s=r/100., native_gradient=slope, comparable=comparable,
                                    same_sign=sign, agrees_25pct=close))
                counts.update(dict(eligible_native_derivatives=int(slope is not None), comparable_derivatives=int(comparable),
                                   same_sign=int(sign is True), within_25pct=int(close is True)))
                if r == 20:
                    video_pairs.append((low, high))
    matches(audit, dict(counts))
    tick = min(len(x)-1 for pair in video_pairs for x in pair)
    assert audit["video_last_common_tick"] == tick == 31
    drift = np.array([[max(0., x[0]-x[tick])*1000 for x in pair] for pair in video_pairs])
    gain = drift[:, 0] - drift[:, 1]
    for values, key in ((drift, "video_rollback_mm_range_at_common_tick"),
                        (gain, "video_positive_minus_negative_benefit_mm_range")):
        np.testing.assert_allclose(audit[key], [values.min(), values.max()], rtol=0., atol=1e-12)


def test_posthoc_contacts_from_signed_geometry_and_native_sources():
    report = read(REPORT)
    with gzip.open(ROOT / "reports/cold_start94_contacts.json.gz", "rt") as stream:
        audit = json.load(stream)
    expected = {}
    for seed in report["seeds"]:
        for group in seed["rounds"]:
            for ep in group["fit_episodes"]:
                expected[ep["sha256"]] = ("fit", seed["seed"], ep["reset"], 0., ep["ticks"], 50, ep["native_events"])
    for cell in report["calibration"]:
        for k, sha in cell["source_hashes"]["native"].items():
            expected[sha] = ("calibration", cell["seed"], cell["reset"], int(k)/100.,
                             cell["score"]["arms"][k]["real_ticks"], 50, cell["native_events"][k])
    with np.load(CURVES, allow_pickle=False) as curves:
        for pair in report["evaluation"]:
            for arm, sha in pair["source_hashes"].items():
                n = len(curves[pair["endpoints_prefix"]+"_"+arm])-1
                expected[sha] = ("evaluation", pair["seed"], pair["reset"], 0., n, 100, pair["native_events"][arm])
    sources = {s["sha256"]: s for s in audit["sources"]}
    assert set(sources) == set(expected) and len(sources) == len(audit["sources"])
    for sha, (kind, seed, reset, offset, n, horizon, events) in expected.items():
        s = sources[sha]
        matches(s, dict(kind=kind, seed=seed, reset=reset, offset_rad_s=offset, ticks=n, requested_ticks=horizon,
                        early_stop=n<horizon, final_archived_flags={k:bool(events[k]) for k in EVENTS}))
    for kind in ("fit", "calibration", "evaluation"):
        items = [s for s in sources.values() if s["kind"] == kind]
        matches(audit["by_kind"][kind], dict(episodes=len(items), early_stops=sum(s["early_stop"] for s in items),
                                           illegal_episodes=sum(s["final_archived_flags"]["illegal"] for s in items)))
    cases = audit["cases"]
    assert sorted(c["sha256"] for c in cases) == sorted(h for h,s in sources.items() if s["final_archived_flags"]["illegal"])
    matches(audit, dict(flagged_episodes=len(cases), position_queries=2*len(cases), physics_integrations=0,
                        policy_calls=0, force_queries=0))
    for phase, shift in (("previous", -1), ("terminal", 0)):
        by_body, paired = Counter(), Counter()
        any_touch = bilateral = 0
        for c in cases:
            matches(c, sources[c["sha256"]])
            ep = c[phase]
            tick = c["ticks"] + shift
            flags = c["final_archived_flags"] if phase == "terminal" else dict.fromkeys(EVENTS, False)
            matches(ep, dict(tick=tick, time_s=tick/50., archived_interval_ending_here=flags if tick else None))
            bodies = {p["body"] for p in ep["floor_nonwheel_candidates"] if p["signed_distance_m"] <= 0.}
            limbs = {b[:2] for b in bodies if b[:3] in ("FL_", "FR_", "RL_", "RR_")}
            both = bool(limbs & {"FL", "RL"} and limbs & {"FR", "RR"})
            pairs = {axle+"_"+b[3:] for b in bodies for prefix, axle in (("F", "front"), ("R", "rear"))
                     if b.startswith(prefix+"L_") and prefix+"R_"+b[3:] in bodies}
            assert ep["touching_or_penetrating_bodies"] == sorted(bodies) and ep["left_and_right"] is both
            assert set(ep["paired_axle_parts"]) == pairs
            by_body.update(bodies)
            paired.update(pairs)
            any_touch += bool(bodies)
            bilateral += both
        assert audit["phase_counts"][phase] == dict(body_episode_counts=dict(by_body), bilateral_episode_count=bilateral,
                            paired_axle_part_episode_counts=dict(paired), any_nonwheel_touch=any_touch)
