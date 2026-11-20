"""Recompute the crossed standing results using only files available in a clean checkout."""
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import struct

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "reports"
PLAN = ROOT / "docs/experiments93/stand_data.json"
PLAN_SHA = "2412b8aa31e78ebb918f30a1890c6319e916d184127470ca063f25ed88964e8d"
PARENTS = (87011, 87023, 87037)
RESETS = (90101, 90102, 90103)
MODELS = (91011, 91023, 91037)
ARMS = ("original", "common")
CELLS = [(p, r) for p in PARENTS for r in RESETS]
DIAGONAL = set(zip(PARENTS, RESETS, strict=True))
ZERO_ACTION = hashlib.sha256(bytes(16*4)).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    assert isinstance(value, str) and len(value) == 64 and len(bytes.fromhex(value)) == 32


def keyed(rows, *fields):
    index = {tuple(row[k] for k in fields): row for row in rows}
    assert len(index) == len(rows), "duplicate published key"
    return index


def curves(path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def matches(actual, expected):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for k, v in expected.items():
            matches(actual[k], v)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            matches(a, b)
    elif isinstance(expected, bool) or expected is None:
        assert actual is expected
    elif isinstance(expected, float):
        assert type(actual) is not bool and actual == pytest.approx(expected, abs=1e-12, rel=0.)
    else:
        assert actual == expected


@pytest.fixture(scope="module")
def public():
    report, plan = read(PUB / "stand_data93.json"), read(PLAN)
    assert sha(PLAN) == report["protocol_sha256"] == PLAN_SHA
    tape = PUB / "stand_data93_curves.jsonl.gz"
    coverage = PUB / "stand_data93_coverage.json"
    assert sha(tape) == report["curves_sha256"] and sha(coverage) == report["coverage_sha256"]
    rows = keyed(curves(tape), "family", "model_seed", "parent_seed", "reset_seed", "arm")
    assert set(rows) == {(family, model, p, r, arm) for p,r in CELLS for arm in ARMS
                         for family, seeds in (("physics", (None,)), ("fixed", MODELS)) for model in seeds}
    cells = keyed(report["cells"], "parent_seed", "reset_seed")
    assert set(cells) == set(CELLS) and len(rows) == 72
    return plan, report, rows, cells, read(coverage)


def rollback(row, h):
    if row["status"] != "COMPLETE" and h == 50:
        return None
    xs = row["world_x_m"]
    return max(0., xs[0] - xs[h]) if len(xs) > h else None


def grade(truth, estimate):
    effect = truth[0] - truth[1]
    complete = None not in estimate
    pred = estimate[0] - estimate[1] if complete else None
    delta = abs(pred - effect) if complete else None
    absolute = [abs(p-t) if p is not None else None for p,t in zip(estimate, truth, strict=True)]
    checks = dict(both_complete=complete, positive=complete and pred > 0,
                  effect_tolerance=complete and delta <= .003, beats_zero_effect=complete and delta < abs(effect),
                  both_absolute_arms=complete and max(absolute) <= .030)
    return dict(truth=truth, prediction=estimate, true_effect=effect, predicted_effect=pred, checks=checks,
                effect_abs_error=delta, zero_effect_abs_error=abs(effect), arm_abs_errors=absolute,
                passed=all(checks.values()))


def sign(values):
    if None in values:
        return "INCOMPLETE"
    if all(v > 0 for v in values):
        return "HIGH"
    if all(v < 0 for v in values):
        return "LOW"
    return "MIXED"


def attribution(grid):
    assert set(grid) == set(CELLS)
    if "INCOMPLETE" in grid.values():
        return "INCOMPLETE"
    for tag, groups in (("RESET_PATTERN", [[grid[p,r] for p in PARENTS] for r in RESETS]),
                        ("PARENT_PATTERN", [[grid[p,r] for r in RESETS] for p in PARENTS])):
        if all(len(set(g)) == 1 for g in groups) and {g[0] for g in groups} == {"HIGH", "LOW"}:
            return tag
    return "MIXED"


def spans(masks):
    # Row index 0 records physical endpoint 1. Include the source and all eight targets.
    assert len(masks) == 50
    return [all(masks[t-1] == 15 for t in range(end-8, end+1)) for end in range(33, 51)]


def late_contrast(real, model):
    if real["status"] != "COMPLETE" or model["status"] != "COMPLETE":
        return None
    full, other = [], []
    for end, every_wheel in zip(range(33, 51), spans(real["loaded_mask"]), strict=True):
        start = end-8
        px, tx = model["world_x_m"], real["world_x_m"]
        error = abs((px[end]-px[start]) - (tx[end]-tx[start]))
        (full if every_wheel else other).append(error)
    return median(full)-median(other) if full and other else None


def test_strict_signs_gates_and_nine_endpoint_displacement_contrast():
    reset_pattern = {(p,r): "HIGH" if r == RESETS[0] else "LOW" for p,r in CELLS}
    assert attribution(reset_pattern) == "RESET_PATTERN"
    assert attribution({(p,r): "HIGH" if p == PARENTS[0] else "LOW" for p,r in CELLS}) == "PARENT_PATTERN"
    assert attribution(dict.fromkeys(CELLS, "HIGH")) == "MIXED"
    assert sign([1., 0.]) == sign([1., -1.]) == "MIXED"
    assert sign([1., None]) == "INCOMPLETE"
    reset_pattern[CELLS[0]] = "INCOMPLETE"
    assert attribution(reset_pattern) == "INCOMPLETE"
    # A gap at the source endpoint alone disqualifies the first late window.
    masks = [15]*50
    masks[24] = 7
    assert spans(masks) == [False] + [True]*17
    truth = dict(status="COMPLETE", loaded_mask=masks, world_x_m=[1.]*51)
    pred = dict(status="COMPLETE", world_x_m=[1.]*51)
    pred["world_x_m"][25] = 1.008
    assert late_contrast(truth, pred) == pytest.approx(-.008)
    assert late_contrast(dict(truth, loaded_mask=[15]*50), pred) is None
    assert not grade([.008, 0.], [1.008, 1.])["passed"]
    assert not grade([.008, 0.], [None, 0.])["passed"]


def test_inputs_first_endpoints_and_immutable_diagonal(public):
    plan, _, rows, _, coverage = public
    assert (tuple(plan["parents"]), tuple(plan["resets"]), tuple(plan["model_seeds"]), tuple(plan["arms"])) == (
        PARENTS, RESETS, MODELS, ARMS)
    for name, expected in plan["source_sha256"].items():
        if not name.startswith("evidence/"):
            assert sha(ROOT / name) == expected
    olds = [row for row in curves(PUB / "fixed_actions92_curves.jsonl.gz") if row["family"] in ("fixed", "physics")
            and row["task"] == "stand"]
    old_curves = keyed(olds, "family", "model_seed", "parent_seed", "reset_seed", "arm")
    phys90 = keyed(read(PUB / "stand_wbc90.json")["stand"]["cases"], "parent_seed", "reset_seed", "arm")
    prior_masks = keyed(coverage["archived_validation90"]["cases"], "parent_seed", "reset_seed", "arm")
    starts, starts_with_obs, firsts = {}, {}, {}
    for key, row in rows.items():
        family, model, parent, reset, arm = key
        n = len(row["ticks"])-1
        assert row["ticks"] == list(range(n+1)) and 0 <= n <= 50
        if family == "physics" or row["status"] == "COMPLETE":
            assert row["status"] == "COMPLETE" and n == 50
        for k in ("world_x_m", "world_vx_m_s", "body_vx_m_s"):
            assert len(row[k]) == n+1 and all(math.isfinite(x) for x in row[k])
        assert row["world_x_m"][0] == 1.10 and row["world_vx_m_s"][0] == 0.
        for k in ("trace_sha256", "first_state_sha256", "applied_sha256"):
            fingerprint(row[k])
        assert starts.setdefault(reset, row["first_state_sha256"]) == row["first_state_sha256"]
        if n:
            assert row["first_applied_sha256"] == ZERO_ACTION
            fingerprint(row["first_endpoint_sha256"])
            group = family, model, reset
            assert firsts.setdefault(group, row["first_endpoint_sha256"]) == row["first_endpoint_sha256"]
        if family == "physics":
            fingerprint(row["initial_state_observation_sha256"])
            assert starts_with_obs.setdefault(reset, row["initial_state_observation_sha256"]) == row["initial_state_observation_sha256"]
            assert row["policy_sha256"] == plan["policies_sha256"][str(parent)]
            masks = row["loaded_mask"]
            assert len(masks) == 50 and all(type(m) is int and 0 <= m < 16 for m in masks)
            assert len(row["pitch_rad"]) == 50 and all(math.isfinite(v) for v in row["pitch_rad"])
        elif n == 50:
            assert row["applied_sha256"] == rows["physics",None,parent,reset,arm]["applied_sha256"]
        assert len(row["wheel_speed_rad_s"]) == n
        assert all(len(v) == 4 and all(math.isfinite(x) for x in v) for v in row["wheel_speed_rad_s"])
        if (parent, reset) in DIAGONAL:
            old = old_curves[key]
            assert row["first_state_sha256"] == old["initial_state_sha256"]
            for k in ("first_applied_sha256", "first_endpoint_sha256", "applied_sha256", "trace_sha256", "ticks",
                      "world_x_m", "world_vx_m_s", "body_vx_m_s", "status"):
                assert row[k] == old[k]
            if family == "physics":
                earlier = prior_masks[parent,reset,arm]
                assert row["trace_sha256"] == phys90[parent,reset,arm]["trace_sha256"] == earlier["trace_sha256"]
                assert row["initial_state_observation_sha256"] == phys90[parent,reset,arm]["initial_state_sha256"]
                assert row["loaded_mask"] == [sum(bit << i for i,bit in enumerate(mask)) for mask in earlier["support_mask_01"]]
    assert len(set(starts.values())) == len(set(starts_with_obs.values())) == len(RESETS)


def test_all_grades_biases_and_registered_attribution(public):
    _, report, rows, cells, _ = public
    prior = read(PUB / "fixed_actions92.json")
    olds = {(m["seed"], p["reset_seed"]): p for m in prior["models"] for p in m["pairs"] if p["task"] == "stand"}
    passed_by_model, primary = dict.fromkeys(MODELS, 0), {}
    for (p,r), cell in cells.items():
        assert cell["source"] == ("90/92" if (p,r) in DIAGONAL else "93")
        measured = [rows["physics",None,p,r,arm] for arm in ARMS]
        truth = [rollback(v, 50) for v in measured]
        matches(cell["truth"], truth)
        matches(cell["true_effect_m"], truth[0]-truth[1])
        scores = keyed(cell["models"], "model_seed")
        assert scores.keys() == {(m,) for m in MODELS}
        for m in MODELS:
            pred = [rollback(rows["fixed",m,p,r,arm], 50) for arm in ARMS]
            score = grade(truth, pred)
            matches(scores[m,]["grade"], score)
            passed_by_model[m] += int(score["passed"])
            if (p,r) in DIAGONAL:
                matches(olds[m,r]["fixed"], score)
        for h in (16,32,50):
            errs = []
            for m in MODELS:
                for arm, real in zip(ARMS, measured, strict=True):
                    predicted = rollback(rows["fixed",m,p,r,arm], h)
                    errs.append(None if predicted is None else predicted-rollback(real,h))
            band = cell["bias"][str(h)]
            matches(band["errors_m"], errs)
            assert band["sign"] == sign(errs)
            if h == 50:
                primary[p,r] = sign(errs)
    assert report["model_pair_passed"] == sum(passed_by_model.values())
    matches(report["per_model_passed"], {str(k): n for k,n in passed_by_model.items()})
    assert report["all_models_qualified"] is all(n == 9 for n in passed_by_model.values())
    assert report["attribution"] == attribution(primary)
    assert report["physical_data_complete"] is True
    expect_complete = all(row["status"] == "COMPLETE" for row in rows.values())
    assert report["status"] == ("COMPLETE" if expect_complete else "INCOMPLETE_MODEL_DIAGNOSTIC")


def test_late_phase_training_coverage_and_eligibility(public):
    _, report, rows, cells, coverage = public
    episodes = coverage["slope_stand"]["episodes"]
    assert len(episodes) == coverage["counts"]["slope_stand_episodes"] == 6
    assert coverage["counts"]["legal_windows"] == 11472 and coverage["counts"]["slope_stand_windows"] == 252
    train_persistent = 0
    for e in episodes:
        masks = [sum(bit << i for i,bit in enumerate(row)) for row in e["support_mask_01"]]
        train_persistent += sum(spans(masks))
    curriculum = report["curriculum"]
    assert curriculum["training_late_all_four"] == train_persistent
    eligible, witnessed = {}, set()
    for (p,r), cell in cells.items():
        actual = {arm: rows["physics",None,p,r,arm] for arm in ARMS}
        persistent = {arm: sum(spans(c["loaded_mask"])) for arm,c in actual.items()}
        matches(cell["late_persistent_by_arm"], persistent)
        if any(persistent.values()):
            witnessed.add((p,r))
        per_trace = [late_contrast(actual[arm], rows["fixed",m,p,r,arm]) for m in MODELS for arm in ARMS]
        matches(cell["late_arm_model_contrasts"], per_trace)
        present = [x for x in per_trace if x is not None]
        single = median(present) if present else None
        matches(cell["late_contrast_m"], single)
        if single is not None:
            eligible[p,r] = single
    qualified_parents = {p for p,r in eligible}
    by_reset = {str(r): [eligible[p,r] for p in PARENTS if (p,r) in eligible] for r in RESETS}
    reset_medians = {key: median(values) if values else None for key,values in by_reset.items()}
    assert curriculum["eligible_cells"] == len(eligible)
    matches(curriculum["eligible_by_reset"], {key: len(values) for key,values in by_reset.items()})
    matches(curriculum["reset_median_contrasts_m"], reset_medians)
    gap = train_persistent == 0 and {p for p,r in witnessed} == set(PARENTS) and {r for p,r in witnessed} == set(RESETS)
    assert curriculum["coverage_gap"] is gap
    complete = all(row["status"] == "COMPLETE" for row in rows.values())
    expected = (complete and gap and len(eligible) >= 6 and qualified_parents == set(PARENTS)
                and all(v is not None and v > 0. for v in reset_medians.values()))
    assert curriculum["advise_sampling_test"] is expected


def test_resource_receipts_and_actual_work(public):
    plan, report, rows, cells, _ = public
    budget, count = plan["budget"], report["counts"]
    fixed = [c for c in rows.values() if c["family"] == "fixed"]
    new_fixed = [c for c in fixed if (c["parent_seed"],c["reset_seed"]) not in DIAGONAL]
    new_physical = [c for c in rows.values() if c["family"] == "physics" and (c["parent_seed"],c["reset_seed"]) not in DIAGONAL]
    ticks = sum(len(row["ticks"])-1 for row in new_physical)
    expected = dict(physical_cells=len(cells), physical_arms=18, physical_reset_states=3, new_physical_arms=len(new_physical),
                    reused_physical_arms=6, new_policy_steps=ticks, new_pd_updates=5*ticks, new_native_steps=10*ticks,
                    model_pairs=len(cells)*3, fixed_arms=len(fixed), new_fixed_arms=len(new_fixed), reused_fixed_arms=18,
                    new_model_accepted_steps=sum(len(c["ticks"])-1 for c in new_fixed), world_model_updates=0, policy_updates=0)
    for k,v in expected.items():
        assert count[k] == v
    assert count["new_model_accepted_steps"] <= count["new_model_steps"] <= budget["new_model_steps"]
    if all(c["status"] == "COMPLETE" for c in new_fixed):
        assert count["new_model_accepted_steps"] == count["new_model_steps"] == 1800
    for key in ("new_physical_arms", "reused_physical_arms", "new_policy_steps", "new_pd_updates", "new_native_steps",
                "world_model_updates", "policy_updates"):
        assert expected[key] == budget[key]
    assert len(new_fixed) == budget["new_model_arms"] and len(fixed)-len(new_fixed) == budget["reused_fixed_model_arms"]
    names = {"coverage": "coverage93.py", "collect": "collect93.py", "inputs": "model93.py", "predict": "model93.py"}
    assert report["resources"].keys() == names.keys()
    for stage, receipt in report["resources"].items():
        assert receipt["source_sha256"] == sha(ROOT / "tools93" / names[stage])
        for k in ("guard_sha256", "receipt_sha256"):
            fingerprint(receipt[k])
        assert receipt["admitted_effective_bytes"] >= plan["resources"]["minimum_effective_gib_at_start"] << 30
        assert receipt["minimum_running_effective_bytes"] >= plan["resources"]["minimum_effective_gib_during_run"] << 30
        assert len(receipt["cpu_affinity"]) == 1 and receipt["elapsed_s"] > 0
        assert 0 < receipt["peak_group_bytes"] <= plan["resources"]["memory_max_gib"] << 30
        assert all(receipt["final_memory_events"][k] == 0 for k in ("high","max","oom","oom_kill"))
    assert report["resources"]["coverage"]["receipt_sha256"] == report["coverage_sha256"]


def boxes(data):
    """Read sequential ISO base media boxes by their declared size, including extended sizes."""
    pos = 0
    while pos < len(data):
        assert pos+8 <= len(data)
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            assert pos+16 <= len(data)
            size, = struct.unpack_from(">Q", data, pos+8)
            header = 16
        elif size == 0:
            size = len(data)-pos
        assert header <= size <= len(data)-pos
        yield kind, data[pos+header:pos+size]
        pos += size


def one_box(data, kind):
    selected = [box for tag,box in boxes(data) if tag == kind]
    assert len(selected) == 1
    return selected[0]


def video_samples(contents):
    """Recover the video dimensions, sample count and duration from the actual MP4 sample table."""
    root = memoryview(contents)
    one_box(root, b"ftyp")
    video = []
    for name, track in boxes(one_box(root, b"moov")):
        if name == b"trak":
            media = one_box(track, b"mdia")
            if one_box(media, b"hdlr")[8:12] == b"vide":
                video.append(media)
    assert len(video) == 1
    media = video[0]
    header = one_box(media, b"mdhd")
    if header[0] == 0:
        scale, duration = struct.unpack_from(">II", header, 12)
    else:
        assert header[0] == 1
        scale, duration = struct.unpack_from(">IQ", header, 20)
    table = one_box(one_box(media, b"minf"), b"stbl")
    entry = one_box(one_box(table, b"stsd")[8:], b"avc1")
    width, height = struct.unpack_from(">HH", entry, 24)
    sizes = one_box(table, b"stsz")
    size, samples = struct.unpack_from(">II", sizes, 4)
    if not size:
        assert len(sizes) == 12+4*samples
    times = one_box(table, b"stts")
    n, = struct.unpack_from(">I", times, 4)
    runs = list(struct.iter_unpack(">II", times[8:]))
    assert len(runs) == n and sum(c for c,_ in runs) == samples
    assert sum(c*d for c,d in runs) == duration
    return width, height, samples, duration/scale


def test_video_contains_all_nine_paired_physical_measurements(public):
    _, report, rows, cells, _ = public
    video = read(PUB / "stand_data93_video.json")
    filename = ROOT / "docs/assets/stand_data93.mp4"
    assert video["status"] == "COMPLETE" and video["protocol_sha256"] == PLAN_SHA
    assert video["sha256"] == sha(filename) and video["bytes"] == filename.stat().st_size
    assert 0 < video["bytes"] <= 5_000_000
    assert video["score_sha256"] == sha(PUB / "stand_data93.json")
    assert video["physical_seal_sha256"] == report["resources"]["collect"]["receipt_sha256"]
    assert video["diagonal_report_sha256"] == read(PUB / "stand_wbc90.json")["stand_sha256"]
    assert video["physical_cases"] == 18
    assert all(video[key] == 0 for key in ("new_physics_steps", "policy_calls", "model_steps", "training_updates"))
    assert video["source_dt_s"] == .02 and video["fps"] == 25 and video["playback_speed"] == .5
    stream = video["stream"]
    assert stream["codec_name"] == "h264" and stream["r_frame_rate"] == "25/1"
    width, height, frame_count, duration = video_samples(filename.read_bytes())
    assert (width, height) == (stream["width"], stream["height"]) == (1280,720)
    assert frame_count == int(stream["nb_read_frames"]) == video["frames"]
    assert duration == pytest.approx(float(stream["duration"]), rel=0, abs=1e-6)
    assert duration == pytest.approx(frame_count / video["fps"], abs=1e-8)
    counts = video["frame_counts"]
    assert counts["movement_per_pair"] / video["fps"] * video["playback_speed"] == .02*50
    assert video["rendered_poses"] == 2*len(CELLS)*51
    position = counts["intro"]
    assert [(c["parent_seed"], c["reset_seed"]) for c in video["cells"]] == CELLS
    for clip in video["cells"]:
        p, r = clip["parent_seed"], clip["reset_seed"]
        assert clip["clip_first_frame"] == position
        position += counts["movement_per_pair"]
        assert clip["endpoint_first_frame"] == position
        position += counts["endpoint_per_pair"]
        assert clip["clip_last_frame"] == position-1
        for i, arm in enumerate(ARMS):
            assert clip["source_sha256"][arm] == rows["physics",None,p,r,arm]["trace_sha256"]
            assert clip["rollback_mm"][arm] == pytest.approx(1000*cells[p,r]["truth"][i], abs=1e-6, rel=0)
        assert clip["true_effect_mm"] == pytest.approx(1000*cells[p,r]["true_effect_m"], abs=1e-6, rel=0)
    assert position + counts["summary"] == counts["total"] == frame_count
