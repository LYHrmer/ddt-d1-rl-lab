"""Verify the temporal definition of sustained support and read back the published 93 training strata."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools93.coverage93 import phase_at_source, run, support_summary, support_windows


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "reports/stand_data93_coverage.json"
PLAN = ROOT / "docs/experiments93/stand_data.json"


def bits(codes):
    return np.array([[bool(code & (1 << wheel)) for wheel in range(4)] for code in codes])


def test_first_transition_counts_and_targets_have_a_different_denominator():
    codes = [0] + [15]*8
    rows = support_windows(bits(codes), episode_id=10)
    assert [(r.source_tick, r.last_tick) for r in rows] == [(1, 9)]
    summary = support_summary(rows)
    main, secondary = summary["source_and_targets"], summary["targets_only"]
    assert main["groups"]["mask_changed"] == {"windows": 1, "episodes": 1}
    assert main["groups"]["all_four"]["windows"] == 0
    assert main["by_switch_count"][1] == {"switches": 1, "windows": 1, "episodes": 1}
    assert secondary["groups"]["all_four"] == {"windows": 1, "episodes": 1}
    assert len(secondary["by_switch_count"]) == 8
    assert len(main["by_switch_count"]) == 9


def test_windows_do_not_bridge_episodes_and_repeated_windows_are_not_extra_episodes():
    assert support_windows(bits([15]*8), episode_id=1) == []
    rows = (support_windows(bits([9]*12), episode_id=1, first_tick=10)
            + support_windows(bits([0,15,0,15,0,15,0,15,0]), episode_id=2))
    s = support_summary(rows)
    assert [(r.source_tick,r.last_tick) for r in rows[:4]] == [(10,18),(11,19),(12,20),(13,21)]
    assert s["windows"] == 5 and s["episodes"] == 2
    assert s["source_and_targets"]["groups"]["stable_other"] == {"windows": 4, "episodes": 1}
    assert s["source_and_targets"]["by_switch_count"][8] == {"switches": 8, "windows": 1, "episodes": 1}
    assert s["targets_only"]["by_switch_count"][7] == {"switches": 7, "windows": 1, "episodes": 1}
    assert [phase_at_source(t) for t in (42,43,49,50)] == ["stand","command_change","command_change","move"]
    with pytest.raises(ValueError, match="support"):
        support_windows(np.zeros((9,4), np.uint8), episode_id=3)


def test_refuses_an_existing_coverage_attempt(tmp_path):
    (tmp_path / "marker").write_text("preserve")
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert list(tmp_path.iterdir()) == [tmp_path / "marker"]


def text_json(path):
    return json.loads(path.read_bytes())


def independent_windows(cases, key):
    """Rebuild physical endpoints from public 50x4 masks without importing the producer's indexing."""
    result = []
    for case in cases:
        masks = case["support_mask_01"]
        assert case["first_support_tick"] == 1 and case["last_support_tick"] == len(masks) == 50
        assert all(len(row)==4 and all(type(v) is int and v in (0,1) for v in row) for row in masks)
        endpoints = {tick: tuple(row) for tick, row in enumerate(masks, start=1)}
        for last in range(9,51):
            result.append((case[key], last, [endpoints[t] for t in range(last-8,last+1)]))
    return result


def independent_summary(rows):
    def pop(ids):
        return dict(windows=len(ids), episodes=len(set(ids)))

    summary = pop([key for key, _, _ in rows])
    for name, offset in (("source_and_targets", 0), ("targets_only", 1)):
        labels = {label: [] for label in ("all_four", "stable_other", "mask_changed")}
        counts = [[] for _ in range(9-offset)]
        for key, _, endpoints in rows:
            masks = endpoints[offset:]
            switched = sum(before != after for before,after in zip(masks, masks[1:]))
            if all(all(wheels) for wheels in masks):
                labels["all_four"].append(key)
            elif len(set(masks)) == 1:
                labels["stable_other"].append(key)
            else:
                labels["mask_changed"].append(key)
            counts[switched].append(key)
        summary[name] = dict(groups={k: pop(v) for k,v in labels.items()},
                             by_switch_count=[dict(switches=i, **pop(vals)) for i,vals in enumerate(counts)])
    return summary


def check_bands(bands, rows):
    assert [(b["first_last_tick"], b["final_last_tick"]) for b in bands] == [(9,16),(17,32),(33,50)]
    for b in bands:
        first, last = b["first_last_tick"], b["final_last_tick"]
        assert b == dict(first_last_tick=first, final_last_tick=last,
                         **independent_summary([row for row in rows if first <= row[1] <= last]))


def test_public_windows_follow_the_training_manifest_and_exclude_held_out_resets():
    record = text_json(PUBLIC)
    assert record["status"] == "COMPLETE" and record["horizon"] == 8
    assert record["protocol_sha256"] == hashlib.sha256(PLAN.read_bytes()).hexdigest()
    for name, sha in record["source_reports_sha256"].items():
        assert sha == text_json(PLAN)["source_sha256"][name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    world = text_json(ROOT / "reports/terrain_world91.json")["dataset"]
    trained = {m["train_seed"]: m for m in text_json(ROOT / "reports/uphill_curriculum89.json")["training"]["models"]
               if m["arm"] == "mixed"}
    assert len(record["sources"]) == len(world["sources"]) == len(trained) == 3
    source_episodes = {}
    for source, old in zip(record["sources"], world["sources"], strict=True):
        assert all(source[k]==v for k,v in old.items())
        m = trained[source["training_seed"]]
        assert source["trace_sha256"] == m["train_trace_sha256"]
        assert source["report_sha256"] == m["report_sha256"]
        assert source["parent_seed"] == m["parent_seed"] and source["parent_policy_sha256"] == m["parent_policy_sha256"]
        source_episodes[source["training_seed"]] = [e for e in m["episodes"] if e["terminated"] or e["truncated"]]
    candidates = set()
    phases = defaultdict(Counter)
    rows = record["episodes"]
    assert len(rows) == len(world["episodes"]) == 42
    for row, archived in zip(rows, world["episodes"], strict=True):
        assert all(row[k] == v for k,v in archived.items())
        source_ep = source_episodes[row["training_seed"]][row["source_episode_index"]]
        assert row["slope_start"] is source_ep["slope_start"]
        assert source_ep["ticks"] == row["end_row_exclusive"] - row["start_row"]
        assert row["legal_windows"] == row["stop_window_source_row_exclusive"] - row["first_window_source_row"]
        assert row["first_window_source_row"] == row["start_row"]
        assert row["legal_windows"] == max(0,source_ep["ticks"]-8)
        if row["slope_start"]:
            candidates.add(row["id"])
        if row["terrain"] == 1:
            for last in range(9,source_ep["ticks"]+1):
                phase = "stand" if last<=50 else "move" if last>=58 else "command_change"
                phases[row["slope_start"],phase][row["id"]] += 1
    public_phases = {(r["slope_start"],r["command_phase"]):r["summary"] for r in record["uphill_by_start_and_command"]}
    assert phases.keys() == public_phases.keys()
    for k, ids in phases.items():
        s = public_phases[k]
        assert s["windows"] == sum(ids.values()) and s["episodes"] == len(ids)
        for mode in ("source_and_targets","targets_only"):
            assert sum(i["windows"] for i in s[mode]["groups"].values()) == s["windows"]
            assert sum(i["windows"] for i in s[mode]["by_switch_count"]) == s["windows"]
    cnt = record["counts"]
    assert cnt["completed_episodes"] == len(rows) and cnt["training_transitions"] == sum(r["transitions"] for r in rows)
    assert cnt["terminated_episodes"] == sum(r["terminated"] for r in rows) == 4
    assert cnt["truncated_episodes"] == sum(r["truncated"] for r in rows) == 38
    assert cnt["legal_windows"] == sum(r["legal_windows"] for r in rows) == 11472
    assert cnt["discarded_unfinished_rows"] == sum(r["discarded_unfinished_rows"] for r in record["sources"]) == 480
    cases = record["slope_stand"]["episodes"]
    assert {c["episode_id"] for c in cases} == candidates and cnt["slope_stand_episodes"] == len(cases) == 6
    for c in cases:
        item = rows[c["episode_id"]]
        assert c["first_source_row"] == item["start_row"] and item["slope_start"]
        assert all(c[k] == item[k] for k in ("training_seed","parent_seed","source_episode_index"))
    assert cnt["slope_stand_windows"] == len(cases)*42 == 252
    assert cnt["slope_stand_distinct_reset_observations"] == len({c["reset_proprio627_sha256"] for c in cases}) == 6
    assert cnt["slope_stand_distinct_first_post_states"] == len({c["first_post_state45_sha256"] for c in cases}) == 6
    validation = record["archived_validation90"]["cases"]
    assert {c["trace_sha256"] for c in validation}.isdisjoint(s["trace_sha256"] for s in record["sources"])


def test_public_source_vs_target_support_and_late_windows_by_episode():
    record = text_json(PUBLIC)
    focus = record["slope_stand"]
    cases = focus["episodes"]
    windows = independent_windows(cases,"episode_id")
    assert focus["summary"] == independent_summary(windows)
    check_bands(focus["by_last_tick"], windows)
    assert focus["summary"]["source_and_targets"]["groups"]["all_four"] == {"windows":5, "episodes":1}
    assert focus["summary"]["targets_only"]["groups"]["all_four"] == {"windows":10, "episodes":4}
    assert focus["by_last_tick"][-1]["source_and_targets"]["groups"]["all_four"] == {"windows":0,"episodes":0}
    for case in cases:
        assert case["summary"] == independent_summary(independent_windows([case],"episode_id"))
    assert len(focus["by_parent"]) == 3
    for group in focus["by_parent"]:
        selected = [c for c in cases if c["parent_seed"] == group["parent_seed"]]
        assert len(selected) == 2
        assert group["summary"] == independent_summary(independent_windows(selected,"episode_id"))
    validation = record["archived_validation90"]
    parent_report = text_json(ROOT / "reports/stand_wbc90.json")["stand"]["cases"]
    truth = {(c["arm"],c["reset_seed"]): c for c in parent_report}
    assert len(validation["cases"]) == len(truth) == 6
    for case in validation["cases"]:
        old = truth[case["arm"],case["reset_seed"]]
        assert all(case[k] == old[k] for k in ("parent_seed","initial_state_sha256","trace_sha256"))
        assert case["summary"] == independent_summary(independent_windows([case],"reset_seed"))
    for arm,expected in (("original",48),("common",19)):
        group = validation["by_arm"][arm]
        subset = [c for c in validation["cases"] if c["arm"] == arm]
        test_rows = independent_windows(subset,"reset_seed")
        assert group["summary"] == independent_summary(test_rows)
        assert group["summary"]["source_and_targets"]["groups"]["all_four"] == {"windows":expected,"episodes":3}
        check_bands(group["by_last_tick"],test_rows)
