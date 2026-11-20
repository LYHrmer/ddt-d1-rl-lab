"""Audit contiguous, episode-grouped eight-step windows in the frozen 91 training set."""
from collections import defaultdict
import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np

from tools93.common93 import ROOT, PLAN_SHA, checked_plan, digest, read, require, write


H = 8
STAND = 50
TRAIN_SEEDS = (89011, 89023, 89037)
PARENT_SEEDS = (87011, 87023, 87037)
WHEELS = ("FL", "FR", "RL", "RR")
REPORTS = ("reports/uphill_curriculum89.json", "reports/terrain_world91.json", "reports/stand_wbc90.json")
LABELS = ("all_four", "stable_other", "mask_changed")
TARGET_BANDS = ((9, 16), (17, 32), (33, 50))


class Window(NamedTuple):
    episode_id: int
    source_tick: int
    last_tick: int
    span_four: bool
    target_four: bool
    span_changes: int
    target_changes: int


def support_windows(loaded, *, episode_id, first_tick=1):
    """Enumerate H8 windows within one continuous episode on its recorded endpoint grid."""
    loaded = np.asarray(loaded)
    require(loaded.ndim == 2 and loaded.shape[1] == 4 and loaded.dtype == bool
            and first_tick >= 0, "need ordered boolean support at four wheels for one episode")
    masks = loaded @ np.array([1, 2, 4, 8], dtype=np.uint8)
    rows = []
    for index in range(len(masks) - H):
        span = masks[index:index + H + 1]
        changes = span[1:] != span[:-1]
        tick = first_tick + index
        rows.append(Window(episode_id, tick, tick + H, bool(np.all(span == 15)),
                           bool(np.all(span[1:] == 15)), int(changes.sum()), int(changes[1:].sum())))
    return rows


def support_summary(windows):
    """Count overlapping windows and distinct episodes in each mutually exclusive support group."""
    windows = list(windows)

    def population(rows):
        return {"windows": len(rows), "episodes": len({row.episode_id for row in rows})}

    result = population(windows)
    for name in ("source_and_targets", "targets_only"):
        full_span = name == "source_and_targets"
        by_change_count = [[] for _ in range(H + int(full_span))]
        groups = {key: [] for key in LABELS}
        for row in windows:
            four = row.span_four if full_span else row.target_four
            changes = row.span_changes if full_span else row.target_changes
            label = "all_four" if four else ("mask_changed" if changes else "stable_other")
            groups[label].append(row)
            by_change_count[changes].append(row)
        result[name] = {
            "groups": {key: population(rows) for key, rows in groups.items()},
            "by_switch_count": [dict(switches=i, **population(rows)) for i, rows in enumerate(by_change_count)],
        }
    return result


def checked_source(path, expected):
    require(path.is_file() and not path.is_symlink() and digest(path) == expected, f"source identity: {path}")


def phase_at_source(source_tick):
    # target action intervals are source_tick + 1 through source_tick + H, inclusive.
    if source_tick + H <= STAND:
        return "stand"
    if source_tick >= STAND:
        return "move"
    return "command_change"


def endpoint_bands(windows):
    return [dict(first_last_tick=a, final_last_tick=b,
                 **support_summary([w for w in windows if a <= w.last_tick <= b])) for a, b in TARGET_BANDS]


def training_coverage(public89, public91):
    dataset = public91["dataset"]
    models = {m["train_seed"]: m for m in public89["training"]["models"] if m["arm"] == "mixed"}
    require(tuple(s["training_seed"] for s in dataset["sources"]) == TRAIN_SEEDS
            and set(models) == set(TRAIN_SEEDS), "registered three mixed sources required")
    episode_rows, stand_cases, stand_rows = [], [], []
    per_parent, per_phase = defaultdict(list), defaultdict(list)
    parents, runs, reset_observations, first_states = set(), set(), set(), set()
    sources = []
    for source, expected_parent in zip(dataset["sources"], PARENT_SEEDS, strict=True):
        seed, parent = source["training_seed"], source["parent_seed"]
        require(parent == expected_parent, "training seed/parent pairing differs")
        archive = models[seed]
        path = ROOT / "evidence/uphill89/training_01" / f"mixed_{seed}/train_trace.npz"
        require(source["path"] == str(path.relative_to(ROOT)) and archive["parent_seed"] == parent
                and source["trace_sha256"] == archive["train_trace_sha256"]
                and source["report_sha256"] == archive["report_sha256"], "training sources disagree")
        checked_source(path, source["trace_sha256"])
        checked_source(path.parent / "report.json", source["report_sha256"])
        receipt = read(path.parent / "report.json")
        require(receipt["status"] == "COMPLETE" and receipt["arm"] == "mixed"
                and receipt["training_seed"] == seed and receipt["parent_seed"] == parent
                and receipt["parent_policy_sha256"] == archive["parent_policy_sha256"], "partial training source")
        with np.load(path, allow_pickle=False) as raw:
            keys = ("episode_tick", "slope_start", "terrain", "command_vx_m_s", "terminated", "truncated",
                    "loaded_wheels_past_toe", "qpos", "qvel", "proprio627")
            data = {k: raw[k] for k in keys}
            require("initial_qpos" not in raw and "initial_qvel" not in raw, "unexpected reset schema")
        for k, a in data.items():
            require(hashlib.sha256(a.tobytes()).hexdigest() == receipt["trace"]["array_sha256"][k], f"array: {k}")
        tick, loaded = data["episode_tick"], data["loaded_wheels_past_toe"]
        done = data["terminated"] | data["truncated"]
        starts = np.flatnonzero(tick == 1)
        require(len(tick) == receipt["trace"]["completed_recorded_ticks"] == 4096
                and loaded.shape == (len(tick), 4) and loaded.dtype == bool
                and np.array_equal(starts, np.r_[0, 1 + np.flatnonzero(done[:-1])])
                and not np.any(data["terminated"] & data["truncated"]), "episode boundaries or support columns")
        ended = [e for e in dataset["episodes"] if e["training_seed"] == seed]
        n = len(ended)
        require(n == source["completed_episodes"] == int(done.sum()) and n <= len(starts) <= n + 1,
                "ended episode inventory")
        discarded = int(len(tick) - starts[-1]) if n < len(starts) else 0
        require(discarded == source["discarded_unfinished_rows"], "unfinished tail")
        sources.append(dict(source, parent_policy_sha256=receipt["parent_policy_sha256"]))
        for i, e in enumerate(ended):
            lo, hi = int(starts[i]), int(starts[i+1]) if i+1 < len(starts) else len(tick)
            end = hi-1
            length = hi-lo
            t = tick[lo:hi]
            slope = bool(data["slope_start"][lo])
            require(e["id"] == len(episode_rows) and (lo, hi) == (e["start_row"], e["end_row_exclusive"])
                    and e["transitions"] == length-1 and np.array_equal(t, np.arange(1, length+1))
                    and np.all(data["terrain"][lo:hi] == e["terrain"])
                    and np.all(data["slope_start"][lo:hi] == slope)
                    and np.array_equal(data["command_vx_m_s"][lo:hi] > 0, t > STAND)
                    and np.all(data["command_vx_m_s"][lo:hi][t <= STAND] == 0)
                    and e["terminated"] == bool(data["terminated"][end])
                    and e["truncated"] == bool(data["truncated"][end]), "episode metadata or command schedule")
            legal = max(0, length-H)
            episode_rows.append(dict(e, parent_seed=parent, source_episode_index=i,
                                     slope_start=slope, legal_windows=legal, first_window_source_row=lo,
                                     stop_window_source_row_exclusive=lo+legal))
            if e["terrain"] != 1:
                require(not slope and not loaded[lo:hi].any(), "slope-only label on another terrain")
                continue
            rows = support_windows(loaded[lo:hi], episode_id=e["id"])
            require(len(rows) == legal, "support windows and training windows differ")
            for row in rows:
                per_phase[slope, phase_at_source(row.source_tick)].append(row)
            if not slope:
                continue
            require(length >= STAND, "standing archive ends before registered span")
            stationary = [r for r in rows if phase_at_source(r.source_tick) == "stand"]
            require(len(stationary) == STAND-H, "stationary windows")
            reset_sha = hashlib.sha256(data["proprio627"][lo].tobytes()).hexdigest()
            state_sha = hashlib.sha256(data["qpos"][lo].tobytes() + data["qvel"][lo].tobytes()).hexdigest()
            reset_observations.add(reset_sha)
            first_states.add(state_sha)
            parents.add(parent)
            runs.add(seed)
            stand_cases.append(dict(episode_id=e["id"], training_seed=seed, parent_seed=parent, source_episode_index=i,
                                    first_source_row=lo, first_support_tick=1, last_support_tick=STAND,
                                    reset_proprio627_sha256=reset_sha, first_post_state45_sha256=state_sha,
                                    support_mask_01=loaded[lo:lo+STAND].astype(np.uint8).tolist(),
                                    summary=support_summary(stationary)))
            stand_rows.extend(stationary)
            per_parent[parent].extend(stationary)
    counts = {
        "completed_episodes": len(episode_rows), "terminated_episodes": sum(e["terminated"] for e in episode_rows),
        "truncated_episodes": sum(e["truncated"] for e in episode_rows),
        "training_transitions": sum(e["transitions"] for e in episode_rows),
        "legal_windows": sum(e["legal_windows"] for e in episode_rows), "slope_stand_windows": len(stand_rows),
        "slope_stand_episodes": len(stand_cases), "slope_stand_distinct_reset_observations": len(reset_observations),
        "slope_stand_distinct_first_post_states": len(first_states), "slope_stand_parent_policies": len(parents),
        "slope_stand_rollout_files": len(runs), "discarded_unfinished_rows": sum(s["discarded_unfinished_rows"] for s in sources),
    }
    require(counts["completed_episodes"] == dataset["completed_episodes"] == 42
            and counts["terminated_episodes"] == dataset["terminated_episodes"] == 4
            and counts["truncated_episodes"] == dataset["truncated_episodes"] == 38
            and counts["training_transitions"] == dataset["transitions"] == 11766
            and counts["legal_windows"] == 11472 and counts["slope_stand_episodes"] == 6
            and counts["slope_stand_windows"] == 252, "frozen training denominator")
    for m in public91["families"]["candidate"]["models"]:
        fit = m["source"]["model"]
        require(fit["horizon"] == H and fit["training_windows"] == counts["legal_windows"], "91 fit receipt")
    return dict(counts=counts, sources=sources, episodes=episode_rows,
                uphill_by_start_and_command=[dict(slope_start=s, command_phase=c, summary=support_summary(v))
                                             for (s,c),v in sorted(per_phase.items())],
                slope_stand=dict(episodes=stand_cases, summary=support_summary(stand_rows),
                                by_parent=[dict(parent_seed=p, summary=support_summary(rows))
                                           for p, rows in sorted(per_parent.items())],
                                by_last_tick=endpoint_bands(stand_rows)))


def archived_validation(public90):
    cases, by_arm = [], defaultdict(list)
    for e in public90["stand"]["cases"]:
        reset, arm = e["reset_seed"], e["arm"]
        require((reset, e["parent_seed"]) in zip((90101, 90102, 90103), PARENT_SEEDS)
                and arm in ("original", "common") and e["clean"] and e["status"] == "COMPLETE"
                and e["ticks"] == 100, "unqualified physical reference")
        path = ROOT / "evidence/wbc90/stand_01" / f"{arm}_{reset}/trace.npz"
        checked_source(path, e["trace_sha256"])
        with np.load(path, allow_pickle=False) as raw:
            command = raw["command"]
            loaded = raw["wheel_contacts_past_toe"]
            require(loaded.shape == (100, 4) and loaded.dtype == bool and command.shape == (100, 3)
                    and np.all(command[:STAND] == 0) and np.allclose(raw["time_s"][:STAND], .02*np.arange(1,STAND+1),
                                                                       rtol=0, atol=1e-9), "90 standing phase")
        masks = loaded[:STAND]
        windows = support_windows(masks, episode_id=reset)  # grouped within each arm
        by_arm[arm].extend(windows)
        cases.append(dict(reset_seed=reset, parent_seed=e["parent_seed"], arm=arm, trace_sha256=e["trace_sha256"],
                          initial_state_sha256=e["initial_state_sha256"], first_support_tick=1, last_support_tick=STAND,
                          support_mask_01=masks.astype(np.uint8).tolist(), summary=support_summary(windows)))
    require({(c["reset_seed"], c["arm"]) for c in cases} ==
            {(s, a) for s in (90101,90102,90103) for a in ("original","common")} and len(cases) == 6, "90 matrix")
    return dict(cases=cases, by_arm={arm: dict(summary=support_summary(rows), by_last_tick=endpoint_bands(rows))
                                    for arm, rows in by_arm.items()})


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(protocol_sha256=PLAN_SHA, status="PARTIAL", horizon=H)
    try:
        plan = checked_plan()
        require(tuple(plan["parents"]) == PARENT_SEEDS, "parent policies changed")
        public = {}
        source_hashes = {name: plan["source_sha256"][name] for name in REPORTS}
        for name, expected in source_hashes.items():
            path = ROOT / name
            checked_source(path, expected)
            public[name] = read(path)
        result["source_reports_sha256"] = source_hashes
        result["semantics"] = {
            "wheels": WHEELS, "mask": "Per-wheel floor contact >= the uphill toe with normal force >1 N at the observed endpoint.",
            "primary": "Nine observed endpoints (source and eight targets), eight adjacent comparisons.",
            "secondary": "Eight target endpoints, seven adjacent comparisons; the source-to-first-target switch is omitted.",
            "partitions": "all_four, stable_other and mask_changed are mutually exclusive within a basis. A switch counts a changed mask once per time interval even when multiple wheels change.",
            "counts": "Episode counts are distinct within each bin. Several overlapping windows from one episode can enter different bins.",
            "rows": "Zero-based source file rows, stop exclusive. A window starting at row j uses state[j], actions[j+1:j+9], states[j+1:j+9]. Its eight target-interval commands define the phase.",
            "split": "89 mixed is the complete 91 fit set. Group future fit/dev splits by whole rollout/episode before windowing. Keep all histories from a validation reset together across parents and arms. The physical 90/93 panels are reserved for evaluation.",
            "reset": "Training starts at post-step tick 1. The zero-time actor observation is retained; a recorded float64 reset state45 is absent.",
        }
        result.update(training_coverage(public["reports/uphill_curriculum89.json"], public["reports/terrain_world91.json"]))
        result["archived_validation90"] = archived_validation(public["reports/stand_wbc90.json"])
        result["status"] = "COMPLETE"
    except BaseException as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write(out / "coverage.json", result)
    return result
