"""Paired uphill wheel-target intervention on the unmodified C++34 closed loop."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from tools87.run87 import ROOT, cpp_actor, parent_inputs, read, sha, snapshot, write
from tools87.scoring87 import score
from training.d1_rl.interface import ACTOR_DIM, LEG_INDICES, WHEEL_INDICES as WHEELS

PLAN = ROOT / "docs/experiments88/uphill_wheel_probe.json"
BASE = ROOT / "evidence/uphill88"
OUTPUT = BASE / "probe_01"


def apply_intervention(raw, tick, arm, plan):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (16,) or not np.isfinite(raw).all() or (np.abs(raw) > 1).any():
        raise ValueError("invalid C++34 action")
    issued = raw.copy()
    offset = plan["arm_delta_rad_s"][arm] if tick >= plan["issue_start_tick"] else 0.0
    requested = raw[WHEELS] + np.float32(offset / plan["wheel_scale_rad_s"])
    issued[WHEELS] = np.clip(requested, -1, 1)
    return issued, requested != issued[WHEELS]


def mean_score_speed(endpoints, interval):
    start, end = interval
    if len(endpoints) <= end:
        return None
    speeds = np.asarray(endpoints[start + 1:end + 1], dtype=float)
    if not np.isfinite(speeds).all():
        raise ValueError("nonfinite speed in scoring window")
    return float(np.mean(speeds))


def passes_pair(delta, threshold, original_completed, plus_completed):
    return bool(original_completed and plus_completed and delta is not None and delta >= threshold)


def check_trace(data, arm, plan, course):
    issued = np.asarray(data["issued_action"], dtype=np.float32)
    raw = np.asarray(data["raw_cpp_action"], dtype=np.float32)
    applied = np.asarray(data["applied_action"], dtype=np.float32)
    clipped = np.asarray(data["wheel_offset_clipped"], dtype=bool)
    if (course["action_delay_ticks"] != 1 or not np.array_equal(applied[1:], issued[:-1])
            or not np.array_equal(issued[:, LEG_INDICES], raw[:, LEG_INDICES])
            or not np.array_equal(np.asarray(data["wheel_target_rad_s"]),
                                  plan["wheel_scale_rad_s"] * applied[:, WHEELS].astype(float))):
        raise ValueError("recorded FIFO, leg actions or applied wheel targets disagree")
    for tick in range(len(raw) - 1):
        expected, expected_clip = apply_intervention(raw[tick + 1], tick, arm, plan)
        if (not np.array_equal(issued[tick + 1], expected)
                or not np.array_equal(clipped[tick + 1], expected_clip)):
            raise ValueError(f"registered issue schedule differs at decision {tick}")
    return int(np.count_nonzero(clipped))


def episode(seed, arm, plan, parent_plan, binary):
    from tools87.terrain87 import make_env

    folder = OUTPUT / f"{arm}_{seed}"
    folder.mkdir()
    course = parent_plan["course"]
    data, complete = {}, False
    env = make_env(plan["terrain"], seed, perception=False, max_transitions=course["ticks"])
    try:
        with cpp_actor(folder, parent_plan, binary) as predict:
            obs, info = env.reset(seed=seed)
            if (info["delay_ticks"] != course["action_delay_ticks"]
                    or not np.all(env.action_scales[WHEELS] == plan["wheel_scale_rad_s"])):
                raise ValueError("wheel scale or action FIFO differs at reset")
            zero = np.zeros(16, np.float32)
            snapshot(env, data, 0., 0., zero, zero, {}, 0.)
            data["raw_cpp_action"] = [zero]
            data["wheel_offset_clipped"] = [np.zeros(4, bool)]
            for tick in range(course["ticks"]):
                if obs.shape != (643,) or np.any(obs[ACTOR_DIM:]):
                    raise ValueError("unexpected baseline observation")
                raw = np.asarray(predict(obs[:ACTOR_DIM], obs[ACTOR_DIM:]), dtype=np.float32)
                issued, clipped = apply_intervention(raw, tick, arm, plan)
                ranges = env.last_raw_features.copy()
                obs, _, terminated, truncated, info = env.step(issued)
                snapshot(env, data, info["body_vx_m_s"], info["command_during_interval"][0],
                         issued, ranges, info, 0.)
                data["raw_cpp_action"].append(raw.copy())
                data["wheel_offset_clipped"].append(clipped)
                if terminated or truncated:
                    break
        metrics = score(data, course, plan["independent_course_mark_m_past_toe"])
        clipping = check_trace(data, arm, plan, course)
        ticks, counts = metrics["ticks"], env.counters()
        if ((counts["policy_transitions"], counts["pd_updates"], counts["native_steps"])
                != (ticks, ticks * 5, ticks * 10) or not (metrics["full_horizon"] or metrics["terminated"])):
            raise ValueError("incomplete physical record or incorrect native counters")
        window = plan["primary"]["interval_end_ticks_exclusive_inclusive"]
        metrics.update(seed=seed, arm=arm, pd_updates=counts["pd_updates"], native_steps=counts["native_steps"],
                       score_vx_m_s=mean_score_speed(data["body_vx_m_s"], window),
                       wheel_offsets_clipped=clipping)
        complete = True
    finally:
        try:
            if data:
                trace = folder / ("trace.npz" if complete else "partial_trace.npz")
                with trace.open("xb") as stream:
                    np.savez_compressed(stream, **data)
        finally:
            env.close()
    metrics["trace_sha256"] = sha(trace)
    write(folder / "metrics.json", metrics)
    print(json.dumps({k: metrics[k] for k in ("seed", "arm", "ticks", "score_vx_m_s", "reached_mark")}),
          flush=True)
    return metrics


def pair_result(root, original, plus, plan, course):
    seed = original["seed"]
    if seed != plus["seed"] or original["arm"] != "original" or plus["arm"] != "wheel_plus":
        raise ValueError("invalid pair")
    start = plan["issue_start_tick"]
    paths = [root / f"{m['arm']}_{seed}" / "trace.npz" for m in (original, plus)]
    if any(sha(p) != m["trace_sha256"] for p, m in zip(paths, (original, plus), strict=True)):
        raise ValueError("paired trace identity differs")
    with np.load(paths[0]) as a, np.load(paths[1]) as b:
        # Row 181 has integrated up to 3.62 s, using the previously issued action.
        state_stop = min(start + 1 + course["action_delay_ticks"], len(a["qpos"]), len(b["qpos"]))
        issue_stop = min(start + 1, state_stop)
        if (any(not np.array_equal(a[k][:state_stop], b[k][:state_stop]) for k in ("qpos", "qvel", "applied_action"))
                or not np.array_equal(a["issued_action"][:issue_stop], b["issued_action"][:issue_stop])
                or not np.array_equal(a["raw_cpp_action"][:state_stop], b["raw_cpp_action"][:state_stop])):
            raise ValueError("pair diverged during its matched pre-intervention prefix")
        difference = np.flatnonzero(np.any(a["applied_action"][:min(len(a["qpos"]), len(b["qpos"]))] !=
                                          b["applied_action"][:min(len(a["qpos"]), len(b["qpos"]))], axis=1))
        first_applied = (int(difference[0]) - 1) / course["policy_hz"] if difference.size else None
    a_vx, b_vx = original["score_vx_m_s"], plus["score_vx_m_s"]
    delta = None if a_vx is None or b_vx is None else b_vx - a_vx
    return {"seed": seed, "original_vx_m_s": a_vx, "wheel_plus_vx_m_s": b_vx, "delta_vx_m_s": delta,
            "identical_state_through_s": (state_stop - 1) / course["policy_hz"],
            "first_changed_applied_interval_start_s": first_applied,
            "passes": passes_pair(delta, plan["primary"]["per_pair_min_delta_mean_body_vx_m_s"],
                                  original["course_completed"], plus["course_completed"])}


def main():
    from tools27.resource_guard27 import read_memory

    scope = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[-1]
    available = read_memory(scope)["effective_available_bytes"]
    if (available < 4 << 30 or os.getpriority(os.PRIO_PROCESS, 0) != 10
            or len(os.sched_getaffinity(0)) != 1 or not scope.endswith("/d1-uphill88-probe.service")):
        raise RuntimeError("launch via the registered guarded probe service")
    OUTPUT.mkdir(exist_ok=False)
    receipt = {"status": "INCOMPLETE", "protocol_sha256": sha(PLAN), "source_sha256": sha(__file__),
               "cases": [], "pairs": [], "training_transitions": 0, "started_effective_bytes": available}
    try:
        plan = read(PLAN)
        parent = ROOT / plan["parent_protocol"]
        if sha(parent) != plan["parent_protocol_sha256"] or list(WHEELS) != plan["wheel_action_rows"]:
            raise ValueError("registered controller or wheel indices differ")
        parent_plan = read(parent)
        binary = parent_inputs(parent_plan)
        for seed in plan["reset_seeds"]:
            for arm in ("original", "wheel_plus"):
                receipt["cases"].append(episode(seed, arm, plan, parent_plan, binary))
            receipt["pairs"].append(pair_result(OUTPUT, *receipt["cases"][-2:], plan, parent_plan["course"]))
        cases = receipt["cases"]
        receipt.update(policy_transitions=sum(c["ticks"] for c in cases),
                       pd_updates=sum(c["pd_updates"] for c in cases),
                       native_steps=sum(c["native_steps"] for c in cases))
        if (len(cases) != plan["max_episodes"] or receipt["policy_transitions"] > plan["max_policy_transitions"]
                or receipt["native_steps"] > plan["max_native_steps"]):
            raise ValueError("physical budget differs")
        receipt.update(status="COMPLETE", primary_pass=all(p["passes"] for p in receipt["pairs"]))
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write(OUTPUT / "report.json", receipt)


if __name__ == "__main__":
    main()
