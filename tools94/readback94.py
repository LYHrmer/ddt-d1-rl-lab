"""Read back raw native endpoints, model predictions and resource receipts for publication."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools94.common94 import BASE, PLAN, SEEDS, arrays, clocks, digest, read, seed_dir, tag, write
from tools94.score94 import OFFSETS, calibration, control


def checked(path, sha):
    path = Path(path)
    if digest(path) != sha:
        raise ValueError(f"archive differs from its receipt: {path}")
    return path


def guard(stage):
    directory = BASE / f"guard_{stage}_01"
    record = read(directory / "report.json")
    samples = [json.loads(line) for line in (directory / "resources.jsonl").read_text().splitlines()]
    start = [r["effective_available_bytes"] for r in samples if r["stage"] == "startup"][-1]
    runtime = [r["effective_available_bytes"] for r in samples if r["stage"] == "runtime"]
    if record["status"] == "COMPLETE" and (start < 4 << 30 or min(runtime) < 2 << 30
            or not record["owned_cgroup_children_empty"] or len(record["cpu_affinity"]) != 1):
        raise ValueError("violated guarded run")
    return {"receipt_sha256": digest(directory / "report.json"), "startup_effective_bytes": start,
            "runtime_min_effective_bytes": min(runtime) if runtime else None,
            **{k: record.get(k) for k in ("status", "elapsed_s", "cpu_affinity", "peak_group_bytes",
                                          "final_memory_events", "owned_cgroup_children_empty", "error")}}


def load_native(ref, expected, reset):
    path = checked(ref["path"], ref["sha256"])
    with np.load(path, allow_pickle=False) as t:
        x = t["state45"][:, 0].copy()
        n = len(x)-1
        clean = n == expected and all(not t[k].any() for k in ("fallen", "illegal", "terminated", "truncated"))
        if int(t["reset_seed"]) != reset or (int(t["policy_steps"]), int(t["servo_steps"]),
                 int(t["native_steps"])) != (n, 5*n, 10*n):
            raise ValueError("inconsistent native clocks or reset")
        if not np.array_equal(t["applied16"][1:], t["issued16"][:-1]) or np.any(t["applied16"][0]):
            raise ValueError("invalid physical action FIFO")
        counts = {k: int(t[k].any()) for k in ("fallen", "illegal", "terminated", "truncated")}
        counts["target_saturation_ticks"] = int(np.count_nonzero(t["wheel_saturation_count"]))
        return x, clean, counts


def video_pair(seed, reset, r, kind, left, right, status):
    label = {"STOPPED_AT_CALIBRATION": "轮地预测未过门", "STOPPED_AT_POLICY_CHECK": "策略核验未过门",
             "EVALUATED": "独立起点评测"}.get(status, "未完成")
    return dict(seed=seed, reset=reset, round=r, kind=kind, left=left["path"], right=right["path"],
                left_sha256=left["sha256"], right_sha256=right["sha256"], status=status, status_label=label)


def run(out):
    out.mkdir()
    public = {"experiment": "D1 cold-start standing", "protocol_sha256": digest(PLAN),
              "totals": clocks(), "seeds": [], "calibration": [], "evaluation": []}
    curves, pairs = {}, []
    for seed in SEEDS:
        folder = seed_dir(seed)
        saved = read(folder / "report.json")
        if saved["seed"] != seed or saved["protocol_sha256"] != public["protocol_sha256"]:
            raise ValueError("unexpected seed protocol")
        item = {k: saved.get(k) for k in ("seed", "status", "totals", "initial", "error", "failure_receipt")}
        item.update(guard=guard(f"seed{seed}"), raw_report_sha256=digest(folder / "report.json"), rounds=[])
        public["seeds"].append(item)
        actual_ticks = 0
        selected = []
        for row in saved["rounds"]:
            group = {k: row.get(k) for k in ("round", "collector_weights_sha256", "status",
                                              "world_checkpoint", "policy_checkpoint", "policy_update")}
            group["fit_episodes"] = []
            if "world_fit" in row:
                fit = row["world_fit"]
                group["world_fit"] = {k:v for k,v in fit.items() if k != "losses"}
                group["first_world_loss"] = fit["losses"][0]["loss"]
                group["last_world_loss"] = fit["losses"][-1]["loss"]
                checked(folder / f"round{row['round']}" / row["world_checkpoint"]["file"], row["world_checkpoint"]["sha256"])
            for train in row["fit"]:
                x, clean, flags = load_native(train, 50, train["reset"])
                actual_ticks += len(x)-1
                group["fit_episodes"].append({**{k:v for k,v in train.items() if k != "path"},
                                              "native_events": flags, "full_clean_window": clean})
            item["rounds"].append(group)
            selected = []
            for cell in row["calibration"]:
                reset, r = cell["reset"], row["round"]
                prefix = f"s{seed}_r{r}_c{reset}"
                real, predicted, safe = {}, {}, {}
                cell_events = {}
                for k in OFFSETS:
                    name = str(k)
                    x, safe[k], cell_events[name] = load_native(cell["native"][name], 50, reset)
                    actual_ticks += len(x)-1
                    real[k] = curves[prefix + "_native_" + tag(k)] = x
                    ref = cell["models"][name]
                    with np.load(checked(ref["path"], ref["sha256"]), allow_pickle=False) as saved_states:
                        predicted[k] = curves[prefix + "_model_" + tag(k)] = saved_states["state45"][:,0].copy()
                score = calibration(real, predicted, cell["score"]["model_derivative"], safe)
                if score != cell["score"]:
                    raise ValueError(f"calibration score differs at {prefix}")
                public["calibration"].append(dict(seed=seed, round=r, reset=reset, score=score,
                    endpoints_prefix=prefix, native_events=cell_events, model_exceptions=cell["failures"],
                    source_hashes={kind:{k:rec["sha256"] for k,rec in cell[kind].items()} for kind in ("models", "native")}))
                selected.append(video_pair(seed,reset,r,"calibration",cell["native"]["-20"],cell["native"]["20"],saved["status"]))
        if saved["evaluation"]:
            selected = []
            for pair in saved["evaluation"]:
                reset = pair["reset"]
                prefix = f"s{seed}_eval{reset}"
                a, ac, af = load_native(pair["initial"],100,reset)
                b, bc, bf = load_native(pair["final"],100,reset)
                actual_ticks += len(a)+len(b)-2
                curves[prefix+"_initial"], curves[prefix+"_final"] = a, b
                score = control(a, b, ac and bc)
                if score != pair["score"]:
                    raise ValueError("control score differs")
                public["evaluation"].append(dict(seed=seed, reset=reset, endpoints_prefix=prefix, score=score,
                    native_events={"initial": af, "final": bf},
                    source_hashes={arm: pair[arm]["sha256"] for arm in ("initial", "final")}))
                selected.append(video_pair(seed,reset,2,"policy",pair["initial"],pair["final"],saved["status"]))
        if saved["status"] != "PARTIAL" and actual_ticks != saved["totals"]["native_decisions"]:
            raise ValueError("inconsistent native step budget")
        pairs.extend(selected)
        for key in public["totals"]:
            public["totals"][key] += saved["totals"][key]
    budget = read(PLAN)["budget"]
    if any(n > budget[k] for k,n in public["totals"].items()):
        raise ValueError("cold-start budget exceeded")
    public["calibration_cells_passed"] = sum(c["score"]["passed"] for c in public["calibration"])
    public["control_pairs_passed"] = sum(c["score"]["passed"] for c in public["evaluation"])
    public["control_qualified"] = len(public["evaluation"]) == 9 and public["control_pairs_passed"] == 9
    arrays(out / "curves.npz", curves)
    public["curves_sha256"] = digest(out / "curves.npz")
    write(out / "report.json", public)
    write(out / "video_manifest.json", {"pairs": pairs})
