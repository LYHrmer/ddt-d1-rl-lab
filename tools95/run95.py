"""Seal three whole-policy directions, predict the five arms, then measure them once."""
from pathlib import Path

import numpy as np

from tools94.common94 import ROOT, arrays, clocks, digest, read, write
from tools95.launch95 import BASE, PLAN, PLAN_SHA
from tools95.score95 import ARMS, HORIZON, RESETS, SEEDS, reward_m


def sources():
    return {name: digest(ROOT / "tools95" / name) for name in
            ("launch95.py", "run95.py", "model95.py", "score95.py")}


def files_for_seal(folder):
    return {str(path.relative_to(folder)): digest(path) for path in sorted(folder.rglob("*"))
            if path.is_file() and path != folder / "seal.json"}


def checked(stage):
    folder = BASE / f"{stage}_01"
    seal = read(folder / "seal.json")
    if (seal["protocol_sha256"] != PLAN_SHA or seal["sources"] != sources()
            or seal["files"] != files_for_seal(folder)):
        raise ValueError(f"{stage} protocol, producer or sealed files changed")
    return folder


def construct_all(folder):
    from tools95.model95 import construct
    counts = []
    for seed in SEEDS:
        receipt = construct(seed, folder / f"seed_{seed}")
        counts.append({k: receipt[k] for k in ("seed", "status", "attempted_model_steps",
                      "completed_model_steps", "backward_attempted", "backward_completed",
                      "shadow_adam_attempted", "shadow_adam_completed")})
        print("direction", seed, receipt["status"], receipt.get("error", ""), flush=True)
    return {"seeds": counts}


def save_starts(folder):
    from tools94.native94 import _endpoint, _open_env, _reset_uphill
    starts = {}
    for reset in RESETS:
        env = _open_env(reset, HORIZON)
        try:
            observation = _reset_uphill(env, reset)
            row = _endpoint(env, observation, 0)
            if any(env.counters()[k] for k in ("policy_transitions", "pd_updates", "native_steps")):
                raise ValueError("holdout reset consumed an integration")
        finally:
            env.close()
        starts[reset] = {k: np.expand_dims(v, 0) for k, v in row.items()}
        arrays(folder / f"start_{reset}.npz", starts[reset])
    return starts


def predict_all(folder):
    from tools95.model95 import load_sealed, predict_arm
    directions = checked("directions")
    starts = save_starts(folder)
    totals = dict(attempted_model_steps=0, completed_model_steps=0, backward_attempted=0, backward_completed=0)
    for seed in SEEDS:
        parent = directions / f"seed_{seed}"
        if read(parent / "receipt.json")["status"] != "COMPLETE":
            continue
        bundle = load_sealed(seed, parent)
        for reset in RESETS:
            dest = folder / f"seed_{seed}/reset_{reset}"
            dest.mkdir(parents=True)
            for key, alpha in ARMS.items():
                row = predict_arm(bundle, starts[reset], alpha)
                states = np.asarray(row.pop("state45"), dtype=np.float32).reshape(-1, 45)
                path = dest / f"{key}.npz"
                arrays(path, {"state45": states})
                write(dest / f"{key}.json", {**row, "alpha": alpha, "file": path.name,
                                             "state_sha256": digest(path), "reset": reset, "seed": seed})
                for name in totals:
                    totals[name] += row[name]
            print("predicted", seed, reset, flush=True)
    return {"totals": totals, "holdout_reset_previews": len(starts), "preview_physics_substeps": 0}


def native_arm(folder, actor, reset, totals, start, first_endpoint):
    from tools94.native94 import NativeEpisodeError
    from tools94.run94 import measured_summary, physical, safe
    folder.mkdir()
    error = None
    try:
        trace, receipt = physical(folder / "native", actor, reset, totals, ticks=HORIZON)
        path = Path(receipt["path"])
    except NativeEpisodeError as exc:
        trace, error = exc.trace, str(exc)
        path = folder / "native/partial_trace.npz"
    row = {"safe": False, "return_m": None, "trace_file": None, "trace_sha256": None, "error": error}
    if trace is not None and path.is_file():
        row.update(measured_summary(trace), trace_file=str(path.relative_to(folder)), trace_sha256=digest(path))
        if len(trace["state45"]):
            for name in start:
                if not np.array_equal(trace[name][0], start[name][0]):
                    raise ValueError(f"holdout reset differs from sealed model observation in {name}")
            row["return_m"] = reward_m(trace["state45"][:, 0], start["state45"][0, 0])
        if len(trace["applied16"]):
            if not np.array_equal(trace["applied16"][0], np.zeros(16)) or trace["exploration16"].any():
                raise ValueError("unexpected first applied action or exploration noise")
            if first_endpoint is not None and not np.array_equal(trace["state45"][1], first_endpoint):
                raise ValueError("different first native endpoint under the shared one-tick FIFO")
            first_endpoint = trace["state45"][1].copy()
        row["safe"] = error is None and safe(trace, HORIZON)
    write(folder / "native.json", row)
    return first_endpoint


def measure_all(folder):
    from tools95.model95 import load_sealed, variant
    directions, predictions = checked("directions"), checked("predictions")
    totals = clocks()
    arms = 0
    for seed in SEEDS:
        parent = directions / f"seed_{seed}"
        if read(parent / "receipt.json")["status"] != "COMPLETE":
            continue
        bundle = load_sealed(seed, parent)
        for reset in RESETS:
            with np.load(predictions / f"start_{reset}.npz", allow_pickle=False) as archive:
                start = dict(archive)
            dest = folder / f"seed_{seed}/reset_{reset}"
            dest.mkdir(parents=True)
            first = None
            for key, alpha in ARMS.items():
                first = native_arm(dest / key, variant(bundle, alpha), reset, totals, start, first)
                arms += 1
            print("measured", seed, reset, totals["native_decisions"], flush=True)
    return {"totals": totals, "native_episodes": arms}


def run_stage(stage, folder):
    work = {"directions": construct_all, "predictions": predict_all, "native": measure_all}
    write(folder / "start.json", {"stage": stage, "protocol_sha256": digest(PLAN), "sources": sources()})
    result = work[stage](folder)
    seal = {**result, "protocol_sha256": PLAN_SHA, "sources": sources(), "files": files_for_seal(folder)}
    write(folder / "seal.json", seal)
