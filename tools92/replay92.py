"""Replay archived applied actions through the frozen 91 wheel–terrain models."""
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from tools91.data91 import _prefix
from tools91.rollout91 import ARMS, END_TICK, START_FIELDS, START_TICK, STARTS, _fallen
from tools92.common92 import PLAN_SHA, ROOT, checked_plan, digest, read, require, write


WORLD91 = ROOT / "evidence/terrain_world91"
TRACE_ROOT = {
    "move": ROOT / "evidence/uphill88/probe_01",
    "stand": ROOT / "evidence/wbc90/stand_01",
}


def _cases(plan):
    for task, seed, parent in STARTS:
        cfg = plan["tasks"][task]
        start = START_TICK[task]
        horizon = END_TICK[task] - start
        require((cfg["reset_seeds"], cfg["start_tick"], cfg["horizon"], cfg["intervention"]) == (
            [s for t, s, _ in STARTS if t == task], start, horizon, ARMS[task][1]),
            f"registered {task} cases differ from their historical cuts")
        for arm in ARMS[task]:
            yield dict(task=task, reset_seed=seed, parent_seed=parent, arm=arm,
                       start_tick=start, horizon=horizon)


def _name(case):
    return f'{case["task"]}_{case["reset_seed"]}_{case["arm"]}'


def _identity(path, expected):
    require(path.is_file() and not path.is_symlink() and digest(path) == expected,
            f"source SHA256 mismatch: {path}")


def _action_window(task, issued, applied, fifo, start, horizon):
    # 88 includes reset as row zero; 90 row zero follows its first control interval.
    require(task in ARMS and start >= 0 and horizon > 0 and (task != "stand" or start == 0),
            "invalid action window")
    first = start + int(task == "move")
    stop = first + horizon
    require(issued.ndim == 2 and issued.shape == applied.shape and issued.shape[1] == 16
            and len(issued) >= stop and issued.dtype == np.float32 and applied.dtype == np.float64,
            "invalid archived action arrays")
    issued, applied = issued[:stop], applied[:stop]
    require(np.isfinite(issued).all() and (np.abs(issued) <= 1.).all()
            and not np.any(applied[0]) and np.array_equal(applied[1:], issued[:-1]),
            "archived one-tick FIFO mismatch")
    require(fifo.shape == (16,) and fifo.dtype == np.float32 and np.array_equal(applied[first], fifo),
            "first applied action differs from the frozen 91 branch FIFO")
    archived = applied[first:stop]
    actions = archived.astype(np.float32)
    require(np.array_equal(actions.astype(np.float64), archived), "lossy applied-action conversion")
    return actions


def _check_state(state):
    require(state.shape == (45,) and state.dtype == np.float64 and np.isfinite(state).all()
            and abs(np.linalg.norm(state[3:7]) - 1.) <= 1e-6, "invalid branch state45")


def _branch(path, case, claim):
    _identity(path, claim["start_sha256"])
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == set(START_FIELDS), "unexpected archived start fields")
        for key in ("start_tick", "reset_seed", "parent_seed"):
            value = data[key]
            require(value.shape == () and value.dtype == np.int64 and value.item() == case[key],
                    f"mismatched branch {key}")
        state, fifo = data["state45"], data["fifo16"]
    _check_state(state)
    return state, fifo


def _physical_input(path, case, state, fifo):
    task, start, horizon = case["task"], case["start_tick"], case["horizon"]
    with ZipFile(path) as archive:
        if task == "move":
            # These members are read only up through the branch endpoint.
            physical_state = np.r_[_prefix(archive, "qpos", start + 1)[-1],
                                   _prefix(archive, "qvel", start + 1)[-1]]
            issued_key, applied_key, rows = "issued_action", "applied_action", start + 1 + horizon
        else:
            with archive.open("initial_qpos.npy") as q, archive.open("initial_qvel.npy") as v:
                physical_state = np.r_[np.load(q, allow_pickle=False), np.load(v, allow_pickle=False)]
            issued_key, applied_key, rows = "issued", "applied", horizon
        require(np.array_equal(physical_state, state), "physical arm differs from frozen branch state")
        return _action_window(task, _prefix(archive, issued_key, rows),
                              _prefix(archive, applied_key, rows), fifo, start, horizon)


def _physical_receipts():
    move = read(ROOT / "reports/uphill_diagnosis88.json")["cases"]
    stand = read(ROOT / "reports/stand_wbc90.json")["stand"]["cases"]
    receipts = {}
    for task, rows, seed_field in (("move", move, "seed"), ("stand", stand, "reset_seed")):
        for row in rows:
            key = (task, row[seed_field], row["arm"])
            require(key not in receipts, "duplicate physical arm")
            if task == "move":
                require(row["full_clean"] and row["ticks"] == 300, "incomplete moving source")
            else:
                require(row["status"] == "COMPLETE" and row["clean"] and row["ticks"] == 100,
                        "incomplete standing source")
            receipts[key] = row
    return receipts


def _seal(out, report):
    path = out / "seal.json"
    write(path, report)
    path.chmod(0o444)


def prepare(out):
    """Export exactly a branch state and a fixed applied-action tape for each physical arm."""
    out = Path(out)
    out.mkdir(exist_ok=False)
    report = {"protocol_sha256": PLAN_SHA, "status": "PARTIAL", "cases": []}
    try:
        plan = checked_plan()
        old = read(ROOT / "reports/terrain_world91.json")
        starts = {row["case"]: row for row in old["starts"]}
        require(len(starts) == len(old["starts"]) == len(STARTS), "incomplete 91 branch set")
        model_rows = old["families"]["candidate"]["models"]
        require([row["seed"] for row in model_rows] == plan["model_seeds"], "incomplete 91 model set")
        report["models"] = {}
        for row in model_rows:
            receipt = row["source"]["model"]
            key = str(row["seed"])
            require(receipt["seed"] == row["seed"] and receipt["status"] == "COMPLETE"
                    and receipt["weights_sha256"] == plan["weights_sha256"][key], "unregistered model")
            report["models"][key] = receipt
        physical = _physical_receipts()
        cases = list(_cases(plan))
        expected = {(c["task"], c["reset_seed"], c["arm"]) for c in cases}
        require(set(physical) == expected and len(cases) == 12, "incomplete physical arm set")
        for case in cases:
            task, seed, arm = case["task"], case["reset_seed"], case["arm"]
            stem = f"{task}_{seed}"
            claim = starts[stem]
            source = TRACE_ROOT[task] / f"{arm}_{seed}/trace.npz"
            trace_hash = physical[task, seed, arm]["trace_sha256"]
            _identity(source, trace_hash)
            require(claim["source_trace_sha256"] == physical[task, seed, "original"]["trace_sha256"],
                    "unmatched original physical branch")
            if task == "stand":
                require(physical[task, seed, arm]["parent_seed"] == case["parent_seed"]
                        and physical[task, seed, arm]["initial_state_sha256"] == claim["initial_state_sha256"],
                        "unmatched standing reset or parent")
            state, fifo = _branch(WORLD91 / "prepare_01/starts" / (stem + ".npz"), case, claim)
            actions = _physical_input(source, case, state, fifo)
            target = out / (_name(case) + ".npz")
            with target.open("xb") as stream:
                np.savez_compressed(stream, state45=state, applied16=actions)
            target.chmod(0o444)
            report["cases"].append(dict(case, file=target.name, file_sha256=digest(target),
                                        source_trace_sha256=trace_hash, start_sha256=claim["start_sha256"]))
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _seal(out, report)
    return report


def _read_input(path, horizon):
    with np.load(path, allow_pickle=False) as data:
        require(len(data.files) == 2 and set(data.files) == {"state45", "applied16"}, "unexpected replay inputs")
        state, actions = data["state45"], data["applied16"]
    _check_state(state)
    require(actions.shape == (horizon, 16) and actions.dtype == np.float32 and np.isfinite(actions).all()
            and (np.abs(actions) <= 1.).all(), "invalid fixed applied actions")
    return state, actions


def _roll_case(out, case, model, state, actions, limits):
    """Advance only the world model; the current input row is already the applied action."""
    import torch

    folder = out / f'seed{case["model_seed"]}' / _name(case)
    folder.mkdir(parents=True, exist_ok=False)
    terrain = torch.tensor([1], dtype=torch.int64, device="cpu")
    states, rejected = [state.copy()], []
    attempted = 0
    status, reason = "COMPLETE", None
    try:
        with torch.no_grad():
            for index, action in enumerate(actions):
                endpoint = case["start_tick"] + index + 1
                attempted += 1
                try:
                    result = model.step(torch.tensor(states[-1][None], dtype=torch.float32, device="cpu"),
                                        torch.from_numpy(action[None].copy()), terrain)
                except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                    status, reason = "MODEL_ERROR", f"{type(exc).__name__}: {exc}"
                    break
                require(torch.is_tensor(result) and result.dtype == torch.float32 and result.device.type == "cpu"
                        and result.shape == (1, 45), "expected one CPU float32 world-model state [1,45]")
                following = result[0].detach().numpy().astype(np.float64)
                if not np.isfinite(following).all():
                    status, reason = "NONFINITE", f"nonfinite state at endpoint {endpoint}"
                    rejected.append(following)
                    break
                if abs(np.linalg.norm(following[3:7]) - 1.) > 1e-4:
                    status, reason = "DIVERGED", f"invalid unit quaternion at endpoint {endpoint}"
                    rejected.append(following)
                    break
                states.append(following)
                if _fallen(following, *limits):
                    status, reason = "PREDICTED_FALL", f"predicted root height or tilt at endpoint {endpoint}"
                    break
    except BaseException as exc:
        status, reason = "PARTIAL", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trace = folder / ("trace.npz" if status == "COMPLETE" else "partial_trace.npz")
        with trace.open("xb") as stream:
            np.savez_compressed(stream, endpoint_tick=case["start_tick"] + np.arange(len(states), dtype=np.int64),
                                state45=np.asarray(states, dtype=np.float64), applied16=actions[:attempted],
                                rejected_state45=np.asarray(rejected, dtype=np.float64).reshape(-1, 45))
        trace.chmod(0o444)
        row = dict(case, status=status, reason=reason, attempted_steps=attempted, accepted_steps=len(states) - 1,
                   trace=str(trace.relative_to(out)), trace_sha256=digest(trace))
        write(folder / "case.json", row)
    return row


def _limits():
    cfg = read(ROOT / "training/configs/pilot03d.json")
    pose = read(ROOT / "training/configs/diagnostic03b_zero.json")["static_derivation"]
    require(pose["model_xml_sha256"] == cfg["robot_xml_sha256"], "nominal pose has wrong robot identity")
    return (pose["nominal_root_height_m"] - cfg["termination"]["base_height_below_nominal_by_m"],
            cfg["termination"]["tilt_above_rad"])


def predict(out, inputs_dir):
    """Roll all three frozen fits on the sealed tapes, without feedback or future state."""
    import torch
    from tools91.model91 import load_new_models

    out, inputs_dir = Path(out), Path(inputs_dir)
    out.mkdir(exist_ok=False)
    report = {"protocol_sha256": PLAN_SHA, "status": "PARTIAL", "cases": [],
              "physics_steps": 0, "policy_updates": 0, "model_updates": 0, "actor_calls": 0, "sensor_calls": 0}
    try:
        plan = checked_plan()
        seal_path = inputs_dir / "seal.json"
        require(seal_path.is_file() and not seal_path.is_symlink(), "missing sealed inputs")
        report["input_seal_sha256"] = digest(seal_path)
        inputs = read(seal_path)
        require(inputs["status"] == "COMPLETE" and inputs["protocol_sha256"] == PLAN_SHA, "unqualified inputs")
        cases = list(_cases(plan))
        require(len(inputs["cases"]) == len(cases) == 12, "incomplete prepared cases")
        tapes = []
        for case, item in zip(cases, inputs["cases"], strict=True):
            require(all(case[k] == item[k] for k in case) and item["file"] == _name(case) + ".npz",
                    "prepared case identity or order differs")
            path = inputs_dir / item["file"]
            _identity(path, item["file_sha256"])
            state, actions = _read_input(path, case["horizon"])
            tapes.append((dict(case, input_sha256=item["file_sha256"]), state, actions))
        seeds = plan["model_seeds"]
        require(seeds == [91011, 91023, 91037] and len(seeds) * sum(c["horizon"] for c in cases) == 3060,
                "incorrect replay budget")
        model_dir = WORLD91 / "fit_01/models"
        for seed in seeds:
            _identity(model_dir / f"seed{seed}/final_state_dict.pt", plan["weights_sha256"][str(seed)])
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        models = load_new_models(model_dir)
        require([s for s, _, _ in models] == seeds, "incomplete frozen family")
        limits = _limits()
        report.update(model_seeds=seeds, weights_sha256=plan["weights_sha256"])
        for seed, model, receipt in models:
            require(receipt == inputs["models"][str(seed)] and receipt["weights_sha256"] == plan["weights_sha256"][str(seed)]
                    and not model.training and all(not p.requires_grad and p.device.type == "cpu" for p in model.parameters()),
                    "different model identity, device or training state")
            for case, state, actions in tapes:
                entry = dict(case, model_seed=seed)
                try:
                    row = _roll_case(out, entry, model, state, actions, limits)
                except BaseException:
                    saved = out / f"seed{seed}" / _name(case) / "case.json"
                    if saved.is_file():
                        report["cases"].append(read(saved))
                    raise
                report["cases"].append(row)
        require(len(report["cases"]) == len(cases) * len(seeds), "incomplete replay records")
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["attempted_steps"] = sum(r["attempted_steps"] for r in report["cases"])
        report["accepted_steps"] = sum(r["accepted_steps"] for r in report["cases"])
        _seal(out, report)
    return report
