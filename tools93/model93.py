"""Fixed-action inference on the six new parent/reset cells of the standing panel."""
import hashlib
from pathlib import Path

import numpy as np

from tools92.replay92 import _action_window, _check_state, _identity, _limits, _name, _read_input, _roll_case, _seal
from tools93.common93 import MODELS, PARENTS, PLAN_SHA, RESETS, ROOT, checked_plan, digest, read, require


ARMS = ("original", "common")
HORIZON = 50
BASELINE90 = ROOT / "reports/stand_wbc90.json"
WEIGHTS91 = ROOT / "evidence/terrain_world91/fit_01/models"


def _new_keys(plan):
    require((tuple(plan["parents"]), tuple(plan["resets"]), tuple(plan["model_seeds"]),
             tuple(plan["arms"]), plan["horizon"]) == (tuple(PARENTS), tuple(RESETS), tuple(MODELS), ARMS, HORIZON),
            "registered grid or timing changed")
    require(len(set(PARENTS)) == len(set(RESETS)) == len(set(MODELS)) == 3, "three distinct seeds per axis required")
    diagonal = set(zip(PARENTS, RESETS, strict=True))
    return [(p, r, arm) for p in PARENTS for r in RESETS if (p, r) not in diagonal for arm in ARMS]


def _index(cases, keys):
    result = {(c["parent_seed"], c["reset_seed"], c["arm"]): c for c in cases}
    require(len(result) == len(cases) and result.keys() == set(keys),
            "incomplete, duplicate or unregistered physical arm inventory")
    return result


def _file(key):
    parent, reset, arm = key
    return f"parent{parent}_reset{reset}_{arm}.npz"


def _source_file(folder, relative, sha):
    path = folder / relative
    require(not Path(relative).is_absolute() and path.resolve().is_relative_to(folder.resolve()),
            "trace outside the registered stage")
    _identity(path, sha)
    return path


def _initial_fingerprints(plan):
    source = read(BASELINE90)["stand"]["cases"]
    diagonal = [(p, r, arm) for p, r in zip(PARENTS, RESETS, strict=True) for arm in ARMS]
    by_key = _index(source, diagonal)
    fingerprints = {}
    for (parent, reset, _), case in by_key.items():
        require(case["status"] == "COMPLETE" and case["clean"] and case["ticks"] == 100
                and case["policy_sha256"] == plan["policies_sha256"][str(parent)]
                and fingerprints.setdefault(reset, case["initial_state_sha256"]) == case["initial_state_sha256"],
                "inconsistent archived standing branch")
    return fingerprints


def _extract(source, expected_initial):
    """Read the reset and the two action arrays only; physical endpoints stay with the scorer."""
    with np.load(source, allow_pickle=False) as raw:
        q, v, obs = raw["initial_qpos"], raw["initial_qvel"], raw["initial_obs"]
        require(q.shape == (23,) and v.shape == (22,) and q.dtype == v.dtype == np.float64
                and obs.shape == (643,) and obs.dtype == np.float32 and np.isfinite(obs).all()
                and q[0] == 1.10 and not np.any(v), "invalid slope reset")
        initial = hashlib.sha256(q.tobytes() + v.tobytes() + obs.tobytes()).hexdigest()
        require(initial == expected_initial, "reset differs from the matching archived diagonal")
        state = np.concatenate((q, v))
        _check_state(state)
        issued, applied = raw["issued"], raw["applied"]
        require(len(issued) == len(applied) == HORIZON, "physical arm must contain exactly 50 intervals")
        actions = _action_window("stand", issued, applied, np.zeros(16, np.float32), 0, HORIZON)
    return state, actions


def prepare(out, physical_dir):
    out, physical_dir = Path(out), Path(physical_dir)
    out.mkdir(exist_ok=False)
    report = dict(protocol_sha256=PLAN_SHA, status="PARTIAL", cases=[])
    try:
        plan = checked_plan()
        keys = _new_keys(plan)
        seal_path = physical_dir / "seal.json"
        require(seal_path.is_file() and not seal_path.is_symlink(), "missing physical seal")
        physical = read(seal_path)
        report["physical_seal_sha256"] = digest(seal_path)
        require(physical["protocol_sha256"] == PLAN_SHA and physical["status"] == "COMPLETE", "partial physical stage")
        rows = _index(physical["cases"], keys)
        initial = _initial_fingerprints(plan)
        for key in keys:
            parent, reset, arm = key
            row = rows[key]
            require(row["status"] == "COMPLETE" and row["clean"] and row["ticks"] == HORIZON
                    and row["policy_sha256"] == plan["policies_sha256"][str(parent)]
                    and row["initial_state_sha256"] == initial[reset], "physical arm or reset is unqualified")
            trace = _source_file(physical_dir, row["trace"], row["trace_sha256"])
            state, actions = _extract(trace, initial[reset])
            target = out / _file(key)
            with target.open("xb") as stream:
                np.savez_compressed(stream, state45=state, applied16=actions)
            target.chmod(0o444)
            report["cases"].append(dict(parent_seed=parent, reset_seed=reset, arm=arm, task="stand",
                                        start_tick=0, horizon=HORIZON, file=target.name, file_sha256=digest(target),
                                        policy_sha256=row["policy_sha256"], initial_state_sha256=initial[reset],
                                        source_trace_sha256=row["trace_sha256"]))
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _seal(out, report)
    return report


def _sealed_inputs(folder, inputs, keys):
    require(inputs["protocol_sha256"] == PLAN_SHA and inputs["status"] == "COMPLETE", "partial or unregistered inputs")
    cases = _index(inputs["cases"], keys)
    tapes = []
    for key in keys:
        parent, reset, arm = key
        row = cases[key]
        require(row["file"] == _file(key) and (row["task"], row["start_tick"], row["horizon"]) == ("stand", 0, HORIZON),
                "wrong sealed input name or timing")
        path = _source_file(folder, row["file"], row["file_sha256"])
        state, actions = _read_input(path, HORIZON)
        require(not np.any(actions[0]) and not np.any(state[23:]), "standing reset action or velocity differs")
        case = dict(parent_seed=parent, reset_seed=reset, arm=arm, task="stand", start_tick=0, horizon=HORIZON,
                    input_sha256=row["file_sha256"])
        tapes.append((case, state, actions))
    return tapes


def _predict_arm(out, case, model, state, actions, limits, report):
    # 92 names each arm by model/reset/task. The extra parent axis needs its own directory.
    parent_dir = out / f'parent{case["parent_seed"]}'
    case_dir = parent_dir / f'seed{case["model_seed"]}' / _name(case)
    if case_dir.exists():
        raise FileExistsError(case_dir)
    try:
        row = _roll_case(parent_dir, case, model, state, actions, limits)
    except BaseException:
        case_file = case_dir / "case.json"
        if case_file.is_file():
            row = read(case_file)
            report["cases"].append(dict(row, trace=str(Path(parent_dir.name) / row["trace"])))
        raise
    report["cases"].append(dict(row, trace=str(Path(parent_dir.name) / row["trace"])))


def predict(out, inputs_dir):
    from tools91.model91 import load_new_models
    import torch

    out, inputs_dir = Path(out), Path(inputs_dir)
    out.mkdir(exist_ok=False)
    report = dict(protocol_sha256=PLAN_SHA, status="PARTIAL", cases=[], all_arms_complete=False,
                  model_updates=0, policy_updates=0, actor_calls=0, sensor_calls=0, physics_steps=0)
    try:
        plan = checked_plan()
        keys = _new_keys(plan)
        seal = inputs_dir / "seal.json"
        require(seal.is_file() and not seal.is_symlink(), "missing input seal")
        report["input_seal_sha256"] = digest(seal)
        tapes = _sealed_inputs(inputs_dir, read(seal), keys)
        require(len(tapes) == 12 and len(MODELS) * len(tapes) * HORIZON == 1800, "crossed standing replay budget")
        for seed in MODELS:
            _identity(WEIGHTS91 / f"seed{seed}/final_state_dict.pt", plan["weights_sha256"][str(seed)])
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        models = load_new_models(WEIGHTS91)
        require([s for s, _, _ in models] == list(MODELS), "incomplete frozen model family")
        limits = _limits()
        report.update(model_seeds=list(MODELS), weights_sha256=plan["weights_sha256"])
        for seed, model, receipt in models:
            require(receipt["weights_sha256"] == plan["weights_sha256"][str(seed)] and not model.training
                    and all(not p.requires_grad and p.device.type == "cpu" for p in model.parameters()),
                    "world model weights or inference mode differ")
            for base, state, actions in tapes:
                _predict_arm(out, dict(base, model_seed=seed), model, state, actions, limits, report)
        require(len(report["cases"]) == 36, "incomplete prediction inventory")
        report["all_arms_complete"] = all(r["status"] == "COMPLETE" for r in report["cases"])
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["attempted_steps"] = sum(r["attempted_steps"] for r in report["cases"])
        report["accepted_steps"] = sum(r["accepted_steps"] for r in report["cases"])
        _seal(out, report)
    return report
