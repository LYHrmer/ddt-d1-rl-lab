"""Advance frozen round-96 plants on the recorded native applied-action suffix."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tools91.launch91 import require
from tools94.common94 import SEEDS, arrays, digest, read, write
from tools95.score95 import ARMS
from tools96.common96 import BLIND_RESETS, MODELS, checked
from tools97.common97 import PLAN, PLAN_SHA


CUT, END = 16, 32
CLOCKS = ("attempted_model_steps", "completed_model_steps", "retained_model_steps")


def _f32_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def _sources():
    require(digest(PLAN) == PLAN_SHA, "registered suffix replay protocol changed")
    plan = read(PLAN)
    require((plan["cut_tick"], plan["end_tick"]) == (CUT, END) and
            tuple(plan["training_seeds"]) == SEEDS and tuple(plan["resets"]) == BLIND_RESETS and
            tuple(plan["models"]) == MODELS and plan["arms"] == ARMS, "registered suffix grid differs")
    files, seals = {}, {}
    for stage in ("fit", "predictions", "native"):
        folder = files[stage] = checked(stage)
        seal_path = folder / "seal.json"
        require(digest(seal_path) == plan["local_parent_seals_sha256"][stage], f"wrong frozen {stage} seal")
        seals[stage] = read(seal_path)
    for later, earlier in (("predictions", "fit"), ("native", "predictions")):
        require(seals[later]["previous_seal_sha256"] == plan["local_parent_seals_sha256"][earlier],
                "round-96 stage ordering differs")
    return files


def _recorded_native(folder, reset):
    record = read(folder / "native.json")
    require(record["error"] is None and record["trace_file"] == "native/trace.npz" and record["ticks"] == END,
            f"expected complete observed native suffix: {folder}")
    path = folder / record["trace_file"]
    require(digest(path) == record["trace_sha256"], f"native receipt differs: {folder}")
    with np.load(path, allow_pickle=False) as archive:
        require(int(archive["reset_seed"]) == reset and int(archive["requested_ticks"]) == END and
                int(archive["completed_transitions"]) == END and
                (int(archive["policy_steps"]), int(archive["servo_steps"]), int(archive["native_steps"]))
                == (END, 5*END, 10*END), f"invalid native clocks: {path}")
        states, applied, issued = (archive[k] for k in ("state45", "applied16", "issued16"))
        require(states.shape == (END + 1, 45) and states.dtype == np.float64 and np.isfinite(states).all()
                and applied.shape == issued.shape == (END, 16) and applied.dtype == np.float32 and
                np.isfinite(applied).all() and not applied[0].any() and np.array_equal(applied[1:], issued[:-1])
                and np.isfinite(issued).all() and np.all(np.abs(issued) <= 1.), f"invalid native state or FIFO: {path}")
        for key in ("fallen", "illegal", "terminated", "truncated"):
            flags = archive[key]
            require(flags.dtype == bool and flags.shape == (END,) and not flags[:-1].any() and
                    bool(flags[-1]) == record[key], f"native {key} event outside its observed end: {path}")
        require(record["safe"] == (not any(record[k] for k in ("fallen", "illegal", "terminated", "truncated"))),
                f"native safety flags differ: {path}")
        return (states[0].astype(np.float32), states[CUT].astype(np.float32),
                applied[CUT:END].copy(), record["trace_sha256"])


def _forecast_cut(folder, seed, reset, model_name, tag, reset_state):
    record = read(folder / f"{tag}.json")
    require((record["seed"], record["reset"], record["model"], record["alpha"], record["file"])
            == (seed, reset, model_name, ARMS[tag], f"{tag}.npz") and record["ticks"] == END and
            record["safe"] and not record.get("error"), f"invalid round-96 forecast: {folder}/{tag}")
    path = folder / record["file"]
    require(digest(path) == record["state_sha256"], f"forecast receipt differs: {path}")
    with np.load(path, allow_pickle=False) as archive:
        states = archive["state45"]
    require(states.shape == (END + 1, 45) and states.dtype == np.float32 and np.isfinite(states).all()
            and np.array_equal(states[0], reset_state), f"invalid forecast shape or reset: {path}")
    return states[CUT].copy(), record["state_sha256"]


def _state_check(state, limits):
    import torch

    require(isinstance(state, torch.Tensor) and state.device.type == "cpu" and state.dtype == torch.float32 and
            state.shape == (1, 45) and bool(torch.isfinite(state).all()), "invalid or nonfinite model endpoint")
    norm = torch.linalg.vector_norm(state[:, 3:7], dim=1)
    require(bool(((norm - 1.).abs() <= 1e-4).all()), "invalid predicted quaternion")
    upright = 1. - 2. * (state[:, 4].square() + state[:, 5].square())
    require(bool(((state[:, 2] >= limits[0]) & (upright >= limits[1])).all()), "predicted fall")


def _roll(world, start, tape, limits):
    """Only accepted endpoints enter state45. An attempted action stays on record even on failure."""
    import torch

    require(start.shape == (45,) and tape.shape == (END-CUT, 16) and
            start.dtype == tape.dtype == np.float32 and np.isfinite(tape).all() and np.all(np.abs(tape) <= 1.),
            "wrong replay cut or action tape")
    attempted = completed = 0
    error = rejected = None
    accepted = [start.copy()]
    with torch.inference_mode():
        state = torch.from_numpy(start.copy()).unsqueeze(0)
        _state_check(state, limits)
        terrain = torch.ones(1, dtype=torch.long)
        actions = torch.from_numpy(tape.copy())
        for action in actions:
            proposed = None
            try:
                attempted += 1
                proposed = world.step(state, action.unsqueeze(0), terrain)
                completed += 1
                _state_check(proposed, limits)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if isinstance(proposed, torch.Tensor) and proposed.shape == (1, 45):
                    rejected = proposed[0].detach().cpu().numpy().copy()
                break
            state = proposed
            accepted.append(state[0].numpy().copy())
    ticks = len(accepted) - 1
    receipt = {"status": "COMPLETE" if ticks == END - CUT else "PARTIAL", "ticks": ticks, "error": error,
               "attempted_model_steps": attempted, "completed_model_steps": completed, "retained_model_steps": ticks,
               "rejected_endpoint_tick": CUT + attempted if error is not None else None}
    data = {"state45": np.stack(accepted), "applied16": tape[:attempted].copy()}
    if rejected is not None:
        data["rejected_state45"] = rejected
    return data, receipt


def run(stage, folder):
    """Execute one exclusive 90-arm stage. Source/infrastructure errors stop the entire attempt."""
    from tools94.learn94 import _fall_limits
    from tools96.fit96 import load_fit

    require(stage in ("fixed", "anchored"), "unknown suffix replay stage")
    folder = Path(folder)
    require(folder.is_dir(), "stage directory must already exist")
    if any(path.name != "start.json" for path in folder.iterdir()):
        raise FileExistsError(f"preserve existing stage output: {folder}")
    write(folder / "replay_started.json", {"stage": stage, "protocol_sha256": PLAN_SHA})
    totals = dict(arms=0, complete_arms=0, partial_arms=0, **dict.fromkeys(CLOCKS, 0))
    totals.update(native_decisions=0, new_native_decisions=0, new_servo_updates=0,
                  new_physics_substeps=0, new_actor_calls=0, backward_passes=0,
                  world_updates=0, policy_updates=0, ppo_updates=0)
    location = {}
    try:
        files = _sources()
        budget = read(PLAN)["budget"]
        limits = _fall_limits()
        for seed in SEEDS:
            seed_path = f"seed_{seed}"
            parent = files["fit"] / seed_path
            fit = read(parent / "receipt.json")
            require(fit["status"] == "COMPLETE", "round-96 fit comparison was incomplete")
            for model_name in MODELS:
                location = {"seed": seed, "model": model_name}
                world = load_fit(seed, parent, model_name)
                require(not world.training and not any(p.requires_grad for p in world.parameters()), "world is not frozen")
                checkpoint = fit["arms"][model_name]["checkpoint"]["sha256"]
                for reset in BLIND_RESETS:
                    cell = Path(seed_path) / f"reset_{reset}"
                    dest = folder / cell / model_name
                    dest.mkdir(parents=True)
                    for tag, alpha in ARMS.items():
                        location = {"seed": seed, "reset": reset, "model": model_name, "tag": tag}
                        reset_state, native_cut, tape, native_sha = _recorded_native(files["native"] / cell / tag, reset)
                        forecast_cut, forecast_sha = _forecast_cut(files["predictions"] / cell / model_name,
                                                                  seed, reset, model_name, tag, reset_state)
                        start = forecast_cut if stage == "fixed" else native_cut
                        require(totals["arms"] < budget["arms_per_stage"] and
                                totals["attempted_model_steps"] + END-CUT <= budget["max_model_steps_per_stage"],
                                "suffix stage budget exhausted")
                        data, row = _roll(world, start, tape, limits)
                        totals["arms"] += 1
                        totals["complete_arms" if row["status"] == "COMPLETE" else "partial_arms"] += 1
                        for counter in CLOCKS:
                            totals[counter] += row[counter]
                        state_file = dest / f"{tag}.npz"
                        arrays(state_file, data)
                        write(dest / f"{tag}.json", {**location, **row, "stage": stage, "alpha": alpha,
                              "cut_tick": CUT, "end_tick": END, "state_file_sha256": digest(state_file),
                              "native_trace_sha256": native_sha, "forecast96_sha256": forecast_sha,
                              "checkpoint_sha256": checkpoint, "start_state_sha256": _f32_sha(start),
                              "applied_tape_sha256": _f32_sha(tape)})
        require(totals["arms"] == budget["arms_per_stage"], "missing replay arms")
    except Exception as exc:
        write(folder / "replay_failure.json", {"stage": stage, "location": location, "totals": totals,
                                               "error": f"{type(exc).__name__}: {exc}"})
        raise
    return totals
