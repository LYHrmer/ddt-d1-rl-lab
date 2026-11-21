"""Collect train-only pairs, fit matched worlds, seal predictions, then measure blind answers."""
from dataclasses import replace

import numpy as np

from tools95.run95 import native_arm
from tools96.common96 import (ARMS, BASE, BLIND_RESETS, HORIZON, MODELS, ORIGINAL, PLAN_SHA, SEEDS,
                              STAGES, TRAIN_RESETS, arrays, checked, clocks, digest, direction,
                              files_for_seal, protocol, read, sources_for, write)


def reset_previews(folder, resets):
    from tools94.native94 import _endpoint, _open_env, _reset_uphill

    for reset in resets:
        env = _open_env(reset, HORIZON)
        try:
            obs = _reset_uphill(env, reset)
            state = _endpoint(env, obs, 0)
            if any(env.counters()[name] for name in ("policy_transitions", "pd_updates", "native_steps")):
                raise ValueError("unexpected physical integration in reset-only preview")
            arrays(folder / f"start_{reset}.npz", {key: v[None] for key, v in state.items()})
        finally:
            env.close()


def measure(folder, preview_folder, seeds, resets, tags):
    from tools95.model95 import variant

    totals = clocks()
    count = 0
    for seed in seeds:
        bundle = direction(seed)
        for reset in resets:
            with np.load(preview_folder / f"start_{reset}.npz", allow_pickle=False) as saved:
                start = dict(saved)
            dest = folder / f"seed_{seed}/reset_{reset}"
            dest.mkdir(parents=True)
            first = None
            for tag in tags:
                first = native_arm(dest / tag, variant(bundle, ARMS[tag]), reset, totals, start, first)
                count += 1
            print("measured", seed, reset, totals["native_decisions"], flush=True)
    limit = len(seeds) * len(resets) * len(tags) * HORIZON
    if totals["native_decisions"] > limit or totals["servo_updates"] > 5 * limit or totals["physics_substeps"] > 10 * limit:
        raise ValueError("native counters exceed stage cap")
    return {"totals": totals, "native_episodes": count}


def collect(folder):
    reset_previews(folder, TRAIN_RESETS)
    return {**measure(folder, folder, SEEDS, TRAIN_RESETS, ("m1", "p1")),
            "reset_previews": len(TRAIN_RESETS), "preview_physics_substeps": 0}


def fit(folder):
    from tools96.fit96 import fit_seed

    collection = checked("collection")
    totals = dict(updates_attempted=0, updates_completed=0, attempted_model_steps=0, completed_model_steps=0)
    seeds = []
    for seed in SEEDS:
        receipt = fit_seed(seed, collection, folder / f"seed_{seed}")
        for key in totals:
            totals[key] += receipt[key]
        seeds.append({"seed": seed, "status": receipt["status"],
                      "arms": {a: receipt["arms"][a]["status"] for a in MODELS}})
        print("fit", seed, seeds[-1]["arms"], flush=True)
    budget = protocol()["budget"]
    if totals["updates_attempted"] > budget["world_updates"] or totals["attempted_model_steps"] > budget["fit_model_steps"]:
        raise ValueError("fit counters exceed stage cap")
    return {"totals": totals, "seeds": seeds}


def complete_models(folder, seed):
    receipt = read(folder / f"seed_{seed}/receipt.json")
    return [name for name in MODELS if receipt["arms"][name]["status"] == "COMPLETE"
            and receipt["arms"][name]["updates_completed"] == 50]


def predict(folder):
    from tools95.model95 import predict_arm
    from tools96.fit96 import load_fit

    fits = checked("fit")
    reset_previews(folder, BLIND_RESETS)
    totals = dict(attempted_model_steps=0, completed_model_steps=0, backward_attempted=0, backward_completed=0)
    for seed in SEEDS:
        parent = direction(seed)
        for model_name in complete_models(fits, seed):
            world = load_fit(seed, fits / f"seed_{seed}", model_name)
            bundle = replace(parent, world=world)
            for reset in BLIND_RESETS:
                dest = folder / f"seed_{seed}/reset_{reset}/{model_name}"
                dest.mkdir(parents=True)
                with np.load(folder / f"start_{reset}.npz", allow_pickle=False) as saved:
                    start = dict(saved)
                for tag, alpha in ARMS.items():
                    row = predict_arm(bundle, start, alpha)
                    states = np.asarray(row.pop("state45"), dtype=np.float32).reshape(-1, 45)
                    file = dest / f"{tag}.npz"
                    arrays(file, {"state45": states})
                    write(dest / f"{tag}.json", {**row, "seed": seed, "reset": reset, "model": model_name,
                                                 "alpha": alpha, "file": file.name, "state_sha256": digest(file)})
                    for key in totals:
                        totals[key] += row[key]
            print("predicted", seed, model_name, flush=True)
    budget = protocol()["budget"]
    if (totals["attempted_model_steps"] > budget["validation_model_steps"]
            or totals["backward_attempted"] > budget["validation_backwards"]):
        raise ValueError("validation model counters exceed stage cap")
    return {"totals": totals, "reset_previews": len(BLIND_RESETS), "preview_physics_substeps": 0}


def blind_native(folder):
    fits = checked("fit")
    predictions = checked("predictions")
    seeds = [seed for seed in SEEDS if complete_models(fits, seed)]
    return measure(folder, predictions, seeds, BLIND_RESETS, tuple(ARMS))


def run_stage(stage, folder):
    actions = {"collection": collect, "fit": fit, "predictions": predict, "native": blind_native}
    previous = STAGES.index(stage) - 1
    provenance = {"stage": stage, "protocol_sha256": PLAN_SHA, "sources": sources_for(stage),
                  "direction_seal95_sha256": digest(ORIGINAL / "seal.json")}
    if previous >= 0:
        provenance["previous_seal_sha256"] = digest(BASE / f"{STAGES[previous]}_01/seal.json")
    write(folder / "start.json", provenance)
    result = actions[stage](folder)
    if provenance["sources"] != sources_for(stage):
        raise ValueError("producer changed during stage")
    write(folder / "seal.json", {**result, **provenance, "files": files_for_seal(folder)})
