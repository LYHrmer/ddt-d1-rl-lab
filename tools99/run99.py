"""Fork the registered worlds, seal their forecasts, then collect the shared native answers."""
from dataclasses import replace

import numpy as np

from tools91.launch91 import require
from tools94.common94 import arrays, digest, read, write
from tools96.common96 import direction
from tools96.run96 import measure, reset_previews
from tools99.common99 import ARMS, HORIZON, MODELS, RESETS, SEEDS, checked_stage, protocol


def fit(folder):
    from tools99.fit99 import fit_seed

    budget = protocol()["budget"]
    totals = dict(updates_attempted=0, updates_completed=0, attempted_model_steps=0, completed_model_steps=0,
                  policy_updates=0, ppo_updates=0)
    roster = []
    for seed in SEEDS:
        row = fit_seed(seed, folder / f"seed_{seed}")
        for name in totals:
            totals[name] += row[name]
        arms = {model: row["arms"][model]["status"] for model in MODELS}
        roster.append({"seed": seed, "status": row["status"], "arms": arms})
        print("fit", seed, arms, row.get("error"), flush=True)
    require(totals["updates_completed"] <= totals["updates_attempted"] <= budget["world_updates"]
            and totals["completed_model_steps"] <= totals["attempted_model_steps"] <= budget["fit_model_steps"]
            and not (totals["policy_updates"] or totals["ppo_updates"]), "training clocks exceed registered limits")
    return {"totals": totals, "seeds": roster}


def predict(folder):
    from tools95.model95 import predict_arm
    from tools99.fit99 import load_fit

    budget, fits = protocol()["budget"], checked_stage("fit")
    reset_previews(folder, RESETS)
    starts = {}
    for reset in RESETS:
        with np.load(folder / f"start_{reset}.npz", allow_pickle=False) as archive:
            starts[reset] = dict(archive)
    totals = dict(attempted_model_steps=0, completed_model_steps=0, backward_attempted=0, backward_completed=0)
    forecasts = 0
    for seed in SEEDS:
        fit_folder = fits / f"seed_{seed}"
        row = read(fit_folder / "receipt.json")
        if row["status"] != "COMPLETE":
            print("incomplete matched fit", seed, row.get("error"), flush=True)
            continue
        parent = direction(seed)
        for model in MODELS:
            fit_arm = row["arms"][model]
            require(fit_arm["status"] == "COMPLETE" and fit_arm["updates_completed"] == budget["updates_per_arm"],
                    "incomplete checkpoint in qualified training pair")
            bundle = replace(parent, world=load_fit(seed, fit_folder, model))
            for reset in RESETS:
                dest = folder / f"seed_{seed}/reset_{reset}/{model}"
                dest.mkdir(parents=True)
                for tag, alpha in ARMS.items():
                    require(forecasts < budget["blind_forecasts"]
                            and totals["attempted_model_steps"] + HORIZON <= budget["validation_model_steps"]
                            and (alpha != 0 or totals["backward_attempted"] < budget["validation_backwards"]),
                            "blind forecast exceeds registered limits")
                    forecast = predict_arm(bundle, starts[reset], alpha)
                    states = np.asarray(forecast.pop("state45"), dtype=np.float32).reshape(-1, 45)
                    path = dest / f"{tag}.npz"
                    arrays(path, {"state45": states})
                    write(dest / f"{tag}.json", {**forecast, "seed": seed, "reset": reset, "model": model,
                          "alpha": alpha, "file": path.name, "state_sha256": digest(path),
                          "checkpoint_sha256": fit_arm["checkpoint"]["sha256"]})
                    for name in totals:
                        totals[name] += forecast[name]
                    forecasts += 1
            print("predicted", seed, model, flush=True)
    return {"totals": totals, "forecasts": forecasts, "reset_previews": len(starts), "preview_physics_substeps": 0}


def native(folder):
    predictions, fits = checked_stage("predictions"), checked_stage("fit")
    complete = [seed for seed in SEEDS if read(fits / f"seed_{seed}/receipt.json")["status"] == "COMPLETE"]
    require(read(predictions / "seal.json")["forecasts"] == len(complete)*len(RESETS)*len(MODELS)*len(ARMS),
            "seal all forecasts from each matched checkpoint before native measurements")
    return measure(folder, predictions, complete, RESETS, tuple(ARMS))
