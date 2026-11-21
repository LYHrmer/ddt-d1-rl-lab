"""Fit the matched worlds, seal all autonomous forecasts, then collect shared native answers."""
from dataclasses import replace

import numpy as np

from tools91.launch91 import require
from tools94.common94 import arrays, digest, read, write
from tools96.common96 import direction
from tools96.run96 import measure, reset_previews
from tools98.common98 import ARMS, HORIZON, MODELS, RESETS, SEEDS, checked_stage, protocol


def fit(folder):
    from tools98.fit98 import fit_seed

    totals = dict(updates_attempted=0, updates_completed=0, attempted_model_steps=0, completed_model_steps=0,
                  policy_updates=0, ppo_updates=0)
    roster = []
    for seed in SEEDS:
        row = fit_seed(seed, folder / f"seed_{seed}")
        for name in totals:
            totals[name] += row[name]
        states = {m: row["arms"][m]["status"] for m in MODELS}
        roster.append({"seed": seed, "status": row["status"], "arms": states})
        print("fit", seed, states, row.get("error"), flush=True)
    budget = protocol()["budget"]
    require(totals["updates_completed"] <= totals["updates_attempted"] <= budget["world_updates"]
            and totals["completed_model_steps"] <= totals["attempted_model_steps"] <= budget["fit_model_steps"]
            and not (totals["policy_updates"] or totals["ppo_updates"]), "fit exceeds its registered budget")
    return {"totals": totals, "seeds": roster}


def predict(folder):
    from tools95.model95 import predict_arm
    from tools98.fit98 import load_fit

    fits = checked_stage("fit")
    plan = protocol()
    budget = plan["budget"]
    reset_previews(folder, RESETS)
    totals = dict(attempted_model_steps=0, completed_model_steps=0, backward_attempted=0, backward_completed=0)
    forecasts = 0
    for seed in SEEDS:
        fit_folder = fits / f"seed_{seed}"
        trained = read(fit_folder / "receipt.json")
        if trained["status"] != "COMPLETE":
            print("incomplete matched fit", seed, trained.get("error"), flush=True)
            continue
        parent = direction(seed)
        for model in MODELS:
            fit_arm = trained["arms"][model]
            if fit_arm["status"] != "COMPLETE" or fit_arm["updates_completed"] != budget["updates_per_arm"]:
                continue
            world = load_fit(seed, fit_folder, model)
            bundle = replace(parent, world=world)
            for reset in RESETS:
                dest = folder / f"seed_{seed}/reset_{reset}/{model}"
                dest.mkdir(parents=True)
                with np.load(folder / f"start_{reset}.npz", allow_pickle=False) as archive:
                    start = dict(archive)
                for tag, alpha in ARMS.items():
                    require(totals["attempted_model_steps"] + HORIZON <= budget["validation_model_steps"]
                            and forecasts < budget["blind_forecasts"], "blind prediction budget exhausted")
                    row = predict_arm(bundle, start, alpha)
                    states = np.asarray(row.pop("state45"), dtype=np.float32).reshape(-1, 45)
                    path = dest / f"{tag}.npz"
                    arrays(path, {"state45": states})
                    write(dest / f"{tag}.json", {**row, "seed": seed, "reset": reset, "model": model, "alpha": alpha,
                          "file": path.name, "state_sha256": digest(path), "checkpoint_sha256": fit_arm["checkpoint"]["sha256"]})
                    for name in totals:
                        totals[name] += row[name]
                    forecasts += 1
            print("predicted", seed, model, flush=True)
    require(totals["backward_attempted"] <= budget["validation_backwards"], "blind derivative budget exceeded")
    return {"totals": totals, "forecasts": forecasts, "reset_previews": len(RESETS), "preview_physics_substeps": 0}


def native(folder):
    predictions = checked_stage("predictions")
    return measure(folder, predictions, SEEDS, RESETS, tuple(ARMS))
