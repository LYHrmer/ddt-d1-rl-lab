"""Prepare the round-98 training prefixes and their recorded round-96 action pairs."""
import argparse
from dataclasses import dataclass
import json

import numpy as np
import torch

from tools94.common94 import ROOT, SEEDS, fit_reset, read, seed_dir
from tools96.common96 import BASE as BASE96, TRAIN_RESETS
from tools96.fit96 import checked, require
from tools98 import fit98
from tools98.common98 import BASE as BASE98
from tools99.loss99 import late_response_loss, paired_response


FIT98_SEAL = "a8e9b97cd684ffc80bbf28459ff35020242cefc13131a832bc07d0a95a342099"
CURVES96_SHA = "48151366dda596791b426ae503a2501ecefa6f36a4c9108d78b2d76e3d927183"
HORIZON = 32
FLAGS = ("fallen", "illegal", "terminated", "truncated")


@dataclass
class TrainingData:
    prefix_batch: dict
    late_actions: torch.Tensor
    late_targets: torch.Tensor
    early_denom: float
    flagged_endpoints: list


def read_full_arm(path, checksum, reset):
    """A recorded terminal endpoint at tick 32 is usable; no samples follow it."""
    with np.load(checked(path, checksum), allow_pickle=False) as data:
        n = int(data["completed_transitions"])
        require(int(data["reset_seed"]) == reset and n >= HORIZON, "missing paired training horizon")
        state = data["state45"][:HORIZON + 1]
        applied = data["applied16"][:HORIZON]
        issued = data["issued16"][:HORIZON]
        flags = {k: data[k][:HORIZON] for k in FLAGS}
    require(state.shape == (33, 45) and state.dtype == np.float64 and np.isfinite(state).all(),
            "invalid recorded state prefix")
    require(applied.shape == issued.shape == (32, 16) and applied.dtype == issued.dtype == np.float32
            and np.isfinite(issued).all() and (np.abs(issued) <= 1.).all() and not applied[0].any()
            and np.array_equal(applied[1:], issued[:-1]), "invalid one-tick applied-action trace")
    require(all(v.shape == (32,) and v.dtype == np.bool_ and not v[:-1].any() for v in flags.values()),
            "invalid flags or contact before the final paired endpoint")
    return state, applied, [name for name, value in flags.items() if value[-1]]


def load_training_data(seed):
    require(seed in SEEDS, "unknown world-model training seed")
    folder = BASE98 / "fit_01"
    seal = read(checked(folder / "seal.json", FIT98_SEAL))
    key = f"seed_{seed}/receipt.json"
    receipt = read(checked(folder / key, seal["files"][key]))
    entries = receipt["source"]["training_traces"]
    require(receipt["seed"] == seed and receipt["status"] == "COMPLETE" and len(entries) == 18,
            "incomplete source training roster")
    states, actions, suffixes, targets, flagged = [], [], [], [], []
    for i, entry in enumerate(entries[:12]):
        reset = fit_reset(seed, 0, i)
        path = seed_dir(seed) / f"round1/train_{reset}/trace.npz"
        require((entry["role"], entry["reset"], entry["path"]) == ("training94", reset, str(path.relative_to(ROOT))),
                "unexpected original training trajectory")
        state, applied = fit98._prefix(path, entry["sha256"], reset)
        states.append(state)
        actions.append(applied)
    for i, entry in enumerate(entries[12:]):
        arm, reset = ("m1", "p1")[i // 3], TRAIN_RESETS[i % 3]
        dest = BASE96 / f"collection_01/seed_{seed}/reset_{reset}/{arm}"
        path = dest / "native/trace.npz"
        require((entry["role"], entry["arm"], entry["reset"], entry["path"])
                == ("training96", arm, reset, str(path.relative_to(ROOT))), "unexpected paired training trajectory")
        native = read(checked(dest / "native.json", entry["receipt_sha256"]))
        require(native["error"] is None and native["trace_file"] == "native/trace.npz"
                and native["trace_sha256"] == entry["sha256"] and native["ticks"] == HORIZON,
                "incomplete paired training receipt")
        state, applied, flags = read_full_arm(path, entry["sha256"], reset)
        require(native["safe"] == (not flags), "trace and receipt disagree about safety")
        if flags:
            flagged.append({"reset": reset, "arm": arm, "tick": HORIZON, "flags": flags})
        states.append(state[:17])
        actions.append(applied[:16])
        suffixes.append(applied[16:32])
        targets.append(state[:, 0])
    prefix = fit98._batch(np.stack(states), np.stack(actions))
    require(fit98.mapping_sha256(prefix) == receipt["batch"]["tensors_sha256"], "round-98 base batch differs")
    x = torch.from_numpy(np.stack(targets))
    return TrainingData(prefix, torch.from_numpy(np.stack(suffixes)), paired_response(x[:, 17:33], x[:, 0]),
                        receipt["normalization"]["denominator_m2"], flagged)


def public_labels(archive, seed):
    """Read only the six collection curves assigned to this training seed."""
    require(seed in SEEDS, "unknown training seed")
    curves, flagged = [], []
    for arm in ("m1", "p1"):
        for reset in TRAIN_RESETS:
            key = f"collection_{seed}_{reset}_{arm}_x"
            x, flags = archive[key], archive[key + "_flags"]
            require(x.shape == (33,) and x.dtype == np.float64 and np.isfinite(x).all()
                    and flags.shape == (32, 4) and flags.dtype == np.bool_ and not flags[:-1].any(),
                    "unusable public training curve")
            if flags[-1].any():
                flagged.append({"reset": reset, "arm": arm, "tick": HORIZON,
                                "flags": [name for name, active in zip(FLAGS, flags[-1]) if active]})
            curves.append(x)
    x = torch.from_numpy(np.stack(curves))
    require(torch.equal(x[:3, :2], x[3:, :2]), "paired collection origins differ")
    return paired_response(x[:, 17:33], x[:, 0]), flagged


def check_labels(seed, target, flagged):
    """Apply three known forecasts without loading or running a world model."""
    reversed_scale = torch.tensor(-1., dtype=torch.float64, requires_grad=True)
    reversed_loss = late_response_loss(reversed_scale * target, target)
    slope, = torch.autograd.grad(reversed_loss, reversed_scale)
    correct = float(late_response_loss(target, target))
    zero = float(late_response_loss(torch.zeros_like(target), target))
    reversed_value = float(reversed_loss.detach())
    np.testing.assert_allclose([correct, zero, reversed_value, slope.item()], [0., 1., 4., -4.], atol=1e-12, rtol=0)
    return {"seed": seed, "pairs": target.shape[0], "ticks_per_pair": target.shape[1],
            "target_rms_mm": 1000 * float(target.square().mean().sqrt()), "positive_labels": int((target > 0).sum()),
            "flagged_endpoints": flagged, "correct_loss": correct, "zero_loss": zero,
            "reversed_loss": reversed_value, "reversed_scale_gradient": slope.item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", action="store_true", help="also match the local sealed training inputs to the public labels")
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    with np.load(checked(ROOT / "reports/paired_response96_curves.npz", CURVES96_SHA), allow_pickle=False) as archive:
        for seed in SEEDS:
            target, flagged = public_labels(archive, seed)
            if args.raw:
                data = load_training_data(seed)
                require(torch.equal(data.late_targets, target) and data.flagged_endpoints == flagged,
                        "public labels differ from the recorded training inputs")
            results.append(check_labels(seed, target, flagged))
    print(json.dumps(results, ensure_ascii=False, indent=2))
