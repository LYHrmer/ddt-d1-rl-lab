"""Proposed56 candidate: paired8-vs32 training horizons, implementation only.

This source does not authorize, seal or launch a training run.  The56 runner
must separately bind complete53 training data, sources and resource admission.
Both arms use the same legal32-step starts, ridge/statistics, initialization and
sampled window rows.  Only the prefix length entering the frozen52 loss differs.
No claim that longer training resolves the observed control failures is made.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from tools52 import dynamics52 as frozen

MODEL_SEEDS = (56001, 56002, 56003)
HORIZONS = (8, 32)
COMMON_HORIZON = 32
UPDATES = 1000
BATCH_SIZE = 64
BATCH_SEED = 52000
LEARNING_RATE = 3e-4
GRADIENT_CLIP = 1.0
DATA_KEYS = ("s", "a", "target", "episode", "tick", "done")
require = frozen.require


def _array_identity(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({"shape": list(array.shape), "dtype": array.dtype.str},
                        sort_keys=True, separators=(",", ":")).encode()
    return header, array.tobytes(order="C")


def array_sha256(value):
    """Hash shape, exact dtype and contiguous C-order values without pickle."""
    header, content = _array_identity(value)
    return hashlib.sha256(header + b"\0" + content).hexdigest()


def mapping_sha256(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy()
        header, content = _array_identity(value)
        digest.update(name.encode() + b"\0" + header + b"\0" + content)
    return digest.hexdigest()


def common_window_indices(data):
    """Return all legal32-step windows, entirely within complete episodes."""
    return frozen.window_indices(data, horizon=COMMON_HORIZON)


def sampled_window_rows(window_count, *, updates=UPDATES, batch_size=BATCH_SIZE):
    """CPU int64 [updates,batch] rows; model seed/horizon never enter this RNG.

    The per-update randint calls match52's sampling semantics.  Optional small
    dimensions serve synthetic sampler tests; fit_model always fixes1000×64.
    """
    require(type(window_count) is int and window_count > 0, "positive window count required")
    require(type(updates) is int and updates > 0 and type(batch_size) is int and batch_size > 0,
            "positive sampler dimensions required")
    generator = torch.Generator(device="cpu").manual_seed(BATCH_SEED)
    return torch.stack([torch.randint(window_count, (batch_size,), generator=generator,
                                     device="cpu", dtype=torch.int64) for _ in range(updates)])


def selected_indices(windows, chosen, horizon):
    """Select the same common window rows, then take only the8/32-step prefix."""
    require(type(horizon) is int and horizon in HORIZONS, "horizon must be8 or32")
    require(windows.dtype == chosen.dtype == torch.int64
            and windows.device.type == chosen.device.type == "cpu", "CPU int64 window indices required")
    require(windows.ndim == 2 and windows.shape[1] == COMMON_HORIZON
            and chosen.ndim == 1 and len(chosen) > 0, "common window/batch shapes differ")
    require(bool(((chosen >= 0) & (chosen < len(windows))).all()), "chosen window row is out of range")
    return windows[chosen][:, :horizon]


def paired_sequence_loss(model, tensors, windows, chosen, horizon):
    """Apply the unchanged52 seven-group loss to one paired window prefix."""
    indices = selected_indices(windows, chosen, horizon)
    return frozen.sequence_loss(model, tensors["s"][indices[:, 0]],
                                tensors["a"][indices], tensors["target"][indices])


def _write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _deadline(cpu_deadline, stage):
    if time.process_time() >= cpu_deadline:
        raise TimeoutError("CPU budget exhausted " + stage)


def fit_model(data, seed, horizon, output, cpu_deadline):
    """One new final-only fit; no overwrite/resume or best-checkpoint selection.

    Call once per registered seed/arm under the external serial resource guard.
    Returns(model, receipt) only after1000 completed optimizer updates.  Failure
    preserves initial/sampling evidence, flushed update/event prefixes and an
    INCOMPLETE receipt; any final weights already written remain ineligible.
    """
    require(type(seed) is int and seed in MODEL_SEEDS, "unregistered proposed56 initialization seed")
    require(type(horizon) is int and horizon in HORIZONS, "horizon must be8 or32")
    require(isinstance(cpu_deadline, (int, float)) and math.isfinite(cpu_deadline),
            "finite absolute process CPU deadline required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    receipt = {"status": "INCOMPLETE", "candidate": "56_paired_training_horizon",
               "seed": seed, "horizon": horizon, "common_window_horizon": COMMON_HORIZON,
               "planned_updates": UPDATES, "updates": 0, "updates_attempted": 0,
               "batch_size": BATCH_SIZE, "batch_seed": BATCH_SEED,
               "learning_rate": LEARNING_RATE, "gradient_clip_norm": GRADIENT_CLIP,
               "ridge_lambda": .001, "dtype": "float32", "device": "cpu", "envs": 0,
               "checkpoint_selection": "final_only", "eligible_for_evaluation": False,
               "retry_or_resume": False, "failure": None, "stage": "created"}
    _write_json(output / "fit_started.json", receipt)
    schedule = None
    completed = attempted = 0
    with (output / "fit_events.jsonl").open("x", buffering=1) as events, \
            (output / "updates.jsonl").open("x", buffering=1) as updates:
        def event(name, **fields):
            events.write(json.dumps({"event": name, "updates_completed": completed,
                                     "updates_attempted": attempted, **fields}, allow_nan=False) + "\n")
            events.flush()

        try:
            event("START", seed=seed, horizon=horizon)
            _deadline(cpu_deadline, "before fit")
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            receipt["stage"] = "sampling_plan"
            arrays = frozen.validate_data(data)
            receipt["training_data_sha256"] = mapping_sha256({key: arrays[key] for key in DATA_KEYS})
            common = common_window_indices(arrays).astype("<i8", copy=False)
            windows = torch.from_numpy(common)
            schedule = sampled_window_rows(len(windows))
            sampled = schedule.numpy().astype("<i8", copy=False)
            receipt.update(common_windows_sha256=array_sha256(common),
                           planned_sampling_rows_sha256=array_sha256(sampled),
                           training_transitions=len(arrays["s"]), training_windows=len(windows),
                           training_episodes=np.unique(arrays["episode"]).tolist())
            with (output / "sampling_plan.npz").open("xb") as stream:
                np.savez_compressed(stream, common_windows=common, sampled_window_rows=sampled)
            _deadline(cpu_deadline, "after sampling plan")
            receipt["stage"] = "ridge"
            baseline = frozen.fit_ridge(arrays)
            receipt["ridge_and_normalization_sha256"] = mapping_sha256(baseline)
            _deadline(cpu_deadline, "after ridge")
            torch.manual_seed(seed)
            model = frozen.Dynamics52(**baseline)
            require(all(value.device.type == "cpu" and value.dtype == torch.float32
                        for value in (*model.parameters(), *model.buffers())),
                    "frozen52 model must initialize as CPU float32")
            receipt["initial_model_state_sha256"] = mapping_sha256(model.state_dict())
            _write_json(output / "pairing_identity.json", {key: value for key, value in receipt.items()
                        if key.endswith("sha256") or key in ("seed", "horizon", "common_window_horizon",
                                                              "training_windows", "batch_seed", "batch_size")})
            tensors = {key: torch.as_tensor(arrays[key], dtype=torch.float32, device="cpu")
                       for key in ("s", "a", "target")}
            optimizer = torch.optim.Adam(model.residual.parameters(), lr=LEARNING_RATE)
            final_loss = None
            receipt["stage"] = "updates"
            for update in range(UPDATES):
                _deadline(cpu_deadline, "during fit")
                chosen = schedule[update]
                attempted += 1
                event("UPDATE_BEGIN", update=update + 1,
                      sampled_rows_sha256=array_sha256(chosen.numpy().astype("<i8", copy=False)))
                loss = paired_sequence_loss(model, tensors, windows, chosen, horizon)
                require(bool(torch.isfinite(loss)), "nonfinite multistep loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.residual.parameters(), GRADIENT_CLIP,
                                                     error_if_nonfinite=True)
                optimizer.step()
                completed += 1
                final_loss = float(loss.detach())
                updates.write(json.dumps({"update": completed, "loss": final_loss,
                    "gradient_norm_before_clip": float(norm), "cpu_seconds": time.process_time() - started,
                    "sampled_rows_sha256": array_sha256(chosen.numpy().astype("<i8", copy=False))},
                    allow_nan=False) + "\n")
                updates.flush()
            receipt["stage"] = "finalization"
            _deadline(cpu_deadline, "after fit")
            require(completed == attempted == UPDATES, "training update counts incomplete")
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "nonfinite final model")
            model.eval().requires_grad_(False)
            with (output / "final_state_dict.pt").open("xb") as stream:
                torch.save(model.state_dict(), stream)
            _deadline(cpu_deadline, "after final serialization")
            receipt.update(status="COMPLETE", stage="complete", updates=completed,
                           updates_attempted=attempted, eligible_for_evaluation=True,
                           final_training_minibatch_loss=final_loss,
                           weights_file="final_state_dict.pt",
                           final_model_state_sha256=mapping_sha256(model.state_dict()),
                           completed_sampling_rows_sha256=array_sha256(sampled[:completed]),
                           cpu_seconds=time.process_time() - started,
                           loss_groups=list(frozen.GROUPS), physical_scale45=frozen.PHYSICAL_SCALE.tolist(),
                           zero_input_features=list(frozen.ZERO_FEATURES),
                           pd_semantics="implicit in learned executed-action to20ms state transition")
            _deadline(cpu_deadline, "before complete receipt")
            event("COMPLETE")
            _write_json(output / "fit_receipt.json", receipt)
            return model, receipt
        except BaseException as exc:
            receipt.update(status="INCOMPLETE", eligible_for_evaluation=False, updates=completed,
                           updates_attempted=attempted, cpu_seconds=time.process_time() - started,
                           failure=f"{type(exc).__name__}: {exc}",
                           final_weights_written=(output / "final_state_dict.pt").exists())
            if schedule is not None:
                receipt["completed_sampling_rows_sha256"] = array_sha256(
                    schedule[:completed].numpy().astype("<i8", copy=False))
            event("INCOMPLETE", failure=receipt["failure"], stage=receipt["stage"])
            with (output / "failure.txt").open("x") as stream:
                stream.write(traceback.format_exc())
            _write_json(output / "partial_receipt.json", receipt)
            raise
