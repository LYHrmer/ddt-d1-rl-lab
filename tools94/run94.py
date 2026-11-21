"""One independent cold start. Calibration and held-out trajectories never enter fitting."""
from __future__ import annotations

import numpy as np
import torch

from tools56.fit56 import mapping_sha256
from tools94 import score94
from tools94.actor94 import make_actor
from tools94.common94 import CALIBRATION, EVALUATION, PLAN, SEEDS, arrays, clocks, digest, fit_reset, seed_dir, source_files, tag, write
from tools94.learn94 import LearnFailure, fit_world, probe, update_actor
from tools94.native94 import NativeEpisodeError, collect_episode


def checkpoint(folder, name, net, optimizer=None):
    path = folder / (name + ".pt")
    data = {"network": net.state_dict()}
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    with path.open("xb") as stream:
        torch.save(data, stream)
    return {"file": path.name, "sha256": digest(path), "weights_sha256": mapping_sha256(data["network"])}


def physical(folder, actor, reset, totals, *, ticks=50, exploration=None, offset=0):
    folder.mkdir()
    write(folder / "start.json", dict(reset=reset, ticks=ticks, explore_seed=exploration, offset_rad_s=offset))
    # A killed unit retains completed native endpoints and applied actions up to the last callback.
    last = 0
    with (folder / "endpoints.f64").open("xb", buffering=0) as journal:
        def progress(trace):
            nonlocal last
            n = len(trace["applied16"])
            for i in range(last, n):
                row = np.concatenate((trace["state45"][i+1], trace["issued16"][i], trace["applied16"][i]))
                journal.write(np.asarray(row, dtype="<f8").tobytes())
            last = n
        try:
            trace = collect_episode(actor, reset, ticks=ticks, explore_seed=exploration,
                                    wheel_offset_rad_s=offset, progress=progress)
        except NativeEpisodeError as exc:
            if getattr(exc, "trace", None) is not None:
                save_trace(folder, exc.trace, totals, partial=True)
            raise
    file = save_trace(folder, trace, totals, partial=False)
    return trace, file


def save_trace(folder, trace, totals, *, partial):
    path = folder / ("partial_trace.npz" if partial else "trace.npz")
    arrays(path, trace)
    for field, key in (("native_decisions", "policy_steps"), ("servo_updates", "servo_steps"),
                       ("physics_substeps", "native_steps")):
        totals[field] += int(trace[key])
    return {"path": str(path), "sha256": digest(path), "ticks": len(trace["applied16"])}


def safe(trace, n):
    return (len(trace["applied16"]) == n and not any(np.any(trace[key])
            for key in ("fallen", "illegal", "terminated", "truncated")))


def measured_summary(trace):
    return {"ticks": len(trace["applied16"]), "fallen": bool(trace["fallen"].any()),
            "illegal": bool(trace["illegal"].any()), "terminated": bool(trace["terminated"].any()),
            "truncated": bool(trace["truncated"].any()), "mean_four_wheel_support": (
                float(np.mean(np.all(trace["support4"], axis=1))) if len(trace["applied16"]) else None)}


def calibration_cell(folder, seed, reset, r, world, actor, totals):
    folder.mkdir()
    traces, files, preds, failures = {}, {}, {}, {}
    for k in score94.OFFSETS:
        trace, file = physical(folder / tag(k), actor, reset, totals, offset=k/100.)
        traces[k], files[k] = trace, file
    reference = traces[-20]
    # Each arm has its own delayed policy feedback. Reset state and first applied step must match.
    for trace in traces.values():
        for name in ("state45", "obs643"):
            if not np.array_equal(reference[name][0], trace[name][0]):
                raise ValueError(f"calibration initial condition differs: {name}")
        if len(trace["applied16"]) and (not np.array_equal(trace["applied16"][0], np.zeros(16))
                or not np.array_equal(trace["state45"][1], reference["state45"][1])):
            raise ValueError("different physical first endpoint before FIFO delivery")
    inputs = {name: reference[name][:1].copy() for name in
              ("state45", "obs643", "filtered3", "ray_raw16", "ray_due", "sensor_delivery_tick", "sensor_capture_tick")}
    derivative = None
    prediction_files = {}
    for k in (*score94.OFFSETS, 0):
        try:
            states, d = probe(world, actor, inputs, k/100., derivative=(k == 0))
            if k == 0:
                derivative = float(d) if d is not None and np.isfinite(d) else None
        except LearnFailure as exc:
            failures[str(k)] = {"error": str(exc), "receipt": exc.receipt}
            states = getattr(exc, "partial_states", None)
            if states is None:
                states = inputs["state45"].copy()
        states = np.asarray(states)
        if states.ndim != 2 or states.shape[1] != 45 or not np.isfinite(states).all():
            raise ValueError("invalid retained model endpoints")
        path = folder / f"model_{tag(k)}.npz"
        arrays(path, {"state45": states})
        prediction_files[str(k)] = dict(path=str(path), sha256=digest(path), ticks=len(states)-1)
        preds[k] = states[:, 0]
    x = {k: t["state45"][:, 0] for k, t in traces.items()}
    clean = {k: safe(t, 50) for k, t in traces.items()}
    score = score94.calibration(x, preds, derivative, clean)
    cell = dict(seed=seed, round=r+1, reset=reset, native={str(k): f for k,f in files.items()},
                models=prediction_files, failures=failures, score=score)
    write(folder / "score.json", cell)
    return cell


def evaluations(folder, initial_actor, final_actor, totals):
    folder.mkdir()
    results = []
    for reset in EVALUATION:
        pair = folder / str(reset)
        pair.mkdir()
        a, af = physical(pair / "initial", initial_actor, reset, totals, ticks=100)
        b, bf = physical(pair / "final", final_actor, reset, totals, ticks=100)
        if not np.array_equal(a["state45"][:2], b["state45"][:2]) or not np.array_equal(a["obs643"][0], b["obs643"][0]):
            raise ValueError("initial and final actors have different physical initial conditions")
        score = score94.control(a["state45"][:,0], b["state45"][:,0], safe(a,100) and safe(b,100))
        row = dict(reset=reset, initial=af, final=bf, score=score)
        write(pair / "score.json", row)
        results.append(row)
    return results


def run(seed):
    if seed not in SEEDS:
        raise ValueError("unregistered training seed")
    folder = seed_dir(seed)
    folder.mkdir()
    totals = clocks()
    seed_report = dict(seed=seed, status="PARTIAL", totals=totals, protocol_sha256=digest(PLAN),
                       sources=source_files(), rounds=[], evaluation=[], error=None)
    write(folder / "started.json", seed_report)
    active_update = None
    try:
        initial = make_actor(seed)
        actor = make_actor(seed)
        seed_report["initial"] = checkpoint(folder, "initial", initial)
        if mapping_sha256(actor.state_dict()) != seed_report["initial"]["weights_sha256"]:
            raise ValueError("initial control differs from the cold training actor")
        training_batches = []
        previous = policy_optimizer = None
        for r in range(2):
            stage = folder / f"round{r+1}"
            stage.mkdir()
            row = dict(round=r+1, fit=[], calibration=[], status="COLLECTING",
                       collector_weights_sha256=mapping_sha256(actor.state_dict()))
            seed_report["rounds"].append(row)
            fresh = []
            for j in range(12):
                reset = fit_reset(seed, r, j)
                noise_seed = int(np.random.SeedSequence([seed, r, j, 94]).generate_state(1)[0])
                t, file = physical(stage / f"train_{reset}", actor, reset, totals, exploration=noise_seed)
                fresh.append(t)
                row["fit"].append({**file, "reset": reset, **measured_summary(t)})
            training_batches.append(fresh)
            write(stage / "training_inputs.json", row["fit"])
            active_update = "world_updates"
            world, optimizer, fit = fit_world(training_batches, seed, previous=previous)
            active_update = None
            totals["world_updates"] += int(fit["updates_completed"])
            row["world_fit"] = fit
            row["world_checkpoint"] = checkpoint(stage, "world", world, optimizer)
            write(stage / "fit.json", {"fit": fit, "checkpoint": row["world_checkpoint"]})
            previous = (world, optimizer)
            if fit["updates_completed"] != 100:
                raise ValueError("incomplete 100-update world model")
            for reset in CALIBRATION[r]:
                row["calibration"].append(calibration_cell(stage / f"cal_{reset}", seed, reset, r, world, actor, totals))
            row["calibration_passed"] = all(c["score"]["passed"] for c in row["calibration"])
            if not row["calibration_passed"]:
                row["status"] = seed_report["status"] = "STOPPED_AT_CALIBRATION"
                write(stage / "report.json", row)
                break
            active_update = "policy_updates"
            actor, policy_optimizer, update = update_actor(world, actor, fresh, seed+100*r,
                                                          previous_optimizer=policy_optimizer)
            active_update = None
            totals["policy_updates"] += int(update["updates_completed"])
            row.update(policy_update=update, policy_checkpoint=checkpoint(stage, "actor", actor, policy_optimizer))
            if update["updates_completed"] != 4 or update["status"] != "COMPLETE":
                row["status"] = seed_report["status"] = "STOPPED_AT_POLICY_CHECK"
                write(stage / "report.json", row)
                break
            row["status"] = "COMPLETE"
            write(stage / "report.json", row)
        else:
            seed_report["evaluation"] = evaluations(folder / "evaluation", initial, actor, totals)
            seed_report["status"] = "EVALUATED"
    except Exception as exc:
        seed_report["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, LearnFailure):
            seed_report["failure_receipt"] = exc.receipt
            if active_update:
                totals[active_update] += int(exc.receipt.get("updates_completed", 0))
            partial = exc.partial_model if active_update == "world_updates" else exc.partial_actor
            if partial is not None:
                seed_report["partial_checkpoint"] = checkpoint(folder, "partial", partial, exc.partial_optimizer)
        raise
    finally:
        write(folder / "report.json", seed_report)
    return seed_report
