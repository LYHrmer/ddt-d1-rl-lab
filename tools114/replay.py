"""Factor the round-113 stopping error into initial-state and applied-target substitutions."""
from itertools import product
from pathlib import Path

import numpy as np

from tools91.launch91 import digest, read, require
from tools94.common94 import arrays
from tools110.course import command
from tools111.control import SPEED_GAIN
from tools111.wheel_response import wheel_signals
from tools113.common import PLAN, PUBLIC, SEEDS
from tools113.model import CARRY, GAIN, STATES, TICKS, load_model, physical_targets, state_vector
from tools113.results import stage_report, verify_inputs


RESET = 525
TAIL = 575
STOP_LIMIT = .05
PARENT_SHA = "409a86c9f06fe9fce8df10cda0f0abf0c17b53574c41fae64abc4ef12d59b1dc"
SEAL_SHA = "2007f11e93469dfd95d17c24b3fda8079e4cc63455435cab6a84936bcdce2eee"
SOURCES = ("sealed", "native")
BUNDLE = "stop_replay.npz"


def fixed_suffix(world, start, state, targets, destination=None):
    """Advance exactly once per delivered physical target using the original route clock."""
    state = np.asarray(state, dtype=float)
    targets = np.asarray(targets, dtype=float)
    require(state.shape == (STATES,) and targets.ndim == 2 and targets.shape[1] == 16
            and 0 <= start < start+len(targets) <= TICKS, "invalid stopping replay slice")
    shape = (len(targets)+1, STATES)
    path = np.empty(shape) if destination is None else destination
    require(path.shape == shape and path.dtype == np.float64, "invalid suffix destination")
    path[0] = state
    for j, action in enumerate(targets):
        path[j+1] = world.step(start+j, path[j], action)
        require(np.isfinite(path[j+1]).all(), "nonfinite model estimate")
    return path


def stopping(path):
    require(path.shape == (TICKS-RESET+1, STATES) and np.isfinite(path).all(), "incomplete braking suffix")
    velocities = path[TAIL-RESET:,3:5]
    speeds = np.linalg.norm(velocities,axis=1)
    peak = float(speeds.max())
    return dict(peak_m_s=peak, peak_tick=TAIL+int(speeds.argmax()), passes_stop=peak <= STOP_LIMIT,
                endpoints_above_limit=int(np.count_nonzero(speeds > STOP_LIMIT)),
                mean_forward_m_s=float(velocities[:-1,0].mean()))


def _parent():
    p = PUBLIC/"results.json"
    seal = PLAN.parent/"forecast_seal.json"
    require(digest(p) == PARENT_SHA and digest(seal) == SEAL_SHA, "round-113 source or pre-physics seal changed")
    manifest = read(p)
    stages = manifest["stages"]
    verify_inputs(stages)
    selected = ["forecast"] + [f"{stage}_{s}" for s in SEEDS for stage in ("native","fit")]
    records = {}
    for stage in selected:
        r = stage_report(stages,stage)
        require(r is not None and r["status"] == "COMPLETE", f"incomplete round-113 {stage}")
        name = stages[stage]["guard"]
        receipt = manifest["guards"][name]
        guard = PUBLIC/"resources"/name/"report.json"
        require(receipt["status"] == "COMPLETE" and digest(guard) == receipt["files"]["report.json"]
                and read(guard)["owned_cgroup_children_empty"], f"unsealed resource guard for {stage}")
        records[stage] = r["result"]
    return records


def _signals(z):
    expected = SPEED_GAIN*np.array([command(k) for k in range(TICKS)])
    np.testing.assert_allclose(z["base_cmd"], expected, atol=1e-12, rtol=0.)
    # Exclude the initial standing interval. The final falling edge is the stop request.
    edges = np.flatnonzero((expected[:-1] > 0.) & (expected[1:] == 0.))+1
    np.testing.assert_array_equal(edges,[RESET])
    np.testing.assert_array_equal(expected[RESET:],0.)
    np.testing.assert_allclose(z["time"], np.arange(TICKS+1)/50., atol=1e-9, rtol=0.)
    target, encoder, torque = wheel_signals(z)
    wheel_sat = z["wheel_saturation"][TAIL:]
    target, encoder, torque = target[TAIL:], encoder[TAIL:], torque[TAIL:]
    return dict(first_stop_request_tick=RESET, stop_request_m_s=float(expected[RESET]),
                actor_vx_at_stop_request_m_s=float(z["actor_vx"][RESET]),
                tail_actor_vx_range_m_s=[float(z["actor_vx"][TAIL:].min()),float(z["actor_vx"][TAIL:].max())],
                wheel_samples=target.size, mean_target_rad_s=float(target.mean()),
                mean_encoder_rad_s=float(encoder.mean()), negative_targets_per_wheel=(target<0.).sum(axis=0).tolist(),
                negative_encoders_per_wheel=(encoder<0.).sum(axis=0).tolist(),
                negative_first_torques_per_wheel=(torque<0.).sum(axis=0).tolist(),
                mean_first_torque_nm=float(torque.mean()), maximum_abs_first_torque_nm=float(np.abs(torque).max()),
                saturated_wheel_intervals_per_wheel=wheel_sat.sum(axis=0,dtype=int).tolist())


def _sources(records, seed):
    fit = records[f"fit_{seed}"]
    native = records[f"native_{seed}"]
    row, = [r for r in records["forecast"]["rows"] if r["seed"] == seed]
    model_path = PUBLIC/f"fit_{seed}"/"action.npz"
    forecast_path = PUBLIC/"forecast"/f"pred_{seed}.npz"
    trace_path = PUBLIC/f"native_{seed}"/"trace.npz"
    require(digest(model_path) == row["model_sha256"] == fit["arms"]["action"]["checkpoint_sha256"]
            and digest(forecast_path) == row["sha256"] and row["file"] == forecast_path.name
            and digest(trace_path) == native["trace_sha256"] and native["motion"]["ticks"] == TICKS
            and native["seed"] == fit["heldout_seed"] == seed and native["trim"] == "brake_aware",
            "unmatched held-out model, on-forecast or native trace")
    world = load_model(model_path)
    require(world.uses_action and world.heldout == seed, "incorrect world-model arm")
    with np.load(forecast_path,allow_pickle=False) as z:
        sealed = z["on_state"][RESET:]
        targets = z["on_targets"][RESET:]
        # Tick k uses the previous action, including at the braking boundary.
        np.testing.assert_array_equal(targets,physical_targets(z["on_issued"][RESET-1:-1]))
    with np.load(trace_path,allow_pickle=False) as z:
        actual = state_vector(z["qpos"][RESET:],z["qvel"][RESET:])
        actions = physical_targets(z["applied"][RESET:])
        np.testing.assert_array_equal(z["applied"][RESET:],z["issued"][RESET-1:-1])
        signals = _signals(z)
    np.testing.assert_allclose(stopping(actual)["peak_m_s"],native["motion"]["stop_tail_max_m_s"],atol=1e-9,rtol=0.)
    np.testing.assert_allclose(stopping(sealed)["peak_m_s"],row["on"]["stop_peak_m_s"],atol=1e-9,rtol=0.)
    sha = dict(model=row["model_sha256"],forecast=row["sha256"],native=native["trace_sha256"])
    return world, {"sealed":sealed,"native":actual}, {"sealed":targets,"native":actions}, signals, sha


def _row(seed, paths, native, sealed, signals, hashes):
    truth = stopping(native)
    arms = {}
    for label, path in paths.items():
        m = stopping(path)
        error = m["peak_m_s"]-truth["peak_m_s"]
        arms[label] = dict(**m, peak_error_m_s=error, abs_peak_error_m_s=abs(error),
                           classification_matches_native=m["passes_stop"] == truth["passes_stop"],
                           false_pass=m["passes_stop"] and not truth["passes_stop"])
    p = {key: val["peak_m_s"] for key,val in arms.items()}
    return dict(seed=seed, source_sha256=hashes, native_stop=truth, sealed_stop=stopping(sealed), arms=arms,
                delta_peak_m_s=dict(state_with_sealed_targets=p["native_sealed"]-p["sealed_sealed"],
                                    state_with_native_targets=p["native_native"]-p["sealed_native"],
                                    targets_with_sealed_state=p["sealed_native"]-p["sealed_sealed"],
                                    targets_with_native_state=p["native_native"]-p["native_sealed"],
                                    interaction=p["native_native"]-p["native_sealed"]-p["sealed_native"]+p["sealed_sealed"]),
                signals=signals)


def diagnose(out):
    """Generate all 900 new model steps in the caller's reserved stage directory."""
    out = Path(out)
    require(out.is_dir() and not (out/BUNDLE).exists(), "missing fresh output directory")
    records = _parent()
    # Write each endpoint as it becomes available. An interrupted run keeps its partially
    # populated file; a row of NaNs marks an uncomputed endpoint.
    progress = out/"partial_stop_replay.npy"
    shape = (len(SEEDS),len(SOURCES),len(SOURCES),TICKS-RESET+1,STATES)
    with progress.open("xb") as stream:
        np.save(stream,np.full(shape,np.nan))
    scratch = np.lib.format.open_memmap(progress,mode="r+")
    packed, rows = {}, []
    for i,seed in enumerate(SEEDS):
        world, states, targets, signals, hashes = _sources(records,seed)
        paths = {}
        for (ia,a),(ib,b) in product(enumerate(SOURCES),repeat=2):
            key = a+"_"+b
            paths[key] = fixed_suffix(world, RESET, states[a][0], targets[b], scratch[i,ia,ib])
            scratch.flush()
            packed[f"{seed}_{key}"] = paths[key]
        np.testing.assert_allclose(paths["sealed_sealed"],states["sealed"],rtol=1e-10,atol=1e-8)
        rows.append(_row(seed,paths,states["native"],states["sealed"],signals,hashes))
    arrays(out/BUNDLE, packed)
    progress.unlink()
    return dict(parent_sha256=PARENT_SHA,forecast_seal_sha256=SEAL_SHA,bundle_sha256=digest(out/BUNDLE),
                start_tick=RESET,tail_first_tick=TAIL,end_tick=TICKS,stop_limit_m_s=STOP_LIMIT,
                model_steps=sum(len(v)-1 for v in packed.values()),actor_forwards=0,
                world_updates=0,policy_updates=0,physics=0,rows=rows)


def _check_transitions(world, path, targets):
    """Check stored transition identities in one matrix operation, without a new rollout."""
    ref = world.reference_state[RESET:]
    normalized = (path-ref)/world.state_scale
    u = (targets-world.reference_target[RESET:])/world.target_scale
    f = np.concatenate((normalized[:-1],u),axis=1)
    force = GAIN*np.tanh(f@world.weight + world.bias)
    residual = (normalized[1:] - CARRY*normalized[:-1] - force)*world.state_scale
    np.testing.assert_allclose(residual, 0., rtol=0., atol=1e-8)


def _compare(actual, stored):
    if isinstance(actual, dict):
        require(isinstance(stored,dict) and set(actual) == set(stored), "incomplete stopping metric")
        for key in actual:
            _compare(actual[key], stored[key])
    elif isinstance(actual, float):
        np.testing.assert_allclose(actual,stored,rtol=0.,atol=1e-8)
    else:
        require(actual == stored, "changed stopping labels, counts or source checksums")


def verify(out, result):
    """Check all reported numbers and the recorded state-transition residuals from public data."""
    file = Path(out)/BUNDLE
    header = dict(parent_sha256=PARENT_SHA,forecast_seal_sha256=SEAL_SHA,bundle_sha256=digest(file),
                  start_tick=RESET,tail_first_tick=TAIL,end_tick=TICKS,stop_limit_m_s=STOP_LIMIT,
                  model_steps=900,actor_forwards=0,world_updates=0,policy_updates=0,physics=0)
    require(all(result.get(k) == v for k,v in header.items()), "invalid fixed-action budget or evidence")
    require([r["seed"] for r in result["rows"]] == list(SEEDS), "incorrect stopping roster")
    records = _parent()
    with np.load(file,allow_pickle=False) as z:
        expected = {f"{s}_{a}_{b}" for s in SEEDS for a,b in product(SOURCES,repeat=2)}
        require(set(z.files) == expected, "missing or unexpected suffix")
        for seed,stored in zip(SEEDS,result["rows"]):
            world, states, targets, signals, hashes = _sources(records,seed)
            paths = {}
            for a,b in product(SOURCES,repeat=2):
                key = a+"_"+b
                path = z[f"{seed}_{key}"]
                np.testing.assert_array_equal(path[0],states[a][0])
                _check_transitions(world,path,targets[b])
                paths[key] = path
            np.testing.assert_allclose(paths["sealed_sealed"],states["sealed"],rtol=1e-10,atol=1e-8)
            _compare(_row(seed,paths,states["native"],states["sealed"],signals,hashes),stored)
    return {"cases":len(SEEDS),"recorded_transition_checks":900,"new_rollout_steps":0}
