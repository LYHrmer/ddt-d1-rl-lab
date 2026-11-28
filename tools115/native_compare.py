"""Recompute the first-braking-pulse comparison from paired, native trajectories."""
from pathlib import Path

import numpy as np

from tools91.launch91 import ROOT, digest, read, require
from tools113.results import compare_tree
from tools114.closed_results import _actuation_and_clock, check_prefix, check_targets, score_native
from tools114.common import BLIND, DEV
from tools115.signals import APPLY, ARCHIVES, ISSUE, WHEELS


SEEDS = (*DEV, *BLIND)
FIRST_STATE, LAST_STATE = APPLY+1, APPLY+10


def reference114(seed):
    require(seed in SEEDS, "unregistered paired starting state")
    relative, sha = ARCHIVES[114]
    index = ROOT / relative
    require(digest(index) == sha, "round-114 parking archive changed")
    record = read(index)
    stage = f"native_{seed}"
    entry = record["stages"][stage]
    guard = record["guards"][entry["guard"]]
    directory = index.parent / stage
    path = directory / "trace.npz"
    report = directory / "report.json"
    require(entry["status"] == guard["status"] == "COMPLETE" and guard["stage"] == stage and guard["child_started"]
            and digest(report) == entry["files"]["report.json"], "unbound source trajectory")
    worker = read(report)
    row = worker["result"]
    case, = [r for r in record["cases"] if r["seed"] == seed]
    require(row["seed"] == seed and worker["stage"] == stage and worker["status"] == row["status"] == "COMPLETE"
            and row["motion"]["ticks"] == 600 and row["motion"]["passed"] and case["complete"] and case["gate_pass"]
            and digest(path) == row["trace_sha256"] == case["trace_sha256"] == entry["files"]["trace.npz"]
            == worker["files"]["trace.npz"], "incomplete or changed round-114 control")
    return path, row


def tilt_peak(qpos):
    """Return the peak over the ten post-integration endpoints affected by intervals 526..535."""
    if len(qpos) <= LAST_STATE:
        return None, None
    q = qpos[FIRST_STATE:LAST_STATE+1, 3:7]
    angle = np.rad2deg(np.arccos(np.clip(1.-2.*(q[:,1]**2+q[:,2]**2),-1.,1.)))
    peak = int(np.argmax(angle))
    return float(angle[peak]), FIRST_STATE+peak


def check_probe(z):
    raw, parked, issued, applied = (z[k] for k in ("raw", "parking_issued_before_probe", "issued", "applied"))
    n = len(raw)
    for block in (parked,issued,applied):
        require(block.shape == (n,16) and block.dtype == np.float32 and np.isfinite(block).all(),
                "invalid delivered motor target")
    zero = np.zeros((1,16),np.float32)
    # Reconstruct the unperturbed round-114 target and its FIFO from this arm's
    # own raw and current state. The subsequent network trajectory is free to respond.
    as114 = dict(z, issued=parked, applied=np.concatenate((zero,parked[:-1])))
    check_targets(as114)
    pulse = np.arange(n) == ISSUE
    require(z["rf_contrast_probe"].dtype == np.bool_, "invalid pulse marker")
    np.testing.assert_array_equal(z["rf_contrast_probe"],pulse)
    np.testing.assert_array_equal(issued[~pulse],parked[~pulse])
    np.testing.assert_array_equal(issued.reshape(n,4,4)[:,:,:3],parked.reshape(n,4,4)[:,:,:3])
    np.testing.assert_array_equal(applied,np.concatenate((zero,issued[:-1])))
    before = after = None
    if n > ISSUE:
        before = 5.*parked[ISSUE,WHEELS].astype(float)
        after = 5.*issued[ISSUE,WHEELS].astype(float)
        others = [0,2,3]
        # Orthogonal projection onto the complement of (-1,3,-1,-1),
        # independent of the execution code's update via RF minus the four-wheel mean.
        axis = np.array([-1.,3.,-1.,-1.])
        expect = before - axis*(axis @ before)/12.
        np.testing.assert_allclose(after,expect,rtol=0.,atol=1e-6)
        np.testing.assert_allclose(after.mean(),before.mean(),rtol=0.,atol=1e-6)
        np.testing.assert_allclose(after[1],after[others].mean(),rtol=0.,atol=1e-6)
        np.testing.assert_allclose(np.diff(after[others]),np.diff(before[others]),rtol=0.,atol=1e-6)
        require(before[1] != after[1], "the registered pulse had no right-front contrast to remove")
    return dict(first_changed_issue_tick=ISSUE if n > ISSUE else None,
                first_changed_applied_tick=APPLY if n > APPLY else None,
                interventions=int(pulse.sum()), fifo_intervals_checked=n, policy_history_windows_checked=n,
                wheel_order=["FL","FR","RL","RR"],
                reference_mapping_at_pulse_rad_s=None if before is None else before.tolist(),
                delivered_at_pulse_rad_s=None if after is None else after.tolist())


def score_candidate(path, seed):
    """Read an episode, validate clocks and the paired intervention, and return JSON-safe measurements."""
    path = Path(path)
    require(path.name == "trace.npz" and not any((path.parent/name).exists()
            for name in ("partial.npz", "partial_counts.json", "worker_partial.json")), "candidate is partial")
    reference_path, parent = reference114(seed)
    with np.load(path,allow_pickle=False) as traces, np.load(reference_path,allow_pickle=False) as control:
        z, old = dict(traces), dict(control)
    n = len(z["raw"])
    require(1 <= n <= 600, "candidate outside registered duration")
    wiring = check_probe(z)
    packets = len(z["capture_tick"])
    counts = dict(decisions=n, servo=5*n, physics=10*n, packets=packets, ray_queries=1040*packets)
    _actuation_and_clock(z, {"counts":counts})
    wiring["paired_prefix"] = check_prefix(z,old,wiring)
    observed = min(n,APPLY+1)
    for name in ("actor_previous_raw", "actor_history_previous_raw", "driver_command_xyz",
                 "parking_active", "velocity_trim", "unprojected_issued"):
        np.testing.assert_array_equal(z[name][:observed],old[name][:observed],err_msg=name)
    np.testing.assert_array_equal(z["parking_issued_before_probe"][:observed],old["issued"][:observed])
    if n > APPLY:
        np.testing.assert_array_equal(z["issued"][APPLY],old["issued"][APPLY])
        wiring["reference_first_wheel_torque_nm"] = old["first_torque_nm"][APPLY,WHEELS].tolist()
        wiring["candidate_first_wheel_torque_nm"] = z["first_torque_nm"][APPLY,WHEELS].tolist()
    height = parent["nominal_base_height_m"]
    motion, stop = score_native(z,height)
    reference_motion, reference_stop = score_native(old,height)
    compare_tree(reference_motion,parent["motion"])
    base_peak, base_tick = tilt_peak(old["qpos"])
    candidate_peak, candidate_tick = tilt_peak(z["qpos"])
    tilt = dict(first_state_index=FIRST_STATE, last_state_index=LAST_STATE,
                window_complete=candidate_peak is not None, reference_peak_deg=base_peak, reference_peak_tick=base_tick,
                candidate_peak_deg=candidate_peak, candidate_peak_tick=candidate_tick,
                delta_peak_deg=None if candidate_peak is None else candidate_peak-base_peak)
    return dict(seed=seed, reference_sha256=parent["trace_sha256"], trace_sha256=digest(path),
                complete_horizon=n == 600, gate_pass=n == 600 and motion["passed"],
                motion=motion, stop=stop, reference_motion=reference_motion, reference_stop=reference_stop,
                counts=counts, wiring=wiring, tilt=tilt)
