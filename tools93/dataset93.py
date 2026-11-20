"""Package the 18 measured first-second standing trajectories with their applied actions."""
import hashlib
from pathlib import Path

import numpy as np

from tools93.common93 import ARMS, BASE, PARENTS, PLAN_SHA, RESETS, ROOT, checked_plan, digest, read, require


H = 50
WHEEL_ACTIONS = (3, 7, 11, 15)
ORDER = tuple((p, r, a) for p in PARENTS for r in RESETS for a in ARMS)
DIAGONAL = set(zip(PARENTS, RESETS, strict=True))


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def payload_rows(raw, *, ticks):
    def initial(k, size):
        a = raw[k]
        require(a.shape == (size,) and a.dtype == np.float64 and np.isfinite(a).all(), f"invalid reset {k}")
        return a

    def interval(k, tail, dtype):
        a = raw[k]
        require(a.shape == (ticks, *tail) and a.dtype == dtype and np.isfinite(a[:H]).all(), f"invalid field {k}")
        return a[:H]

    require(ticks in (50, 100) and np.allclose(interval("time_s", (), np.float64), np.arange(1,51)*.02,
                                              atol=1e-9, rtol=0), "physical times differ from the 50 Hz endpoints")
    require(not interval("command", (3,), np.float64).any(), "standing phase must have a zero command")
    for k in ("fallen", "illegal", "terminated", "truncated"):
        require(not interval(k, (), np.bool_).any(), f"incomplete or invalid physical arm: {k}")
    q0, v0 = initial("initial_qpos", 23), initial("initial_qvel", 22)
    q, v = interval("qpos", (23,), np.float64), interval("qvel", (22,), np.float64)
    state = np.vstack((np.r_[q0, v0], np.column_stack((q, v))))
    require(not v0.any() and q0[0] == 1.10 and
            np.allclose(np.linalg.norm(state[:,3:7], axis=1), 1., rtol=0, atol=1e-6), "invalid physical state45")
    issued = interval("issued", (16,), np.float32)
    applied64 = interval("applied", (16,), np.float64)
    applied = applied64.astype(np.float32)
    require(same_bits(applied.astype(np.float64), applied64), "lossy conversion of recorded applied actions")
    require(not applied[0].any() and np.all(np.abs(issued) <= 1.) and same_bits(applied[1:], issued[:-1]),
            "physical action FIFO is not exactly one interval")
    indices = raw["wheel_dof_indices"]
    scales = raw["wheel_action_scales_rad_s"]
    require(indices.shape == (4,) and indices.dtype.kind in "iu" and len(set(indices.tolist())) == 4
            and np.all((indices >= 6) & (indices < 22)) and np.array_equal(scales, [5.]*4),
            "invalid wheel actuator mapping")
    measured = v[:, indices]
    if "measured_wheel_speed_rad_s" in raw:
        require(same_bits(interval("measured_wheel_speed_rad_s", (4,), np.float64), measured),
                "measured wheel speeds do not match qvel")
    wheel_targets = interval("applied_wheel_target_rad_s", (4,), np.float64)
    require(np.array_equal(wheel_targets, applied64[:, WHEEL_ACTIONS] * scales), "wheel target/action mismatch")
    raw_action = interval("policy_raw", (16,), np.float32)
    require(np.all(np.abs(raw_action) <= 1.), "policy action outside its physical contract")
    fields = dict(state45=state, issued16=issued, applied16=applied, policy_raw16=raw_action,
                  support_mask=interval("wheel_contacts_past_toe", (4,), np.bool_),
                  wheel_target_rad_s=wheel_targets, wheel_speed_rad_s=measured,
                  body_vx_m_s=interval("body_vx_m_s", (), np.float64), pitch_rad=interval("pitch_rad", (), np.float64),
                  last_torque_nm=interval("last_torque_nm", (16,), np.float64),
                  range_features16=interval("ranges", (16,), np.float32))
    return fields, indices.astype(np.int64)


def references():
    plan = checked_plan()
    old_path = ROOT / "reports/stand_wbc90.json"
    fresh_path = BASE / "collect_01/seal.json"
    old = read(old_path)["stand"]
    fresh = read(fresh_path)
    require(old["status"] == fresh["status"] == "COMPLETE"
            and fresh["protocol_sha256"] == PLAN_SHA, "unsealed physical collection")
    records = {}
    for kind, cases in (("90",old["cases"]), ("93",fresh["cases"])):
        for e in cases:
            p, r, a = e["parent_seed"], e["reset_seed"], e["arm"]
            key = (p,r,a)
            require(key in ORDER and key not in records and ((p,r) in DIAGONAL) == (kind == "90")
                    and e["status"] == "COMPLETE" and e["clean"]
                    and e["ticks"] == (100 if kind == "90" else H)
                    and e["policy_sha256"] == plan["policies_sha256"][str(p)],
                    "wrong source, reused arm or incomplete physical case")
            records[key] = e
    require(records.keys() == set(ORDER), "incomplete sealed three-by-three grid")
    return records, {"physical90_report_sha256": digest(old_path), "physical93_seal_sha256": digest(fresh_path)}


def export(out, sources):
    """Write one exclusive CPU/NumPy archive from sealed (Path, SHA256) physical sources."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    require(out.suffix == ".npz" and sources.keys() == set(ORDER), "expected all 18 sealed physical .npz inputs")
    records, source_seals = references()
    starts = {}
    cases, panels = [], []
    dof_indices = None
    for i, key in enumerate(ORDER):
        p, r, a = key
        source, raw_sha = sources[key]
        source = Path(source)
        receipt = records[key]
        require(source.is_absolute() and source.is_file() and not source.is_symlink()
                and digest(source) == raw_sha == receipt["trace_sha256"], "physical SHA256 or grid identity differs")
        with np.load(source, allow_pickle=False) as raw:
            panel, idx = payload_rows(raw, ticks=receipt["ticks"])
            initial_hash = hashlib.sha256()
            for k in ("initial_qpos", "initial_qvel", "initial_obs"):
                initial_hash.update(raw[k].tobytes())
        require(initial_hash.hexdigest() == receipt["initial_state_sha256"], "wrong physical reset/observation identity")
        if dof_indices is None:
            dof_indices = idx
        require(same_bits(idx, dof_indices), "wheel mapping changed between physical arms")
        if a == "original":
            require(same_bits(panel["issued16"], panel["policy_raw16"]), "original arm changes policy actions")
        # Every parent sees the same initial condition, zero FIFO and the same first
        # physical endpoint at each reset. Later observed states stay with their own arm.
        ref = (initial_hash.hexdigest(), array_sha(panel["state45"][:2]), array_sha(panel["applied16"][0]))
        require(starts.setdefault(r, ref) == ref, "same-reset initial state or first physical transition differs")
        panels.append(panel)
        cases.append(dict(row=i, parent_seed=p, reset_seed=r, arm=a, source_round=90 if (p,r) in DIAGONAL else 93,
                          source_trace_sha256=raw_sha, policy_sha256=receipt["policy_sha256"],
                          initial_state_sha256=receipt["initial_state_sha256"], source_ticks=receipt["ticks"]))
    require(len({v[0] for v in starts.values()}) == len(RESETS), "physical resets are not distinct")
    arrays = {k: np.stack([panel[k] for panel in panels]) for k in panels[0]}
    arrays.update(parent_seed=np.array([e["parent_seed"] for e in cases], dtype=np.int64),
                  reset_seed=np.array([e["reset_seed"] for e in cases], dtype=np.int64),
                  arm=np.array([e["arm"] for e in cases]), source_trace_sha256=np.array([e["source_trace_sha256"] for e in cases]),
                  policy_sha256=np.array([e["policy_sha256"] for e in cases]), state_ticks=np.arange(H+1,dtype=np.int64),
                  interval_end_ticks=np.arange(1,H+1,dtype=np.int64), wheel_qvel_columns=dof_indices,
                  wheel_action_columns=np.array(WHEEL_ACTIONS, dtype=np.int64), wheel_order=np.array(["FL","FR","RL","RR"]),
                  protocol_sha256=np.array(PLAN_SHA), evaluation_only=np.array(True))
    # The raw arm files are immutable. Write only after all sources agree, and read
    # every typed array back before publishing its receipt, with pickle disabled.
    with out.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(out, allow_pickle=False) as saved:
        require(set(saved.files) == arrays.keys() and len(saved.files) == len(arrays), "export array inventory")
        for k, v in arrays.items():
            require(v.dtype.kind != "O" and same_bits(saved[k], v), f"portable array round trip failed: {k}")
    schema = {k: dict(shape=list(v.shape), dtype=v.dtype.str, sha256=array_sha(v)) for k,v in arrays.items()}
    return dict(status="COMPLETE", file=out.name, protocol_sha256=PLAN_SHA, sha256=digest(out), bytes=out.stat().st_size,
                evaluation_only=True, n_arms=len(cases), horizon=H, policy_period_s=.02, arrays=schema, cases=cases,
                source_seals=source_seals, reset_groups={str(r): [c["row"] for c in cases if c["reset_seed"] == r] for r in RESETS},
                parent_groups={str(p): [c["row"] for c in cases if c["parent_seed"] == p] for p in PARENTS},
                indexing="state45[i,t] = raw float64 (qpos23, qvel22). Interval j=0..49 goes from state j to j+1; "
                         "applied16[i,j] drove that interval; issued16 and policy_raw16 were computed at its start. "
                         "support_mask, wheel speeds, body_vx_m_s, pitch_rad and last_torque_nm are observed at its end. "
                         "range_features16 is the causal, delayed 16-vector available to the actor at interval start.",
                support_definition="FL,FR,RL,RR contacts at or beyond the 0.60 m toe carrying greater than 1 N at the endpoint.",
                split="Evaluation only. A whole reset and all its policies, arms and derived windows form one group. "
                      "These three previously used reset states are excluded from world-model fitting and normalization.")
