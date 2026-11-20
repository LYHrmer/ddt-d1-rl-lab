"""Exercise the new parent axis, causal input boundary and retained prefixes with synthetic data."""
import hashlib
import json

import numpy as np
import pytest

from tools93 import model93 as replay


def _write(path, obj):
    path.write_text(json.dumps(obj))


def _start(reset):
    q = np.zeros(23, dtype=np.float64)
    q[[0, 2, 3, 7]] = [1.10, .45, 1., replay.RESETS.index(reset) / 8.]
    v, obs = np.zeros(22, dtype=np.float64), np.zeros(643, dtype=np.float32)
    initial = hashlib.sha256(q.tobytes() + v.tobytes() + obs.tobytes()).hexdigest()
    return q, v, obs, initial


@pytest.fixture
def collection(tmp_path, monkeypatch):
    plan = dict(parents=list(replay.PARENTS), resets=list(replay.RESETS), model_seeds=list(replay.MODELS),
                arms=list(replay.ARMS), horizon=50, policies_sha256={str(p): f"{p:064x}" for p in replay.PARENTS})
    monkeypatch.setattr(replay, "checked_plan", lambda: plan)
    reference = tmp_path / "reference.json"
    old = []
    for p, r in zip(replay.PARENTS, replay.RESETS, strict=True):
        for arm in replay.ARMS:
            old.append(dict(parent_seed=p, reset_seed=r, arm=arm, policy_sha256=plan["policies_sha256"][str(p)],
                            initial_state_sha256=_start(r)[3], ticks=100, status="COMPLETE", clean=True))
    _write(reference, {"stand": {"cases": old}})
    monkeypatch.setattr(replay, "BASELINE90", reference)
    physical = tmp_path / "collect"
    physical.mkdir()
    report = dict(status="COMPLETE", protocol_sha256=replay.PLAN_SHA, cases=[])
    for i, (p, r, arm) in enumerate(replay._new_keys(plan), 1):
        q, v, obs, fingerprint = _start(r)
        issued = np.full((50, 16), i/32, dtype=np.float32)
        applied = np.zeros((50, 16), dtype=np.float64)
        applied[1:] = issued[:-1]
        file = physical / f"source{i}.npz"
        # Accessing either subsequent-state member requires pickle and fails on the export path.
        forbidden = np.array([None], dtype=object)
        np.savez(file, initial_qpos=q, initial_qvel=v, initial_obs=obs, issued=issued, applied=applied,
                 qpos=forbidden, qvel=forbidden)
        report["cases"].append(dict(parent_seed=p, reset_seed=r, arm=arm, policy_sha256=plan["policies_sha256"][str(p)],
                                    initial_state_sha256=fingerprint, ticks=50, status="COMPLETE", clean=True,
                                    trace=file.name, trace_sha256=replay.digest(file)))
    _write(physical / "seal.json", report)
    return plan, physical


def test_export_off_diagonal_only_and_preserve_the_parent_axis(collection, tmp_path):
    plan, physical = collection
    out = tmp_path / "inputs"
    report = replay.prepare(out, physical)
    assert report["status"] == "COMPLETE"
    assert report["physical_seal_sha256"] == replay.digest(physical / "seal.json")
    keys = replay._new_keys(plan)
    assert len(set(keys)) == 12 and not {(p, r) for p, r, _ in keys}.intersection(zip(replay.PARENTS, replay.RESETS))
    assert len({case["file"] for case in report["cases"]}) == 12
    raw_cases = replay.read(physical / "seal.json")["cases"]
    for row, raw, (case, state, actions) in zip(report["cases"], raw_cases, replay._sealed_inputs(out, report, keys), strict=True):
        with np.load(out / row["file"], allow_pickle=False) as exported:
            assert set(exported.files) == {"state45", "applied16"}
        assert case["input_sha256"] == row["file_sha256"] == replay.digest(out / row["file"])
        with np.load(physical / raw["trace"], allow_pickle=False) as native:
            np.testing.assert_array_equal(state, np.r_[native["initial_qpos"], native["initial_qvel"]])
            np.testing.assert_array_equal(actions, native["applied"])
        assert row["source_trace_sha256"] == raw["trace_sha256"]
    with pytest.raises(FileExistsError):
        replay.prepare(out, physical)


@pytest.mark.parametrize(("corruption", "message"), [
    ("partial", "unqualified"), ("duplicate", "inventory"), ("initial", "unqualified"),
    ("bytes", "source SHA256"), ("shift", "one-tick FIFO"),
])
def test_unqualified_physical_data_are_retained_without_admission(collection, tmp_path, corruption, message):
    _, physical = collection
    seal = physical / "seal.json"
    report = replay.read(seal)
    row = report["cases"][-1]
    if corruption == "partial":
        row.update(status="TERMINATED", ticks=15)
    elif corruption == "duplicate":
        report["cases"][-1] = report["cases"][-2]
    elif corruption == "initial":
        row["initial_state_sha256"] = "0"*64
    elif corruption == "bytes":
        row["trace_sha256"] = "f"*64
    else:
        source = physical / row["trace"]
        with np.load(source, allow_pickle=False) as raw:
            fields = {k: raw[k] for k in ("initial_qpos", "initial_qvel", "initial_obs", "applied", "issued")}
        fields["applied"][30, 1] += 2**-40
        np.savez(source, **fields)
        row["trace_sha256"] = replay.digest(source)
    _write(seal, report)
    out = tmp_path / "inputs"
    with pytest.raises(ValueError, match=message):
        replay.prepare(out, physical)
    saved = replay.read(out / "seal.json")
    assert saved["status"] == "PARTIAL" and saved["reason"]
    assert len(saved["cases"]) == (0 if corruption == "duplicate" else 11)
    with pytest.raises(FileExistsError):
        replay.prepare(out, physical)


def test_same_reset_and_arm_get_distinct_traces_per_parent_and_keep_failed_prefix(tmp_path):
    pytest.importorskip("torch")

    class Stand:
        def step(self, state, action, terrain):
            next_state = state.clone()
            next_state[:, 0] += action[:, 0]
            return next_state

    class Falls:
        def step(self, state, action, terrain):
            next_state = state.clone()
            next_state[:, 2] = 0.
            return next_state

    p1, p2, _ = replay.PARENTS
    r = replay.RESETS[-1]
    q, v, _, _ = _start(r)
    state = np.r_[q, v]
    report = dict(cases=[])
    actions = np.zeros((2, 16), dtype=np.float32)
    actions[1, 0] = .125
    base = dict(reset_seed=r, arm="original", task="stand", start_tick=0, horizon=2,
                model_seed=replay.MODELS[0], input_sha256="synthetic")
    for p, model in ((p1, Stand()), (p2, Falls())):
        replay._predict_arm(tmp_path, dict(base, parent_seed=p), model, state, actions, (.27, .6), report)
    first, second = report["cases"]
    assert first["trace"] != second["trace"] and first["status"] == "COMPLETE" and first["accepted_steps"] == 2
    assert second["status"] == "PREDICTED_FALL" and second["attempted_steps"] == second["accepted_steps"] == 1
    for row in report["cases"]:
        assert replay.digest(tmp_path / row["trace"]) == row["trace_sha256"]
        with np.load(tmp_path / row["trace"], allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["endpoint_tick"], np.arange(row["accepted_steps"] + 1))
            np.testing.assert_array_equal(raw["applied16"], actions[:row["attempted_steps"]])
    with pytest.raises(FileExistsError):
        replay._predict_arm(tmp_path, dict(base, parent_seed=p1), Falls(), state, actions, (.27, .6), report)
    assert len(report["cases"]) == 2
