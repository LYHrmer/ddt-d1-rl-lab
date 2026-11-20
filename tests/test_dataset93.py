"""Exercise the portable standing-data package using only generated NumPy arrays."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools93 import dataset93 as pack


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/stand_data93.json"
CURVES = ROOT / "reports/stand_data93_curves.jsonl.gz"
BUNDLE = ROOT / "reports/stand_data93_validation.npz"


def synthetic(parent, reset, arm, ticks):
    q0 = np.zeros(23, np.float64)
    q0[[0, 2, 3, 7]] = [1.10, .45, 1., (reset-90100)*.01]
    v0 = np.zeros(22, np.float64)
    obs0 = np.full(643, (reset-90100)*.01, np.float32)
    q = np.repeat(q0[None], ticks, axis=0)
    v = np.zeros((ticks, 22), np.float64)
    parent_value = float(pack.PARENTS.index(parent)+1)
    arm_adjustment = .03 if arm == "common" else 0.
    # All arms have identical first transitions at the same reset.
    q[:, 0] -= .001*np.arange(ticks)*(parent_value+arm_adjustment)
    v[:, 9] = .002*np.arange(ticks)*(parent_value+arm_adjustment)
    raw = np.full((ticks,16), parent_value/10, np.float32)
    issued = raw.copy()
    issued[:,pack.WHEEL_ACTIONS] += np.float32(arm_adjustment)
    applied = np.zeros((ticks,16), np.float64)
    applied[1:] = issued[:-1]
    dofs = np.array([9, 13, 17, 21], np.int32)
    z = dict(initial_qpos=q0, initial_qvel=v0, initial_obs=obs0, qpos=q, qvel=v,
             time_s=np.arange(1,ticks+1)*.02, command=np.zeros((ticks,3)), wheel_dof_indices=dofs,
             wheel_action_scales_rad_s=np.full(4,5.), issued=issued, applied=applied, policy_raw=raw,
             applied_wheel_target_rad_s=applied[:,pack.WHEEL_ACTIONS]*5., body_vx_m_s=v[:,0], pitch_rad=q[:,8],
             wheel_contacts_past_toe=np.ones((ticks,4),dtype=bool), last_torque_nm=np.zeros((ticks,16)),
             ranges=np.zeros((ticks,16),np.float32))
    for name in ("fallen", "illegal", "terminated", "truncated"):
        z[name] = np.zeros(ticks,dtype=bool)
    if ticks == pack.H:
        z["measured_wheel_speed_rad_s"] = v[:,dofs]
    else:
        z["command"][50:,0] = .2  # the archived physical reference also contains a moving second
    return z


def reset_hash(z):
    return hashlib.sha256(b"".join(z[name].tobytes() for name in ("initial_qpos","initial_qvel","initial_obs"))).hexdigest()


@pytest.fixture
def grid(tmp_path,monkeypatch):
    rows, records = {}, {}
    folder = tmp_path / "sources"
    folder.mkdir()
    for i,key in enumerate(pack.ORDER):
        p,r,a = key
        ticks = 100 if (p,r) in pack.DIAGONAL else pack.H
        z = synthetic(p,r,a,ticks)
        path = folder / f"physical{i}.npz"
        np.savez_compressed(path, **z)
        sha = pack.digest(path)
        rows[key] = (path,sha)
        records[key] = dict(trace_sha256=sha, ticks=ticks, policy_sha256=hashlib.sha256(str(p).encode()).hexdigest(),
                            initial_state_sha256=reset_hash(z))
    monkeypatch.setattr(pack,"references",lambda:(records,{"test":"synthetic"}))
    return rows,records


def test_exports_full_physics_rows_and_18_grouped_cases_with_reproducible_shapes(tmp_path,grid):
    rows, records = grid
    dest = tmp_path / "validation.npz"
    manifest = pack.export(dest, dict(reversed(list(rows.items()))))
    assert manifest["status"] == "COMPLETE" and manifest["evaluation_only"] is True
    assert (manifest["n_arms"],manifest["horizon"]) == (18,50)
    assert manifest["sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()
    assert manifest["bytes"] == dest.stat().st_size
    with np.load(dest,allow_pickle=False) as data:
        assert data["state45"].shape == (18,51,45) and data["state45"].dtype == np.float64
        assert data["applied16"].shape == data["issued16"].shape == (18,50,16)
        assert data["applied16"].dtype == data["issued16"].dtype == np.float32
        assert data["support_mask"].shape == (18,50,4) and data["support_mask"].dtype == bool
        assert data["evaluation_only"].item() and data["protocol_sha256"].item() == pack.PLAN_SHA
        assert tuple(zip(data["parent_seed"],data["reset_seed"],data["arm"])) == pack.ORDER
        for name,m in manifest["arrays"].items():
            arr = data[name]
            assert arr.dtype.kind != "O"
            assert m == dict(dtype=arr.dtype.str, shape=list(arr.shape), sha256=hashlib.sha256(arr.tobytes()).hexdigest())
        for i,key in enumerate(pack.ORDER):
            p,r,a = key
            assert i in manifest["parent_groups"][str(p)] and i in manifest["reset_groups"][str(r)]
            assert manifest["cases"][i]["source_trace_sha256"] == rows[key][1] == data["source_trace_sha256"][i]
            with np.load(rows[key][0],allow_pickle=False) as raw:
                np.testing.assert_array_equal(data["state45"][i,0],np.r_[raw["initial_qpos"],raw["initial_qvel"]])
                np.testing.assert_array_equal(data["state45"][i,1:],np.column_stack((raw["qpos"][:50],raw["qvel"][:50])))
                np.testing.assert_array_equal(data["applied16"][i],raw["applied"][:50])
                np.testing.assert_array_equal(data["wheel_speed_rad_s"][i],raw["qvel"][:50,raw["wheel_dof_indices"]])
    with pytest.raises(FileExistsError):
        pack.export(dest,{})
    assert manifest["sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()


@pytest.mark.parametrize("fault,error", [("fifo","FIFO"),("rounded","lossy"),("too_short","invalid field"),
                                        ("fall","incomplete"),("velocity","wheel speeds"),("target","wheel target")])
def test_rejects_invalid_raw_motor_inputs_and_partial_runs(fault,error):
    z = synthetic(87011,90102,"common",50)
    if fault == "fifo":
        z["applied"][1,0] = .75
    elif fault == "rounded":
        z["applied"][1,0] = np.nextafter(z["applied"][1,0],1.)
    elif fault == "too_short":
        z["qvel"] = z["qvel"][:-1]
    elif fault == "fall":
        z["fallen"][8] = True
    elif fault == "velocity":
        z["measured_wheel_speed_rad_s"][3,0] += .01
    else:
        z["applied_wheel_target_rad_s"][4,0] += .01
    with pytest.raises(ValueError,match=error):
        pack.payload_rows(z,ticks=50)


def test_rejects_a_swapped_source_and_a_mismatched_same_reset_endpoint(tmp_path,grid):
    rows, records = grid
    out = tmp_path / "never_written.npz"
    k, other = pack.ORDER[0],pack.ORDER[1]
    swapped = dict(rows)
    swapped[k] = rows[other]
    with pytest.raises(ValueError,match="grid identity"):
        pack.export(out, swapped)
    assert not out.exists()
    p,r,a = other
    z = synthetic(p,r,a,100)
    z["qpos"][0,1] += .001
    altered = tmp_path / "different.npz"
    np.savez_compressed(altered, **z)
    new = pack.digest(altered)
    rows[other] = (altered,new)
    records[other]["trace_sha256"] = new
    with pytest.raises(ValueError,match="same-reset"):
        pack.export(out,rows)
    assert not out.exists()


def test_public_npz_matches_the_18_sealed_physics_curves_and_grouped_splits():
    report = json.loads(REPORT.read_bytes())
    manifest = report["validation_dataset"]
    assert report["dataset_exporter_sha256"] == hashlib.sha256((ROOT / "tools93/dataset93.py").read_bytes()).hexdigest()
    assert manifest["status"] == "COMPLETE" and manifest["evaluation_only"] is True
    assert manifest["bytes"] == BUNDLE.stat().st_size and manifest["sha256"] == hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    assert manifest["protocol_sha256"] == report["protocol_sha256"] == pack.PLAN_SHA
    plan = json.loads((ROOT / "docs/experiments93/stand_data.json").read_bytes())
    assert manifest["source_seals"]["physical93_seal_sha256"] == report["resources"]["collect"]["receipt_sha256"]
    assert manifest["source_seals"]["physical90_report_sha256"] == plan["source_sha256"]["reports/stand_wbc90.json"]
    assert manifest["policy_period_s"] == .02
    assert (manifest["n_arms"], manifest["horizon"]) == (18,50)
    assert hashlib.sha256(CURVES.read_bytes()).hexdigest() == report["curves_sha256"]
    with gzip.open(CURVES, "rt", encoding="utf8") as f:
        curves = [row for row in map(json.loads,f) if row["family"] == "physics"]
    measured = {(e["parent_seed"],e["reset_seed"],e["arm"]):e for e in curves}
    assert len(measured) == len(curves) == 18
    order = [(p,r,a) for p in plan["parents"] for r in plan["resets"] for a in plan["arms"]]
    assert measured.keys() == set(order)
    assert [(e["parent_seed"],e["reset_seed"],e["arm"]) for e in manifest["cases"]] == order
    assert manifest["reset_groups"] == {str(r):[i for i,k in enumerate(order) if k[1] == r] for r in plan["resets"]}
    assert manifest["parent_groups"] == {str(p):[i for i,k in enumerate(order) if k[0] == p] for p in plan["parents"]}
    diagonal = set(zip(plan["parents"],plan["resets"],strict=True))
    masks = np.array([1,2,4,8],np.uint8)
    with np.load(BUNDLE,allow_pickle=False) as data:
        assert set(data.files) == manifest["arrays"].keys() and len(data.files) == len(manifest["arrays"])
        payload = {k:data[k] for k in data.files}
    for k,arr in payload.items():
        assert arr.dtype.kind != "O"
        assert manifest["arrays"][k] == dict(dtype=arr.dtype.str,shape=list(arr.shape),
                                              sha256=hashlib.sha256(arr.tobytes()).hexdigest())
    assert payload["protocol_sha256"].item() == pack.PLAN_SHA and payload["evaluation_only"].item() is True
    assert list(zip(payload["parent_seed"],payload["reset_seed"],payload["arm"])) == order
    np.testing.assert_array_equal(payload["state_ticks"],np.arange(51))
    np.testing.assert_array_equal(payload["interval_end_ticks"],np.arange(1,51))
    assert payload["state45"].shape == (18,51,45) and payload["state45"].dtype == np.float64
    assert payload["applied16"].shape == payload["issued16"].shape == payload["policy_raw16"].shape == (18,50,16)
    assert payload["applied16"].dtype == payload["issued16"].dtype == payload["policy_raw16"].dtype == np.float32
    assert payload["support_mask"].shape == (18,50,4) and payload["support_mask"].dtype == np.bool_
    np.testing.assert_array_equal(payload["wheel_qvel_columns"],[9,13,17,21])
    np.testing.assert_array_equal(payload["wheel_action_columns"],[3,7,11,15])
    np.testing.assert_array_equal(payload["wheel_order"],["FL","FR","RL","RR"])
    first = {}
    for i,(p,r,a) in enumerate(order):
        case, curve = manifest["cases"][i], measured[p,r,a]
        assert case["row"] == i and case["source_round"] == (90 if (p,r) in diagonal else 93)
        assert case["source_ticks"] == (100 if (p,r) in diagonal else 50)
        assert case["source_trace_sha256"] == curve["trace_sha256"] == payload["source_trace_sha256"][i]
        assert case["policy_sha256"] == curve["policy_sha256"] == plan["policies_sha256"][str(p)] == payload["policy_sha256"][i]
        assert case["initial_state_sha256"] == curve["initial_state_observation_sha256"]
        state, applied, issued = (payload[k][i] for k in ("state45","applied16","issued16"))
        assert hashlib.sha256(state[0].tobytes()).hexdigest() == curve["first_state_sha256"]
        assert hashlib.sha256(state[1].tobytes()).hexdigest() == curve["first_endpoint_sha256"]
        assert hashlib.sha256(applied[0].tobytes()).hexdigest() == curve["first_applied_sha256"]
        assert hashlib.sha256(applied.tobytes()).hexdigest() == curve["applied_sha256"]
        common = (state[:2].tobytes(),applied[0].tobytes(),case["initial_state_sha256"])
        assert first.setdefault(r,common) == common
        assert not np.any(applied[0]) and applied[1:].tobytes() == issued[:-1].tobytes()
        if a == "original":
            np.testing.assert_array_equal(issued,payload["policy_raw16"][i])
        assert curve["ticks"] == payload["state_ticks"].tolist()
        np.testing.assert_array_equal(state[:,0],curve["world_x_m"])
        np.testing.assert_array_equal(state[:,23],curve["world_vx_m_s"])
        np.testing.assert_allclose(payload["body_vx_m_s"][i],curve["body_vx_m_s"][1:],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(payload["support_mask"][i]@masks,curve["loaded_mask"])
        np.testing.assert_array_equal(payload["pitch_rad"][i],curve["pitch_rad"])
        np.testing.assert_array_equal(payload["wheel_speed_rad_s"][i],curve["wheel_speed_rad_s"])
        np.testing.assert_array_equal(state[1:,23+payload["wheel_qvel_columns"]],payload["wheel_speed_rad_s"][i])
        np.testing.assert_array_equal(applied[:,payload["wheel_action_columns"]].astype(np.float64)*5.,payload["wheel_target_rad_s"][i])
    assert len({x[0] for x in first.values()}) == 3
