"""Check new holdouts, frame provenance, and independent full-horizon forecasts."""
from hashlib import sha256
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from test_policy_direction95_video import mp4_box
from tools98 import video98 as v


SEEDS = (94011, 94023, 94037)
RESETS = (98201, 98202, 98203)
PLAN_SHA = "f"*64


def save_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def seal(folder, stage, previous=None):
    files = {str(p.relative_to(folder)): v.digest(p) for p in folder.rglob("*") if p.is_file() and p.name != "seal.json"}
    path = folder/"seal.json"
    save_json(path, dict(stage=stage, files=files, protocol_sha256=PLAN_SHA, previous_seal_sha256=previous))
    return v.digest(path)


@pytest.fixture
def sealed_pair(tmp_path):
    native, predictions = tmp_path/"native", tmp_path/"predictions"
    fit = tmp_path/"fit_01"
    native.mkdir()
    predictions.mkdir()
    reset, seed = RESETS[0], SEEDS[0]
    start = np.zeros(45, dtype=np.float64)
    start[0], start[3] = 1.100000013, 1.
    np.savez(predictions/f"start_{reset}.npz", state45=start[None])
    rel = f"seed_{seed}/reset_{reset}"
    for source in v.KINDS:
        if source != "native":
            trained = fit/f"seed_{seed}"/source
            trained.mkdir(parents=True)
            (trained/"world.pt").write_bytes(f"synthetic {source}".encode())
            checkpoint = v.digest(trained/"world.pt")
            save_json(trained/"receipt.json", dict(status="COMPLETE", updates_completed=25, checkpoint={"sha256":checkpoint}))
        for tag in ("p1", "zero"):
            old_terminal = source == "native" and tag == "p1"
            predicted_stop = source == "control" and tag == "p1"
            n = 14 if predicted_stop else 32
            states = np.repeat(start[None], n+1, axis=0)
            speed_mm = 1. if tag == "p1" else 3. if source == "endpoint" else 2.
            states[:, 0] -= np.arange(n+1) * speed_mm/1000
            if source == "endpoint" and tag == "p1":
                states[16, 1:3] += [.003, .004]
            if source != "native":
                states = states.astype(np.float32)
            reward = -float(np.abs(states[1:, 0].astype(np.float64)-start[0]).mean()) if n == 32 else None
            if source == "native":
                folder = native/rel/tag
                (folder/"native").mkdir(parents=True)
                file = folder/"native/trace.npz"
                flags = {key: np.zeros(n, dtype=bool) for key in ("fallen", "illegal", "terminated", "truncated")}
                flags["illegal"][-1] = flags["terminated"][-1] = old_terminal
                np.savez(file, state45=states, completed_transitions=n, requested_ticks=32, explore_seed=-1,
                         wheel_offset_rad_s=0., reset_seed=reset, time_s=np.arange(n+1)*.02, **flags)
                save_json(folder/"native.json", dict(trace_file="native/trace.npz", trace_sha256=v.digest(file),
                                                     safe=not old_terminal, return_m=reward))
            else:
                folder = predictions/rel/source
                folder.mkdir(parents=True, exist_ok=True)
                file = folder/f"{tag}.npz"
                np.savez(file, state45=states)
                save_json(folder/f"{tag}.json", dict(model=source, alpha=1 if tag=="p1" else 0, seed=seed, reset=reset,
                         safe=not predicted_stop, error="predicted fall" if predicted_stop else None, ticks=n,
                         state_sha256=v.digest(file), checkpoint_sha256=checkpoint, return_m=reward))
    seal(native, "native", seal(predictions, "predictions", seal(fit, "fit")))
    return native, predictions


def test_holdouts_compute_own_zero_gain_and_heldout_body_xyz_in_mm(sealed_pair):
    cases, used, hashes = v.load_cases(*sealed_pair, SEEDS, RESETS, PLAN_SHA)
    assert [(c.seed,c.reset) for c in cases] == [(s,r) for s in SEEDS for r in RESETS]
    assert len(used) == 20 and set(hashes) == {"fit", "native", "predictions"}
    assert all(len(h) == 64 for h in hashes.values())
    real, ctrl, endpoint = [cases[0].panels[k] for k in v.KINDS]
    snap = cases[0].snapshot()["panels"]
    assert real.summary()["h32_gain_mm"] == pytest.approx(16.5)
    assert not real.summary()["h32_both_safe"] and v.arm_note(real.p1, 32).startswith("非法接触")
    assert v.arm_note(real.p1, 31) == v.arm_note(real.zero, 32) == ""
    assert ctrl.summary()["h32_gain_mm"] is None and len(ctrl.gains) == 14
    assert snap["control"]["h16_p1_body_position_error_mm"] is None
    assert snap["endpoint"]["h16_p1_body_position_error_mm"] == pytest.approx(5., abs=1e-6)
    assert endpoint.summary()["h32_gain_mm"] == pytest.approx(33., abs=.0001)
    assert v.gain_scale(cases) >= max(max(abs(p.gains)) for p in (real, ctrl, endpoint))
    assert all(c.last_tick == 0 for c in cases[1:])


def test_missing_or_mixed_blind_stages_detected_before_replay(sealed_pair):
    native, predictions = sealed_pair
    seal(native, "native", "wrong")
    with pytest.raises(ValueError, match="different blind forecasts"):
        v.load_cases(native, predictions, SEEDS, RESETS, PLAN_SHA)
    seal(native, "native", v.digest(predictions/"seal.json"))
    path = predictions/f"seed_{SEEDS[0]}/reset_{RESETS[0]}/endpoint/p1.json"
    row = json.loads(path.read_bytes())
    row["model"] = "control"
    save_json(path, row)
    with pytest.raises(ValueError, match="stage seal"):
        v.load_cases(native, predictions, SEEDS, RESETS, PLAN_SHA)
    fit = native.parent/"fit_01"
    seal(native, "native", seal(predictions, "predictions", v.digest(fit/"seal.json")))
    with pytest.raises(ValueError, match="arm, policy seed or reset"):
        v.load_cases(native, predictions, SEEDS, RESETS, PLAN_SHA)
    row.update(model="endpoint", checkpoint_sha256="0"*64)
    save_json(path, row)
    seal(native, "native", seal(predictions, "predictions", v.digest(fit/"seal.json")))
    with pytest.raises(ValueError, match="different world checkpoint"):
        v.load_cases(native, predictions, SEEDS, RESETS, PLAN_SHA)


def test_older_reset_or_different_origin_cannot_stand_in_for_a_new_holdout(sealed_pair):
    native, predictions = sealed_pair
    path = predictions/f"start_{RESETS[0]}.npz"
    with np.load(path, allow_pickle=False) as z:
        state = z["state45"].copy()
    state[0, 0] += .01
    np.savez(path, state45=state)
    seal(native, "native", seal(predictions, "predictions", v.digest(native.parent/"fit_01/seal.json")))
    with pytest.raises(ValueError, match="different physical reset"):
        v.load_cases(native, predictions, SEEDS, RESETS, PLAN_SHA)


def test_proposed_but_rejected_sixteenth_state_has_no_xyz_score():
    arm = SimpleNamespace(trace=SimpleNamespace(ticks=16))
    row = dict(ticks=16, safe=False, attempted_model_steps=16, error="predicted fall")
    assert not v.accepted_h16(arm, row)
    row.update(attempted_model_steps=17, error="next step failed before producing a state")
    assert v.accepted_h16(arm, row)


def test_public_movie_crosschecks_all_holdouts_scores_and_frame_table():
    root = Path(__file__).resolve().parents[1]
    movie, receipt = root/"docs/assets/endpoint98.mp4", root/"reports/endpoint98_video.json"
    report, curves = root/"reports/endpoint98.json", root/"reports/endpoint98_curves.npz"
    if not any(p.exists() for p in (movie, receipt, report, curves)):
        pytest.skip("new blind-test results have not been published")
    video, trial = (json.loads(p.read_bytes()) for p in (receipt,report))
    assert video["protocol_sha256"] == trial["protocol_sha256"] and v.digest(curves) == trial["curves_sha256"]
    assert video["displayed_native_alphas"] == [0.,1.]
    assert all(video[k] == 0 for k in ("cover_frames","rendered_model_poses","new_physics_steps","model_steps","policy_calls","training_updates"))
    for stage in ("fit","native","predictions"):
        assert video["stage_seals_sha256"][stage] == trial["stages"][stage]["seal_sha256"]
    assert [(c["seed"],c["reset"]) for c in video["cells"]] == [(s,r) for s in SEEDS for r in RESETS]
    by_key = {(c["training_seed"],c["reset"]): c for c in trial["cells"]}
    fits = {f["seed"]: f for f in trial["fits"]}
    pos = poses = 0
    with np.load(curves,allow_pickle=False) as data:
        for cell in video["cells"]:
            actual = by_key[cell["seed"], cell["reset"]]
            assert set(cell["panels"]) == set(v.KINDS)
            last_tick = 0
            for kind, panel in cell["panels"].items():
                raw = [actual["arms"][tag][kind] for tag in ("p1","zero")]
                for tag, src in zip(("p1","zero"), raw, strict=True):
                    a = panel[tag]
                    for field in ("source_sha256","receipt_sha256"):
                        assert a.get(field) == src[field]
                    if kind != "native" and a.get("receipt_sha256"):
                        assert a["checkpoint_sha256"] == fits[cell["seed"]]["arms"][kind]["checkpoint"]["sha256"]
                    if kind == "native" and a.get("source_sha256") and a.get("ticks") is not None:
                        flags = data[src["flags_key"]]
                        assert ("非法接触" in a["stop_reason"]) == bool(flags[:,1].any())
                        assert ("跌倒" in a["stop_reason"]) == bool(flags[:,0].any())
                    assert bool(a.get("h16_scoreable")) == (src["h16_status"] in ("OBSERVED","ACCEPTED"))
                    last_tick = max(last_tick, a.get("ticks", 0))
                n = panel["common_ticks"]
                if n == 32:
                    xp, xz = (data[r["curve_key"]].astype(np.float64) for r in raw)
                    origin = actual["origin_x_m"]
                    expected = 1000*(np.abs(xz[1:33]-origin).mean()-np.abs(xp[1:33]-origin).mean())
                    assert panel["h32_gain_mm"] == pytest.approx(expected, abs=1e-10)
                    assert panel["h32_both_safe"] == all(r["safe"] for r in raw)
                else:
                    assert panel["h32_gain_mm"] is None and not panel["h32_both_safe"]
                if kind != "native":
                    model16 = data[raw[0]["h16_key"]]
                    real16 = data[actual["arms"]["p1"]["native"]["h16_key"]]
                    if raw[0]["h16_status"] == "ACCEPTED" and actual["arms"]["p1"]["native"]["h16_status"] == "OBSERVED":
                        err = 1000*np.linalg.norm(model16.astype(np.float64)[:3]-real16[:3])
                        assert panel["h16_p1_body_position_error_mm"] == pytest.approx(err, abs=1e-9)
                    else:
                        assert panel["h16_p1_body_position_error_mm"] is None
            if video["status"] == "COMPLETE":
                assert cell["first_frame"] == pos and cell["endpoint_frame"] == pos+last_tick
                assert cell["last_frame"] == pos+last_tick+video["endpoint_hold_frames"]
                native = cell["panels"]["native"]
                poses += (1+last_tick)*sum(native[tag].get("ticks") is not None for tag in ("p1","zero"))
                pos = cell["last_frame"]+1
    if video["status"] == "NO_MATCHED_NATIVE_AND_MODEL_MOTION":
        assert not movie.exists() and video["matched_cells"] == 0
        return
    assert video["status"] == "COMPLETE" and video["matched_cells"] > 0 and video["rendered_native_poses"] == poses
    blob = movie.read_bytes()
    assert 0 < len(blob) == video["bytes"] <= 5_000_000 and sha256(blob).hexdigest() == video["video_sha256"]
    tables = memoryview(blob)
    for box in (b"moov",b"trak",b"mdia",b"minf",b"stbl"):
        tables = mp4_box(tables,box)
    assert struct.unpack_from(">I",mp4_box(tables,b"stsz"),8)[0] == pos == video["frames"]
    avc = mp4_box(mp4_box(tables,b"stsd")[8:], b"avc1")
    assert struct.unpack_from(">HH",avc,24) == (v.WIDTH,v.HEIGHT)
    assert video["duration_s"] == pytest.approx(pos/v.FPS)
