"""Check forecast-versus-physics replay using small sealed files and no renderer."""
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from tools96 import video96 as v
from test_policy_direction95_video import mp4_box


def put_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def seal(root):
    files = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in root.rglob("*")
             if p.is_file() and p.name != "seal.json"}
    put_json(root/"seal.json", dict(protocol_sha256="f"*64, files=files))


@pytest.fixture
def sources(tmp_path):
    native, predicted = tmp_path/"native", tmp_path/"predicted"
    native.mkdir()
    predicted.mkdir()
    seed, reset = v.SEEDS[0], v.RESETS[0]
    key = f"seed_{seed}/reset_{reset}"
    physical = np.zeros(45, dtype=np.float64)
    physical[0], physical[3] = 1.1000000111, 1.
    np.savez(predicted/f"start_{reset}.npz", state45=physical[None, :])
    for source, speed1, speed0 in (("native", .001, .002), ("control", .001, .004), ("paired", .003, .002)):
        for tag, speed in (("p1", speed1), ("zero", speed0)):
            early = source == "control" and tag == "p1"
            terminal = source == "native" and tag == "p1"
            n = 14 if early else 32
            states = np.repeat(physical[None,:], n+1, axis=0)
            states[1:,0] -= np.arange(1, n+1) * speed
            if source != "native":
                states = states.astype(np.float32)
            score = -float(np.abs(states[1:,0].astype(np.float64)-physical[0]).mean()) if n == 32 else None
            if source == "native":
                folder = native/key/tag
                (folder/"native").mkdir(parents=True)
                path = folder/"native/trace.npz"
                flags = {k: np.zeros(n, dtype=bool) for k in ("fallen","illegal","terminated","truncated")}
                flags["illegal"][-1] = flags["terminated"][-1] = terminal
                np.savez(path, state45=states, completed_transitions=n, requested_ticks=32, explore_seed=-1,
                         wheel_offset_rad_s=0., reset_seed=reset, time_s=np.arange(n+1)*.02, **flags)
                put_json(folder/"native.json", {"safe": not terminal, "return_m": score, "trace_file": "native/trace.npz",
                                               "trace_sha256": sha256(path.read_bytes()).hexdigest()})
            else:
                folder = predicted/key/source
                folder.mkdir(parents=True, exist_ok=True)
                path = folder/f"{tag}.npz"
                np.savez(path, state45=states)
                put_json(folder/f"{tag}.json", dict(model=source, alpha=1 if tag=="p1" else 0, seed=seed,
                         reset=reset, safe=not early, error="predicted fall" if early else None, ticks=n,
                         state_sha256=sha256(path.read_bytes()).hexdigest(), return_m=score))
    seal(native)
    seal(predicted)
    return native, predicted


def test_each_prediction_uses_its_own_zero_and_contact_preserves_only_the_measured_window(sources):
    cells, used, proto = v.load_cells(*sources)
    assert len(cells) == 9 and {(c.seed,c.reset) for c in cells} == {(s,r) for s in v.SEEDS for r in v.RESETS}
    assert len(used) == 15 and proto == "f"*64
    assert cells[0].last_tick == 32
    real, control, paired = cells[0].panels
    assert np.allclose(real.gains, .5*np.arange(2,34))
    assert real.summary()["h32_gain_mm"] == pytest.approx(16.5)
    assert real.summary()["h32_both_safe"] is False and real.p1.trace.stop_reason == "非法接触"
    assert control.p1.trace.stop_reason == "预测跌倒" and len(control.gains) == 14
    assert control.summary()["h32_gain_mm"] is None
    assert control.gains[-1] == pytest.approx(22.5, abs=.0001)
    assert paired.summary()["h32_gain_mm"] == pytest.approx(-16.5, abs=.0001)
    assert paired.summary()["h32_both_safe"] is True
    assert all(p.p1.trace is None for cell in cells[1:] for p in cell.panels)


def test_changed_file_and_swapped_model_identity_are_detected(sources):
    native, predicted = sources
    path = predicted/f"seed_{v.SEEDS[0]}/reset_{v.RESETS[0]}/control/p1.json"
    row = json.loads(path.read_bytes())
    row["model"] = "paired"
    put_json(path, row)
    with pytest.raises(ValueError, match="stage seal"):
        v.load_cells(native,predicted)
    seal(predicted)
    with pytest.raises(ValueError, match="arm, policy seed or reset"):
        v.load_cells(native,predicted)


def test_no_forecast_motion_produces_a_receipt_without_a_standin_movie(tmp_path):
    native, predicted = tmp_path/"native", tmp_path/"predicted"
    for root in (native,predicted):
        root.mkdir()
        seal(root)
    target, receipt = tmp_path/"new.mp4", tmp_path/"receipt.json"
    report = v.render(native,predicted,target,receipt)
    assert report["status"] == "NO_MATCHED_NATIVE_AND_MODEL_MOTION" and len(report["cells"]) == 9
    assert report["cover_frames"] == report["new_physics_steps"] == report["model_steps"] == 0
    assert not target.exists() and json.loads(receipt.read_bytes()) == report
    with pytest.raises(FileExistsError):
        v.render(native,predicted,target,receipt)


def test_public_movie_matches_blind_forecasts_native_gains_and_source_hashes():
    root = Path(__file__).resolve().parents[1]
    report = root/"reports/paired_response96.json"
    receipt = root/"reports/paired_response96_video.json"
    curves = root/"reports/paired_response96_curves.npz"
    movie = root/"docs/assets/paired_response96.mp4"
    if not any(p.exists() for p in (report, receipt, curves, movie)):
        pytest.skip("world-model comparison has not been published in this checkout")
    results = json.loads(report.read_bytes())
    replay = json.loads(receipt.read_bytes())
    assert replay["protocol_sha256"] == results["protocol_sha256"]
    for stage in ("native", "predictions"):
        assert replay[stage+"_seal_sha256"] == results["stages"][stage]["seal_sha256"]
    assert replay["displayed_alpha"] == 1 and replay["reference_alpha"] == 0
    assert all(replay[k] == 0 for k in ("cover_frames", "new_physics_steps", "model_steps", "policy_calls", "training_updates"))
    expected = {(s,r) for s in v.SEEDS for r in v.RESETS}
    measured = {(c["training_seed"],c["reset"]): c for c in results["cells"]}
    assert len(results["cells"]) == len(replay["cells"]) == 9
    assert set(measured) == {(c["seed"],c["reset"]) for c in replay["cells"]} == expected
    assert sha256(curves.read_bytes()).hexdigest() == results["curves_sha256"]
    frames = 0
    with np.load(curves, allow_pickle=False) as data:
        for c in replay["cells"]:
            source = measured[c["seed"],c["reset"]]
            assert set(c["panels"]) == set(v.KINDS)
            for kind, p in c["panels"].items():
                rows = [source["arms"][tag][kind] for tag in ("p1","zero")]
                for tag,row in zip(("p1","zero"),rows,strict=True):
                    for field in ("source_sha256","receipt_sha256"):
                        assert p[tag].get(field) == row[field]
                if p["common_ticks"] == v.HORIZON:
                    origin = source["origin_x_m"]
                    a,b = (np.asarray(data[row["curve_key"]], dtype=np.float64) for row in rows)
                    gain = 1000*(np.abs(b[1:33]-origin).mean()-np.abs(a[1:33]-origin).mean())
                    assert p["h32_gain_mm"] == pytest.approx(gain, abs=1e-10)
                    assert p["h32_both_safe"] == all(row["safe"] for row in rows)
                else:
                    assert p["h32_gain_mm"] is None and p["h32_both_safe"] is False
            if replay["status"] == "COMPLETE":
                assert c["first_frame"] == frames
                assert c["endpoint_frame"] == frames + max(p["p1"].get("ticks",0) for p in c["panels"].values())
                assert c["last_frame"] == c["endpoint_frame"]+replay["endpoint_hold_frames"]
                frames = c["last_frame"]+1
    if replay["status"] == "NO_MATCHED_NATIVE_AND_MODEL_MOTION":
        assert replay["matched_cells"] == 0 and not movie.exists()
        return
    assert replay["status"] == "COMPLETE" and replay["file"] == movie.name and replay["matched_cells"] > 0
    blob = movie.read_bytes()
    assert 0 < len(blob) == replay["bytes"] <= 5_000_000
    assert sha256(blob).hexdigest() == replay["video_sha256"]
    tables = memoryview(blob)
    for name in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
        tables = mp4_box(tables,name)
    assert struct.unpack_from(">I",mp4_box(tables,b"stsz"),8)[0] == frames == replay["frames"]
    avc = mp4_box(mp4_box(tables,b"stsd")[8:], b"avc1")
    assert struct.unpack_from(">HH",avc,24) == (v.WIDTH,v.HEIGHT)
    assert replay["duration_s"] == pytest.approx(frames/v.FPS)
