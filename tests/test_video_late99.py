"""Verify the two plotted windows independently against fresh native and model positions."""
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from test_policy_direction95_video import mp4_box
from tools99 import video99 as v


SEEDS, RESETS = (94011, 94023, 94037), (99201, 99202, 99203)
PROTOCOL = "a"*64


def save(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def seal(base, stage, previous=None):
    folder = base/(stage+"_01")
    files = {str(p.relative_to(folder)): v.digest(p) for p in folder.rglob("*") if p.is_file() and p.name != "seal.json"}
    target = folder/"seal.json"
    save(target, dict(stage=stage, protocol_sha256=PROTOCOL, files=files, previous_seal_sha256=previous))
    return v.digest(target)


@pytest.fixture
def grid(tmp_path):
    fit, pred, native = (tmp_path/(k+"_01") for k in ("fit","predictions","native"))
    for path in (fit,pred,native):
        path.mkdir()
    seed, reset = SEEDS[0], RESETS[0]
    start = np.zeros(45, dtype=np.float64)
    start[0], start[3] = 1.100000013, 1.
    np.savez(pred/f"start_{reset}.npz", state45=start[None])
    rel = f"seed_{seed}/reset_{reset}"
    for kind in v.KINDS:
        if kind != "native":
            folder = fit/f"seed_{seed}"/kind
            folder.mkdir(parents=True)
            (folder/"world.pt").write_bytes(("synthetic "+kind).encode())
            ckpt = v.digest(folder/"world.pt")
            save(folder/"receipt.json", dict(status="COMPLETE", updates_completed=25, checkpoint={"sha256":ckpt}))
        for tag in v.TAGS:
            early = kind == "control" and tag == "p1"
            contact = kind == "native" and tag == "p1"
            n = 19 if early else 32
            states = np.repeat(start[None],n+1,axis=0)
            states[1:16,0] -= .005 if tag == "p1" else .010
            states[16,0] += -.1 if tag == "p1" else .1
            if kind == "late":
                states[17:,0] -= .004 if tag == "p1" else .001
            else:
                states[17:,0] -= .002 if tag == "p1" else .004
            if kind != "native":
                states = states.astype(np.float32)
            ret = -float(np.abs(states[1:,0].astype(np.float64)-start[0]).mean()) if n == 32 else None
            if kind == "native":
                folder = native/rel/tag
                (folder/"native").mkdir(parents=True)
                path = folder/"native/trace.npz"
                events = {k: np.zeros(n,dtype=bool) for k in ("fallen","illegal","terminated","truncated")}
                events["illegal"][-1] = events["terminated"][-1] = contact
                np.savez(path,state45=states,requested_ticks=32,reset_seed=reset,explore_seed=-1,wheel_offset_rad_s=0.,
                         completed_transitions=n,time_s=np.arange(n+1)*.02,**events)
                save(folder/"native.json",dict(safe=not contact,return_m=ret,trace_file="native/trace.npz",trace_sha256=v.digest(path)))
            else:
                folder = pred/rel/kind
                folder.mkdir(parents=True,exist_ok=True)
                path = folder/f"{tag}.npz"
                np.savez(path,state45=states)
                save(folder/f"{tag}.json",dict(seed=seed,reset=reset,model=kind,alpha=1 if tag=="p1" else 0,
                     file=path.name,ticks=n,safe=not early,return_m=ret,error="predicted fall" if early else None,
                     state_sha256=v.digest(path),checkpoint_sha256=ckpt))
    seal(tmp_path,"native",seal(tmp_path,"predictions",seal(tmp_path,"fit")))
    return tmp_path


def test_same_physical_origin_both_windows_own_baseline_and_no_extrapolation(grid):
    cases, used, seals = v.load_cases(grid, SEEDS, RESETS, PROTOCOL)
    assert [(c.seed,c.reset) for c in cases] == [(s,r) for s in SEEDS for r in RESETS]
    assert len(used) == 20 and set(seals) == {"fit","predictions","native"}
    case = cases[0]
    rows = case.summary()["panels"]
    assert rows["native"]["h32_gain_mm"] == pytest.approx((15*5+16*2)/32)
    assert rows["native"]["h32_both_safe"] is False and "非法接触" in rows["native"]["p1"]["stop_reason"]
    assert rows["native"]["late_gain_mm_by_tick"] == pytest.approx([2.]*16)
    assert rows["late"]["late_gain_mm"] == pytest.approx(-3.,abs=.0001)
    assert rows["control"]["h32_gain_mm"] is None and rows["control"]["late_gain_mm"] is None
    assert rows["control"]["common_ticks"] == 19 and rows["control"]["late_common_ticks"] == 3
    assert v.reading(case.late["control"],4) == "缺少同期对照" and v.reading(case.late["control"],0) == "待计分"
    scale = v.shared_scale(cases)
    assert all(abs(val) <= scale for k,p in rows.items() for g in (p["full_gain_mm_by_tick"],p["late_gain_mm_by_tick"]) for val in g)
    assert all(c.end == 0 for c in cases[1:])


def test_changed_forecast_and_wrong_fit_cannot_be_displayed(grid):
    path = grid/f"predictions_01/seed_{SEEDS[0]}/reset_{RESETS[0]}/late/p1.json"
    data = json.loads(path.read_bytes())
    data["checkpoint_sha256"] = "4"*64
    save(path,data)
    with pytest.raises(ValueError,match="stage seal"):
        v.load_cases(grid,SEEDS,RESETS,PROTOCOL)
    seal(grid,"native",seal(grid,"predictions",v.digest(grid/"fit_01/seal.json")))
    with pytest.raises(ValueError,match="different model fit"):
        v.load_cases(grid,SEEDS,RESETS,PROTOCOL)
    seal(grid,"native","wrong")
    with pytest.raises(ValueError,match="sealed predecessor"):
        v.load_cases(grid,SEEDS,RESETS,PROTOCOL)


def test_forecast_cannot_be_relabelled_as_a_different_family_or_reset(grid):
    path = grid/f"predictions_01/seed_{SEEDS[0]}/reset_{RESETS[0]}/late/zero.json"
    data = json.loads(path.read_bytes())
    for wrong in (dict(model="control"),dict(model="late",reset=98201)):
        data.update(wrong)
        save(path,data)
        seal(grid,"native",seal(grid,"predictions",v.digest(grid/"fit_01/seal.json")))
        with pytest.raises(ValueError,match="arm, policy seed or reset"):
            v.load_cases(grid,SEEDS,RESETS,PROTOCOL)


def test_public_movie_all_plotted_points_sources_checkpoints_and_frames():
    root = Path(__file__).resolve().parents[1]
    movie, receipt = root/"docs/assets/late_response99.mp4", root/"reports/late_response99_video.json"
    report, curves = root/"reports/late_response99.json", root/"reports/late_response99_curves.npz"
    if not any(p.exists() for p in (movie,receipt,report,curves)):
        pytest.skip("late-response blind-test assets have not been published")
    video, trial = (json.loads(p.read_bytes()) for p in (receipt,report))
    assert v.digest(curves) == trial["curves_sha256"] and video["protocol_sha256"] == trial["protocol_sha256"]
    assert video["full_ticks"] == [1,32] and video["late_ticks"] == [17,32] and video["displayed_native_alphas"] == [0,1]
    assert [(c["seed"],c["reset"]) for c in video["cells"]] == [(s,r) for s in SEEDS for r in RESETS]
    assert all(video[k] == 0 for k in ("cover_frames","rendered_model_poses","new_physics_steps","model_steps","policy_calls","training_updates"))
    for stage in ("fit","predictions","native"):
        assert video["stage_seals_sha256"][stage] == trial["stages"][stage]["seal_sha256"]
    fits = {row["seed"]:row for row in trial["fits"]}
    by_cell = {(c["training_seed"],c["reset"]):c for c in trial["cells"]}
    frames = poses = 0
    with np.load(curves,allow_pickle=False) as data:
        for c in video["cells"]:
            ref = by_cell[c["seed"],c["reset"]]
            assert c["physical_origin_x_m"] == ref["origin_x_m"] and set(c["panels"]) == set(v.KINDS)
            end = 0
            for kind,p in c["panels"].items():
                xs, ticks, safes = [], [], []
                for tag in v.TAGS:
                    a, row = p[tag],ref["arms"][tag][kind]
                    for field in ("source_sha256","receipt_sha256"):
                        assert a.get(field) == row.get(field)
                    if kind != "native" and a.get("receipt_sha256"):
                        assert a["checkpoint_sha256"] == fits[c["seed"]]["arms"][kind]["checkpoint"]["sha256"]
                    if kind == "native" and a.get("ticks") is not None:
                        flags = data[row["flags_key"]]
                        assert ("非法接触" in a["stop_reason"]) == bool(flags[:,1].any())
                        assert ("跌倒" in a["stop_reason"]) == bool(flags[:,0].any())
                    xs.append(data[row["curve_key"]].astype(np.float64))
                    ticks.append(a.get("ticks",0))
                    safes.append(row["safe"])
                end = max(end,*ticks)
                for cut, field, endpoint, count in (
                        (0,"full_gain_mm_by_tick","h32_gain_mm","common_ticks"),
                        (16,"late_gain_mm_by_tick","late_gain_mm","late_common_ticks")):
                    n = max(0,min(ticks)-cut)
                    assert n == p[count] == len(p[field])
                    if n:
                        plus, zero = (np.abs(x[cut+1:cut+n+1]-c["physical_origin_x_m"]) for x in xs)
                        # Independent scalar averaging, including every intermediate movie point.
                        expected = [1000*(zero[:t].mean()-plus[:t].mean()) for t in range(1,n+1)]
                        assert np.allclose(p[field],expected,atol=1e-10,rtol=1e-10)
                        assert max(map(abs,expected)) <= video["gain_scale_mm"]
                    if n == 32-cut:
                        assert p[endpoint] == pytest.approx(expected[-1],abs=1e-10)
                    else:
                        assert p[endpoint] is None
                assert p["h32_both_safe"] == (p["common_ticks"] == 32 and all(safes))
            if video["status"] == "COMPLETE":
                assert c["first_frame"]==frames and c["endpoint_frame"]==frames+end
                assert c["last_frame"]==c["endpoint_frame"]+video["endpoint_hold_frames"]
                frames = c["last_frame"]+1
                poses += (end+1)*sum(c["panels"]["native"][t].get("ticks") is not None for t in v.TAGS)
    if video["status"] == "NO_MATCHED_NATIVE_AND_MODEL_MOTION":
        assert not movie.exists() and video["matched_cells"] == 0
        return
    assert video["status"] == "COMPLETE" and video["matched_cells"] > 0 and video["rendered_native_poses"] == poses
    image = movie.read_bytes()
    assert 0 < len(image) == video["bytes"] <= 5_000_000 and sha256(image).hexdigest() == video["video_sha256"]
    table = memoryview(image)
    for tag in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
        table = mp4_box(table,tag)
    assert struct.unpack_from(">I",mp4_box(table,b"stsz"),8)[0] == frames == video["frames"]
    assert struct.unpack_from(">HH",mp4_box(mp4_box(table,b"stsd")[8:],b"avc1"),24) == (v.WIDTH,v.HEIGHT)
    assert video["duration_s"] == pytest.approx(frames/v.FPS)
