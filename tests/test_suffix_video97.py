"""Verify the scored suffix, native footage identity and both fixed-tape interventions."""
from hashlib import sha256
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from test_policy_direction95_video import mp4_box
from tools97 import video97 as v


def arm(states, source_hash="a"*64):
    return SimpleNamespace(trace=SimpleNamespace(states=states, ticks=len(states)-1), info={"source_sha256":source_hash})


def test_suffix_gain_excludes_cut_and_uses_the_physical_reset_even_for_anchored_trajectories():
    native = np.zeros((33,45),dtype=np.float64)
    native[:,0] = 1.0
    native[16,0] = 100.
    base = native.copy()
    native[17:,0] = np.tile([1.01,.99],8)
    base[17:,0] = np.tile([1.02,.96],8)
    plus, zero = (v.recorded_suffix(arm(s)) for s in (native,base))
    actual = v.gains(plus,zero,1.0)
    assert len(actual) == 16 and actual[0] == pytest.approx(10.) and actual[-1] == pytest.approx(20.)
    assert len(v.gains(plus,v.Branch(zero.x[:3],{}),1.0)) == 3


def test_suffix_binds_action_tape_cut_state_model_and_available_steps(tmp_path):
    seed,reset,tag = 94011,96201,"p1"
    tape = np.arange(16*16,dtype=np.float32).reshape(16,16)/300.
    real = np.zeros((33,45),np.float64)
    real[:,3],real[:,0] = 1.,1.1000000111
    model = real.astype(np.float32)
    model[16,0] -= .04
    a, f = arm(real),arm(model,"b"*64)
    checkpoint = "c"*64
    for stage, n in (("fixed",16),("anchored",3)):
        folder = tmp_path/stage
        where = folder/f"seed_{seed}/reset_{reset}/paired"
        where.mkdir(parents=True)
        start = (model if stage=="fixed" else real)[16].astype(np.float32)
        states = np.repeat(start[None,:],n+1,axis=0)
        states[1:,0] -= .001*np.arange(1,n+1)
        attempts = n if n==16 else n+1
        file,receipt = where/"p1.npz",where/"p1.json"
        np.savez(file,state45=states,applied16=tape[:attempts])
        row = dict(stage=stage,seed=seed,reset=reset,model="paired",tag=tag,alpha=1.,cut_tick=16,end_tick=32,
                   status="COMPLETE" if n==16 else "PARTIAL",error=None if n==16 else "predicted fall",
                   ticks=n,attempted_model_steps=attempts,retained_model_steps=n,state_file_sha256=v.digest(file),
                   native_trace_sha256=a.info["source_sha256"],forecast96_sha256=f.info["source_sha256"],
                   checkpoint_sha256=checkpoint,start_state_sha256=v.f32sha(start),applied_tape_sha256=v.f32sha(tape))
        receipt.write_text(json.dumps(row),encoding="utf-8")
        seal = {"files":{str(p.relative_to(folder)):v.digest(p) for p in (file,receipt)}}
        branch = v.suffix_branch(folder,seal,{},stage,seed,reset,tag,a,f,tape,checkpoint)
        assert branch.info["scored_ticks"] == n and np.array_equal(branch.x,states[1:,0])
        assert branch.info["status"] == row["status"]
        for wrong_start, wrong_tape, wrong_checkpoint in ((real[16].astype(np.float32)+.125,tape,checkpoint),
                                                         (start,tape[::-1],checkpoint),(start,tape,"d"*64)):
            row["start_state_sha256"] = v.f32sha(wrong_start)
            receipt.write_text(json.dumps(row),encoding="utf-8")
            seal["files"][str(receipt.relative_to(folder))] = v.digest(receipt)
            with pytest.raises(ValueError,match="start, actions or world"):
                v.suffix_branch(folder,seal,{},stage,seed,reset,tag,a,f,wrong_tape,wrong_checkpoint)


def test_measured_footage_requires_all_cells_in_order_and_exact_frame_spans():
    cells = []
    for i,(s,r) in enumerate((s,r) for s in (1,2,3) for r in (10,20,30)):
        cells.append(dict(seed=s,reset=r,panels={"native":{"p1":{"ticks":32}}},
                          first_frame=51*i,endpoint_frame=51*i+32,last_frame=51*i+50))
    source = dict(status="COMPLETE",fps=25,cover_frames=0,displayed_alpha=1,reference_alpha=0,cells=cells,
                  endpoint_hold_frames=18,frames=459,stream={"nb_read_frames":"459"})
    order = [(r["seed"],r["reset"]) for r in cells]
    v.timeline(source,order)
    with pytest.raises(ValueError,match="registered order"):
        v.timeline(source,order[::-1])
    cells[4]["first_frame"] += 1
    with pytest.raises(ValueError,match="clip mapping"):
        v.timeline(source,order)


def test_public_movie_curves_sources_safety_and_frame_table():
    root=Path(__file__).resolve().parents[1]
    movie=root/"docs/assets/suffix_replay97.mp4"
    receipt=root/"reports/suffix_replay97_video.json"
    result=root/"reports/suffix_replay97.json"
    curves=root/"reports/suffix_replay97_curves.npz"
    if not any(p.exists() for p in (movie,receipt,result,curves)):
        pytest.skip("round-97 diagnostic movie has not been published")
    video, report = (json.loads(p.read_bytes()) for p in (receipt,result))
    check96=json.loads((root/"reports/paired_response96_video.json").read_bytes())
    old_curves=root/"reports/paired_response96_curves.npz"
    assert v.digest(root/"docs/assets/paired_response96.mp4")==video["source_video_sha256"]==check96["video_sha256"]
    assert v.digest(root/"reports/paired_response96_video.json")==video["source_video_receipt_sha256"]
    assert video["status"]=="COMPLETE" and video["model"]=="paired" and video["protocol_sha256"]==report["protocol_sha256"]
    assert video["scored_ticks"]==[17,32] and video["cut_tick"]==16 and video["displayed_alpha"]==1 and video["reference_alpha"]==0
    assert video["source_crop_xywh"]==[4,82,504,280]
    for key in ("cover_frames","new_physics_steps","forward_kinematics_calls","model_steps","policy_calls","training_updates"):
        assert video[key]==0
    for method in ("fixed","anchored"):
        assert video["stage_seals_sha256"][method]==report["stages"][method]["seal_sha256"]
    order=[(s,r) for s in (94011,94023,94037) for r in (96201,96202,96203)]
    assert [(c["seed"],c["reset"]) for c in video["cells"]]==order
    measured={(c["training_seed"],c["reset"]):c for c in report["cells"]}
    with np.load(old_curves,allow_pickle=False) as prev, np.load(curves,allow_pickle=False) as new:
        for c,source in zip(video["cells"],check96["cells"],strict=True):
            cell=measured[c["seed"],c["reset"]]
            assert c["physical_reset_x_m"]==cell["origin_x_m"]
            assert all(c[k]==source[k] for k in ("first_frame","endpoint_frame","last_frame"))
            for method,item in c["series"].items():
                x=[]
                for tag in v.TAGS:
                    arm=cell["arms"][tag]
                    row=arm["native"] if method=="native" else arm["models"]["paired"][method]
                    info=item[tag]
                    if method in ("fixed","anchored"):
                        assert info["source_sha256"]==row["state_file_sha256"] and info["receipt_sha256"]==row["receipt_sha256"]
                        for k in ("start_state_sha256","applied_tape_sha256","native_trace_sha256","forecast96_sha256","checkpoint_sha256"):
                            assert info.get(k)==row.get(k)
                        x.append(new[row["curve_key"]][1:])
                    else:
                        assert info["source_sha256"]==row["source_sha256"] and info["safe"]==row["safe"]
                        x.append(prev[row["curve_key"]][17:33])
                        if method=="native":
                            assert ("非法接触" in info["stop_reason"])==bool(row["flag_ticks"]["illegal"])
                    assert info["scored_ticks"]==row["ticks"]
                n=min(map(len,x))
                assert n==item["common_scored_ticks"]
                if n==16:
                    origin=c["physical_reset_x_m"]
                    plus,zero=(np.asarray(xx,dtype=np.float64) for xx in x)
                    assert item["suffix_gain_mm"]==pytest.approx(1000*(np.abs(zero-origin).mean()-np.abs(plus-origin).mean()),abs=1e-10)
                else:
                    assert item["suffix_gain_mm"] is None
    blob=movie.read_bytes()
    assert 0<len(blob)==video["bytes"]<=5_000_000 and sha256(blob).hexdigest()==video["video_sha256"]
    tables=memoryview(blob)
    for box in (b"moov",b"trak",b"mdia",b"minf",b"stbl"):
        tables=mp4_box(tables,box)
    assert struct.unpack_from(">I",mp4_box(tables,b"stsz"),8)[0]==video["decoded_native_crops"]==video["frames"]==check96["frames"]
    assert video["duration_s"]==pytest.approx(video["frames"]/25)
