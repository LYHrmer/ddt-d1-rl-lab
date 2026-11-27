"""Audit and publish complete native traces alongside every unmeasured registered case."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import ROOT, digest, read, write
from tools109.readback109 import copy_new
from tools111.control import SPEED_GAIN
from tools111.course import SEEDS, calibration_gain, step_bounds
from tools111.individual import ARMS, origin
from tools111.kinematics import mechanism, vendor_targets
from tools111.replay import check_episode
from tools111.run import BASE, PLAN, PLAN_SHA, complete, sources


PUBLIC = ROOT/"reports/lidar_compound111"
MOVIE = ROOT/"docs/assets/lidar111.mp4"


def approach(z, layout):
    if layout == "flat":
        return []
    front = z["front_extent_x"]
    edge = step_bounds(layout)[0]
    rows = []
    for offset in (-.35,-.25,-.15,-.05,.10):
        marker = edge+offset
        crossing = np.flatnonzero((front[:-1]<marker) & (front[1:]>=marker))
        if len(crossing):
            k = int(crossing[0])
            alpha = (marker-front[k])/(front[k+1]-front[k])
            wz = z["measured_relative_z"]
            height = 1000*(wz[k]+alpha*(wz[k+1]-wz[k]))
            rows.append(dict(offset_m=offset, wheel_relative_z_mm=height.tolist()))
    return rows


def details(z, row, chain):
    check_episode(z,row)
    if chain is not None:
        baseline, _ = chain.forward(vendor_targets(z["raw"]))
        issued, _ = chain.forward(vendor_targets(z["issued"]))
        measured, _ = chain.forward(z["qpos"][:,7:])
        np.testing.assert_allclose((issued-baseline)[:,:,2],z["issued_lift_fk"],atol=1e-9,rtol=0.)
        np.testing.assert_allclose(measured[:,:,2],z["measured_relative_z"],atol=1e-9,rtol=0.)
    x = z["xy"][:-1,0]
    ticks = np.arange(len(x))
    yaw = z["imu_yaw"]
    vx = np.sum(z["vxy"][:-1]*np.column_stack((np.cos(yaw),np.sin(yaw))),axis=1)
    odom = .087*np.median(z["encoder_wheel_rad_s"],axis=1)
    summary = {}
    windows = (("initial_stand",(ticks>=25)&(ticks<50)), ("approach_hold",(ticks>=150)&(ticks<170)),
               ("middle",(x>=3)&(x<=6)), ("post_course_hold",(x>7)&(ticks<450)), ("stop",ticks>=575))
    for name, select in windows:
        if np.any(select):
            summary[name] = dict(samples=int(select.sum()), body_forward_mean=float(vx[select].mean()),
                                 wheel_odom_mean=float(odom[select].mean()),
                                 actor_command_mean=float(z["actor_vx"][select].mean()),
                                 wheel_odom_median_abs_error=float(np.median(np.abs(odom[select]-vx[select]))))
    reset = hashlib.sha256(z["qpos"][0].tobytes()+z["qvel"][0].tobytes()).hexdigest()
    return dict(approach=approach(z,row["layout"]), speed_windows=summary, initial_sha256=reset)


def ledger():
    """Distinct stages and one-shot reservations; the interrupted development unit always stays incomplete."""
    from tools88.export88 import resources
    parent = origin()
    flat_out, flat, flat_usage = complete("flat")
    assert calibration_gain(flat["episodes"]) is None
    cases = []
    for row in flat["episodes"]:
        name = f"{row['seed']}_{row['arm']}"
        cases.append((flat_out/name, flat_out.name, "COMPLETE", row))
    resources_by_unit = {flat_out.name:flat_usage}
    for seed in SEEDS["development"]:
        for arm in ARMS:
            name = f"{seed}_{arm}"
            reserved = list(BASE.glob("development_??/"+name+"/reservation.json"))
            reserved += list(BASE.glob("individual_*/"+name+"/reservation.json"))
            if not reserved:
                cases.append((None,None,"NOT_STARTED",dict(seed=seed,arm=arm,layout="development")))
                continue
            if len(reserved) != 1:
                raise ValueError(f"duplicate reservation: {name}")
            trial = reserved[0].parent
            unit = trial.parent
            guard = BASE/("guard_"+unit.name)
            status = read(guard/"report.json")["status"]
            if status == "COMPLETE" and unit != parent:
                record = read(unit/"report.json")
                assert record["status"] == "COMPLETE" and record["sources"] == sources()
                assert record["episode"] == read(trial/"report.json")
                resources_by_unit[unit.name] = resources(guard)
            if (trial/"report.json").is_file():
                row = read(trial/"report.json")
                assert row["status"] == "COMPLETE" and row["seed"] == seed and row["arm"] == arm
                cases.append((trial,unit.name,status,row))
            else:
                cases.append((None,unit.name,status,read(reserved[0])))
    return cases, resources_by_unit


def audit_for_video(out):
    cases, usage = ledger()
    geometry, robot_sha = mechanism()
    slots = []
    for trial, unit, guard_status, row in cases:
        item = dict(unit=unit, unit_status=guard_status, seed=row["seed"], arm=row["arm"], layout=row["layout"])
        if trial is None:
            item["case_status"] = "NOT_STARTED" if unit is None else "PARTIAL"
        else:
            file = trial/"trace.npz"
            assert digest(file) == row["trace_sha256"]
            with np.load(file,allow_pickle=False) as z:
                stats = details(dict(z),row,geometry)
            item.update(case_status="COMPLETE",episode=row,measurements=stats,
                        file=f"trajectories/{row['seed']}_{row['arm']}.npz")
        slots.append(item)
    if any(s["case_status"] == "NOT_STARTED" for s in slots):
        raise ValueError("a registered development cell is still pending")
    guards = {}
    for path in sorted(BASE.glob("guard_*/report.json")):
        r = read(path)
        if r["status"] != "COMPLETE":
            guards[path.parent.name] = {k:r[k] for k in ("status","error","child_started","minimum_effective_bytes","signals")}
            guards[path.parent.name]["sha256"] = digest(path)
    proof = dict(cases=slots, complete_units=usage, incomplete_units=guards, sources=sources(), robot_sha256=robot_sha,
                 plan_sha256=PLAN_SHA, development_stage_status="INCOMPLETE",blind_trials=0,policy_updates=0,world_updates=0)
    write(out/"validated.json",proof)
    return digest(out/"validated.json")


def install(source, target, sha):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with TemporaryDirectory(dir=target.parent) as temp:
            staged = Path(temp)/target.name
            copy_new(source,staged)
            assert digest(staged) == sha
            try:
                os.link(staged,target)
            except FileExistsError:
                pass
    if digest(target) != sha:
        raise ValueError(f"preserve existing different file {target}")


def publish():
    folder, film, usage = complete("video")
    assert digest(folder/"validated.json") == film["validated_sha256"]
    record = read(folder/"validated.json")
    reservations = {}
    for slot in record["cases"]:
        name = f"{slot['seed']}_{slot['arm']}"
        if slot["case_status"] == "COMPLETE":
            install(BASE/slot["unit"]/name/"trace.npz", PUBLIC/slot["file"], slot["episode"]["trace_sha256"])
        elif slot["case_status"] == "PARTIAL":
            receipt = BASE/slot["unit"]/name/"reservation.json"
            file = f"reservations/{name}.json"
            reservations[name] = dict(file=file,sha256=digest(receipt))
            install(receipt,PUBLIC/file,reservations[name]["sha256"])
    for unit, facts in record["incomplete_units"].items():
        install(BASE/unit/"report.json",PUBLIC/"resources"/(unit+".json"),facts["sha256"])
    install(folder/film["filename"], MOVIE, film["sha256"])
    with TemporaryDirectory(dir=PUBLIC) as temp:
        manifest = Path(temp)/"native.json"
        write(manifest,{**record,"partial_reservations":reservations,"video":film,"video_resource":usage})
        install(manifest,PUBLIC/manifest.name,digest(manifest))


def verify():
    proof = read(PUBLIC/"native.json")
    assert proof["plan_sha256"] == digest(PLAN) == PLAN_SHA and proof["sources"] == sources()
    assert proof["development_stage_status"] == "INCOMPLETE" and proof["blind_trials"] == 0
    expected = [(s,a) for s in SEEDS["flat"] for a in ("heading_off","heading_on")]
    expected += [(s,a) for s in SEEDS["development"] for a in ARMS]
    assert [(r["seed"],r["arm"]) for r in proof["cases"]] == expected
    # The unit is incomplete even where individual traces can be independently reconstructed.
    assert proof["incomplete_units"]["guard_development_01"]["status"] != "COMPLETE"
    for unit, facts in proof["incomplete_units"].items():
        path = PUBLIC/"resources"/(unit+".json")
        assert digest(path) == facts["sha256"]
        receipt = read(path)
        assert all(receipt[k] == v for k,v in facts.items() if k != "sha256")
    totals = dict(decisions=0, servo=0, physics=0, packets=0, ray_queries=0)
    verified = []
    partial = 0
    for slot in proof["cases"]:
        if slot["case_status"] != "COMPLETE":
            assert "file" not in slot and slot["case_status"] in ("PARTIAL","NOT_STARTED")
            if slot["case_status"] == "PARTIAL":
                partial += 1
                info = proof["partial_reservations"][f"{slot['seed']}_{slot['arm']}"]
                path = PUBLIC/info["file"]
                assert digest(path) == info["sha256"]
                assert read(path) == {**{k:slot[k] for k in ("seed","arm","layout")},"gain":SPEED_GAIN}
            continue
        row = slot["episode"]
        assert row["status"] == "COMPLETE" and row["gain"] == SPEED_GAIN
        path = PUBLIC/slot["file"]
        assert digest(path) == row["trace_sha256"]
        with np.load(path,allow_pickle=False) as z:
            check = details(dict(z),row,None)
        stored = slot["measurements"]
        assert check["initial_sha256"] == stored["initial_sha256"]
        for a,b in zip(check["approach"],stored["approach"],strict=True):
            assert a["offset_m"] == b["offset_m"]
            np.testing.assert_allclose(a["wheel_relative_z_mm"],b["wheel_relative_z_mm"],atol=1e-8,rtol=0.)
        assert check["speed_windows"].keys() == stored["speed_windows"].keys()
        for key, stats in check["speed_windows"].items():
            for name,value in stats.items():
                assert abs(value-stored["speed_windows"][key][name]) < 1e-9
        for key in totals:
            totals[key] += row["counts"][key]
        verified.append(row)
    for seed in {s for s,_ in expected}:
        resets = {r["measurements"]["initial_sha256"] for r in proof["cases"]
                  if r["seed"] == seed and r["case_status"] == "COMPLETE"}
        assert len(resets) <= 1, f"initial state differs across arms for seed {seed}"
    assert totals["decisions"]+600*partial <= 14400 and totals["servo"] == 5*totals["decisions"]
    assert totals["physics"]+6000*partial <= 144000 and totals["ray_queries"]+74880*partial <= 1797120
    assert proof["policy_updates"] == proof["world_updates"] == 0
    video = proof["video"]
    assert video["physical_substeps"] == 0 and MOVIE.stat().st_size == video["bytes"] and digest(MOVIE)==video["sha256"]
    for clip in video["clips"]:
        row, = [r for r in verified if r["seed"] == clip["seed"] and r["arm"] == clip["arm"]]
        assert clip["trace_sha256"] == row["trace_sha256"] and clip["end_tick"] == row["motion"]["ticks"]
    return dict(complete_traces=len(verified), partial_cases=partial, confirmed=totals, blind_trials=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publish",action="store_true")
    if p.parse_args().publish:
        publish()
    print(json.dumps(verify()))
