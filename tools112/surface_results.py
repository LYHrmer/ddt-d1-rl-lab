"""Publish the matched measurements and independently reconstruct both scores from native states."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools88.export88 import resources
from tools91.launch91 import ROOT, digest, read, require, write
from tools111.control import SPEED_GAIN
from tools111.replay import check_episode
from tools111.results import install
from tools111.wheel_response import wheel_signals
from tools112.run_surface import recorded
from tools112.surface import BASE, PLAN_SHA, PROBES, PUBLIC, REFERENCE_SHA, SEEDS, bindings, paired_scores, reference


MOVIE = ROOT/"docs/assets/surface112.mp4"


def measure_stop(z):
    if len(z["time"]) != 601:
        return None
    yaw = z["imu_yaw"][575:]
    direction = np.column_stack((np.cos(yaw),np.sin(yaw)))
    dx = (np.diff(z["xy"][575:],axis=0)*direction).sum()
    target, _, torque = wheel_signals(z)
    return dict(forward_displacement_mm=float(dx*1000.),
                negative_targets_per_wheel=(target[575:]<0.).sum(axis=0).tolist(),
                negative_first_torques=int(np.count_nonzero(torque[575:]<0.)))


def compare(path, row):
    plane_file, plane_row = reference(row["seed"])
    vendor = read(ROOT/"reports/speed110/results.json")["preflight"]
    require(row["arm"] == "heading_on" and row["layout"] == "flat" and row["gain"] == SPEED_GAIN
            and row["surface"] == "level_heightfield" and row["reference_sha256"] == plane_row["trace_sha256"]
            and row["actor_sha256"] == vendor["model_sha256"] and row["actuator_config_sha256"] == vendor["config_sha256"]
            and digest(path) == row["trace_sha256"],"invalid native pair")
    with np.load(path,allow_pickle=False) as archive, np.load(plane_file,allow_pickle=False) as original:
        field, plane = dict(archive), dict(original)
    check_episode(plane,plane_row)
    check_episode(field,row)
    n = row["motion"]["ticks"]
    for key in ("qpos","qvel","raw","applied","first_torque_nm"):
        np.testing.assert_array_equal(field[key][0],plane[key][0])
    for key in ("time","base_cmd","actor_vx","capture_tick"):
        np.testing.assert_array_equal(field[key],plane[key][:len(field[key])])
    require(not np.any(field["applied_lift_fk"]) and not np.any(plane["applied_lift_fk"]),"leg preview changed")
    first_change = {}
    for key in ("qpos","qvel","raw","applied"):
        diff = np.max(np.abs(field[key]-plane[key][:len(field[key])]),axis=1)
        changed = np.flatnonzero(diff>1e-8)
        first_change[key] = int(changed[0]) if len(changed) else None
    wheel_xy = field["wheels_xy"]
    edge = min(float((wheel_xy[:,:,0]+5.).min()),float((17.-wheel_xy[:,:,0]).min()),
               float((2.2-np.abs(wheel_xy[:,:,1])).min()))
    require(edge > .087,"wheel close enough to sample the finite grid boundary")
    return dict(**paired_scores(plane_row["motion"],row["motion"]),ticks=n,
                plane_speed_m_s=plane_row["motion"]["speed_m_s"],field_speed_m_s=row["motion"]["speed_m_s"],
                plane_stop_peak_m_s=plane_row["motion"]["stop_tail_max_m_s"],
                field_stop_peak_m_s=row["motion"]["stop_tail_max_m_s"],
                plane_stop=measure_stop(plane),field_stop=measure_stop(field),
                first_change_tick=first_change,minimum_wheel_center_to_grid_boundary_m=edge)


def verify_survey(s):
    require(s["rays"] == 32 and s["physics"] == 0 and s["positions_m"] == [list(p) for p in PROBES]
            and s["grid_shape"] == [89,551],"invalid equal-height survey count or geometry")
    heights = np.array([s["heights_m"]["plane"],s["heights_m"]["heightfield"]])
    difference = float(np.max(np.abs(heights[0]-heights[1])))
    require(heights.shape == (2,16) and np.max(np.abs(heights)) <= .0001 and difference == s["max_difference_m"]
            and difference <= .0001,"unequal physical heights")
    _, ref = reference(SEEDS[0])
    require(s["contact_pose"]["seed"] == SEEDS[0] and s["contact_pose"]["tick"] == 575
            and s["contact_pose"]["trace_sha256"] == ref["trace_sha256"], "contact pose source changed")
    for wheels in s["contact_pose"]["contacts"].values():
        require(len(wheels) == 4 and all(wheels), "missing wheel contacts")
        normals = np.array([c["normal_floor_to_wheel"] for wheel in wheels for c in wheel])
        np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1.,atol=1e-8,rtol=0.)


def verdict(cases):
    measured = [c for c in cases if c["case_status"] == "COMPLETE"]
    full = len(measured) == len(SEEDS) and all(c["comparison"]["ticks"] == 600 for c in measured)
    return dict(complete_pairs=len(measured),cruise_supported=full and all(c["comparison"]["cruise_supported"] for c in measured),
                stop_supported=full and all(c["comparison"]["stop_supported"] for c in measured))


def install_json(target, content):
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as temp:
        file = Path(temp)/target.name
        write(file,content)
        install(file,target,digest(file))


def publish():
    _, survey_guard, s = recorded("survey")
    verify_survey(s["survey"])
    cases = []
    for seed in SEEDS:
        record = recorded(str(seed))
        if record is None:
            reservations = list(BASE.glob(f"{seed}_??/reservation.json"))
            require(len(reservations)<=1,"duplicate heightfield reservation")
            case = dict(seed=seed,case_status="PARTIAL" if reservations else "NOT_STARTED")
            if reservations:
                file = reservations[0]
                case.update(reservation_sha256=digest(file),unit=file.parent.name)
                install(file,PUBLIC/"reservations"/f"{seed}.json",digest(file))
        else:
            out, guard, r = record
            row = r["episode"]
            file = f"trajectories/heightfield_{seed}.npz"
            comparison = compare(out/"trace.npz",row)
            install(out/"trace.npz",PUBLIC/file,row["trace_sha256"])
            case = dict(seed=seed,case_status="COMPLETE",file=file,episode=row,comparison=comparison,
                        unit=out.name,resources=resources(guard))
        cases.append(case)
    failed = {}
    physical_guards = {"guard_survey"}
    physical_guards.update(f"guard_{seed}" for seed in SEEDS)
    for f in sorted(BASE.glob("guard_*/report.json")):
        if f.parent.name.rsplit("_",1)[0] in physical_guards and read(f)["status"] != "COMPLETE":
            failed[f.parent.name] = digest(f)
            install(f,PUBLIC/"resources"/(f.parent.name+".json"),digest(f))
    install_json(PUBLIC/"results.json",dict(sources=bindings(),proposal_sha256=PLAN_SHA,reference_manifest_sha256=REFERENCE_SHA,
                 survey=s["survey"],survey_resources=resources(survey_guard),cases=cases,verdict=verdict(cases),
                 failed_units=failed,policy_updates=0,world_updates=0,blind_trials=0))


def verify():
    report = read(PUBLIC/"results.json")
    require(report["sources"] == bindings() and report["proposal_sha256"] == PLAN_SHA
            and report["reference_manifest_sha256"] == REFERENCE_SHA,"unbound public results")
    verify_survey(report["survey"])
    cases = report["cases"]
    require([c["seed"] for c in cases] == list(SEEDS) and report["verdict"] == verdict(cases),"invalid diagnostic roster")
    totals = dict(decisions=0,servo=0,physics=0,packets=0,ray_queries=0)
    for c in cases:
        if c["case_status"] != "COMPLETE":
            require(c["case_status"] in ("PARTIAL","NOT_STARTED") and "comparison" not in c,"unmeasured score")
            if c["case_status"] == "PARTIAL":
                require(digest(PUBLIC/"reservations"/f"{c['seed']}.json") == c["reservation_sha256"],"missing reservation")
            continue
        row = c["episode"]
        require(row["seed"] == c["seed"] and row["status"] == "COMPLETE" and row["mujoco_version"] == "3.3.0",
                "wrong native source")
        check = compare(PUBLIC/c["file"],row)
        for key,val in check.items():
            stored = c["comparison"][key]
            if key in ("plane_stop","field_stop") and val is not None:
                np.testing.assert_allclose(val["forward_displacement_mm"],stored["forward_displacement_mm"],atol=1e-8,rtol=0.)
                require(all(stored[k] == v for k,v in val.items() if k != "forward_displacement_mm"),key)
            elif isinstance(val,float):
                np.testing.assert_allclose(val,stored,atol=1e-9,rtol=0.)
            else:
                require(val == stored,key)
        for k in totals:
            totals[k] += row["counts"][k]
    for unit, sha in report["failed_units"].items():
        require(digest(PUBLIC/"resources"/(unit+".json")) == sha,"missing failed guard")
    for u in [report["survey_resources"]]+[c["resources"] for c in cases if c["case_status"]=="COMPLETE"]:
        require(u["startup_effective_bytes"] >= 4<<30 and u["minimum_effective_bytes"] >= 2<<30
                and u["owned_cgroup_children_empty"] and len(u["cpu_affinity"]) == 1,"unexpected resource receipt")
    partial = sum(c["case_status"]=="PARTIAL" for c in cases)
    require(totals["physics"]+6000*partial<=18000 and totals["ray_queries"]+74880*partial<=224640
            and not any(report[k] for k in ("world_updates","policy_updates","blind_trials")),"budget exceeded")
    return dict(**report["verdict"],confirmed=totals)


def video(publish_video):
    if publish_video:
        from tools112.keyframe_video import INITIAL, completed
        out, guard, r = completed()
        f = r["film"]
        install(out/f["filename"],MOVIE,f["sha256"])
        admission_stops = {}
        for file in sorted(BASE.glob("guard_d1-surface112-keyframes-*/report.json")):
            if file.parent != guard:
                require(not read(file)["child_started"], "unaccounted started film")
                admission_stops[file.parent.name] = digest(file)
                install(file,PUBLIC/"resources"/(file.parent.name+".json"),digest(file))
        install_json(PUBLIC/"video.json",dict(film=f,resources=resources(guard),renderer_sha256=r["renderer_sha256"],
                     unit=r["unit"],first_render_guard_sha256=digest(INITIAL),
                     first_render_reservation_sha256=digest(BASE/"video_01/reservation.json"),admission_stops=admission_stops))
    receipt = read(PUBLIC/"video.json")
    f = receipt["film"]
    _, ref = reference(SEEDS[0])
    row = read(PUBLIC/"results.json")["cases"][0]["episode"]
    require(f["seed"] == SEEDS[0] and f["trace_sha256"] == [ref["trace_sha256"],row["trace_sha256"]]
            and f["physics"] == 0 and digest(MOVIE) == f["sha256"] and MOVIE.stat().st_size == f["bytes"]
            and 0 < f["bytes"] <= 5_000_000 and f["frames"] == int(f["stream"]["nb_read_frames"])
            and f["stream"]["codec_name"] == "h264" and "llvmpipe" in f["driver"]["renderer"].lower()
            and receipt["renderer_sha256"] == digest(ROOT/"tools112/surface_video.py"),"invalid paired video")
    original = PUBLIC/"resources/video_01.json"
    require(read(original)["status"] == "INCOMPLETE" and read(original)["child_started"]
            and digest(original) == receipt["first_render_guard_sha256"]
            and digest(PUBLIC/"reservations/video_01.json") == receipt["first_render_reservation_sha256"],
            "missing interrupted full-frame export")
    for unit, sha in receipt["admission_stops"].items():
        require(digest(PUBLIC/"resources"/(unit+".json")) == sha,"missing render admission receipt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish",action="store_true")
    parser.add_argument("--video",action="store_true")
    args = parser.parse_args()
    if args.publish:
        publish()
    print(json.dumps(verify()))
    if args.video:
        video(args.publish)
