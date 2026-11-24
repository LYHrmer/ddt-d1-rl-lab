"""Independently read back the public D1 contact experiment using NumPy and recorded measurements."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, digest, read
from tools108 import model108, preview108, run108
from training.d1_rl.interface import DEFAULT_Q


PUBLIC = ROOT / "reports/contact108"
STAGES = ("development", "fit", "blind", "score", "video")
CHANNELS = ("time_s", "axle_x_m", "base_x_m", "step_force_n", "knee_offset_rad", "fallen", "illegal_contact")


def _match(actual, expected, name):
    """Verify required report fields. Floating statistics use absolute or relative roundoff only."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() <= actual.keys(), f"missing fields at {name}")
        for field, v in expected.items():
            _match(actual[field], v, f"{name}.{field}")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, list) and len(expected) == len(actual), f"size differs at {name}")
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _match(a, e, f"{name}[{i}]")
    elif isinstance(expected, (float, np.floating)):
        require(type(actual) in (int, float) and np.isclose(actual, expected, atol=1e-9, rtol=1e-11), name)
    else:
        require(type(actual) is type(expected) and actual == expected, name)


def _episode(data, report, arm, spec):
    time, axle, base, force, knees, fallen, illegal = (np.asarray(data[k]) for k in CHANNELS)
    t, horizon = len(time)-1, spec["horizon"]
    require(0 < t <= spec["ticks"] and time.shape == axle.shape == base.shape == (t+1,)
            and force.shape == (t*10, 4) and knees.shape == (t, 2)
            and fallen.shape == illegal.shape == (t,) and fallen.dtype == illegal.dtype == bool,
            "unexpected measurement clocks or shape")
    require(all(np.isfinite(x).all() for x in (time, axle, base, force, knees)) and (force >= 0.).all(),
            "nonfinite measurement or negative normal load")
    np.testing.assert_allclose(time, np.arange(t+1)*.02, rtol=0., atol=1e-9)
    hits = np.flatnonzero(np.max(force[:, :2], axis=1) >= spec["trigger_n"])
    first = int(hits[0]) if len(hits) else None
    k = first//10 + 1 if first is not None else None
    ok = k is not None and k+horizon <= t
    pulse = np.zeros(t, dtype=bool)
    if arm == "flexion" and k is not None:
        pulse[k:k+spec["pulse_ticks"]] = True
    require(not knees[~pulse].any() and (knees[pulse] <= 1e-7).all()
            and (knees[pulse] >= spec["pulse_rad"]-1e-7).all(), "pulse outside registered window or limits")
    path = (axle[k+1:k+horizon+1] - axle[k]) if ok else None
    base_delta = float(1000*(base[k+horizon]-base[k])) if ok else None
    expected = dict(arm=arm, ticks=t, trigger_tick=k, first_contact_s=None if first is None else .002*(first+1),
                    max_front_contact_force_n=float(force[:, :2].max()), window_complete=ok,
                    fallen=bool(fallen.any()), illegal_contact=bool(illegal.any()),
                    changed_knee_ticks=int((np.abs(knees[:, 0])>1e-6).sum()), final_axle_x_m=float(axle[-1]),
                    final_base_x_m=float(base[-1]), axle_mm=float(1000*path[-1]) if ok else None, base_mm=base_delta)
    _match(report, expected, f"{report['seed']}.{arm}")
    require(report["status"] in ("COMPLETE", "PARTIAL") and (report["status"] != "COMPLETE" or t == spec["ticks"]),
            "incomplete episode marked complete")
    return dict(k=k, contact_s=expected["first_contact_s"], path=path, base_mm=base_delta, ticks=t,
                sha=report["sha256"], complete=report["status"] == "COMPLETE",
                safe=not expected["fallen"] and not expected["illegal_contact"] and report["status"] == "COMPLETE")


def _limits(values):
    return [float(values.min()), float(values.max())] if len(values) else None


def _horizontal_face(loaded, edge, height):
    return (loaded[:, 4] >= edge) & (loaded[:, 6] >= height-.002) & (loaded[:, 9] >= .9)


def _wheel_peak(forces, start):
    i, wheel = (int(v) for v in np.unravel_index(np.argmax(forces), forces.shape))
    return dict(force_n=float(forces[i, wheel]), wheel=("FL", "FR")[wheel],
                native_substep=start+i, substep_end_s=(start+i+1)*.002)


def _contact_surface(events, force, spec, k=None):
    """Replay the archived horizontal-face test, and expose the measured geometry of loaded front contacts."""
    require(events.ndim == 2 and events.shape[1] == 10 and np.isfinite(events).all(),
            "malformed sparse step contacts")
    # Columns are 500 Hz index, wheel slot (-1 is a nonwheel), robot geom, normal force,
    # three world position coordinates, and the terrain-to-robot world normal.
    require(np.equal(events[:, :3], np.floor(events[:, :3])).all()
            and (np.diff(events[:, 0]) >= 0).all() and (events[:, 0] >= 0).all()
            and (events[:, 0] < len(force)).all() and np.isin(events[:, 1], [-1, 0, 1, 2, 3]).all()
            and (events[:, 2] >= 0).all() and (events[:, 3] > 0).all(), "invalid sparse contact indices or load")
    np.testing.assert_allclose(np.linalg.norm(events[:, 7:10], axis=1), 1., atol=1e-6, rtol=0.)
    wheels = events[events[:, 1] >= 0]
    total = np.zeros_like(force)
    np.add.at(total, (wheels[:, 0].astype(np.intp), wheels[:, 1].astype(np.intp)), wheels[:, 3])
    np.testing.assert_allclose(total, force, rtol=1e-12, atol=1e-10)
    threshold = spec["trigger_n"]
    edge, height = spec["geometry"]["front_x_m"], spec["geometry"]["height_m"]
    loaded = wheels[wheels[:, 3] >= threshold]
    top = loaded[_horizontal_face(loaded, edge, height)]
    front = loaded[loaded[:, 1] < 2]
    result = dict(top_by_wheel=[bool((top[:, 1] == i).any()) for i in range(4)],
                  max_nonwheel_step_force_n=float(np.max(events[events[:, 1] < 0, 3], initial=0.)),
                  loaded_front_events=len(front), front_x_minus_edge_mm=_limits(1000*(front[:, 4]-edge)),
                  front_z_minus_top_mm=_limits(1000*(front[:, 6]-height)), front_normal_x=_limits(front[:, 7]),
                  front_normal_z=_limits(front[:, 9]))
    if k is not None:
        start, end = 10*k, 10*(k+spec["horizon"])
        observed = front[(front[:, 0] >= start) & (front[:, 0] < end)]
        on_top = _horizontal_face(observed, edge, height)
        other = observed[~on_top]
        result["window"] = dict(whole_episode_peak_front_wheel_step_normal=_wheel_peak(force[:, :2], 0),
                    window_peak_front_wheel_step_normal=_wheel_peak(force[start:end, :2], start),
                    window_front_wheel_loaded_contact_records=len(observed),
                    window_top_loaded_records=int(on_top.sum()),
                    window_top_loaded_native_substeps=len(np.unique(observed[on_top, 0])),
                    window_other_loaded_records=len(other), window_other_contact_x_minus_edge_mm=_limits(1000*(other[:, 4]-edge)),
                    window_other_contact_z_minus_top_mm=_limits(1000*(other[:, 6]-height)),
                    window_other_contact_unit_normal_x=_limits(other[:, 7]), window_other_contact_unit_normal_z=_limits(other[:, 9]))
    return result


def _motion_series(data, axle_x):
    calves, wheel_z, base_z, quat = (data[key] for key in ("calves_rad", "wheel_z_m", "base_z_m", "base_quat"))
    n = len(axle_x)
    require(calves.shape == wheel_z.shape == (n, 2) and base_z.shape == (n,) and quat.shape == (n, 4)
            and all(np.isfinite(a).all() for a in (calves, wheel_z, base_z, quat)), "invalid measured motion")
    np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1., rtol=0., atol=1e-12)
    w, x, y, z = quat.T
    pitch = np.rad2deg(np.arcsin(np.clip(2*(w*y-x*z), -1., 1.)))
    return dict(mean_front_calf_angle_rad=calves.mean(axis=1), mean_front_wheel_center_height_mm=1000*wheel_z.mean(axis=1),
                front_axle_forward_x_mm=1000*axle_x, base_height_mm=1000*base_z, base_pitch_deg=pitch)


def _check_packets(row, sensor_plan, spec, pair, name, motion, measured, obs):
    hz, rate, delay = (sensor_plan[k] for k in ("policy_hz", "sensor_hz", "delay_ticks"))
    require((hz, rate, delay) == (50, 6, 2), "sparse ranging clock")
    k = pair["k"]
    pulses = np.arange((k*rate)//hz+1)
    capture = (pulses*hz + rate-1)//rate
    capture = capture[capture+delay < k]
    delivery = capture+delay
    np.testing.assert_array_equal(motion[name+"_packet_ticks"], np.column_stack((capture, delivery)))
    features = motion[name+"_range_features"]
    require(features.dtype == np.float32 and features.shape == (len(capture), 16)
            and np.isfinite(features).all(), "actor packet must have eight ranges and eight flags")
    np.testing.assert_array_equal(features[-1], obs[627:])
    captures, deliveries = motion[name+"_capture_pose"], motion[name+"_delivery_pose"]
    for ticks, poses in ((capture, captures), (delivery, deliveries)):
        require(poses.shape == (len(capture), 7) and np.isfinite(poses).all(), "invalid packet pose")
        np.testing.assert_array_equal(poses[:, 0], measured[name+"_base_x_m"][ticks])
        np.testing.assert_array_equal(poses[:, 2], motion[name+"_base_z_m"][ticks])
        np.testing.assert_array_equal(poses[:, 3:7], motion[name+"_base_quat"][ticks])
    sensor = sensor_plan["geometry"]
    raw_dirs, sparse, _ = preview108._grid(sensor)
    directions = np.tile(raw_dirs, (2, 1))[sparse]
    mounts = np.repeat(sensor["mounts_m"], 4, axis=0)
    angles = np.asarray(sensor["sparse_depression_deg"])
    stop_ms = round(1000*pair["contact_s"])
    packets, tops, residual_c, residual_d = [], [], [], []
    for i, delivered in enumerate(delivery):
        top, floor, riser, other, invalid, depths, errors_c, errors_d = preview108._packet(
            features[i], captures[i], deliveries[i], mounts, directions, sensor, spec["geometry"])
        c, d = int(capture[i]), int(delivered)
        lead = stop_ms - 20*d
        left, right = top[:4], top[4:]
        packets.append([c, d, lead, preview108._bits(left), preview108._bits(right),
                        int(floor.sum()), int(riser.sum()), int(other.sum()), int(invalid.sum())])
        if top.any():
            tops.append(dict(capture_tick=c, delivery_tick=d, delivery_s=d/hz, lead_ms=lead,
                             top_depressions_deg_left=angles[left].tolist(), top_depressions_deg_right=angles[right].tolist(),
                             top_x_beyond_front_edge_mm=_limits(depths)))
        residual_c.extend(errors_c)
        residual_d.extend(errors_d)
    counts = dict(top=sum(p[3].bit_count()+p[4].bit_count() for p in packets),
                  floor=sum(p[5] for p in packets), front_riser=sum(p[6] for p in packets),
                  unclassified_valid=sum(p[7] for p in packets), invalid=sum(p[8] for p in packets))
    require(sum(counts.values()) == len(capture)*8, "surface labels do not partition the recorded beams")
    _match(row, dict(paired_precontact=True, first_contact_ms=stop_ms, original_trace_sha256=pair["hashes"][0],
           precontact_packets=packets, counts=counts, packets_with_top_both_sides=sum(bool(p[3] and p[4]) for p in packets),
           first_top_packet=tops[0] if tops else None, last_top_packet=tops[-1] if tops else None,
           max_capture_pose_labeled_surface_residual_um=1e6*max(residual_c) if residual_c else None,
           max_delivery_pose_same_surface_residual_mm=1000*max(residual_d) if residual_d else None), name+" preview")


def _motion_and_preview(folder, spec, pairs, mechanics, observations):
    preview = read(folder / "preview_readback.json")
    sensor_plan = read(preview108.BEAMS)
    require(digest(ROOT/"tools87/terrain87.py") == sensor_plan["parents_sha256"]["tools87/terrain87.py"],
            "native ranging implementation changed")
    _match(preview, dict(split=mechanics["split"], audit="post_score_physical_surface_preview",
                         original_packet_counted_once_per_pair=True, packet_fields=preview108.PACKET_FIELDS,
                         top_mask_bits_in_depression_deg=sensor_plan["geometry"]["sparse_depression_deg"]), "preview contract")
    require([r["seed"] for r in preview["rows"]] == list(pairs), "wrong packet roster")
    motion_fields = ("calves_rad", "wheel_z_m", "base_z_m", "base_quat")
    packet_fields = ("packet_ticks", "range_features", "capture_pose", "delivery_pose")
    with (np.load(folder/"motion_observations.npz", allow_pickle=False) as motion,
          np.load(folder/"measurements.npz", allow_pickle=False) as measured):
        keys = {f"seed{s}_{a}_{f}" for s in pairs for a in spec["arms"] for f in motion_fields}
        keys.update(f"seed{s}_original_{f}" for s in pairs for f in packet_fields)
        require(set(motion.files) == keys and len(motion.files) == len(keys), "wrong public motion/packet fields")
        for j, (seed, pair) in enumerate(pairs.items()):
            k, window = pair["k"], slice(pair["k"], pair["k"]+spec["horizon"]+1)
            signals = []
            orig_name = f"seed{seed}_original"
            for arm in spec["arms"]:
                name = f"seed{seed}_{arm}"
                data = {f: motion[name+"_"+f] for f in motion_fields}
                for f in motion_fields:
                    np.testing.assert_array_equal(data[f][:k+1], motion[orig_name+"_"+f][:k+1])
                signals.append(_motion_series(data, measured[name+"_axle_x_m"]))
            times = measured[orig_name+"_time_s"]
            for key, original in signals[0].items():
                change = signals[1][key][window] - original[window]
                p = int(abs(change).argmax())
                peak = float(change[p])
                _match(mechanics["pairs"][j]["flexion_minus_original"][key],
                       dict(peak_signed=peak, peak_tick=k+p, peak_ticks_after_trigger=p, peak_time_s=float(times[k+p]),
                            at_horizon=float(change[-1]), horizon_abs_fraction_of_peak=abs(float(change[-1])/peak) if peak else None),
                       f"{seed} {key}")
            obs = observations[j]
            np.testing.assert_allclose(motion[orig_name+"_calves_rad"][k-1]-DEFAULT_Q[[2,6]], obs[[11,15]], atol=2e-7, rtol=0.)
            _check_packets(preview["rows"][j], sensor_plan, spec, pair, orig_name, motion, measured, obs)


def split_readback(folder, split, spec, seal):
    summary = read(folder / "contacts.json")
    mechanics = read(folder / "physical_readback.json")
    roster = [(s, a) for s in spec[split] for a in spec["arms"]]
    require(summary["split"] == split and [(r["seed"], r["arm"]) for r in summary["episodes"]] == roster
            and [p["seed"] for p in summary["pairs"]] == spec[split], "wrong or duplicate episode roster")
    require(summary["archival_index_sha256"] == seal["files"]["episodes/index.json"], "source episode index")
    _match(mechanics, dict(split=split, analysis="post_hoc_mechanics", window_ticks=spec["horizon"],
                           episode_index_sha256=summary["archival_index_sha256"]), "physical readback")
    require([r["seed"] for r in mechanics["pairs"]] == spec[split], "wrong mechanics roster")
    paired, total = {}, 0
    with (np.load(folder / "measurements.npz", allow_pickle=False) as tape,
          np.load(folder / "step_contacts.npz", allow_pickle=False) as sparse):
        expected_keys = {f"seed{s}_{a}_{c}" for s, a in roster for c in CHANNELS}
        require(len(tape.files) == len(expected_keys) and set(tape.files) == expected_keys, "wrong measured fields")
        require(set(sparse.files) == {f"seed{s}_{a}" for s, a in roster}
                and len(sparse.files) == len(roster), "wrong step event roster")
        for j, seed in enumerate(spec[split]):
            records, signals = [], []
            for a, arm in enumerate(spec["arms"]):
                name = f"seed{seed}_{arm}"
                data = {c: tape[f"{name}_{c}"] for c in CHANNELS}
                row = summary["episodes"][2*j+a]
                trace = "trace.npz" if row["status"] == "COMPLETE" else "partial_trace.npz"
                require(row["sha256"] == seal["files"][f"episodes/{name}/{trace}"], "native trajectory identity")
                record = _episode(data, row, arm, spec)
                require(record["path"] is not None, f"incomplete measured contact window for {name}")
                surface = _contact_surface(sparse[name], data["step_force_n"], spec, record["k"])
                _match(row, {key: surface[key] for key in ("top_by_wheel", "max_nonwheel_step_force_n")}, name)
                _match(mechanics["pairs"][j]["arms"][arm], surface.pop("window"), f"{name} contact window")
                record["surface"] = surface
                total += record["ticks"]
                records.append(record)
                signals.append(data)
            lo, hi = records
            k = lo["k"]
            paired_record = summary["pairs"][j]
            require(k == hi["k"] and lo["path"] is not None and hi["path"] is not None, "incomplete paired model window")
            _match(mechanics["pairs"][j], dict(paired_window_available=True, trigger_tick=k,
                   first_native_contact_end_s=lo["contact_s"],
                   window_s=signals[0]["time_s"][[k, k+spec["horizon"]]].tolist()), f"{split}.{seed} mechanics")
            # These shared coordinates and force samples are public. Full state/actor prefix identity
            # is retained in the separately sealed native acquisition receipt.
            for c in ("axle_x_m", "base_x_m", "step_force_n"):
                stop = k*10 if c == "step_force_n" else k+1
                np.testing.assert_array_equal(signals[0][c][:stop], signals[1][c][:stop])
            np.testing.assert_array_equal(lo["path"][0], hi["path"][0])
            qualified = (all(r["complete"] for r in records) and
                         np.all(np.sum(np.abs(signals[1]["knee_offset_rad"])>1e-6, axis=0) == spec["pulse_ticks"]))
            _match(paired_record, dict(matched_prefix=True, qualified=bool(qualified),
                                       axle_effect_mm=float(1000*(hi["path"][-1]-lo["path"][-1])),
                                       base_effect_mm=hi["base_mm"]-lo["base_mm"]), f"{split}.{seed}")
            paired[seed] = dict(k=k, contact_s=lo["contact_s"], paths=np.stack([lo["path"], hi["path"]]),
                                base_mm=[r["base_mm"] for r in records], safe=[r["safe"] for r in records],
                                hashes=[r["sha"] for r in records], surfaces=[r["surface"] for r in records],
                                qualified=bool(qualified))
    _match(summary, dict(data_eligible=all(p["qualified"] for p in paired.values()), native_decisions=total,
                         physics_substeps=total*10), split)
    with np.load(folder / "onset.npz", allow_pickle=False) as onset:
        require(set(onset.files) == {"resets", "trigger_ticks", "observations"}
                and np.array_equal(onset["resets"], spec[split])
                and np.array_equal(onset["trigger_ticks"], [p["k"] for p in paired.values()]), "onset clock")
        observations = onset["observations"]
        require(observations.shape == (len(paired), 643) and observations.dtype == np.float32
                and np.isfinite(observations).all(), "onset observation dimensions")
        for pair, obs in zip(paired.values(), observations, strict=True):
            x = obs.astype(np.float64)
            pair["features"] = np.array([.087*x[[28,32,36,40]].mean()/.05, x[[11,15]].mean()])
            pair["observation_sha256"] = sha256(np.ascontiguousarray(obs).tobytes()).hexdigest()
        _motion_and_preview(folder, spec, paired, mechanics, observations)
    return paired, total


def _validated_seals(folder, spec):
    prov = read(folder / "provenance.json")
    require(prov["plan_sha256"] == digest(model108.PLAN) == run108.PLAN_SHA, "published plan is different")
    sources = prov["sources_sha256"]
    require({f"tools108/{x}108.py" for x in ("contact","model","score","video","run")} <= sources.keys(),
            "production sources are missing")
    for name, sha in sources.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and digest(ROOT / path) == sha, f"source {name}")
    seals, hashes = prov["stage_seals"], prov["stage_seal_sha256"]
    require(set(seals) == set(hashes) == set(STAGES), "missing acquisition stage")
    def json_digest(value):
        return sha256((json.dumps(value, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    for stage in STAGES:
        # Reconstruct the exact producer serialization even if a JSON distributor reorders object fields.
        canonical_seal = dict(status="COMPLETE", stage=stage, plan_sha256=prov["plan_sha256"],
                              files=dict(sorted(seals[stage]["files"].items())))
        require(canonical_seal == seals[stage] and json_digest(canonical_seal) == hashes[stage],
                f"invalid {stage} seal")
        inputs = dict(plan_sha256=prov["plan_sha256"], launcher_sha256=sources["tools108/run108.py"], stage=stage)
        if stage in ("development", "blind"):
            inputs["native_sources"] = {name: digest(ROOT/name) for name in run108.NATIVE_FILES}
        if stage in ("fit", "score"):
            inputs["model_sha256"] = sources["tools108/model108.py"]
        require(json_digest(inputs) == canonical_seal["files"]["inputs.json"], f"source binding for {stage}")
    require(hashes["fit"] == prov["blind_fit_seal_sha256"] and
            seals["blind"]["files"]["fit_binding.json"] == json_digest({"fit_seal_sha256": hashes["fit"]}),
            "blind collection is not bound to these fitted models")
    public = prov["public_files_sha256"]
    bindings = {
        "development/contacts.json": ("development", "contacts.json"),
        "development/measurements.npz": ("development", "measurements.npz"),
        "blind/contacts.json": ("score", "contacts.json"), "blind/measurements.npz": ("score", "measurements.npz"),
        "fit/fit.json": ("fit", "fit/fit.json"), "fit/models.npz": ("fit", "fit/models.npz"),
        "score/score.json": ("score", "score.json"), "score/predictions.npz": ("score", "forecast/predictions.npz"),
        "score/forecast_seal.json": ("score", "forecast/forecast_seal.json")}
    split_extractions = {f"{split}/{file}" for split in ("development", "blind")
                         for file in ("onset.npz", "step_contacts.npz", "physical_readback.json",
                                      "motion_observations.npz", "preview_readback.json")}
    require({*bindings, "resources.json", *split_extractions} <= public.keys(),
            "incomplete public artifact roster")
    for name, sha in public.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and digest(folder/path) == sha, f"artifact {name}")
    for name,(stage, private) in bindings.items():
        require(public[name] == seals[stage]["files"][private], f"{name} differs from its stage")
    usage, limits = read(folder / "resources.json"), spec["resources"]
    require(set(usage) == set(STAGES), "missing resource receipt")
    for stage, r in usage.items():
        _match(r, limits, f"resource configuration for {stage}")
        require(r["startup_effective_bytes"] >= limits["admission_bytes"] and
                r["minimum_effective_bytes"] >= limits["reserve_bytes"] and
                0 < r["peak_group_bytes"] <= limits["memory_max_bytes"] and r["elapsed_s"] > 0
                and len(r["cpu_affinity"]) == 1 and r["owned_cgroup_children_empty"] is True
                and re.fullmatch(rf"d1-contact108-{stage}-\d{{2}}", r["unit"])
                and re.fullmatch(r"[0-9a-f]{64}", r["guard_sha256"]), f"bad resource values for {stage}")
    return seals


def _cell(predicted, reference, mean, gradient, resolution, limit):
    value, target = float(predicted[-1]), float(reference[-1])
    error = abs(value-target)
    reasons = []
    if abs(target) < resolution:
        reasons.append("below_engineering_resolution")
    if value*target <= 0:
        reasons.append("direction_mismatch")
    if error > limit*abs(target):
        reasons.append("outside_native_25pct")
    return dict(passed=not reasons, reasons=reasons, predicted_mm=value, native_mm=target, error_mm=error,
                relative_error=None if target == 0 else error/abs(target), train_mean_mm=mean,
                train_mean_error_mm=abs(mean-target), zero_error_mm=abs(target),
                signed_action_derivative_mm_per_rad=float(1000*gradient[-1]/model108.PULSE_RAD),
                trajectory_mae_mm=float(np.mean(np.abs(predicted-reference))))


def check(folder=PUBLIC):
    folder = Path(folder)
    spec = model108._protocol()
    seals = _validated_seals(folder, spec)
    train, steps1 = split_readback(folder/"development", "development", spec, seals["development"])
    blind, steps2 = split_readback(folder/"blind", "blind", spec, seals["blind"])
    fit_dir, score_dir = folder/"fit", folder/"score"
    fit = model108.check_fit(fit_dir)
    fit_seeds = list(model108.MODEL_SEEDS)
    horizon, budget = spec["horizon"], spec["budget"]
    paths = np.stack([p["paths"] for p in train.values()])
    mean = (paths[:, 1] - paths[:, 0]).mean(axis=0)
    with np.load(fit_dir/"models.npz", allow_pickle=False) as z:
        require(np.array_equal(z["development"], list(train)), "fit source identities")
        np.testing.assert_allclose(z["training_paths_m"], paths, rtol=0., atol=1e-12)
        np.testing.assert_array_equal(z["training_features"], np.stack([p["features"] for p in train.values()]))
        np.testing.assert_allclose(z["mean_response_m"], mean, rtol=0, atol=1e-12)
        coeff, center, scale = z["coefficients"], z["knee_center_rad"].item(), z["knee_scale_rad"].item()
        updates = len(fit_seeds) * (z["normalized_parameter_path"].shape[1]-1)
    _match(fit, dict(source_sha256={str(s): p["hashes"] for s,p in train.items()},
                     world_updates=updates, policy_updates=0, initializations=fit_seeds,
                     source_safety={str(s): p["safe"] for s,p in train.items()},
                     development_response_mm={str(s): float(1000*(p["paths"][1,-1]-p["paths"][0,-1]))
                                              for s,p in train.items()}, baseline_endpoint_mm=float(1000*mean[-1])), "fit")
    forecast = read(score_dir/"forecast_seal.json")
    _match(forecast, dict(status="SEALED", spec_sha256=digest(model108.PLAN), models_sha256=digest(fit_dir/"models.npz"),
                          training_fit_sha256=digest(fit_dir/"fit.json"), predictions_sha256=digest(score_dir/"predictions.npz"),
                          fitted_updates=0, policy_updates=0), "forecast seal")
    require(set(forecast["onset"]) == {str(s) for s in blind}, "blind onset roster")
    for seed,p in blind.items():
        _match(forecast["onset"][str(seed)], dict(seed=seed, trigger_tick=p["k"], observation_tick=p["k"]-1,
                                                  observation_sha256=p["observation_sha256"]), "sealed onset")
    with np.load(score_dir/"predictions.npz", allow_pickle=False) as z:
        pred, gradients = z["paths_m"], z["gradient_m_per_unit_pulse"]
        require(np.array_equal(z["fit_seeds"], fit_seeds) and np.array_equal(z["resets"], list(blind))
                and pred.shape == (len(fit_seeds), len(blind), 2, horizon)
                and gradients.shape == (len(fit_seeds), len(blind), horizon), "prediction roster")
        np.testing.assert_array_equal(z["start_features"], np.stack([p["features"] for p in blind.values()]))
        np.testing.assert_allclose(z["mean_response_m"], mean, rtol=0., atol=1e-12)
    np.testing.assert_allclose(gradients, pred[:,:,1]-pred[:,:,0], rtol=0., atol=1e-12)
    cfg = spec["model"]
    outcomes, safety = {}, {}
    for seed, p in blind.items():
        effect = 1000*(p["paths"][1] - p["paths"][0])
        outcomes[str(seed)] = dict(native_paths_mm=(1000*p["paths"]).tolist(), native_response_mm=effect.tolist(),
                                   endpoint_mm=float(effect[-1]), source_sha256=p["hashes"])
        safety[str(seed)] = dict(zip(spec["arms"], p["safe"], strict=True))
    cells, errors, beats = [], {}, {}
    for i, model in enumerate(fit_seeds):
        model_cells = []
        for j,(seed, p) in enumerate(blind.items()):
            for arm, strength in enumerate((0., 1.)):
                expected = model108.trajectory_m(p["features"], strength, coeff[i], center, scale)
                np.testing.assert_allclose(pred[i,j,arm], expected, atol=1e-12, rtol=1e-10)
            effect = np.asarray(outcomes[str(seed)]["native_response_mm"])
            stats = _cell(1000*(pred[i,j,1]-pred[i,j,0]), effect, float(1000*mean[-1]), gradients[i,j],
                          cfg["resolution_mm"], cfg["relative_error_max"])
            model_cells.append(dict(model_seed=model, reset=seed, **stats))
        cells.extend(model_cells)
        mae = {name: float(np.mean([c[field] for c in model_cells])) for name,field in
               (("candidate_mm","error_mm"), ("training_mean_mm","train_mean_error_mm"), ("zero_mm","zero_error_mm"))}
        errors[str(model)] = mae
        beats[str(model)] = bool(mae["candidate_mm"] < mae["training_mean_mm"])
    passed = sum(c["passed"] for c in cells)
    qualifies = passed == len(cells) and all(beats.values())
    safe = all(all(v.values()) for v in safety.values())
    positive = all(o["endpoint_mm"] >= cfg["resolution_mm"] for o in outcomes.values())
    supported = bool(qualifies and safe and positive)
    _match(read(score_dir/"score.json"), dict(status="SCORED", forecast_sha256=forecast["predictions_sha256"], cells=cells,
           cells_passed=passed, cells_planned=len(cells), blind_mae=errors, strictly_beats_training_mean=beats,
           outcomes=outcomes, arm_safety=safety, all_native_pairs_safe=safe, all_native_positive_resolved=positive,
           model_qualified=qualifies, supports_next_local_policy_experiment=supported, policy_updates=0), "blind score")
    require(steps1+steps2 == budget["total_native_decisions_max"] and 10*(steps1+steps2) == budget["total_physics_substeps_max"]
            and updates == budget["world_updates_max"] and budget["policy_updates"] == 0, "registered budget")
    contact_results = {}
    for split, pairs in (("development", train), ("blind", blind)):
        rows = []
        for seed, p in pairs.items():
            axle = 1000*p["paths"][:, -1]
            rows.append(dict(seed=seed, first_contact_s=p["contact_s"], trigger_tick=p["k"],
                             axle_mm=dict(zip(spec["arms"], axle.tolist(), strict=True)),
                             base_mm=dict(zip(spec["arms"], p["base_mm"], strict=True)),
                             axle_effect_mm=float(axle[1]-axle[0]), base_effect_mm=p["base_mm"][1]-p["base_mm"][0],
                             surfaces=dict(zip(spec["arms"], p["surfaces"], strict=True))))
        contact_results[split] = rows
    return dict(native_decisions=steps1+steps2, physics_substeps=10*(steps1+steps2), world_updates=updates,
                policy_updates=0, contacts=contact_results, blind_cells_passed=passed, blind_mae=errors,
                model_qualified=qualifies, supports_next_local_policy_experiment=supported)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PUBLIC)
    print(json.dumps(check(parser.parse_args().root), indent=2, allow_nan=False))
