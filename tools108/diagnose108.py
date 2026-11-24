"""Post-hoc mechanics of the paired #108 contact, using saved measurements only."""
from pathlib import Path

import numpy as np

from tools91.launch91 import require
from tools94.common94 import digest, read
from training.d1_rl.interface import DEFAULT_Q, JOINT_NAMES


CALVES = ("FL_calf_joint", "FR_calf_joint")
CALF_IDS = np.array([JOINT_NAMES.index(name) for name in CALVES])
# The archived D1 model has a seven-coordinate free base followed by JOINT_NAMES.
# Verify these addresses against the independent named encoder channels in every trace.
CALF_QPOS = 7 + CALF_IDS


def _range(values):
    return [float(values.min()), float(values.max())] if values.size else None


def _peak_difference(original, flexion, times, k):
    delta = flexion - original
    i = int(np.argmax(np.abs(delta)))
    return dict(peak_signed=float(delta[i]), peak_tick=k+i, peak_ticks_after_trigger=i,
                peak_time_s=float(times[i]), at_horizon=float(delta[-1]),
                horizon_abs_fraction_of_peak=float(abs(delta[-1]/delta[i])) if delta[i] else None)


def _force_peak(forces, native_start, dt):
    i, wheel = (int(v) for v in np.unravel_index(np.argmax(forces), forces.shape))
    return dict(force_n=float(forces[i, wheel]), wheel=("FL", "FR")[wheel],
                native_substep=native_start+i, substep_end_s=(native_start+i+1)*dt)


def _contact_details(trace, plan, k, end, wheels):
    substeps, dt = 10, .002
    start_sub, end_sub = substeps*k, substeps*end
    step = trace["step_force_n"][:, :2]
    floor = trace["ground_force_n"][start_sub:end_sub, :2]
    raw = trace["step_contacts"]
    edge, height = plan["geometry"]["front_x_m"], plan["geometry"]["height_m"]
    threshold = plan["trigger_n"]
    loaded = raw[(raw[:, 0] >= start_sub) & (raw[:, 0] < end_sub)
                 & np.isin(raw[:, 1], [0, 1]) & (raw[:, 3] >= threshold)]
    # Same top-support geometry used by score108.measurements, now windowed and
    # restricted to the front wheels. MuJoCo's normal here points from box to wheel.
    top = ((loaded[:, 4] >= edge) & (loaded[:, 6] >= height-.002) & (loaded[:, 9] >= .9))
    other = loaded[~top]
    both = (step[start_sub:end_sub] >= threshold) & (floor >= threshold)
    return dict(
        whole_episode_peak_front_wheel_step_normal=_force_peak(step, 0, dt),
        window_peak_front_wheel_step_normal=_force_peak(step[start_sub:end_sub], start_sub, dt),
        window_front_wheel_loaded_contact_records=len(loaded),
        window_top_loaded_records=int(top.sum()),
        window_top_loaded_native_substeps=int(np.unique(loaded[top, 0]).size),
        window_other_loaded_records=len(other),
        window_other_contact_x_minus_edge_mm=_range(1000*(other[:, 4]-edge)),
        window_other_contact_z_minus_top_mm=_range(1000*(other[:, 6]-height)),
        window_other_contact_unit_normal_x=_range(other[:, 7]),
        window_other_contact_unit_normal_z=_range(other[:, 9]),
        window_simultaneous_floor_and_step_loaded_substeps_FL_FR=both.sum(axis=0).tolist(),
        window_front_wheel_x_minus_edge_mm=_range(1000*(wheels[k:end+1, :, 0]-edge)),
        window_end_wheel_x_minus_edge_mm_FL_FR=(1000*(wheels[end, :, 0]-edge)).tolist(),
        episode_end_wheel_x_minus_edge_mm_FL_FR=(1000*(wheels[-1, :, 0]-edge)).tolist(),
    )


def _arm_data(folder, meta, plan):
    path = folder / "trace.npz"
    require(meta["trace_file"] == path.name and digest(path) == meta["trace_sha256"], "wrong contact trace")
    k = meta["trigger_tick"]
    end = k + plan["horizon"]
    ticks = meta["ticks_completed"]
    with np.load(path, allow_pickle=False) as trace:
        time, qpos = trace["time_s"], trace["qpos"]
        wheels = trace["wheel_xyz_m"][:, :2, :]
        encoders = trace["actor_observation"][:, 9+CALF_IDS]
        calves = qpos[:, CALF_QPOS]
        require(qpos.shape == (ticks+1, 23) and wheels.shape == (ticks+1, 2, 3)
                and np.allclose(time, .02*np.arange(ticks+1), rtol=0., atol=1e-9)
                and np.allclose(calves-DEFAULT_Q[CALF_IDS], encoders, rtol=0., atol=2e-7),
                "saved named encoders or decision clock do not match qpos")
        require(trace["step_force_n"].shape == (10*ticks, 4) and trace["ground_force_n"].shape == (10*ticks, 4)
                and trace["step_contacts"].shape[1] == 10, "unexpected native contact channels")
        axle = wheels[:, :, 0].mean(axis=1)
        require(np.array_equal(axle, trace["front_axle_x_m"]), "stored front axle disagrees with wheel centers")
        loads = trace["step_force_n"][:, :2]
        first = np.flatnonzero((loads >= plan["trigger_n"]).any(axis=1))
        require(first.size and first[0] == meta["first_contact_substep"] and int(first[0])//10+1 == k,
                "paired diagnostic contact clock disagrees with native load")
        window = slice(k, end+1)
        qw, qx, qy, qz = qpos[window, 3:7].T
        base_pitch_deg = np.degrees(np.arcsin(np.clip(2*(qw*qy-qz*qx), -1., 1.)))
        return (qpos[:k+1], time[window],
                dict(mean_front_calf_angle_rad=calves[window].mean(axis=1),
                     mean_front_wheel_center_height_mm=1000*wheels[window, :, 2].mean(axis=1),
                     front_axle_forward_x_mm=1000*axle[window],
                     base_height_mm=1000*qpos[window, 2], base_pitch_deg=base_pitch_deg),
                _contact_details(trace, plan, k, end, wheels))


def diagnose(episodes_dir, plan, split):
    """Compare registered complete pairs in one explicitly supplied split; return JSON-safe values."""
    directory = Path(episodes_dir)
    index = read(directory / "index.json")
    require(split in ("development", "blind") and index["reset_seeds"] == plan[split]
            and index["arms"] == plan["arms"] == ["original", "flexion"], "unexpected diagnostic roster")
    pairs = []
    for seed in plan[split]:
        folders = [directory / f"seed{seed}_{arm}" for arm in plan["arms"]]
        meta = [read(folder / "report.json") for folder in folders]
        require(all(m["seed"] == seed and m["arm"] == arm and m["geometry"] == plan["geometry"]
                    and m["checkpoint_sha256"] == plan["checkpoint_sha256"] for m, arm in zip(meta, plan["arms"])),
                "diagnostic rollout identity mismatch")
        row = dict(seed=seed, paired_window_available=False,
                   arm_status={m["arm"]: m["status"] for m in meta})
        pairs.append(row)
        if any(m["status"] != "COMPLETE" or not m["full_horizon"] or m["trigger_tick"] is None
               or m["trigger_tick"]+plan["horizon"] > m["ticks_completed"] for m in meta):
            row["reason"] = "A complete triggered pair and the full measurement window are required."
            continue
        if meta[0]["first_contact_substep"] != meta[1]["first_contact_substep"]:
            row["reason"] = "Native trigger times differ."
            continue
        lo, hi = [_arm_data(folder, m, plan) for folder, m in zip(folders, meta)]
        require(np.array_equal(lo[0], hi[0]) and np.array_equal(lo[1], hi[1]), "paired pre-issue state mismatch")
        k = meta[0]["trigger_tick"]
        row.update(paired_window_available=True, trigger_tick=k,
                   first_native_contact_end_s=(meta[0]["first_contact_substep"]+1)*.002,
                   window_s=[float(lo[1][0]), float(lo[1][-1])],
                   flexion_minus_original={key: _peak_difference(val, hi[2][key], lo[1], k)
                                           for key, val in lo[2].items()},
                   arms={"original": lo[3], "flexion": hi[3]})
    return dict(split=split, analysis="post_hoc_mechanics", window_ticks=plan["horizon"],
                episode_index_sha256=digest(directory/"index.json"),
                named_qpos_addresses=dict(zip(CALVES, CALF_QPOS.tolist())),
                definitions=dict(
                    peak_signed="Signed flexion minus original at the largest absolute difference from k through k+horizon, sampled at 50 Hz inclusive. Ties use the earliest boundary.",
                    front_axle="Mean world x of the FL and FR wheel centers. Positive points toward the step.",
                    step_normal_force="Sum of positive normal forces per wheel against step108 at each 500 Hz substep. Face and rim summaries use individual loaded contacts.",
                    top_loaded="Front-wheel contact with at least the trigger load, x at or past the front edge, z within 2 mm below the top or higher, and upward unit normal z at least 0.9 (score108).",
                    other_loaded="All other loaded front-wheel step contacts. Signed offsets and terrain-to-wheel normals distinguish an oblique rim reaction from horizontal top support."),
                pairs=pairs)
