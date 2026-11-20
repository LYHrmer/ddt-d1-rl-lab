"""Localize illegal contacts in the saved native endpoints of the six training rollouts."""
import hashlib
import json
import math

import mujoco
import numpy as np

from tools87.readback87 import check
from tools87.terrain87 import TERRAINS, ground_height, make_env
from tools89.common89 import BASE, PLAN, check_worker, complete_stage, read, registered_plan, sha
from tools89.course89 import loaded_wheels_past_toe, wheel_geom_slots
from tools89.eval89 import checked_models
from training.d1_rl.interface import inverse_rotate


def source_events(folder, model_sha, hz):
    report = read(folder / "report.json")
    path = folder / "train_trace.npz"
    keep = {"qpos", "qvel", "last_torque_nm", "illegal_contact", "terminated", "fallen", "episode_tick",
            "terrain", "slope_start", "start_x_m", "command_vx_m_s", "body_vx_m_s", "loaded_wheels_past_toe"}
    declared = report["trace"]["array_sha256"]
    n = report["trace"]["completed_recorded_ticks"]
    check(n == report["budget"] and report["policy_sha256"] == model_sha, folder.name + " identity/budget")
    arrays = {}
    with np.load(path, allow_pickle=False) as z:
        check(set(z.files) == set(declared) and keep <= set(z.files), folder.name + " training arrays")
        for key in z.files:
            array = z[key]
            check(len(array) == n and hashlib.sha256(array.tobytes()).hexdigest() == declared[key],
                  folder.name + " " + key)
            if key in keep:
                arrays[key] = array
    flagged, fallen = arrays["illegal_contact"], arrays["fallen"]
    check(np.array_equal(arrays["terminated"], flagged | fallen)
          and int(flagged.sum()) == report["coverage"]["illegal_contact_ticks"]
          and int(fallen.sum()) == report["coverage"]["fallen_ticks"], folder.name + " safety flags")
    episodes = np.cumsum(arrays["episode_tick"] == 1) - 1
    events = []
    for i in np.flatnonzero(flagged):
        window = range(max(0, i - min(2, int(arrays["episode_tick"][i]) - 1)), i + 1)
        terrain = TERRAINS[int(arrays["terrain"][i])]
        source = [{"index": int(j), **{k: arr[j] for k, arr in arrays.items()}} for j in window]
        check(all(episodes[j] == episodes[i] and arrays["terrain"][j] == arrays["terrain"][i] for j in window),
              folder.name + " crossed reset when selecting contact snapshots")
        events.append(({"model": folder.name, "terrain": terrain, "training_seed": report["training_seed"],
                        "episode_index": int(episodes[i]), "terminal_rollout_row": int(i),
                        "end_time_s": float(arrays["episode_tick"][i] / hz),
                        "slope_start": bool(arrays["slope_start"][i]), "start_x_m": float(arrays["start_x_m"][i])}, source))
    return {"model": folder.name, "train_report_sha256": sha(folder / "report.json"), "policy_sha256": model_sha,
            "trace_sha256": sha(path), "verified_arrays": len(declared), "archived_ticks": n,
            "illegal_contact_intervals": int(flagged.sum()), "fallen_intervals": int(fallen.sum())}, events


def native_contacts(env, threshold):
    """Enumerate the contacts and cross-check them against the frozen diagnostic."""
    model, data = env.model, env.data
    support_types = {mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_HFIELD}
    permitted_support = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0 and model.geom_contype[g]
                         and model.geom_type[g] in support_types}
    other_ground_contacts = []
    force = np.empty(6)
    for i, c in enumerate(data.contact[:data.ncon]):
        a, b = int(c.geom1), int(c.geom2)
        if (a in permitted_support) == (b in permitted_support):
            continue
        world, geom = (a, b) if a in permitted_support else (b, a)
        body = int(model.geom_bodyid[geom])
        if not body or geom in env.wheel_geoms:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        other_ground_contacts.append({"body": model.body(body).name, "robot_geom_id": geom,
            "robot_geom_type": mujoco.mjtGeom(int(model.geom_type[geom])).name,
            "world_geom": model.geom(world).name, "contact_world_xyz_m": c.pos.tolist(),
            "road_height_at_contact_x_m": float(ground_height(env.current_terrain, c.pos[0])),
            "normal_world_from_ground": (c.frame[:3] if world == a else -c.frame[:3]).tolist(),
            "signed_contact_distance_m": float(c.dist), "cold_normal_force_n": float(force[0]),
            "above_threshold": bool(force[0] > threshold)})
    reconstructed, self_contacts = env._contact_diagnostics()
    check(reconstructed == any(c["above_threshold"] for c in other_ground_contacts), "frozen contact classification")
    return bool(reconstructed), int(self_contacts), other_ground_contacts


def boundary(env, source, hz, threshold, slots):
    model, data = env.model, env.data
    qpos, qvel = source["qpos"], source["qvel"]
    mujoco.mj_resetData(model, data)
    data.time = float(source["episode_tick"] / hz)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.qfrc_applied[env.dof_ids] = source["last_torque_nm"]
    mujoco.mj_forward(model, data)
    reproduced, self_contacts, contacts = native_contacts(env, threshold)
    wheel_mask = loaded_wheels_past_toe(env, slots)
    quat = env._sensor("trunk_quat")
    w, x, y, z = quat
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(float(np.clip(2*(w*y-z*x), -1., 1.)))
    vx = inverse_rotate(quat, env._sensor("frame_vel"))[0]
    check(abs(vx - source["body_vx_m_s"]) <= 1e-9, "recorded pose and forward-speed sensor disagree")
    return {"rollout_row": source["index"], "episode_tick": int(source["episode_tick"]),
            "time_s": data.time, "base_world_xyz_m": qpos[:3].tolist(), "base_quat_wxyz": quat.tolist(),
            "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch),
            "base_vertical_height_above_road_m": float(qpos[2] - ground_height(env.current_terrain, qpos[0])),
            "command_m_s": float(source["command_vx_m_s"]), "body_vx_m_s": float(vx),
            "archived_interval_illegal": bool(source["illegal_contact"]), "archived_fallen": bool(source["fallen"]),
            "archived_wheel_support_mask": source["loaded_wheels_past_toe"].tolist(),
            "cold_wheel_support_mask": wheel_mask.tolist(), "cold_illegal_at_endpoint": reproduced,
            "cold_self_contact_count": self_contacts, "nonwheel_support_contacts": contacts}


def audit():
    plan = registered_plan()
    course = read(PLAN.parents[2] / plan["parent_protocol"])["course"]
    complete_stage("training")
    folder = BASE / "training_01"
    models = checked_models(folder, plan)
    summary, pending = [], []
    for (arm, seed), (weights, digest) in models.items():
        row, cases = source_events(weights.parent, digest, course["policy_hz"])
        summary.append(row)
        pending.extend(cases)
    events = []
    for terrain in TERRAINS:
        selected = [(row, source) for row, source in pending if row["terrain"] == terrain]
        if not selected:
            continue
        env = make_env(terrain, selected[0][0]["training_seed"], max_transitions=1)
        try:
            env.reset(seed=selected[0][0]["training_seed"])
            slots = wheel_geom_slots(env)
            threshold = env.config["termination"]["illegal_support_contact_normal_force_above_n"]
            for row, source in selected:
                snapshots = [boundary(env, s, course["policy_hz"], threshold, slots) for s in source]
                bodies = sorted({c["body"] for c in snapshots[-1]["nonwheel_support_contacts"] if c["above_threshold"]})
                events.append({**row, "normal_force_threshold_n": threshold,
                               "confirmed_bodies_at_endpoint": bodies, "threshold_at_endpoint_reconstructed": bool(bodies),
                               "history": snapshots})
            check(all(env.counters()[k] == 0 for k in ("native_steps", "policy_transitions", "pd_updates")),
                  "training endpoint readback integrated physics")
        finally:
            env.close()
    return {"status": "COMPLETE", "protocol_sha256": sha(PLAN), "source_sha256": sha(__file__),
            "training_report_sha256": sha(folder / "report.json"), "native_steps": 0, "optimizer_steps": 0,
            "method": "Cold forward dynamics from recorded qpos, qvel and last applied PD torques, "
                      "using the frozen illegal-contact classifier. Its recorded interval flag is an OR over ten native "
                      "steps and the final boundary. Only 20 ms endpoints are stored; original solver warm-starts, "
                      "substep forces and the first offending part within an interval are unavailable.",
            "models": summary, "events": events, "observed_flagged_intervals": len(events),
            "cold_positive_endpoints": sum(case["threshold_at_endpoint_reconstructed"] for case in events),
            "checked_earlier_endpoints": sum(len(case["history"])-1 for case in events)}


if __name__ == "__main__":
    check_worker("sensor")
    with (BASE / "training_contacts_01.json").open("x") as stream:
        try:
            result = audit()
        except BaseException as exc:
            json.dump({"status": "INCOMPLETE", "error": str(exc)}, stream)
            raise
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
