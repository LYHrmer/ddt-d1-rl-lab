"""Run reset contact checks followed by one paired PPO rollout per arm and training seed."""
import hashlib
import json
from pathlib import Path

import numpy as np

from tools89.common89 import BASE, PLAN, complete_stage, parent_folder, read, registered_plan, sha, write


def preflight_case(folder, arm, seed, predict, plan):
    from tools89.course89 import make_env

    ticks = plan["preflight"]["ticks_per_episode"]
    path = folder / f"{arm}_{seed}"
    path.mkdir()
    env = make_env("up", seed, curriculum=(arm == "mixed"), slope_start_x_m=plan["curriculum"]["slope_start_x_m"],
                   perception=True, max_transitions=ticks)
    rows = {k: [] for k in ("qpos", "body_vx_m_s", "command", "issued", "applied", "ranges",
                            "wheel_contacts_past_toe", "fallen", "illegal")}
    complete = False
    try:
        env.reset(seed=seed)
        obs, info = env.reset()
        initial_pose = env.data.qpos.copy()
        if info["delay_ticks"] != 1:
            raise ValueError("preflight action delay differs")
        for _ in range(ticks):
            raw = env.last_raw_features.copy()
            if obs.shape != (643,) or not np.array_equal(obs[627:], raw):
                raise ValueError("preflight did not deliver the actual range packet")
            issued = predict(obs[:627], raw)
            obs, reward, terminated, truncated, info = env.step(issued)
            if info["slope_start"] != (arm == "mixed") or not np.isfinite(reward):
                raise ValueError("reset placement or reward differs")
            for key, value in (("qpos", env.data.qpos.copy()), ("body_vx_m_s", info["body_vx_m_s"]),
                               ("command", info["command_during_interval"][0]), ("issued", issued),
                               ("applied", info["applied_action"]), ("ranges", raw),
                               ("wheel_contacts_past_toe", info["loaded_wheels_past_toe"]),
                               ("fallen", info["fallen"]), ("illegal", info["illegal_body_contact"])):
                rows[key].append(value)
            if terminated or truncated:
                break
        mask = np.asarray(rows["wheel_contacts_past_toe"], dtype=bool)
        moving = np.asarray(rows["command"]) > 0
        n = len(moving)
        both = mask[:, :2].any(axis=1) & mask[:, 2:].any(axis=1)
        counts = env.counters()
        if ((counts["policy_transitions"], counts["pd_updates"], counts["native_steps"]) != (n, 5*n, 10*n)
                or not np.isfinite(rows["qpos"]).all() or not np.isfinite(rows["body_vx_m_s"]).all()
                or not np.any(np.asarray(rows["ranges"])[:, 8:]) or not np.array_equal(rows["applied"][1:], rows["issued"][:-1])):
            raise ValueError("preflight trace or physical counters differ")
        clean = n == ticks and not (any(rows["fallen"]) or any(rows["illegal"]))
        both_n, all_n = int(np.sum(both & moving)), int(np.sum(mask.all(axis=1) & moving))
        touched_rear = bool(np.any(mask[:, 2:]))
        gate = clean and int(moving.sum()) == 50 and ((both_n >= 25 and all_n >= 1) if arm == "mixed" else not touched_rear)
        result = {"arm": arm, "seed": seed, "ticks": n, "pd_updates": 5*n, "native_steps": 10*n,
                  "initial_base_x_m": float(initial_pose[env.free_qpos]), "moving": int(moving.sum()),
                  "both_axles_on_slope_moving": both_n, "four_wheels_on_slope_moving": all_n,
                  "any_rear_on_slope": touched_rear, "clean": clean, "passed": gate}
        complete = True
    finally:
        try:
            if rows["qpos"]:
                trace = path / ("trace.npz" if complete else "partial_trace.npz")
                with trace.open("xb") as stream:
                    np.savez_compressed(stream, **rows, initial_qpos=initial_pose)
        finally:
            env.close()
    result["trace_sha256"] = sha(trace)
    write(path / "metrics.json", result)
    return result


def paired_training(folder, seed, plan):
    """Read both native rollouts from disk, retaining the first differing reset in the record."""
    data, reports = {}, {}
    for arm in plan["training"]["arms"]:
        path = folder / f"{arm}_{seed}"
        report = reports[arm] = read(path / "report.json")
        with np.load(path / "train_trace.npz", allow_pickle=False) as z:
            data[arm] = {k: z[k] for k in z.files}
        if (report["status"] != "COMPLETE" or any(hashlib.sha256(v.tobytes()).hexdigest()
                != report["trace"]["array_sha256"][k] for k, v in data[arm].items())):
            raise ValueError(f"{path}: training archive integrity failure")
    original, mixed = (data[k] for k in ("original", "mixed"))
    provenance_keys = ("parent_policy_sha256", "inheritance", "initial_probe_actions_sha256", "pre_rollout_torch_rng_sha256")
    if any(reports["original"][k] != reports["mixed"][k] for k in provenance_keys):
        raise ValueError(f"seed {seed}: paired model, Adam state or initial RNG differs")
    n = plan["training"]["transitions_per_model"]
    if len(original["reward"]) != n or len(mixed["reward"]) != n or np.any(original["slope_start"]):
        raise ValueError("unexpected original training placement or budget")
    onsets = np.flatnonzero(mixed["slope_start"])
    if not onsets.size:
        raise ValueError("the registered training curriculum never activated")
    prefix = int(onsets[0])
    if prefix == 0:
        raise ValueError("both arms must start on the same course")
    for key in original:
        if not np.array_equal(original[key][:prefix], mixed[key][:prefix]):
            raise ValueError(f"seed {seed}: {key} differs before the first changed reset")
    measures = {}
    for arm, a in data.items():
        up = a["terrain"] == 1
        moving = up & (a["command_vx_m_s"] > 0)
        contacts = a["loaded_wheels_past_toe"]
        if contacts.shape != (n, 4) or not moving.any():
            raise ValueError("invalid four-wheel coverage data")
        both = contacts[:, :2].any(axis=1) & contacts[:, 2:].any(axis=1)
        n_four = int(np.sum(contacts.all(axis=1) & moving))
        measures[arm] = {"uphill_ticks": int(up.sum()), "uphill_moving_ticks": int(moving.sum()),
                         "both_axles_on_slope_moving": int(np.sum(both & moving)), "all_four_on_slope_moving": n_four,
                         "all_four_fraction": n_four / int(moving.sum()),
                         "uphill_moving_base_past_toe": int(np.sum(a["base_origin_past_toe"] & moving)),
                         "terminated_episodes": int(a["terminated"].sum())}
        if (n_four != reports[arm]["coverage"]["uphill_moving_four_loaded_ticks"]
                or int(moving.sum()) != reports[arm]["coverage"]["uphill_moving_ticks"]):
            raise ValueError(f"seed {seed}: independently computed support counts differ")
    gain = measures["mixed"]["all_four_fraction"] - measures["original"]["all_four_fraction"]
    return {"train_seed": seed, "identical_prefix_ticks": prefix, **measures, "fraction_increase": gain,
            "coverage_pass": measures["mixed"]["all_four_fraction"] >= plan["coverage"]["minimum_mixed_fraction"]
            and gain >= plan["coverage"]["minimum_paired_fraction_increase"]}


def run(stage):
    plan = registered_plan()
    if stage == "smoke":
        geometry = complete_stage("admission")["zero_preflight"]
        if geometry["native_steps"] or len(geometry["mixed_uphill_starts"]) != len(plan["preflight"]["reset_seeds"]):
            raise RuntimeError("reset geometry not qualified for physical smoke")
    if stage == "training" and not complete_stage("smoke")["preflight_pass"]:
        raise RuntimeError("preflight contact gate failed; training is closed for this run")
    folder = BASE / f"{stage}_01"
    folder.mkdir(exist_ok=False)
    receipt = {"stage": stage, "status": "INCOMPLETE", "protocol_sha256": sha(PLAN),
               "sources": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
               "cases": [], "models": []}
    try:
        train = plan["training"]
        if stage == "admission":
            from tools89.course89 import zero_preflight

            for parent in train["parent_seeds"]:
                parent_folder(parent, plan)
            params = plan["preflight"]
            receipt["zero_preflight"] = zero_preflight(seeds=params["reset_seeds"], slope_start_x_m=plan["curriculum"]["slope_start_x_m"])
        elif stage == "smoke":
            from tools87.ppo87 import load_predictor

            params = plan["preflight"]
            for parent, seed in zip(train["parent_seeds"], params["reset_seeds"], strict=True):
                predict = load_predictor(parent_folder(parent, plan))
                for arm in train["arms"]:
                    c = preflight_case(folder, arm, seed, predict, plan)
                    receipt["cases"].append(c)
                    print(json.dumps(c), flush=True)
            receipt["preflight_pass"] = (len(receipt["cases"]) == params["max_episodes"]
                                         and all(c["passed"] for c in receipt["cases"]))
        else:
            from tools89.train89 import train_one

            receipt["coverage_pairs"] = []
            for parent, seed in zip(train["parent_seeds"], train["seeds"], strict=True):
                for arm in train["arms"]:
                    name = f"{arm}_{seed}"
                    print(f"Starting {name}", flush=True)
                    model_folder = folder / name
                    train_one(arm, parent, seed, model_folder)
                    model = read(model_folder / "report.json")
                    if (model["status"] != "COMPLETE" or sha(model_folder / "policy.zip") != model["policy_sha256"]
                            or model["counts"]["policy_transitions"] != train["transitions_per_model"]):
                        raise ValueError(f"incomplete model for {name}")
                    receipt["models"].append({"model": name, "parent_seed": parent, "train_seed": seed, "arm": arm,
                                              "report_sha256": sha(model_folder / "report.json"),
                                              "policy_sha256": model["policy_sha256"], "optimizer_steps": model["optimizer_steps"]})
                pair = paired_training(folder, seed, plan)
                receipt["coverage_pairs"].append(pair)
                print(json.dumps(pair), flush=True)
            receipt["coverage_pass"] = all(p["coverage_pass"] for p in receipt["coverage_pairs"])
        receipt["status"] = "COMPLETE"
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write(folder / "report.json", receipt)
