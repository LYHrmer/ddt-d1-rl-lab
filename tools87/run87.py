"""Run the bounded native D1 full-body policy and real perceptive PPO comparison."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from tools87.scoring87 import score, select_video

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/experiments87/full_body_perception.json"
REPORT34 = ROOT / "evidence/cpp_reference34/execution_01/report.json"
MODEL = ROOT / "evidence/low_memory_delay25/execution_01/random/export/policy.onnx"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, values):
    with path.open("x") as stream:
        json.dump(values, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_inputs(plan):
    identity = plan["parent"]
    report = read(REPORT34)
    binary = Path(report["cpp_build"]["binary"])
    if (sha(REPORT34) != identity["report34_sha256"] or report["status"] != "COMPLETE"
            or not report["cpp_original_and_holdout_qualified"] or not report["cleanup_complete"]
            or sha(binary) != identity["cpp34_sha256"] or sha(MODEL) != identity["onnx_sha256"]
            or sha(ROOT / identity["checkpoint"]) != identity["checkpoint_sha256"]):
        raise ValueError("complete policy25/C++34 identities do not match the protocol")
    return binary


@contextmanager
def cpp_actor(folder, plan, binary):
    from tools34.cpp_reference34 import CppReferenceActor, SOURCE
    from tools19.cpp_actor19 import ORT_LIBRARY

    parent = read(REPORT34)["stages"]["formal"]["cases"][0]["cpp_reference34"]["transport"]["identities"]
    if sha(SOURCE) != parent["source_sha256"] or sha(ORT_LIBRARY) != parent["ort_library_sha256"]:
        raise ValueError("original C++34 source or ORT library changed")
    actor = CppReferenceActor(binary, MODEL, output=folder / "cpp34",
                              expected_binary_sha256=plan["parent"]["cpp34_sha256"],
                              expected_model_sha256=plan["parent"]["onnx_sha256"],
                              expected_source_sha256=parent["source_sha256"],
                              source=SOURCE, max_requests=plan["course"]["ticks"])
    try:
        actor.start()
        yield lambda body, ranges: actor.predict(body)
    finally:
        receipt = actor.close()
        if receipt["status"] != "CLOSED" or not receipt["cleanup_complete"]:
            raise RuntimeError("C++34 transport did not close cleanly")


@contextmanager
def predictor(folder, arm, training_dir, plan, binary):
    if arm == "baseline":
        with cpp_actor(folder, plan, binary) as predict:
            yield predict
    else:
        from tools87.ppo87 import load_predictor
        yield load_predictor(training_dir)


def snapshot(env, data, vx, cmd, action, ranges, info, mask_difference):
    # Boundary i>0 describes interval [i-1,i). Ranges are the raw causal packet
    # available when the actor issued that interval's action, before integration.
    row = {
        "qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(), "time_s": float(env.data.time),
        "body_x_m": float(env.data.qpos[env.free_qpos]), "body_vx_m_s": vx, "command_vx_m_s": cmd,
        "pitch_rad": info.get("base_pitch_rad", 0.0), "roll_rad": info.get("base_roll_rad", 0.0),
        "front_contact_x_m": info.get("front_contact_x_m"), "front_axle_x_m": info.get("front_axle_x_m"),
        "wheel_saturation_count": info.get("wheel_saturation_count", 0),
        "fallen": bool(info.get("fallen", False)), "illegal_body_contact": bool(info.get("illegal_body_contact", False)),
        "issued_action": action, "applied_action": info.get("applied_action", np.zeros(16)),
        "raw_ranges16": ranges, "mask_action_max_abs": mask_difference,
        "last_torque_nm": info.get("last_torque_nm", np.zeros(16)),
        "leg_target_rad": info.get("leg_target_rad", np.zeros(12)),
        "wheel_target_rad_s": info.get("wheel_target_rad_s", np.zeros(4)),
    }
    for key, value in row.items():
        data.setdefault(key, []).append(np.nan if value is None else value)


def episode(output, terrain, seed, arm, training_seed, training_dir, plan, binary):
    from tools87.terrain87 import make_env

    case = f"{arm}_{training_seed or 0}_{terrain}_{seed}"
    folder = output / case
    folder.mkdir()
    course = plan["course"]
    env = make_env(terrain, seed, perception=arm != "masked", max_transitions=course["ticks"])
    data = {}
    complete = False
    try:
        with predictor(folder, arm, training_dir, plan, binary) as predict:
            obs, reset_info = env.reset(seed=seed)
            if reset_info["delay_ticks"] != course["action_delay_ticks"] or obs.shape != (643,):
                raise ValueError("D1 reset contract changed")
            snapshot(env, data, 0., 0., np.zeros(16), np.zeros(16), {}, 0.)
            for _ in range(course["ticks"]):
                raw = env.last_raw_features.copy()
                body = obs[:627]
                features = obs[627:]
                expected = raw if arm != "masked" else np.zeros_like(raw)
                if not np.array_equal(expected, features):
                    raise ValueError("actor did not receive the registered terrain observation")
                action = np.asarray(predict(body, features), dtype=np.float32)
                if action.shape != (16,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
                    raise ValueError("invalid whole-body action")
                difference = (float(np.max(np.abs(action - predict(body, np.zeros_like(raw)))))
                              if arm == "range" else 0.0)
                obs, _, terminated, truncated, info = env.step(action)
                snapshot(env, data, info["body_vx_m_s"], info["command_during_interval"][0],
                         action.copy(), raw, info, difference)
                if terminated or truncated:
                    break
        metrics = score(data, course, plan["evaluation"]["course_mark_m_after_toe"])
        counts = env.counters()
        n = metrics["ticks"]
        if (counts["policy_transitions"], counts["native_steps"], counts["pd_updates"]) != (n, 10*n, 5*n):
            raise ValueError("native/servo/actor counters differ from interval trace")
        if not metrics["full_horizon"] and not metrics["terminated"]:
            raise ValueError("unexplained incomplete evaluation trial")
        complete = True
    finally:
        if data:
            trace = folder / ("trace.npz" if complete else "partial_trace.npz")
            with trace.open("xb") as stream:
                np.savez_compressed(stream, **{k: np.asarray(v) for k, v in data.items()})
        env.close()
    metrics.update(case=case, status="COMPLETE", arm=arm, training_seed=training_seed, seed=seed,
                   terrain=terrain, native_steps=counts["native_steps"], pd_updates=counts["pd_updates"],
                   trace_sha256=sha(trace), protocol_sha256=sha(PLAN))
    write(folder / "metrics.json", metrics)
    print(json.dumps({k: metrics[k] for k in ("case","ticks","fixed_speed_rmse_m_s","reached_mark","fallen")}), flush=True)
    return metrics


def evidence(path, stage):
    if path is None:
        raise ValueError(f"missing completed {stage} output")
    receipt = read(path / "report.json")
    if (receipt["stage"] != stage or receipt["status"] != "COMPLETE"
            or receipt["protocol_sha256"] != sha(PLAN)):
        raise ValueError(f"unqualified {stage} input")
    return receipt


def geometry_preflight(course):
    """Use native ray intersections and reset poses without advancing dynamics."""
    import mujoco
    from tools87.terrain87 import make_env, ground_height

    results = {}
    initial_qpos = None
    for terrain in course["terrains"]:
        env = make_env(terrain, 87101)
        try:
            obs, info = env.reset(seed=87101)
            if (obs.shape != (643,) or np.count_nonzero(obs[627:]) or info["delay_ticks"] != 1
                    or (initial_qpos is not None and not np.array_equal(initial_qpos, env.data.qpos))):
                raise ValueError("terrain reset pose, initial measurements or delay differs")
            initial_qpos = env.data.qpos.copy()
            heights = []
            for x in (0.0, 1.0, 1.6):
                geom = np.array([-1], np.int32)
                distance = mujoco.mj_ray(env.model, env.data, np.array([x, 0.8, 1.0]),
                                         np.array([0., 0., -1.]), None, True, -1, geom)
                measured = 1 - float(distance)
                if geom[0] != env.floor_geom or abs(measured - ground_height(terrain, x)) > 2e-6:
                    raise ValueError(f"native terrain ray differs at {terrain}/{x}: {measured}")
                heights.append(measured)
            if env.counters()["native_steps"] != 0:
                raise ValueError("geometry preflight integrated the plant")
            results[terrain] = heights
        finally:
            env.close()
    return results


def run(args):
    from tools27.resource_guard27 import read_memory

    scope = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[-1]
    available = read_memory(scope)["effective_available_bytes"]
    if available < 4 << 30 or os.getpriority(os.PRIO_PROCESS, 0) != 10 or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError("requires a single-CPU Nice10 systemd unit with 4 GiB startup headroom")
    args.output.mkdir(parents=True, exist_ok=False)
    plan = read(PLAN)
    binary = parent_inputs(plan)
    receipt = {"stage": args.stage, "status": "INCOMPLETE", "protocol_sha256": sha(PLAN),
               "cases": [], "models": [], "started_effective_bytes": available,
               "source_sha256": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    started = time.monotonic()
    try:
        if args.stage == "preflight":
            from tools87.ppo87 import preflight
            receipt["ppo"] = preflight()
            receipt["native_heights_at_x_0_1_1p6_m"] = geometry_preflight(plan["course"])
        if args.stage == "baseline":
            for seed in plan["baseline"]["reset_seeds"]:
                for terrain in plan["course"]["terrains"]:
                    receipt["cases"].append(episode(args.output, terrain, seed, "baseline", None, None, plan, binary))
        if args.stage == "training":
            evidence(args.baseline, "baseline")
            evidence(args.preflight, "preflight")
            from tools87.ppo87 import train
            for seed in plan["training"]["seeds"]:
                for arm in plan["training"]["arms"]:
                    model = f"{arm}_{seed}"
                    print(f"Training {model}", flush=True)
                    train(arm, seed, args.output / model)
                    result = read(args.output / model / "report.json")
                    if result["status"] != "COMPLETE":
                        raise ValueError(f"incomplete training model {model}")
                    receipt["models"].append({"model": model, "arm": arm, "training_seed": seed,
                                              "report_sha256": sha(args.output / model / "report.json")})
                blind, sensed = [read(args.output / f"{a}_{seed}" / "report.json")
                                 for a in plan["training"]["arms"]]
                keys = ("raw_range", "issued", "applied", "qpos", "reward", "terrain", "terminated", "truncated")
                same = (blind["trace"]["proprio627_sha256"] == sensed["trace"]["proprio627_sha256"]
                        and all(blind["trace"]["array_sha256"][k] == sensed["trace"]["array_sha256"][k]
                                for k in keys))
                receipt.setdefault("paired_rollout_equal", {})[str(seed)] = same
                if not same:
                    raise ValueError(f"the pre-update native rollouts for seed {seed} differ")
        if args.stage == "evaluate":
            training = evidence(args.training, "training")
            identities = {m["model"]: m["report_sha256"] for m in training["models"]}
            for train_seed, seed in zip(plan["training"]["seeds"], plan["evaluation"]["heldout_reset_seeds"], strict=True):
                for terrain in plan["course"]["terrains"]:
                    for arm in ["baseline", *plan["training"]["arms"]]:
                        model_id = f"{arm}_{train_seed}"
                        folder = args.training / model_id if arm != "baseline" else None
                        if folder is not None and sha(folder / "report.json") != identities[model_id]:
                            raise ValueError(f"changed training receipt for {model_id}")
                        receipt["cases"].append(episode(args.output, terrain, seed, arm,
                            train_seed if folder else None, folder, plan, binary))
            receipt["video_selection"] = select_video(receipt["cases"])
        receipt["status"] = "COMPLETE"
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["elapsed_s"] = time.monotonic() - started
        write(args.output / "report.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "baseline", "training", "evaluate"))
    parser.add_argument("--output", required=True, type=Path)
    for name in ("baseline", "preflight", "training"):
        parser.add_argument("--" + name, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
