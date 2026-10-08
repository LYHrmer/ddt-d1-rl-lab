#!/usr/bin/env python3
"""One bounded fixed-action diagnostic; no training and no automatic reset."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import mujoco
import numpy as np
from d1_rl.env import D1Env

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                               default=json_numpy)+"\n")

def json_numpy(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-native", type=int, required=True)
    args = parser.parse_args()
    if not 10 <= args.max_native <= 600 or args.max_native % 10:
        parser.error("native budget must be a multiple of 10, within [10,600]")
    fixture = json.loads(args.fixture.read_text())
    actions = np.asarray(fixture["actions"], dtype=np.float32)
    if (actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all()
            or len(actions)*10 > args.max_native):
        parser.error("fixed fixture actions must fit the native budget")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__).resolve(), args.config.resolve(), args.fixture.resolve(),
             ROOT/"training/d1_rl/env.py", ROOT/"training/d1_rl/interface.py"]
    identities = {str(p):sha(p) for p in paths}
    dump(args.output/"protocol.json", {"mode":"fixed_action_diagnostic_only", "max_native":args.max_native,
         "seed":fixture.get("seed",7001), "delay_ticks":fixture.get("delay_ticks",0),
         "inputs_sha256":identities,"mujoco_version":mujoco.__version__,
         "teacher_or_training":False,"extra_forward_during_trace":False})
    dump(args.output/"actions.json", fixture)
    env = None
    native_count = 0
    started = time.monotonic()
    original_step = mujoco.mj_step
    rows = []
    native_stream = (args.output/"native_trace.jsonl").open("x")
    policy_stream = (args.output/"policy_trace.jsonl").open("x")
    status, error = "ERROR", None
    def counted_step(model, data):
        nonlocal native_count
        if native_count >= args.max_native or time.monotonic()-started > 30:
            raise RuntimeError("Diagnostic hard native/wall limit reached")
        native_count += 1
        (args.output/"native_count.json").write_text(json.dumps({"attempted":native_count}))
        original_step(model, data)
        row = {"native_step":native_count,"time_s":float(data.time),
               "qpos":data.qpos.tolist(),"qvel":data.qvel.tolist(),
               "qfrc_applied":data.qfrc_applied.tolist(),
               "qacc_warmstart":data.qacc_warmstart.tolist()}
        native_stream.write(json.dumps(row,allow_nan=False)+"\n")
    try:
        env = D1Env(config=args.config, seed=fixture.get("seed",7001),
                    eval_delay_ticks=fixture.get("delay_ticks",0), max_transitions=len(actions))
        env.reset()
        env.set_command(0.,0.)
        dump(args.output/"initial_state.json",env.get_initial_state())
        mujoco.mj_step = counted_step
        for index, action in enumerate(actions):
            obs,reward,terminated,truncated,info = env.step(action)
            row = {"policy_tick":index+1,"time_s":float(env.data.time),
                   "issued_action":action.tolist(),"reward":reward,
                   "terminated":terminated,"truncated":truncated,
                   "observation":obs.tolist(),"info":info}
            rows.append(row)
            policy_stream.write(json.dumps(row,allow_nan=False,default=json_numpy)+"\n")
            if terminated or truncated:
                status="TERMINATED" if terminated else "TRUNCATED"
                break
        else:
            status="COMPLETED_FIXED_HORIZON"
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
        (args.output/"error.txt").write_text(traceback.format_exc())
    finally:
        mujoco.mj_step = original_step
        native_stream.close(); policy_stream.close()
        result={"status":status,"error":error,"native_attempts":native_count,
                "actual_counts":env.counters() if env else None,
                "elapsed_s":time.monotonic()-started,
                "inputs_unchanged":all(sha(p)==h for p,h in identities.items()),
                "last_metrics":rows[-1]["info"] if rows else None,
                "peak_tilt_rad":max((r["info"]["tilt_rad"] for r in rows),default=None),
                "min_height_m":min((r["info"]["base_height_m"] for r in rows),default=None)}
        if env: env.close()
        dump(args.output/"report.json",result)
        print(json.dumps(result,indent=2,ensure_ascii=False,default=json_numpy))
    return 1 if status=="ERROR" or not result["inputs_unchanged"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
