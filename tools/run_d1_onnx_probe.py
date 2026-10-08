#!/usr/bin/env python3
"""Bounded learned-ONNX Python trajectory for independent C++ comparison."""
import argparse
import json
from pathlib import Path
import time
import traceback

import mujoco
from d1_rl.factory03 import environment_class, environment_sources
from evaluate_d1_ppo import make_predictor, sha256
from run_d1_fixed_probe import dump, json_numpy

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-native",type=int,default=500)
    a=p.parse_args()
    if not 10<=a.max_native<=500 or a.max_native%10:
        p.error("Allowed native cap is a multiple of 10 within [10,500]")
    a.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__).resolve(),ROOT/"tools/evaluate_d1_ppo.py",ROOT/"tools/run_d1_fixed_probe.py",
             *environment_sources(a.config),a.config.resolve(),
             a.onnx.resolve(),a.onnx.resolve().parent/"metadata.json"]
    identities={str(path):sha256(path) for path in sources}
    dump(a.output/"protocol.json",{"mode":"learned_onnx_dynamic_parity", "max_native":a.max_native,
         "seed":7001,"delay_ticks":0,"policy_sha256":sha256(a.onnx),
         "onnx_metadata_sha256":sha256(a.onnx.parent/"metadata.json"),
         "inputs_sha256":identities,"mujoco_version":mujoco.__version__,
         "extra_forward_during_trace":False})
    env=None; native=0; rows=[]; status="ERROR"; error=None; began=time.monotonic()
    step=mujoco.mj_step
    native_stream=(a.output/"native_trace.jsonl").open("x")
    policy_stream=(a.output/"policy_trace.jsonl").open("x")
    def record_step(model,data):
        nonlocal native
        if native>=a.max_native or time.monotonic()-began>60:
            raise RuntimeError("ONNX pair probe native/wall cap")
        native+=1
        (a.output/"native_count.json").write_text(json.dumps({"attempted":native}))
        step(model,data)
        row={"native_step":native,"time_s":float(data.time),"qpos":data.qpos.tolist(),
             "qvel":data.qvel.tolist(),"qfrc_applied":data.qfrc_applied.tolist(),
             "issued_action":env._last_policy_action.tolist(),
             "executed_action":env._applied_action.tolist()}
        native_stream.write(json.dumps(row,allow_nan=False)+"\n")
    try:
        predictor,provenance=make_predictor(None,a.onnx.resolve(),a.config.resolve())
        dump(a.output/"policy_provenance.json",provenance)
        env=environment_class(a.config)(config=a.config,seed=7001,eval_delay_ticks=0,max_transitions=a.max_native//10)
        obs,_=env.reset(); env.set_command(0.,0.);obs=env.current_observation()
        dump(a.output/"initial_state.json",env.get_initial_state())
        mujoco.mj_step=record_step
        for tick in range(a.max_native//10):
            action=predictor(obs)
            obs,reward,terminated,truncated,info=env.step(action)
            row={"policy_tick":tick+1,"time_s":float(env.data.time),"issued_action":action.tolist(),
                 "reward":reward,"terminated":terminated,"truncated":truncated,"info":info}
            rows.append(row);policy_stream.write(json.dumps(row,default=json_numpy,allow_nan=False)+"\n")
            if terminated or truncated:
                status="TERMINATED" if terminated else "TRUNCATED";break
        else: status="COMPLETED_FIXED_HORIZON"
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}";(a.output/"error.txt").write_text(traceback.format_exc())
    finally:
        mujoco.mj_step=step;native_stream.close();policy_stream.close()
        report={"status":status,"error":error,"native_attempts":native,
                "actual_counts":env.counters() if env else None,
                "inputs_unchanged":all(sha256(path)==identity for path,identity in identities.items()),
                "elapsed_s":time.monotonic()-began,"last_metrics":rows[-1]["info"] if rows else None}
        dump(a.output/"report.json",report)
        if env:env.close()
        print(json.dumps({k:v for k,v in report.items() if k!="last_metrics"},indent=2))
    return int(status=="ERROR" or not report["inputs_unchanged"])

if __name__=="__main__":raise SystemExit(main())
