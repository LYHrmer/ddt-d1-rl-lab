#!/usr/bin/env python3
"""One separately budgeted 640-transition environment profile, no learning."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import time
import traceback

import numpy as np
from d1_rl.env import D1Env

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--mode",required=True,choices=["timing","profile"])
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    config=ROOT/"training/configs/pilot03b.json"
    fixture=ROOT/"training/configs/diagnostic03b_static_support.json"
    sources=[config,fixture,ROOT/"training/d1_rl/env.py",ROOT/"training/d1_rl/interface.py",Path(__file__).resolve()]
    ids={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (a.output/"protocol.json").write_text(json.dumps({"max_transitions":640,"max_native_steps":6400,
        "purpose":"fixed-support engineering profile; no training or policy qualification",
        "inputs_sha256":ids,"seed":7001,"max_wall_seconds":60,"mode":a.mode,
        "cold_policy_ticks":64,"hot_policy_ticks":576},indent=2)+"\n")
    env=None;profiler=cProfile.Profile();started=time.monotonic();status="ERROR";error=None
    setup_s=cold_s=hot_s=None
    try:
        env=D1Env(config=config,seed=7001,eval_delay_ticks=0,max_transitions=640)
        env.reset();env.set_command(0.,0.)
        action=np.asarray(json.loads(fixture.read_text())["actions"][0],dtype=np.float32)
        setup_s=time.monotonic()-started
        loop_start=time.monotonic();hot_start=None
        if a.mode=="profile":profiler.enable()
        for tick in range(640):
            if time.monotonic()-started>60:raise RuntimeError("Profile wall cap")
            _,_,done,truncated,_=env.step(action)
            if tick==63:
                hot_start=time.monotonic();cold_s=hot_start-loop_start
            if done or truncated:
                status="EARLY_TERMINATION";break
        else:
            status="COMPLETED";hot_s=time.monotonic()-hot_start
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}";(a.output/"error.txt").write_text(traceback.format_exc())
    finally:
        profiler.disable()
        if a.mode=="profile":
            profiler.dump_stats(str(a.output/"profile.pstats"))
            with (a.output/"profile.txt").open("x") as stream:
                pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats("cumulative").print_stats(60)
        result={"status":status,"error":error,"elapsed_total_s":time.monotonic()-started,
                "actual_counts":env.counters() if env else None,"setup_seconds":setup_s,
                "cold_64_seconds":cold_s,"hot_576_seconds":hot_s,
                "hot_transitions_per_s":576/hot_s if hot_s else None,
                "inputs_unchanged":all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in ids.items())}
        (a.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        if env:env.close()
        print(json.dumps(result,indent=2))
    return int(status!="COMPLETED" or not result["inputs_unchanged"])

if __name__=="__main__":raise SystemExit(main())
