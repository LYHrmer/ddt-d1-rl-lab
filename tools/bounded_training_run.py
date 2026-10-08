#!/usr/bin/env python3
"""Independent process-group deadline around a D1 training CLI invocation."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deadline",required=True,type=float)
    p.add_argument("training_args",nargs=argparse.REMAINDER)
    a=p.parse_args()
    args=a.training_args
    if args and args[0]=="--": args=args[1:]
    if not 15<=a.deadline<=10860: p.error("deadline outside bounded contract range")
    if "--output" not in args: p.error("training arguments must include --output")
    output=Path(args[args.index("--output")+1]).resolve()
    if output.exists(): p.error("output already exists")
    guard_path=output.parent/(output.name+".watchdog.json")
    guard_path.parent.mkdir(parents=True,exist_ok=True)
    with guard_path.open("x") as stream: stream.write('{"status":"starting"}\n')
    started=time.monotonic()
    process=subprocess.Popen(["rtk","proxy",str(ROOT/"run_training03.sh"),
                              str(ROOT/"tools/train_d1_ppo.py"),*args],start_new_session=True)
    status="finished"; error=None
    try:
        while process.poll() is None:
            if time.monotonic()-started>=a.deadline-10:
                status="deadline"; break
            time.sleep(.2)
    except BaseException as exc:
        status="interrupted"; error=f"{type(exc).__name__}: {exc}"
    finally:
        if process.poll() is None:
            try: os.killpg(process.pid,signal.SIGINT)
            except ProcessLookupError: pass
            try: process.wait(timeout=7)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(timeout=2)
        receipt={"status":status,"child_returncode":process.returncode,
                 "elapsed_wall_s":time.monotonic()-started,"deadline_s":a.deadline,"error":error,
                 "output":str(output)}
        guard_path.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)
    return 0 if status=="finished" and process.returncode==0 else 1

if __name__=="__main__": raise SystemExit(main())
