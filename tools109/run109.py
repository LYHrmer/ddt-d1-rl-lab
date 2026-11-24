"""Launch isolated 1.5 m/s flat-course stages and verify the native trajectory receipts."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from tools91.launch91 import PRESSURE, ROOT, THREADS, digest, read, require, write
from tools92.launch92 import scope_for


BASE = ROOT / "evidence/speed109"
PROPOSAL = ROOT / "docs/speed109_proposal.md"
PROPOSAL_SHA = "2ccd7127e006b6db1b0a3c85120b690aee3bc39df00427ab41b4e4ded54d3066"
GUARD_SHA = "236ef8f29a315435d4960cb95bc121851901e31e2a12475ee78035681628267d"
STAGES = ("actors", "reference", "blind", "video")
NATIVE = ("training/d1_rl/env.py", "training/d1_rl/interface.py", "training/d1_rl/task03d.py",
          "training/d1_rl/delay_env18.py", "training/d1_rl/fast_checks.py", "training/configs/pilot03d.json",
          "tools87/terrain87.py", "tools87/ppo87.py", "tools89/train89.py", "tools109/cruise109.py",
          "tools109/episodes109.py", "tools109/score109.py", "tools109/run109.py")


def paths(stage, attempt):
    require(stage in STAGES and 1 <= attempt <= 99, "invalid stage or attempt")
    name = f"{stage}_{attempt:02d}"
    return BASE/name, BASE/("guard_"+name), f"d1-speed109-{stage}-{attempt:02d}"


def sources():
    return {n: digest(ROOT/n) for n in NATIVE}


def completed(stage):
    from tools88.export88 import resources
    from tools109.cruise109 import BLIND, DEV

    candidates = []
    for file in sorted(BASE.glob(f"{stage}_??/report.json")):
        folder, guard, _ = paths(stage, int(file.parent.name[-2:]))
        if not (guard/"report.json").is_file() or read(guard/"report.json")["status"] != "COMPLETE":
            continue
        usage = resources(guard)
        result = read(file)
        require(result["status"] == "COMPLETE" and result["stage"] == stage
                and result["proposal_sha256"] == digest(PROPOSAL) == PROPOSAL_SHA and result["sources"] == sources(),
                f"stage binding differs for {folder}")
        if stage == "video":
            require(digest(folder/result["filename"]) == result["sha256"], "invalid video digest")
            candidates.append((folder, result, usage))
            continue
        seeds = BLIND if stage == "blind" else DEV
        expected = [(seed, arm) for seed in seeds for arm in result["arms"]]
        rows = result["episodes"]
        require([(r["seed"], r["arm"]) for r in rows] == expected, "incomplete episode matrix")
        for r in rows:
            ep = folder/f'{r["seed"]}_{r["arm"]}'
            require(r == read(ep/"report.json") and r["status"] == "COMPLETE"
                    and digest(ep/"trace.npz") == r["trace_sha256"], "bad native episode receipt")
        candidates.append((folder, result, usage))
    require(len(candidates) == 1, f"no unique complete {stage} result")
    return candidates[0]


def select_candidate():
    from tools109.cruise109 import DEV
    dev = [completed(s)[1] for s in ("actors", "reference")]
    rows = [row for part in dev for row in part["episodes"]]
    for seed in DEV:
        group = [r for r in rows if r["seed"] == seed]
        require(len(group) == 3 and len({r["initial_state_sha256"] for r in group}) == 1, "unpaired reset")
    for arm in ("actor24", "reference24"):
        group = [r for r in rows if r["arm"] == arm]
        if len(group) == len(DEV) and all(r["score"]["passed"] for r in group):
            return arm
    return None


def worker(stage, attempt):
    from tools27.resource_guard27 import read_memory
    from tools109.cruise109 import BLIND, DEV

    out, _, unit = paths(stage, attempt)
    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::"+scope and len(os.sched_getaffinity(0)) == 1
            and os.getpriority(os.PRIO_PROCESS, 0) == 10 and all(os.environ.get(k) == "1" for k in THREADS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and read_memory(scope)["effective_available_bytes"] >= 4<<30,
            "speed109 requires the admitted single-CPU unit")
    out.mkdir(exist_ok=False)
    try:
        require(digest(PROPOSAL) == PROPOSAL_SHA, "speed109 protocol changed")
        common = dict(stage=stage, proposal_sha256=PROPOSAL_SHA, sources=sources())
        write(out/"inputs.json", common)
        if stage == "video":
            from tools109.video109 import render
            result = render(out)
        else:
            from tools109.episodes109 import collect

            if stage == "actors":
                arms = ["actor5", "actor24"]
            elif stage == "reference":
                completed("actors")
                arms = ["reference24"]
            else:
                choice = select_candidate()
                require(choice is not None, "no qualifying development candidate for blind rollouts")
                arms = [choice]
            result = {"arms": arms, "episodes": collect(out, arms, BLIND if stage == "blind" else DEV)}
        write(out/"report.json", dict(status="COMPLETE", **common, **result))
    except BaseException as exc:
        write(out/"partial.json", {"status": "PARTIAL", "stage": stage, "error": repr(exc)})
        raise


def launch(stage, attempt):
    out, guard, unit = paths(stage, attempt)
    require(digest(PRESSURE) == GUARD_SHA and digest(PROPOSAL) == PROPOSAL_SHA, "supervisor or plan changed")
    BASE.mkdir(parents=True, exist_ok=True)
    with (ROOT/"evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not out.exists() and not guard.exists(), "preserve existing attempt")
        if stage != "video":
            require(not any(BASE.glob(f"{stage}_??/*/reservation.json")), "a native episode was already reserved")
        env = dict.fromkeys(THREADS, "1")
        env.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT/".python-deps", ROOT/"training", ROOT))),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="", MUJOCO_GL="disable",
                   MPLBACKEND="Agg")
        if stage == "video":
            from tools102.mesa_video102 import GL_ENV
            env.update(GL_ENV)
        command = ["systemd-run", "--user", "--collect", "--wait", "--pipe", "--slice=app.slice", f"--unit={unit}",
                   "-p", "Nice=10", "-p", "CPUWeight=25", "-p", "MemoryHigh=2G", "-p", "MemoryMax=3G",
                   "-p", "MemorySwapMax=0", "-p", "OOMPolicy=kill", f"--working-directory={ROOT}",
                   *(f"--setenv={k}={v}" for k, v in env.items()), sys.executable, "-B", str(PRESSURE),
                   "--scope", scope_for(unit), "--output", str(guard), "--timeout", "900",
                   "--", sys.executable, "-B", "-m", "tools109.run109", "worker", stage, "--attempt", str(attempt)]
        subprocess.run(command, check=True)
        require(read(guard/"report.json")["status"] == "COMPLETE" and (out/"report.json").is_file(),
                "stage is incomplete")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("launch", "worker"))
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--attempt", type=int, default=1)
    args = p.parse_args()
    {"launch": launch, "worker": worker}[args.mode](args.stage, args.attempt)
