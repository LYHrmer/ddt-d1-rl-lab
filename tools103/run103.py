"""Serial, resource-guarded stages of the recorded D1 action-window comparison."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from tools91.launch91 import PRESSURE, THREADS, require
from tools92.launch92 import scope_for
from tools94.common94 import ROOT, digest, read, write
from tools95.run95 import files_for_seal
from tools101.common101 import PUBLIC as PARENT


PLAN = ROOT / "docs/experiments103/action_window.json"
PLAN_SHA = "23eb7acbe80c21c6563be318c88a1a4fc636f97231e66367f6598c14fbfc4836"
BASE = ROOT / "evidence/action_window103"
PUBLIC = ROOT / "reports/action_window103"
STAGES = ("prepare", "fit", "evaluate")
DEPENDENCIES = {"prepare": (), "fit": ("prepare",), "evaluate": ("prepare", "fit")}
MODULES = {"prepare": ("data103",), "fit": ("data103", "fit103"), "evaluate": ("data103", "fit103", "score103")}


def protocol():
    require(digest(PLAN) == PLAN_SHA, "action-window registration changed")
    plan = read(PLAN)
    for relative, expected in plan["parents_sha256"].items():
        require(digest(ROOT / relative) == expected, f"registered input changed: {relative}")
    return plan


def folder(stage):
    return BASE / f"{stage}_01"


def guard(stage):
    return BASE / f"guard_{stage}_01"


def unit(stage):
    return f"d1-action103-{stage}-01"


def sources(stage):
    return {f"tools103/{name}.py": digest(ROOT / f"tools103/{name}.py") for name in ("run103", *MODULES[stage])}


def completed(stage):
    from tools88.export88 import resources

    output, supervisor = folder(stage), guard(stage)
    seal, receipt = read(output / "seal.json"), read(supervisor / "report.json")
    require(seal["status"] == "COMPLETE" and seal["stage"] == stage and seal["unit"] == unit(stage)
            and seal["protocol_sha256"] == PLAN_SHA and seal["files"] == files_for_seal(output)
            and seal["sources"] == sources(stage), f"incomplete or changed {stage} output")
    require(seal["parents"] == {p: digest(folder(p) / "seal.json") for p in DEPENDENCIES[stage]},
            f"{stage} bound to different predecessor")
    limits = read(PLAN)["resources"]
    usage = resources(supervisor)
    require(receipt["scope"] == scope_for(unit(stage)) and receipt["child_started"]
            and usage["supervisor_sha256"] == limits["supervisor_sha256"], "unexpected resource supervisor")
    return output, dict(**usage, scope=receipt["scope"], receipt_sha256=digest(supervisor / "report.json"),
                        samples_sha256=digest(supervisor / "resources.jsonl"))


def execute(stage, output, plan):
    if stage == "prepare":
        from tools103.data103 import build_archive, window_inputs

        audits = {}
        for split in ("train", "holdout"):
            tape = output / f"{split}_actions.npz"
            audit = build_archive(split, tape)
            windows = window_inputs(split, PARENT / f"{split}.npz", tape)
            size = plan["data"][split]
            require((audit["episodes"], audit["native_intervals"], audit["native_transition_ticks_per_clock"],
                     audit["packet_pairs"]) ==
                    (size["native_episodes"], size["native_intervals"], size["native_action_ticks_per_clock"],
                     size["pairs"]) and windows.shape == (size["rows"], 2, 2, 9, 17)
                    and audit["actions_sha256"] == plan["data"]["source_action_bytes_sha256"][split],
                    f"registered {split} clocks or issued actions differ")
            audits[split] = audit
        write(output / "audit.json", dict(status="COMPLETE", protocol_sha256=PLAN_SHA, splits=audits,
                                          head_updates=0, policy_updates=0, physics_substeps=0))
        return

    import torch
    from tools103.fit103 import fit, predict

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    write(output / "runtime.json", dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads(),
                                        interop_threads=torch.get_num_interop_threads(),
                                        deterministic=torch.are_deterministic_algorithms_enabled()))
    data = folder("prepare")
    if stage == "fit":
        models = output / "models"
        models.mkdir()
        fit(PARENT / "train.npz", data / "train_actions.npz", models)
        result = read(models / "fit.json")
        require(result["status"] == "COMPLETE" and result["updates"] == plan["fit"]["total_head_updates"],
                "incomplete predictor fitting")
        return

    from tools103.score103 import score

    models = folder("fit") / "models"
    packets, tape, forecasts = PARENT / "holdout.npz", data / "holdout_actions.npz", output / "forecast.npz"
    predict(packets, tape, models, forecasts)
    write(output / "report.json", dict(status="COMPLETE", protocol_sha256=PLAN_SHA,
                                        fit_sha256=digest(models / "fit.json"), holdout_sha256=digest(packets),
                                        tape_sha256=digest(tape), forecast_sha256=digest(forecasts),
                                        counts=dict(head_updates=plan["fit"]["total_head_updates"], policy_updates=0,
                                                    encoder_updates=0, dynamics_updates=0, native_decisions=0,
                                                    physics_substeps=0, new_sensor_rays=0),
                                        score=score(packets, forecasts)))


def worker(stage, name):
    from tools27.resource_guard27 import read_memory

    plan = protocol()
    limits = plan["resources"]
    scope = scope_for(name)
    require(name == unit(stage) and Path("/proc/self/cgroup").read_text().strip() == "0::" + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == limits["nice"] and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and read_memory(scope)["effective_available_bytes"] >= limits["admission_min_bytes"],
            "worker must run in the admitted single-core unit")
    output = folder(stage)
    output.mkdir()
    start = dict(stage=stage, unit=name, protocol_sha256=PLAN_SHA, sources=sources(stage),
                 parents={p: digest(folder(p) / "seal.json") for p in DEPENDENCIES[stage]})
    write(output / "start.json", start)
    try:
        execute(stage, output, plan)
        require(sources(stage) == start["sources"], "producer changed while the unit was running")
        write(output / "seal.json", dict(status="COMPLETE", files=files_for_seal(output), **start))
    except BaseException as error:
        write(output / "partial.json", dict(status="PARTIAL", error=repr(error), **start))
        raise


def launch(stage):
    limits = protocol()["resources"]
    require(digest(PRESSURE) == limits["supervisor_sha256"], "unexpected pressure supervisor")
    BASE.mkdir(parents=True, exist_ok=True)
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if folder(stage).exists() or guard(stage).exists():
            raise FileExistsError(f"preserving previous {stage} attempt")
        for parent in DEPENDENCIES[stage]:
            completed(parent)
        sources(stage)
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", MUJOCO_GL="disable",
                   PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))))
        name = unit(stage)
        cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={name}",
               "--property=CPUWeight=25", "--property=OOMPolicy=kill", f"--property=Nice={limits['nice']}",
               f"--property=MemoryHigh={limits['memory_high_bytes']}", f"--property=MemoryMax={limits['memory_max_bytes']}",
               f"--property=MemorySwapMax={limits['swap_max_bytes']}", f"--working-directory={ROOT}"]
        cmd.extend(f"--setenv={key}={value}" for key, value in env.items())
        cmd.extend([sys.executable, "-B", str(PRESSURE), "--scope", scope_for(name), "--output", str(guard(stage)),
                    "--timeout", "1200", "--", sys.executable, "-B", "-m", "tools103.run103", stage,
                    "--worker-unit", name])
        subprocess.run(cmd, check=True)
        completed(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--worker-unit")
    args = parser.parse_args()
    if args.worker_unit:
        worker(args.stage, args.worker_unit)
    else:
        launch(args.stage)
