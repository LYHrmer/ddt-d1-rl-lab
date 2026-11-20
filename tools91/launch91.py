"""Run the preregistered offline transfer test under D1's serial pressure supervisor."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/terrain_world91"
PLAN = ROOT / "docs/experiments91/terrain_world_model.json"
PLAN_SHA = "aedb3c121d026f18b7d8197f2bfda91f001197a875704717f7e4192f6d0aa929"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "LP_NUM_THREADS", "BLIS_NUM_THREADS")
STAGES = ("prepare", "archived", "fit", "candidate", "score", "display", "tests")
PRESSURE = ROOT.parent / "wheel-legged-control-lab/scripts/run_pressure83.py"
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, allow_nan=False, indent=2)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ensure_guard(stage):
    receipt = read(BASE / f"guard_{stage}_01/report.json")
    require(receipt["status"] == "COMPLETE" and receipt["owned_cgroup_children_empty"]
            and receipt["child_started"] and receipt["returncode"] == 0, f"incomplete {stage} guard")
    return receipt


def prepare(folder, plan):
    from tools91.data91 import fit_data, prepare_starts

    _, dataset = fit_data()
    require([row["trace_sha256"] for row in dataset["sources"]] == plan["fit"]["trace_sha256"]
            and dataset["transitions"] == plan["fit"]["usable_transitions"], "training inputs differ")
    starts = prepare_starts(folder / "starts")
    write(folder / "report.json", {"dataset": dataset, **starts, "protocol_sha256": PLAN_SHA})


def predict(stage, folder, plan):
    import torch
    from tools91.model91 import load_new_models, load_old_models
    from tools91.rollout91 import predict_suite

    ensure_guard("prepare")
    if stage == "candidate":
        ensure_guard("fit")
    starts = BASE / "prepare_01/starts"
    for source in read(starts.parent / "report.json")["starts"]:
        require(digest(starts / (source["case"] + ".npz")) == source["start_sha256"], "branch input changed")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    models = load_old_models() if stage == "archived" else load_new_models(BASE / "fit_01/models")
    expected = plan["old_models" if stage == "archived" else "fit"]["seeds"]
    require([s for s, _, _ in models] == expected, "incomplete model family")
    receipts = []
    for seed, model, model_receipt in models:
        case_folder = folder / f"seed{seed}"
        with torch.no_grad():
            suite = predict_suite(model, starts, case_folder)
        require(suite == read(case_folder / "predictions.json"), "inconsistent model predictions")
        receipts.append({"seed": seed, "model": model_receipt, "predictions_sha256": digest(case_folder / "predictions.json")})
    write(folder / "seal.json", {"protocol_sha256": PLAN_SHA, "family": stage, "models": receipts})


def worker(stage, unit):
    from tools27.resource_guard27 import read_memory

    scope = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}.service"
    require(Path("/proc/self/cgroup").read_text().strip() == "0::" + scope and
            len(os.sched_getaffinity(0)) == 1 and os.getpriority(os.PRIO_PROCESS, 0) == 10 and
            all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and
            read_memory(scope)["effective_available_bytes"] >= 4 << 30, "unit, affinity or memory admission mismatch")
    plan = read(PLAN)
    require(digest(PLAN) == PLAN_SHA and all(digest(ROOT / path) == h for path, h in plan["parents_sha256"].items()),
            "registered inputs changed")
    if stage == "tests":
        tests = sorted((ROOT / "tests").glob("test*91.py"))
        require(len(tests) == 4, "missing experiment tests")
        subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", *tests], check=True)
        subprocess.run([sys.executable, "-B", "-m", "tools91.check_ray91"], check=True)
        return
    require(unit == f"d1-terrain91-{stage}-01", "unregistered scientific attempt")
    folder = BASE / f"{stage}_01"
    folder.mkdir()
    if stage == "prepare":
        prepare(folder, plan)
    elif stage in ("archived", "candidate"):
        predict(stage, folder, plan)
    elif stage == "fit":
        from tools91.data91 import fit_data
        from tools91.model91 import fit_models

        ensure_guard("prepare")
        arrays, dataset = fit_data()
        require(dataset == read(BASE / "prepare_01/report.json")["dataset"], "training data changed")
        fit_models(arrays, folder / "models")
    elif stage == "score":
        from tools91.score91 import score_experiment

        write(folder / "report.json", score_experiment())
    elif stage == "display":
        ensure_guard("score")
        require(digest(ROOT / "reports/terrain_world91.json") == digest(BASE / "score_01/report.json"),
                "published scores differ from the guarded readback")
        subprocess.run([sys.executable, "-B", "-m", "tools91.kinematics91", folder / "kinematics.json"], check=True)
        subprocess.run([sys.executable, "-B", "-m", "tools91.plot91", folder / "terrain_world91.png"], check=True)


def launch(stage):
    tag = str(time.time_ns()) if stage == "tests" else "01"
    unit = f"d1-terrain91-{stage}-{tag}"
    guard = BASE / f"guard_{stage}_{tag}"
    if guard.exists() or (BASE / f"{stage}_{tag}").exists():
        raise FileExistsError(f"preserving the existing {stage} attempt")
    env = dict.fromkeys(THREADS, "1")
    env.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
               CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
               MUJOCO_GL="disable", MPLBACKEND="Agg", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    uid = os.getuid()
    scope = f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}.service"
    cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={unit}",
           "--property=Nice=10", "--property=MemoryHigh=2G", "--property=MemoryMax=3G", "--property=MemorySwapMax=0",
           "--property=OOMPolicy=kill", "--property=CPUWeight=25", f"--working-directory={ROOT}"]
    cmd += [f"--setenv={key}={value}" for key, value in env.items()]
    cmd += [sys.executable, "-B", str(PRESSURE), "--scope", scope, "--output", str(guard), "--timeout", "1200", "--",
            sys.executable, "-B", str(Path(__file__).resolve()), stage, "--worker-unit", unit]
    subprocess.run(cmd, check=True)
    receipt = read(guard / "report.json")
    require(receipt["status"] == "COMPLETE" and receipt["owned_cgroup_children_empty"], "invalid resource receipt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--worker-unit")
    args = parser.parse_args()
    if args.worker_unit:
        worker(args.stage, args.worker_unit)
    else:
        BASE.mkdir(exist_ok=True, parents=True)
        with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            launch(args.stage)
