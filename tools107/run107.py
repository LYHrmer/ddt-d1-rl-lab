"""Install the staged mount, conditionally acquire the paired scans, and verify published evidence."""
import argparse
import fcntl
from hashlib import sha1
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from tools91.launch91 import PRESSURE, THREADS, require
from tools92.launch92 import scope_for
from tools94.common94 import ROOT, digest, read, write
from tools107 import visibility107


BASE = ROOT / "evidence/mount107"
PUBLIC = visibility107.PUBLIC
MANIFEST = visibility107.PLAN.with_name("sources.sha256")
FILES = {
    "installation": ("geometry.json", "clearance_original.json", "clearance_forward.json", "clearance_samples.npz", "sightlines.json"),
    "capture": ("geometry.json", "capture.json", "clouds.npz", "predictions.npz", "optical.json"),
}


def registration():
    for line in MANIFEST.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        require(digest(ROOT / relative) == expected, f"registered source changed: {relative}")
    return read(visibility107.PLAN)


def verify_vendor_geometry():
    listed = read(visibility107.PLAN.with_name("vendor_geometry.json"))["git_blobs_in_local_d1_description"]
    root = ROOT / "src/ddt_ros2_control/urdfs/d1_description"
    for name, expected in listed.items():
        content = (root / name).read_bytes()
        blob = sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        require(blob == expected, f"vendor geometry changed: {name}")


def locations(stage, attempt):
    require(stage in FILES and 1 <= attempt <= 99, "unknown stage or attempt")
    name = f"{stage}_{attempt:02d}"
    return BASE / name, BASE / ("guard_" + name), f"d1-mount107-{stage}-{attempt:02d}"


def seal(folder, stage):
    return dict(status="COMPLETE", stage=stage, manifest_sha256=digest(MANIFEST),
                files={name: digest(folder / name) for name in FILES[stage]})


def validate_resources(folder):
    usage = read(folder / "resources.json")
    limits = read(visibility107.PLAN)["resources"]
    require(usage["startup_effective_bytes"] >= limits["admission_bytes"]
            and usage["minimum_effective_bytes"] >= limits["reserve_bytes"]
            and usage["owned_cgroup_children_empty"] and len(usage["cpu_affinity"]) == 1
            and all(usage[k] == v for k, v in limits.items()), "invalid resource receipt")


def approved_installation():
    path = PUBLIC / "installation"
    require(read(path / "seal.json") == seal(path, "installation"), "missing or damaged installation receipt")
    validate_resources(path)
    require(read(path / "geometry.json")["passed"] and read(path / "sightlines.json")["passed"],
            "the two prerequisites for acquisition did not pass")
    return path


def worker(stage, attempt):
    from tools27.resource_guard27 import read_memory
    from tools107 import capture107
    folder, _, unit = locations(stage, attempt)
    scope = scope_for(unit)
    registration()
    require(Path("/proc/self/cgroup").read_text().strip() == "0::" + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and read_memory(scope)["effective_available_bytes"] >= 4 << 30, "worker not in admitted serial unit")
    folder.mkdir(exist_ok=False)
    try:
        verify_vendor_geometry()
        if stage == "installation":
            capture107.prepare_geometry(folder)
            write(folder / "sightlines.json", visibility107.evaluate())
        else:
            source = approved_installation()
            shutil.copyfile(source / "geometry.json", folder / "geometry.json")
            capture107.capture_initial(folder)
            from tools107.score107 import score
            score(folder)
        write(folder / "seal.json", seal(folder, stage))
    except BaseException as exc:
        write(folder / "partial.json", dict(stage=stage, status="PARTIAL", error=repr(exc)))
        raise


def launch(stage, attempt):
    folder, guard, unit = locations(stage, attempt)
    limits = registration()["resources"]
    require(digest(PRESSURE) == limits["supervisor_sha256"], "pressure supervisor changed")
    BASE.mkdir(parents=True, exist_ok=True)
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not folder.exists() and not guard.exists(), "preserving an earlier attempt")
        if stage == "capture":
            approved_installation()
            require(not any(BASE.glob("capture_*/capture_started.json")), "the acquisition budget has already started")
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", MUJOCO_GL="disable",
                   PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))))
        command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={unit}",
                   "--property=Nice=10", "--property=CPUWeight=25", "--property=OOMPolicy=kill",
                   "--property=MemoryHigh=2147483648", "--property=MemoryMax=3221225472", "--property=MemorySwapMax=0",
                   f"--working-directory={ROOT}", *(f"--setenv={k}={v}" for k, v in env.items()),
                   sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard),
                   "--timeout", "600", "--", sys.executable, "-B", "-m", "tools107.run107", "worker", stage,
                   "--attempt", str(attempt)]
        subprocess.run(command, check=True)


def publish(stage, attempt):
    from tools88.export88 import resources
    plan = registration()
    folder, guard, unit = locations(stage, attempt)
    require(read(folder / "seal.json") == seal(folder, stage), "only complete sealed stages can be published")
    usage = resources(guard)
    receipt = read(guard / "report.json")
    require(receipt["scope"] == scope_for(unit) and usage["minimum_effective_bytes"] >= 2 << 30, "invalid resource log")
    destination = PUBLIC / stage
    PUBLIC.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), "preserving existing publication")
    with TemporaryDirectory(dir=PUBLIC, prefix=".staging-") as temp:
        safe = Path(temp) / "complete"
        safe.mkdir()
        for f in (*FILES[stage], "seal.json"):
            shutil.copyfile(folder / f, safe / f)
        require(read(safe / "seal.json") == seal(safe, stage), "incomplete transfer")
        write(safe / "resources.json", {**usage, **plan["resources"], "unit":unit, "guard_sha256":digest(guard / "report.json")})
        validate_resources(safe)
        safe.rename(destination)


def check():
    from tools107.score107 import score
    registration()
    install = PUBLIC / "installation"
    require(read(install / "seal.json") == seal(install, "installation"), "invalid installation seal")
    validate_resources(install)
    geo, visibility = read(install / "geometry.json"), read(install / "sightlines.json")
    require(visibility107.evaluate() == visibility and geo["checked_arm_poses"] == 918
            and geo["optical_calls"] == geo["physics_steps"] == 0, "invalid initial stages")
    from tools107.capture107 import check_geometry
    check_geometry(install)
    actual = PUBLIC / "capture"
    if not (geo["passed"] and visibility["passed"]):
        require(not actual.exists(), "acquisition despite a failed prerequisite")
        return
    require(read(actual / "seal.json") == seal(actual, "capture")
            and digest(install / "geometry.json") == digest(actual / "geometry.json"), "capture not bound to admission")
    validate_resources(actual)
    entry = read(actual / "capture.json")
    require(entry["status"] == "COMPLETE" and entry["physical_clouds"] == 216 and entry["first_return_beams"] == 112320
            and entry["native_ray_calls"] == 224640 and entry["physics_steps"] == entry["native_decisions"] == 0
            and entry["model_updates"] == entry["policy_updates"] == 0 and entry["cloud_sha256"] == digest(actual / "clouds.npz")
            and len(entry["native_packets_sha256"]) == 216, "incomplete registered acquisition")
    score(actual, verify=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "worker", "publish", "check"))
    parser.add_argument("stage", choices=tuple(FILES), nargs="?")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.action == "check":
        check()
    elif args.stage is None:
        parser.error("a stage is required")
    else:
        {"launch": launch, "worker": worker, "publish": publish}[args.action](args.stage, args.attempt)
