"""Dispatch still-unreserved development cases, one guarded physics worker at a time."""
import os
from pathlib import Path
import time

from tools91.launch91 import THREADS, read
from tools92.launch92 import scope_for


def ready():
    """Sample the host and the ancestors of the next unit. Its own guard makes the admission decision."""
    root = Path("/sys/fs/cgroup")
    ancestor = (root/scope_for("d1-lidar111-probe").lstrip("/")).parent
    available = next(int(x.split()[1])*1024 for x in Path("/proc/meminfo").read_text().splitlines()
                     if x.startswith("MemAvailable:"))
    pressure = [Path("/proc/pressure/memory")]
    while True:
        pressure.append(ancestor/"memory.pressure")
        if ancestor == root:
            break
        maximum = (ancestor/"memory.max").read_text().strip()
        if maximum != "max":
            available = min(available, int(maximum)-int((ancestor/"memory.current").read_text()))
        ancestor = ancestor.parent
    full = [float(next(s for s in p.read_text().splitlines() if s.startswith("full ")).split()[1].split("=")[1])
            for p in pressure]
    return available >= 4 << 30 and max(full) < 1.


def wait_for_admission():
    deadline = time.monotonic()+900
    quiet_since = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        quiet_since = (quiet_since or now) if ready() else None
        if quiet_since is not None and now-quiet_since >= 10:
            return
        time.sleep(1)
    raise TimeoutError("no sustained memory/pressure admission window for the next unstarted case")


def main():
    for key in THREADS:
        os.environ[key] = "1"
    from subprocess import CalledProcessError
    from tools111.course import SEEDS
    from tools111.individual import ARMS, BASE, launch, origin, paths
    origin()
    for seed in SEEDS["development"]:
        for arm in ARMS:
            name = f"{seed}_{arm}"
            if any(BASE.glob(f"*/{name}/reservation.json")):
                continue
            guards = sorted(BASE.glob("guard_individual_"+name+"_??"))
            for guard in guards:
                receipt = read(guard/"report.json")
                if receipt["child_started"] or receipt["status"] == "COMPLETE":
                    raise ValueError(f"investigate {guard.name} before dispatching another worker")
            attempt = 1+max((int(g.name[-2:]) for g in guards),default=0)
            while attempt <= 99:
                wait_for_admission()
                print(f"dispatching {name} attempt {attempt}",flush=True)
                try:
                    launch(seed,arm,attempt)
                except CalledProcessError:
                    out, guard, _ = paths(seed,arm,attempt)
                    receipt = read(guard/"report.json")
                    if (out/name/"reservation.json").exists():
                        print(f"retained interrupted reservation {name}",flush=True)
                        break
                    if receipt["child_started"] or not any(s in str(receipt["error"]) for s in
                            ("effective memory", "admission not sustained", "PSI")):
                        raise
                    attempt += 1
                    continue
                break
            else:
                raise RuntimeError(f"exhausted distinct admission directories for {name}")


if __name__ == "__main__":
    main()
