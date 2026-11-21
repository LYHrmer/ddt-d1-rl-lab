"""Registration, input identities and paths for the paired D1 sensing replay."""
from tools91.launch91 import require
from tools94.common94 import ROOT, digest, read


PLAN = ROOT / "docs/experiments100/pointcloud_density.json"
PLAN_SHA = "d7a65f1184c5f205d1e87c50b9f91d51f272548d9aa41ba7b498f2d0589cf5ca"
BASE = ROOT / "evidence/pointcloud100"
PUBLIC_REPORT = ROOT / "reports/pointcloud100.json"
PUBLIC_MANIFEST = ROOT / "reports/pointcloud100_manifest.json"
PUBLIC_CLOUDS = ROOT / "reports/pointcloud100_clouds.npz"
PUBLIC_VIDEO = ROOT / "docs/assets/pointcloud100.mp4"
PUBLIC_MOVIE_RECEIPT = ROOT / "reports/pointcloud100_video.json"


def protocol(private=False):
    require(digest(PLAN) == PLAN_SHA, "point-cloud protocol has changed")
    plan = read(PLAN)
    for name, expected in plan["parents_sha256"].items():
        require(digest(ROOT / name) == expected, f"changed registered input {name}")
    if private:
        for source in plan["sources"]:
            path = ROOT / source["path"]
            receipt = read(path.with_name("metrics.json"))
            require(receipt["status"] == "COMPLETE" and receipt["seed"] == source["seed"]
                    and receipt["terrain"] == "flat" and receipt["arm"] == "range"
                    and receipt["ticks"] == 300 and receipt["full_horizon"]
                    and receipt["trace_sha256"] == source["sha256"] == digest(path),
                    f"incomplete or changed source {path}")
    return plan


def producer(stage):
    shared = ["tools100/common100.py", "tools100/launch100.py", "tools100/score100.py"]
    if stage == "video":
        shared += ["tools100/video100.py"]
    elif stage == "replay":
        shared += ["tools100/replay100.py", "tools100/sensor100.py", "tools100/readback100.py"]
    else:
        raise ValueError(f"unknown stage {stage}")
    return {name: digest(ROOT / name) for name in shared}
