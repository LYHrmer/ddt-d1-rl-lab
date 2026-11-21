"""Locked inputs of the fixed-area wheel-track occlusion replay."""
from tools91.launch91 import require
from tools94.common94 import ROOT, digest, read


PLAN = ROOT / "docs/experiments102/occlusion_position.json"
PLAN_SHA = "ead58eb2e893a6c3bc58bbeba95b97612cd5c449c77c8de5810b29dd1dbcdec0"
BASE = ROOT / "evidence/occlusion102"
PUBLIC = ROOT / "reports/occlusion102"
MOVIE = ROOT / "docs/assets/occlusion102.mp4"
STAGES = ("replay", "video")


def protocol():
    require(digest(PLAN) == PLAN_SHA, "occlusion registration changed")
    plan = read(PLAN)
    for filename, expected in plan["parents_sha256"].items():
        require(digest(ROOT / filename) == expected, f"changed registered parent: {filename}")
    folder = ROOT / "reports/jeplo101/fit"
    fit = read(folder / "fit.json")
    require(fit["status"] == "COMPLETE" and fit["seeds"] == plan["seeds"], "incomplete round 101 fit")
    for seed in plan["seeds"]:
        for arm in plan["arms"]:
            ckpt = fit["per_seed"][str(seed)]["arms"][arm]
            require(ckpt["updates"] == 50 and digest(folder / ckpt["file"]) == ckpt["sha256"],
                    "unsealed or incomplete checkpoint")
    return plan


def sources(stage):
    files = ["common102.py", "launch102.py", "mask102.py"]
    files.append({"replay": "replay102.py", "video": "video102.py"}[stage])
    return {f"tools102/{f}": digest(ROOT / "tools102" / f) for f in files}
