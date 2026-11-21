"""Fixed inputs and output identities of the D1 next-packet perception study."""
from tools91.launch91 import require
from tools94.common94 import ROOT, digest, read


PLAN = ROOT / "docs/experiments101/predictive_perception.json"
PLAN_SHA = "6499aedbc6e578a81fa95e3c51885a50f897f8ba00eab2010972df175803e20d"
BASE = ROOT / "evidence/jeplo101"
PUBLIC = ROOT / "reports/jeplo101"
MOVIE = ROOT / "docs/assets/jeplo101.mp4"
STAGES = ("prepare", "fit", "evaluate", "video")


def protocol():
    require(digest(PLAN) == PLAN_SHA, "perception registration changed")
    plan = read(PLAN)
    for name, expected in plan["parents_sha256"].items():
        require(digest(ROOT / name) == expected, f"registered parent changed: {name}")
    return plan


def producer(stage):
    names = {
        "prepare": ("data101.py",),
        "fit": ("fit101.py",),
        "evaluate": ("fit101.py", "score101.py"),
        "video": ("video101.py", "score101.py"),
    }
    files = ("common101.py", "launch101.py", *names[stage])
    return {f"tools101/{name}": digest(ROOT / "tools101" / name) for name in files}
