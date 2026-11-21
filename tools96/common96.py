"""Round-96 protocol, frozen inputs and immutable stage receipts."""
from tools94.common94 import ROOT, SEEDS, arrays, clocks, digest, read, write
from tools95.run95 import files_for_seal
from tools95.score95 import ARMS, HORIZON

BASE = ROOT / "evidence/paired_response96"
PLAN = ROOT / "docs/experiments96/paired_response.json"
PLAN_SHA = "9b7b33c88b8229b79d851acb12d8569e2a55f3b39d1c2cd1a358cb43892773dd"
ORIGINAL = ROOT / "evidence/policy_direction95/directions_01"
TRAIN_RESETS = (96101, 96102, 96103)
BLIND_RESETS = (96201, 96202, 96203)
MODELS = ("control", "paired")
STAGES = ("collection", "fit", "predictions", "native", "video")


def protocol():
    if digest(PLAN) != PLAN_SHA:
        raise ValueError("round-96 protocol changed")
    plan = read(PLAN)
    parents = ((ROOT / "docs/supermimic_paired_response96_proposal.md", plan["proposal_sha256"]),
               (ROOT / "reports/policy_direction95.json", plan["direction_report95_sha256"]),
               (ORIGINAL / "seal.json", plan["direction_seal95_sha256"]))
    if any(digest(path) != expected for path, expected in parents):
        raise ValueError("the proposal or frozen round-95 directions differ")
    if (tuple(plan["training_seeds"]) != SEEDS or tuple(plan["new_training_resets"]) != TRAIN_RESETS or
            tuple(plan["blind_resets"]) != BLIND_RESETS or plan["validation_multipliers"] != ARMS or
            tuple(plan["arms"]) != MODELS or plan["horizon_ticks"] != HORIZON):
        raise ValueError("runner indices do not match the protocol")
    return plan


def sources_for(stage):
    paths = ["tools96/common96.py", "tools96/launch96.py", "tools96/run96.py",
             "tools95/run95.py", "tools95/model95.py", "tools95/score95.py",
             "tools94/native94.py", "tools94/run94.py", "tools94/learn94.py", "tools91/model91.py"]
    if stage in ("fit", "predictions", "native"):
        paths.append("tools96/fit96.py")
    if stage in ("predictions", "native"):
        paths.append("tools96/score96.py")
    if stage == "video":
        paths.append("tools96/video96.py")
    return {path: digest(ROOT / path) for path in paths}


def checked(stage):
    folder = BASE / f"{stage}_01"
    seal = read(folder / "seal.json")
    if (seal["protocol_sha256"] != PLAN_SHA or seal["sources"] != sources_for(stage)
            or seal["files"] != files_for_seal(folder)):
        raise ValueError(f"{stage} inputs, code or sealed files changed")
    return folder


def direction(seed):
    from tools95.model95 import load_sealed

    parent = ORIGINAL / f"seed_{seed}"
    expected = read(ORIGINAL / "seal.json")["files"][f"seed_{seed}/receipt.json"]
    if digest(parent / "receipt.json") != expected:
        raise ValueError(f"round-95 direction receipt differs for {seed}")
    return load_sealed(seed, parent)
