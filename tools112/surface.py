"""Protocol and matched reference trajectories for the equal-height surface comparison."""
from tools91.launch91 import ROOT, digest, read, require
from tools111.course import SEEDS as COURSE_SEEDS
from tools111.run import sources as native_sources


SEEDS = COURSE_SEEDS["flat"]
PLAN = ROOT / "docs/speed_response112_proposal.md"
PLAN_SHA = "b2e8b95fab1bf2c58aaa15c80ce90ed1c0e5e12c86c361bbc4113abc9b5ada92"
REFERENCE = ROOT / "reports/lidar_compound111/native.json"
REFERENCE_SHA = "f0c78571880588577622f6130225ebbb59230dbd25dea9faa71bc0c0cacd3ea3"
BASE = ROOT / "evidence/surface112"
PUBLIC = ROOT / "reports/surface112"
PROBES = [(x,y) for x in (-.33, 1.63, 2.71, 3.47, 4.61, 5.37, 7.23, 10.13) for y in (-.26,.26)]


def bindings():
    require(digest(PLAN) == PLAN_SHA and digest(REFERENCE) == REFERENCE_SHA, "proposal or reference changed")
    original = native_sources()
    require(read(REFERENCE)["sources"] == original, "the physics or controller differs from the plane reference")
    return {**original, **{p:digest(ROOT/p) for p in
                           ("tools112/surface.py","tools112/surface_native.py","tools112/run_surface.py")}}


def reference(seed):
    require(seed in SEEDS and digest(REFERENCE) == REFERENCE_SHA, "unregistered reference seed or manifest")
    slot, = [r for r in read(REFERENCE)["cases"] if r["seed"] == seed and r["arm"] == "heading_on"]
    require(slot["case_status"] == slot["unit_status"] == "COMPLETE", "incomplete plane reference")
    path = REFERENCE.parent / slot["file"]
    require(digest(path) == slot["episode"]["trace_sha256"], "plane trace checksum mismatch")
    return path, slot["episode"]


def paired_scores(plane, field):
    if any(row.get(k) is None for row in (plane,field) for k in ("speed_m_s","stop_tail_max_m_s")):
        return dict(cruise_loss_m_s=None, stop_increase_m_s=None, cruise_supported=False, stop_supported=False)
    loss = plane["speed_m_s"] - field["speed_m_s"]
    stop = field["stop_tail_max_m_s"] - plane["stop_tail_max_m_s"]
    return dict(cruise_loss_m_s=loss, stop_increase_m_s=stop,
                cruise_supported=loss >= .08, stop_supported=stop >= .03)
