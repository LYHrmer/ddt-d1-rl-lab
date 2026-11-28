"""Remove the first parked right-front wheel contrast in the frozen round-114 native episode."""
from pathlib import Path
from types import FunctionType

import numpy as np

from tools91.launch91 import require
from tools94.common94 import write
from tools113.results import compare_tree
from tools114 import native as parking
from tools114.control import StopProjection
from tools115.native_compare import SEEDS, reference114, score_candidate
from tools115.signals import ISSUE, WHEELS


def release_rf_contrast(parked):
    issued = parked.copy()
    wheels = parked[WHEELS].astype(np.float64)
    mean = wheels.mean()
    delta = wheels[1]-mean
    wheels[[0,2,3]] += delta / 3.
    wheels[1] = mean
    issued[WHEELS] = wheels
    return issued


def _bind(function, **bindings):
    namespace = dict(function.__globals__, **bindings)
    return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


def with_rf_probe(drive, rows):
    class Probe(StopProjection):
        def apply(self, driver_command, targets):
            parked = super().apply(driver_command,targets)
            tick = len(rows["rf_contrast_probe"])
            pulse = tick == ISSUE
            if pulse:
                require(self.active and np.all(np.asarray(driver_command) == 0.), "pulse at an unexpected request")
            rows["parking_issued_before_probe"].append(parked.copy())
            rows["rf_contrast_probe"].append(pulse)
            return release_rf_contrast(parked) if pulse else parked

    return _bind(parking.with_parking, StopProjection=Probe)(drive,rows)


def episode(out, seed):
    out = Path(out)
    ref, source = reference114(seed)

    def matched_reference(requested_seed):
        require(requested_seed == seed, "native loop changed paired seed")
        return ref, source

    # All six starts are matched. The original episode checks qpos/qvel at reset
    # against this start's saved round-114 state, then runs the original servo loop.
    paired_episode = _bind(parking.episode, DEV=SEEDS, BLIND=(), baseline=matched_reference, with_parking=with_rf_probe)
    row = paired_episode(out,seed)
    try:
        result = score_candidate(out/"trace.npz",seed)
        require(row["nominal_base_height_m"] == source["nominal_base_height_m"]
                and result["counts"] == row["counts"] and result["trace_sha256"] == row["trace_sha256"]
                and row["reference_sha256"] == result["reference_sha256"], "native report and trajectory disagree")
        compare_tree(result["motion"],row["motion"])
        return dict(row,cohort="paired_reused_round114",intervention="release_rf_contrast_once",
                    legacy_reference113_sha256=source["reference_sha256"],assessment=result)
    except BaseException as exc:
        # A trace may already be on disk if the independent post-run audit fails.
        write(out/"partial_counts.json",dict(error=repr(exc),counts=row["counts"]))
        raise
