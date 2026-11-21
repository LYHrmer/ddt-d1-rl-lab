"""Post-hoc diagnostics at native endpoints that actually exist, without extrapolation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools94.common94 import digest, read, tag, write
from tools94.score94 import agrees, return32


def run(report, curves):
    record = read(report)
    if digest(curves) != record["curves_sha256"]:
        raise ValueError("published endpoint archive changed")
    cells, display = [], []
    final_rollback = []
    native_lengths, model_lengths = [], []
    with np.load(curves, allow_pickle=False) as data:
        for cell in record["calibration"]:
            prefix = cell["endpoints_prefix"]
            d = cell["score"]["model_derivative"]
            pairs, motions = [], {}
            for radius in (10, 20):
                lo, hi = (data[prefix+"_native_"+tag(k)] for k in (-radius, radius))
                for key, x in ((-radius,lo), (radius,hi)):
                    native_lengths.append(len(x)-1)
                    model_lengths.append(len(data[prefix+"_model_"+tag(key)])-1)
                    final_rollback.append(1000.*max(0., x[0]-x[-1]))
                    motions[key] = x
                native = ((return32(hi)-return32(lo))/(2*radius/100.)) if min(len(hi),len(lo)) >= 33 else None
                comparable = d is not None and native is not None
                pairs.append(dict(radius_rad_s=radius/100., native_gradient=native,
                                  comparable=comparable, same_sign=(d*native>0 if comparable else None),
                                  agrees_25pct=(agrees(d,native) if comparable else None)))
            cells.append(dict(seed=cell["seed"], reset=cell["reset"], model_derivative=d, radii=pairs))
            display.append((motions[-20], motions[20]))
    tick = min(len(arm) for pair in display for arm in pair)-1
    drift, benefit = [], []
    for pair in display:
        left, right = (1000.*max(0., x[0]-x[tick]) for x in pair)
        drift.extend((left,right))
        benefit.append(left - right)
    radii = [r for cell in cells for r in cell["radii"]]
    return {"status": "POST_HOC", "curves_sha256": digest(curves), "cells": cells,
            "eligible_native_derivatives": sum(r["native_gradient"] is not None for r in radii),
            "comparable_derivatives": sum(r["comparable"] for r in radii),
            "same_sign": sum(r["same_sign"] is True for r in radii),
            "within_25pct": sum(r["agrees_25pct"] is True for r in radii),
            "native_ticks_range": [min(native_lengths), max(native_lengths)],
            "native_arms_reaching_50": native_lengths.count(50), "model_arms_reaching_50": model_lengths.count(50),
            "rollback_mm_at_each_native_stop": [min(final_rollback), max(final_rollback)],
            "video_last_common_tick": tick, "video_rollback_mm_range_at_common_tick": [min(drift), max(drift)],
            "video_positive_minus_negative_benefit_mm_range": [min(benefit), max(benefit)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("curves", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    write(args.out, run(args.report, args.curves))
