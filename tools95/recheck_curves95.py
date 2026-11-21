"""Recompute wheel-direction agreement from the frozen round-94 trajectories."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = "reports/cold_start94.json"
CURVES = "reports/cold_start94_curves.npz"
OUTPUT = ROOT / "reports/cold_start95_wheel_direction.json"
SHA256 = {
    RUN: "a6e770eec5cda66cbc9614182ee19afaa02b1e89c08f1797df3ddecb92108872",
    CURVES: "a91a35b6ff66b3ed4b95699b90b1d8ea8eaec3f768a00eed2124c9d0580080eb",
}
RADII = (10, 20)  # Hundredths of a radian per second.


def archived_bytes(name):
    data = (ROOT / name).read_bytes()
    if sha256(data).hexdigest() != SHA256[name]:
        raise ValueError(f"Frozen source checksum mismatch for {name}")
    return data


def agreement(first, second):
    if first is None or second is None:
        return None
    scale = max(abs(first), abs(second))
    relative_error = abs(first - second) / scale if scale else None
    same_sign = first * second > 0
    return {
        "same_sign": same_sign,
        "within_25pct": same_sign and relative_error <= 0.25,
        "relative_error": relative_error,
    }


def tally(comparisons, include_error=False):
    available = [c for c in comparisons if c is not None]
    summary = {
        "comparable": len(available),
        "same_sign": sum(c["same_sign"] for c in available),
        "within_25pct": sum(c["within_25pct"] for c in available),
    }
    if include_error:
        errors = [c["relative_error"] for c in available if c["relative_error"] is not None]
        summary["max_relative_error"] = max(errors, default=None)
    return summary


def central_difference(negative, positive, origin, horizon, radius):
    if min(len(negative), len(positive)) <= horizon:
        return None, None
    returns = [
        -math.fsum(abs(x - origin) for x in curve[1:horizon + 1]) / horizon
        for curve in (negative, positive)
    ]
    difference = returns[1] - returns[0]
    return difference / (2 * radius), difference


def load_cells(calibration, archive):
    cells = []
    for cell in calibration:
        prefix = cell["endpoints_prefix"]
        tracks = {}
        native_origins = set()
        for source in ("native", "model"):
            for offset in (-20, -10, 10, 20):
                tag = ("m" if offset < 0 else "p") + str(abs(offset))
                array = archive[f"{prefix}_{source}_{tag}"]
                if array.ndim != 1 or array.size < 2:
                    raise ValueError(f"Invalid x trajectory at {prefix}/{source}/{tag}")
                values = array.tolist()  # Promote archived float32 samples before subtraction.
                if not all(math.isfinite(x) for x in values):
                    raise ValueError(f"Nonfinite x at {prefix}/{source}/{tag}")
                tracks[source, offset] = values
                if source == "native":
                    native_origins.add(values[0])
        if len(native_origins) != 1:
            raise ValueError(f"Unpaired native reset x in {prefix}")
        cells.append((cell, tracks, native_origins.pop()))
    return cells


def rows_at(cells, horizon, primary):
    rows = []
    for cell, tracks, origin in cells:
        for offset in RADII:
            radius = offset / 100
            row = {"seed": cell["seed"], "reset": cell["reset"], "radius_rad_s": radius}
            for source in ("native", "model"):
                slope, delta = central_difference(
                    tracks[source, -offset], tracks[source, offset], origin, horizon, radius
                )
                row[source] = slope
                if horizon == primary:
                    row[f"{source}_positive_minus_negative_return_mm"] = 1000 * delta
            row["agreement"] = agreement(row["model"], row["native"])
            if horizon == 32:
                row["autograd"] = cell["score"]["model_derivative"]
            rows.append(row)
    return rows


def summarize(rows):
    summary = {"model_vs_native": tally(r["agreement"] for r in rows)}
    for field in ("seed", "reset", "radius_rad_s"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row["agreement"])
        summary["by_" + field] = {str(value): tally(group) for value, group in groups.items()}

    cells = defaultdict(list)
    for row in rows:
        cells[row["seed"], row["reset"]].append(row)
    radius_agreement = {}
    for source in ("native", "model"):
        radius_agreement[source] = tally(agreement(a[source], b[source]) for a, b in cells.values())
        summary[source + "_positive"] = sum(r[source] is not None and r[source] > 0 for r in rows)
    paired = [(a["agreement"], b["agreement"]) for a, b in cells.values()
              if a["agreement"] is not None and b["agreement"] is not None]
    radius_agreement["model_vs_native_at_both_radii"] = {
        "comparable_cells": len(paired),
        "same_sign_at_both": sum(a["same_sign"] and b["same_sign"] for a, b in paired),
        "within_25pct_at_both": sum(a["within_25pct"] and b["within_25pct"] for a, b in paired),
    }
    summary["between_radii"] = radius_agreement
    return summary


def triangle(rows, cells):
    complete = [r for r in rows if None not in (r["autograd"], r["model"], r["native"])]
    result = {}
    for label, group in (("all_available", rows), ("all_three_available", complete)):
        result[label] = {}
        for a, b in (("autograd", "model"), ("autograd", "native"), ("model", "native")):
            result[label][a + "_vs_" + b] = tally((agreement(r[a], r[b]) for r in group), True)

    omissions = []
    for cell, tracks, _ in cells:
        for offset in RADII:
            reasons = []
            for source in ("native", "model"):
                for arm in (-offset, offset):
                    last_tick = len(tracks[source, arm]) - 1
                    if last_tick < 32:
                        reason = {"source": source, "offset_rad_s": arm / 100, "last_tick": last_tick}
                        if source == "native":
                            reason["terminal_events"] = cell["native_events"][str(arm)]
                        reasons.append(reason)
            if cell["score"]["model_derivative"] is None:
                reasons.append({"source": "autograd", "nominal_model_exception": cell["model_exceptions"]["0"]})
            if reasons:
                omissions.append({"seed": cell["seed"], "reset": cell["reset"], "radius_rad_s": offset / 100,
                                  "missing": reasons})
    result["omitted_from_all_three"] = omissions
    return result


def build_report():
    import numpy as np

    run = json.loads(archived_bytes(RUN))
    if run["curves_sha256"] != SHA256[CURVES]:
        raise ValueError("Run record does not identify the frozen endpoint archive")
    with np.load(BytesIO(archived_bytes(CURVES)), allow_pickle=False) as archive:
        cells = load_cells(run["calibration"], archive)
    terminal_ticks = Counter()
    for cell, tracks, _ in cells:
        for arm, outcome in cell["native_events"].items():
            if not (outcome["terminated"] and outcome["illegal"]):
                raise ValueError("Unexpected native calibration termination in frozen run")
            terminal_ticks[len(tracks["native", int(arm)]) - 1] += 1
    primary = min(terminal_ticks) - 1
    horizons = {}
    for horizon in sorted({8, 16, 24, primary, 31, 32}):
        rows = rows_at(cells, horizon, primary)
        horizons[str(horizon)] = {"seconds": horizon / 50, "summary": summarize(rows), "pairs": rows}

    main_rows = horizons[str(primary)]["pairs"]
    delta_return = [r["native_positive_minus_negative_return_mm"] for r in main_rows]
    return {
        "status": "post_hoc_archive_recheck",
        "source_sha256": SHA256,
        "round94_protocol_sha256": run["protocol_sha256"],
        "new_simulation_steps": 0, "new_model_evaluations": 0, "training_updates": 0,
        "definitions": {
            "action": "One common offset to the four wheel targets in each closed-loop rollout (rad/s).",
            "return_m": "R_H = -mean(abs(x[t] - native_x_at_reset)), t=1..H; the same origin for both sources.",
            "native_and_model": "Central differences (R_H(+radius) - R_H(-radius)) / (2*radius), in m/(rad/s).",
            "autograd": "Archived zero-offset H32 model derivative of the return in m/(rad/s).",
            "within_25pct": "Nonzero and same sign, and |a-b| <= 0.25*max(|a|,|b|).",
            "primary_horizon": "Last common 20 ms endpoint strictly before the earliest native termination.",
            "return_difference_mm": "Positive minus negative return; time-averaged absolute drift improvement in mm.",
            "dependence": "Each seed has its own actor and model. The two radii reuse a cell, the three resets recur "
                          "for all seeds, and all horizons reuse trajectory prefixes. Only the common wheel direction is probed.",
        },
        "native_x_at_reset_m": [{"seed": c["seed"], "reset": c["reset"], "x": x} for c, _, x in cells],
        "native_illegal_terminal_tick_counts": dict(sorted(terminal_ticks.items())),
        "primary_horizon": primary,
        "primary_native_return_difference_mm_range": [min(delta_return), max(delta_return)],
        "horizons": horizons,
        "h32_triangle": triangle(horizons["32"]["pairs"], cells),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="Recompute and compare the existing report byte for byte")
    args = parser.parse_args()
    payload = (json.dumps(build_report(), indent=2, allow_nan=False) + "\n").encode()
    if args.check:
        if args.output.read_bytes() != payload:
            raise SystemExit(f"Recomputed report differs from {args.output}")
        print(f"Verified {args.output}")
    else:
        with args.output.open("xb") as output:
            output.write(payload)
        print(f"Created {args.output}")


if __name__ == "__main__":
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[variable] = "1"
    main()
