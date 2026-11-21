"""Physical quantities and preregistered gates, with world x in metres."""
from __future__ import annotations

import math

import numpy as np


OFFSETS = (-20, -10, 10, 20)  # centiradians/sec, exact dictionary and filename keys


def position(x, ticks):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) <= ticks or not np.isfinite(x[:ticks+1]).all():
        raise ValueError(f"missing or nonfinite position at or before tick {ticks}")
    return x[:ticks+1]


def return32(x):
    x = position(x, 32)
    return -float(np.abs(x[1:] - x[0]).mean())


def rollback_mm(x):
    x = position(x, 50)
    return 1000. * max(0., float(x[0] - x[-1]))


def agrees(a, b):
    return bool(math.isfinite(a) and math.isfinite(b) and a*b > 0
                and abs(a-b) <= .25 * max(abs(a), abs(b)))


def calibration(real, imagined, derivative, clean):
    """Score the complete four-arm cell. Preserve each separately testable gate."""
    arms = {}
    for key in OFFSETS:
        measured = real[key]
        predicted = imagined[key]
        a = rollback_mm(measured) if len(measured) >= 51 else None
        b = rollback_mm(predicted) if len(predicted) >= 51 else None
        arms[str(key)] = dict(real_ticks=len(measured)-1, model_ticks=len(predicted)-1,
                              complete_safe=bool(clean[key]), real_rollback_mm=a, model_rollback_mm=b,
                              model_error_mm=b-a if a is not None and b is not None else None)
    available = all(len(real[k]) >= 51 and len(imagined[k]) >= 51 for k in OFFSETS)
    if not available or derivative is None or not math.isfinite(derivative):
        return dict(passed=False, available=False, arms=arms, radii={}, model_derivative=derivative,
                    gates={"all_arms_complete_safe": all(clean.values()), "full_numeric_comparison": False})
    radii = {}
    for k in (10, 20):
        lo, hi = arms[str(-k)], arms[str(k)]
        native_delta = lo["real_rollback_mm"] - hi["real_rollback_mm"]
        model_delta = lo["model_rollback_mm"] - hi["model_rollback_mm"]
        slope = (return32(real[k])-return32(real[-k]))/(2*k/100.)
        radii[str(k)] = dict(native_gradient=slope, native_effect_mm=native_delta, model_effect_mm=model_delta,
                             signed_effect_error_mm=model_delta-native_delta, gradient_agrees=agrees(derivative, slope),
                             effect_sign=bool(native_delta * model_delta > 0),
                             effect_within_3mm=abs(native_delta-model_delta) <= 3.,
                             beats_zero=abs(native_delta-model_delta) < abs(native_delta))
    gates = {
        "all_arms_complete_safe": all(clean.values()),
        "native_gradient_stable": agrees(radii["10"]["native_gradient"], radii["20"]["native_gradient"]),
        "model_gradient_matches_native": all(row["gradient_agrees"] for row in radii.values()),
        "effect_sign": all(row["effect_sign"] for row in radii.values()),
        "effect_within_3mm": all(row["effect_within_3mm"] for row in radii.values()),
        "beats_zero": all(row["beats_zero"] for row in radii.values()),
        "all_arms_within_30mm": all(abs(row["model_error_mm"]) <= 30. for row in arms.values()),
    }
    return dict(passed=all(gates.values()), available=True, arms=arms, radii=radii,
                model_derivative=float(derivative), gates=gates)


def control(initial, final, clean):
    result = {"clean": bool(clean), "passed": False}
    for label, values in (("initial", initial), ("final", final)):
        if len(values) < 101:
            return result
        x = position(values, 100)
        result[label] = {
            "rollback_mm": rollback_mm(x),
            "maximum_absolute_drift_mm": 1000. * float(np.abs(x[:51]-x[0]).max()),
            "second_second_displacement_mm": 1000. * float(x[100]-x[50]),
        }
    a, b = result["initial"], result["final"]
    benefit = a["rollback_mm"] - b["rollback_mm"]
    result.update(rollback_reduction_mm=benefit, passed=bool(clean and benefit >= 30.
                  and b["maximum_absolute_drift_mm"] <= a["maximum_absolute_drift_mm"]))
    return result
