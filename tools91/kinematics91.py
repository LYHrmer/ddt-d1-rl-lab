"""Check stand x against world vx from saved traces. Write a fresh public audit."""
import argparse
from pathlib import Path

import numpy as np

from tools91.launch91 import BASE, PLAN_SHA, ROOT, digest, read, require, write


RESETS = (90101, 90102, 90103)
ARMS = ("original", "common")


def _measure(xs, vxs):
    xs, vxs = (np.asarray(a, dtype=np.float64) for a in (xs, vxs))
    require(xs.shape == vxs.shape == (51,) and np.isfinite(xs).all() and np.isfinite(vxs).all(),
            "stand kinematics requires every endpoint 0..50 in float64")
    delta_x = float(xs[-1] - xs[0])
    integrated = float(.01 * np.sum(vxs[:-1] + vxs[1:], dtype=np.float64))
    return {"world_vx_m_s": vxs.tolist(), "world_dx_m": delta_x, "integrated_world_vx_m": integrated,
            "dx_minus_integral_m": delta_x - integrated,
            "rollback_from_x_m": max(0., -delta_x), "rollback_from_vx_m": max(0., -integrated)}


def _record(diagnostic, family, model_seed, cases):
    require(set(cases) == {(s, a) for s in RESETS for a in ARMS}, "incomplete stand arm inventory")
    for reset in RESETS:
        key = {"family": family, "model_seed": model_seed, "task": "stand", "reset_seed": reset}
        for arm in ARMS:
            diagnostic["arms"].append(dict(**key, arm=arm, **cases[reset, arm]))
        original, common = (cases[reset, arm] for arm in ARMS)
        x = original["rollback_from_x_m"] - common["rollback_from_x_m"]
        v = original["rollback_from_vx_m"] - common["rollback_from_vx_m"]
        diagnostic["pairs"].append(dict(**key, effect_x_m=x, effect_integrated_vx_m=v, difference_m=x - v))


def run():
    scoring = BASE / "score_01/report.json"
    report = read(scoring)
    require(report["protocol_sha256"] == PLAN_SHA, "missing registered experiment score")
    physical_summary = ROOT / "reports/stand_wbc90.json"
    require(digest(physical_summary) == report["truth_sha256"]["reports/stand_wbc90.json"], "90 report changed")
    measured = {}
    for case in read(physical_summary)["stand"]["cases"]:
        seed, arm = case["reset_seed"], case["arm"]
        trace = ROOT / f"evidence/wbc90/stand_01/{arm}_{seed}/trace.npz"
        require(case["status"] == "COMPLETE" and digest(trace) == case["trace_sha256"], "90 source changed")
        with np.load(trace, allow_pickle=False) as rows:
            require(np.allclose(rows["time_s"][:50], np.arange(1, 51) * .02, rtol=0., atol=1e-9),
                    "90 physical endpoints are off the registered 50 Hz grid")
            item = _measure(np.r_[rows["initial_qpos"][0], rows["qpos"][:50, 0]],
                            np.r_[rows["initial_qvel"][0], rows["qvel"][:50, 0]])
        require(abs(item["rollback_from_x_m"] - case["rollback_m"]) < 1e-12, "90 rollback differs")
        measured[seed, arm] = item
    diagnostic = {
        "design": "post hoc; existing physical and imagined stand traces",
        "quantity": "1 s world-x displacement and trapezoidal integral of world vx at 50 Hz",
        "integration_note": "Endpoint quadrature approximates the native 500 Hz displacement.",
        "report_sha256": digest(scoring), "curves_sha256": report["curves_sha256"],
        "prediction_seals": report["prediction_seals"], "policy_hz": 50,
        "endpoint_tick": list(range(51)), "arms": [], "pairs": [],
    }
    _record(diagnostic, "physics", None, measured)
    for family in ("archived", "candidate"):
        folder = BASE / f"{family}_01"
        seal = folder / "seal.json"
        require(digest(seal) == report["prediction_seals"][family], "prediction seal changed")
        for source in read(seal)["models"]:
            model_seed = source["seed"]
            base = folder / f"seed{model_seed}"
            manifest = base / "predictions.json"
            require(digest(manifest) == source["predictions_sha256"], "predictions changed")
            cases = {}
            for row in read(manifest)["cases"]:
                if row["task"] != "stand":
                    continue
                path = base / row["trace"]
                require(row["status"] == "COMPLETE" and digest(path) == row["trace_sha256"],
                        "incomplete or changed imagined stand")
                with np.load(path, allow_pickle=False) as trace:
                    require(np.array_equal(trace["endpoint_tick"], np.arange(51)), "missing imagined endpoints")
                    state = trace["state45"]
                    item = _measure(state[:, 0], state[:, 23])  # qpos world x and qvel world vx
                require(abs(item["rollback_from_x_m"] - row["score"]) < 1e-12, "sealed rollback differs")
                cases[row["reset_seed"], row["arm"]] = item
            _record(diagnostic, family, model_seed, cases)
    require(len(diagnostic["arms"]) == 42 and len(diagnostic["pairs"]) == 21, "incomplete stand comparison")
    return diagnostic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new JSON output file")
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError(output)
    write(output, run())
