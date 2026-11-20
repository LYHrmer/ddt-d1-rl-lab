"""Reconstruct synchronized wheel/ground contact at archived D1-87 observation boundaries."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools87.readback87 import PROTOCOL, digest


AXLES = ("front", "rear")
POSITION_ATOL_M = 1e-6


@dataclass(frozen=True)
class LoadedPoint:
    x: float
    normal_force: float
    longitudinal_speed: float | None
    tangential_speed: float
    tangential_force: float
    sliding_friction_min: float
    contact_dimension: int


def tangent_speeds(velocity_world, normal):
    """Wheel material velocity along projected world +x and magnitude in the contact plane."""
    tangent_velocity = velocity_world - normal * np.dot(velocity_world, normal)
    forward = np.array([1., 0., 0.]) - normal * normal[0]
    norm = np.linalg.norm(forward)
    longitudinal = float(np.dot(tangent_velocity, forward) / norm) if norm >= 1e-8 else None
    return longitudinal, float(np.linalg.norm(tangent_velocity))


def window_summary(points_by_tick, axle_x, toe):
    points = [p for tick in points_by_tick for p in tick]
    n = len(points_by_tick)
    force = sum(p.normal_force for p in points)
    result = {
        "boundaries": n,
        "loaded_boundaries": sum(bool(tick) for tick in points_by_tick),
        "loaded_contact_samples": len(points),
        "mean_axle_normal_force_n": force / n if n else None,
        "axle_x_at_last_sample_m": float(axle_x[-1]) if n else None,
        "normal_load_fraction_past_toe": None,
        "longitudinal_direction_defined_load_fraction": None,
        "longitudinal_slip_signed_m_s": None,
        "longitudinal_slip_abs_m_s": None,
        "tangential_slip_m_s": None,
        "sum_tangential_force_magnitude_over_sum_normal_force": None,
        "min_configured_sliding_friction": None,
        "contact_dimensions": sorted({p.contact_dimension for p in points}),
    }
    if force:
        longitudinal = [p for p in points if p.longitudinal_speed is not None]
        longitudinal_load = sum(p.normal_force for p in longitudinal)
        result.update(
            normal_load_fraction_past_toe=sum(p.normal_force for p in points if p.x >= toe) / force,
            longitudinal_direction_defined_load_fraction=longitudinal_load / force,
            tangential_slip_m_s=sum(p.normal_force*p.tangential_speed for p in points) / force,
            sum_tangential_force_magnitude_over_sum_normal_force=sum(p.tangential_force for p in points) / force,
            min_configured_sliding_friction=min(p.sliding_friction_min for p in points),
        )
        if longitudinal_load:
            signed = sum(p.normal_force * p.longitudinal_speed for p in longitudinal)
            absolute = sum(p.normal_force * abs(p.longitudinal_speed) for p in longitudinal)
            result["longitudinal_slip_signed_m_s"] = signed / longitudinal_load
            result["longitudinal_slip_abs_m_s"] = absolute / longitudinal_load
    return result


def compare_positions(original, restored):
    original, restored = np.asarray(original), np.asarray(restored)
    if original.shape != restored.shape or np.isinf(original).any() or np.isinf(restored).any():
        raise ValueError("invalid contact-position series")
    a, b = np.isfinite(original), np.isfinite(restored)
    distance = np.abs(original[a & b] - restored[a & b])
    mismatches = int(np.count_nonzero(a != b))
    excess = int(np.count_nonzero(distance > POSITION_ATOL_M))
    return {
        "boundaries": len(original), "common_finite": len(distance),
        "missingness_mismatches": mismatches, "differences_over_tolerance": excess,
        "max_common_error_m": float(distance.max()) if len(distance) else None,
        "passed": mismatches == 0 and excess == 0,
    }


def geometry(model, mujoco):
    from training.d1_rl.env import FOOT_BODIES
    from training.d1_rl.interface import JOINT_NAMES

    by_geom = {}
    for name in FOOT_BODIES:
        body = model.body(name).id
        candidates = [
            g for g in range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
            if model.geom_contype[g] and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]
        if len(candidates) != 1:
            raise ValueError(f"expected one colliding wheel cylinder on {name}")
        by_geom[candidates[0]] = ("front" if name.startswith("F") else "rear", body)
    floor = model.geom("floor").id
    if model.geom_bodyid[floor] != 0:
        raise ValueError("terrain floor must be attached to the static world")
    dofs = [model.jnt_dofadr[model.joint(name).id] for name in JOINT_NAMES]
    return floor, by_geom, dofs


def analyze_trace(model, trace, case, course):
    import mujoco

    floor, wheel_geoms, dofs = geometry(model, mujoco)
    axles = {a: [g for g, (axle, _) in wheel_geoms.items() if axle == a] for a in AXLES}
    data = mujoco.MjData(model)
    jacobian, force = np.empty((3, model.nv)), np.empty(6)
    n = len(trace["time_s"]) - 1
    hz, toe = course["policy_hz"], course["toe_x_m"]
    if (not 0 < n <= course["ticks"] or n != case["ticks"]
            or bool(n == course["ticks"]) != case["full_horizon"]
            or not np.allclose(trace["time_s"], np.arange(n+1)/hz, atol=1e-8, rtol=0)):
        raise ValueError(f"{case['case']}: archive clock/termination mismatch")
    for key, width in (("qpos", model.nq), ("qvel", model.nv), ("last_torque_nm", len(dofs))):
        if trace[key].shape != (n+1, width) or not np.isfinite(trace[key]).all():
            raise ValueError(f"{case['case']}: invalid {key}")
    points = {a: [] for a in AXLES}
    centers = {a: [] for a in AXLES}
    farthest = {a: [] for a in AXLES}
    # Row zero is a synthetic score placeholder; 87 did not archive its front contact.
    for i in range(1, n+1):
        data.time = trace["time_s"][i]
        data.qpos[:] = trace["qpos"][i]
        data.qvel[:] = trace["qvel"][i]
        data.qfrc_applied[dofs] = trace["last_torque_nm"][i]
        data.qacc_warmstart[:] = 0.
        mujoco.mj_forward(model, data)
        current = {a: [] for a in AXLES}
        for j, contact in enumerate(data.contact[:data.ncon]):
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if floor not in (g1, g2):
                continue
            wheel = g2 if g1 == floor else g1
            if wheel not in wheel_geoms:
                continue
            mujoco.mj_contactForce(model, data, j, force)
            if force[0] <= 1.0:  # Same normal-load cutoff as D1Terrain87.front_contact_x_m.
                continue
            axle, body = wheel_geoms[wheel]
            mujoco.mj_jac(model, data, jacobian, None, contact.pos, body)
            longitudinal, tangential = tangent_speeds(jacobian @ data.qvel, contact.frame[:3])
            current[axle].append(LoadedPoint(
                float(contact.pos[0]), float(force[0]), longitudinal, tangential,
                float(np.linalg.norm(force[1:3])), float(min(contact.friction[:2])), int(contact.dim),
            ))
        for axle in AXLES:
            points[axle].append(current[axle])
            centers[axle].append(float(data.geom_xpos[axles[axle], 0].mean()))
            farthest[axle].append(max((p.x for p in current[axle]), default=np.nan))

    # Half-open on the left, closed on the right. A shortened episode never acquires missing ticks.
    windows = {}
    for label, end_s in (("seconds_2_to_3", 3), ("seconds_4_to_5", 5), ("last_second", 6)):
        sl = slice((end_s-1)*hz, end_s*hz)
        windows[label] = {a: window_summary(points[a][sl], centers[a][sl], toe) for a in AXLES}
    first, maximum = {}, {}
    for axle in AXLES:
        x = np.asarray(farthest[axle])
        crossed = np.flatnonzero(x >= toe)
        first[axle] = float(trace["time_s"][crossed[0]+1]) if len(crossed) else None
        maximum[axle] = float(np.nanmax(x)) if np.isfinite(x).any() else None
    return {
        **{k: case[k] for k in ("case", "arm", "seed", "training_seed", "terrain", "ticks", "full_horizon", "trace_sha256")},
        "terminated": case["terminated"], "first_loaded_contact_past_toe_s": first,
        "max_loaded_contact_x_m": maximum, "windows": windows,
        "validation": {
            "front_contact_x_m": compare_positions(trace["front_contact_x_m"][1:], farthest["front"]),
            "front_axle_x_m": compare_positions(trace["front_axle_x_m"][1:], centers["front"]),
        },
    }


def build(root):
    from tools87.terrain87 import load_model

    plan = json.loads(PROTOCOL.read_text())
    protocol_sha = digest(PROTOCOL)
    models = {}
    rows = []
    for split, folder, stage, expected in (
        ("development", "baseline_01", "baseline", plan["baseline"]["max_episodes"]),
        ("heldout", "evaluate_01", "evaluate", plan["evaluation"]["max_episodes"]),
    ):
        directory = root / folder
        report = json.loads((directory / "report.json").read_text())
        cases = report["cases"]
        if (report["stage"] != stage or report["status"] != "COMPLETE"
                or report["protocol_sha256"] != protocol_sha or len(cases) != expected
                or len({c["case"] for c in cases}) != expected):
            raise ValueError(f"{directory}: incomplete or inconsistent archive")
        for case in cases:
            path = directory / case["case"] / "trace.npz"
            if (case["status"] != "COMPLETE" or case["protocol_sha256"] != protocol_sha
                    or json.loads(path.with_name("metrics.json").read_text()) != case
                    or digest(path) != case["trace_sha256"]):
                raise ValueError(f"{path}: identity mismatch or incomplete source")
            terrain = case["terrain"]
            if terrain not in models:
                models[terrain] = load_model(terrain)
            with np.load(path, allow_pickle=False) as z:
                keys = ("qpos", "qvel", "time_s", "last_torque_nm", "front_contact_x_m", "front_axle_x_m")
                trace = {k: z[k] for k in keys}
            rows.append({"split": split, **analyze_trace(models[terrain], trace, case, plan["course"])})
    passed = all(v["passed"] for row in rows for v in row["validation"].values())
    return {
        "status": "COMPLETE", "protocol87_sha256": protocol_sha, "validation_passed": passed,
        "source_sha256": digest(Path(__file__)), "boundary_forward_calls": sum(row["ticks"] for row in rows),
        "integrated_physics_steps": 0, "training_transitions": 0,
        "position_validation_atol_m": POSITION_ATOL_M,
        "method": "Forward dynamics without integration at each archived 20 ms endpoint, using qpos, qvel, last applied PD torque and zero solver warm-start. Native substep forces and original warm starts were not archived.",
        "weighting": "Wheel material point velocity J(q,p) qvel relative to the fixed floor. Longitudinal axis is world +x projected on each contact plane. Speed means use reconstructed normal load above 1 N. Axle normal force sums all such loads per boundary, including zero for unloaded boundaries. Tangential force uses the sliding components only.",
        "windows": "(2,3], (4,5], (5,6] seconds from episode start. Empty/truncated windows retain only measured boundaries. Toe x=0.6 m is also used as a reference line on flat.",
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        try:
            report = build(args.root)
        except Exception as error:
            json.dump({"status": "INCOMPLETE", "error": str(error)}, stream, indent=2)
            raise
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"cases": len(report["cases"]), "validation_passed": report["validation_passed"]}))
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
