"""Public synthetic wheel-bias regression. Run with python -m tools90.check_rolling_bias90."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tools90.rolling_bias90 import RollingBiasWBC90
from tools90.wbc90 import EstimatedState90, WBC90


FD_DT = 1e-6
MATCH_TOLERANCE_M_S2 = 1e-6
MINIMUM_SLIP_M_S = .05
MINIMUM_OLD_ERROR_M_S2 = .01
LIMBS = ("FL", "FR", "RL", "RR")


def fixture_model():
    """Primitive shapes only, with the D1 controller's free joint and motor-channel names."""
    xml = ET.Element("mujoco", model="rolling-bias-regression")
    world = ET.SubElement(xml, "worldbody")
    root = ET.SubElement(world, "body", name="base_link")
    ET.SubElement(root, "freejoint", name="floating_base_joint")
    ET.SubElement(root, "geom", type="box", size=".25 .12 .075", mass="10")
    actuator = ET.SubElement(xml, "actuator")
    for index, limb in enumerate(LIMBS):
        x, y = (.27 if index < 2 else -.27), (.16 if index % 2 == 0 else -.16)
        parent = root
        for segment, pos, axis in (("hip", f"{x} {y} 0", "1 0 0"),
                                   ("thigh", "0 0 -.04", "0 1 0"),
                                   ("calf", "0 0 -.16", "0 1 0"),
                                   ("foot", "0 0 -.17", "0 1 0")):
            parent = ET.SubElement(parent, "body", name=f"{limb}_{segment}", pos=pos)
            joint = f"{limb}_{segment}_joint"
            ET.SubElement(parent, "joint", type="hinge", name=joint, axis=axis)
            ET.SubElement(actuator, "motor", joint=joint)
            if segment == "foot":
                # A cylinder's local z axis is its axle; align that axis with the hinge's +y.
                ET.SubElement(parent, "geom", name=f"{limb}_wheel", type="cylinder",
                              size=".087 .020", euler="-90 0 0", mass=".45")
            else:
                ET.SubElement(parent, "geom", type="capsule", size=".018 .05", mass=".3")
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))


def _state(support, moving):
    quaternion = np.roll(Rotation.from_euler("xyz", [.195, -.12, .31]).as_quat(), 1)
    normals = np.array([[-.12, .025, 1.], [.075, -.08, 1.], [-.04, .10, 1.], [.02, -.05, 1.]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    q = np.array([[.17, .41, -.78, .31], [-.08, .55, -.72, -.48],
                  [.12, .36, -.66, -.21], [-.09, .62, -.84, .39]]).ravel()
    rates = np.array([[.31, -.57, .41, 3.2], [-.36, .46, -.58, -2.4],
                      [.40, .68, -.55, 1.7], [-.52, -.33, .67, -3.6]]).ravel()
    return EstimatedState90(quaternion, np.array([.04, -.03, .48]), moving*np.array([.37, -.19, .09]),
                            moving*np.array([.65, -.43, .32]), q, moving*rates,
                            np.asarray(support, dtype=bool), normals)


def _native_slip(model, data, active, normals):
    """Material slip from the cylinder center's spatial velocity and its finite rim geometry."""
    components, edges = [], []
    for i in active:
        geom = model.geom(f"{LIMBS[i]}_wheel").id
        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        normal = normals[i]
        along = np.cross(axis, normal)
        along /= np.linalg.norm(along)
        frame = np.stack((along, np.cross(normal, along), normal))
        edge = np.sign(axis @ normal)
        radial = normal - (axis @ normal)*axis
        radius, half_width = model.geom_size[geom, :2]
        arm = -radius*radial/np.linalg.norm(radial) - half_width*edge*axis
        twist = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom, twist, 0)
        components.append(frame @ (twist[3:] + np.cross(twist[:3], arm)))
        edges.append(edge)
    return np.asarray(components), np.asarray(edges)


def _finite_difference(model, q, v, qdd, state, dt):
    """Kinematic perturbation on the free-joint manifold. No applied force or time step."""
    active = np.flatnonzero(state.support)
    slips, edges = [], []
    for h in (-dt, 0., dt):
        trial = mujoco.MjData(model)
        trial.qpos[:] = q
        mujoco.mj_integratePos(model, trial.qpos, v + .5*h*qdd, h)
        trial.qvel[:] = v + h*qdd
        mujoco.mj_kinematics(model, trial)
        mujoco.mj_comPos(model, trial)
        mujoco.mj_comVel(model, trial)
        slip, edge = _native_slip(model, trial, active, state.normals_world)
        slips.append(slip)
        edges.append(edge)
    if not np.array_equal(edges[0], edges[1]) or not np.array_equal(edges[1], edges[2]) or not np.all(edges[1]):
        raise ValueError("synthetic finite difference switched or touched a cylinder edge branch")
    derivative = (slips[2] - slips[0]) / (2*dt)
    return derivative + 20.*slips[1], slips[1]


def _row_error(controller, state, qdd, expected):
    controller._reconstruct(state)
    contacts = controller._contacts(state.support, state.normals_world)
    active = np.flatnonzero(state.support)
    projected = np.stack([contacts.frame[i] @ (contacts.jac[i] @ qdd) - contacts.target[i] for i in active])
    if np.any(contacts.target[~state.support]) or np.any(contacts.jac[~state.support]):
        raise ValueError("inactive wheel produced a constraint")
    return np.max(np.abs(projected - expected), axis=1)


def compare():
    model = fixture_model()
    original, corrected = WBC90(model), RollingBiasWBC90(model)
    spec = (("single_front_sliding", [1, 0, 0, 0], True),
            ("diagonal_sliding", [0, 1, 1, 0], True),
            ("all_wheels_sliding", [1, 1, 1, 1], True),
            ("all_wheels_stationary", [1, 1, 1, 1], False))
    rows = []
    qdd = np.linspace(-1.1, .85, model.nv)
    for name, support, moving in spec:
        state = _state(support, moving)
        # Construct the independent coordinate vector by public joint names, independently
        # of the controller's nominal-state reconstruction and private index arrays.
        q, v = np.zeros(model.nq), np.zeros(model.nv)
        q[:3], q[3:7], v[:3], v[3:6] = (state.base_position_m, state.base_quat_wxyz,
                                       state.base_v_world_m_s, state.base_omega_body_rad_s)
        for i, limb in enumerate(LIMBS):
            for j, segment in enumerate(("hip", "thigh", "calf", "foot")):
                joint = model.joint(f"{limb}_{segment}_joint").id
                q[model.jnt_qposadr[joint]] = state.joint_q[4*i+j]
                v[model.jnt_dofadr[joint]] = state.joint_dq[4*i+j]
        for acceleration, trial_qdd in (("zero", np.zeros(model.nv)), ("nonzero", qdd)):
            expected, slip = _finite_difference(model, q, v, trial_qdd, state, FD_DT)
            half_step, _ = _finite_difference(model, q, v, trial_qdd, state, FD_DT/2)
            old = _row_error(original, state, trial_qdd, expected)
            new = _row_error(corrected, state, trial_qdd, expected)
            stability = float(np.max(np.abs(half_step-expected)))
            speeds = np.linalg.norm(slip, axis=1)
            each_old_fails = bool(np.min(old) > MINIMUM_OLD_ERROR_M_S2)
            each_new_passes = bool(np.max(new) <= MATCH_TOLERANCE_M_S2)
            source_check = (np.min(speeds) >= MINIMUM_SLIP_M_S and each_old_fails if moving else
                            np.max(speeds) == 0 and np.max(old) <= MATCH_TOLERANCE_M_S2)
            rows.append(dict(case=name, moving=bool(moving), generalized_acceleration=acceleration,
                             active_wheels=np.flatnonzero(state.support).tolist(), slip_speed_m_s=speeds.tolist(),
                             original_error_per_wheel_m_s2=old.tolist(), corrected_error_per_wheel_m_s2=new.tolist(),
                             fd_step_halving_max_difference_m_s2=stability,
                             passed=bool(source_check and each_new_passes and stability <= MATCH_TOLERANCE_M_S2)))
    return dict(passed=all(r["passed"] for r in rows), mujoco_version=mujoco.__version__, rows=rows,
                moving_wheel_cases=sum(len(r["active_wheels"]) for r in rows if r["moving"]),
                stationary_wheel_cases=sum(len(r["active_wheels"]) for r in rows if not r["moving"]),
                derivative_components=["rolling", "lateral", "normal"], finite_difference_dt_s=FD_DT,
                tolerance_m_s2=MATCH_TOLERANCE_M_S2, minimum_slip_m_s=MINIMUM_SLIP_M_S,
                minimum_detectable_old_error_m_s2=MINIMUM_OLD_ERROR_M_S2,
                original_max_error_m_s2=max(max(r["original_error_per_wheel_m_s2"]) for r in rows),
                corrected_max_error_m_s2=max(max(r["corrected_error_per_wheel_m_s2"]) for r in rows),
                archived_frames_read=0, qp_solves=0, native_steps=0, policy_updates=0)


def run(out: Path):
    """Persist an exclusive worker receipt and preserve a failure if the fixture fails."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "PARTIAL"}
    try:
        result.update(compare())
        if not result["passed"]:
            raise AssertionError("synthetic classical acceleration check failed")
        result["status"] = "COMPLETE"
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with (out / "report.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def test_native_wheel_slip_derivative():
    result = compare()
    print(f'Original max {result["original_max_error_m_s2"]:.8g} m/s², '
          f'corrected max {result["corrected_max_error_m_s2"]:.8g} m/s²')
    assert result["moving_wheel_cases"] == 14 and result["stationary_wheel_cases"] == 8
    assert result["passed"], result["rows"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write an exclusive report directory")
    args = parser.parse_args()
    result = run(args.output) if args.output else compare()
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
