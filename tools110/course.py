"""World geometry, command course and terrain-local qualification for speed110."""
from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from tools91.launch91 import ROOT, digest, read


TERRAINS = ("ridge", "ripple", "step")
SEEDS = {"flat": (110011, 110023, 110037), "calibration": (110041, 110053, 110067),
         "development": (110101, 110103, 110107), "blind": (110201, 110203, 110207)}
CFG = ROOT / "training/configs/pilot03d.json"
TICKS = 600
X_GRID = np.linspace(-5., 17., 551)
Y_GRID = np.linspace(-2.2, 2.2, 89)


def command(tick):
    if not 0 <= tick <= TICKS:
        raise ValueError("outside the speed110 clock")
    return float(np.interp(tick, [0, 50, 150, 450, 525, TICKS], [0., 0., 1.5, 1.5, 0., 0.]))


def height(terrain, x, y):
    x, y = np.broadcast_arrays(x, y)
    if terrain == "flat":
        return np.zeros_like(x, dtype=float)
    if terrain == "ridge":
        return math.tan(math.radians(4.))*np.maximum(0., 2.5-np.abs(x-4.5))
    if terrain == "step":
        return .03*((x >= 4.) & (x <= 5.5) & (np.abs(y) <= 1.4))
    if terrain != "ripple":
        raise ValueError(f"unknown terrain {terrain}")
    edge = np.clip(np.minimum(x-2., 7.-x)/.25, 0., 1.)
    envelope = .5-.5*np.cos(np.pi*edge)
    lanes = np.clip(y/.15, -1., 1.)
    phase = 2.*np.pi*(x-2.)/.65
    return .0125*envelope*(1.+lanes*np.cos(phase))


def scene(directory, terrain):
    cfg = read(CFG)
    original = ROOT/cfg["scene"]
    robot = original.parent/"robot.xml"
    if digest(original) != cfg["scene_sha256"] or digest(robot) != cfg["robot_xml_sha256"]:
        raise ValueError("frozen scene or D1 changed")
    if terrain == "flat":
        return original
    if terrain not in TERRAINS:
        raise ValueError("unregistered terrain")
    directory = Path(directory)
    dst = directory/"mujoco"
    dst.mkdir()
    (dst/"robot.xml").symlink_to(robot)
    (directory/"meshes").symlink_to(original.parent.parent/"meshes", target_is_directory=True)
    root = ET.parse(original)
    floor = root.find("./worldbody/geom[@name='floor']")
    if terrain == "step":
        ET.SubElement(root.find("worldbody"), "geom", name="step110", type="box", pos="4.75 0 0.015",
                      size="0.75 1.4 0.015", rgba=".82 .48 .16 1")
    else:
        ET.SubElement(root.find("asset"), "hfield", name="road110", nrow=str(len(Y_GRID)),
                      ncol=str(len(X_GRID)), size="11 2.2 .25 .05")
        del floor.attrib["size"]
        floor.set("type", "hfield")
        floor.set("hfield", "road110")
        floor.set("pos", "6 0 -0.025")
    p = dst/"scene110.xml"
    root.write(p, encoding="utf-8", xml_declaration=True)
    return p


def set_floor(model, terrain):
    if terrain in ("ridge", "ripple"):
        assert model.nhfield == 1
        z = height(terrain, X_GRID[None, :], Y_GRID[:, None])
        model.hfield_data[:] = ((z+.025)/.25).reshape(-1)


def measured_terrain(model, terrain):
    """Read actual intersections before trials, away from triangle or step edges."""
    import mujoco as mj
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    ids = np.empty(1, dtype=np.int32)
    measured = []
    # Robot sits near the origin, safely away from the middle-course probe points.
    for x in (2.18, 3.13, 4.33, 4.78, 5.69, 6.76, 7.10):
        for y in (-.29, .29):
            distance = mj.mj_ray(model, data, np.array([x, y, 1.]), np.array([0., 0., -1.]), None, True, -1, ids)
            h, predicted = 1.-distance, float(height(terrain, x, y))
            if distance < 0 or abs(h-predicted) > .0005:
                raise ValueError(f"terrain ray mismatch at {(x,y)}: {h} versus {predicted}")
            measured.append([x, y, h, predicted])
    return measured


def assess(z, terrain):
    t, xy, wheel = (np.asarray(z[k]) for k in ("time", "xy", "wheels_xy"))
    n = len(t)-1
    shapes = {"xy": (n+1, 2), "vxy": (n+1, 2), "wheels_xy": (n+1, 4, 2), "fallen": (n,),
              "illegal": (n,), "exit_contact": (n, 4), "top_contact": (n, 4)}
    if not 1 <= n <= TICKS or not np.allclose(t, np.arange(n+1)/50., atol=1e-9, rtol=0.):
        raise ValueError("invalid 50 Hz trajectory clock")
    for name, expected in shapes.items():
        if z[name].shape != expected or not np.isfinite(z[name]).all():
            raise ValueError(f"invalid {name}")
    gates = []
    for marker in (3., 4., 5., 6.):
        pair = np.flatnonzero((xy[:-1, 0] < marker) & (xy[1:, 0] >= marker))
        if not len(pair):
            break
        k = int(pair[0])
        gates.append(float(t[k]+(t[k+1]-t[k])*(marker-xy[k,0])/(xy[k+1,0]-xy[k,0])))
    sections, speed = [], None
    if len(gates) == 4 and np.all(np.diff(gates)>0):
        sections = (1./np.diff(gates)).tolist()
        speed = 3./(gates[-1]-gates[0])
    on_road = (wheel[:, :, 0] >= 2.) & (wheel[:, :, 0] <= 7.)
    lane = float(np.max(np.abs(wheel[:, :, 1][on_road]))) if np.any(on_road) else None
    stop = float(np.linalg.norm(z["vxy"][575:], axis=1).max()) if n == TICKS else None
    exits, tops = np.any(z["exit_contact"], axis=0), np.any(z["top_contact"], axis=0)
    checks = {"horizon": n == TICKS, "forward_speed": speed is not None and 1.50 <= speed <= 1.575,
              "one_metre_speeds": len(sections)==3 and all(1.425 <= v <= 1.575 for v in sections),
              "lane": lane is not None and lane <= 1., "all_wheels_exited": bool(exits.all() and np.all(wheel[-1,:,0]>7.)),
              "step_top_support": terrain != "step" or bool(tops.all()), "stopped": stop is not None and stop <= .05,
              "fall_free": not np.any(z["fallen"]), "legal_support": not np.any(z["illegal"])}
    return dict(passed=all(checks.values()), failed=[k for k, passed in checks.items() if not passed],
                ticks=n, gate_times_s=gates, speed_m_s=speed, sections_m_s=sections, max_lane_y_m=lane,
                exit_contact=exits.tolist(), step_top_contact=tops.tolist(), stop_tail_max_m_s=stop)


def next_gain(rows):
    if len(rows)!=3:
        raise ValueError("calibration needs three complete scores")
    s=[r["score"] for r in rows]
    if all(r["passed"] for r in s):
        return 1.
    if any(set(r["failed"])-{"forward_speed","one_metre_speeds"} or r["speed_m_s"] is None for r in s):
        return None
    speeds = np.array([r["speed_m_s"] for r in s])
    if not (np.all(speeds<1.5) or np.all(speeds>1.575)):
        return None
    if any(np.max(np.abs(np.array(r["sections_m_s"])/r["speed_m_s"]-1.))>.05 for r in s):
        return None
    gain = 1.5375/float(speeds.mean())
    return gain if .8 <= gain <= 1.25 else None
