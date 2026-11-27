"""One continuous terrain course; geometry and ground truth are restricted to the simulator/scorer."""
from xml.etree import ElementTree as ET

import numpy as np

from tools110.course import X_GRID, Y_GRID, assess, scene


SEEDS = {"flat": (111011, 111023, 111037), "calibration": (111041, 111053, 111067),
         "development": (111101, 111103, 111107), "blind": (111201, 111203, 111207)}


def step_bounds(layout):
    if layout not in ("development", "blind"):
        raise ValueError("compound layout required")
    start = 5.12 + (.08 if layout == "blind" else 0.)
    return start, start+.76


def ground(layout, x, y, *, box=True):
    x, y = np.broadcast_arrays(x, y)
    if layout == "flat":
        return np.zeros_like(x, dtype=float)
    start, end = step_bounds(layout)
    ridge = np.tan(np.deg2rad(4.))*np.maximum(0., .75-np.abs(x-3.25))
    edge = np.clip(np.minimum(x-4., 5.-x)/.12, 0., 1.)
    envelope = .5-.5*np.cos(np.pi*edge)
    phase = 2.*np.pi*(x-4.)/.60 + (np.pi/2 if layout == "blind" else 0.)
    ripple = .0125*envelope*(1.+np.clip(y/.15, -1., 1.)*np.cos(phase))
    platform = .03*((x >= start) & (x <= end) & (np.abs(y) <= 1.4)) if box else 0.
    return ridge+ripple+platform


def make_scene(directory, layout):
    if layout == "flat":
        return scene(directory, "flat")
    target = scene(directory, "ridge")
    tree = ET.parse(target)
    start, end = step_bounds(layout)
    ET.SubElement(tree.find("worldbody"), "geom", name="platform111", type="box",
                  pos=f"{(start+end)/2} 0 .015", size=f"{(end-start)/2} 1.4 .015", rgba=".82 .48 .16 1")
    path = target.with_name("compound111.xml")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def install_heightfield(model, layout):
    if layout != "flat":
        z = ground(layout, X_GRID[None, :], Y_GRID[:, None], box=False)
        model.hfield_data[:] = ((z+.025)/.25).reshape(-1)


def calibration_gain(rows):
    scores = [r["motion"] for r in rows if r["arm"] == "heading_on"]
    if len(scores) != 3 or all(s["passed"] for s in scores):
        return None
    if any(s["speed_m_s"] is None or set(s["failed"])-{"one_metre_speeds", "forward_speed"} for s in scores):
        return None
    speeds = np.asarray([s["speed_m_s"] for s in scores])
    if not (all(speeds < 1.5375) or all(speeds > 1.5375)):
        return None
    if any(np.max(np.abs(np.asarray(s["sections_m_s"])/s["speed_m_s"]-1.)) > .05 for s in scores):
        return None
    from tools111.control import SPEED_GAIN
    gain = SPEED_GAIN*1.5375/float(speeds.mean())
    return gain if .8 <= gain <= 1.25 else None


def score(z, layout):
    motion = assess(z, "flat" if layout == "flat" else "step")
    effects = {}
    if layout != "flat":
        edge, _ = step_bounds(layout)
        arrived = (z["front_extent_x"][1:] >= edge) | z["front_platform_touch"]
        indices = np.flatnonzero(arrived)
        end = int(indices[0]) if len(indices) else 0
        start = max(0, end-25)
        k = np.arange(start, end)
        valid = k[z["capture_used"][k] >= 0]
        speed = z["base_cmd"] - z["actor_vx"]
        peak_speed = float(max(speed[valid], default=0.))
        peak_lift = float(z["applied_lift_fk"][valid, :2].max()) if len(valid) else 0.
        effects = dict(first_front_edge_tick=end if len(indices) else None,
                       speed_reduction_m_s=peak_speed, front_lift_m=peak_lift,
                       speed_active=peak_speed >= .020, lift_active=peak_lift >= .005)
    return motion, effects
