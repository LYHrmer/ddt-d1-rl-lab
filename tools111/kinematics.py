"""Independent fixed-base leg geometry from the hash-checked D1 MJCF, without loading MuJoCo."""
import argparse
import json
from xml.etree import ElementTree as ET

import numpy as np

from tools91.launch91 import ROOT, digest, read, write


def vendor_targets(actions):
    pose = np.array([.1,.8,-1.5,0.,-.1,.8,-1.5,0.,.1,1.,-1.5,0.,-.1,1.,-1.5,0.])
    return pose+np.tile([.125,.25,.25,.25],4)*actions


def rotation(q):
    q = np.asarray(q, dtype=float)
    w, x, y, z = np.moveaxis(q/np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    values = (1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
              2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
              2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))
    return np.stack(values, axis=-1).reshape(q.shape[:-1]+(3,3))


class Mechanism:
    def __init__(self, path):
        root = ET.parse(path).find(".//body[@name='base_link']")
        self.offsets = np.empty((4,4,3))
        self.rotations = np.empty((4,4,3,3))
        self.limits = np.empty((4,3,2))
        for i, prefix in enumerate(("FL", "FR", "RL", "RR")):
            parent = root
            for j, name in enumerate(("hip", "thigh", "calf", "foot")):
                parent = parent.find(f"body[@name='{prefix}_{name}']")
                self.offsets[i,j] = np.fromstring(parent.get("pos"), sep=" ")
                self.rotations[i,j] = rotation(np.fromstring(parent.get("quat","1 0 0 0"),sep=" "))
                joint = parent.find("joint")
                assert joint.get("axis") == "0 0 1" and joint.get("pos") == "0 0 0"
                if j < 3:
                    self.limits[i,j] = np.fromstring(joint.get("range"),sep=" ")
            cylinder = parent.find("geom[@type='cylinder']")
            assert cylinder is not None and cylinder.get("pos","0 0 0") == "0 0 0"

    def forward(self, q):
        angles = np.asarray(q, dtype=float).reshape(-1,4,4)
        center = np.zeros(angles.shape[:2]+(3,))
        orient = np.broadcast_to(np.eye(3),center.shape+(3,))
        pivots, axes = [], []
        for j in range(4):
            center = center + (orient @ self.offsets[None,:,j,:,None])[...,0]
            if j < 3:
                fixed = orient @ self.rotations[None,:,j]
                pivots.append(center)
                axes.append(fixed[...,2])
                a = angles[:,:,j]/2.
                hinge = np.stack((np.cos(a),np.zeros_like(a),np.zeros_like(a),np.sin(a)), axis=-1)
                orient = fixed @ rotation(hinge)
        jac = np.stack([np.cross(axis, center-pivot) for pivot, axis in zip(pivots, axes)],axis=-1)
        return center, jac

    def lift(self, nominal, metres):
        nominal = np.asarray(nominal).reshape(-1,4,4)
        initial, jac = self.forward(nominal)
        target = initial.copy()
        target[:,:,2] += metres
        center = initial
        q = nominal.copy()
        low = np.maximum(nominal[:,:,:3]-.12, np.minimum(nominal[:,:,:3],self.limits[:,:,0]))
        high = np.minimum(nominal[:,:,:3]+.12, np.maximum(nominal[:,:,:3],self.limits[:,:,1]))
        for _ in range(2):
            jt = np.swapaxes(jac,-1,-2)
            delta = jt @ np.linalg.solve(jac@jt+.005**2*np.eye(3), (target-center)[...,None])
            q[:,:,:3] = np.clip(q[:,:,:3]+delta[...,0], low, high)
            center, jac = self.forward(q)
        return center-initial


def mechanism():
    cfg = read(ROOT/"training/configs/pilot03d.json")
    path = (ROOT/cfg["scene"]).parent/"robot.xml"
    assert cfg["robot_xml_sha256"] == digest(path)
    return Mechanism(path), cfg["robot_xml_sha256"]


def audit():
    chain, sha = mechanism()
    root = ROOT/"reports/speed110"
    groups = []
    for row in read(root/"results.json")["episodes"]:
        path = root/row["file"]
        assert digest(path) == row["trace_sha256"]
        with np.load(path, allow_pickle=False) as z:
            pose = z["qpos"]
            center, _ = chain.forward(pose[:,7:])
            world = pose[:,None,:3]+(rotation(pose[:,3:7])[:,None]@center[...,None])[...,0]
            max_xy_error = float(np.max(np.linalg.norm(world[:,:,:2]-z["wheels_xy"],axis=2)))
            assert max_xy_error < 1e-8
            moved = chain.lift(vendor_targets(z["issued"]), .012)
            vertical = 1000*moved[:,:,2]
            lateral = 1000*np.linalg.norm(moved[:,:,:2],axis=2)
            groups.append(dict(seed=row["seed"], trace_sha256=row["trace_sha256"], wheel_targets=vertical.size,
                               max_recorded_center_xy_error_m=max_xy_error, minimum_lift_mm=float(vertical.min()),
                               maximum_lift_mm=float(vertical.max()), maximum_horizontal_change_mm=float(lateral.max()),
                               within_0p4mm=int(np.sum((abs(vertical-12.)<=.4)&(lateral<=.4)))))
    return dict(robot_xml_sha256=sha, groups=groups, physical_substeps=0, policy_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    result = audit()
    file = ROOT/"reports/lidar_compound111/kinematics.json"
    if args.record:
        write(file, result)
    saved = read(file)
    assert {k:v for k,v in saved.items() if k != "groups"} == {k:v for k,v in result.items() if k != "groups"}
    for got, expected in zip(result["groups"], saved["groups"], strict=True):
        assert got.keys() == expected.keys()
        for key in got:
            if isinstance(got[key], float):
                np.testing.assert_allclose(got[key], expected[key], rtol=0., atol=1e-9)
            else:
                assert got[key] == expected[key]
    print(json.dumps(result, indent=2))
