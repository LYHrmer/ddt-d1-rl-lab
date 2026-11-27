"""The live plant's first returns and a separate, fixed-base encoder kinematic model."""
import numpy as np
import mujoco as mj

from tools91.launch91 import ROOT, read
from tools100.sensor100 import _grid, _wheel_cylinders
from tools110.native import POSE, SCALE
from tools111.control import Packet, body_to_level
from training.d1_rl.interface import JOINT_NAMES


class Lidar:
    def __init__(self, model):
        self.geometry = read(ROOT / "docs/experiments100/pointcloud_density.json")["geometry"]
        self.directions = _grid(self.geometry)[0]
        self.mounts = np.asarray(self.geometry["mounts_m"])
        self.base = model.body("base_link").id
        self.model = model
        self.ray_calls = 0

    def capture(self, tick, data, quat_imu, odom):
        # World pose is used only by the simulator to generate what the sensor sees.
        rotation = data.xmat[self.base].reshape(3, 3)
        rays = np.ascontiguousarray(self.directions @ rotation.T).ravel()
        level = body_to_level(quat_imu)
        near, far = (self.geometry[k] for k in ("range_min_m", "range_max_m"))
        n = len(self.directions)
        geom = np.empty(n, np.int32)
        points, ranges, validities = [], [], []
        for mount in self.mounts:
            origin = data.xpos[self.base] + rotation @ mount
            all_hits, moving_hits = np.empty(n), np.empty(n)
            for include_static, distances in ((True, all_hits), (False, moving_hits)):
                mj.mj_multiRay(self.model, data, origin, rays, None, include_static, -1, geom, distances, n, far)
                self.ray_calls += n
            valid = (all_hits >= near) & (all_hits <= far)
            valid &= ~((moving_hits >= 0.) & (np.abs(moving_hits-all_hits) <= 1e-8))
            local = (mount + self.directions*all_hits[:, None]) @ level
            local[~valid] = np.nan
            points.append(local)
            ranges.append(all_hits)
            validities.append(valid)
        packet = Packet(tick, np.vstack(points), odom.xy.copy(), odom.yaw)
        return packet, np.concatenate(ranges), np.concatenate(validities)


class LegGeometry:
    """Use only known robot geometry and 16 joint angles; floating-base pose stays at identity."""
    def __init__(self, model):
        self.model = model
        self.state = mj.MjData(model)
        self.state.qpos[:7] = (0., 0., 0., 1., 0., 0., 0.)
        joints = [model.joint(name).id for name in JOINT_NAMES]
        self.qids = model.jnt_qposadr[joints]
        self.dofs = model.jnt_dofadr[joints]
        self.bounds = model.jnt_range[joints]
        self.wheels = _wheel_cylinders(model, mj.mjtGeom.mjGEOM_CYLINDER)
        self.jac = np.empty((3, model.nv))

    def centers(self, q):
        self.state.qpos[self.qids] = q
        mj.mj_kinematics(self.model, self.state)
        return self.state.geom_xpos[self.wheels].copy()

    def raised_targets(self, action, lift):
        if not np.any(lift):
            return action.copy(), np.zeros(4)
        nominal = POSE + SCALE*action
        initial = self.centers(nominal)
        goal = initial.copy()
        goal[:, 2] += lift
        q = nominal.copy()
        centers = initial
        for _ in range(2):
            mj.mj_comPos(self.model, self.state)
            for i in np.flatnonzero(lift):
                leg = slice(i*4, i*4+3)
                mj.mj_jacGeom(self.model, self.state, self.jac, None, int(self.wheels[i]))
                jac = self.jac[:, self.dofs[leg]]
                correction = jac.T @ np.linalg.solve(jac@jac.T + .005**2*np.eye(3), goal[i]-centers[i])
                low = np.maximum(nominal[leg]-.12, np.minimum(nominal[leg], self.bounds[leg, 0]))
                high = np.minimum(nominal[leg]+.12, np.maximum(nominal[leg], self.bounds[leg, 1]))
                q[leg] = np.clip(q[leg]+correction, low, high)
            centers = self.centers(q)
        issued = action + (q-nominal)/SCALE
        return issued, centers[:, 2]-initial[:, 2]
