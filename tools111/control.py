"""Causal wheel-track preview and speed/leg targets using delivered lidar and onboard odometry."""
from collections import deque
from dataclasses import dataclass

import numpy as np

SPEED_GAIN = 0.97792348
ARMS = {"heading_off": (False, False, False), "heading_on": (True, False, False),
        "neither": (True, False, False), "speed": (True, True, False),
        "legs": (True, False, True), "both": (True, True, True)}


def yaw_from_quat(quat):
    w, x, y, z = quat
    return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def axes(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def body_to_level(quat):
    """Columns are the gravity-level front, left and up directions in the IMU/body frame."""
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if quat.shape != (4,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid IMU orientation")
    w, x, y, z = quat/norm
    # Row three of body-to-world rotation expresses world vertical in body coordinates.
    up = np.array([2.*(x*z-w*y), 2.*(y*z+w*x), 1.-2.*(x*x+y*y)])
    forward = np.array([1., 0., 0.]) - up[0]*up
    length = np.linalg.norm(forward)
    if length < 1e-8:
        raise ValueError("the heading axis has no horizontal projection")
    forward /= length
    return np.column_stack((forward, np.cross(up, forward), up))


def heading_rate(reference, measured, moving):
    error = reference - measured
    return float(np.clip(2.5*np.arctan2(np.sin(error), np.cos(error)), -.35, .35)) if moving else 0.


class WheelOdometry:
    def __init__(self, yaw):
        self.yaw = yaw
        self.xy = np.zeros(2)
        self.travel = 0.
        self.previous_speed = None

    def update(self, yaw, wheels_rad_s):
        speed = .087 * float(np.median(wheels_rad_s))
        if self.previous_speed is not None:
            ds = (self.previous_speed + speed) * .01
            dyaw = np.arctan2(np.sin(yaw-self.yaw), np.cos(yaw-self.yaw))
            self.xy += ds * axes(self.yaw+.5*dyaw)[:, 0]
            self.travel += ds
        self.previous_speed, self.yaw = speed, yaw


@dataclass
class Packet:
    tick: int
    level_points: np.ndarray
    odom_xy: np.ndarray
    yaw: float

    def aligned(self, odom):
        points = self.level_points.copy()
        points[:, :2] = ((points[:, :2] @ axes(self.yaw).T + self.odom_xy - odom.xy)
                         @ axes(odom.yaw))
        return points


def track_features(points, wheel_xy):
    """Return distance and signed height change between supported patches in one wheel track."""
    dx = points[:, 0] - wheel_xy[0]
    inside = np.isfinite(points).all(axis=1) & (np.abs(points[:, 1]-wheel_xy[1]) <= .115)
    # Fixed-width bins prevent the denser lower beams from overwhelming a small step top.
    edges = np.arange(.04, 1.031, .09)
    patches = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sample = points[inside & (dx >= lo) & (dx < hi)]
        if len(sample) >= 4:
            patches.append((float(np.median(sample[:, 0])-wheel_xy[0]), float(np.median(sample[:, 2]))))
    features = []
    for i, (x0, z0) in enumerate(patches):
        for x1, z1 in patches[i+1:]:
            if x1-x0 > .40:
                break
            if x1-x0 >= .065 and abs(z1-z0) > .005:
                features.append(((x0+x1)/2., z1-z0))
    return features, len(patches)


class Preview:
    def __init__(self, arm, gain=SPEED_GAIN):
        self.heading, self.use_speed, self.use_legs = ARMS[arm]
        self.gain = gain
        self.events = deque()
        self.last_packet = -1
        self.coverage = np.zeros(2, dtype=int)
        self.speed_delta = 0.
        self.lift = np.zeros(4)

    def observe(self, tick, packet, odom, wheel_xy):
        if packet.tick != tick-2 or packet.tick <= self.last_packet:
            raise ValueError("preview requires the next delivered 40 ms packet")
        self.last_packet = packet.tick
        points = packet.aligned(odom)
        for side in (0, 1):
            features, self.coverage[side] = track_features(points, wheel_xy[side])
            for ahead, dz in features:
                severity = min(1., (abs(dz)-.005)/.025)
                self.events.append((tick, side, odom.travel+ahead, severity, dz > 0))

    def targets(self, tick, command, reference_yaw, odom, wheel_xy):
        # A short physical landmark memory lets the rear axle reach a patch seen by the front.
        while self.events and self.events[0][0] < tick-45:
            self.events.popleft()
        risk = 0.
        lift = np.zeros(4)
        for _, side, position, severity, rising in self.events:
            remaining = position - odom.travel
            wheelbase = wheel_xy[side, 0]-wheel_xy[side+2, 0]
            for index, distance in ((side, remaining), (side+2, remaining+wheelbase)):
                if -.15 <= distance <= .75:
                    risk = max(risk, severity)
                if rising and -.11 <= distance <= .56:
                    lift[index] = max(lift[index], .012*severity)
        moving = float(np.clip(command/1.5, 0., 1.))
        desired_delta = .045*risk*moving if self.use_speed else 0.
        self.speed_delta += float(np.clip(desired_delta-self.speed_delta, -.009, .009))
        desired_lift = lift*moving if self.use_legs else np.zeros(4)
        self.lift += np.clip(desired_lift-self.lift, -.003, .003)
        base = self.gain*command
        speed = max(0., base-self.speed_delta)
        yaw_rate = heading_rate(reference_yaw, odom.yaw, self.heading and command > .01)
        return speed, yaw_rate, self.lift.copy()
