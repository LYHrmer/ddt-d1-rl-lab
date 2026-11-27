"""Estimate a longitudinal speed correction using the issued command and wheel encoders."""
import json
import math
from statistics import median

from tools111.control import SPEED_GAIN


class VelocityTrim:
    def __init__(self):
        self.last_request = None
        self.held_ticks = 0
        self.bias = 0.
        self.trim = 0.

    def update(self, request, last_issued, wheel_rad_s):
        """At 50 Hz, return the increment to the current lidar-shaped actor command."""
        if len(wheel_rad_s) != 4 or not all(math.isfinite(x) for x in (request,last_issued,*wheel_rad_s)):
            raise ValueError("four finite encoder readings and finite commands required")
        self.held_ticks = self.held_ticks+1 if self.last_request == request else 0
        self.last_request = request
        if self.held_ticks >= 10:
            encoder_speed = .087*median(wheel_rad_s)
            weight = -math.expm1(-.02/.40)
            self.bias += weight*(last_issued-encoder_speed-self.bias)
            desired = min(.18,max(0.,(1.025-SPEED_GAIN)*request+self.bias))
            self.trim += min(.0036,max(-.0036,desired-self.trim))
        return self.trim


def archived():
    import numpy as np
    from tools91.launch91 import digest, read
    from tools110.course import command
    from tools111.replay import check_episode
    from tools111.results import PUBLIC
    cases = read(PUBLIC/"native.json")["cases"]
    rows = []
    for case in cases:
        if case["case_status"] != "COMPLETE" or case["arm"] not in ("heading_on","both"):
            continue
        path = PUBLIC/case["file"]
        assert digest(path) == case["episode"]["trace_sha256"]
        with np.load(path,allow_pickle=False) as z:
            if case["arm"] == "heading_on":
                # Verify that enabling both lidar channels on this recorded flat path reproduces every issued command.
                check_episode(z,{**case["episode"],"arm":"both"})
            obs = VelocityTrim()
            actual = z["actor_vx"]
            wheels = z["encoder_wheel_rad_s"]
            values = np.array([obs.update(command(k),actual[k-1] if k else 0.,w) for k,w in enumerate(wheels)])
            body_x = z["xy"][:-1,0]
            middle = (body_x>=3.) & (body_x<=6.)
            tail = np.arange(len(values))>=575
            rows.append(dict(seed=case["seed"],arm=case["arm"],trace_sha256=digest(path),
                             middle_median_m_s=float(np.median(values[middle])),
                             stop_median_m_s=float(np.median(values[tail])),
                             maximum_m_s=float(values.max())))
    return rows


if __name__ == "__main__":
    print(json.dumps(archived(),indent=2))
