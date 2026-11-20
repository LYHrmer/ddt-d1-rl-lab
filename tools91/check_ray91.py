"""Local geometry check on synthetic poses for 91. No physics integration."""
import numpy as np

from tools91.rollout91 import _PredictedRanges
from training.d1_rl.interface import DEFAULT_Q


def check():
    import mujoco
    from tools87.terrain87 import D1Terrain87, load_model

    geometry = load_model("up")
    predicted, reference = _PredictedRanges(geometry), _PredictedRanges(geometry)
    state = np.zeros(45, dtype=np.float64)
    state[2:4] = [.46, 1.]
    state[7:23] = DEFAULT_Q
    packets = []
    for tick in range(12):
        state[0] = .14 * tick
        reference.data.qpos[:] = state[:23]
        reference.data.qvel[:] = state[23:]
        mujoco.mj_forward(geometry, reference.data)
        D1Terrain87._update_measurement(reference, tick)
        packets.append(predicted.at(tick, state))
        if not np.array_equal(packets[-1], reference.last_raw_features):
            raise ValueError(f"imagined ray packet differs from the 87 native sensor at tick {tick}")
    if (np.any(packets[:2]) or not np.any(packets[2]) or not np.array_equal(packets[2], packets[10])
            or np.array_equal(packets[10], packets[11]) or reference.data.time or predicted.data.time):
        raise ValueError("two-tick delay, hold, new capture or zero-integration clock differs")
    return {"status": "PASS", "synthetic_poses": len(packets), "identical_packets": len(packets),
            "delivery_ticks": [2, 11], "physics_steps": 0}


if __name__ == "__main__":
    import json

    print(json.dumps(check()))
