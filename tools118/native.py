"""Measure the round-117 final controller at a new initial condition on the original course."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import require
from tools94.common94 import arrays, digest, write
from tools110.course import CFG, TICKS
from tools111.control import SPEED_GAIN
from tools111.course import score
from tools114.closed_results import parking_stop
from tools116.native import with_preview_input
from tools116.score import first_failed_section, lidar_effects
from tools117.native import ReleaseActor, with_parking


def episode(out, seed):
    """Use a fresh seeded reset; there is no baseline trajectory for a held-out start."""
    out = Path(out)
    require(out.is_dir() and {f.name for f in out.iterdir()} == {"reservation.json"},
            "blind run needs a fresh reservation")
    counts = dict(decisions=0, servo=0, physics=0, packets=0, ray_queries=0)
    rows = defaultdict(list)
    actor = height = None
    try:
        from tools110.native import MODEL_SHA, YAML_SHA, Plant, load_policy, mj
        from tools111.course import install_heightfield, make_scene
        from tools111.native import drive
        from training.d1_rl.interface import WHEEL_INDICES

        layout, arm = "development", "both"
        session = load_policy()
        with TemporaryDirectory(prefix="d1-compound118-") as temp:
            plant = Plant(config=CFG, scene=make_scene(temp, layout), seed=seed, arm="B", eval_delay_ticks=1,
                          eval_command_sequence=[{"start_s": 0., "vx": 0., "wz": 0.}], max_transitions=TICKS)
            try:
                install_heightfield(plant.model, layout)
                plant.reset(seed=seed)
                height = float(plant.nominal_base_height_m)
                wheels = np.asarray(plant.dof_ids)[WHEEL_INDICES]
                actor = ReleaseActor(session, lambda: plant.data.qvel[wheels])
                loop = with_preview_input(with_parking(drive, rows), actor)
                loop(plant, actor, rows, counts, layout, arm, SPEED_GAIN)
            finally:
                plant.close()
        trace = actor.trace(rows)
        trace["nominal_base_height_m"] = np.asarray(height)
        motion, _ = score(trace, layout)
        lidar = lidar_effects(trace, trace["velocity_trim"])
        ticks = motion["ticks"]
        packets = len(trace["capture_tick"])
        require(0 < ticks <= TICKS and packets <= 72 and counts == dict(decisions=ticks, servo=5*ticks,
                physics=10*ticks, packets=packets, ray_queries=1040*packets), "blind clocks differ")
        require(ticks == TICKS or trace["fallen"][-1] or trace["illegal"][-1], "unexplained early termination")
        np.testing.assert_array_equal(trace["preview_actor_vx"], rows["actor_vx"])
        arrays(out / "trace.npz", trace)
        return dict(status="COMPLETE", seed=seed, layout=layout, arm=arm, gain=SPEED_GAIN, parking=True,
                    actor_sha256=MODEL_SHA, actuator_config_sha256=YAML_SHA, mujoco_version=mj.__version__,
                    trace_sha256=digest(out / "trace.npz"), nominal_base_height_m=height, counts=counts,
                    motion=motion, lidar=lidar, stop=parking_stop(trace["qpos"], trace["qvel"]),
                    driving_pass=ticks == TICKS and lidar["passed"] and not (set(motion["failed"]) - {"stopped"}),
                    gate_pass=motion["passed"] and lidar["passed"], first_failed_section=first_failed_section(motion))
    except BaseException as error:
        partial = actor.trace(rows) if actor else {k: np.asarray(v) for k, v in rows.items()}
        if height is not None:
            partial["nominal_base_height_m"] = np.asarray(height)
        arrays(out / "partial.npz", partial)
        write(out / "partial_counts.json", dict(error=repr(error), counts=counts))
        raise
