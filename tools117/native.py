"""Run one registered release or parking trial on the frozen round-111 route."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType

import numpy as np

from tools91.launch91 import require
from tools94.common94 import arrays, digest, write
from tools110.course import CFG, TICKS, command
from tools111.control import SPEED_GAIN
from tools111.course import score
from tools114.closed_results import parking_stop
from tools114.control import StopProjection
from tools116.control import PrevTrimActor
from tools116.native import with_preview_input
from tools116.response_replay import RELEASE_SLEW, SaturationRelease
from tools116.score import first_failed_section, lidar_effects


class ReleaseTrim(SaturationRelease):
    """Connect the registered four-input rule to the round-116 actor callback."""

    def __init__(self):
        super().__init__(RELEASE_SLEW)
        self.preview = None

    def update(self, request, last_actual, wheels):
        require(self.preview is not None, "missing lidar preview for this decision")
        preview, self.preview = self.preview, None
        return self.update_preview(request, last_actual, wheels, preview)


class ReleaseActor(PrevTrimActor):
    def __init__(self, session, read_wheels):
        super().__init__(session, read_wheels)
        self.control = ReleaseTrim()

    def set_preview(self, vx):
        super().set_preview(vx)
        self.control.preview = float(vx)


def with_parking(drive, rows):
    """Insert the round-114 projection after the original lidar leg IK and before the FIFO."""
    geometry = drive.__globals__["LegGeometry"]
    projection = StopProjection()

    class ParkedGeometry(geometry):
        def raised_targets(self, raw, lift):
            issued, height = super().raised_targets(raw, lift)
            request = command(len(rows["raw"]))
            return projection.apply((request, 0., 0.), issued), height

    bindings = dict(drive.__globals__, LegGeometry=ParkedGeometry)
    return FunctionType(drive.__code__, bindings, drive.__name__, drive.__defaults__, drive.__closure__)


def episode(out, seed, reference_path, parking=False):
    out, reference_path = Path(out), Path(reference_path)
    require(out.is_dir() and not any((out / f).exists() for f in
            ("trace.npz", "partial.npz", "partial_counts.json")), "native trial already attempted")
    counts = dict(decisions=0, servo=0, physics=0, packets=0, ray_queries=0)
    rows = defaultdict(list)
    actor = None
    try:
        from tools110.native import MODEL_SHA, YAML_SHA, Plant, load_policy, mj
        from tools111.course import install_heightfield, make_scene
        from tools111.native import drive
        from training.d1_rl.interface import WHEEL_INDICES

        layout, arm = "development", "both"
        session = load_policy()
        with TemporaryDirectory(prefix="d1-compound117-") as tmp:
            plant = Plant(config=CFG, scene=make_scene(tmp, layout), seed=seed, arm="B", eval_delay_ticks=1,
                          eval_command_sequence=[{"start_s": 0., "vx": 0., "wz": 0.}], max_transitions=TICKS)
            try:
                install_heightfield(plant.model, layout)
                plant.reset(seed=seed)
                height = float(plant.nominal_base_height_m)
                with np.load(reference_path, allow_pickle=False) as before:
                    np.testing.assert_array_equal(plant.data.qpos, before["qpos"][0])
                    np.testing.assert_array_equal(plant.data.qvel, before["qvel"][0])
                    require(height == float(before["nominal_base_height_m"]), "different paired reset height")
                wheels = np.asarray(plant.dof_ids)[WHEEL_INDICES]
                actor = ReleaseActor(session, lambda: plant.data.qvel[wheels])
                loop = with_parking(drive, rows) if parking else drive
                with_preview_input(loop, actor)(plant, actor, rows, counts, layout, arm, SPEED_GAIN)
            finally:
                plant.close()
        trace = actor.trace(rows)
        trace["nominal_base_height_m"] = np.asarray(height)
        motion, _ = score(trace, layout)
        lidar = lidar_effects(trace, trace["velocity_trim"])
        n = motion["ticks"]
        require(0 < n <= TICKS and counts == dict(decisions=n, servo=5*n, physics=10*n,
                packets=len(trace["capture_tick"]), ray_queries=1040*len(trace["capture_tick"]))
                and counts["packets"] <= 72, "native clocks differ")
        np.testing.assert_array_equal(trace["preview_actor_vx"], rows["actor_vx"])
        np.testing.assert_array_equal(trace["applied"], np.vstack((np.zeros(16), trace["issued"][:-1])))
        drive_pass = n == TICKS and lidar["passed"] and not (set(motion["failed"]) - {"stopped"})
        arrays(out / "trace.npz", trace)
        return dict(status="COMPLETE", seed=seed, parking=parking, layout=layout, arm=arm, gain=SPEED_GAIN,
                    counts=counts, motion=motion, lidar=lidar, driving_pass=drive_pass,
                    gate_pass=motion["passed"] and lidar["passed"], first_failed_section=first_failed_section(motion),
                    stop=parking_stop(trace["qpos"], trace["qvel"]), nominal_base_height_m=height,
                    actor_sha256=MODEL_SHA, actuator_config_sha256=YAML_SHA, mujoco_version=mj.__version__,
                    trace_sha256=digest(out / "trace.npz"), reference_sha256=digest(reference_path))
    except BaseException as exc:
        partial = actor.trace(rows) if actor else {key: np.asarray(value) for key, value in rows.items()}
        arrays(out / "partial.npz", partial)
        write(out / "partial_counts.json", dict(error=repr(exc), counts=counts))
        raise
