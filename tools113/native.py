"""Evaluate the frozen vendor actor on the original level grid with encoder compensation."""
from collections import defaultdict
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import require
from tools94.common94 import arrays, digest, read, write
from tools110.course import CFG, TICKS, command
from tools111.control import SPEED_GAIN
from tools111.course import score
from tools112.surface import PUBLIC as SURFACE_RESULTS, SEEDS, bindings
from tools113.control import BrakeTrim


class TrimmedActor:
    """Change the forward component of the frame shared by ONNX and its history."""
    def __init__(self, session, read_wheels):
        self.session = session
        self.read_wheels = read_wheels
        self.control = BrakeTrim()
        self.last_actual = 0.
        self.commands = []
        self.increments = []
        self.encoders = []

    def run(self, outputs, inputs):
        tick = len(self.commands)
        request = command(tick)
        preview_speed = SPEED_GAIN * request
        frame = inputs["nn_input0"]
        require(frame.shape == (1, 57) and frame.dtype == np.float32
                and frame[0, 6] == np.float32(2. * preview_speed), "unexpected heading-only vendor input")
        increment = self.control.update(request, self.last_actual, self.read_wheels())
        actual = preview_speed + increment
        frame[0, 6] = np.float32(2. * actual)
        # tools111.drive stores this same frame in history after inference.
        self.last_actual = actual
        self.commands.append(actual)
        self.increments.append(increment)
        self.encoders.append(self.control.encoder_forward_m_s)
        return self.session.run(outputs, inputs)

    def trace(self, rows):
        trace = {key: np.asarray(values) for key, values in rows.items()}
        trace["preview_actor_vx"] = trace.get("actor_vx", np.empty(0))
        trace["actor_vx"] = np.asarray(self.commands)
        trace["velocity_trim"] = np.asarray(self.increments)
        trace["encoder_forward_m_s"] = np.asarray(self.encoders)
        return trace


def grid_reference(seed):
    require(seed in SEEDS, "unregistered heightfield starting state")
    report = read(SURFACE_RESULTS / "results.json")
    require(report["sources"] == bindings(), "reference controller or grid source changed")
    case, = [c for c in report["cases"] if c["seed"] == seed]
    row = case["episode"]
    require(case["case_status"] == row["status"] == "COMPLETE" and row["motion"]["ticks"] == TICKS
            and row["surface"] == "level_heightfield" and row["arm"] == "heading_on" and row["gain"] == SPEED_GAIN,
            "invalid original heightfield case")
    path = SURFACE_RESULTS / case["file"]
    require(digest(path) == row["trace_sha256"], "original heightfield trace checksum mismatch")
    return path, row


def episode(out, seed):
    """Write one newly reserved trajectory, preserving partial arrays on failure."""
    import mujoco as mj
    from tools110.course import scene
    from tools110.native import MODEL_SHA, YAML_SHA, Plant, load_policy
    from tools111.native import drive
    from tools112.surface_native import flatten
    from training.d1_rl.interface import WHEEL_INDICES

    rows = defaultdict(list)
    counts = dict(decisions=0, servo=0, physics=0, packets=0, ray_queries=0)
    actor = None
    try:
        path, reference = grid_reference(seed)
        require(reference["actor_sha256"] == MODEL_SHA and reference["actuator_config_sha256"] == YAML_SHA
                and reference["mujoco_version"] == mj.__version__, "native stack differs from reference")
        session = load_policy()
        with TemporaryDirectory(prefix="d1-speed113-") as directory:
            plant = Plant(config=CFG, scene=scene(directory, "ridge"), seed=seed, arm="B", eval_delay_ticks=1,
                          eval_command_sequence=[{"start_s": 0., "vx": 0., "wz": 0.}], max_transitions=TICKS)
            try:
                flatten(plant.model)
                plant.reset(seed=seed)
                with np.load(path, allow_pickle=False) as off:
                    np.testing.assert_array_equal(plant.data.qpos, off["qpos"][0])
                    np.testing.assert_array_equal(plant.data.qvel, off["qvel"][0])
                wheel_ids = np.asarray(plant.dof_ids)[WHEEL_INDICES]
                actor = TrimmedActor(session, lambda: plant.data.qvel[wheel_ids])
                drive(plant, actor, rows, counts, "flat", "heading_on", SPEED_GAIN)
            finally:
                plant.close()
        trace = actor.trace(rows)
        motion, effects = score(trace, "flat")
        n = motion["ticks"]
        require(counts["decisions"] == n and counts["servo"] == 5*n and counts["physics"] == 10*n <= 6000
                and counts["ray_queries"] == 1040*counts["packets"] <= 74880, "native clock mismatch")
        np.testing.assert_array_equal(trace["actor_vx"], trace["preview_actor_vx"] + trace["velocity_trim"])
        np.testing.assert_array_equal(trace["actor_frame_command"][:, 0], (2. * trace["actor_vx"]).astype(np.float32))
        np.testing.assert_array_equal(trace["encoder_forward_m_s"], .087*np.median(trace["encoder_wheel_rad_s"], axis=1))
        arrays(out / "trace.npz", trace)
        return dict(status="COMPLETE", seed=seed, layout="flat", surface="level_heightfield", arm="heading_on",
                    gain=SPEED_GAIN, trim="brake_aware", counts=counts, motion=motion, effects=effects,
                    trace_sha256=digest(out / "trace.npz"), reference_sha256=reference["trace_sha256"],
                    actor_sha256=MODEL_SHA, actuator_config_sha256=YAML_SHA, mujoco_version=mj.__version__)
    except BaseException as error:
        partial = actor.trace(rows) if actor is not None else {key: np.asarray(vals) for key, vals in rows.items()}
        arrays(out / "partial.npz", partial)
        write(out / "partial_counts.json", dict(error=repr(error), counts=counts))
        raise
