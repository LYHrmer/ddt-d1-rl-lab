"""Evaluate the frozen D1 actor with a wheel-target projection at parking."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType

import numpy as np

from tools91.launch91 import require
from tools94.common94 import arrays, digest, write
from tools110.course import CFG, TICKS
from tools111.control import SPEED_GAIN
from tools111.course import score
from tools113.native import TrimmedActor
from tools114.common import BLIND, DEV, baseline
from tools114.control import StopProjection


class RecordedVendor:
    """Record action feedback at the ONNX input, after the round-113 input correction."""
    def __init__(self, session, rows):
        self.session = session
        self.rows = rows

    def run(self, outputs, inputs):
        self.rows["actor_previous_raw"].append(inputs["nn_input0"][0, 41:57].copy())
        self.rows["actor_history_previous_raw"].append(inputs["nn_input1"][0, :, 41:57].copy())
        return self.session.run(outputs, inputs)


def with_parking(drive, rows):
    """Bind the driver-request and target interfaces on a private copy of the frozen loop."""
    namespace = drive.__globals__.copy()
    original_command = namespace["command"]
    original_geometry = namespace["LegGeometry"]
    stop = StopProjection()
    request = None

    def driver_command(tick):
        nonlocal request
        vx = original_command(tick)
        # The registered course has no driver lateral or yaw command. IMU heading
        # feedback and encoder trim are added later to the actor input.
        request = (vx, 0., 0.)
        return vx

    class ParkingGeometry(original_geometry):
        def raised_targets(self, raw, lift):
            nonlocal request
            require(request is not None, "wheel targets requested without a new driver command")
            unprojected, heights = super().raised_targets(raw, lift)
            issued = stop.apply(request, unprojected)
            rows["driver_command_xyz"].append(request)
            rows["parking_active"].append(stop.active)
            rows["wheel_common_removed_rad_s"].append(stop.removed_rad_s)
            rows["unprojected_issued"].append(unprojected.copy())
            request = None
            return issued, heights

    namespace.update(command=driver_command, LegGeometry=ParkingGeometry)
    # Bind only this episode. Every other caller retains the original drive and
    # geometry, and the motor loop still owns both the raw and the delayed FIFO.
    return FunctionType(drive.__code__, namespace, drive.__name__, drive.__defaults__, drive.__closure__)


def check_feedback(trace):
    raw = trace["raw"]
    previous = trace["actor_previous_raw"]
    np.testing.assert_array_equal(previous[0], np.zeros(16))
    np.testing.assert_array_equal(previous[1:], raw[:-1])
    history_indices = np.maximum(np.arange(len(raw))[:, None] - np.arange(10, 0, -1), 0)
    np.testing.assert_array_equal(trace["actor_history_previous_raw"], previous[history_indices])
    np.testing.assert_array_equal(trace["applied"][0], np.zeros(16))
    np.testing.assert_array_equal(trace["applied"][1:], trace["issued"][:-1])
    np.testing.assert_array_equal(trace["actor_vx"], trace["preview_actor_vx"] + trace["velocity_trim"])
    np.testing.assert_array_equal(trace["actor_frame_command"][:, 0], (2. * trace["actor_vx"]).astype(np.float32))
    np.testing.assert_array_equal(trace["encoder_forward_m_s"], .087*np.median(trace["encoder_wheel_rad_s"], axis=1))


def episode(out, seed):
    """Run one reserved case. Exclusive artifacts preserve interrupted attempts."""
    out = Path(out)
    if not out.is_dir():
        raise FileNotFoundError(f"reserved episode directory is missing: {out}")
    if any((out / name).exists() for name in ("trace.npz", "partial.npz", "partial_counts.json")):
        raise FileExistsError(f"episode output already exists in {out}")
    rows = defaultdict(list)
    counts = dict(decisions=0, servo=0, physics=0, packets=0, ray_queries=0)
    actor = None
    try:
        from tools110.course import scene
        from tools110.native import MODEL_SHA, YAML_SHA, Plant, load_policy, mj
        from tools111.native import drive
        from tools112.surface_native import flatten
        from training.d1_rl.interface import WHEEL_INDICES

        require(seed in (*DEV, *BLIND), "unregistered parking start")
        paired = seed in DEV
        # The blind cases use the same runtime fingerprint with fresh resets.
        source_path, stack = baseline(seed if paired else DEV[0])
        require(stack["actor_sha256"] == MODEL_SHA and stack["actuator_config_sha256"] == YAML_SHA
                and stack["mujoco_version"] == mj.__version__, "native stack differs from round 113")
        session = RecordedVendor(load_policy(), rows)
        with TemporaryDirectory(prefix="d1-parking114-") as directory:
            plant = Plant(config=CFG, scene=scene(directory, "ridge"), seed=seed, arm="B", eval_delay_ticks=1,
                          eval_command_sequence=[{"start_s": 0., "vx": 0., "wz": 0.}], max_transitions=TICKS)
            try:
                flatten(plant.model)
                plant.reset(seed=seed)
                nominal_height = float(plant.nominal_base_height_m)
                if paired:
                    with np.load(source_path, allow_pickle=False) as off:
                        np.testing.assert_array_equal(plant.data.qpos, off["qpos"][0])
                        np.testing.assert_array_equal(plant.data.qvel, off["qvel"][0])
                wheel_ids = np.asarray(plant.dof_ids)[WHEEL_INDICES]
                actor = TrimmedActor(session, lambda: plant.data.qvel[wheel_ids])
                drive_with_parking = with_parking(drive, rows)
                drive_with_parking(plant, actor, rows, counts, "flat", "heading_on", SPEED_GAIN)
            finally:
                plant.close()
        trace = actor.trace(rows)
        motion, effects = score(trace, "flat")
        n = motion["ticks"]
        require(counts["decisions"] == n and counts["servo"] == 5*n and counts["physics"] == 10*n <= 6000
                and counts["ray_queries"] == 1040*counts["packets"] <= 74880, "parking native clock mismatch")
        check_feedback(trace)
        arrays(out / "trace.npz", trace)
        return dict(status="COMPLETE", seed=seed, cohort="development" if paired else "blind", layout="flat",
                    surface="level_heightfield", arm="heading_on", gain=SPEED_GAIN, trim="brake_aware",
                    projection="parked_wheel_common_zero", nominal_base_height_m=nominal_height,
                    counts=counts, motion=motion, effects=effects, trace_sha256=digest(out / "trace.npz"),
                    reference_sha256=stack["trace_sha256"] if paired else None,
                    actor_sha256=MODEL_SHA, actuator_config_sha256=YAML_SHA, mujoco_version=mj.__version__)
    except BaseException as error:
        partial = actor.trace(rows) if actor is not None else {key: np.asarray(vals) for key, vals in rows.items()}
        arrays(out / "partial.npz", partial)
        write(out / "partial_counts.json", dict(error=repr(error), counts=counts))
        raise
