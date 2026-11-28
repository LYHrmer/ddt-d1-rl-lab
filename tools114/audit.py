"""Audit the parking adapter against the three recorded round-113 action sequences."""
from collections import defaultdict
import json

import numpy as np

from tools91.launch91 import require
from tools110.course import TICKS, command
from tools111.control import SPEED_GAIN
from tools114.common import DEV, baseline
from tools114.native import RecordedVendor, with_parking


class LegGeometry:
    """Supply the actual stored output of the old leg-height stage to the parking adapter."""
    def __init__(self, trace):
        self.trace = trace
        self.tick = 0

    def raised_targets(self, raw, lift):
        k = self.tick
        self.tick += 1
        return self.trace["issued"][k], self.trace["issued_lift_fk"][k]


class ArchivedActorOutputs:
    def __init__(self, raw):
        self.raw = iter(raw)

    def run(self, outputs, inputs):
        return [next(self.raw)[None, :]]


def _play(trace, rows):
    """Replay only action-memory columns and the actuator FIFO, without evaluating a policy."""
    geometry = LegGeometry(trace)
    actor = RecordedVendor(ArchivedActorOutputs(trace["raw"]), rows)
    frame = np.zeros((1, 57), np.float32)
    history = np.zeros((1, 10, 57), np.float32)
    previous_issued = np.zeros(16, np.float32)
    issued, applied = [], []
    for k in range(TICKS):
        command(k)  # The parking adapter captures the driver's request before the actor correction.
        raw = actor.run(["recorded"], {"nn_input0": frame, "nn_input1": history})[0][0]
        targets, _ = geometry.raised_targets(raw, trace["lift_demand"][k])
        issued.append(targets)
        applied.append(previous_issued)
        history[:, :-1] = history[:, 1:].copy()
        history[:, -1] = frame
        frame[0, 41:57] = raw
        previous_issued = targets
    return np.stack(issued), np.stack(applied)


def audit_seed(seed):
    path, row = baseline(seed)
    with np.load(path, allow_pickle=False) as archive:
        fields = ("raw", "issued", "applied", "issued_lift_fk", "lift_demand", "base_cmd")
        z = {k: archive[k] for k in fields}
    for name in ("raw", "issued", "applied"):
        block = z[name]
        require(block.shape == (TICKS, 16) and block.dtype == np.float32 and np.isfinite(block).all(),
                "malformed recorded action array")
    np.testing.assert_array_equal(z["raw"], z["issued"])
    np.testing.assert_array_equal(z["lift_demand"], np.zeros((TICKS, 4)))
    old_fifo = np.vstack((np.zeros(16, np.float32), z["issued"][:-1]))
    np.testing.assert_array_equal(old_fifo, z["applied"])
    for block in z.values():
        block.setflags(write=False)

    rows = defaultdict(list)
    issued, applied = with_parking(_play, rows)(z, rows)
    driver = np.array(rows["driver_command_xyz"])
    np.testing.assert_array_equal(driver[:, 1:], np.zeros((TICKS, 2)))
    np.testing.assert_allclose(driver[:, 0] * SPEED_GAIN, z["base_cmd"], atol=1e-12, rtol=0.)
    positive = np.flatnonzero(z["base_cmd"] > 0.)
    first_stop = int(positive[-1]) + 1
    require(first_stop == 525 and np.all(z["base_cmd"][first_stop:] == 0.), "changed archived stop request")
    active = np.arange(TICKS) >= first_stop
    np.testing.assert_array_equal(rows["parking_active"], active)
    np.testing.assert_array_equal(rows["unprojected_issued"], z["issued"])
    np.testing.assert_array_equal(issued[~active], z["issued"][~active])
    leg_delta = (issued - z["issued"]).reshape(TICKS, 4, 4)[:, :, :3].astype(float) * [.125, .25, .25]
    np.testing.assert_array_equal(leg_delta, np.zeros_like(leg_delta))
    before = 5. * z["issued"][:, 3::4].astype(float)
    after = 5. * issued[:, 3::4].astype(float)
    old_mean, new_mean = before.mean(axis=1), after.mean(axis=1)
    removed = np.where(active, old_mean, 0.)
    np.testing.assert_allclose(rows["wheel_common_removed_rad_s"], removed, atol=1e-12, rtol=0.)
    predicted = before - removed[:, None]
    # The driver writes float32 actions back to the vendor interface.
    np.testing.assert_array_equal(issued[:, 3::4], (predicted / 5.).astype(np.float32))
    wheel_diffs = ((after[:, :, None] - after[:, None, :]) -
                   (before[:, :, None] - before[:, None, :]))
    require(np.max(abs(wheel_diffs)) < 1e-5 and np.max(abs(new_mean[active])) < 1e-5,
            "incorrect physical differential speed or parking common component")

    np.testing.assert_array_equal(applied, np.vstack((np.zeros(16, np.float32), issued[:-1])))
    changed = np.flatnonzero((issued != z["issued"]).any(axis=1))
    delayed = np.flatnonzero((applied != old_fifo).any(axis=1))
    np.testing.assert_array_equal(delayed, changed[changed + 1 < TICKS] + 1)
    require(len(changed) > 0 and len(delayed) > 0, "recorded stop did not exercise the adapter")
    previous_raw = np.vstack((np.zeros(16, np.float32), z["raw"][:-1]))
    np.testing.assert_array_equal(rows["actor_previous_raw"], previous_raw)
    history_index = np.maximum(np.arange(TICKS)[:, None] + np.arange(-10, 0), 0)
    np.testing.assert_array_equal(rows["actor_history_previous_raw"], previous_raw[history_index])
    return dict(seed=seed, trace_sha256=row["trace_sha256"], replayed_decisions=TICKS,
                active_tick_range_inclusive=[first_stop, TICKS-1], active_intervals=int(active.sum()),
                unchanged_pre_stop_intervals=first_stop, first_changed_issue_tick=int(changed[0]),
                first_changed_applied_tick=int(delayed[0]), changed_issued_intervals=len(changed),
                changed_applied_intervals=len(delayed), delay_ticks=1,
                active_mean_wheel_target_before_rad_s=float(old_mean[active].mean()),
                active_mean_wheel_target_after_rad_s=float(new_mean[active].mean()),
                active_peak_absolute_wheel_mean_before_rad_s=float(abs(old_mean[active]).max()),
                active_peak_absolute_wheel_mean_after_rad_s=float(abs(new_mean[active]).max()),
                max_leg_target_error_rad=float(abs(leg_delta).max()),
                max_pairwise_wheel_difference_error_rad_s=float(abs(wheel_diffs).max()),
                reconstructed_action_history_windows=TICKS, history_width=10,
                policy_feedback_exact=True, delayed_targets_exact=True)


def audit_archives():
    return dict(source_round=113, evidence="fixed_archived_actor_outputs",
                action_history="reconstructed_last_action_columns_only", actor_forwards=0,
                physics=0, policy_updates=0, world_updates=0, rows=[audit_seed(seed) for seed in DEV])


if __name__ == "__main__":
    print(json.dumps(audit_archives(), ensure_ascii=False, indent=2, allow_nan=False))
