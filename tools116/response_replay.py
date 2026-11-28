"""Read the development traces at fixed measured states to examine two follow-up control changes."""
import json

import numpy as np

from tools91.launch91 import ROOT, digest, read, require
from tools110.course import command
from tools111.control import SPEED_GAIN
from tools113.control import BrakeTrim
from tools114.control import StopProjection


INDEX = ROOT / 'reports/compound116/results.json'
INDEX_SHA = 'b47f1f3bf7c0da97c427f7871d1f103ba32e76070d0369fc1d284994d5115b6a'
ORIGINAL_SLEW = .0036
TRIM_CAP = .18
RELEASE_SLEW = 2 * ORIGINAL_SLEW


class SaturationRelease(BrakeTrim):
    """Use faster downward slew only while the lidar is slowing cruise and trim is unwinding from its cap."""

    def __init__(self, release_slew):
        super().__init__()
        self.release_slew = release_slew
        self.at_cap_before_release = False

    def update_preview(self, request, previous_actual, wheels, preview):
        previous_inner, previous_outer = self.cruise.trim, self.trim
        original = super().update(request, previous_actual, wheels)
        steady_preview = request == 1.5 and self.cruise.held_ticks >= 10 and preview < SPEED_GAIN * request
        if not steady_preview:
            self.at_cap_before_release = False
            return original

        # This is the desired trim before the frozen inner slew limiter.
        desired = min(TRIM_CAP, max(0., (1.025 - SPEED_GAIN) * request + self.cruise.bias))
        if previous_outer >= TRIM_CAP - 1e-12:
            self.at_cap_before_release = True
        elif desired >= previous_inner:
            self.at_cap_before_release = False
        if not self.at_cap_before_release or desired >= previous_inner:
            return original

        # The two original limiters are in series. Use one shared falling rate for both.
        # At this fixed forward request the original outer cap is at least TRIM_CAP.
        self.cruise.trim = previous_inner + max(-self.release_slew, desired - previous_inner)
        self.trim = previous_outer + min(ORIGINAL_SLEW, max(-self.release_slew, self.cruise.trim - previous_outer))
        if self.cruise.trim <= desired + 1e-12:
            self.at_cap_before_release = False
        return self.trim


def trim_at_recorded_states(encoders, preview, release_slew):
    controller = SaturationRelease(release_slew)
    previous_actual = 0.
    increments = []
    for tick, (wheel_speeds, preview_speed) in enumerate(zip(encoders, preview, strict=True)):
        increment = controller.update_preview(command(tick), previous_actual, wheel_speeds, preview_speed)
        increments.append(increment)
        previous_actual = preview_speed + increment
    return np.asarray(increments)


def speed_readback(trace, gate_times):
    preview, encoders = trace['preview_actor_vx'], trace['encoder_wheel_rad_s']
    baseline = trim_at_recorded_states(encoders, preview, ORIGINAL_SLEW)
    np.testing.assert_array_equal(baseline, trace['velocity_trim'])
    np.testing.assert_array_equal(preview + baseline, trace['actor_vx'])
    candidate = trim_at_recorded_states(encoders, preview, RELEASE_SLEW)
    times = trace['time'][:-1]
    windows = [(times >= a) & (times < b) for a, b in zip(gate_times[:-1], gate_times[1:], strict=True)]
    last = windows[-1]
    falling = np.r_[0., baseline[:-1]] - baseline
    return dict(total_decisions=len(times), identical_full_command_trace=bool(np.array_equal(candidate, baseline)),
                reached_cap_with_lidar=bool(np.any((baseline >= TRIM_CAP-1e-12) & (preview < trace['base_cmd']))),
                last_metre_ticks=int(last.sum()), original_down_limit_ticks=int(np.isclose(
                    falling[last], ORIGINAL_SLEW, rtol=0., atol=1e-12).sum()),
                mean_policy_command_change_m_s_by_metre=[float((candidate[mask] - baseline[mask]).mean())
                                                         for mask in windows])


def parking_readback(trace):
    projection = StopProjection()
    issued, applied = trace['issued'], trace['applied']
    new_issued = np.stack([projection.apply((command(k), 0., 0.), action) for k, action in enumerate(issued)])
    new_applied = np.vstack((np.zeros_like(issued[0]), new_issued[:-1]))
    np.testing.assert_array_equal(new_applied[:526], applied[:526])
    np.testing.assert_array_equal(new_applied.reshape(-1, 4, 4)[:, :, :3], applied.reshape(-1, 4, 4)[:, :, :3])
    wheels = trace['encoder_wheel_rad_s'][575:]
    old_target = 5. * applied[575:, 3::4].astype(float)
    new_target = 5. * new_applied[575:, 3::4].astype(float)
    np.testing.assert_allclose(new_target.mean(axis=1), 0., atol=1e-6, rtol=0.)
    np.testing.assert_allclose(np.diff(old_target,axis=1), np.diff(new_target,axis=1), atol=1e-6, rtol=0.)
    measured = trace['first_torque_nm'][575:, 3::4]
    np.testing.assert_allclose(measured, np.clip(.5 * (old_target - wheels), -12., 12.), atol=2e-9, rtol=0.)
    projected = np.clip(.5 * (new_target - wheels), -12., 12.)
    return dict(samples=int(wheels.size), reverse_targets=int((old_target < 0.).sum()),
                positive_shaft_power_samples=int((wheels * measured > 0.).sum()),
                mean_measured_first_servo_torque_nm=float(measured.mean()),
                mean_projected_torque_at_same_measured_speeds_nm=float(projected.mean()))


def audit():
    require(digest(INDEX) == INDEX_SHA, 'different development archive')
    cases = []
    for row in read(INDEX)['cases']:
        trace = INDEX.parent / f"native_{row['seed']}/trace.npz"
        require(row['stage_status'] == 'COMPLETE' and row['complete_horizon'] and digest(trace) == row['trace_sha256'],
                'unsealed or incomplete source')
        with np.load(trace, allow_pickle=False) as archive:
            speed = speed_readback(archive, row['motion']['gate_times_s'])
            parking = parking_readback(archive)
        cases.append(dict(seed=row['seed'], speed=speed, parking=parking))
    return dict(cases=cases, new_native_episodes=0, policy_forwards=0, policy_updates=0, world_updates=0)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
