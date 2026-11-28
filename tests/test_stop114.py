import unittest
from collections import defaultdict

import numpy as np

from tools114.control import StopProjection


class TestStopProjection(unittest.TestCase):
    def test_first_stopped_tick_removes_only_wheel_common_mode(self):
        control = StopProjection()
        original = np.arange(16, dtype=np.float32) / 8
        original[3::4] = [.75, .25, -.25, .25]
        running = control.apply((1.5, 0., 0.), original)
        np.testing.assert_array_equal(running, original)

        stopped = control.apply((0., 0., 0.), original)
        expected = original.copy()
        expected[3::4] = [.5, 0., -.5, 0.]
        np.testing.assert_array_equal(stopped, expected)
        np.testing.assert_array_equal(original[3::4], [.75, .25, -.25, .25])
        self.assertTrue(control.active)
        self.assertEqual(control.removed_rad_s, 1.25)

    def test_all_driver_axes_rearm_and_release_on_the_first_tick(self):
        control = StopProjection()
        targets = np.ones(16, dtype=np.float32)
        commands = [
            ((0., 0., 0.), False), ((0., 0., 0.), False),
            ((-1., 0., 0.), False), ((0., 0., 0.), False),
            ((0., 0., .5), False), ((0., 0., 0.), False),
            ((1.e-9, 0., 0.), False), ((0., 0., 0.), True), ((0., 0., 0.), True),
            ((0., .01, 0.), False), ((0., 0., 0.), False),
            ((1., 0., 0.), False), ((0., 0., 0.), True),
            ((0., 0., .01), False), ((0., 0., 0.), False),
            ((1., 0., 0.), False), ((0., 0., 0.), True), ((1., 0., 0.), False),
        ]
        for tick, (driver, active) in enumerate(commands):
            with self.subTest(tick=tick):
                issued = control.apply(driver, targets)
                self.assertEqual(control.active, active)
                self.assertEqual(control.removed_rad_s, 5. if active else 0.)
                np.testing.assert_array_equal(issued[3::4], np.full(4, 0. if active else 1.))

    def test_wheel_differences_survive_even_outside_the_unit_interval(self):
        control = StopProjection()
        targets = np.zeros(16, dtype=np.float32)
        targets[3::4] = [-.75, -.75, -.75, .75]
        control.apply((1., 0., 0.), targets)
        stopped = control.apply((0., 0., 0.), targets)
        np.testing.assert_array_equal(stopped[3::4], [-.375, -.375, -.375, 1.125])
        self.assertEqual(control.removed_rad_s, -1.875)

    def test_archived_following_command_is_zero_despite_nonzero_actor_trim(self):
        from tools110.course import command as driver_vx
        from tools114.common import baseline

        path, _ = baseline(111011)
        control = StopProjection()
        with np.load(path, allow_pickle=False) as z:
            raw = z["raw"][524:576]
            unprojected = z["issued"][524:576]
            np.testing.assert_array_equal(raw, unprojected)
            self.assertGreater(z["actor_vx"][575], .04)
            self.assertEqual(driver_vx(575), 0.)

        seen = []
        removed = []
        for tick, original in enumerate(unprojected, start=524):
            issued = control.apply((driver_vx(tick), 0., 0.), original)
            seen.append(control.active)
            removed.append(control.removed_rad_s)
            if tick == 524:
                np.testing.assert_array_equal(issued, original)
            else:
                self.assertLess(abs(np.mean(issued[3::4], dtype=np.float64)), 3e-8)
        self.assertEqual(seen, [False] + [True] * 51)
        self.assertAlmostEqual(removed[1], -1.1043328419327736)
        self.assertAlmostEqual(removed[-1], -.9863163717091084)
        np.testing.assert_array_equal(raw, unprojected)


def command(tick):
    return (1., 0., 0., 1., 1.)[tick]


class LegGeometry:
    def __init__(self, model):
        pass

    def raised_targets(self, raw, lift):
        return raw.copy(), lift


def sample_drive(actor, rows):
    """Exercise the same driver, network and action-queue interfaces without a physics engine."""
    geometry = LegGeometry(None)
    last_raw = last_issued = np.zeros(16, np.float32)
    history = np.zeros((10, 57), np.float32)
    for tick in range(5):
        frame = np.zeros(57, np.float32)
        frame[6] = 2. * command(tick)
        frame[-16:] = last_raw
        last_raw = actor.run(["nn_output"], {"nn_input0": frame[None], "nn_input1": history[None]})[0][0]
        issued, _ = geometry.raised_targets(last_raw, np.zeros(4))
        rows["raw"].append(last_raw)
        rows["issued"].append(issued)
        rows["applied"].append(last_issued)
        last_issued = issued
        history[:-1], history[-1] = history[1:].copy(), frame


class SequenceVendor:
    def __init__(self):
        self.next_value = 0.

    def run(self, outputs, inputs):
        self.next_value += .25
        return [np.full((1, 16), self.next_value, dtype=np.float32)]


class TestStopExecution(unittest.TestCase):
    def test_vendor_history_keeps_raw_while_the_servo_receives_delayed_projection(self):
        from tools114.native import RecordedVendor, with_parking

        rows = defaultdict(list)
        bound = with_parking(sample_drive, rows)
        bound(RecordedVendor(SequenceVendor(), rows), rows)
        np.testing.assert_array_equal(rows["parking_active"], [False, True, True, False, False])
        np.testing.assert_array_equal(np.asarray(rows["issued"])[:, 3], [.25, 0., 0., 1., 1.25])
        np.testing.assert_array_equal(np.asarray(rows["applied"])[:, 3], [0., .25, 0., 0., 1.])
        np.testing.assert_array_equal(np.asarray(rows["actor_previous_raw"])[:, 3], [0., .25, .5, .75, 1.])
        np.testing.assert_array_equal(np.asarray(rows["actor_history_previous_raw"])[4, -3:, 3], [.25, .5, .75])
        originals = np.repeat(np.array([[.25], [.5], [.75], [1.], [1.25]]), 16, axis=1)
        np.testing.assert_array_equal(rows["unprojected_issued"], originals)
        np.testing.assert_array_equal(rows["raw"], originals)

        original_rows = defaultdict(list)
        sample_drive(SequenceVendor(), original_rows)
        np.testing.assert_array_equal(original_rows["issued"], originals)
