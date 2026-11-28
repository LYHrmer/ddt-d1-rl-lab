import unittest
from types import SimpleNamespace

from tools117.parking_video import END, FPS, STOP_TICK, ZERO, shots_for


class TestCompoundParkingFilm(unittest.TestCase):
    def test_driver_release_tail_closeup_and_last_pose_are_held_on_one_clock(self):
        shots = shots_for([SimpleNamespace(last=END), SimpleNamespace(last=END)])
        self.assertEqual(shots[0]["tick"], 500)
        self.assertEqual(shots[-1]["tick"], END)
        self.assertEqual([s["tick"] for s in shots], sorted(s["tick"] for s in shots))
        hold = next(s for s in shots if s["tick"] == ZERO)
        self.assertGreaterEqual(hold["frames"], FPS)
        self.assertEqual(hold["phase"], "overview")
        close = [s for s in shots if s["phase"] == "front_axle"]
        self.assertEqual((close[0]["tick"], close[-1]["tick"]), (STOP_TICK, END))
        self.assertGreaterEqual(close[0]["frames"], FPS)
        self.assertGreaterEqual(close[-1]["frames"], 2*FPS)
        self.assertEqual(close[-1]["first_frame"], sum(s["frames"] for s in shots[:-1]))
