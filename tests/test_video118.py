import unittest

from tools118.video import END, FPS, STOP_TICK, caption, timeline


class TestBlindFilm(unittest.TestCase):
    def test_complete_course_holds_contact_crossing_and_final_measured_pose(self):
        gates = [4.39, 5.06, 5.73, 6.36]
        shots = timeline(END, gates, 284)
        ticks = [s["tick"] for s in shots]
        self.assertEqual(ticks, sorted(set(ticks)))
        self.assertTrue({220, 253, 284, 287, 318, 525, STOP_TICK, END} <= set(ticks))
        self.assertTrue(all(next(s["frames"] for s in shots if s["tick"] == k) >= FPS for k in (284, 318, 525, 575)))
        self.assertGreaterEqual(shots[-1]["frames"], 2*FPS)
        self.assertEqual(shots[-1]["first_frame"], sum(s["frames"] for s in shots[:-1]))
        self.assertLess(sum(s["frames"] for s in shots)/FPS, 30)
        info = dict(last_tick=END, end_reason="记录结束", gate_times_s=gates, contact_tick=284,
                    speed_m_s=1.5209, sections_m_s=[1.5, 1.51, 1.56])
        self.assertNotIn("均速", caption(info, 317, "step"))
        self.assertIn("1.5209", caption(info, 318, "step"))
        self.assertIn("有载接触", caption(info, 284, "step"))

    def test_early_terminal_before_first_gate_has_no_stop_or_future_pose(self):
        shots = timeline(122, [], None)
        self.assertEqual(shots[-1]["tick"], 122)
        self.assertGreaterEqual(shots[-1]["frames"], 2*FPS)
        self.assertEqual({s["phase"] for s in shots}, {"cruise"})
        info = dict(last_tick=122, end_reason="非法接触终止", gate_times_s=[], contact_tick=None, stop=None)
        self.assertIn("非法接触终止", caption(info, 122, "cruise"))
