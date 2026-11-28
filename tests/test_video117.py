import unittest
from types import SimpleNamespace

from tools117.video import FPS, caption, timeline


class TestStepReplay(unittest.TestCase):
    def test_paired_clock_and_hold_after_both_cars_cross(self):
        readings = [dict(entry_s=5.725, exit_s=6.358, section_m_s=1.58, contact_tick=284),
                    dict(entry_s=5.725, exit_s=6.361, section_m_s=1.57, contact_tick=284)]
        shots = timeline([600, 600], readings)
        ticks = [s["tick"] for s in shots]
        self.assertEqual(sorted(set(ticks)), ticks)
        self.assertTrue({284, 287, 318, 319, 575, 600} <= set(ticks))
        joint, = [s for s in shots if s["tick"] == 319]
        self.assertGreaterEqual(joint["frames"], 2 * FPS)
        self.assertEqual(sum(s["frames"] for s in shots[:-1]), shots[-1]["first_frame"])
        self.assertLess(sum(s["frames"] for s in shots) / FPS, 25)
        track = SimpleNamespace(last=600, end_reason="记录结束")
        self.assertIn("1.5800 m/s", caption(track, readings[0], 318, "step", None))
        self.assertIn("已用时", caption(track, readings[1], 318, "step", None))
        self.assertIn("1.5700 m/s", caption(track, readings[1], 319, "step", None))

    def test_terminal_before_gates_is_shown_with_its_measured_time(self):
        readings = [dict(entry_s=5.725, exit_s=6.358, section_m_s=1.58, contact_tick=284),
                    dict(entry_s=None, exit_s=None, section_m_s=None, contact_tick=None)]
        shots = timeline([600, 205], readings)
        terminal, = [s for s in shots if s["tick"] == 205]
        self.assertGreaterEqual(terminal["frames"], FPS)
        track = SimpleNamespace(last=205, end_reason="跌倒终止")
        self.assertEqual(caption(track, readings[1], 600, "tail", None), "跌倒终止于 4.10 秒")
