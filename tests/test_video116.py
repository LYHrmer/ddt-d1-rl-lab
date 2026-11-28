import unittest
from types import SimpleNamespace

import numpy as np

from tools116.video import cameras, caption, timeline


class TestCompoundVideo(unittest.TestCase):
    def test_shared_clock_holds_both_measured_contacts_stop_and_early_terminal(self):
        end = [600,305]
        events = [dict(crest=230,ripple=270,front_contact=294,rear_clear=336),
                  dict(crest=215,ripple=260,front_contact=284,rear_clear=None)]
        shots = timeline(end,events)
        ticks = [s["tick"] for s in shots]
        self.assertEqual(sorted(set(ticks)),ticks)
        for k in (284,294,305,525,575,600):
            s, = [s for s in shots if s["tick"] == k]
            self.assertGreaterEqual(s["frames"],25)
        self.assertLess(ticks[0],230)
        self.assertIn("slope",{s["phase"] for s in shots})
        self.assertIn("tail",{s["phase"] for s in shots})
        self.assertEqual(sum(s["frames"] for s in shots[:-1]),shots[-1]["first_frame"])
        self.assertLess(sum(s["frames"] for s in shots)/25,35.)
        tracks = []
        for i,n in enumerate(end):
            pose = np.zeros((n+1,23)); pose[:,0] = np.arange(n+1)*.01+i*.1
            tracks.append(SimpleNamespace(pose=pose,last=n,end_reason="跌倒终止" if i else "记录结束"))
        view = cameras(tracks,shots,[dict(xy_m=[5.7,.1]),None])
        self.assertEqual(view["step"][0],view["step"][1])
        self.assertIsNone(view["tail"][1])
        message = caption(tracks[1],events[1],305,"step",0.,None)
        self.assertIn("跌倒终止",message)

    def test_missing_contact_is_never_captioned_as_a_crossing(self):
        track = SimpleNamespace(last=600,end_reason="记录结束",pose=np.zeros((601,23)))
        track.pose[225,0] = 5.4
        events = dict(crest=None,ripple=None,front_contact=None,rear_clear=None)
        self.assertEqual(caption(track,events,225,"step",0.,None),"台阶区域")
