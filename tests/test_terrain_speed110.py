import unittest

import numpy as np

from tools110.course import assess, command, height, next_gain


def straight(v):
    time = np.arange(601)/50.
    speed = np.array([command(k)*v/1.5 for k in range(601)])
    x = np.r_[0., np.cumsum((speed[1:]+speed[:-1])/100.)]
    xy = np.column_stack((x, np.zeros_like(x)))
    offsets = np.array([[.45,.25],[.45,-.25],[-.15,.25],[-.15,-.25]])
    wheels = xy[:,None,:]+offsets
    return dict(time=time, xy=xy, vxy=np.column_stack((speed, np.zeros_like(speed))), wheels_xy=wheels,
                fallen=np.zeros(600,bool), illegal=np.zeros(600,bool),
                exit_contact=wheels[1:,:,0]>7., top_contact=(wheels[1:,:,0]>4.01)&(wheels[1:,:,0]<5.49))


class TerrainScore(unittest.TestCase):
    def test_geometry(self):
        self.assertAlmostEqual(float(height("ridge",4.5,0)),2.5*np.tan(np.deg2rad(4.)))
        self.assertEqual(float(height("step",4.3,.4)),.03)
        self.assertEqual(float(height("step",4.3,1.5)),0.)
        x=np.linspace(3.,5.,200)
        np.testing.assert_allclose(height("ripple",x,.3)+height("ripple",x,-.3),.025)
        self.assertGreater(np.ptp(height("ripple",x,.3)),.0249)

    def test_inside_terrain_and_all_four_wheels(self):
        z=straight(1.53)
        self.assertTrue(assess(z,"step")["passed"])
        z["top_contact"][:,3]=False
        self.assertEqual(assess(z,"step")["failed"],["step_top_support"])
        self.assertTrue(assess(z,"ridge")["passed"])
        z["exit_contact"][:,1]=False
        self.assertIn("all_wheels_exited",assess(z,"ridge")["failed"])
        z["wheels_xy"][210,1,1]=1.2
        z["vxy"][-5,1]=.1
        failed=assess(z,"ridge")["failed"]
        self.assertIn("lane",failed)
        self.assertIn("stopped",failed)

    def test_calibration_requires_paired_safe_systematic_bias(self):
        rows=[{"score":assess(straight(v),"flat")} for v in (1.34,1.35,1.36)]
        self.assertAlmostEqual(next_gain(rows),1.5375/1.35)
        rows[0]["score"]["failed"].append("illegal_support")
        self.assertIsNone(next_gain(rows))
        rows[0]={"score":assess(straight(1.65),"flat")}
        self.assertIsNone(next_gain(rows))
        self.assertEqual(next_gain([{"score":assess(straight(1.53),"flat")} for _ in range(3)]),1.)


if __name__=="__main__":
    unittest.main()
