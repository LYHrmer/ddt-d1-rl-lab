import unittest

import numpy as np

from tools113.model import ACTIONS, STATES, GridWorld, loss_gradient, physical_targets, state_vector


class SpeedWorld(unittest.TestCase):
    def test_servo_units_and_local_heading(self):
        q = np.zeros(23)
        v = np.zeros(22)
        q[2], q[3], q[6] = .39, np.sqrt(.5), np.sqrt(.5)
        q[7:] = np.arange(16)
        v[:2] = [0., 1.5]
        v[6:] = np.arange(16) - 2.
        s = state_vector(q, v)
        self.assertEqual(s.shape, (37,))
        np.testing.assert_allclose(s[:6],[.39, 0., 0., 1.5, 0., 0.], atol=1e-14)
        np.testing.assert_array_equal(s[9:21],[0,1,2,4,5,6,8,9,10,12,13,14])
        np.testing.assert_array_equal(s[21:], v[6:])
        targets = physical_targets(np.full(16,4.))
        np.testing.assert_allclose(targets[:4],[.6,1.8,-.5,20.], atol=1e-14)

    def test_rollout_derivative_matches_central_differences(self):
        random = np.random.default_rng(43)
        w = random.normal(0.,.1,(STATES+ACTIONS,STATES))
        b = random.normal(0.,.1,STATES)
        x = random.normal(size=(3,5,STATES))
        u = random.normal(size=(3,4,ACTIONS))
        _, dw, db = loss_gradient(w,b,x,u)
        for p,g,idx in ((w,dw,(3,3)), (w,dw,(40,7)), (w,dw,(51,3)), (b,db,3)):
            original = p[idx]
            eps = 1e-5
            p[idx] = original+eps
            plus = loss_gradient(w,b,x,u)[0]
            p[idx] = original-eps
            minus = loss_gradient(w,b,x,u)[0]
            p[idx] = original
            self.assertAlmostEqual(g[idx],(plus-minus)/(2.*eps),places=8)

    def test_masked_world_does_not_respond_to_real_targets(self):
        w = np.zeros((STATES+ACTIONS,STATES))
        w[STATES+3,3] = 1.
        m = GridWorld(np.zeros((601,STATES)), np.zeros((600,ACTIONS)),
                      np.ones(STATES), np.ones(ACTIONS), w, np.zeros(STATES), True, 111011)
        s, u = np.zeros(STATES), np.zeros(ACTIONS)
        u[3] = .3
        self.assertGreater(m.step(3,s,u)[3],0.)
        m.uses_action = False
        np.testing.assert_array_equal(m.step(3,s,u),m.step(3,s,-u))
