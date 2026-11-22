"""Which terrain changes are observable in the existing left-minus-right target?"""

import numpy as np
import pytest

from tools100.score100 import estimate_delta_mm


@pytest.mark.parametrize(
    "left_mm, right_mm, expected_delta_mm",
    [(0, 0, 0), (30, 30, 0), (150, 150, 0), (250, 250, 0),
     (30, 0, 30), (0, 30, -30)],
)
def test_lateral_target_on_ideal_returns(left_mm, right_mm, expected_delta_mm):
    # The robot is on level ground at the near section. Both tracks have a return
    # at the far section, so any loss of terrain information is due to the target.
    near = np.array([[0.25, 0.22, 0.0], [0.25, -0.22, 0.0]])
    far = np.array([[0.50, 0.22, left_mm / 1000], [0.50, -0.22, right_mm / 1000]])
    returns = np.vstack((near, far))

    assert estimate_delta_mm(returns, far[:, :2]) == pytest.approx(expected_delta_mm)
    np.testing.assert_allclose(1000 * (far[:, 2] - near[:, 2]), [left_mm, right_mm])
