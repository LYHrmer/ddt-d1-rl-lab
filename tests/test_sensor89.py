import numpy as np
import pytest

from tools89.sensor89 import stats


def test_signed_and_absolute_target_responses_keep_all_paired_traces_and_wheels():
    delta = np.ones((3, 100, 4))
    delta[0, :, 0] = -3
    delta[2] = -delta[1]
    result = stats(delta)
    assert result["signed_mean_rad_s"] == 0.
    assert result["mean_abs_rad_s"] == pytest.approx(7/6)
    assert result["per_wheel_signed_mean_rad_s"] == pytest.approx([-1., 1/3, 1/3, 1/3])
    assert result["reference_signed_means_rad_s"] == [0., 1., -1.]
    with pytest.raises(ValueError):
        stats(delta[:, :-1])
