import numpy as np

from tools94.common94 import read
from tools104.motion104 import PUBLIC, capture_query, measured_motion, reference_motion
from tools104.support104 import evidence


def test_query_translates_forward_and_turns_with_the_body():
    # A positive quarter-turn after a forward move puts a delivered forward
    # query to the left of the original heading, at the translated origin.
    pose_c = np.array([4., -2., 0.5, 1., 0., 0., 0.])
    pose_d = np.array([4.01, -2., 0.8, np.sqrt(0.5), 0., 0., np.sqrt(0.5)])
    motion = reference_motion([pose_c, pose_d])
    query = np.array([[0.35, 0.2], [0.35, -0.2]])
    np.testing.assert_allclose(capture_query(query, motion), [[-0.19, 0.35], [0.21, 0.35]], atol=1e-14)


def test_wheel_speed_is_integrated_over_only_the_two_elapsed_ticks():
    up_to_down = np.tile([0., 0., -1.], (3, 1))
    wheels = np.array([[0.] * 4, [1.] * 4, [2.] * 4]) / 0.087
    motion = measured_motion(np.zeros((3, 3)), up_to_down, wheels)
    np.testing.assert_allclose(motion, [0.04, 0., 0.], atol=1e-14)


def test_gyro_recovers_the_heading_rate_on_a_tilted_body():
    up = np.array([0.3, 0.4, np.sqrt(0.75)])
    gravity = np.tile(-up, (3, 1))
    gyro = np.tile(1.5 * up, (3, 1))
    np.testing.assert_allclose(measured_motion(gyro, gravity, np.zeros((3, 4))), [0., 0., 0.06])


def test_surface_audit_replays_the_original_stale_height_and_new_regression():
    result = evidence()
    assert result == read(PUBLIC / "returns.json")
    by_id = {row["row_id"]: row for row in result["failures"]}
    assert by_id[127]["truth_mm"] == 0 and by_id[127]["estimate_mm"] == -30
    assert by_id[27]["original_passed"] and by_id[27]["nearest_surfaces"] == ["front_face"]
