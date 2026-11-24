from math import sqrt

import numpy as np
import pytest

from tools106.sightline106 import box_sightline, mount_origins


LOWER = (0., 0., 0.)
UPPER = (1., 1., .25)


def test_flat_topped_box_intercepts_diagonal_ground_sightline():
    # The line enters the roof at x=.75 and leaves the rear face at x=1.
    origin = (-.25, .5, .75)
    target = (1.25, .5, 0.)
    state, length = box_sightline(origin, target, LOWER, UPPER)
    assert state == "blocked"
    assert length == pytest.approx(sqrt(.25**2 + .125**2))
    assert box_sightline(target, origin, LOWER, UPPER) == (state, length)


@pytest.mark.parametrize(("origin", "target", "expected"), [
    ((-.25, -.125, .75), (1.25, -.125, 0.), "clear"),       # passes beside the fixture
    ((-.25, .5, .75), (.25, .5, .5), "clear"),              # infinite ray could hit after the endpoint
    ((-1., 1., .125), (1., -1., .125), "tangent"),          # touches only one vertical edge
    ((-1., .5, .25), (2., .5, .25), "tangent"),             # follows the roof
    ((-1., .5, .5), (0., .5, .25), "tangent"),             # touches only at the final endpoint
    ((1., .5, .125), (2., .5, .125), "tangent"),            # starts on the rear face and moves away
])
def test_clear_and_zero_length_interior_contacts(origin, target, expected):
    assert box_sightline(origin, target, LOWER, UPPER) == (expected, 0.)


def test_ground_endpoint_on_the_base_still_crosses_positive_interior():
    assert box_sightline((.5, .5, .75), (.5, .5, 0.), LOWER, UPPER) == ("blocked", .25)


def test_mounts_use_full_roll_pitch_and_parent_translation():
    h = sqrt(.5)
    # Pitch the base by 90 degrees. Its forward lever arm now points down.
    pose = np.array([4., 2., 3., h, 0., h, 0.])
    mounts = [[.3, .2, .05], [.3, -.2, .05]]
    np.testing.assert_allclose(mount_origins(pose, mounts), [[4.05, 2.2, 2.7], [4.05, 1.8, 2.7]], atol=1e-12)
