import numpy as np
import pytest

from tools108.video108 import FPS, first_contact, timeline


def test_replay_clock_uses_front_wheel_force_on_the_step_at_the_substep_end():
    force = np.zeros((30, 4))
    force[0, 2:] = 100
    force[1, :2] = [.99, .99]
    assert first_contact(force) == (None, None)
    force[10, 1] = 1
    assert first_contact(force) == (.022, 2)
    force[9, 0] = 2
    assert first_contact(force) == (.020, 1)
    with pytest.raises(ValueError):
        first_contact(force[:-1])


def test_slow_replay_retains_every_pose_and_marks_contact_and_endpoint_holds():
    shots = list(timeline(60, [20, 22]))
    running = [s for s in shots if not s.hold]
    assert [s.tick for s in running] == list(range(61))
    assert running[19].frames == 2 and running[19].speed == .25
    assert running[59].frames == 1 and running[59].speed == .5
    assert [(s.tick, s.frames) for s in shots if s.hold] == [(0, 12), (20, FPS), (60, FPS)]
    assert [s.tick for s in timeline(60, [20, 30]) if s.hold] == [0, 20, 30, 60]
    assert [s.tick for s in timeline(10, [None, None]) if s.hold] == [0, 10]
    assert len(list(timeline(0, [None, None]))) == 1
    with pytest.raises(ValueError):
        list(timeline(60, [61, None]))
