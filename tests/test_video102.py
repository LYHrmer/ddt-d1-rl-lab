import numpy as np
import pytest

from tools102.mask102 import CAPTURE_TICKS
from tools102.video102 import FPS, FRAMES_PER_TICK, clocks


def test_slow_replay_holds_only_delivered_packets_and_retains_every_native_pose():
    deliveries = np.array(CAPTURE_TICKS) + 2
    schedule = list(clocks(deliveries))
    assert [tick for tick, _, _ in schedule] == list(range(27, 153))
    assert all(deliveries[packet] <= tick and (packet == 15 or tick < deliveries[packet+1])
               for tick, packet, _ in schedule)
    events = [(tick, packet, hold//FPS) for tick, packet, hold in schedule if hold]
    assert events == [(27, 0, 4), (77, 6, 7), (86, 7, 7), (94, 8, 6), (152, 15, 4)]
    assert sum(FRAMES_PER_TICK + hold for _, _, hold in schedule) == 1064
    assert (1/50) / (FRAMES_PER_TICK/FPS) == pytest.approx(.1)
    with pytest.raises(ValueError, match="clocks"):
        list(clocks(np.array(CAPTURE_TICKS)))
