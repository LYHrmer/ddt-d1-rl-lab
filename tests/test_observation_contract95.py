import numpy as np
import pytest

from tools95.check_privileged_observation95 import EXAMPLE, audit_trace, packing_check


def test_public_trace_reconstructs_actor_frames_and_delayed_packets():
    result = audit_trace(EXAMPLE)
    assert (result["decisions"], result["endpoints"]) == (32, 33)
    assert result["capture_ticks"] == [0, 9, 17, 25]
    assert result["delivery_ticks"] == [2, 11, 19, 27]
    assert result["valid_capture_beams"] == 32 and result["invalid_capture_beams"] == 0
    assert packing_check() == {"synthetic_beams": 8, "valid": 3, "invalid": 5}


@pytest.mark.parametrize("fault,reason", [("history", "history"), ("future_packet", "visible packet"),
                                         ("delivery_before_capture_delay", "delivery clock")])
def test_audit_detects_mutated_observation_and_delivery_clock(tmp_path, fault, reason):
    with np.load(EXAMPLE, allow_pickle=False) as source:
        trace = dict(source)
    if fault == "history":
        # Replace the oldest frame with the current frame at a tick where they differ.
        trace["obs643"][12, 57:114] = trace["obs643"][12, :57]
    elif fault == "future_packet":
        trace["obs643"][1, 627:] = trace["ray_raw16"][0]
    else:
        trace["sensor_delivery_tick"][2] = 1
    path = tmp_path / "altered.npz"
    np.savez(path, **trace)
    with pytest.raises(ValueError, match=reason):
        audit_trace(path)
