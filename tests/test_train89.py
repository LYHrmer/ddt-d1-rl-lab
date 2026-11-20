"""Check that uphill exposure counts real simultaneous support and parent Adam slots survive."""
import io
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from tools89.train89 import coverage, inherit_state


def test_coverage_keeps_all_commanded_uphill_ticks_and_partial_episodes():
    support = np.array([
        [1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1],
        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1],
    ], dtype=bool)
    rows = dict(terrain=[1, 1, 1, 2, 2, 1, 1, 1], command_vx_m_s=[0., .2, .2, 0., .2, 0., .2, .2],
                loaded_wheels_past_toe=support, episode_tick=[1, 2, 3, 1, 2, 1, 2, 3],
                start_x_m=[1.1, 1.1, 1.1, 0, 0, 0, 0, 0], slope_start=[1, 1, 1, 0, 0, 0, 0, 0],
                base_origin_past_toe=np.array([1, 1, 1, 0, 0, 0, 0, 1], bool),
                truncated=[0, 0, 1, 0, 1, 0, 0, 0], terminated=[0]*8, fallen=[0]*8, illegal_contact=[0]*8)
    result = coverage(rows)
    assert result["uphill_moving_ticks"] == 4
    assert result["uphill_moving_four_loaded_ticks"] == 2
    assert result["uphill_moving_four_loaded_fraction"] == .5
    assert result["uphill_moving_both_axles_loaded_ticks"] == 3
    assert result["uphill_episodes"][0]["truncated"]
    assert not result["uphill_episodes"][1]["truncated"]
    rows["loaded_wheels_past_toe"][4] = True
    with pytest.raises(ValueError, match="terrain support"):
        coverage(rows)


def test_same_architecture_inherits_parent_weights_and_each_adam_slot(tmp_path):
    def network():
        model = torch.nn.Linear(3, 2)
        model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        return model

    source, dest = network(), network()
    for index, p in enumerate(source.parameters(), start=1):
        source.optimizer.state[p] = {"step": torch.tensor(200. + index),
                                     "exp_avg": torch.full_like(p, index / 10),
                                     "exp_avg_sq": torch.full_like(p, index / 20)}
    path = tmp_path / "policy.zip"
    with zipfile.ZipFile(path, "x") as archive:
        for name, state in (("policy.pth", source.state_dict()), ("policy.optimizer.pth", source.optimizer.state_dict())):
            stream = io.BytesIO()
            torch.save(state, stream)
            archive.writestr(name, stream.getvalue())
    checked = inherit_state(dest, path)
    assert checked["inherited_adam_steps"] == {"weight": 201, "bias": 202}
    for a, b in zip(source.parameters(), dest.parameters(), strict=True):
        assert torch.equal(a, b)
        for key, slot in source.optimizer.state[a].items():
            assert torch.equal(slot, dest.optimizer.state[b][key])
