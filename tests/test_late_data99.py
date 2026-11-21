"""Recorded endpoints, training-only access and public late-response labels."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools99 import data99


def archive_arm(path, n=32, terminal=None):
    issued = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None] / 64., (n, 16)).copy()
    applied = np.concatenate([np.zeros_like(issued[:1]), issued[:-1]])
    state = np.zeros((n + 1, 45), dtype=np.float64)
    state[:, 0] = 1.1 + np.arange(n + 1, dtype=np.float64) * 1e-9
    flags = {k: np.zeros(n, dtype=bool) for k in data99.FLAGS}
    if terminal is not None:
        flags["illegal"][terminal - 1] = flags["terminated"][terminal - 1] = True
    np.savez(path, reset_seed=96101, completed_transitions=n, state45=state,
             issued16=issued, applied16=applied, **flags)
    return data99.fit98.digest(path)


def test_final_contact_keeps_observed_state_and_action_delay(tmp_path):
    path = tmp_path / "train.npz"
    checksum = archive_arm(path, terminal=32)
    state, applied, flags = data99.read_full_arm(path, checksum, 96101)
    assert flags == ["illegal", "terminated"] and state[32, 0] == 1.1 + 32e-9
    assert applied[16, 0] == np.float32(15 / 64)
    with pytest.raises(ValueError, match="changed source"):
        data99.read_full_arm(path, "0" * 64, 96101)


@pytest.mark.parametrize(("n", "terminal"), [(31, None), (32, 31)])
def test_missing_or_post_terminal_sample_cannot_fill_h32(tmp_path, n, terminal):
    path = tmp_path / "train.npz"
    checksum = archive_arm(path, n, terminal)
    with pytest.raises(ValueError, match="horizon|contact"):
        data99.read_full_arm(path, checksum, 96101)


def test_all_public_training_labels_without_access_to_blind_arrays():
    path = data99.checked(data99.ROOT / "reports/paired_response96_curves.npz", data99.CURVES96_SHA)
    expected_rms_mm = (4.066153393031237, 3.7894095578864273, 3.525263816956316)
    with np.load(path, allow_pickle=False) as archive:
        for seed, rms in zip(data99.SEEDS, expected_rms_mm, strict=True):
            # Only this seed's six public training arms are visible to the loader.
            curves = {k: archive[k] for k in archive.files if k.startswith(f"collection_{seed}_")}
            target, flagged = data99.public_labels(curves, seed)
            answer = data99.check_labels(seed, target, flagged)
            assert answer["pairs"] == 3 and answer["positive_labels"] == 48
            assert answer["target_rms_mm"] == pytest.approx(rms, abs=1e-11)
            assert [answer[k] for k in ("correct_loss", "zero_loss", "reversed_loss")] == [0., 1., 4.]
            assert flagged == ([{"reset": 96103, "arm": "m1", "tick": 32, "flags": ["illegal", "terminated"]}]
                               if seed == 94011 else [])
