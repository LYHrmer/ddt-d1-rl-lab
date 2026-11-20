import json

import numpy as np
import pytest

from tools89.common89 import PLAN, read, sha
from tools89.video89 import ARMS, paired_sources


def record(folder, arm, plan, *, base_x=0.):
    train, seed = plan["training"]["seeds"][0], plan["evaluation"]["reset_seeds"][0]
    case = f"{arm}_{train}_up_{seed}"
    path = folder / case
    path.mkdir()
    x = np.array([np.nan, .58, .61, .62])
    q = np.zeros((4, 23))
    q[:, 0] = base_x
    np.savez(path / "trace.npz", time_s=np.arange(4)*.02, qpos=q, qvel=np.zeros((4, 22)),
             body_vx_m_s=np.zeros(4), command_vx_m_s=np.zeros(4), raw_ranges16=np.zeros((4, 16)),
             front_contact_x_m=x)
    meta = {"status": "COMPLETE", "arm": arm, "seed": seed, "training_seed": train, "terrain": "up",
            "case": case, "reached_mark": False, "full_horizon": False, "terminated": True,
            "protocol_sha256": sha(PLAN), "trace_sha256": sha(path / "trace.npz"),
            "max_loaded_front_contact_past_toe_m": float(np.nanmax(x) - plan["curriculum"]["toe_x_m"])}
    (path / "metrics.json").write_text(json.dumps(meta))
    return meta


def test_video_selects_fixed_curriculum_pair_even_if_it_terminated(tmp_path):
    plan = read(PLAN)
    rows = [record(tmp_path, arm, plan) for arm in ARMS]
    traces, contacts = paired_sources(tmp_path, {"status": "COMPLETE", "protocol_sha256": sha(PLAN), "cases": rows}, plan)
    assert [r.meta["arm"] for r in traces] == list(ARMS)
    assert all(r.meta["terminated"] for r in traces)
    assert all(np.isnan(x[0]) and np.isclose(x[-1], .02) for x in contacts)


def test_video_rejects_moving_candidate_start_in_heldout_replay(tmp_path):
    plan = read(PLAN)
    rows = [record(tmp_path, "original", plan), record(tmp_path, "mixed", plan, base_x=1.1)]
    with pytest.raises(ValueError, match="same pose"):
        paired_sources(tmp_path, {"status": "COMPLETE", "protocol_sha256": sha(PLAN), "cases": rows}, plan)
