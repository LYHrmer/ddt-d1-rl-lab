"""Verify the complete measured grid and the correspondence between displayed and physical time."""
import json

import numpy as np
import pytest

from tools93 import video93 as movie


def _save(path, value):
    path.write_text(json.dumps(value))


def test_source_ticks_at_half_speed_and_endpoint_dwell():
    frames = movie.frame_counts()
    assert movie.FPS == 25 and movie.DT*movie.FPS == .5
    assert frames == dict(intro=75, movement_per_pair=50, endpoint_per_pair=32, summary=125, total=938)
    for i in range(9):
        begin = frames["intro"] + i*(movie.TICKS + frames["endpoint_per_pair"])
        held = begin + frames["movement_per_pair"]
        assert (held-begin)/movie.FPS == 2 and movie.TICKS*movie.DT == 1
        assert frames["endpoint_per_pair"]/movie.FPS == 1.28
    assert (begin + 50 + 32 + frames["summary"])/movie.FPS == frames["total"]/movie.FPS == 37.52


@pytest.fixture
def panel(tmp_path):
    old, fresh = tmp_path / "90", tmp_path / "93"
    old.mkdir()
    fresh.mkdir()
    old_seal = dict(status="COMPLETE", cases=[])
    new_seal = dict(status="COMPLETE", protocol_sha256=movie.PLAN_SHA, cases=[])
    score = dict(protocol_sha256=movie.PLAN_SHA, cells=[])
    plan = dict(policies_sha256={str(p): f"{p:064x}" for p in movie.PARENTS})
    for i, p in enumerate(movie.PARENTS):
        for j, r in enumerate(movie.RESETS):
            folder, seal, n = (old, old_seal, 100) if i == j else (fresh, new_seal, 50)
            a = .10 + i*.01 + j*.001
            # Make one intervention increase rollback, so sign and selection get tested as well.
            b = a + .005 if (i, j) == (1, 0) else a - .008
            for arm, distance in zip(movie.ARMS, (a, b), strict=True):
                path = folder / f"{p}_{r}_{arm}.npz"
                start = np.zeros(23)
                start[[0, 2, 3, 7]] = [1.10, .45, 1., j/8.]
                q = np.tile(start, (n, 1))
                t = np.arange(1, n+1)*.02
                q[:, 0] -= distance*t
                if n == 100:
                    q[50:, 0] = 2.  # Only the first second of archived physics enters this film.
                np.savez(path, qpos=q, qvel=np.zeros((n, 22)), time_s=t, command=np.zeros((n, 3)),
                         initial_qpos=start, initial_qvel=np.zeros(22))
                seal["cases"].append(dict(parent_seed=p, reset_seed=r, arm=arm, status="COMPLETE", clean=True,
                                          ticks=n, policy_sha256=plan["policies_sha256"][str(p)],
                                          trace=path.name, trace_sha256=movie.digest(path), rollback_m=distance))
            score["cells"].append(dict(parent_seed=p, reset_seed=r, truth=[a, b], true_effect_m=a-b))
    old_seal["cases"].reverse()
    new_seal["cases"].reverse()
    score["cells"].reverse()
    _save(old / "report.json", old_seal)
    _save(fresh / "seal.json", new_seal)
    return plan, score, old, fresh


def test_nine_pairs_use_measured_states_and_both_intervention_signs(panel):
    pairs = movie.load_pairs(*panel)
    assert [(p.parent, p.reset) for p in pairs] == [(p, r) for p in movie.PARENTS for r in movie.RESETS]
    assert [p.effect_mm(50) for p in pairs] == pytest.approx([8., 8., 8., -5., 8., 8., 8., 8., 8.])
    for p in pairs:
        for a in p.arms:
            assert a.q.shape == (51, 23) and a.v.shape == (51, 22)
            assert a.rollback_mm[0] == 0 and a.rollback_mm[-1] > 0
            assert a.path.parent == (panel[2] if movie.PARENTS.index(p.parent) == movie.RESETS.index(p.reset) else panel[3])


@pytest.mark.parametrize("case", ["duplicate", "missing", "score", "partial", "hash"])
def test_refuse_wrong_matrix_altered_score_or_unverified_source(panel, case):
    _, score, _, fresh = panel
    seal = movie.read(fresh / "seal.json")
    if case == "duplicate":
        seal["cases"].append(seal["cases"][0])
    elif case == "missing":
        seal["cases"].pop()
    elif case == "score":
        score["cells"][0]["true_effect_m"] += .001
    elif case == "partial":
        seal["cases"][0]["status"] = "TERMINATED"
    else:
        seal["cases"][0]["trace_sha256"] = "0"*64
    _save(fresh / "seal.json", seal)
    with pytest.raises(ValueError):
        movie.load_pairs(*panel)


def test_existing_video_stage_is_preserved(tmp_path):
    with pytest.raises(FileExistsError):
        movie.run(tmp_path)
