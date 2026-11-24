import numpy as np
import pytest

from tools94.common94 import arrays, digest, read, write
from tools108 import model108 as model


def _obs(speed, knee):
    obs = np.zeros(643, dtype=np.float32)
    obs[[28, 32, 36, 40]] = speed/model.WHEEL_RADIUS_M*.05
    obs[[11, 15]] = knee
    return obs


def test_start_uses_only_observable_encoder_channels():
    obs = _obs(.2, -.1)
    np.testing.assert_allclose(model.start_features(obs), [.2, -.1], atol=1e-7)
    keep = np.zeros(643, dtype=bool)
    keep[[11, 15, 28, 32, 36, 40]] = True
    obs[~keep] = np.random.default_rng(1).normal(size=(~keep).sum())
    np.testing.assert_allclose(model.start_features(obs), [.2, -.1], atol=1e-7)


def test_fixed_waveform_uses_the_fifo_and_has_an_independent_finite_difference_gradient():
    coefficients = np.linspace(-.8, 1., model.PARAMETERS)
    start = np.array([.19, -.13])
    args = coefficients, -.1, .08
    original = model.trajectory_m(start, 0., *args)
    flex = model.trajectory_m(start, 1., *args)
    delta = model.gain_gradient_m(start, *args)
    assert delta[0] == 0 and delta[1] != 0
    np.testing.assert_allclose(flex-original, delta, atol=1e-14)
    eps = 1.e-5
    derivative = (model.trajectory_m(start, .4+eps, *args)-model.trajectory_m(start, .4-eps, *args))/(2*eps)
    np.testing.assert_allclose(derivative, delta, atol=1e-11)


def test_prefix_only_blind_reader_does_not_require_flexion_or_future_rows(tmp_path):
    for j, seed in enumerate(model.BLIND):
        folder = tmp_path / f"seed{seed}_original"
        folder.mkdir()
        calls = np.full((10, 643), np.nan, dtype=np.float32)
        calls[3] = _obs(.1+j*.02, -.2+j*.01)
        # The last frame available prior to substep contact n=38 is decision 3.
        arrays(folder/"trace.npz", dict(trigger_tick=4, actor_observation=calls))
    starts = model.load_blind_starts(tmp_path)
    for j, seed in enumerate(model.BLIND):
        assert starts[seed]["observation_tick"] == 3
        np.testing.assert_allclose(starts[seed]["features"], [.1+j*.02, -.2+j*.01], atol=1e-7)


def _pair(seed, gain_mm, safe=True):
    obs = _obs(.1, -.05)
    t = np.arange(1, model.H+1)
    paths = np.stack([.002*t, .002*t + gain_mm/1000*(t-1)/(model.H-1)])
    return dict(seed=seed, onset=model._onset(seed, 12, obs), safe=[safe, True],
                source_sha256=["synthetic_original", "synthetic_flexion"], paths_m=paths)


def _native_fixture(folder, seed, arm, mismatch=False):
    spec, t, k = model._protocol(), model.TOTAL_TICKS, 10
    destination = folder / f"seed{seed}_{arm}"
    destination.mkdir()
    force = np.zeros((10*t, 4))
    force[99, 0] = 1.
    pulse = np.zeros(t)
    axle = .002*np.arange(t+1)
    if arm == "flexion":
        pulse[k:k+model.PULSE_TICKS] = model.PULSE_RAD
        axle[k+2:] += .0001*np.arange(1,t-k)
    nominal = np.full((t, 16), .17, dtype=np.float32)
    issued = nominal.copy()
    issued[:, [2,6]] += (pulse/.5).astype(np.float32)[:,None]
    applied = np.zeros_like(issued)
    applied[1:] = issued[:-1]
    qpos = np.zeros((t+1, 23))
    qpos[k,0] = float(mismatch)
    path = destination/"trace.npz"
    flags = np.zeros(t, dtype=bool)
    arrays(path, dict(qpos=qpos, qvel=np.zeros((t+1,22)), actor_observation=np.tile(_obs(.21,-.03),(t+1,1)),
                      nominal_action=nominal, issued_action=issued, applied_action=applied,
                      pulse_target_rad=pulse, actual_issue_offset_rad=.5*(issued-nominal)[:,[2,6]],
                      front_axle_x_m=axle, step_force_n=force, trigger_tick=k,
                      fallen=flags, illegal_contact=flags, terminated=flags))
    write(destination/"report.json", dict(status="COMPLETE", full_horizon=True, seed=seed, arm=arm,
           checkpoint_sha256=spec["checkpoint_sha256"], geometry=spec["geometry"], trace_file=path.name,
           trace_sha256=digest(path), trigger_tick=k, first_contact_substep=99))


def test_native_loader_clock_fifo_labels_and_identical_prefix(tmp_path):
    seed = model.TRAIN[0]
    for name, different in (("matched", False), ("mismatched", True)):
        folder = tmp_path/name
        folder.mkdir()
        _native_fixture(folder, seed, "original")
        _native_fixture(folder, seed, "flexion", mismatch=different)
        if different:
            with pytest.raises(ValueError, match="identical native prefix"):
                model.load_pairs(folder, "train")
            continue
        pairs = model.load_pairs(folder, "train")
        paired = pairs[seed]
        assert paired["onset"]["observation_tick"] == 9 and paired["safe"] == [True,True]
        assert paired["paths_m"].shape == (2, model.H)
        np.testing.assert_allclose(np.diff(paired["paths_m"],axis=0)[0], .0001*np.arange(model.H), atol=1e-14)
        assert pairs[model.TRAIN[1]]["reasons"] == {"original":"missing_report", "flexion":"missing_report"}


def _forecast(folder, pairs, gains_mm, mean_mm=1.):
    shape = len(model.MODEL_SEEDS), len(model.BLIND), model.H
    curve = np.arange(model.H)/(model.H-1)
    gains = np.broadcast_to(np.asarray(gains_mm)[None,:,None]*curve/1000, shape).copy()
    paths = np.stack([np.zeros(shape), gains], axis=2)
    path = folder/"predictions.npz"
    arrays(path, dict(fit_seeds=model.MODEL_SEEDS, resets=model.BLIND, paths_m=paths, gradient_m_per_unit_pulse=gains,
                      mean_response_m=mean_mm/1000*curve, start_features=[pairs[s]["onset"]["features"] for s in model.BLIND]))
    starts = {str(s):{k:v for k,v in pairs[s]["onset"].items() if k != "features"} for s in model.BLIND}
    write(folder/"forecast_seal.json", dict(status="SEALED", spec_sha256=digest(model.PLAN),
                                           predictions_sha256=digest(path), onset=starts))


def test_all_nine_directions_require_resolved_axes_and_a_strict_mean_response_baseline(tmp_path):
    gains = [2., 3., 5.]
    pairs = {s: _pair(s, v) for s, v in zip(model.BLIND, gains, strict=True)}
    _forecast(tmp_path, pairs, gains)
    score = model.score_blind(pairs, tmp_path)
    assert score["cells_passed"] == 9 and score["supports_next_local_policy_experiment"]
    # The fit always predicting the training mean can satisfy the 25% cells yet
    # adds nothing over that competing predictor.
    for s in pairs:
        pairs[s] = _pair(s, 2.)
    hold = tmp_path/"constant"
    hold.mkdir()
    _forecast(hold, pairs, [2.,2.,2.], mean_mm=2.)
    result = model.score_blind(pairs, hold)
    assert result["cells_passed"] == 9 and not result["model_qualified"]
    pairs[model.BLIND[1]] = _pair(model.BLIND[1], .4, safe=False)
    pairs[model.BLIND[2]] = dict(seed=model.BLIND[2], reasons={"flexion":"partial:terminated"})
    result = model.score_blind(pairs, hold)
    assert len(result["cells"]) == 9 and result["cells_passed"] == 3 and not result["all_native_pairs_safe"]
    assert "below_engineering_resolution" in result["cells"][1]["reasons"]
    assert result["cells"][2]["reasons"] == ["incomplete_native_pair"]


def test_relative_error_uses_physical_effect_and_handles_wrong_sign(tmp_path):
    pairs = {s:_pair(s, 2.) for s in model.BLIND}
    _forecast(tmp_path, pairs, [2.55, -2., 2.49])
    result = model.score_blind(pairs, tmp_path)
    assert result["cells_passed"] == 3
    assert "outside_native_25pct" in result["cells"][0]["reasons"]
    assert "direction_mismatch" in result["cells"][1]["reasons"]


def test_synthetic_two_step_optimizer_audit_and_forecast_separation(monkeypatch, tmp_path):
    # Small algebraic fixture only. No native observations or actual fitting budget is used.
    monkeypatch.setattr(model, "UPDATES", 2)
    monkeypatch.setattr(model, "_protocol", lambda: {})
    train = {s:_pair(s, v) for s,v in zip(model.TRAIN, (1.,2.,3.), strict=True)}
    fit = model.fit_models(train, tmp_path/"fit")
    assert fit["world_updates"] == 6 and model.check_fit(tmp_path/"fit") == fit
    with np.load(tmp_path/"fit/models.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["coefficients"][:,-1], np.zeros(3))
    with pytest.raises(FileExistsError):
        model.fit_models(train, tmp_path/"fit")
    wrong = {s:_pair(s, 2.) for s in model.BLIND}
    with pytest.raises(ValueError, match="development resets"):
        model.fit_models(wrong, tmp_path/"bad")
    starts = {s:r["onset"] for s,r in wrong.items()}
    seal = model.predict_blind(starts, tmp_path/"fit", tmp_path/"pred")
    assert seal == read(tmp_path/"pred/forecast_seal.json") and seal["policy_updates"] == 0
    starts[model.BLIND[0]]["actual_mm"] = 2.
    with pytest.raises(ValueError, match="unexpected blind inputs"):
        model.predict_blind(starts, tmp_path/"fit", tmp_path/"reject")
