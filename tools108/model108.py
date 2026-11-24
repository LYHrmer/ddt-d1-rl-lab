"""Local 50 Hz forced dynamics of front-axle displacement under the frozen D1 policy."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read, write


PLAN = ROOT / "docs/experiments108/contact.json"
TRAIN = (108011, 108012, 108013)
BLIND = (108101, 108102, 108103)
MODEL_SEEDS = (108201, 108202, 108203)
ARMS = ("original", "flexion")
H, PULSE_TICKS, TOTAL_TICKS = 32, 16, 200
DT, PULSE_RAD, WHEEL_RADIUS_M = .02, -.08, .087
UPDATES = 100
# Four acceleration intervals, identical knots for the free and forced dynamics;
# one additional slope lets the start knee configuration affect the forced response.
PARAMETERS = 9
ACTUATOR_FILTER = 1. - np.exp(-DT / .08)
PARAMETER_PENALTY = .001
LEARNING_RATE = .05


def _protocol():
    spec = read(PLAN)
    require(spec["development"] == list(TRAIN) and spec["blind"] == list(BLIND)
            and not set(TRAIN).intersection(BLIND) and spec["arms"] == list(ARMS)
            and (spec["horizon"], spec["pulse_ticks"], spec["pulse_rad"], spec["ticks"]) ==
            (H, PULSE_TICKS, PULSE_RAD, TOTAL_TICKS) and spec["trigger_n"] == 1.
            and spec["model"]["updates_per_model"] == UPDATES and spec["model"]["initializations"] == 3
            and spec["model"]["resolution_mm"] == 1. and spec["model"]["relative_error_max"] == .25,
            "contact-response registration changed")
    return spec


def _sha_array(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def start_features(observation):
    """Extract encoder-based approach speed [m/s] and mean front-calf offset [rad] at k-1."""
    obs = np.asarray(observation)
    require(obs.shape == (643,) and np.isfinite(obs).all(), "invalid pre-contact actor observation")
    obs = obs.astype(np.float64)
    # The first 57 channels are the current frame. D1 scales encoder rates by .05.
    velocity = WHEEL_RADIUS_M * obs[[28, 32, 36, 40]].mean() / .05
    return np.array([velocity, obs[[11, 15]].mean()])


def _prefix_observation(path):
    """Read only trigger time and the single last pre-contact observation from a compressed native trace."""
    with ZipFile(path) as z:
        with z.open("trigger_tick.npy") as f:
            tick = int(np.lib.format.read_array(f, allow_pickle=False).item())
        if tick < 1:
            return None
        with z.open("actor_observation.npy") as f:
            version = np.lib.format.read_magic(f)
            shape, order, dtype = np.lib.format._read_array_header(f, version)
            require(not order and dtype == np.float32 and len(shape) == 2 and shape[1] == 643
                    and tick < shape[0], "invalid onset observation serialization")
            width = dtype.itemsize * shape[1]
            f.seek(width * (tick - 1), 1)
            obs = np.frombuffer(f.read(width), dtype=dtype)
            require(obs.shape == (643,), "incomplete onset observation")
    return tick, obs.copy()


def _onset(seed, tick, obs):
    return dict(seed=seed, trigger_tick=tick, observation_tick=tick-1,
                observation_sha256=_sha_array(obs), features=start_features(obs))


def load_blind_starts(episodes_dir):
    """Read one original-arm pre-contact frame per blind reset. Never open an outcome array or the flexion arm."""
    _protocol()
    result = {}
    for seed in BLIND:
        folder = Path(episodes_dir) / f"seed{seed}_original"
        path = folder / "trace.npz"
        # A partial episode is not eligible for an independent forecast.
        start = _prefix_observation(path) if path.is_file() else None
        result[seed] = (dict(seed=seed, reason="missing_complete_original_or_contact") if start is None
                        else _onset(seed, *start))
    return result


def _arm_data(folder, seed, arm, spec):
    receipt_file = folder / f"seed{seed}_{arm}" / "report.json"
    if not receipt_file.is_file():
        return None, "missing_report"
    rec = read(receipt_file)
    require((rec["seed"], rec["arm"], rec["checkpoint_sha256"]) == (seed, arm, spec["checkpoint_sha256"])
            and rec["geometry"] == spec["geometry"], "native episode or checkpoint identity changed")
    if rec["status"] != "COMPLETE" or not rec["full_horizon"]:
        return None, "partial:" + str(rec.get("terminal_reason"))
    path = receipt_file.with_name("trace.npz")
    require(rec["trace_file"] == path.name and digest(path) == rec["trace_sha256"], "incorrect native trace seal")
    with np.load(path, allow_pickle=False) as z:
        names = ("qpos", "qvel", "actor_observation", "issued_action", "applied_action", "nominal_action",
                 "pulse_target_rad", "actual_issue_offset_rad", "front_axle_x_m", "step_force_n",
                 "trigger_tick", "fallen", "illegal_contact", "terminated")
        data = {k: z[k] for k in names}
    k = int(data["trigger_tick"])
    require(data["step_force_n"].shape == (10 * TOTAL_TICKS, 4)
            and np.isfinite(data["step_force_n"]).all(), "invalid 500 Hz contact stream")
    hits = np.flatnonzero((data["step_force_n"][:, :2] >= spec["trigger_n"]).any(axis=1))
    if not len(hits):
        require(k == -1 and rec["trigger_tick"] is None, "trigger without front-wheel contact")
        return None, "no_front_contact"
    require(k == rec["trigger_tick"] == int(hits[0])//10 + 1 and rec["first_contact_substep"] == int(hits[0]),
            "onset does not follow the first force-qualified contact")
    if k + H > TOTAL_TICKS:
        return None, "short_postcontact_window"
    for key, width in (("qpos", 23), ("qvel", 22), ("actor_observation", 643)):
        require(data[key].shape == (TOTAL_TICKS+1, width) and np.isfinite(data[key]).all(), "invalid native boundary data")
    expected_pulse = np.zeros(TOTAL_TICKS)
    if arm == "flexion":
        expected_pulse[k:k+PULSE_TICKS] = PULSE_RAD
    nominal, issued, applied = (data[name] for name in ("nominal_action", "issued_action", "applied_action"))
    require(all(a.shape == (TOTAL_TICKS, 16) and np.isfinite(a).all() for a in (nominal, issued, applied))
            and np.array_equal(expected_pulse, data["pulse_target_rad"]) and not applied[0].any()
            and np.array_equal(applied[1:], issued[:-1]), "incorrect applied-action FIFO or pulse clock")
    expected = nominal.copy()
    expected[:, [2, 6]] = np.clip(nominal[:, [2, 6]] + (expected_pulse/.5).astype(nominal.dtype)[:, None], -1., 1.)
    require(np.array_equal(issued, expected) and np.allclose(data["actual_issue_offset_rad"],
            .5*(issued[:, [2, 6]] - nominal[:, [2, 6]]), rtol=0, atol=1e-8), "unregistered front-calf command")
    if arm == "flexion" and not (data["actual_issue_offset_rad"][k:k+PULSE_TICKS] < 0).any(axis=0).all():
        return None, "intervention_fully_clipped_at_a_front_knee"
    axle = data["front_axle_x_m"]
    require(axle.shape == (TOTAL_TICKS+1,) and np.isfinite(axle).all(), "invalid front axle clock")
    safe = not any(data[f].any() for f in ("fallen", "illegal_contact", "terminated"))
    prefixes = {name: _sha_array(data[name][:k+1]) for name in ("qpos", "qvel", "actor_observation")}
    prefixes.update({name: _sha_array(data[name][:k]) for name in ("issued_action", "nominal_action")})
    prefixes["applied_action"] = _sha_array(data["applied_action"][:k+1])
    return dict(onset=_onset(seed, k, data["actor_observation"][k-1]), prefix=prefixes, safe=safe,
                trace_sha256=rec["trace_sha256"], path_m=axle[k+1:k+H+1] - axle[k]), None


def load_pairs(episodes_dir, split):
    """Load signed native training or scoring labels only after the relevant stage is sealed."""
    require(split in ("train", "blind"), "unknown contact-response split")
    spec = _protocol()
    roster = TRAIN if split == "train" else BLIND
    pairs = {}
    for seed in roster:
        records = [_arm_data(Path(episodes_dir), seed, arm, spec) for arm in ARMS]
        reasons = {arm: why for arm, (_, why) in zip(ARMS, records, strict=True) if why}
        if reasons:
            pairs[seed] = dict(seed=seed, reasons=reasons)
            continue
        original, flex = (rec for rec, _ in records)
        require(original["prefix"] == flex["prefix"] and original["onset"]["trigger_tick"] == flex["onset"]["trigger_tick"],
                "intervention arms do not branch from an identical native prefix")
        require(original["path_m"][0] == flex["path_m"][0], "matched native FIFO arms differ before pulse delivery")
        pairs[seed] = dict(seed=seed, onset=original["onset"], paths_m=np.stack([original["path_m"], flex["path_m"]]),
                           safe=[original["safe"], flex["safe"]], source_sha256=[r["trace_sha256"] for r, _ in records])
    return pairs


def _trajectory_basis(knee, strength):
    """Propagate sensitivities of (x,v) to nine acceleration weights for a prescribed normalized pulse."""
    require(np.isfinite(knee) and np.isfinite(strength), "nonfinite start or pulse")
    x, velocity = np.zeros(PARAMETERS), np.zeros(PARAMETERS)
    path, actuator = [], 0.
    for t in range(H):
        # The action issued at t=0 reaches the plant in interval 1. At the first
        # endpoint the two inputs and their integrated responses are identical.
        applied = strength if 1 <= t <= PULSE_TICKS else 0.
        actuator += ACTUATOR_FILTER * (applied - actuator)
        block = t // 8
        acceleration = np.zeros(PARAMETERS)
        acceleration[block] = 1.
        acceleration[4+block] = actuator
        acceleration[-1] = actuator * knee
        x = x + DT*velocity + .5*DT**2*acceleration
        velocity = velocity + DT*acceleration
        path.append(x.copy())
    return np.stack(path)


def trajectory_m(features, strength, coefficients, center_rad, scale_rad):
    """Predict 32 signed axle displacements from the branch boundary in metres."""
    speed, knee_rad = np.asarray(features, dtype=np.float64)
    coefficients = np.asarray(coefficients)
    require(coefficients.shape == (PARAMETERS,) and np.isfinite(coefficients).all()
            and np.isfinite(speed) and np.isfinite(center_rad) and np.isfinite(scale_rad) and scale_rad > 0,
            "invalid local model inputs or coefficients")
    basis = _trajectory_basis((knee_rad-center_rad)/scale_rad, strength)
    return DT*np.arange(1, H+1)*speed + basis @ coefficients


def gain_gradient_m(features, coefficients, center_rad, scale_rad):
    """Exact derivative with respect to normalized pulse strength; dynamics are affine in this command."""
    knee = (features[1]-center_rad) / scale_rad
    change = _trajectory_basis(knee, 1.) - _trajectory_basis(knee, 0.)
    return change @ coefficients


def _objective(features, y):
    require(features.shape == (len(TRAIN), 2) and y.shape == (len(TRAIN), 2, H)
            and np.isfinite(features).all() and np.isfinite(y).all(), "invalid development matrix")
    center = float(np.mean(features[:, 1]))
    scale = max(float(np.std(features[:, 1])), .05)
    speed_path = DT * np.arange(1, H+1) * features[:, 0, None]
    paths = y - speed_path[:, None, :]
    effects = y[:, 1] - y[:, 0]
    mats = np.array([[_trajectory_basis((k-center)/scale, s) for s in (0., 1.)] for k in features[:, 1]])
    gains = mats[:, 1] - mats[:, 0]
    # Equal-weight trajectory, signed response, and terminal response losses.
    # Each denominator uses development outcomes only, with a 1 mm RMS floor.
    components = [(mats, paths), (gains, effects), (gains[:, -1], effects[:, -1])]
    X, target, rms = [], [], []
    for design, values in components:
        unit = max(.001, float(np.sqrt(np.mean(values**2))))
        denominator = np.sqrt(values.size)*unit
        X.append(design.reshape(-1, PARAMETERS)/denominator)
        target.append(values.ravel()/denominator)
        rms.append(unit)
    matrix, target = np.vstack(X), np.concatenate(target)
    # Scale down high-leverage columns; never amplify a barely excited input.
    column_scale = np.maximum(np.linalg.norm(matrix, axis=0), 1.)
    return matrix/column_scale, target, column_scale, center, scale, rms


def _adam_fit(design, targets, seed):
    rng = np.random.default_rng(seed)
    params = rng.normal(0., .02, PARAMETERS)
    first, second = np.zeros_like(params), np.zeros_like(params)
    losses, path = [], [params.copy()]
    for step in range(1, UPDATES+1):
        residual = design @ params - targets
        losses.append(float(residual @ residual + PARAMETER_PENALTY*(params @ params)))
        gradient = 2.*(design.T @ residual + PARAMETER_PENALTY*params)
        first = .9*first + .1*gradient
        second = .999*second + .001*gradient**2
        params -= LEARNING_RATE*(first/(1.-.9**step))/(np.sqrt(second/(1.-.999**step))+1e-8)
        path.append(params.copy())
    require(np.isfinite(params).all() and np.isfinite(losses).all(), "nonfinite local response fit")
    return params, losses, path, first, second


def _new_output(folder, marker, data):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / marker, data)
    return folder


def fit_models(train_pairs, out_dir):
    """Use only the three registered development reset pairs; return and save fit.json and models.npz."""
    _protocol()
    require(set(train_pairs) == set(TRAIN), "development resets must match the complete registered roster")
    folder = _new_output(out_dir, "fit_started.json", dict(development=list(TRAIN), blind=list(BLIND)))
    missing = {str(s): train_pairs[s]["reasons"] for s in TRAIN if "reasons" in train_pairs[s]}
    if missing:
        report = dict(status="HOLD", reasons=missing, world_updates=0, policy_updates=0)
        write(folder / "fit.json", report)
        return report
    require(all(train_pairs[s]["seed"] == train_pairs[s]["onset"]["seed"] == s for s in TRAIN),
            "development record belongs to a different reset")
    features = np.stack([train_pairs[s]["onset"]["features"] for s in TRAIN])
    paths = np.stack([train_pairs[s]["paths_m"] for s in TRAIN])
    design, target, column_scale, center, scale, rms = _objective(features, paths)
    coefficients, history, parameter_paths, moments1, moments2 = [], [], [], [], []
    for seed in MODEL_SEEDS:
        normalized, losses, parameter_path, m, v = _adam_fit(design, target, seed)
        coefficients.append(np.where(np.any(design != 0, axis=0), normalized/column_scale, 0.))
        history.append(losses)
        parameter_paths.append(parameter_path)
        moments1.append(m)
        moments2.append(v)
    baseline = (paths[:, 1] - paths[:, 0]).mean(axis=0)
    model_file = folder / "models.npz"
    arrays(model_file, dict(development=TRAIN, model_seeds=MODEL_SEEDS, coefficients=coefficients, losses=history,
                            knee_center_rad=center, knee_scale_rad=scale, component_rms_m=rms,
                            column_scale=column_scale, mean_response_m=baseline, normalized_parameter_path=parameter_paths,
                            first_moment=moments1, second_moment=moments2,
                            training_features=features, training_paths_m=paths))
    report = dict(status="COMPLETE", model_sha256=digest(model_file), spec_sha256=digest(PLAN),
                  development=list(TRAIN), initializations=list(MODEL_SEEDS), world_updates=len(MODEL_SEEDS)*UPDATES,
                  policy_updates=0, source_sha256={str(s): train_pairs[s]["source_sha256"] for s in TRAIN},
                  source_safety={str(s): train_pairs[s]["safe"] for s in TRAIN},
                  development_response_mm={str(s): float(1000.*(p[1,-1]-p[0,-1])) for s, p in zip(TRAIN, paths, strict=True)},
                  baseline_endpoint_mm=float(1000*baseline[-1]), empirical_weight_source="development pairs only",
                  weights_per_fit=PARAMETERS, actuator_time_constant_s=.08, model_step_s=DT,
                  optimizer=dict(name="full-batch Adam", learning_rate=LEARNING_RATE, betas=[.9,.999], epsilon=1e-8,
                                 penalty=PARAMETER_PENALTY, updates_per_initialization=UPDATES),
                  target="World-x displacement of mean front-wheel axle from trigger to each endpoint, metres.")
    write(folder / "fit.json", report)
    return report


def _models(folder):
    folder = Path(folder)
    fit = read(folder / "fit.json")
    require(fit["status"] == "COMPLETE" and fit["spec_sha256"] == digest(PLAN)
            and fit["development"] == list(TRAIN) and fit["model_sha256"] == digest(folder / "models.npz"),
            "incomplete or altered development fit")
    with np.load(folder / "models.npz", allow_pickle=False) as z:
        values = {k: z[k] for k in ("model_seeds", "coefficients", "mean_response_m", "knee_center_rad", "knee_scale_rad")}
    require(np.array_equal(values["model_seeds"], MODEL_SEEDS) and values["coefficients"].shape ==
            (len(MODEL_SEEDS), PARAMETERS), "invalid fitted models")
    return fit, values


def check_fit(folder):
    """Verify each stored optimizer increment, moment and loss directly from archived development supervision."""
    fit, _ = _models(folder)
    with np.load(Path(folder)/"models.npz", allow_pickle=False) as z:
        features, paths = z["training_features"], z["training_paths_m"]
        design, target, scales, center, scale, rms = _objective(features, paths)
        np.testing.assert_allclose(z["component_rms_m"], rms, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(z["column_scale"], scales, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(z["mean_response_m"], (paths[:,1]-paths[:,0]).mean(axis=0), atol=1e-12)
        require(z["knee_center_rad"].item() == center and z["knee_scale_rad"].item() == scale,
                "incorrect normalization of knee features")
        params, losses = z["normalized_parameter_path"], z["losses"]
        require(params.shape == (len(MODEL_SEEDS), UPDATES+1, PARAMETERS), "incorrect parameter audit trail")
        for i, seed in enumerate(MODEL_SEEDS):
            np.testing.assert_array_equal(params[i,0], np.random.default_rng(seed).normal(0., .02, PARAMETERS))
            m, v = np.zeros(PARAMETERS), np.zeros(PARAMETERS)
            for step in range(1, UPDATES+1):
                p = params[i,step-1]
                r = design@p - target
                np.testing.assert_allclose(losses[i,step-1], r@r + PARAMETER_PENALTY*(p@p), atol=1e-12, rtol=1e-10)
                g = 2.*(design.T@r + PARAMETER_PENALTY*p)
                m, v = .9*m+.1*g, .999*v+.001*g**2
                increment = -LEARNING_RATE*(m/(1.-.9**step))/(np.sqrt(v/(1.-.999**step))+1e-8)
                np.testing.assert_allclose(np.diff(params[i,step-1:step+1], axis=0)[0], increment, atol=1e-10, rtol=1e-8)
            physical = np.where(np.any(design != 0, axis=0), params[i,-1]/scales, 0.)
            np.testing.assert_allclose(z["coefficients"][i], physical, atol=1e-10, rtol=1e-8)
            np.testing.assert_allclose(z["first_moment"][i], m, atol=1e-10, rtol=1e-8)
            np.testing.assert_allclose(z["second_moment"][i], v, atol=1e-10, rtol=1e-8)
    return fit


def predict_blind(starts, fit_dir, out_dir):
    """Seal forecasts using start features only, before the scorer opens the paired blind endpoints."""
    _protocol()
    require(set(starts) == set(BLIND), "blind onset roster changed")
    fit, model = _models(fit_dir)
    folder = _new_output(out_dir, "predict_started.json", dict(blind=list(BLIND), models_sha256=fit["model_sha256"]))
    pred = np.full((len(MODEL_SEEDS), len(BLIND), 2, H), np.nan)
    gradient = np.full((len(MODEL_SEEDS), len(BLIND), H), np.nan)
    contexts = np.full((len(BLIND), 2), np.nan)
    receipts = {}
    for j, seed in enumerate(BLIND):
        onset = starts[seed]
        require(onset["seed"] == seed, "different blind onset identity")
        if "reason" in onset:
            require(set(onset) == {"seed", "reason"}, "unexpected fields in missing onset")
            receipts[str(seed)] = onset
            continue
        require(set(onset) == {"seed", "trigger_tick", "observation_tick", "observation_sha256", "features"}
                and onset["observation_tick"] == onset["trigger_tick"]-1, "unexpected blind inputs or clock")
        inputs = np.asarray(onset["features"])
        require(inputs.shape == (2,) and np.isfinite(inputs).all(), "invalid blind start features")
        contexts[j] = inputs
        receipts[str(seed)] = {k:v for k,v in onset.items() if k != "features"}
        for n, coeff in enumerate(model["coefficients"]):
            args = coeff, model["knee_center_rad"].item(), model["knee_scale_rad"].item()
            for a, strength in enumerate((0., 1.)):
                pred[n,j,a] = trajectory_m(inputs, strength, *args)
            gradient[n,j] = gain_gradient_m(inputs, *args)
            np.testing.assert_allclose(gradient[n,j], pred[n,j,1]-pred[n,j,0], rtol=0., atol=1e-12)
    path = folder / "predictions.npz"
    arrays(path, dict(fit_seeds=MODEL_SEEDS, resets=BLIND, paths_m=pred, gradient_m_per_unit_pulse=gradient,
                      start_features=contexts, mean_response_m=model["mean_response_m"]))
    seal = dict(status="SEALED", predictions_sha256=digest(path), models_sha256=fit["model_sha256"],
                training_fit_sha256=digest(Path(fit_dir)/"fit.json"), spec_sha256=digest(PLAN),
                onset=receipts, fitted_updates=0, policy_updates=0)
    write(folder / "forecast_seal.json", seal)
    return seal


def score_blind(blind_pairs, pred_dir):
    """Read sealed forecasts, then assess each of the three paired finite contrasts for every model."""
    spec = _protocol()["model"]
    require(set(blind_pairs) == set(BLIND), "incomplete blind scoring roster")
    folder = Path(pred_dir)
    seal = read(folder / "forecast_seal.json")
    path = folder / "predictions.npz"
    require(seal["status"] == "SEALED" and seal["spec_sha256"] == digest(PLAN)
            and seal["predictions_sha256"] == digest(path), "blind forecasts are not sealed")
    with np.load(path, allow_pickle=False) as z:
        paths, gradients, baseline = z["paths_m"], z["gradient_m_per_unit_pulse"], z["mean_response_m"]
        start = z["start_features"]
        require(np.array_equal(z["fit_seeds"], MODEL_SEEDS) and np.array_equal(z["resets"], BLIND)
                and paths.shape == (len(MODEL_SEEDS), len(BLIND), 2, H)
                and gradients.shape == paths.shape[:2]+(H,) and baseline.shape == (H,), "wrong blind array roster")
    truth, safety, cells = {}, {}, []
    for j, reset in enumerate(BLIND):
        pair = blind_pairs[reset]
        if "reasons" in pair:
            truth[str(reset)] = dict(reasons=pair["reasons"])
            safety[str(reset)] = None
            continue
        require(pair["seed"] == pair["onset"]["seed"] == reset, "mixed blind pair identity")
        onset = seal["onset"][str(reset)]
        if "reason" not in onset:
            require(onset["observation_sha256"] == pair["onset"]["observation_sha256"]
                    and onset["trigger_tick"] == pair["onset"]["trigger_tick"]
                    and np.array_equal(start[j], pair["onset"]["features"]), "blind scoring onset differs from prediction onset")
        native = np.asarray(pair["paths_m"])
        require(native.shape == (2, H) and np.isfinite(native).all(), "incomplete native response curve")
        gain = 1000.*(native[1]-native[0])
        truth[str(reset)] = dict(native_response_mm=gain.tolist(), native_paths_mm=(1000*native).tolist(),
                                endpoint_mm=float(gain[-1]), source_sha256=pair["source_sha256"])
        safety[str(reset)] = dict(zip(ARMS, pair["safe"], strict=True))
    mean_blind_errors, baseline_ok = {}, {}
    for i, model_seed in enumerate(MODEL_SEEDS):
        errors, constant_errors, zero_errors = [], [], []
        for j, reset in enumerate(BLIND):
            case = truth[str(reset)]
            cell = dict(model_seed=model_seed, reset=reset, passed=False)
            cells.append(cell)
            if "reasons" in case:
                cell["reasons"] = ["incomplete_native_pair"]
                continue
            if "reason" in seal["onset"][str(reset)]:
                cell["reasons"] = ["incomplete_forecast_start"]
                continue
            require(np.isfinite(paths[i,j]).all() and np.isfinite(gradients[i,j]).all(), "missing sealed blind forecast")
            response = (paths[i,j,1] - paths[i,j,0]) * 1000.
            np.testing.assert_allclose(response, 1000*gradients[i,j], atol=1e-8, rtol=0)
            pred, actual, mean = float(response[-1]), case["endpoint_mm"], float(1000*baseline[-1])
            error = abs(pred-actual)
            reasons = []
            if abs(actual) < spec["resolution_mm"]:
                reasons.append("below_engineering_resolution")
            if pred*actual <= 0:
                reasons.append("direction_mismatch")
            if error > spec["relative_error_max"]*abs(actual):
                reasons.append("outside_native_25pct")
            cell.update(passed=not reasons, reasons=reasons, predicted_mm=pred, native_mm=actual,
                        error_mm=error, relative_error=None if actual == 0 else error/abs(actual),
                        signed_action_derivative_mm_per_rad=1000.*float(gradients[i,j,-1])/PULSE_RAD,
                        train_mean_mm=mean, train_mean_error_mm=abs(mean-actual), zero_error_mm=abs(actual),
                        trajectory_mae_mm=float(np.mean(abs(response-np.asarray(case["native_response_mm"])))))
            errors.append(error)
            constant_errors.append(abs(mean-actual))
            zero_errors.append(abs(actual))
        complete = len(errors) == len(BLIND)
        mean_blind_errors[str(model_seed)] = (dict(candidate_mm=float(np.mean(errors)),
                                                  training_mean_mm=float(np.mean(constant_errors)),
                                                  zero_mm=float(np.mean(zero_errors))) if complete else None)
        baseline_ok[str(model_seed)] = bool(complete and np.mean(errors) < np.mean(constant_errors))
    count = sum(c["passed"] for c in cells)
    qualifies = count == len(MODEL_SEEDS)*len(BLIND) and all(baseline_ok.values())
    safe = all(s is not None and all(s.values()) for s in safety.values())
    positive = all("endpoint_mm" in r and r["endpoint_mm"] >= spec["resolution_mm"] for r in truth.values())
    return dict(status="SCORED", forecast_sha256=seal["predictions_sha256"], outcomes=truth, arm_safety=safety,
                cells=cells, cells_passed=count, cells_planned=len(cells), blind_mae=mean_blind_errors,
                strictly_beats_training_mean=baseline_ok, model_qualified=qualifies,
                all_native_pairs_safe=safe, all_native_positive_resolved=positive,
                supports_next_local_policy_experiment=bool(qualifies and safe and positive),
                policy_updates=0, observable="Front-axle world-x displacement at trigger+32, relative to trigger.")
