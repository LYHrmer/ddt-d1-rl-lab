"""Learn 50 Hz dynamics around the time-aligned D1 grid reference, in physical actuator units."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from tools94.common94 import arrays, digest
from tools111.kinematics import rotation, vendor_targets
from tools112.surface import SEEDS


TICKS = 600
HORIZON = 16
UPDATES = 50
STATES = 37
ACTIONS = 16
FORWARD = 3
LEGS = [j for j in range(ACTIONS) if j % 4 != 3]
ARMS = ("action", "masked")
CARRY = .80
GAIN = .35
ADAM_LR = .008
STATE_FLOOR = np.array([.015, .04, .04, .08, .08, .05, .25, .25, .25] + [.06]*12
                       + [.35, .35, .35, 1.0]*4)
TARGET_FLOOR = np.tile([.05, .05, .05, .8], 4)


def state_vector(qpos, qvel):
    """Return z/RPY(roll,pitch)/heading velocities/vz/body omega/leg q/all joint dq."""
    q = np.asarray(qpos, dtype=np.float64)
    v = np.asarray(qvel, dtype=np.float64)
    if q.shape[-1:] != (23,) or v.shape != (*q.shape[:-1], 22):
        raise ValueError("expected matching D1 qpos[23] and qvel[22] endpoints")
    r = rotation(q[..., 3:7])
    yaw = np.arctan2(r[..., 1, 0], r[..., 0, 0])
    roll = np.arctan2(r[..., 2, 1], r[..., 2, 2])
    pitch = np.arctan2(-r[..., 2, 0], np.hypot(r[..., 0, 0], r[..., 1, 0]))
    c, s = np.cos(yaw), np.sin(yaw)
    forward = c*v[..., 0] + s*v[..., 1]
    lateral = -s*v[..., 0] + c*v[..., 1]
    base = np.stack((q[..., 2], roll, pitch, forward, lateral, v[..., 2]), axis=-1)
    return np.concatenate((base, v[..., 3:6], q[..., 7:][..., LEGS], v[..., 6:22]), axis=-1)


def physical_targets(raw16):
    """Map a delayed, applied vendor action to interleaved leg radians and wheel rad/s."""
    raw = np.asarray(raw16, dtype=np.float64)
    if raw.shape[-1:] != (ACTIONS,):
        raise ValueError("expected interleaved 16-component vendor actions")
    targets = vendor_targets(raw)
    targets[..., 3::4] = 5.*raw[..., 3::4]
    return targets


@dataclass
class Trajectory:
    seed: int
    domain: str
    path: Path
    states: np.ndarray
    targets: np.ndarray
    x: np.ndarray


def read_trajectory(seed, domain, path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as trace:
        time = trace["time"]
        q, v = trace["qpos"], trace["qvel"]
        applied, issued = trace["applied"], trace["issued"]
        np.testing.assert_allclose(time, np.arange(TICKS+1)/50., atol=1e-9, rtol=0)
        if q.shape != (601,23) or v.shape != (601,22) or applied.shape != (600,16) or issued.shape != (600,16):
            raise ValueError(f"expected one complete 12-second native trajectory in {path}")
        np.testing.assert_array_equal(applied[0], np.zeros(ACTIONS))
        np.testing.assert_array_equal(applied[1:], issued[:-1])
        np.testing.assert_array_equal(v[:-1,9::4], trace["encoder_wheel_rad_s"])
        states, targets = state_vector(q, v), physical_targets(applied)
    if not np.isfinite(states).all() or not np.isfinite(targets).all():
        raise ValueError(f"nonfinite native states or actuator targets in {path}")
    return Trajectory(int(seed), domain, path, states, targets, q[:,0])


def _starts(trace):
    starts = np.arange(TICKS-HORIZON+1)
    x = np.lib.stride_tricks.sliding_window_view(trace.x, HORIZON+1)
    cruise = np.all((x >= 3.) & (x <= 6.), axis=1)
    return {"cruise": starts[cruise], "braking": starts[starts >= 450], "all": starts}


@dataclass
class GridWorld:
    reference_state: np.ndarray
    reference_target: np.ndarray
    state_scale: np.ndarray
    target_scale: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    uses_action: bool
    heldout: int

    def step(self, tick, state, physical_target):
        """One free predicted endpoint. Tick, state and SI targets can be batched."""
        k = np.asarray(tick)
        if k.dtype.kind not in "iu" or np.any((k < 0) | (k >= TICKS)):
            raise ValueError("time outside the registered command course")
        x = (np.asarray(state) - self.reference_state[k]) / self.state_scale
        u = (np.asarray(physical_target) - self.reference_target[k]) / self.target_scale
        if x.shape[-1:] != (STATES,) or u.shape != (*x.shape[:-1], ACTIONS):
            raise ValueError("expected state[37] and physical target[16] with matching batch axes")
        if not self.uses_action:
            u = np.zeros_like(u)
        features = np.concatenate((x, u), axis=-1)
        y = CARRY*x + GAIN*np.tanh(features @ self.weight + self.bias)
        return self.reference_state[k+1] + self.state_scale*y

    def save(self, path):
        arrays(path, {"format_version": np.int64(1), **vars(self)})


def load_model(path):
    """Load frozen, CPU-only inference arrays; reading a checkpoint does not train it."""
    with np.load(path, allow_pickle=False) as data:
        if int(data["format_version"]) != 1:
            raise ValueError("unknown speed-world checkpoint")
        shapes = {"reference_state": (TICKS+1, STATES), "reference_target": (TICKS, ACTIONS),
                  "state_scale": (STATES,), "target_scale": (ACTIONS,),
                  "weight": (STATES+ACTIONS, STATES), "bias": (STATES,)}
        packed = {key: data[key] for key in shapes}
        if any(value.shape != shapes[key] or not np.isfinite(value).all() for key,value in packed.items()):
            raise ValueError("malformed speed-world checkpoint")
        if np.any(packed["state_scale"] <= 0) or np.any(packed["target_scale"] <= 0):
            raise ValueError("nonpositive speed-world normalization")
        return GridWorld(**packed, uses_action=bool(data["uses_action"]), heldout=int(data["heldout"]))


def _initial(train, heldout):
    grid = [row for row in train if row.domain == "grid"]
    mean_s = np.mean([row.states for row in grid], axis=0)
    mean_u = np.mean([row.targets for row in grid], axis=0)
    states = np.stack([row.states - mean_s for row in train])
    actions = np.stack([row.targets - mean_u for row in train])
    scale_s = np.maximum(STATE_FLOOR, np.sqrt(np.mean(states*states, axis=(0,1))))
    scale_u = np.maximum(TARGET_FLOOR, np.sqrt(np.mean(actions*actions, axis=(0,1))))
    model = GridWorld(mean_s, mean_u, scale_s, scale_u, np.zeros((STATES+ACTIONS,STATES)),
                      np.zeros(STATES), True, heldout)
    return model, states/scale_s, actions/scale_u


def _batches(train, heldout):
    """Each update uses eight windows per domain and phase; both arms consume identical windows."""
    pools = []
    starts = [_starts(row) for row in train]
    for domain in ("grid", "compound"):
        for phase in ("cruise","braking","all"):
            items = [(i,k) for i,row in enumerate(train) if row.domain == domain for k in starts[i][phase]]
            if not items:
                raise ValueError(f"no {domain} {phase} training windows")
            pools.append(np.asarray(items, dtype=np.int64))
    random = np.random.default_rng(1130000+heldout)
    return np.array([np.concatenate([p[random.integers(len(p), size=8)] for p in pools])
                     for _ in range(UPDATES)])


def loss_gradient(weight, bias, states, actions):
    """Exact BPTT for all normalized endpoints and an extra forward-speed MSE term."""
    batch, endpoints, dims = states.shape
    horizon = endpoints-1
    if dims != STATES or actions.shape != (batch, horizon, ACTIONS) or horizon == 0:
        raise ValueError("invalid normalized training window")
    features = np.empty((horizon,batch,STATES+ACTIONS))
    activations = np.empty((horizon,batch,STATES))
    errors = np.empty_like(activations)
    x = states[:,0]
    for j in range(horizon):
        features[j,:,:STATES] = x
        features[j,:,STATES:] = actions[:,j]
        activations[j] = np.tanh(features[j] @ weight + bias)
        x = CARRY*x + GAIN*activations[j]
        errors[j] = x - states[:,j+1]
    loss = np.mean(errors**2) + np.mean(errors[:,:,FORWARD]**2)
    direct = errors / STATES
    direct[:,:,FORWARD] += errors[:,:,FORWARD]
    direct *= 2. / (batch*horizon)
    propagated = np.zeros_like(x)
    grad_w, grad_b = np.zeros_like(weight), np.zeros_like(bias)
    for j in range(horizon-1, -1, -1):
        total = direct[j] + propagated
        affine = GAIN*total*(1.-activations[j]**2)
        grad_w += features[j].T @ affine
        grad_b += affine.sum(axis=0)
        propagated = CARRY*total + affine @ weight[:STATES].T
    return float(loss), grad_w, grad_b


def _train_arm(model, states, actions, batches, log, name, out):
    params = (model.weight, model.bias)
    moment = [np.zeros_like(p) for p in params]
    variance = [np.zeros_like(p) for p in params]
    losses = []
    try:
        for step,batch in enumerate(batches, start=1):
            log.write(json.dumps({"arm": name, "update": step, "event": "attempt"})+"\n")
            rows, starts = batch.T
            indices = starts[:,None] + np.arange(HORIZON+1)
            window_s = states[rows[:,None], indices]
            window_u = actions[rows[:,None], indices[:,:-1]]
            if not model.uses_action:
                window_u.fill(0.)
            loss, dw, db = loss_gradient(*params, window_s, window_u)
            norm = float(np.sqrt(np.sum(dw*dw)+np.sum(db*db)))
            if not np.isfinite(loss) or not np.isfinite(norm):
                raise ValueError("nonfinite rollout loss or gradient")
            for p,m,v,g in zip(params,moment,variance,(dw,db)):
                g = g / max(1.,norm)
                m *= .9
                m += .1*g
                v *= .999
                v += .001*g*g
                p -= ADAM_LR*(m/(1.-.9**step))/(np.sqrt(v/(1.-.999**step))+1e-8)
            losses.append(loss)
            log.write(json.dumps({"arm": name,"update": step,"event": "complete",
                                  "loss": loss,"gradient_norm": norm}, allow_nan=False)+"\n")
        if len(losses) != UPDATES or not all(np.isfinite(p).all() for p in params):
            raise ValueError("invalid final optimizer state")
        model.save(out/(name+".npz"))
    except BaseException:
        model.save(out/("partial_"+name+".npz"))
        raise
    return {"updates":len(losses),"first_batch_loss":losses[0],"last_batch_loss":losses[-1],
            "checkpoint_sha256":digest(out/(name+".npz"))}


def heldout_predictions(model, trace):
    """Condition only on the start state and the 16 recorded applied targets in each window."""
    masks = _starts(trace)
    starts = np.sort(np.union1d(masks["cruise"], masks["braking"]))
    if len(masks["cruise"]) == 0 or len(masks["braking"]) == 0:
        raise ValueError("held-out evaluation window is empty")
    state = trace.states[starts].copy()
    for j in range(HORIZON):
        ticks = starts+j
        state = model.step(ticks, state, trace.targets[ticks])
    prediction = state[:, FORWARD]
    truth = trace.states[starts+HORIZON, FORWARD]
    raw = {"starts":starts,"truth":truth,"prediction":prediction}
    score = {}
    for phase in ("cruise","braking"):
        mask = np.isin(starts,masks[phase])
        raw[phase] = mask
        score[phase] = {"windows": int(mask.sum()),
                        "mae_m_s": float(np.mean(np.abs(prediction[mask]-truth[mask])))}
    return raw, score


def fit_fold(out, cases, heldout):
    """Fit action and masked worlds on whole training trajectories, then open the reserved grid trace."""
    out = Path(out)
    cases = sorted(((int(s), d, Path(p)) for s,d,p in cases), key=lambda row: (row[1],row[0],row[2].name))
    if heldout not in SEEDS or len(cases) != 10 or len(set(p for _,_,p in cases)) != 10:
        raise ValueError("unexpected fold or archive count")
    if sorted(s for s,d,_ in cases if d == "grid") != sorted(SEEDS) or sum(d=="compound" for _,d,_ in cases) != 7:
        raise ValueError("expected three level-grid and seven complete compound trajectories")
    held = [row for row in cases if row[0] == heldout]
    if len(held) != 1 or held[0][1] != "grid":
        raise ValueError("held-out reset spans unexpected trajectories")
    train = [read_trajectory(*r) for r in cases if r[0] != heldout]
    base, states, actions = _initial(train, heldout)
    batches = _batches(train, heldout)
    arms = {}
    with (out/"fit_updates.jsonl").open("x",buffering=1) as log:
        for name in ARMS:
            model = GridWorld(base.reference_state,base.reference_target,base.state_scale,base.target_scale,
                              base.weight.copy(),base.bias.copy(),name == "action",int(heldout))
            arms[name] = _train_arm(model,states,actions,batches,log,name,out)
    test = read_trajectory(*held[0])
    archive = {}
    for name in ARMS:
        signals, score = heldout_predictions(load_model(out/(name+".npz")),test)
        archive[name] = signals.pop("prediction")
        for key,value in signals.items():
            if key in archive:
                np.testing.assert_array_equal(archive[key],value)
            archive[key] = value
        arms[name]["holdout"] = score
    arrays(out/"heldout_h16.npz", archive)
    passed = {s: arms["action"]["holdout"][s]["mae_m_s"] < arms["masked"]["holdout"][s]["mae_m_s"]
              for s in ("cruise","braking")}
    return {"heldout_seed":int(heldout),"heldout_sha256":digest(test.path),"arms":arms,
            "training":[{"seed":r.seed,"domain":r.domain,"file":r.path.name,"sha256":digest(r.path)} for r in train],
            "sample_seed":1130000+int(heldout),"sample_schedule_sha256":hashlib.sha256(batches.tobytes()).hexdigest(),
            "world_updates":sum(r["updates"] for r in arms.values()),"policy_updates":0,"physics_steps":0,
            "holdout_scores_sha256":digest(out/"heldout_h16.npz"),"updates_sha256":digest(out/"fit_updates.jsonl"),
            "passed":passed,"both_phases_passed":all(passed.values())}
