"""Prospective frozen-actor rollouts from the held-out reset only, with no physics."""
import numpy as np

from tools91.launch91 import digest, require
from tools94.common94 import arrays
from tools110.course import command
from tools111.control import SPEED_GAIN, heading_rate, yaw_from_quat
from tools113.common import SEEDS, grid_case, roster
from tools113.control import BrakeTrim
from tools113.model import LEGS, TICKS, load_model, physical_targets, state_vector


DT = .02
POSE = physical_targets(np.zeros(16)).astype(np.float32)


def actor_input(state, vx, wz, last_action, previous_gyro):
    roll, pitch = state[1:3]
    gyro = .97*state[6:9] + .03*previous_gyro
    gravity = (np.sin(pitch), -np.cos(pitch)*np.sin(roll), -np.cos(pitch)*np.cos(roll))
    frame = np.zeros(57, np.float32)
    frame[:3] = .25*gyro
    frame[3:6] = gravity
    frame[6:9] = (2.*np.clip(vx,-3.,3.), 0., .25*wz)
    frame[9:25][LEGS] = state[9:21].astype(np.float32) - POSE[LEGS]
    frame[25:41] = state[21:]*.05
    frame[41:] = last_action
    return frame, gyro


def onnx_policy(session):
    def apply(tick, frame, history):
        return session.run(["nn_output"], {"nn_input0":frame[None,:], "nn_input1":history[None,:,:]})[0][0]
    return apply


def advance_history(history, frame):
    history[:-1] = history[1:].copy()
    history[-1] = frame


def check_actor(apply, training_path):
    """Rebuild all 600 policy frames from one *training* archive, without integrating physics."""
    with np.load(training_path, allow_pickle=False) as z:
        state = state_vector(z["qpos"],z["qvel"])
        vx, wz, truth = z["actor_vx"], z["actor_wz"], z["raw"]
    last = np.zeros(16, np.float32)
    gyro = np.zeros(3)
    history = None
    worst = 0.
    for k in range(TICKS):
        frame, gyro = actor_input(state[k],vx[k],wz[k],last,gyro)
        if history is None:
            history = np.tile(frame,(10,1))
        action = apply(k,frame,history)
        worst = max(worst,float(np.max(np.abs(action-truth[k]))))
        advance_history(history,frame)
        last = truth[k]
    require(worst <= 3e-5,"forecast policy interface differs from the saved native actor: "+str(worst))
    return {"frames":TICKS,"maximum_raw_action_error":worst,"trace_sha256":digest(training_path)}


def horizontal_velocity(state, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = state[3:5]
    return np.array([c*vx-s*vy,s*vx+c*vy])


def yaw_rate(state):
    roll, pitch = state[1:3]
    require(abs(np.cos(pitch)) > .05, "predicted orientation left heading coordinates")
    return (np.sin(roll)*state[7]+np.cos(roll)*state[8])/np.cos(pitch)


def rollout(world, q0, v0, enabled, apply):
    state = np.empty((TICKS+1,37))
    xy = np.empty((TICKS+1,2))
    yaw = np.empty(TICKS+1)
    issued = np.empty((TICKS,16),np.float32)
    targets = np.empty((TICKS,16))
    commands = np.empty((TICKS,2))
    state[0], xy[0], yaw[0] = state_vector(q0,v0), q0[:2], yaw_from_quat(q0[3:7])
    trim = BrakeTrim() if enabled else None
    previous = np.zeros(16,np.float32)
    gyro = np.zeros(3)
    previous_vx = 0.
    history = None
    for k in range(TICKS):
        current = state[k]
        request = command(k)
        increment = trim.update(request,previous_vx,current[21:][3::4]) if enabled else 0.
        vx = SPEED_GAIN*request + increment
        wz = heading_rate(yaw[0],yaw[k],request > .01)
        commands[k] = (vx,wz)
        frame, gyro = actor_input(current,vx,wz,previous,gyro)
        if history is None:
            history = np.tile(frame,(10,1))
        output = np.asarray(apply(k,frame,history))
        require(output.shape == (16,) and np.isfinite(output).all(),"invalid imagined actor output")
        issued[k] = output
        advance_history(history,frame)
        # Both native and imagined PD act on the previous decision, including zero at reset.
        targets[k] = physical_targets(previous)
        following = world.step(k,current,targets[k])
        require(np.isfinite(following).all(),"nonfinite isolated world forecast")
        state[k+1] = following
        yaw[k+1] = yaw[k]+.5*DT*(yaw_rate(current)+yaw_rate(following))
        xy[k+1] = xy[k]+.5*DT*(horizontal_velocity(current,yaw[k])+horizontal_velocity(following,yaw[k+1]))
        previous, previous_vx = issued[k], vx
    return dict(state=state,xy=xy,yaw=yaw,issued=issued,targets=targets,commands=commands)


def metrics(trace):
    x = trace["xy"][:,0]
    times = []
    for marker in (3.,4.,5.,6.):
        indices = np.flatnonzero((x[:-1]<marker)&(x[1:]>=marker))
        if not len(indices):
            break
        k = int(indices[0])
        times.append(DT*(k+(marker-x[k])/(x[k+1]-x[k])))
    speed = 3./(times[-1]-times[0]) if len(times)==4 and np.all(np.diff(times)>0) else None
    return dict(cruise_m_s=None if speed is None else float(speed),
                stop_peak_m_s=float(np.linalg.norm(trace["state"][575:,3:5],axis=1).max()))


def differences(off, on):
    return {k:None if on[k] is None or off[k] is None else on[k]-off[k] for k in off}


def initial(path):
    with np.load(path, allow_pickle=False) as z:
        return z["qpos"][0], z["qvel"][0]


def make_forecast(out, model_paths):
    from tools110.native import MODEL_SHA, load_policy
    apply = onnx_policy(load_policy())
    training_path = next(p for _,domain,p in roster() if domain=="compound")
    check = check_actor(apply,training_path)
    rows = []
    for seed in SEEDS:
        reference, _ = grid_case(seed)
        q0, v0 = initial(reference)
        path = model_paths[seed]
        world = load_model(path)
        require(world.heldout == seed and world.uses_action,"wrong prospective fold or model arm")
        packed, scores = {}, {}
        for arm in ("off","on"):
            trace = rollout(world,q0,v0,arm=="on",apply)
            packed.update({arm+"_"+key:value for key,value in trace.items()})
            scores[arm] = metrics(trace)
        file = out/f"pred_{seed}.npz"
        arrays(file,packed)
        rows.append(dict(seed=seed,model_sha256=digest(path),file=file.name,sha256=digest(file),**scores,
                         delta=differences(scores["off"],scores["on"])))
    return dict(rows=rows,actor_forwards=2*TICKS*len(SEEDS),model_steps=2*TICKS*len(SEEDS),
                interface_audit=check,actor_sha256=MODEL_SHA,physics=0,policy_updates=0)


def replay_prediction(file, model_file, ref_path, seed):
    world = load_model(model_file)
    require(world.heldout == seed and world.uses_action,"forecast used an ineligible checkpoint")
    q0, v0 = initial(ref_path)
    scores = {}
    with np.load(file,allow_pickle=False) as bundle:
        for arm in ("off","on"):
            actions = bundle[arm+"_issued"]
            trace = rollout(world,q0,v0,arm=="on",lambda k,frame,history: actions[k])
            for key, values in trace.items():
                np.testing.assert_allclose(values,bundle[arm+"_"+key],atol=1e-8,rtol=1e-10)
            scores[arm] = metrics(trace)
    return dict(**scores,delta=differences(scores["off"],scores["on"]))
