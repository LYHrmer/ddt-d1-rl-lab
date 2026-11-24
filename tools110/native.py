"""Synchronous D1 manufacturer policy benchmark, with measured wheel support and native clocks."""
from __future__ import annotations

from tempfile import TemporaryDirectory

import mujoco as mj
import numpy as np

from tools94.common94 import ROOT, arrays, digest, write
from tools100.sensor100 import _wheel_cylinders
from tools110.course import CFG, TICKS, assess, command, height, measured_terrain, scene, set_floor
from training.d1_rl.delay_env18 import D1Delay18Env
from training.d1_rl.interface import DEFAULT_Q, TORQUE_LIMIT, WHEEL_INDICES, build_observation


VENDOR = ROOT/"src/ddt_ros2_control/controller/rl_controller/config/d1"
MODEL_SHA = "698ca9339eb1eda8812d552bbace2977959a7f3f4f41c58c9dd84f3fce5d9850"
YAML_SHA = "cedf96cedd405b61c7933d0128cdb6954668501115fbce48c2ae2d65f448c6fe"
POSE = np.array([.1,.8,-1.5,0., -.1,.8,-1.5,0., .1,1.,-1.5,0., -.1,1.,-1.5,0.])
KP = np.tile([100.,60.,60.,10.],4)
KD = np.tile([4.,1.5,1.5,.5],4)
SCALE = np.tile([.125,.25,.25,.25],4)


class Plant(D1Delay18Env):
    def _place_static_pose(self, joint_q, yaw):
        # Preserve the original reset perturbations about the registered vendor posture.
        super()._place_static_pose(joint_q+(POSE-DEFAULT_Q), yaw)


def load_policy():
    import onnxruntime as ort
    path = VENDOR/"flat.onnx"
    if digest(path) != MODEL_SHA or digest(VENDOR/"controllers.yaml") != YAML_SHA:
        raise ValueError("vendor model or actuator configuration changed")
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])
    signature = [(i.name, i.shape) for i in session.get_inputs()]
    if signature != [("nn_input0", [1,57]), ("nn_input1", [1,10,57])]:
        raise ValueError("wrong vendor policy inputs")
    return session


def actor_frame(plant, cmd, last, gyro):
    frame, gyro = build_observation(plant._sensor("trunk_gyro"), plant._sensor("trunk_quat"),
                                    np.array([cmd,0.,0.]), plant.data.qpos[plant.qpos_ids],
                                    plant.data.qvel[plant.dof_ids], last, gyro)
    frame[6] = 2.*np.clip(cmd, -3., 3.)
    pose = plant.data.qpos[plant.qpos_ids].astype(np.float32)-POSE.astype(np.float32)
    pose[WHEEL_INDICES] = 0.
    frame[9:25] = pose
    return frame, gyro


def torque(action, q, dq):
    # The vendor P wheel path is tau=10*(.25*a)-.5*dq, exactly a .5 wheel servo with target 5*a.
    action = np.asarray(action, dtype=np.float64)
    raw = KP*(POSE+SCALE*action-q)-KD*dq
    raw[WHEEL_INDICES] = KD[WHEEL_INDICES]*(5.*action[WHEEL_INDICES]-dq[WHEEL_INDICES])
    return np.clip(raw,-TORQUE_LIMIT,TORQUE_LIMIT), np.abs(raw)>=TORQUE_LIMIT


def support(plant, mapping, target_id):
    after = np.zeros(4, bool)
    top = np.zeros(4, bool)
    force = np.empty(6)
    for i, c in enumerate(plant.data.contact[:plant.data.ncon]):
        a,b = int(c.geom1),int(c.geom2)
        if a in mapping:
            wheel, road = mapping[a], b
        elif b in mapping:
            wheel, road = mapping[b], a
        else:
            continue
        if plant.model.geom_bodyid[road] != 0:
            continue
        mj.mj_contactForce(plant.model, plant.data, i, force)
        if force[0] <= 1.:
            continue
        x,y,z = c.pos
        if x>7. and abs(y)<=1.:
            after[wheel] = True
        if (road==target_id and 4.002<x<5.498 and abs(y)<1. and abs(z-.03)<.002
                and abs(c.frame[2])>.95):
            top[wheel] = True
    return after, top


def episode(out, seed, terrain, gain, session):
    out.mkdir(exist_ok=False)
    write(out/"reservation.json",dict(seed=seed, terrain=terrain, gain=gain))
    rows={k:[] for k in ("time","xy","vxy","qpos","qvel","wheels_xy","cmd","actor_cmd","issued","applied",
                         "wheel_saturation","exit_contact","top_contact","fallen","illegal","tilt","clearance")}
    count = {"decisions":0,"servo":0,"physics":0,"policy_forwards":0}
    try:
        with TemporaryDirectory(prefix="d1-speed110-") as directory:
            plant=Plant(config=CFG,scene=scene(directory,terrain),seed=seed,arm="B",eval_delay_ticks=1,
                        eval_command_sequence=[{"start_s":0., "vx":0., "wz":0.}],max_transitions=TICKS)
            try:
                set_floor(plant.model,terrain)
                rays=measured_terrain(plant.model,terrain)
                plant.reset(seed=seed)
                wheels=_wheel_cylinders(plant.model,mj.mjtGeom.mjGEOM_CYLINDER)
                slot={int(g):i for i,g in enumerate(wheels)}
                target_id=plant.model.geom("step110").id if terrain=="step" else -1
                data=plant.data

                def append_state():
                    rows["time"].append(float(data.time))
                    rows["qpos"].append(data.qpos.copy())
                    rows["qvel"].append(data.qvel.copy())
                    rows["xy"].append(data.qpos[:2].copy())
                    rows["vxy"].append(data.qvel[:2].copy())
                    rows["wheels_xy"].append(data.geom_xpos[wheels,:2].copy())

                append_state()
                last=np.zeros(16,np.float32)
                gyro=np.zeros(3)
                history=None
                for k in range(TICKS):
                    count["decisions"]+=1
                    cmd=command(k)
                    frame,gyro=actor_frame(plant, gain*cmd,last,gyro)
                    if history is None:
                        history=np.tile(frame,(10,1))
                    fresh=session.run(["nn_output"], {"nn_input0":frame[None,:],"nn_input1":history[None,:,:]})[0][0]
                    count["policy_forwards"]+=1
                    if fresh.shape!=(16,) or not np.isfinite(fresh).all():
                        raise ValueError("nonfinite 16-output actor")
                    applied=last
                    history[:-1]=history[1:].copy()
                    history[-1]=frame
                    last=fresh
                    sat,exits,tops=(np.zeros(4,bool) for _ in range(3))
                    illegal=False
                    for substep in range(10):
                        if substep%2==0:
                            tau,tau_sat=torque(applied,data.qpos[plant.qpos_ids],data.qvel[plant.dof_ids])
                            data.qfrc_applied[:]=0.
                            data.qfrc_applied[plant.dof_ids]=tau
                            data.ctrl[:]=0.
                            count["servo"]+=1
                            sat |= tau_sat[WHEEL_INDICES]
                        count["physics"]+=1
                        mj.mj_step(plant.model,data)
                        plant._check_finite_state()
                        illegal_now,_=plant._contact_diagnostics()
                        illegal |= illegal_now
                        beyond,above=support(plant,slot,target_id)
                        exits |= beyond
                        tops |= above
                    mj.mj_forward(plant.model,data)
                    plant._check_finite_state()
                    illegal |= plant._contact_diagnostics()[0]
                    append_state()
                    qw,qx,qy,qz=data.qpos[3:7]
                    tilt=float(np.arccos(np.clip(1.-2.*(qx*qx+qy*qy),-1.,1.)))
                    clearance=float(data.qpos[2]-height(terrain,*data.qpos[:2]))
                    fallen=clearance < plant.nominal_base_height_m-.18 or tilt>.6
                    vals=dict(cmd=cmd,actor_cmd=gain*cmd, issued=fresh,applied=applied,
                              wheel_saturation=sat, exit_contact=exits,top_contact=tops,fallen=fallen,
                              illegal=illegal, tilt=tilt,clearance=clearance)
                    for key,val in vals.items():
                        rows[key].append(val)
                    if illegal or fallen:
                        break
            finally:
                plant.close()
        trace={k:np.asarray(v) for k,v in rows.items()}
        result=assess(trace,terrain)
        n=result["ticks"]
        assert count==dict(decisions=n,servo=5*n,physics=10*n,policy_forwards=n)
        arrays(out/"trace.npz",trace)
        report=dict(status="COMPLETE",seed=seed,terrain=terrain,gain=gain,score=result,counts=count,
                    initial_pose=trace["qpos"][0].tolist(), terrain_rays=rays,trace_sha256=digest(out/"trace.npz"))
        write(out/"report.json",report)
        return report
    except BaseException as error:
        arrays(out/"partial.npz",{k:np.asarray(v) for k,v in rows.items()})
        write(out/"partial.json",dict(status="PARTIAL",error=repr(error),counts=count))
        raise
