"""Change only the floor contact representation and run the original 50 Hz D1 control loop."""
from collections import defaultdict
from tempfile import TemporaryDirectory

import mujoco as mj
import numpy as np

from tools91.launch91 import require
from tools94.common94 import arrays, digest, write
from tools100.sensor100 import _wheel_cylinders
from tools110.course import CFG, TICKS, X_GRID, Y_GRID, scene
from tools110.native import MODEL_SHA, YAML_SHA, Plant, load_policy
from tools111.control import SPEED_GAIN
from tools111.course import score
from tools111.native import drive
from tools112.surface import PROBES, SEEDS, reference


def flatten(model):
    require(model.nhfield == 1 and tuple(model.hfield_nrow) == (len(Y_GRID),)
            and tuple(model.hfield_ncol) == (len(X_GRID),), "heightfield grid differs from the compound route")
    floor = model.geom("floor").id
    np.testing.assert_array_equal(model.hfield_size[0], [11., 2.2, .25, .05])
    np.testing.assert_array_equal(model.geom_pos[floor], [6., 0., -.025])
    model.hfield_data[:] = -model.geom_pos[floor,2] / model.hfield_size[0,2]
    require(float(model.hfield_data.min()) == float(model.hfield_data.max()), "nonuniform floor")


def wheel_contacts(model, data, floor):
    ids = _wheel_cylinders(model, mj.mjtGeom.mjGEOM_CYLINDER)
    by_wheel = [[] for _ in ids]
    for c in data.contact[:data.ncon]:
        a, b = int(c.geom1), int(c.geom2)
        if floor not in (a,b):
            continue
        wheel = b if a == floor else a
        matches = np.flatnonzero(ids == wheel)
        if len(matches):
            normal = c.frame[:3] * (1. if a == floor else -1.)
            by_wheel[int(matches[0])].append(dict(position_m=c.pos.tolist(), distance_m=float(c.dist),
                                                   normal_floor_to_wheel=normal.tolist()))
    return by_wheel


def survey():
    with TemporaryDirectory(prefix="d1-equal-height-") as temp:
        plane = mj.MjModel.from_xml_path(str(scene(temp,"flat")))
        field = mj.MjModel.from_xml_path(str(scene(temp,"ridge")))
    flatten(field)
    floor = plane.geom("floor").id
    require(field.geom("floor").id == floor and plane.ngeom == field.ngeom and
            plane.geom_type[floor] == mj.mjtGeom.mjGEOM_PLANE and field.geom_type[floor] == mj.mjtGeom.mjGEOM_HFIELD,
            "surface types or robot geometry changed")
    matched = ("geom_friction","geom_priority","geom_condim","geom_solref","geom_solimp","geom_margin","geom_gap",
               "geom_contype","geom_conaffinity","geom_matid","geom_bodyid","body_mass","body_inertia","body_pos",
               "body_quat","jnt_range","dof_damping","dof_frictionloss","dof_armature","actuator_gear")
    for key in matched:
        np.testing.assert_array_equal(getattr(plane,key),getattr(field,key),err_msg=key)
    robot = np.arange(plane.ngeom) != floor
    for key in ("geom_type","geom_size","geom_pos","geom_quat","geom_dataid"):
        np.testing.assert_array_equal(getattr(plane,key)[robot],getattr(field,key)[robot],err_msg=key)
    np.testing.assert_array_equal(plane.opt.gravity,field.opt.gravity)
    require(plane.opt.integrator == field.opt.integrator and plane.opt.timestep == field.opt.timestep, "physics differs")
    path, ref = reference(SEEDS[0])
    with np.load(path,allow_pickle=False) as z:
        q, dq = z["qpos"][575], z["qvel"][575]
    heights, contacts = {}, {}
    for name, model in (("plane",plane),("heightfield",field)):
        data = mj.MjData(model)
        data.qpos[:], data.qvel[:] = q, dq
        mj.mj_fwdPosition(model,data)
        contacts[name] = wheel_contacts(model,data,floor)
        heights[name] = []
        geom_id = np.empty(1,np.int32)
        for x,y in PROBES:
            distance = mj.mj_ray(model,data,np.array([x,y,1.]),np.array([0.,0.,-1.]),None,True,-1,geom_id)
            require(distance >= 0. and geom_id[0] == floor, f"floor ray missed at {(x,y)}")
            heights[name].append(float(1.-distance))
        require(data.time == 0., "static survey advanced physics")
    maximum = float(np.max(np.abs(np.subtract(heights["plane"],heights["heightfield"]))))
    require(maximum <= .0001 and np.max(np.abs(list(heights.values()))) <= .0001, "unequal physical surface heights")
    return dict(positions_m=PROBES, heights_m=heights, max_difference_m=maximum, rays=2*len(PROBES), physics=0,
                equal_fields=matched, grid_shape=[len(Y_GRID),len(X_GRID)], heightfield_value=float(field.hfield_data[0]),
                contact_pose=dict(seed=SEEDS[0],tick=575,trace_sha256=ref["trace_sha256"],contacts=contacts))


def episode(out, seed):
    path, plane = reference(seed)
    settings = dict(seed=seed,layout="flat",arm="heading_on",gain=SPEED_GAIN,surface="level_heightfield")
    rows = defaultdict(list)
    counts = dict(decisions=0,servo=0,physics=0,packets=0,ray_queries=0)
    session = load_policy()
    try:
        with TemporaryDirectory(prefix="d1-level-field-") as directory:
            plant = Plant(config=CFG,scene=scene(directory,"ridge"),seed=seed,arm="B",eval_delay_ticks=1,
                          eval_command_sequence=[{"start_s":0.,"vx":0.,"wz":0.}],max_transitions=TICKS)
            try:
                flatten(plant.model)
                plant.reset(seed=seed)
                with np.load(path,allow_pickle=False) as z:
                    np.testing.assert_array_equal(plant.data.qpos,z["qpos"][0])
                    np.testing.assert_array_equal(plant.data.qvel,z["qvel"][0])
                drive(plant,session,rows,counts,"flat","heading_on",SPEED_GAIN)
            finally:
                plant.close()
        trace = {key:np.asarray(values) for key,values in rows.items()}
        motion, effects = score(trace,"flat")
        n = motion["ticks"]
        require(counts["decisions"] == n and counts["servo"] == 5*n and counts["physics"] == 10*n <= 6000
                and counts["ray_queries"] == 1040*counts["packets"] <= 74880,"diagnostic count mismatch")
        arrays(out/"trace.npz",trace)
        return dict(**settings,status="COMPLETE",counts=counts,motion=motion,effects=effects,
                    trace_sha256=digest(out/"trace.npz"),reference_sha256=plane["trace_sha256"],
                    actor_sha256=MODEL_SHA,actuator_config_sha256=YAML_SHA,mujoco_version=mj.__version__)
    except BaseException as error:
        arrays(out/"partial.npz",{key:np.asarray(v) for key,v in rows.items()})
        write(out/"partial_counts.json",dict(error=repr(error),counts=counts))
        raise
