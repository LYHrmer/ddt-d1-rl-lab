"""Fixed-input tests of the frozen D1 vendor policy at the first parking request."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from runpy import run_path
from xml.etree import ElementTree as ET

import numpy as np

from tools91.launch91 import ROOT, digest, read, require
from tools94.common94 import arrays, write
from tools111.kinematics import vendor_targets
from tools113.results import compare_tree
from tools114.common import BLIND, DEV, PUBLIC

# Importing the d1_rl package eagerly imports its MuJoCo environment. Run just the
# registered NumPy observation source, so input reconstruction and CI need no simulator.
INTERFACE = ROOT / "training/d1_rl/interface.py"
INTERFACE_SHA = "69f8020b5cd16d6625fcbd681e975760d04e7a5ff21830589fbb432c57866232"
require(digest(INTERFACE) == INTERFACE_SHA, "changed frozen observation interface")
_interface = run_path(str(INTERFACE))
WHEEL_INDICES = _interface["WHEEL_INDICES"]
build_observation = _interface["build_observation"]


SEEDS = (*DEV, *BLIND)
TICK = 525
ARMS = ("original", "equal_encoders", "equal_previous_actions")
ENCODER_COLUMNS = 25 + WHEEL_INDICES
FEEDBACK_COLUMNS = 41 + WHEEL_INDICES
RAW_LIMIT = 1e-5
PHYSICAL_LIMIT = 1e-5
MIN_GAP_RAD_S = .05
REDUCTION = .25
BUNDLE = "policy_probe.npz"
JOURNAL = "policy_calls.jsonl"
IMU_RECEIPT = "imu_contract.json"
CFG = ROOT / "training/configs/pilot03d.json"
CFG_SHA = "9cd6546e8a8df50c7d578a664003681fc79ddca7fa0a27b80158f0f5b6842bea"
INDEX = PUBLIC / "closed_loop.json"
INDEX_SHA = "8c7be8254f8ac3e6631e1cab39fa9c6140f3968f402f5697ba1c82d9c55d04b5"
ACTOR = ROOT / "src/ddt_ros2_control/controller/rl_controller/config/d1/flat.onnx"
ACTOR_SHA = "698ca9339eb1eda8812d552bbace2977959a7f3f4f41c58c9dd84f3fce5d9850"
ROBOT_SHA = "64b7ba2887d011cfaefe22fe589c71596c0fa2ba0c0d64c5533c26fbe7e64757"
PHYSICAL_SCALE = np.tile([.125, .25, .25, 5.], 4)
POSE = vendor_targets(np.zeros(16)).astype(np.float32)


@dataclass
class Source:
    seed: int
    frames: np.ndarray
    raw: np.ndarray
    trace_sha256: str


def _expected_imu_contract():
    """Public checks use the tracked configuration and registered sensor convention."""
    cfg = read(CFG)
    require(digest(CFG) == CFG_SHA and cfg["robot_xml_sha256"] == ROBOT_SHA, "changed registered D1 configuration")
    source = (Path(cfg["scene"]).parent / "robot.xml").as_posix()
    return dict(schema="d1-parking115-imu-v1", config_sha256=CFG_SHA, robot_xml=source, robot_xml_sha256=ROBOT_SHA,
                base_body="base_link", free_joint="floating_base_joint", imu_site="trunk_imu",
                position_in_body_m=[0.,0.,0.], quaternion_in_body_wxyz=[1.,0.,0.,0.],
                gyro={"name":"trunk_gyro","site":"trunk_imu"},
                attitude={"name":"trunk_quat","objtype":"site","objname":"trunk_imu"},
                gyro_qvel_columns=[3,4,5])


def _check_imu_contract():
    """Read and hash the installed vendor XML once, then extract the fields supporting the gyro mapping."""
    expected = _expected_imu_contract()
    content = (ROOT/expected["robot_xml"]).read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    require(sha == expected["robot_xml_sha256"], "unregistered D1 IMU geometry")
    xml = ET.fromstring(content)
    base = xml.find("./worldbody/body[@name='base_link']")
    require(base is not None, "missing native base body")
    site = base.find("site[@name='trunk_imu']")
    joint = base.find("joint[@type='free']")
    gyro = xml.find("./sensor/gyro[@name='trunk_gyro']")
    attitude = xml.find("./sensor/framequat[@name='trunk_quat']")
    require(all(node is not None for node in (site,joint,gyro,attitude)), "missing native IMU or free joint")
    require(not (set(site.attrib) & {"euler","axisangle","xyaxes","zaxis","fromto"})
            and not xml.findall(".//default/site"), "unexpected IMU mount override")
    actual = dict(schema=expected["schema"],config_sha256=CFG_SHA,robot_xml=expected["robot_xml"],
                  robot_xml_sha256=sha,base_body=base.get("name"),free_joint=joint.get("name"),
                  imu_site=site.get("name"),position_in_body_m=[float(x) for x in site.get("pos","0 0 0").split()],
                  quaternion_in_body_wxyz=[float(x) for x in site.get("quat","1 0 0 0").split()],
                  gyro={k:gyro.get(k) for k in ("name","site")},
                  attitude={k:attitude.get(k) for k in ("name","objtype","objname")},gyro_qvel_columns=[3,4,5])
    require(actual == expected, "IMU is not aligned with the registered free-joint body")
    return actual


def rebuild_inputs(z, tick=TICK):
    """Return 11 independent rows, oldest previous frame through the current frame."""
    n = len(z["raw"])
    require(10 <= tick < n, "probe requires ten actual previous frames")
    q, dq, imu = z["qpos"], z["qvel"], z["imu_quat"]
    require(q.shape == (n+1,23) and dq.shape == (n+1,22) and imu.shape == (n,4), "invalid native endpoints")
    indices = np.arange(tick-10,tick+1)
    # Warm up the gyro filter from the actual episode reset, retaining all its prior state.
    gyro = np.zeros(3)
    for omega in dq[:indices[0],3:6]:
        gyro = .97*omega+.03*gyro
    frames = np.empty((11,57),np.float32)
    for i,k in enumerate(indices):
        previous_raw = z["raw"][k-1] if k else np.zeros(16,np.float32)
        command = np.array([z["actor_vx"][k],0.,z["actor_wz"][k]])
        frame, gyro = build_observation(dq[k,3:6],imu[k],command,q[k,7:],dq[k,6:],previous_raw,gyro)
        # The vendor overrides joint offsets using float32 subtraction. Later the IMU heading
        # controller and encoder compensation write the actual command given to this same frame.
        frame[9:25] = q[k,7:].astype(np.float32) - POSE
        frame[9+WHEEL_INDICES] = 0.
        expected_cmd = np.array([2.*command[0],0.,.25*command[2]],np.float32)
        np.testing.assert_array_equal(z["actor_frame_command"][k],expected_cmd)
        frame[6:9] = z["actor_frame_command"][k]
        frames[i] = frame
    np.testing.assert_array_equal(dq[indices,9::4],z["encoder_wheel_rad_s"][indices])
    np.testing.assert_array_equal(frames[:,ENCODER_COLUMNS], (.05*z["encoder_wheel_rad_s"][indices]).astype(np.float32))
    np.testing.assert_array_equal(frames[:,41:],z["actor_previous_raw"][indices])
    np.testing.assert_array_equal(frames[:-1,41:],z["actor_history_previous_raw"][tick])
    return frames


def prepare(seed):
    """Rebuild a registered sample without loading a runtime or evaluating a policy."""
    require(seed in SEEDS and digest(INDEX) == INDEX_SHA, "wrong parking source or seed")
    index = read(INDEX)
    case, = [r for r in index["cases"] if r["seed"] == seed]
    tag = f"native_{seed}"
    stage = index["stages"][tag]
    folder = PUBLIC / tag
    guard = index["guards"][stage["guard"]]
    require(case["complete"] and case["gate_pass"] and guard["child_started"] and guard["stage"] == tag
            and stage["status"] == guard["status"] == "COMPLETE"
            and digest(folder/"report.json") == stage["files"]["report.json"], "incomplete parking source")
    row = read(folder/"report.json")["result"]
    path = folder/"trace.npz"
    sha = digest(path)
    require(sha == case["trace_sha256"] == stage["files"]["trace.npz"] == row["trace_sha256"]
            and row["seed"] == seed and row["actor_sha256"] == ACTOR_SHA and row["motion"]["ticks"] == 600,
            "changed native source or vendor policy")
    keys = ("raw","qpos","qvel","imu_quat","actor_vx","actor_wz","actor_frame_command","encoder_wheel_rad_s",
            "actor_previous_raw","actor_history_previous_raw")
    with np.load(path,allow_pickle=False) as archive:
        z = {key:archive[key] for key in keys}
        require(archive["parking_active"][TICK] and not archive["parking_active"][TICK-1]
                and archive["driver_command_xyz"][TICK-1,0] > 0.
                and np.all(archive["driver_command_xyz"][TICK] == 0.), "changed parking boundary")
    return Source(seed, rebuild_inputs(z), z["raw"][TICK].copy(), sha)


def altered(frames, arm):
    """Replace only the chosen four slots by their simultaneous mean in each frame."""
    require(frames.shape == (11,57) and frames.dtype == np.float32 and np.isfinite(frames).all()
            and arm in ARMS, "invalid fixed policy inputs or arm")
    inputs = frames.copy()
    if arm != "original":
        columns = ENCODER_COLUMNS if arm == "equal_encoders" else FEEDBACK_COLUMNS
        means = frames[:,columns].astype(float).mean(axis=1)
        inputs[:,columns] = means[:,None]
    return inputs


def _parity(actual, reference):
    errors = abs(np.asarray(actual,dtype=float)-reference)
    wheel_error = float((errors[WHEEL_INDICES]*5.).max())
    legs = (errors*PHYSICAL_SCALE).reshape(4,4)[:,:3]
    leg_error = float(legs.max())
    return dict(max_raw_error=float(errors.max()), max_leg_target_error_rad=leg_error,
                max_wheel_target_error_rad_s=wheel_error,
                all_outputs_match=bool(errors.max() <= RAW_LIMIT and max(leg_error,wheel_error) <= PHYSICAL_LIMIT))


def wheel_gap(raw):
    wheels = 5.*np.asarray(raw,dtype=float)[WHEEL_INDICES]
    return float(wheels[1]-wheels[[0,2,3]].mean())


def summarize(sources,outputs):
    """Score joint-output parity first; report fixed-input sensitivities only if all six match."""
    require([s.seed for s in sources] == list(SEEDS) and outputs.shape == (len(SEEDS),len(ARMS),16)
            and outputs.dtype == np.float32 and np.isfinite(outputs[:,0]).all(), "incomplete baseline roster")
    parity = [_parity(v[0],s.raw) for s,v in zip(sources,outputs)]
    passed = all(p["all_outputs_match"] for p in parity)
    require(np.isfinite(outputs[:,1:]).all() if passed else np.isnan(outputs[:,1:]).all(),
            "counterfactual inference did not respect the all-seed parity gate")
    rows = []
    for i,source in enumerate(sources):
        original = wheel_gap(outputs[i,0])
        variants = {}
        if passed:
            for a,arm in enumerate(ARMS[1:],start=1):
                value = wheel_gap(outputs[i,a])
                relative = 1.-abs(value)/abs(original) if abs(original) > MIN_GAP_RAD_S else None
                variants[arm] = dict(right_front_gap_rad_s=value,delta_gap_rad_s=value-original,
                                     absolute_gap_reduction_fraction=relative,
                                     reaches_reduction=relative is not None and relative >= REDUCTION)
        rows.append(dict(seed=source.seed,trace_sha256=source.trace_sha256,parity=parity[i],
                         reference_wheels_rad_s=(5.*source.raw[WHEEL_INDICES].astype(float)).tolist(),
                         original_gap_rad_s=original,variants=variants))
    counts = {arm:sum(r["variants"][arm]["reaches_reduction"] for r in rows) for arm in ARMS[1:]} if passed else {}
    return dict(tick=TICK,source_index_sha256=INDEX_SHA,actor_sha256=ACTOR_SHA,robot_xml_sha256=ROBOT_SHA,
                arms=list(ARMS),rows=rows,baseline_parity=passed,reduction_fraction=REDUCTION,
                minimum_baseline_gap_rad_s=MIN_GAP_RAD_S, counts_reaching_reduction=counts,
                all_six_reach_reduction={arm:n == len(SEEDS) for arm,n in counts.items()},
                actor_forwards=len(SEEDS)*(len(ARMS) if passed else 1),simulation_steps=0,world_updates=0,policy_updates=0)


def open_actor():
    """Load precisely the frozen ONNX with the producer's sequential, single-thread CPU configuration."""
    import onnxruntime as ort

    require(digest(ACTOR) == ACTOR_SHA,"different frozen vendor actor")
    cfg = ort.SessionOptions()
    cfg.intra_op_num_threads = cfg.inter_op_num_threads = 1
    cfg.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(ACTOR),cfg,providers=["CPUExecutionProvider"])
    require([(v.name,v.shape) for v in session.get_inputs()] ==
            [("nn_input0",[1,57]),("nn_input1",[1,10,57])] and session.get_providers() == ["CPUExecutionProvider"],
            "unexpected vendor policy or provider")
    return session


def _forward(session,frames):
    value, = session.run(["nn_output"],{"nn_input0":frames[-1:],"nn_input1":frames[None,:-1]})
    require(value.shape == (1,16) and value.dtype == np.float32 and np.isfinite(value).all(),
            "invalid vendor policy output")
    return value[0]


def run_probe(out):
    """Execute 6 original forwards, conditionally followed by 12 isolated input substitutions."""
    out = Path(out)
    scratch = out/"partial_policy_outputs.npy"
    require(out.is_dir() and not any((out/name).exists() for name in (BUNDLE,JOURNAL,IMU_RECEIPT,scratch.name)),
            "preserve previous policy probe")
    write(out/IMU_RECEIPT,_check_imu_contract())
    sources = [prepare(s) for s in SEEDS]
    inputs = np.stack([[altered(s.frames,a) for a in ARMS] for s in sources])
    with scratch.open("xb") as stream:
        np.save(stream,np.full((len(SEEDS),len(ARMS),16),np.nan,dtype=np.float32))
    outputs = np.lib.format.open_memmap(scratch,mode="r+")
    actor = open_actor()
    with (out/JOURNAL).open("x",buffering=1) as log:
        for a,arm in enumerate(ARMS):
            if a and not all(_parity(outputs[i,0],s.raw)["all_outputs_match"] for i,s in enumerate(sources)):
                break
            for i,source in enumerate(sources):
                entry = {"arm":arm,"seed":source.seed}
                log.write(json.dumps(dict(entry,event="attempt"))+"\n")
                outputs[i,a] = _forward(actor,inputs[i,a])
                outputs.flush()
                log.write(json.dumps(dict(entry,event="complete"))+"\n")
    result = summarize(sources,outputs)
    arrays(out/BUNDLE,dict(inputs=inputs,outputs=outputs,registered_raw=np.stack([s.raw for s in sources])))
    scratch.unlink()
    return dict(**result,bundle_sha256=digest(out/BUNDLE),journal_sha256=digest(out/JOURNAL),
                imu_contract_sha256=digest(out/IMU_RECEIPT))


def verify(out,expected):
    """Check saved inputs, counts and output-derived results without new policy evaluations."""
    out = Path(out)
    imu_sha = digest(out/IMU_RECEIPT)
    require(imu_sha == expected["imu_contract_sha256"] and read(out/IMU_RECEIPT) == _expected_imu_contract(),
            "changed native IMU receipt or protocol")
    sources = [prepare(s) for s in SEEDS]
    with np.load(out/BUNDLE,allow_pickle=False) as z:
        require(set(z.files) == {"inputs","outputs","registered_raw"}, "incorrect evidence keys")
        inputs = np.stack([[altered(s.frames,a) for a in ARMS] for s in sources])
        require(z["inputs"].dtype == inputs.dtype, "changed ONNX input precision")
        np.testing.assert_array_equal(z["inputs"],inputs)
        np.testing.assert_array_equal(z["registered_raw"],np.stack([s.raw for s in sources]))
        result = summarize(sources,z["outputs"])
    events = [json.loads(line) for line in (out/JOURNAL).read_text().splitlines()]
    actual_arms = ARMS if result["baseline_parity"] else ARMS[:1]
    wanted = [dict(arm=a,seed=s,event=e) for a in actual_arms for s in SEEDS for e in ("attempt","complete")]
    require(events == wanted,"unexpected inference ledger")
    compare_tree(dict(**result,bundle_sha256=digest(out/BUNDLE),journal_sha256=digest(out/JOURNAL),
                      imu_contract_sha256=imu_sha),expected)
    return dict(archived_forwards=result["actor_forwards"],new_forwards=0,tensor_pairs_checked=len(SEEDS)*len(ARMS))
