"""Reconstruct the compound-course signal path on recorded native states and recorded actor outputs."""
import json
import math

import numpy as np

from tools91.launch91 import ROOT, digest, read, require
from tools110.course import TICKS, command
from tools111.control import SPEED_GAIN
from tools111.kinematics import Mechanism, rotation, vendor_targets
from tools111.replay import check_episode
from tools113.forecast import actor_input, advance_history
from tools113.model import state_vector
from tools116.control import PrevTrimActor


INDEX = ROOT / "reports/lidar_compound111/native.json"
INDEX_SHA = "f0c78571880588577622f6130225ebbb59230dbd25dea9faa71bc0c0cacd3ea3"
ROBOT_SHA = "64b7ba2887d011cfaefe22fe589c71596c0fa2ba0c0d64c5533c26fbe7e64757"
SEEDS = (111101,111103,111107)


class ArchiveGeometry(Mechanism):
    """Wheel-center chain from the registered robot XML. Verify it against all archived native endpoints."""
    def __init__(self):
        side = np.array([1.,-1.,1.,-1.])
        self.offsets = np.zeros((4,4,3))
        self.offsets[:,0,0] = [.282,.282,-.093,-.093]
        self.offsets[:,0,1] = side*.0775
        self.offsets[:,1,0] = side*.0905
        self.offsets[:,1,2] = -.0945
        self.offsets[:,2:,0] = .25
        self.offsets[:,3,2] = side*.052166
        quat = np.zeros((4,4,4))
        quat[...,0] = 1.
        quat[:,0] = [.499998,.5,.500002,.5]
        quat[[0,2],1] = [.500098,-.5001,.499902,-.4999]
        quat[[1,3],1] = [.499898,-.4999,.500102,-.5001]
        quat[[1,3],3] = [.363988,0.,0.,.931404]
        self.rotations = rotation(quat)
        self.limits = np.tile([[-.785398,.785398],[-1.8326,3.40339],[-2.775,-.855]],(4,1,1))


def _check_leg_targets(z,chain):
    candidates = vendor_targets(z["issued"])
    baseline = vendor_targets(z["raw"])
    before = chain.forward(baseline)[0]
    after = chain.forward(candidates)[0]
    fk = (after-before)[:,:,2]
    demand = z["lift_demand"]
    from_demand = chain.lift(baseline,demand)[:,:,2]
    np.testing.assert_allclose(from_demand,fk,rtol=0.,atol=1e-8)
    np.testing.assert_allclose(fk,z["issued_lift_fk"],rtol=0.,atol=1e-8)
    np.testing.assert_array_equal(z["applied_lift_fk"],np.vstack((np.zeros(4),z["issued_lift_fk"][:-1])))
    # Validate the public chain using native wheel measurements independent of the recorded target FK.
    q = z["qpos"]
    measured = chain.forward(q[:,7:])[0]
    world = q[:,None,:3]+(rotation(q[:,3:7])[:,None]@measured[...,None])[...,0]
    np.testing.assert_allclose(world[:,:,:2],z["wheels_xy"],rtol=0.,atol=1e-8)
    np.testing.assert_allclose(measured[:,:,2],z["measured_relative_z"],rtol=0.,atol=1e-8)
    return dict(active_lift_intervals=int(np.any(demand > 0.,axis=1).sum()),
                maximum_requested_lift_m=float(demand.max()),
                max_error_fk_from_requested_lift_m=float(abs(from_demand-fk).max()),
                max_error_logged_fk_m=float(abs(fk-z["issued_lift_fk"]).max()),
                max_error_native_wheel_xy_m=float(abs(world[:,:,:2]-z["wheels_xy"]).max()),
                max_error_native_relative_z_m=float(abs(measured[:,:,2]-z["measured_relative_z"]).max()))


def reference_speeds(preview,encoders,enabled):
    """Read-back of the R113 recurrence with the previous preview-plus-trim command in its bias estimate."""
    n = len(preview)
    require(encoders.shape == (n,4), "wrong encoder clock")
    observed = .087*np.median(encoders,axis=1)
    commands, increments = np.empty(n), np.empty(n)
    requests = np.array([command(k) for k in range(n)])
    held = 0
    bias = cruise = increment = previous = 0.
    weight = -math.expm1(-.02/.40)
    for k, request in enumerate(requests):
        held = held+1 if k and request == requests[k-1] else 0
        if held >= 10:
            bias += weight*(previous-observed[k]-bias)
            desired = np.clip((1.025-SPEED_GAIN)*request+bias,0.,.18)
            cruise += np.clip(desired-cruise,-.0036,.0036)
        cap = .18*np.clip(request/1.5,0.,1.)+max(0.,-observed[k])
        increment += np.clip(min(cruise,cap)-increment,-.0036,.0036)
        increments[k] = increment if enabled else 0.
        commands[k] = previous = preview[k] + increments[k]
    return commands, increments, observed


class _ArchivedOutputs:
    def __init__(self, raw):
        self.raw = raw
        self.tick = 0

    def run(self, outputs, inputs):
        self.current = inputs["nn_input0"].copy()
        self.history = inputs["nn_input1"].copy()
        output = self.raw[self.tick]
        self.tick += 1
        return [output[None,:]]


def _replay(z, states, enabled):
    raw, preview, encoders = z["raw"], z["actor_vx"], z["encoder_wheel_rad_s"]
    wanted, trim, observed = reference_speeds(preview,encoders,enabled)
    session = _ArchivedOutputs(raw)
    adapter = PrevTrimActor(session,lambda: encoders[session.tick],enabled=enabled)
    previous_raw = np.zeros(16,np.float32)
    gyro, history = np.zeros(3), None
    for k,vx in enumerate(preview):
        frame, gyro = actor_input(states[k],vx,z["actor_wz"][k],previous_raw,gyro)
        np.testing.assert_array_equal(frame[6:9],z["actor_frame_command"][k])
        expected = frame.copy()
        expected[6] = np.float32(2.*wanted[k])
        if history is None:
            history = np.tile(frame,(10,1))
        old_history = history.copy()
        adapter.set_preview(vx)
        action = adapter.run(["recorded"],{"nn_input0":frame[None,:], "nn_input1":history[None,:,:]})[0][0]
        np.testing.assert_array_equal(session.current[0],expected)
        np.testing.assert_array_equal(session.history[0],old_history)
        np.testing.assert_array_equal(history,old_history)
        np.testing.assert_array_equal(action,raw[k])
        advance_history(history,frame)
        previous_raw = action
    t = adapter.trace({})
    np.testing.assert_array_equal(t["preview_actor_vx"],preview)
    np.testing.assert_allclose(t["velocity_trim"],trim,rtol=0.,atol=1e-12)
    np.testing.assert_allclose(t["actor_vx"],wanted,rtol=0.,atol=1e-12)
    np.testing.assert_array_equal(t["encoder_forward_m_s"],observed)
    np.testing.assert_array_equal(t["trim_previous_actor_vx"],np.r_[0.,t["actor_vx"][:-1]])

    # R111 did not save the action-memory columns. Independently reconstruct their expected raw
    # epochs and check them against the current and history slots observed by the new adapter.
    prev = np.vstack((np.zeros(16,np.float32),raw[:-1]))
    history_indices = np.maximum(np.arange(len(raw))[:,None]+np.arange(-10,0),0)
    np.testing.assert_array_equal(t["actor_previous_raw"],prev)
    np.testing.assert_array_equal(t["actor_history_previous_raw"],prev[history_indices])
    commands = z["actor_frame_command"].copy()
    commands[:,0] = (2.*wanted).astype(np.float32)
    np.testing.assert_array_equal(t["actor_history_frame_command"],commands[history_indices])
    return t


def audit():
    """Replay existing 1,800 timestamps with both switch positions, with no ONNX or physics."""
    require(digest(INDEX) == INDEX_SHA,"changed compound archive")
    index = read(INDEX)
    require(index["robot_sha256"] == ROBOT_SHA,"old trace used different geometry")
    cases = [c for c in index["cases"] if c["layout"] == "development" and c["arm"] == "both"]
    require([c["seed"] for c in cases] == list(SEEDS),"incomplete archived roster")
    chain = ArchiveGeometry()
    rows = []
    for c in cases:
        episode = c["episode"]
        p = INDEX.parent / c["file"]
        require(c["case_status"] == episode["status"] == "COMPLETE" and episode["motion"]["ticks"] == TICKS
                and episode["seed"] == c["seed"] and episode["trace_sha256"] == digest(p),
                "missing or altered individual trajectory")
        with np.load(p,allow_pickle=False) as archive:
            z = dict(archive)
        # This reconstructs the frozen lidar responses from recorded ranges, IMU and encoders.
        check_episode(z,episode)
        geometry = _check_leg_targets(z,chain)
        states = state_vector(z["qpos"],z["qvel"])
        off, on = (_replay(z,states,enabled) for enabled in (False,True))
        preview, base = z["actor_vx"], z["base_cmd"]
        np.testing.assert_array_equal(off["actor_vx"],preview)
        same_trim_masked = base + on["velocity_trim"]
        expected_delta = base - preview
        actual_delta = same_trim_masked - on["actor_vx"]
        np.testing.assert_allclose(actual_delta,expected_delta,rtol=0.,atol=1e-12)
        require(np.any(expected_delta >= .020) and geometry["maximum_requested_lift_m"] >= .005,
                "old trace did not exercise both lidar channels")
        rows.append(dict(seed=c["seed"],source=c["file"],trace_sha256=episode["trace_sha256"],
                         case_status=c["case_status"],unit_status=c["unit_status"],ticks=TICKS,
                         max_error_off_preview_m_s=float(abs(off["actor_vx"]-preview).max()),
                         lidar_slowdown_intervals=int(np.count_nonzero(expected_delta > 1e-12)),
                         max_lidar_slowdown_m_s=float(expected_delta.max()),
                         max_equal_trim_slowdown_error_m_s=float(abs(actual_delta-expected_delta).max()),
                         max_replayed_trim_m_s=float(on["velocity_trim"].max()),
                         archived_raw_differs_from_issued_intervals=int(np.any(z["raw"] != z["issued"],axis=1).sum()),
                         **geometry))
    return dict(source_index_sha256=INDEX_SHA,source_robot_sha256=ROBOT_SHA,
                reference="R111 paired wiring on recorded states and fixed native actor outputs",
                source_action_history="R111 raw outputs and stored command triplets; full ONNX inputs were not logged",
                ticks=TICKS*len(rows), switch_positions=["off","on"], history_width=10,
                equal_trim_reference="base_cmd plus the identical replayed trim on the same archived inputs",
                verified_input="full reconstructed frame: only forward command may change; history from previous frames",
                new_actor_forwards=0,physics=0,policy_updates=0,world_updates=0,passed=True,cases=rows)


if __name__ == "__main__":
    print(json.dumps(audit(),indent=2,allow_nan=False))
