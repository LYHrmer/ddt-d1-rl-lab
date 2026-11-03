# Independent D1 adapter82 core and synthetic native probe

This is a new, explicit **test-only / effort-only** entry point. It does not
replace, patch or launch `rl_controller`, `hardware_bridge`, `runtime03`,
`runtime34`, ROS, ONNX Runtime, a robot model, or any physical actuator. It is not
a repair already deployed to the existing ROS controller and gives no hardware
or stopping qualification. No model weights or frozen files are changed.

## Contract

`adapter82.hpp` is a dependency-free C++17 core with one serialized owner:

1. `reset(steady_ns)` clears filtered gyro, ten-frame history, last clipped
   action, pending request and command. Both cold start and warm reentry require
   a fresh command. Each reset creates a new token generation. When later
   connected to `actor_server34`, the wrapper **must also start a new server
   process/session**: local tokens cannot reset its private reference shaper.
2. `receive_command(vx_vy_wz, steady_ns)` rejects nonfinite values and clips
   finite commands to `[-1,1]`. The **new adapter82 command-age policy** accepts
   age up to and including 500ms; greater age faults. This is not a claim that
   the original ROS controller already had a command-age watchdog.
3. `begin_tick(sample, steady_ns)` emits raw float32 627 values: current 57,
   then the ten previous frames oldest first. Gyro is filtered once with
   `.97*current+.03*previous` in float64. Quaternion is finite, normalized wxyz;
   gravity is its inverse rotation of world `[0,0,-1]`. Command scales are
   `[2,2,.25]`; wheel position features are zero; dq scale is `.05`. Reset
   prehistory repeats the first frame. Last action is the previously accepted
   clipped action. History is appended only when inference is accepted.
4. Send this **unshaped** observation to the unchanged 34 actor server; its
   gain1.5/tau0.1s shaping must not be duplicated here. The core does not load,
   identify, hash-check or run a model. A future wrapper must bind the exact
   completed25 model and34 server identities before authorizing execution.
5. `complete_tick(token, float32_action16, steady_ns)` checks generation/sequence,
   finite actions, current command age and inference duration. Exactly20ms is
   accepted;20ms+1ns is rejected. Actions are clipped to `[-1,1]` before being
   stored as previous action. Duplicate/old responses and backward clocks fault.
6. `torque_output(q16,dq16,steady_ns)` returns explicit effort commands using
   the25/34 contract: legs `q0+.5*a`, wheel velocity `5*a`; leg KP60/KD1.5,
   wheel KP0/KD.5; leg effort clipped to `[-80,80]`, wheel to `[-12,12]`Nm.
   The accepted request also expires20ms after its observation timestamp.

Faults latch until a valid reset. Fault/reset/no-action/pending outputs are
`authorized=false`, `targets=nullopt`, and16 zero efforts. They are **not zero
actions fed through position PD**, and do not command the default posture.
There is no asynchronous old-action hold while a request is pending. A future
ROS wrapper must explicitly accommodate this conservative scheduling policy;
this core cannot establish that inserting zero-effort intervals is physically
safe. Zero effort stops driving, but does not guarantee braking, upright balance
or bounded stopping distance. Separate physical acceptance remains necessary.

All timestamps must come from one monotonic clock at the actual receive/sample/
completion/output boundaries. The core cannot detect a caller fabricating age,
an internally stale sensor value, a downstream missed write, or a blocked caller
that never polls. It has no hardware watchdog. No mutex/data-race claim is made.
It must not be used by multiple threads without external serialization.
The20ms checks limit response/action age; they do not implement a50Hz scheduler,
a250Hz PD loop, or a ROS simulation-clock integration policy.

No MIT PVT output is sent. Applying gains and targets inside a motor is not the
same as applying these explicitly clipped efforts; the motor/SDK limits and
transport failure response remain unverified.

## Explicit native probe and checks

The caller must run these only within the project's serial resource budget.
Compiling does not start a model or ROS node. Use a fresh build directory:

```bash
rtk proxy c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  runtime82/probe82.cpp -o /path/to/new_build/probe82
rtk proxy /path/to/new_build/probe82 --scenario nominal
rtk proxy /path/to/new_build/probe82 --scenario late
rtk proxy /path/to/new_build/probe82 --scenario stale_command
rtk proxy /path/to/new_build/probe82 --scenario reset

# Standard-library driver compiles and calls the actual native core and probe.
rtk proxy python3 -m unittest discover -s tests -p test_adapter82.py -v
```

The four probe scenarios use fixed sensors and explicitly synthetic actions,
not a policy. JSON includes raw627, authorization, fault reason and16 efforts;
authorized nominal output additionally includes clipped actions and PD targets.
The native tests check independently specified values, both torque signs,
nonfinite/zero quaternions, sensor/action failures, command/deadline boundaries,
clock regression, history ordering, reset and cross-generation responses.

The static old-source test observes the discarded, reversed-bound `std::clamp`
in `FSM.cpp`; it does not execute that invalid call. Where local03 evidence is
available, the test checks its22 fixtures/19 native calls and recorded stale
action after warm reentry. Those historical results remain unchanged:
[`d1_interface_audit_03.md`](../docs/d1_interface_audit_03.md).

Independent bounded execution is recorded in the project82 delivery report.
Passing synthetic checks establishes this core's stated behavior, not full34
cross-language parity, ROS/DDS deadlines, live stopping, or hardware acceptance.
