# 83 startup-contract correction — 2026-11-04

Two implementation defects were discovered before their expensive stage began:

- Policy geometry83 rejected six absolute paths inherited unchanged from the fixed81 source manifest. The first record remains INCOMPLETE, with zero model/actor calls, zero physics and zero updates.
- ROS83 required eight map entries although its documented command supplies seven. The first nominal call exited before constructing an actor or ROS context. The supervisor preserves stdout/stderr and the generated manifest; there is no per-scenario report directory. The remaining three calls to that same defective entry point were not started.

The original sources, plans and all failure artifacts remain frozen. Correct the startup contracts in independent checked entry points; do not change the model, data, controller, metrics, gates, seeds or experiment parameters. First run regression tests against each exact defect, then run the full checked command in a validate-only mode that cannot construct an actor, Torch model or ROS context. Bind the new source and successful cold-admission receipts before a fresh independent validation. This is an engineering correction, not resumption of either failed record or tuning in response to a performance outcome.

The original aggregate budgets remain: policy geometry at most2016 dynamics forwards and2016 actor-head calls, zero updates/physics; ROS at most26 request attempts across24 nominal,1 late receive,1 late response and0 stale-command inference requests. No repeat of the already successful64 native calls. Original ROS first-attempt request/model/context counts are0, policy first-attempt model/actor counts are0. Each fresh ROS mode receives a15s child deadline and7s CPU limit (four modes at most60s/28s, leaving room within the original30 CPU seconds for failed startup). Heavy work remains serial with the unchanged83 resource guard.

All original failures stay visible in the final delivery. A checked entry point passing admission is not itself a control-performance or real-time/hardware result.
