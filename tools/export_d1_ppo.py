#!/usr/bin/env python3
"""Export a completed, self-trained SB3 D1 PPO checkpoint to legacy ONNX.

No MuJoCo model is loaded and no physics is run. The output directory must be
new. Python ORT parity is checked here; C++ ORT 1.10 parity is a separate gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import stable_baselines3
from stable_baselines3 import PPO
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from d1_rl.interface import ACTION_DIM, ACTION_SCALE, ACTOR_DIM, HISTORY_LENGTH, OBS_DIM  # noqa: E402

DEFAULT_CONFIG = ROOT / "training/configs/pilot03.json"
FIXTURE_SEED = 20261009
FIXTURE_COUNT = 64
ALLOWED_ONNX_OPS = {"Concat", "Reshape", "Gemm", "Tanh", "Clip", "Constant", "Identity"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def check_checkpoint(checkpoint: Path, config_path: Path):
    if checkpoint.name != "policy.zip":
        raise ValueError("Only a completed self-trained policy.zip may be exported")
    protocol_path = checkpoint.parent / "protocol.json"
    result_path = checkpoint.parent / "result.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    hyper = protocol["hyperparameters"]
    if (result["status"] != "COMPLETE" or not result["inputs_unchanged"] or
            result["trained_transitions"] <= 0 or
            result["trained_transitions"] != protocol["planned_transitions"] or
            hyper["random_initialization"] is not True or
            hyper["teacher_policy_used"] is not False or
            hyper["warm_start"] is not False):
        raise ValueError("Checkpoint lacks completed random-initialized/no-teacher provenance")
    if (hyper["policy_arch"] != [128, 128] or hyper["activation"] != "tanh" or
            protocol["config"] != json.loads(config_path.read_text(encoding="utf-8"))):
        raise ValueError("Checkpoint architecture/config differs from frozen pilot03")
    for path_string, expected in protocol["inputs_sha256"].items():
        path = Path(path_string)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Training source changed since checkpoint: {path}")
    if any("vecnormalize" in path.name.lower() for path in checkpoint.parent.iterdir()):
        raise ValueError("VecNormalize artifacts found; the frozen actor has no normalization wrapper")
    return protocol_path, protocol, result_path, result


def check_policy(policy):
    if tuple(policy.observation_space.shape) != (ACTOR_DIM,) or tuple(policy.action_space.shape) != (ACTION_DIM,):
        raise ValueError("SB3 observation/action space is not 627/16")
    if not (np.array_equal(policy.action_space.low, -np.ones(ACTION_DIM)) and
            np.array_equal(policy.action_space.high, np.ones(ACTION_DIM))):
        raise ValueError("Action space is not [-1,1]^16")
    from stable_baselines3.common.torch_layers import FlattenExtractor
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    if not isinstance(policy.features_extractor, FlattenExtractor):
        raise ValueError("Only the frozen unnormalized flat MLP feature extractor is supported")
    if not isinstance(policy.action_dist, DiagGaussianDistribution):
        raise ValueError("The frozen PPO actor requires an unsquashed diagonal Gaussian")
    layers = list(policy.mlp_extractor.policy_net)
    expected = ((torch.nn.Linear, ACTOR_DIM, 128), (torch.nn.Tanh, None, None),
                (torch.nn.Linear, 128, 128), (torch.nn.Tanh, None, None))
    if len(layers) != len(expected):
        raise ValueError("Actor MLP layer count differs from frozen [128,128] Tanh")
    for layer, (kind, incoming, outgoing) in zip(layers, expected):
        if not isinstance(layer, kind):
            raise ValueError("Actor MLP layer type differs from frozen architecture")
        if kind is torch.nn.Linear and (layer.in_features, layer.out_features) != (incoming, outgoing):
            raise ValueError("Actor MLP dimensions differ from frozen architecture")
    if not isinstance(policy.action_net, torch.nn.Linear) or (
            policy.action_net.in_features, policy.action_net.out_features) != (128, ACTION_DIM):
        raise ValueError("Actor action head is not Linear(128,16)")


class DeterministicActor(torch.nn.Module):
    """SB3 policy mean with the deployment action clip, no sampling."""

    def __init__(self, policy):
        super().__init__()
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_head = policy.action_net

    def forward(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        actor = torch.cat((obs.reshape(1, OBS_DIM),
                           history.reshape(1, HISTORY_LENGTH * OBS_DIM)), dim=1)
        mean = self.action_head(self.policy_net(actor))
        return torch.clamp(mean, -1.0, 1.0)


def onnx_signature(model):
    def item(info):
        tensor = info.type.tensor_type
        return {"name": info.name, "elem_type": tensor.elem_type,
                "shape": [dim.dim_value for dim in tensor.shape.dim]}
    return {"inputs": [item(x) for x in model.graph.input],
            "outputs": [item(x) for x in model.graph.output]}


def check_onnx(path):
    model = onnx.load(str(path))
    opsets = [(x.domain, x.version) for x in model.opset_import]
    if opsets != [("", 11)]:
        raise ValueError(f"Expected only default-domain ONNX opset 11, got {opsets}")
    ops = sorted({node.op_type for node in model.graph.node})
    if not set(ops).issubset(ALLOWED_ONNX_OPS):
        raise ValueError(f"Unexpected ONNX operations for ORT 1.10: {ops}")
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    signature = onnx_signature(model)
    if signature != {"inputs": [
            {"name": "obs", "elem_type": onnx.TensorProto.FLOAT, "shape": [1, 57]},
            {"name": "history", "elem_type": onnx.TensorProto.FLOAT, "shape": [1, 10, 57]}],
            "outputs": [{"name": "actions", "elem_type": onnx.TensorProto.FLOAT,
                         "shape": [1, 16]}]}:
        raise ValueError(f"Unexpected ONNX input/output signature: {signature}")
    return ops, signature


def golden_parity(actor, sb3_model, onnx_path):
    rng = np.random.default_rng(FIXTURE_SEED)
    obs = rng.normal(0, 0.5, size=(FIXTURE_COUNT, OBS_DIM)).astype(np.float32)
    history = rng.normal(0, 0.5, size=(FIXTURE_COUNT, HISTORY_LENGTH, OBS_DIM)).astype(np.float32)
    obs[0] = 0
    history[0] = 0
    flat = np.concatenate((obs, history.reshape(FIXTURE_COUNT, -1)), axis=1)
    sb3_action, _ = sb3_model.predict(flat, deterministic=True)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if ([(x.name, x.shape, x.type) for x in session.get_inputs()] != [
            ("obs", [1, 57], "tensor(float)"),
            ("history", [1, 10, 57], "tensor(float)")] or
            [(x.name, x.shape, x.type) for x in session.get_outputs()] != [
                ("actions", [1, 16], "tensor(float)")]):
        raise ValueError("Python ORT did not load the expected two-input/one-output signature")
    fixtures = []
    max_torch_ort = 0.0
    max_sb3_torch = 0.0
    with torch.no_grad():
        for index in range(FIXTURE_COUNT):
            one_obs = obs[index:index + 1]
            one_history = history[index:index + 1]
            torch_action = actor(torch.from_numpy(one_obs),
                                 torch.from_numpy(one_history)).cpu().numpy()
            ort_action = session.run(["actions"], {"obs": one_obs,
                                                   "history": one_history})[0]
            max_torch_ort = max(max_torch_ort, float(np.max(np.abs(torch_action - ort_action))))
            max_sb3_torch = max(max_sb3_torch, float(np.max(np.abs(sb3_action[index] - torch_action[0]))))
            fixtures.append({"index": index, "obs": one_obs[0].tolist(),
                             "history_oldest_to_newest": one_history[0].tolist(),
                             "torch_actions": torch_action[0].tolist(),
                             "python_ort_actions": ort_action[0].tolist()})
    if max_torch_ort > 1e-5 or max_sb3_torch > 1e-6:
        raise ValueError(f"ONNX parity failed: Torch/ORT={max_torch_ort}, SB3/Torch={max_sb3_torch}")
    return fixtures, max_torch_ort, max_sb3_torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    config_path = args.config.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    protocol_path, protocol, result_path, result = check_checkpoint(checkpoint, config_path)
    torch.set_num_threads(1)
    model = PPO.load(str(checkpoint), device="cpu")
    if model.num_timesteps != result["trained_transitions"]:
        raise ValueError("Loaded checkpoint step count differs from completed training result")
    check_policy(model.policy)
    model.policy.eval()
    actor = DeterministicActor(model.policy).eval()
    output.mkdir(parents=True, exist_ok=False)
    onnx_path = output / "policy.onnx"
    with torch.no_grad():
        torch.onnx.export(actor, (torch.zeros((1, 57), dtype=torch.float32),
                                  torch.zeros((1, 10, 57), dtype=torch.float32)),
                          str(onnx_path), input_names=["obs", "history"],
                          output_names=["actions"], opset_version=11,
                          dynamo=False, do_constant_folding=True)
    ops, signature = check_onnx(onnx_path)
    fixtures, max_torch_ort, max_sb3_torch = golden_parity(actor, model, onnx_path)
    fixture_path = output / "golden_fixtures.json"
    with fixture_path.open("x", encoding="utf-8") as stream:
        stream.write(json_text({"format": "d1-onnx-two-input-golden-64-v1",
                                "seed": FIXTURE_SEED, "count": FIXTURE_COUNT,
                                "input_order": ["obs", "history"],
                                "history_order": "oldest_to_newest_previous_only",
                                "fixtures": fixtures}))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scene = ROOT / config["scene"]
    robot = scene.parent / "robot.xml"
    schema = ROOT / config["interface_schema"]
    audited_schema = ROOT / config.get("audited_interface_schema", config["interface_schema"])
    if sha256(audited_schema) != config.get(
            "audited_interface_schema_sha256", config["interface_schema_sha256"]):
        raise ValueError("Audited observation schema SHA256 mismatch")
    action_scales = config.get("action_scales", ACTION_SCALE.tolist())
    metadata = {
        "format": "d1-pure-ppo-onnx-export-v1",
        "interface_id": config.get("interface_id", "d1-selftrained-pilot03-v1"),
        "config_version": config["version"],
        "action_scales": action_scales,
        "audited_interface_schema_sha256": config.get(
            "audited_interface_schema_sha256", config["interface_schema_sha256"]),
        "training_kind": protocol["kind"], "training_arm": protocol["arm"],
        "training_seed": protocol["seed"],
        "trained_transitions": result["trained_transitions"],
        "random_initialization": True, "teacher_policy_used": False,
        "warm_start": False, "vecnormalize_used": False,
        "policy_operation": "deterministic Gaussian mean, clipped to [-1,1]",
        "history_order": "oldest_to_newest_previous_only",
        "onnx_ir_version": 7, "onnx_opset": 11, "onnx_node_types": ops,
        "onnx_signature": signature,
        "sha256": {"checkpoint": sha256(checkpoint), "policy_onnx": sha256(onnx_path),
                   "training_protocol": sha256(protocol_path),
                   "training_result": sha256(result_path),
                   "scene_xml": sha256(scene), "robot_xml": sha256(robot),
                   "interface_schema": sha256(schema),
                   "audited_interface_schema": sha256(audited_schema),
                   "config": sha256(config_path),
                   "golden_fixtures": sha256(fixture_path),
                   "export_script": sha256(Path(__file__).resolve())},
        "golden_parity": {"fixtures": FIXTURE_COUNT, "fixture_seed": FIXTURE_SEED,
                          "torch_vs_python_ort_max_abs_error": max_torch_ort,
                          "sb3_predict_vs_torch_max_abs_error": max_sb3_torch,
                          "python_ort_version": ort.__version__,
                          "cxx_ort_1_10_status": "not_tested_here"},
        "versions": {"torch": torch.__version__, "sb3": stable_baselines3.__version__,
                     "onnx": onnx.__version__, "python_ort": ort.__version__},
    }
    with (output / "metadata.json").open("x", encoding="utf-8") as stream:
        stream.write(json_text(metadata))
    print(json_text({"output": str(output), "onnx_sha256": metadata["sha256"]["policy_onnx"],
                     "fixtures": FIXTURE_COUNT, "max_torch_python_ort_error": max_torch_ort,
                     "cxx_ort_1_10_status": "not_tested_here"}))


if __name__ == "__main__":
    main()
