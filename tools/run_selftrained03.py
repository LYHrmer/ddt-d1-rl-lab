#!/usr/bin/env python3
"""Short, identity-checked entry point for the independent D1 C++ runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export",type=Path,required=True,help="directory containing policy.onnx and metadata.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sequence",choices=["forward_stop","reverse_stop","turn_stop"],default="forward_stop")
    p.add_argument("--delay-ticks",type=int,choices=range(4),default=0)
    p.add_argument("--initial-state",type=Path)
    p.add_argument("--seed",type=int,default=7001)
    p.add_argument("--validate-only",action="store_true",help="zero integration; load identities and static state")
    p.add_argument("--print-command",action="store_true",help="check identities and print argv without running")
    a=p.parse_args()
    if a.output.exists():p.error("Output directory already exists; choose a new name")
    export=a.export.resolve(); meta=json.loads((export/"metadata.json").read_text())
    if (meta.get("interface_id")!="d1-selftrained-pilot03b-v2" or
            meta.get("random_initialization") is not True or meta.get("teacher_policy_used") is not False):
        p.error("Expected self-trained 03b metadata; this launcher does not accept colleague policies")
    config_path=ROOT/"training/configs/pilot03b.json"
    config=json.loads(config_path.read_text())
    schema_path=ROOT/"runtime03/schema_pilot03b.json"
    schema=json.loads(schema_path.read_text())
    scene=ROOT/config["scene"]
    paths={"policy_onnx":export/"policy.onnx","config":config_path,"scene_xml":scene,
           "robot_xml":scene.parent/"robot.xml","interface_schema":ROOT/config["interface_schema"],
           "golden_fixtures":export/"golden_fixtures.json"}
    for name,path in paths.items():
        if sha(path)!=meta["sha256"][name]:p.error(f"Metadata SHA mismatch: {name}")
    if (schema["interface_id"]!=meta["interface_id"] or
            schema["training_config_sha256"]!=sha(config_path) or
            schema["training_interface_schema_sha256"]!=meta["sha256"]["interface_schema"]):
        p.error("C++ runtime schema differs from the trained policy interface")
    sequence=ROOT/"runtime03/sequences03b"/(a.sequence+".json")
    if not sequence.is_file():p.error(f"Missing explicit 12-second sequence: {sequence}")
    cmd=["rtk","proxy",str(ROOT/"runtime03/build/d1_runtime03"),
         "--schema",str(schema_path),"--model",str(scene),"--policy",str(paths["policy_onnx"]),
         "--expected-model-sha256",sha(scene),"--expected-policy-sha256",meta["sha256"]["policy_onnx"],
         "--expected-schema-sha256",sha(schema_path),"--sequence",str(sequence),
         "--mode","replay","--seconds","12","--max-native","6000",
         "--delay-ticks",str(a.delay_ticks),"--seed",str(a.seed),"--output",str(a.output.resolve())]
    if a.initial_state:cmd.extend(["--initial-state",str(a.initial_state.resolve())])
    if a.validate_only:cmd.append("--validate-only")
    print(json.dumps({"training_kind":meta["training_kind"],"training_seed":meta["training_seed"],
          "trained_transitions":meta["trained_transitions"],"interface":meta["interface_id"],
          "native_cap":0 if a.validate_only or a.print_command else 6000,
          "qualification":"Run completion alone does not mean control thresholds passed"},indent=2),flush=True)
    if a.print_command:
        print(json.dumps(cmd,ensure_ascii=False,indent=2));return 0
    env=os.environ.copy()
    extra="/opt/ros/humble/lib:"+str(ROOT/"install/mujoco/lib")
    env["LD_LIBRARY_PATH"]=extra+(":"+env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return subprocess.run(cmd,env=env,check=False).returncode

if __name__=="__main__":raise SystemExit(main())
