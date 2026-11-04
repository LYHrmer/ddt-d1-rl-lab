"""Read hashes only; absolute entries retain their original frozen meaning."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);args=p.parse_args()
if args.output.exists():raise FileExistsError(args.output)
work=Path('/home/lyh/project83_work_20261104')
before=json.loads((work/'integrity_before83.json').read_text())
seals=json.loads((work/'sealed_execution83.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
groups=[];failures=[]
for row in before['groups']:
    root=Path(row['root']);plan=root/row['plan']
    if sha(plan)!=row['sha256']:failures.append(str(plan))
    entries=json.loads(plan.read_text())['source_sha256']
    for name,digest in entries.items():
        if sha(root/name)!=digest:failures.append(str(root/name))
    groups.append(dict(row,verified_files=len(entries)))
for name,digest in seals['plans'].items():
    plan=Path(name);root=plan.parents[2]
    if sha(plan)!=digest:failures.append(name)
    data=json.loads(plan.read_text());entries=data['source_sha256']
    for source,expected in entries.items():
        if sha(root/source)!=expected:failures.append(str(root/source))
    groups.append({'plan':name,'sha256':digest,'verified_files':len(entries)})
for name,digest in seals['supervision'].items():
    if sha(Path(name))!=digest:failures.append(name)
runtime=Path('/home/lyh/ddt_ros2_ws/docs/experiments83/runtime83_preregistered.json')
if sha(runtime)!='437fcda05945ece13549f210760a66fecceec84318938472d62441d9804dcd3d':failures.append(str(runtime))
for name,digest in json.loads(runtime.read_text())['source_sha256'].items():
    if sha(Path(name))!=digest:failures.append(name)
manifest=Path('/home/lyh/wheel-legged-control-lab/runtime/d1_b22/runtime_manifest_v27.json')
if sha(manifest)!=before['v27_sha256']:failures.append(str(manifest))
result={'status':'PASS' if not failures else 'FAIL','groups':groups,'runtime83_original_sources':43,
        'v27_sha256':sha(manifest),'failures':failures,'new_model_calls':0,'new_physics_steps':0}
with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result));raise SystemExit(bool(failures))
