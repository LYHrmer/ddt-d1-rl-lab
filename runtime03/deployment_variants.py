"""Frozen training-variant identities used by deployment entry points.

03b, 03c and 03d share the actor/control interface but differ in task configs.
Selection is by the export's explicit config_version, never its directory name.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'd1-pure-ppo-pilot-03b': {
        'config': 'training/configs/pilot03b.json',
        'config_sha256': '3996e6e7c677be59e2852692d3f71ef7d0de8a689b6be3493f9ba466d92a589f',
        'schema': 'runtime03/schema_pilot03b.json',
        'schema_sha256': 'a9eba4202468ef0eb2aa52dfde838b8efcb23d8ac25c3067064adc3aed941b55',
    },
    'd1-pure-ppo-pilot-03c': {
        'config': 'training/configs/pilot03c.json',
        'config_sha256': '562bdc6b26fba7bbb4906f7045ba11cbe70848a3335bd8d5e1659540a3f2b604',
        'schema': 'runtime03/schema03c.json',
        'schema_sha256': 'c3019b6737a358ebbdcd8c5cdb43bdd6b5a9783f36deeb31c30c87f8f820a538',
    },
    'd1-pure-ppo-pilot-03d': {
        'config': 'training/configs/pilot03d.json',
        'config_sha256': '9cd6546e8a8df50c7d578a664003681fc79ddca7fa0a27b80158f0f5b6842bea',
        'schema': 'runtime03/schema03d.json',
        'schema_sha256': '64473031cbf027c950469a3db5a6e39b868f244da2b3f244fa174ab503f64c1c',
    },
}


def select_deployment(metadata):
    version = metadata.get('config_version')
    if version not in VARIANTS:
        raise ValueError(f'Unsupported exported config_version: {version!r}; expected explicit 03b, 03c or 03d')
    variant = VARIANTS[version]
    config_path, schema_path = ROOT/variant['config'], ROOT/variant['schema']
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(config_path) != variant['config_sha256'] or digest(schema_path) != variant['schema_sha256']:
        raise ValueError('Frozen deployment config/schema changed; qualify a new variant')
    config, schema = json.loads(config_path.read_text()), json.loads(schema_path.read_text())
    if (config['version'] != version or metadata.get('sha256', {}).get('config') != variant['config_sha256'] or
            Path(schema['training_config']).resolve() != config_path.resolve() or
            schema['training_config_sha256'] != variant['config_sha256']):
        raise ValueError('Export configuration identity differs from the selected deployment variant')
    if (metadata.get('interface_id') != 'd1-selftrained-pilot03b-v2' or
            schema['interface_id'] != metadata['interface_id'] or
            config['interface_id'] != metadata['interface_id']):
        raise ValueError('Export/config/runtime control interface identities disagree')
    return config_path, schema_path, config, schema
