"""Contract 04 deployment variants: the frozen 03 table plus 04a.

runtime03/deployment_variants.py is referenced verbatim by the 03b/03c/03d
deployment receipts, so it is not edited. This module reuses its table and
selection logic and adds the one 04a entry. 04a shares the 03b control
interface exactly; only the training config path and SHA differ, mirroring how
schema03d differed from schema03c.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'runtime03'))

from deployment_variants import VARIANTS as VARIANTS_03, select_deployment as select_03  # noqa: E402

VARIANTS = dict(VARIANTS_03)
VARIANTS['d1-pure-ppo-pilot-04a'] = {
    'config': 'training/configs/pilot04a.json',
    'config_sha256': '042d15a1f9108b90b5e2bad5ce0ba89673cf16a7a0c6e534171858c02cb2ddd6',
    'schema': 'runtime04/schema04a.json',
    'schema_sha256': '3a3a0b88f3f2b754ecff38083dc080259346827bcf109ea7aaf6bca4ec6e475c',
}
VARIANTS['d1-pure-ppo-pilot-04b'] = {
    'config': 'training/configs/pilot04b.json',
    'config_sha256': '9d855063fb654f1d9bf0ae4d392f7f70294c328c5e944ef79f8e93902fb82e0a',
    'schema': 'runtime04/schema04b.json',
    'schema_sha256': 'ffca72c5b8942b79ab33c49fff4c5d4ae9ad06bdb699e41f340f156e20a29f6e',
}
CONTRACT_04_VERSIONS = ('d1-pure-ppo-pilot-04a', 'd1-pure-ppo-pilot-04b')


def select_deployment(metadata):
    """Same contract as the 03 selector, extended to the contract 04 versions."""
    version = metadata.get('config_version')
    if version not in CONTRACT_04_VERSIONS:
        return select_03(metadata)
    variant = VARIANTS[version]
    config_path, schema_path = ROOT/variant['config'], ROOT/variant['schema']

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if digest(config_path) != variant['config_sha256'] or digest(schema_path) != variant['schema_sha256']:
        raise ValueError('Frozen 04a deployment config/schema changed; qualify a new variant')
    config, schema = json.loads(config_path.read_text()), json.loads(schema_path.read_text())
    if (config['version'] != version or
            metadata.get('sha256', {}).get('config') != variant['config_sha256'] or
            Path(schema['training_config']).resolve() != config_path.resolve() or
            schema['training_config_sha256'] != variant['config_sha256']):
        raise ValueError('Export configuration identity differs from the 04a deployment variant')
    if (metadata.get('interface_id') != 'd1-selftrained-pilot03b-v2' or
            schema['interface_id'] != metadata['interface_id'] or
            config['interface_id'] != metadata['interface_id']):
        raise ValueError('Export/config/runtime control interface identities disagree')
    return config_path, schema_path, config, schema
