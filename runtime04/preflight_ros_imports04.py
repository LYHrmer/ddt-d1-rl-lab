#!/usr/bin/env python3
"""Zero-physics import preflight for the contract 04 ROS launch entry.

Confirms that the full ROS Humble environment loaded by runtime04/ros_env04.sh
can import every module the live publisher needs, and records the resolved
environment identity so a later live trial can be frozen against it.

This tool deliberately does less than a trial: it calls no rclpy.init, creates
no node, publishes and subscribes to nothing, imports no MuJoCo and integrates
zero native steps. A PASS here is an import/environment result only; it is not
a watchdog validation and not a control qualification.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

# The 03 live attempt died because this entry was missing from PYTHONPATH.
REQUIRED_PATH_ENTRY = '/opt/ros/humble/lib/python3.10/site-packages'
REQUIRED_MODULES = ('rpyutils', 'rclpy', 'geometry_msgs.msg', 'std_msgs.msg')
# Importing any of these would mean the preflight is no longer physics-free.
FORBIDDEN_MODULES = ('mujoco', 'torch', 'onnxruntime')


def module_identity(name):
    module = sys.modules[name]
    file = getattr(module, '__file__', None)
    root = sys.modules[name.split('.')[0]]
    return {'module': name,
            'file': file,
            'package_version': getattr(root, '__version__', None),
            'file_sha256': hashlib.sha256(Path(file).read_bytes()).hexdigest() if file and Path(file).is_file() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True,
                        help='receipt directory; must not already exist')
    args = parser.parse_args()

    checks = []

    def check(name, ok, detail):
        checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})
        return ok

    path_entries = (os.environ.get('PYTHONPATH') or '').split(os.pathsep)
    check('ros_distro_is_humble', os.environ.get('ROS_DISTRO') == 'humble',
          {'ROS_DISTRO': os.environ.get('ROS_DISTRO')})
    check('ament_prefix_path_has_humble',
          '/opt/ros/humble' in (os.environ.get('AMENT_PREFIX_PATH') or '').split(os.pathsep),
          {'AMENT_PREFIX_PATH': os.environ.get('AMENT_PREFIX_PATH')})
    check('pythonpath_has_rpyutils_site_packages', REQUIRED_PATH_ENTRY in path_entries,
          {'required': REQUIRED_PATH_ENTRY, 'pythonpath_entries': path_entries,
           'why': 'the frozen 03 launcher overrode PYTHONPATH and dropped this entry'})
    check('ros_localhost_only', os.environ.get('ROS_LOCALHOST_ONLY') == '1',
          {'ROS_LOCALHOST_ONLY': os.environ.get('ROS_LOCALHOST_ONLY')})
    check('ros_domain_id_is_84', os.environ.get('ROS_DOMAIN_ID') == '84',
          {'ROS_DOMAIN_ID': os.environ.get('ROS_DOMAIN_ID')})

    imported = []
    for name in REQUIRED_MODULES:
        try:
            __import__(name)
        except Exception as error:  # noqa: BLE001 - the identity of the failure is the result
            check('import:'+name, False, {'error': type(error).__name__, 'message': str(error)})
            continue
        identity = module_identity(name)
        imported.append(identity)
        check('import:'+name, True, identity)

    # Zero-physics and zero-node assertions, verified rather than asserted in prose.
    leaked = [name for name in FORBIDDEN_MODULES if name in sys.modules]
    check('no_simulation_or_inference_module_imported', not leaked, {'imported': leaked})

    context_ok = None
    if 'rclpy' in sys.modules:
        try:
            context_ok = bool(sys.modules['rclpy'].utilities.get_default_context().ok())
        except Exception as error:  # noqa: BLE001
            context_ok = 'unavailable: '+type(error).__name__
    check('rclpy_context_not_initialised', context_ok is False, {'default_context_ok': context_ok})

    failed = [item['check'] for item in checks if item['status'] == 'FAIL']
    receipt = {
        'format': 'd1-contract04-ros-import-preflight-v1',
        'status': 'PASS' if not failed else 'FAIL',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'scope': 'import_and_environment_identity_only',
        'physics_steps_executed': 0,
        'transitions_consumed': 0,
        'ros_nodes_created': 0,
        'ros_messages_published': 0,
        'ros_messages_received': 0,
        'watchdog_validated': False,
        'control_qualification_claimed': False,
        'note': ('PASS means the four publisher imports resolve under the full ROS Humble '
                 'environment. It does not validate the 0.5 s command-timeout stop, any '
                 'physical behaviour, or any qualification gate.'),
        'checks': checks,
        'failed_checks': failed,
        'modules': imported,
        'environment': {
            'ROS_DISTRO': os.environ.get('ROS_DISTRO'),
            'ROS_DOMAIN_ID': os.environ.get('ROS_DOMAIN_ID'),
            'ROS_LOCALHOST_ONLY': os.environ.get('ROS_LOCALHOST_ONLY'),
            'RMW_IMPLEMENTATION': os.environ.get('RMW_IMPLEMENTATION'),
            'AMENT_PREFIX_PATH': os.environ.get('AMENT_PREFIX_PATH'),
            'PYTHONPATH': os.environ.get('PYTHONPATH'),
            'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH'),
        },
        'interpreter': {
            'executable': sys.executable,
            'version': sys.version,
            'dont_write_bytecode': bool(sys.dont_write_bytecode),
            'platform': platform.platform(),
            'node': platform.node(),
        },
        'entry_sha256': {
            'preflight_ros_imports04.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'ros_env04.sh': hashlib.sha256((Path(__file__).parent/'ros_env04.sh').read_bytes()).hexdigest(),
            'launch_env04.sh': hashlib.sha256((Path(__file__).parent/'launch_env04.sh').read_bytes()).hexdigest(),
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'status': receipt['status'], 'failed_checks': failed,
                      'physics_steps_executed': 0,
                      'output': str(args.output/'report.json')}, indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
