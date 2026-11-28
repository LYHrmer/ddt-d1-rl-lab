"""Registered sources and original parking comparisons on the level D1 grid."""
from tools91.launch91 import ROOT, digest, read, require
from tools113 import common as parent


DEV = tuple(parent.SEEDS)
BLIND = (114211, 114227, 114239)
BASE = ROOT / 'evidence/parking114'
PUBLIC = ROOT / 'reports/parking114'
PLAN = ROOT / 'docs/experiments114/parking.json'
METHOD = ROOT / 'docs/parking114_proposal.md'
PREVIOUS = parent.PUBLIC / 'results.json'


def registration():
    plan = read(PLAN)
    require(plan['development'] == list(DEV) and plan['conditional_blind'] == list(BLIND), 'unexpected parking seeds')
    require(plan['round113_registration_sha256'] == parent.registration(), 'prior speed protocol changed')
    for name, sha in plan['sources_sha256'].items():
        require(digest(ROOT / name) == sha, 'registered parking source changed: ' + name)
    return digest(PLAN)


def baseline(seed):
    require(seed in DEV, 'no paired reference at this starting state')
    tag = f'native_{seed}'
    stage = read(PREVIOUS)['stages'][tag]
    directory = PREVIOUS.parent / tag
    trace = directory / 'trace.npz'
    report = directory / 'report.json'
    require(stage['status'] == 'COMPLETE' and digest(report) == stage['files']['report.json'], 'unbound baseline')
    row = read(report)['result']
    require(row['status'] == 'COMPLETE' and row['seed'] == seed and row['motion']['ticks'] == 600 and
            row['motion']['failed'] == ['stopped'] and row['trace_sha256'] == stage['files']['trace.npz'] == digest(trace),
            'wrong or incomplete parking baseline')
    return trace, row
