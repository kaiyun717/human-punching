#!/usr/bin/env python3
"""Saved-only 120-row curve ledger plus unchanged independent raw-trace audit."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from human_punching.phase2_curve_saved_v2 import files,ledger,pilot


def review(run,output,*,namespace_proof=None,ledger_only=False,pending=False):
    run=Path(run).resolve(strict=True);output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'outputs/task10b') or output.exists():
        raise ValueError('Fresh owned audit output required')
    if pending and not ledger_only:raise ValueError('Pending jobs have no independent execution audit')
    result=ledger(run,pending=pending)
    output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (output/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    save('declared_ledger.json',result)
    report=dict(schema='task10b_phase2_cross_curve_saved_review_v2',planned_trials=120,
        status='ledger_only',ledger_sha256=pilot.sha(output/'declared_ledger.json'),
        all_declared_trials_independently_audited=False,phase2_qualification=False,
        route_credit=False,control_regression_is_not_a_new_physical_fault=True,
        source_sha256=pilot.sha(__file__),inputs={},exception=None)
    if not ledger_only:
        try:
            for update in pilot.UPDATES:
                panel=run/f'validation_{update:05d}/summary.json'
                if panel.exists():report['inputs'].update(files(ROOT,dict(run=str(run),panel=str(panel)),
                    'outputs/four-punch-milestone','outputs/task10/release_baseline_02',
                    'outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json',namespace_proof=namespace_proof))
            spec=importlib.util.spec_from_file_location('task10b_curve_saved_original_auditor',ROOT/'scripts/task10b_phase2_audit_clock_v2.py')
            audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
            measured=audit.audit_run(run,output/'trace_audit')
            report['audit']=dict(path=str(output/'trace_audit/audit.json'),sha256=pilot.sha(output/'trace_audit/audit.json'))
            report['retained_trace_audit_passed']=measured['passed']
            complete=(len(result['trials'])==120 and all(r['status']=='executed_pending_independent_audit' for r in result['trials'])
                and measured.get('trials_audited')==120 and measured.get('panels_audited')==15 and measured['passed'])
            report['all_declared_trials_independently_audited']=bool(complete)
            report['status']='complete_evidence' if complete else 'partial_evidence' if measured['passed'] else 'audit_failed'
        except Exception as error:
            report['status']='unresolved_audit_failure';report['exception']=dict(type=type(error).__name__,message=str(error))
    save('review.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--namespace-proof');parser.add_argument('--ledger-only',action='store_true');parser.add_argument('--pending',action='store_true')
    a=parser.parse_args();result=review(a.run,a.output,namespace_proof=a.namespace_proof,ledger_only=a.ledger_only,pending=a.pending)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 1 if result['status'] in ('unresolved_audit_failure','audit_failed') else 0


if __name__=='__main__':raise SystemExit(main())
