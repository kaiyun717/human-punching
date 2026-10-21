#!/usr/bin/env python3
"""Saved-only physical audit adapter for distinct direct-clock launch schemas.

CPU policy inference in the original auditor is retained. No actor actions are
applied to a simulator. Native bytes and the original audit source stay intact.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('task10b_direct_expanded_raw_audit',ROOT/'scripts/task10b_phase2_audit.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
from human_punching.phase2_expanded_entry_v2 import LAUNCH_SCHEMA,VERSION
from human_punching.phase2_expanded_review_v2 import verify_event_evidence
from human_punching.wbc.provenance import writable_path,write_json,sha256_file


class TypedLaunchJSON:
    def __getattr__(self,name):return getattr(json,name)
    def loads(self,value,*args,**kwargs):
        result=json.loads(value,*args,**kwargs)
        if isinstance(result,dict) and result.get('schema')==LAUNCH_SCHEMA:
            if result.get('direct_consumer_version')!=VERSION:
                raise ValueError('New launch lacks its exact direct consumer version')
            return {**result,'schema':'task10b_phase2_expanded_native_launch_v1'}
        return result


def audit_run(run,output):
    run=Path(run).resolve(strict=True);folder=writable_path(output)
    if not run.is_relative_to(ROOT/'outputs/task10b') or folder.exists():
        raise ValueError('Owned saved run and fresh audit output required')
    launch=json.loads((run/'launch.json').read_text())
    if launch.get('schema')!=LAUNCH_SCHEMA or launch.get('direct_consumer_version')!=VERSION:
        raise ValueError('This adapter cannot relabel historical or shared native runs')
    folder.mkdir(parents=True,exist_ok=False)
    try:clock=verify_event_evidence(run)
    except Exception as error:clock=dict(passed=False,exception=dict(type=type(error).__name__,message=str(error)))
    original=parent.json
    try:
        parent.json=TypedLaunchJSON()
        result=parent.audit_run(run,folder/'original_raw_audit')
    except Exception as error:
        result=dict(schema='task10b_phase2_independent_trace_audit_v1',run=str(run),passed=False,
            status='unresolved_audit_failure',panels_audited=0,trials_audited=0,
            exception=dict(type=type(error).__name__,message=str(error)),
            outcome_resolution_passed=False,closed_protocol_launch_audited=False)
    finally:parent.json=original
    result={**result,'passed':bool(result.get('passed') is True and clock.get('passed') is True),
        'direct_consumer_version':VERSION,'direct_event_clock_review':clock,
        'native_launch_schema':LAUNCH_SCHEMA,'adapter_source_sha256':sha256_file(__file__),
        'original_raw_auditor_sha256':sha256_file(parent.__file__),
        'typed_in_memory_schema_view_only':True,'native_artifacts_rewritten':False,
        'new_physics_steps':0,'policy_actions_applied_to_simulator':0,'formal_acceptance':False}
    if not result['passed']:result['status']='failed'
    write_json(folder/'audit.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);result=audit_run(args.run,args.output)
    print(json.dumps({k:result[k] for k in ('passed','panels_audited','trials_audited')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
