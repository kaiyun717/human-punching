#!/usr/bin/env python3
"""Schema-aware saved-only family control audit; no scene, integration or actor action."""
from pathlib import Path
import argparse,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from human_punching.phase2_family_control_audit_v1 import audit_run as closed_audit,sha,bind,read
from human_punching.phase2_nonuniform_control import Inputs


def review(packet_path,output,*,namespace_proof):
    ctx=Inputs(namespace_proof,root=ROOT);path=ctx.owned(packet_path);packet=read(path)
    if (packet.get('schema')!='task10b_phase2_family_target_control_packet_v1'
            or packet.get('mode')!='family-target-control' or packet.get('family') not in ('cross','right_hook')
            or packet.get('allocation_wall_seconds')!=360 or packet.get('application_wall_seconds')!=300
            or packet.get('target_control_child_wall_seconds')!=210 or packet.get('target_control_trials')!=32
            or packet.get('training') is not False or packet.get('fresh_transition_cap')!=0):
        raise ValueError('Explicit accounted frozen family-control packet required')
    for name,digest in packet['files'].items():
        ctx.bound(dict(path=name,sha256=digest));ctx.bound(dict(path=path.parent/'source_snapshot'/name,sha256=digest))
    terminal=read(path.parent/'terminal_accounting.json');execution=read(path.parent/'execution.json')
    if (terminal['packet_sha256']!=sha(path) or execution['packet_sha256']!=sha(path)
            or terminal['job_id']!=execution['job_id'] or not 0<terminal['allocated_gpu_seconds']<=360
            or terminal['allocated_tres'].get('gres/gpu')!='1'):
        raise ValueError('Exact terminal native job/accounting required')
    run=ctx.owned(ROOT/packet['output']);out=Path(output).resolve()
    if not out.is_relative_to(ROOT/'outputs/task10b') or out.exists():raise ValueError('Fresh owned audit directory required')
    out.mkdir(parents=True,exist_ok=False)
    native=read(run/'execution.json');control=read(run/'control_plan.json')
    if control!=read(ctx.owned(packet['inputs']['plan'])):raise ValueError('Native plan differs from packet plan')
    closed=closed_audit(run,out/'closed_launch.json')
    spec=importlib.util.spec_from_file_location('family_control_preserved_clock_v2_audit',ROOT/'scripts/task10b_phase2_audit_clock_v2.py')
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    measured=None;error=None
    try:measured=auditor.audit_run(run,out/'trace_audit')
    except Exception as caught:error=dict(type=type(caught).__name__,message=str(caught))
    physical_integrity=bool(measured and measured['body_unchanged'] and all(measured['runtime_checks'].values())
        and measured['panels'] and all(p.get('prefix_integrity_pass') is True for p in measured['panels']))
    exact=bool(measured and measured['trials_audited']==control['native_attempt_capacity']==closed['executed_trials_audited'])
    passed=bool(closed['passed'] and physical_integrity and exact and native.get('frozen_actor_unchanged') is True)
    report=dict(schema='task10b_phase2_family_control_independent_saved_review_v1',family=packet['family'],
        passed=passed,status='complete_measurement_evidence' if passed else 'audit_unresolved',
        packet=bind(path),terminal_accounting=bind(path.parent/'terminal_accounting.json'),source=bind(__file__),
        control_plan=bind(run/'control_plan.json'),closed_launch_audit=bind(out/'closed_launch.json'),
        raw_audit=None if measured is None else bind(out/'trace_audit/audit.json'),exception=error,
        planned_trials=32,native_attempt_capacity=control['native_attempt_capacity'],
        native_numeric_complete_cycles=native.get('numeric_complete_cycles'),
        reference_unavailable_trials=closed['reference_unavailable_trials'],
        native_measurement_integrity=physical_integrity,exact_native_count=exact,
        independent_numeric_complete_cycles=None if measured is None else sum(t['numeric_complete_cycle'] for p in measured['panels'] for t in p['trials']),
        physical_flags=[] if measured is None else [dict(trial_id=t['trial_id'],checks=t['hard_flag_checks']) for p in measured['panels'] for t in p['trials'] if not t['hard_flags_absent']],
        frozen_actor_unchanged=native.get('frozen_actor_unchanged'),physical_failures_preserved=True,
        event_clock_semantics='New family adapter declares prep0; independent closed/current-state clock supplement validates this. The preserved D0 raw auditor retains its historical preparation label; no saved evidence is relabeled.',
        limits='All32 requested cycles include explicit unavailable references. Integrity does not imply cycle success, video acceptance, learned qualification or shared-start feasibility.',
        new_physics_steps=0,policy_actions_applied_to_simulator=0,route_credit=False,phase2_qualification=False,formal_acceptance=False)
    with(out/'review.json').open('x')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',required=True);p.add_argument('--output',required=True)
    p.add_argument('--namespace-proof',required=True);a=p.parse_args()
    r=review(a.packet,a.output,namespace_proof=a.namespace_proof)
    print(json.dumps({k:r[k] for k in ('family','status','planned_trials','native_attempt_capacity','reference_unavailable_trials','independent_numeric_complete_cycles')}));return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
