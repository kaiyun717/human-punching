#!/usr/bin/env python3
"""Saved-only grouped raw physics/actor audits with complete480-request ledger."""
from __future__ import annotations
import argparse,importlib.util,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from human_punching.phase2_grouped_saved_v1 import ledger,binding,read,UPDATES
from human_punching.phase2_expanded_clock_audit_v1 import audit_batch


def independent_event_comparison(measured,folder):
    """Compare metadata to events recomputed by the separate FK/raw auditor."""
    checks={}
    for panel in measured['panels']:
        batch=Path(folder)/Path(panel['path']).name
        if not (batch/'summary.json').exists():
            checks[str(batch)+':saved_summary']=False;continue
        saved={r['trial_id']:r for r in read(batch/'summary.json')['trials']}
        for trial in panel['trials']:
            if trial['trial_id'] not in saved:
                checks[trial['trial_id']+':saved_trial']=False;continue
            events=saved[trial['trial_id']].get('event_clocks',{})
            pairs={'request_time_s':'request_s','preparation_complete_time_s':'preparation_complete_s',
                'first_positional_support_readiness_time_s':'first_positional_support_readiness_s',
                'actual_launch_time_s':'launch_s','outward_complete_time_s':'outward_complete_s',
                'first_active_contact_time_s':'first_active_contact_s','first_inactive_contact_time_s':'first_inactive_contact_s',
                'functional_recovery_time_s':'recovery_time_s'}
            for new,independent in pairs.items():
                a,b=events.get(new),trial.get(independent)
                checks[trial['trial_id']+':'+new]=(a is None and b is None) or (type(a) in (int,float) and type(b) in (int,float) and abs(a-b)<=1.e-9)
    return dict(passed=bool(checks) and all(checks.values()),checks=checks)


def review(packet,output,*,namespace_proof,ledger_only=False,pending=False):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'outputs/task10b') or output.exists():raise ValueError('Fresh owned audit output required')
    if pending and not ledger_only:raise ValueError('Pending jobs support a pending ledger only')
    declared=ledger(packet,namespace_proof,pending=pending);output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (output/name).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    save('declared_ledger.json',declared);run=Path(declared['run']);panels=[]
    if not ledger_only:
        spec=importlib.util.spec_from_file_location('task10b_grouped_original_independent_auditor',ROOT/'scripts/task10b_phase2_audit.py')
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        for update in UPDATES:
            folder=run/'held_out_validation'/f'validation_{update:05d}'
            if not (folder/'launch.json').exists():continue
            item=dict(update=update,passed=False,measurement_integrity=False,exception=None,clock_audits=[])
            try:
                launch=read(folder/'launch.json')
                if (launch.get('schema')!='task10b_phase2_grouped_validation_launch_v1'
                        or launch.get('actual_training_update')!=update
                        or launch.get('actual_completed_fresh_transitions')!=update*128*64
                        or launch.get('checkpoint_template_is_unbound') is not True):
                    raise ValueError('Exact grouped callback launch/checkpoint clock required')
                measured=original.audit_run(folder,output/f'validation_{update:05d}')
                item.update(audit=binding(output/f'validation_{update:05d}/audit.json'),
                    passed=measured['passed'],trials_audited=measured['trials_audited'],
                    measurement_integrity=bool(measured['body_unchanged'] and all(measured['runtime_checks'].values())
                        and measured['panels'] and all(p.get('prefix_integrity_pass') is True for p in measured['panels'])),
                    numeric_complete_cycles=sum(t['numeric_complete_cycle'] for p in measured['panels'] for t in p['trials']),
                    physical_flags=[dict(trial_id=t['trial_id'],checks=t['hard_flag_checks']) for p in measured['panels'] for t in p['trials'] if not t['hard_flags_absent']])
                item['independent_event_comparison']=independent_event_comparison(measured,folder)
                for batch in sorted(folder.glob('batch_[0-9][0-9][0-9][0-9]')):
                    if (batch/'summary.json').exists() and (batch/'closed_protocol_launch.json').exists():
                        item['clock_audits'].append(dict(batch=batch.name,**audit_batch(batch)))
                item['event_metadata_passed']=(bool(item['clock_audits']) and all(c['passed'] for c in item['clock_audits'])
                    and item['independent_event_comparison']['passed'])
            except Exception as error:item['exception']=dict(type=type(error).__name__,message=str(error))
            panels.append(item)
    complete=(len(panels)==5 and all(p.get('trials_audited')==96 and p['measurement_integrity']
        and p.get('event_metadata_passed') for p in panels)
        and declared['counts']=={'completed_native_attempt_pending_independent_audit':480})
    integrity=bool(panels) and all(p['measurement_integrity'] and p.get('event_metadata_passed') for p in panels)
    parity_path=run/'grouped_integrity_v2.json';parity=read(parity_path) if parity_path.exists() else None
    clocks_path=run/'grouped_event_clocks_v1.json';clocks=read(clocks_path) if clocks_path.exists() else None
    report=dict(schema='task10b_phase2_grouped_saved_review_v1',status='ledger_only' if ledger_only else
        'complete_measurement_evidence' if complete else 'partial_measurement_evidence' if integrity else 'audit_unresolved',
        ledger=binding(output/'declared_ledger.json'),panels=panels,planned_trials=480,
        all_declared_trials_independently_audited=complete,retained_measurement_integrity=integrity,
        recorded_final_parity_pass=None if parity is None else parity.get('passed'),
        recorded_event_clock_consumer_pass=None if clocks is None else clocks.get('passed'),
        final_parity_receipt=binding(parity_path) if parity_path.exists() else None,
        event_clock_receipt=binding(clocks_path) if clocks_path.exists() else None,
        physical_failures_retained=True,route_credit=False,phase2_qualification=False,formal_acceptance=False,
        source=binding(__file__),new_physics_steps=0,policy_actions_applied_to_simulator=0,
        limitations='Raw physical failures can have valid measurements; numerical failure is not an audit error. Final checkpoint parity alone grants no route credit. Full source/actor/force/closed-state audit remains explicit in each panel.')
    save('review.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',required=True);p.add_argument('--output',required=True)
    p.add_argument('--namespace-proof',required=True);p.add_argument('--ledger-only',action='store_true');p.add_argument('--pending',action='store_true')
    a=p.parse_args();r=review(a.packet,a.output,namespace_proof=a.namespace_proof,ledger_only=a.ledger_only,pending=a.pending)
    print(json.dumps(r,indent=2,allow_nan=False));return 1 if r['status']=='audit_unresolved' else 0

if __name__=='__main__':raise SystemExit(main())
