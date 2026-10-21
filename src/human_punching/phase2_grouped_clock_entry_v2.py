"""Expanded event clocks around unchanged original grouped PPO and parity gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from . import phase2_grouped_entry_v3 as parent
from .phase2_expanded_clocks_v1 import VERSION,binding,validation_clock_scope
from .phase2_expanded_clock_audit_v1 import audit_batch


def execute_configuration(path,expected_sha256,output):
    folder=parent._owned(output)
    if folder.exists():raise FileExistsError('Fresh clock-adapted grouped output required')
    with validation_clock_scope():result=parent.execute_configuration(path,expected_sha256,folder)
    audits=[];errors=[]
    for batch in sorted((folder/'held_out_validation').glob('validation_*/batch_*')):
        try:
            audit=audit_batch(batch)
        except Exception as error:
            audit=dict(passed=False,error=dict(type=type(error).__name__,message=str(error)))
        audits.append(dict(batch=str(batch.relative_to(folder)),**audit))
    try:
        for panel in sorted((folder/'held_out_validation').glob('validation_*/summary.json')):
            value=json.loads(panel.read_text())
            if value.get('status')=='executed' and not all(t.get('event_clock_version')==VERSION for t in value['trials']):
                errors.append(dict(panel=str(panel),reason='missing_new_trial_event_version'))
    except Exception as error:errors.append(dict(type=type(error).__name__,message=str(error)))
    passed=bool(audits) and all(a['passed'] for a in audits) and not errors
    receipt=dict(schema='task10b_phase2_grouped_clock_entry_v2',event_clock_version=VERSION,
        passed=passed,source=binding(__file__),configuration=dict(path=str(path),sha256=expected_sha256),
        inherited_integrity=result.get('grouped_integrity_v3'),batch_audits=audits,errors=errors,
        original_evaluator_source_modified=False,original_ppo_modified=False,physics_actor_reference_changes=False,
        old_artifacts_rewritten=False,qualification=False,formal_acceptance=False)
    destination=folder/'grouped_event_clocks_v2.json'
    with destination.open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return {**result,'status':result['status'] if passed else 'failed','event_clocks_pass':passed,'event_clock_receipt':binding(destination)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--configuration',type=Path,required=True)
    p.add_argument('--configuration-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv);r=execute_configuration(a.configuration,a.configuration_sha256,a.output)
    print(json.dumps(parent._json_evidence(r),sort_keys=True,allow_nan=False))
    return 0 if r['status']=='bounded_stop' and r.get('grouped_integrity_pass') is True and r['event_clocks_pass'] else 1

if __name__=='__main__':raise SystemExit(main())
