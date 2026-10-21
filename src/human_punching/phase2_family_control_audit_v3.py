#!/usr/bin/env python3
"""Independent saved-array family-control launch supplement. No model or evaluator imports.
Derived transparently from the independent clock-v1 closed-launch pattern, but
uses the actual four-target control plan directly. No expanded schema is forged.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def owned(p):
 p=Path(p);p=(p if p.is_absolute() else ROOT/p).resolve(strict=True)
 if not p.is_relative_to(ROOT):raise ValueError('Unowned saved input')
 return p
def read(p):return json.loads(owned(p).read_text())
def bind(p):return dict(path=str(owned(p)),sha256=sha(owned(p)))
def npz(p):
 with np.load(owned(p),allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def json_identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def launch_snapshot_digest(state):
    """Independent typed digest; imports no launch/evaluator implementation."""
    result=hashlib.sha256()
    for key,value in sorted(state.items()):
        result.update(key.encode())
        if isinstance(value,np.ndarray):
            result.update(value.dtype.str.encode());result.update(str(value.shape).encode())
            result.update(np.ascontiguousarray(value).tobytes())
        else:result.update(json.dumps(value,sort_keys=True,allow_nan=False).encode())
    return result.hexdigest()


def closed_launch_audit(raw,receipt,plan,cells,launch,control,domain,targets,boundary_arrays,bank_q,bank_v):
    """Audit the isolated exact-prefix finalizer, including preserved rejections.

    This checks the supplied native/controller inventory against trace fields.
    It does not claim that inventory is a full mjSTATE_INTEGRATION restore state,
    qualify live G1 interaction, or convert a rejected request into a cycle.
    """
    checks={};accepted_by_world={};worlds=len(plan["candidates"])
    try:
        checks["control_protocol"]=(control.get("version")=="task10b_phase2_family_training_target_frozen_control_v2"
            and control.get("family")=="left_hook" and control.get("reference_engineering_correction_cap_rad")==.210
            and control.get("physical_joint_limits_relaxed") is False)
        checks["schema"]=(receipt["schema"]=="task10b_phase2_closed_protocol_native_receipt_v1"
            and receipt["physical_state_or_history_replaced"] is False
            and receipt["arbitrary_live_state_admission"] is False
            and receipt["real_time_operation_qualified"] is False)
        requests={r["trial_id"]:r for r in control["requests"]}
        checks["control_request_ids_unique"]=len(requests)==len(control["requests"])==32
        groups={g["snapshot"]["command_id"]:g for g in domain["selected_command_groups"]}
        candidates={t["candidate_identity"]:t for t in targets["geometry_candidates"]}
        checks["unique_canonical_targets"]=len(candidates)==len(targets["geometry_candidates"])
        checks["requested_trial_order"]=len(receipt["requests"])==worlds and len(plan["trials"])==worlds
        records=receipt["boundary_snapshots"]["snapshots"]
        checks["snapshot_schema"]=receipt["boundary_snapshots"]["schema"]=="task10b_phase2_lossless_launch_boundaries_v1"
        decoded=[];used_arrays=[]
        for index,record in enumerate(records):
            pair={}
            for side in ("before","after"):
                fields=record[side];state=dict(fields["scalar_fields"])
                if set(state)&set(fields["array_fields"]):raise ValueError("Overlapping boundary state fields")
                for key,name in fields["array_fields"].items():
                    value=boundary_arrays[name]
                    if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                        raise ValueError("Nonfinite or object boundary array")
                    state[key]=value
                    used_arrays.append(name)
                checks[f"snapshot_{index}_{side}_digest"]=launch_snapshot_digest(state)==fields["snapshot_sha256"]
                pair[side]=state
            checks[f"snapshot_{index}_inventory"]=set(pair["before"])==set(pair["after"])
            decoded.append(pair)
        checks["lossless_archive_inventory"]=len(used_arrays)==len(set(used_arrays)) and set(used_arrays)==set(boundary_arrays)
        decisions={}
        checks["unique_world_decisions"]=True
        for item in receipt["decisions"]:
            world=item["world"]
            if type(world) is not int or not 0<=world<worlds:raise ValueError("Invalid decision world")
            if world in decisions:checks["unique_world_decisions"]=False
            decisions[world]=item
        fields=("trial_id","request_identity","family","requested_ttc_s","style","candidate_id",
            "reference_sha256","reference_identity_kind","checkpoint_sha256","target_candidate_identity","snapshot_payload_sha256",
            "snapshot_slot_id","placement","head_point_velocity_w","target_w")
        target_fields=("trial_id","request_identity","target_candidate_identity","snapshot_payload_sha256",
            "snapshot_slot_id","placement","head_point_velocity_w","target_w")
        for world,candidate in enumerate(plan["candidates"]):
            key=f"world_{world}_";row=cells[candidate];trial_id=plan["trials"][world]
            bank=int(trial_id.rsplit('_bank_',1)[1]);target=candidates[row["target_candidate_identity"]]
            group=groups[row["command_id"]];snapshot=group["snapshot"]
            expected=dict(trial_id=trial_id,family=row['family'],requested_ttc_s=row['requested_ttc_s'],
                style=candidate,candidate_id=candidate,reference_sha256=row['reference_sha256'],
                reference_identity_kind=row['reference_identity_kind'],
                target_candidate_identity=target['candidate_identity'],snapshot_payload_sha256=target['payload_sha256'],
                checkpoint_sha256=control['accepted_checkpoint']['sha256'],snapshot_slot_id=target['snapshot_slot_id'],
                placement=target['placement'],head_point_velocity_w=target['head_point_velocity_w'],target_w=row['target_w'])
            initial=dict(version="task10b_independent_family_guard_zero_history_v1",qpos=bank_q[bank].tolist(),qvel=bank_v[bank].tolist(),
                previous_raw_action=[0.]*48,held_action_initialization="original independent sim.reset, before first actor call",
                controller_initialization="fresh native Simulation and original sim.reset; no saved-state restoration")
            initial_identity=json_identity(initial)
            expected['request_identity']=json_identity(dict(version=control['version'],request=expected.copy(),bank_id=bank,
                initial_state_identity=initial_identity))
            checks[key+"canonical_target_binding"]=(json_identity({k:v for k,v in target.items() if k!='candidate_identity'})==target['candidate_identity']
                and target['source_domain_sha256']==domain['bound_sha256']
                and target['command_id']==row['command_id'] and target['split']==group['split']==row['split']=='train'
                and target['original_replica_id']==row['original_replica_id']==snapshot['replica_id']
                and target['snapshot_slot_id']==snapshot['slot_id'] and target['payload_sha256']==snapshot['payload_sha256']
                and target['head_target_w']==row['target_w'] and target['placement']==dict(forward_m=0.,left_m=0.,yaw_rad=0.))
            checks[key+"request_exact"]=receipt['requests'][world]==requests[trial_id]==expected
            checks[key+"declared_initial_state"]=(np.array_equal(raw['qpos'][0,world],bank_q[bank])
                and np.array_equal(raw['qvel'][0,world],bank_v[bank])
                and control['initial_state_identities'][bank]==initial_identity)
            checks[key+"single_initialization"]=bool(np.all(raw["reset_count"][:,world]==1)
                and np.all(raw["policy_previous_raw_action"][0,world]==0)
                and np.all(raw["ctrl"][0,world]==0) and raw["reference_tick"][0,world]==0)
            launch_tick=row["launch_tick"];actor=launch_tick//4
            reached=actor<len(raw["policy_action"])
            checks[key+"reached_boundary_recorded"]=(world in decisions)==reached
            if not reached:continue
            item=decisions[world];index=item["boundary_snapshot_index"]
            before,after=(decoded[index][side] for side in ("before","after"))
            checks[key+"boundary_events"]=(type(launch_tick) is int and launch_tick%4==0
                and records[index]["episode_tick"]==launch_tick and item["reference_tick"]==launch_tick
                and item["trial_id"]==trial_id and expected["reference_identity_kind"]=="reference_content"
                and item["request"]=={k:v for k,v in expected.items() if k!="reference_identity_kind"})
            checks[key+"boundary_digest"]=(launch_snapshot_digest(before)==item["measured_boundary_state_sha256"]
                and launch_snapshot_digest(after)==item["after_planning_state_sha256"]
                and sorted(before)==item["native_fields_captured"])
            sources={"qpos":raw["policy_qpos"][actor],"qvel":raw["policy_qvel"][actor],
                "ctrl":raw["policy_ctrl"][actor],"qacc_warmstart":raw["policy_qacc_warmstart"][actor],
                "qfrc_applied":raw["qfrc_applied"][launch_tick],"xfrc_applied":raw["xfrc_applied"][launch_tick],
                "previous_raw_action":raw["policy_previous_raw_action"][actor],
                "ticks":raw["policy_reference_tick"][actor],"reference_ids":raw["policy_reference_id"][actor],
                "reset_counts":raw["reset_count"][launch_tick],"support":raw["support_contact"][launch_tick],
                "time":raw["time"][launch_tick],"phase_denominators_s":raw["policy_phase_denominator_s"][actor]}
            checks[key+"measured_state_history"]=all(np.array_equal(before[k],value) for k,value in sources.items())
            checks[key+"counter_state"]=(before["physics_steps"]==launch_tick and before["actor_updates"]==actor
                and np.array_equal(before["reference_ids"],plan["reference_ids"])
                and np.array_equal(before["phase_denominators_s"],np.asarray(plan["phase_denominators_s"],np.float32)))
            checks[key+"reference_and_phase_immutable"]=bool(
                np.all(raw["policy_reference_id"][:,world]==plan["reference_ids"][world])
                and np.all(raw["policy_phase_denominator_s"][:,world]==np.float32(plan["phase_denominators_s"][world])))
            checks[key+"actuator_state_inventory"]=before["act"].shape==(worlds,0) and after["act"].shape==(worlds,0)
            clocks=[item[k] for k in ("planning_wall_s","target_callback_wall_s","cache_lookup_wall_s","simulation_pause_wall_s")]
            checks[key+"wall_clocks"]=(all(type(c) in (int,float) and np.isfinite(c) and c>=0 for c in clocks)
                and clocks[1]+clocks[2]<=clocks[0]+1.e-6 and clocks[3]==clocks[0]
                and item["simulated_time_advanced_s"]==0 and np.array_equal(before["time"],after["time"]))
            fresh=item["target_input_receipt"]
            target_ok=(all(fresh.get(k)==expected[k] for k in target_fields)
                and type(fresh.get("capture_episode_tick")) is int and fresh["capture_episode_tick"]==launch_tick
                and not isinstance(fresh.get("input_episode_time_s"),bool) and fresh.get("input_episode_time_s")==launch_tick*.005
                and fresh.get("source_kind")=="fixed_canonical_snapshot" and fresh.get("no_prediction") is True
                and fresh.get("head_geometry")=="head_collision" and fresh.get("head_velocity_method")=="jacobian_qvel")
            current=item["observed_current_request"]
            request_ok=isinstance(current,dict) and all(current.get(k)==expected[k] for k in fields)
            ready=bool(raw["policy_readiness"][actor,world])
            checks[key+"current_readiness"]=item["current_positional_support_ready"] is ready
            changed=launch_snapshot_digest(before)!=launch_snapshot_digest(after)
            reason=("physical_or_controller_state_changed_during_planning" if changed else
                "command_or_reference_identity_changed" if not request_ok else
                "launch_target_input_mismatch_or_stale" if not target_ok else
                "current_positional_support_readiness_failed" if not ready or not before["support"][world].all() else
                "final_planning_late" if clocks[0]>.020 else None)
            accepted=reason is None;accepted_by_world[world]=bool(item["accepted"])
            checks[key+"disposition"]=(item["accepted"] is accepted and item["reason"]==reason
                and item["reasons"]==([] if accepted else [reason])
                and item["actual_launch_tick"]==(launch_tick if accepted else None))
            checks[key+"recorded_launch_flags"]=(bool(raw["policy_launched"][actor,world])==accepted
                and bool(raw["policy_rejected"][actor,world])==(not accepted)
                and not bool(before["launched"][world]) and not bool(before["rejected"][world]))
            interval_after=np.arange(len(raw["reference_tick"]))>launch_tick
            checks[key+"immutable_disposition_through_recovery"]=(
                np.array_equal(raw["launched"][:,world],interval_after&accepted)
                and np.array_equal(raw["rejected"][:,world],interval_after&(not accepted)))
            expected_ticks=np.arange(len(raw["reference_tick"]),dtype=raw["reference_tick"].dtype)
            if not accepted:expected_ticks=np.minimum(expected_ticks,launch_tick)
            checks[key+"uninterrupted_reference_clock"]=np.array_equal(raw["reference_tick"][:,world],expected_ticks)
            checks[key+"scope"]=(item["same_continuous_cached_reference"] is True
                and item["exact_joint_return_required"] is False and item["quiet_hook_gate_added"] is False
                and item["live_g1_state_recaptured"] is False and item["full_saved_state_restore"] is False
                and item["hardware_real_time_qualified"] is False)
        return dict(passed=all(bool(value) for value in checks.values()),checks={k:bool(v) for k,v in checks.items()},
            accepted_by_world=accepted_by_world,decisions=len(decisions),snapshots=len(records),
            complete_saved_state_restore_audited=False,
            live_target_execution_audited=False)
    except (KeyError,IndexError,TypeError,ValueError,OverflowError) as error:
        return dict(passed=False,checks={k:bool(v) for k,v in checks.items()},accepted_by_world=accepted_by_world,
            reason="Malformed or inconsistent closed-protocol evidence",exception=dict(type=type(error).__name__,message=str(error)))


VERSION='task10b_phase2_family_control_saved_audit_v3'
PLAN_SCHEMA='task10b_phase2_family_training_target_frozen_control_v2_plan'
LAUNCH_SCHEMA='task10b_phase2_family_target_control_launch_v2'


def ledger_checks(control,native):
    requests={r['trial_id']:r for r in control['requests']}
    states={r['candidate_id']:r for r in control['reference_dispositions']}
    supplied={r['trial_id']:r for r in native}
    checks={'exact32':len(requests)==len(native)==len(supplied)==32 and set(requests)==set(supplied)}
    negative=[]
    for trial,request in requests.items():
        state=states[request['candidate_id']];row=supplied.get(trial,{})
        checks[trial+':admission']=row.get('native_execution_admitted') is state['native_execution_admitted']
        checks[trial+':reference_disposition']=(row.get('reference_generation_disposition')==state['generation_disposition']
            and row.get('rejection_reasons')==state['rejection_reasons'])
        if not state['native_execution_admitted']:
            checks[trial+':negative_zero_execution']=(row.get('status')=='reference_unavailable'
                and row.get('execution_extent')=='known_zero_before_attempt' and row.get('actual_physics_steps')==0
                and row.get('execution_complete') is False and row.get('numeric_complete_cycle') is False)
            negative.append(trial)
    return dict(passed=all(checks.values()),checks=checks,negative_trial_ids=negative)


def audit_run(run,output):
    from .phase2_expanded_clock_audit_v1 import audit_batch
    run=owned(run);output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'outputs') or output.exists():raise ValueError('Fresh owned output required')
    control=read(run/'control_plan.json');launch=read(run/'launch.json')
    if (control.get('schema')!=PLAN_SCHEMA or control.get('planned_trials')!=32
            or control.get('family')!='left_hook'
            or control.get('version')!='task10b_phase2_family_training_target_frozen_control_v2'
            or control.get('reference_engineering_correction_cap_rad')!=.210
            or control.get('physical_joint_limits_relaxed') is not False
            or launch.get('target_control_version')!=control['version']
            or launch.get('schema')!=LAUNCH_SCHEMA or launch.get('family')!=control['family']
            or launch.get('training') is not False or launch.get('physical_qualification') is not False):
        raise ValueError('Explicit frozen family-control source schemas required')
    for name,digest in control['input_inventory'].items():
        if sha(owned(name))!=digest:raise ValueError('Control bound input changed')
    original_plan=owned(launch['control_plan']['path'])
    if sha(original_plan)!=launch['control_plan']['sha256'] or read(original_plan)!=control:
        raise ValueError('Saved plan differs from immutable input plan')
    parity=read(run/'restored_checkpoint_parity.json') if (run/'restored_checkpoint_parity.json').exists() else None
    checkpoint_ok=parity==dict(parameters_equal=True,actions_equal=True,maximum_action_error=0.0)
    catalog=read(control['catalog']['path']);cells={row['candidate_id']:row for row in catalog['cells']}
    if sha(owned(control['catalog']['path']))!=control['catalog']['sha256']:raise ValueError('Catalog changed')
    domain_path=ROOT/'outputs/task10b/p2domain01/domain.json';target_path=ROOT/'outputs/task10b/p2targets01/target_geometry.json'
    domain=read(domain_path);domain['bound_sha256']=sha(domain_path);targets=read(target_path)
    for p in (domain_path,target_path):
        if control['input_inventory'][str(p.relative_to(ROOT))]!=sha(p):raise ValueError('Canonical snapshot geometry changed')
    bank_binding=launch['release_inputs']['bank'];bank_path=owned(bank_binding['path'])
    if sha(bank_path)!=bank_binding['sha256']:raise ValueError('Original bank changed')
    bank=npz(bank_path);q=bank['qpos'][:8].astype(np.float32);v=bank['qvel'][:8].astype(np.float32)
    native=read(run/'outcomes.json');extent=ledger_checks(control,native);by_id={row['trial_id']:row for row in native}
    if not extent['passed']:raise ValueError('Declared32 ledger or zero-execution negatives changed')
    panels=[];unresolved=[];audited=[]
    for index,candidate in enumerate(control['selected_candidate_ids']):
        folder=run/f'validation_{index:05d}'/candidate;row=cells[candidate]
        selected=[r['trial_id'] for r in control['requests'] if r['candidate_id']==candidate]
        if candidate not in control['admitted_candidate_ids']:
            if folder.exists():raise ValueError('Unavailable reference unexpectedly has a native output directory')
            continue
        if not all((folder/name).exists() for name in ('raw_trace.npz','summary.json','plan.json','closed_protocol_launch.json')):
            unresolved.extend(selected);continue
        raw=npz(folder/'raw_trace.npz');plan=read(folder/'plan.json');receipt=read(folder/'closed_protocol_launch.json')
        path=owned(receipt['boundary_snapshots']['path'])
        if path!=folder/'closed_boundary_snapshots.npz' or sha(path)!=receipt['boundary_snapshots']['sha256']:
            raise ValueError('Boundary archive identity changed')
        measured=closed_launch_audit(raw,receipt,plan,cells,launch,control,domain,targets,npz(path),q,v)
        clocks=audit_batch(folder)
        ref=npz(row['reference_path']);tick=row['launch_tick']
        hold=bool(row['preparation_complete_tick']==0 and row['start_mode']=='family_guard'
            and np.array_equal(ref['qpos'][:tick+1],np.repeat(ref['qpos'][:1],tick+1,axis=0)))
        summary=read(folder/'summary.json');reports={t['trial_id']:t for t in summary['trials']}
        exact=bool(len(reports)==8 and set(reports)==set(selected)==set(plan['trials']))
        counters=bool(len(raw['pre_qpos'])==control['horizons'][candidate]['planned_physics_steps']
            and len(raw['qpos'])==len(raw['pre_qpos'])+1 and len(raw['policy_action'])*4==len(raw['pre_qpos']))
        numeric=[]
        for trial in selected:
            world=plan['trials'].index(trial);saved=by_id[trial];report=reports.get(trial,{})
            accepted=measured.get('accepted_by_world',{}).get(world,False)
            expected=bool(counters and summary['status']=='executed' and report.get('numeric_complete_cycle') and accepted and measured['passed'])
            numeric.append(saved['numeric_complete_cycle']==expected)
        integrity=bool(measured['passed'] and clocks['passed'] and hold and exact and counters and all(numeric))
        panels.append(dict(candidate_id=candidate,passed=integrity,closed_launch=measured,event_clocks=clocks,
            raw_trace=bind(folder/'raw_trace.npz'),closed_receipt=bind(folder/'closed_protocol_launch.json'),
            constant_initial_hold=hold,complete_counters=counters,exact8requests=exact,numeric_ledger_consistent=all(numeric),
            preparation_complete_time_s=0.,source_hold_is_not_connector=True))
        audited.extend(selected)
    result=dict(schema=VERSION,passed=bool(checkpoint_ok and extent['passed'] and panels and all(p['passed'] for p in panels) and not unresolved),
        control_plan=bind(run/'control_plan.json'),launch=bind(run/'launch.json'),family=control['family'],
        checkpoint_parity_passed=checkpoint_ok,panels=panels,planned_trials=32,executed_trials_audited=len(audited),
        reference_unavailable_trials=len(extent['negative_trial_ids']),negative_ledger=extent,unresolved_trial_ids=unresolved,
        source=bind(__file__),pattern_ancestry=bind(ROOT/'outputs/task10b/p2tcaudithost01/scripts/closed_launch.py'),
        no_fake_expanded_schema=True,physical_qualification=False,route_credit=False,new_physics_steps=0,
        limits='Saved current-readiness, exact target/request, controller-history, first-write clocks and complete negative ledger. Raw FK/forces/actor audit is separate; no live G1 or shared-state claim.')
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return result
