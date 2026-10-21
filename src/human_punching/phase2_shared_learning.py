"""Original bounded specialist PPO on exact shared-preparation references.

All training resets are fresh-state rehearsal. The 96 held-out native shared
cycles use a separate full-state/parity-bound callback and retain all failures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass,replace
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .phase2_cache import _owned_path,_preflight
from .phase2_deadline import ExecutionDeadline,BoundedExecutionInterruption
from .phase2_domain import fingerprint
from .phase2_grouped_learning import (GroupedTrainingDeclaration,grouped_stopping_rules,
    validate_validation_descriptor,validate_callback_panel,_owned_input,_read_bound)
from .phase2_learning import (NativeSpecialistEnv,PilotBudget,family_arguments,fresh_policy,
    train_pilot,_tensor_view)
from .phase2_phases import PhaseSchedule
from .phase2_sampling import IntegerReferencePlayback,VERSION as SAMPLING_VERSION
from .phase2_shared_catalog import START_QUANTIZATION,validate_start_bank
from .phase2_shared_reference import resolve_shared_joints
from .phase2_runtime_pack import VERSION as STORAGE_VERSION
from .wbc.provenance import PROJECT_ROOT,sha256_file,writable_path,write_json,runtime_record

RESET_PROTOCOL="task10b_shared_exact_d3_start_half_source_rehearsal_v1"
CONFIGURATION_SCHEMA="task10b_phase2_shared_training_configuration_v1"


@dataclass(frozen=True)
class SharedTrainingDeclaration(GroupedTrainingDeclaration):
    start_bank_sha256:str=""
    start_mode:str="shared_jab_saved_state"
    reset_protocol:str=RESET_PROTOCOL
    start_quantization_version:str=START_QUANTIZATION
    sampling_version:str=SAMPLING_VERSION

    def __post_init__(self):
        super().__post_init__()
        if (self.start_mode!="shared_jab_saved_state" or self.reset_protocol!=RESET_PROTOCOL
                or self.start_quantization_version!=START_QUANTIZATION or self.sampling_version!=SAMPLING_VERSION
                or len(self.start_bank_sha256)!=64 or any(c not in "0123456789abcdef" for c in self.start_bank_sha256)):
            raise ValueError("Exact versioned shared start/sampling/reset identities required")


def select_training_rows(declaration,catalog,domain,targets):
    """Select exactly32×two×four by metadata before any held-out archive opens."""
    if (catalog.get("schema")!="task10b_phase2_reference_catalog_v1"
            or domain.get("schema")!="Task10BPhase2Domain/v1"
            or targets.get("schema")!="Task10BPhase2RegisteredTargets/v1"):
        raise ValueError("Canonical shared catalog/domain/geometry required")
    groups={r['snapshot']['command_id']:r for r in domain['selected_command_groups']}
    geometry={r['candidate_identity']:r for r in targets['geometry_candidates']}
    if len(groups)!=len(domain['selected_command_groups']) or len(geometry)!=len(targets['geometry_candidates']):
        raise ValueError("Duplicate canonical group or target identity")
    if {g for g,r in groups.items() if r['split']=='train'}!=set(declaration.command_groups):
        raise ValueError("Training requires exactly the canonical32 training groups")
    placements=dict(declaration.target_candidate_identities);selected={};excluded=[];names=set();banks=set()
    for row in catalog['cells']:
        candidate=row['candidate_id']
        if not isinstance(candidate,str) or not candidate or candidate in names:raise ValueError("Unique reference candidate IDs required")
        names.add(candidate)
        reason=('other_family' if row.get('family')!=declaration.family else
            'not_a_declared_training_group' if row.get('command_id') not in declaration.command_groups else
            'held_out_or_wrong_split' if row.get('split')!='train' else
            'not_a_trained_timing' if row.get('requested_ttc_s') not in declaration.trained_timings_s else None)
        if reason:excluded.append(dict(candidate_id=candidate,reason=reason));continue
        group=row['command_id'];target=geometry[placements[group]];snapshot=groups[group]['snapshot']
        if (row.get('start_mode')!='shared_jab_saved_state' or type(row.get('start_index')) is not int or row['start_index'] not in range(4)
                or row.get('kinematic_admission') is not True or row.get('original_replica_id')!=0
                or row.get('recipe_sha256')!=declaration.recipe_sha256 or row.get('start_bank_sha256')!=declaration.start_bank_sha256
                or row.get('target_candidate_identity')!=placements[group]
                or fingerprint({k:v for k,v in target.items() if k!='candidate_identity'})!=target['candidate_identity']
                or target['source_domain_sha256']!=declaration.domain_sha256 or target['command_id']!=group
                or target['split']!='train' or target['original_replica_id']!=0 or snapshot['replica_id']!=0
                or target['snapshot_slot_id']!=snapshot['slot_id'] or target['payload_sha256']!=snapshot['payload_sha256']
                or tuple(row.get('target_w',()))!=tuple(target['head_target_w'])
                or row.get('own_family_guard_targets_serialized') is not True):
            raise ValueError("Shared training request lacks exact admitted target/start/guard/recipe binding")
        schedule=PhaseSchedule(row['preparation_complete_tick'],row['launch_tick'],row['outward_end_tick'],
            row['guard_hold_tick'],row['reference_end_tick'],row['phase_semantics_version'])
        if (not 0<schedule.preparation_complete_tick<=200 or schedule.reference_end_tick>6000
                or abs(row['duration_s']-schedule.reference_end_tick*.005)>1e-8
                or row['phase_denominator_s']!=schedule.phase_denominator_s
                or row['episode_duration_s']!=schedule.phase_denominator_s):
            raise ValueError("Explicit bounded preparation and complete reference horizon required")
        key=(group,row['requested_ttc_s'],row['start_index'])
        if key in selected:raise ValueError("Multiple references for one exact shared training request")
        selected[key]=json.loads(json.dumps(row,allow_nan=False));banks.add(row['start_bank_path'])
    expected=set(itertools.product(declaration.command_groups,declaration.trained_timings_s,range(4)))
    if set(selected)!=expected or len(banks)!=1:
        raise ValueError("Exactly256 admitted references and one identical shared start bank required; no resampling")
    rows=[selected[key] for key in itertools.product(declaration.command_groups,declaration.trained_timings_s,range(4))]
    return rows,dict(schema='task10b_phase2_shared_training_selection_v1',declaration_identity=declaration.identity,
        selected_candidate_ids=[r['candidate_id'] for r in rows],selected_rows_sha256=fingerprint(rows),excluded_rows=excluded,
        training_groups=list(declaration.command_groups),trained_timings_s=list(declaration.trained_timings_s),
        interpolation_timing_s=declaration.interpolation_timing_s,interpolation_used_for_training=False,
        sampling_split='train',reset_protocol=RESET_PROTOCOL,no_nearest_reference_substitution=True,
        starts_are_four_distinct_references=True,uninterrupted_cycle_qualification=False)


def preflight_references(rows,declaration,domain,jab_paths):
    """Bound one host load chunk and the complete owned GPU tensor inventory."""
    headers={};total=0;device_reservation=0;host_peak=0
    bank=_owned_path(rows[0]['start_bank_path'])
    if sha256_file(bank)!=declaration.start_bank_sha256:raise ValueError("Shared start bank changed")
    _preflight(bank,declaration.start_bank_sha256,1024**2)
    with np.load(bank,allow_pickle=False) as archive:q,v=archive['qpos'].copy(),archive['qvel'].copy()
    with np.load(jab_paths['reference'],allow_pickle=False) as source:nominal=source['qpos'][0].copy()
    validate_start_bank(q,v,nominal,json.loads(jab_paths['body_metadata'].read_text()),domain,'train')
    for row in rows:
        for name in ('reference','legacy_runtime'):
            path=_owned_path(row[name+'_path']);digest=row[name+'_sha256']
            if path not in headers:
                headers[path]=_preflight(path,digest,2*1024**3);total+=headers[path].uncompressed_array_bytes
                if total>2*1024**3:raise ValueError("Shared training reference pool exceeds2GiB")
            elif headers[path].sha256!=digest:raise ValueError("Conflicting shared reference hash")
        header=headers[_owned_path(row['reference_path'])];schema={a.name:a for a in header.arrays};n=schema['qpos'].shape[0]
        packed=row.get('runtime_storage_version')==STORAGE_VERSION
        if row.get('runtime_storage_version') not in (None,STORAGE_VERSION):raise ValueError("Unknown shared runtime storage recipe")
        if (schema['qpos'].shape!=(n,55) or schema['qvel'].shape!=(n,54) or not 4<=n<=6001
                or (not packed and (schema.get('qacc') is None or schema['qacc'].shape!=(n,54)))
                or schema.get('torque') is None or schema['torque'].shape!=(n,48)
                or schema.get('family_guard_glove_chest_pos') is None or schema['family_guard_glove_chest_pos'].shape!=(2,3)
                or schema.get('family_guard_qpos') is None or schema['family_guard_qpos'].shape!=(55,)):
            raise ValueError("Shared reference dimensions or explicit family guard targets differ")
        # Native ReferenceLibrary retains t/q/v/24bodypositions/twohandposes/
        # rotations/torque in float32.1536bytes/frame conservatively covers that
        # finite inventory and constants. CPU FK lists/load arrays exist one row
        # at a time; they are a separate <=256MiB host workspace reservation.
        device_reservation+=n*1536+8192
        source_path=None
        if packed:
            source_path=_owned_input(row['unpacked_reference_path'])
            if sha256_file(source_path)!=row['unpacked_reference_sha256']:raise ValueError("Full-precision shared evidence changed")
            evidence_header=_preflight(source_path,row['unpacked_reference_sha256'],64*1024**2)
            host_peak=max(host_peak,evidence_header.uncompressed_array_bytes*6+header.uncompressed_array_bytes*3+n*4096)
        else:host_peak=max(host_peak,header.uncompressed_array_bytes*6+n*4096)
        if device_reservation>declaration.resident_reference_bytes or host_peak>256*1024**2:
            raise ValueError("Shared reference persistent tensor or host chunk reservation exceeded")
        with np.load(row['reference_path'],allow_pickle=False) as archive:
            clock=np.asarray(archive['t'],float);start=row['start_index'];prep=row['preparation_complete_tick']
            if (clock.shape!=(n,) or clock[0]!=0 or not np.allclose(np.diff(clock),.005,atol=1e-12,rtol=0)
                    or abs(clock[-1]-row['duration_s'])>1e-8 or not np.array_equal(archive['head_center'],row['target_w'])
                    or not np.array_equal(archive['qpos'][0].astype(np.float32),q[start].astype(np.float32))
                    or not np.array_equal(archive['qvel'][0].astype(np.float32),v[start].astype(np.float32))
                    or not np.array_equal(archive['family_guard_qpos'].astype(np.float32),archive['qpos'][prep].astype(np.float32))
                    or not all(np.isfinite(archive[name]).all() for name in (('qpos','qvel','torque') if packed else ('qpos','qvel','qacc','torque')))
                    or not np.array_equal(archive['qpos'][:,:7].astype(np.float32),np.tile(q[start,:7].astype(np.float32),(n,1)))
                    or np.any(archive['qvel'][:,:6]) or (not packed and np.any(archive['qacc'][:,:6]))
                    or not np.isfinite(archive['family_guard_glove_chest_pos']).all()
                    or int(archive['active_hand_index'])!=(0 if declaration.family in ('jab','left_hook') else 1)):
                raise ValueError("Shared start/grid/guard/target/active-hand identity differs")
            if source_path is not None:
                with np.load(source_path,allow_pickle=False) as evidence:
                    if (evidence['qacc'].shape!=(n,54) or not np.isfinite(evidence['qacc']).all() or np.any(evidence['qacc'][:,:6])
                            or not all(np.array_equal(archive[name],evidence[name].astype(np.float32)) for name in ('qpos','qvel','torque'))
                            or not np.array_equal(archive['t'],evidence['t'])
                            or not np.array_equal(archive['head_center'],evidence['head_center'])
                            or not np.array_equal(archive['family_guard_qpos'],evidence['family_guard_qpos'])
                            or not np.array_equal(archive['family_guard_glove_chest_pos'],evidence['family_guard_glove_chest_pos'])):
                        raise ValueError("Packed shared runtime differs from exact native conversion of preserved source evidence")
    return dict(unique_reference_archives=len(headers),unique_reference_array_bytes=total,
        persistent_reference_tensor_reservation_bytes=device_reservation,host_working_chunk_reservation_bytes=host_peak,
        persistent_reference_limit_bytes=declaration.resident_reference_bytes,host_working_chunk_limit_bytes=256*1024**2,
        packed_offline_derivative_evidence_validated=True,
        memory_scope='Owned reference arrays only, no total process or simulator GPU-memory peak claim')


class SharedSpecialistEnv(IntegerReferencePlayback,NativeSpecialistEnv):
    """Original actor/controller/PPO reward, with explicit shared reset targets."""
    def __init__(self,*args,deadline,resident_limit,**kwargs):
        self.execution_deadline=deadline
        super().__init__(*args,**kwargs)
        self.install_integer_reference_playback()
        guards=[];upper=resolve_shared_joints(self.model)['upper_qpos']
        nominal_upper=torch.as_tensor(self.nominal_qpos[upper],dtype=torch.float32,device=self.device)
        for row,reference in zip(self.library.rows,self.library.arrays,strict=True):
            with np.load(row['reference_path'],allow_pickle=False) as archive:
                centers=torch.as_tensor(archive['family_guard_glove_chest_pos'].copy(),dtype=torch.float32,device=self.device)
                guard=torch.as_tensor(archive['family_guard_qpos'].copy(),dtype=torch.float32,device=self.device)
            if (not torch.allclose(centers,self.nominal_centers,atol=1e-6,rtol=0)
                    or not torch.equal(guard[upper],nominal_upper)):
                raise ValueError("Serialized guard targets differ from the accepted own-family centers/upper posture")
            reference['guard_glove_chest_pos']=centers;guards.append(guard)
        self.family_guard_qpos=torch.stack(guards)
        self.reference_resident_bytes=self.library.bytes+self.family_guard_qpos.numel()*4
        if self.reference_resident_bytes>resident_limit:raise ValueError("Measured shared reference tensor bound exceeded")

    def reset(self,env_ids=None,*,random_phase=True,perturb=True,reference_ids=None,initial_qpos=None,initial_qvel=None):
        # `perturb=True` is the original trainer's reset signature. This version
        # intentionally applies no additional noise: each selected reference
        # already starts at one exact frozen D3 state.
        ids=torch.arange(self.n,device=self.device) if env_ids is None else env_ids
        count=len(ids);self.sim.reset(ids)
        self.ids_ref[ids]=(torch.randint(len(self.library.rows),(count,),device=self.device) if reference_ids is None
            else torch.as_tensor(reference_ids,dtype=torch.long,device=self.device))
        self.ticks[ids]=0;self.rejected[ids]=False;self.launched[ids]=False
        if random_phase:
            rehearsal=torch.rand(count,device=self.device)>=.5
            prep=torch.tensor([r['preparation_complete_tick'] for r in self.library.rows],device=self.device)[self.ids_ref[ids]]
            end=torch.tensor([r['reference_end_tick'] for r in self.library.rows],device=self.device)[self.ids_ref[ids]]
            # Half exact connector starts, half original-style source-phase
            # rehearsal over the first80% of the retained family source clock.
            sampled=prep+(torch.rand(count,device=self.device)*.8*(end-prep)).long()
            self.ticks[ids]=torch.where(rehearsal,sampled,0)
        pose,velocity=self.ref('qpos')[ids].clone(),self.ref('qvel')[ids].clone()
        if initial_qpos is not None or initial_qvel is not None:
            if random_phase or initial_qpos is None or initial_qvel is None:
                raise ValueError("Explicit independent initialization requires both q/v and no random phase")
            pose=torch.as_tensor(initial_qpos,dtype=torch.float32,device=self.device)
            velocity=torch.as_tensor(initial_qvel,dtype=torch.float32,device=self.device)
            if pose.shape!=(count,55) or velocity.shape!=(count,54):raise ValueError("Complete shared rehearsal initialization required")
        _tensor_view(self.sim.data.qpos)[ids]=pose;_tensor_view(self.sim.data.qvel)[ids]=velocity
        self.last_action[ids]=0;self.reset_counts[ids]+=1;self.sim.forward()
        rotation=_tensor_view(self.sim.data.xmat)[ids,self.hips].reshape(count,3,3)
        self.initial_up[ids]=rotation[:,2,:];self.last_support=self.support_contact()
        return self.observe()

    def reward(self,action,peak_limit):
        result=super().reward(action,peak_limit)
        if self.args.guard_joint_weight:
            q=_tensor_view(self.sim.data.qpos)
            old=self.library.sample('qpos',self.ids_ref,torch.zeros_like(self.clock))
            target=self.family_guard_qpos[self.ids_ref]
            old_error=(q[:,self.qadr]-old[:,self.qadr]).abs().amax(-1)
            new_error=(q[:,self.qadr]-target[:,self.qadr]).abs().amax(-1)
            weight=self.args.guard_joint_weight*(self.clock>=self.ref('guard_hold_start')).float()
            result+=weight*(torch.exp(-(new_error/.1).square())-torch.exp(-(old_error/.1).square()))
        return result

    def step(self,action):
        try:self.execution_deadline.checkpoint('shared_training_actor_boundary',progress={
            'physics_steps':self.physics_steps_executed,'actor_updates':self.actor_updates_executed})
        except BoundedExecutionInterruption as error:
            self.deadline_interruption_receipt=error.receipt;raise
        return super().step(action)


def load_configuration(path,expected_sha256):
    data=_read_bound(path,expected_sha256)
    if data.get('schema')!=CONFIGURATION_SCHEMA or data.get('start_mode')!='shared_jab_saved_state':
        raise ValueError("Frozen shared training configuration required")
    items={}
    for name in ('declaration','catalog','domain','target_geometry','predicate','mask','recipe','validation_configuration'):
        row=data[name]
        if set(row)!={'path','sha256'}:raise ValueError("Exact shared configuration input records required")
        items[name]=dict(path=str(_owned_input(row['path'])),sha256=row['sha256'])
        if sha256_file(items[name]['path'])!=row['sha256']:raise ValueError("Shared training configuration input changed")
    declaration=SharedTrainingDeclaration(**_read_bound(items['declaration']['path'],items['declaration']['sha256']))
    for name,key in (('domain','domain_sha256'),('target_geometry','target_geometry_sha256'),('predicate','predicate_sha256'),
            ('mask','declared_mask_sha256'),('recipe','recipe_sha256')):
        if items[name]['sha256']!=getattr(declaration,key):raise ValueError("Shared declaration/input identities differ")
    budget=PilotBudget(**data['budget'])
    if data.get('stopping_rules')!=grouped_stopping_rules(budget):raise ValueError("Strict original finite physical stops must be declared")
    release=Path(data['release']).resolve(strict=True)
    if not release.is_relative_to(PROJECT_ROOT.resolve()):raise ValueError("Accepted release must resolve inside human-punching")
    return declaration,items,budget,data,release


def execute_configuration(path,expected_sha256,output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError("Shared PPO requires a counted Slurm CUDA allocation")
    started=time.monotonic();declaration,items,budget,config,release=load_configuration(path,expected_sha256)
    deadline=ExecutionDeadline(budget.maximum_wall_s,serialization_reserve_s=budget.shutdown_reserve_s,
        session_started_monotonic=started)
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'declaration.json',asdict(declaration));write_json(folder/'stopping_rules.json',config['stopping_rules'])
    write_json(folder/'reset_contract.json',dict(version=RESET_PROTOCOL,exact_selected_connector_start_probability=.5,
        retained_source_rehearsal_probability=.5,source_rehearsal_fraction_end=.8,
        source_rehearsal_tick_definition='preparation_complete_tick + floor(U[0,1)*.8*(reference_end_tick-preparation_complete_tick))',
        additional_position_or_velocity_noise=False,fresh_sim_reset=True,previous_raw_action_initialization='zero after each independent training rehearsal reset',
        no_commanded_cycle_credit=True,within_cycle_qualified_state_restoration=False,
        original_inner_plan_label='Inherited train_pilot labels resets as original half-guard/half sampled-reference rehearsal. This explicit version defines actual shared reset ticks and exact per-start references.'))
    env=None;summary=None;error=None
    try:
        with deadline.signal_handlers():
            from .phase2_shared_validation_callback import SharedValidationCallback
            callback=SharedValidationCallback.from_configuration(items['validation_configuration']['path'],items['validation_configuration']['sha256'],
                declaration,release,folder/'held_out_validation')
            domain=_read_bound(items['domain']['path'],declaration.domain_sha256)
            targets=_read_bound(items['target_geometry']['path'],declaration.target_geometry_sha256)
            catalog=_read_bound(items['catalog']['path'],items['catalog']['sha256'])
            rows,selection=select_training_rows(declaration,catalog,domain,targets)
            args,paths=family_arguments(release,declaration.family);_,jab_paths=family_arguments(release,'jab')
            if sha256_file(paths['checkpoint'])!=declaration.parent_checkpoint_sha256:raise ValueError("Shared PPO must transfer accepted family weights")
            residency=preflight_references(rows,declaration,domain,jab_paths)
            validate_validation_descriptor(declaration,callback.descriptor,domain)
            write_json(folder/'selection.json',selection);write_json(folder/'reference_preflight.json',residency)
            runtime=runtime_record()
            if not runtime['declared_pins_match']:raise ValueError("Phase2 runtime pins differ")
            write_json(folder/'launch.json',dict(schema='task10b_phase2_shared_training_launch_v1',slurm_job_id=os.environ['SLURM_JOB_ID'],
                runtime=runtime,declaration_identity=declaration.identity,budget=asdict(budget),
                reset_protocol=RESET_PROTOCOL,source_rehearsal_no_commanded_cycle_credit=True,
                fresh_transition_cap=budget.transition_cap,selection_sha256=fingerprint(selection),
                validation_descriptor_identity=fingerprint(callback.descriptor),sampling_version=SAMPLING_VERSION,
                native_qualification=False,sources={str(Path(__file__).resolve()):sha256_file(__file__)}))
            torch.set_num_threads(1);torch.manual_seed(declaration.seed);np.random.seed(declaration.seed)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            from .wbc.telemetry.cycle_qualification import native_stream_scope
            with native_stream_scope():
                deadline.checkpoint('before_shared_training_initialization')
                env=SharedSpecialistEnv(args,rows,num_envs=128,deadline=deadline,resident_limit=declaration.resident_reference_bytes)
                deadline.checkpoint('after_shared_training_initialization');policy=fresh_policy(paths,env.device)
                write_json(folder/'reference_residency.json',dict(**residency,actual_reference_tensor_bytes=env.reference_resident_bytes))
                def validate(policy,update):
                    deadline.checkpoint('before_shared_physical_validation',progress=dict(update=update))
                    return validate_callback_panel(callback(policy,update,deadline=deadline),callback.descriptor)
                remaining=budget.maximum_wall_s-(time.monotonic()-started)
                if remaining<=budget.shutdown_reserve_s:raise RuntimeError("Initialization consumed shared training wall allowance")
                summary=train_pilot(env,policy,folder/'training',paths=paths,seed=declaration.seed,validate=validate,
                    budget=replace(budget,maximum_wall_s=remaining))
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
        if isinstance(exc,BoundedExecutionInterruption):error['deadline_receipt']=exc.receipt
    result={**(summary or {}),**{key:getattr(declaration,key) for key in ('recipe_sha256','domain_sha256','target_geometry_sha256',
        'predicate_sha256','declared_mask_sha256','start_bank_sha256','reset_protocol','start_quantization_version','sampling_version')},
        'shared_schema':'task10b_phase2_shared_training_receipt_v1','declaration_identity':declaration.identity,
        'trained_timings_s':list(declaration.trained_timings_s),'interpolation_timing_s':declaration.interpolation_timing_s,
        'interpolation_used_for_training':False,'sampling_split':'train','training_groups':list(declaration.command_groups),
        'source_phase_rehearsal_no_cycle_credit':True,'additional_start_noise_applied':False,
        'wrapper_exception':error,'session_wall_s':time.monotonic()-started,'fresh_transition_cap':budget.transition_cap,
        'phase2_complete':False,'formal_acceptance':False}
    if error is not None or summary is None:result.update(status='failed',exception=error)
    write_json(folder/'shared_training_receipt.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',type=Path,required=True);parser.add_argument('--configuration-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);options=parser.parse_args()
    result=execute_configuration(options.configuration,options.configuration_sha256,options.output)
    print(json.dumps(result,allow_nan=False,sort_keys=True));return 0 if result['status']=='bounded_stop' else 1


if __name__=='__main__':raise SystemExit(main())
