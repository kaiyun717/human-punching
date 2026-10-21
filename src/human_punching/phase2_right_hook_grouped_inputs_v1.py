"""Exact RH8 grouped inputs and completed development/packing reviews."""
from __future__ import annotations
from pathlib import Path
from . import phase2_pipeline_inputs_v2 as parent
from .phase2_right_hook_pack_inputs_v2 import Inputs
from .phase2_nonuniform_control import Inputs as AliasedInputs
from .wbc.provenance import PROJECT_ROOT,sha256_file

VERSION='task10b_phase2_right_hook_grouped_inputs_v1'
KEYS=frozenset(('configuration','packing_audit','control_numerical_review','control_visual_review'))
PACK_AUDIT='task10b_phase2_independent_saved_right_hook_runtime_pack_audit_v1'
PACK_AUDITOR='outputs/task10b/p2rhpackauditprep01/audit_saved_right_hook_pack.py'


def selected_evidence(collector,config,inputs):
    collector.bound(config['configuration_factory'])
    selection=collector.read(collector.bound(config['right_hook_selection']))
    if (selection.get('schema')!='task10b_phase2_right_hook_composite_grouped_selection_v1'
            or selection.get('family')!='right_hook' or selection.get('trained_timings_s')!=[1.,1.2]
            or selection.get('interpolation_timing_s')!=1.1
            or selection.get('historical_timeout_and_unattempted_preserved') is not True
            or selection.get('packed_catalogs',{}).get('train')!=config['catalog']
            or selection.get('packed_recipe')!=config['recipe']):
        raise ValueError('Exact RH composite selection and common packed recipe required')
    for key in ('original_selection','full_recipe','timing_selection','pilot_envelope','selection_basis',
                'suffix_plan','composite_train_catalog','full_validation_catalog','common_unpacked_recipe',
                'reviewed_envelope','packed_recipe'):
        collector.bound(selection[key])
    for key in ('full_plans','saved_audits','packed_catalogs'):
        for item in selection[key].values():collector.bound(item)
    envelope=collector.read(collector.bound(selection['reviewed_envelope']))
    if ([envelope.get(k) for k in ('allocation_s','application_s','child_budget_s','shutdown_reserve_s',
                                  'maximum_updates','fresh_transition_cap','num_envs','rollout_steps')]
            !=[960,900,720,90,8,65536,128,64]
            or envelope.get('validation_updates')!=[0,8]):
        raise ValueError('Exact finite RH8 envelope and original rollout route required')
    audit=collector.read(collector.add(inputs['packing_audit']))
    for key in ('packet','execution','terminal','source','parent_auditor','packed_recipe'):
        collector.bound(audit[key])
    train=collector.read(collector.bound(config['catalog']))
    validation=collector.read(collector.bound(selection['packed_catalogs']['validation']))
    expected=[r['candidate_id'] for c in (train,validation) for r in c['cells']]
    if (audit.get('schema')!=PACK_AUDIT or audit.get('passed') is not True
            or audit.get('packed_recipe')!=config['recipe']
            or collector.bound(audit['source'])!=(collector.root/PACK_AUDITOR).resolve(strict=True)
            or audit.get('counts')!=dict(planned=88,checked=88,train=64,validation=24)
            or audit.get('physics_steps')!=0 or audit.get('actor_calls')!=0
            or audit.get('exact_executed_float32_casts') is not True
            or audit.get('other_retained_arrays_exact') is not True
            or audit.get('source_qacc_and_screens_preserved') is not True
            or audit.get('event_metadata_preserved') is not True
            or [r.get('candidate_id') for r in audit.get('cells',[])]!=expected
            or any(r.get('passed') is not True for r in audit['cells'])):
        raise ValueError('Exact successful independently audited88 packed arrays required')
    packet_path=collector.bound(audit['packet']);packet=collector.read(packet_path)
    if (packet.get('schema')!='task10b_phase2_right_hook_pack_packet_v2'
            or Path(config['catalog']['path']).resolve(strict=True)!=packet_path.parent/'run/catalog_00.json'
            or Path(selection['packed_catalogs']['validation']['path']).resolve(strict=True)!=packet_path.parent/'run/catalog_01.json'):
        raise ValueError('Packed catalogs must come from the exact audited packing packet')
    return selection


def control_reviews(collector,inputs):
    numeric_path=collector.add(inputs['control_numerical_review']);numeric=collector.read(numeric_path)
    visual=collector.read(collector.add(inputs['control_visual_review']))
    if (numeric.get('schema')!='task10b_phase2_family_control_independent_saved_review_v2'
            or numeric.get('family')!='right_hook' or numeric.get('passed') is not True
            or numeric.get('planned_trials')!=32 or numeric.get('native_attempt_capacity')!=32
            or numeric.get('native_measurement_integrity') is not True or numeric.get('exact_native_count') is not True
            or numeric.get('frozen_actor_unchanged') is not True or numeric.get('physical_failures_preserved') is not True
            or numeric.get('independent_numeric_complete_cycles')!=8 or numeric.get('new_physics_steps')!=0
            or visual.get('schema')!='task10b_phase2_family_frozen_control_manual_review_v1'
            or visual.get('family')!='right_hook' or visual.get('required_clips')!=25
            or visual.get('manually_reviewed_clips')!=25 or visual.get('reviewed_numeric_failure_clips')!=24
            or visual.get('movement_exploit_observed') is not False
            or visual.get('recognizable_single_right_hook_in_all_sampled_clips') is not True
            or collector.bound(visual['independent_trace_audit'])!=numeric_path):
        raise ValueError('Complete measured RH frozen-control review required; failures cannot be hidden')
    # Exact reviewed two-root proof handles old native receipts without generic
    # pathname rewriting or walking their unrelated historical metadata.
    aliases=AliasedInputs(collector.root/'outputs/task10b/p2hostalias01/probe.json')
    for key in ('packet','terminal_accounting','source','control_plan','closed_launch_audit','raw_audit'):
        path=aliases.bound(numeric[key]);collector.add(path,numeric[key]['sha256'])
    for key in ('decode','media_integrity'):collector.bound(visual[key])


def files(inputs,*,root=PROJECT_ROOT,provenance_files=None):
    if set(inputs)!=KEYS:raise ValueError('Exact configuration, packing audit and two development reviews required')
    collector=Inputs(root,provenance_files)
    config_path=collector.add(inputs['configuration']);config=collector.read(config_path)
    parent.grouped_files(collector,config_path)
    selected_evidence(collector,config,inputs)
    control_reviews(collector,inputs)
    return collector.finish()
