"""Synthetic timing-pilot admission checks; no learned route or native execution."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import human_punching.phase2_timing_pilot as m
import human_punching.phase2_interiors as ownership
from human_punching import phase2_learning as original,phase2_d0_clock_v2 as clock


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


def archive(path,ttc=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=np.arange(401)*.005
    data=dict(time_s=t,t=t,qpos=np.zeros((len(t),55)),qvel=np.zeros((len(t),54)))
    if ttc is not None:data['requested_ttc_s']=ttc
    np.savez(path,**data)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'human-punching';root.mkdir();monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(ownership,'ROOT',root)
    for name in m.SOURCES:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# synthetic '+name)
    release=root/'outputs/four-punch-milestone';family='cross';artifacts={}
    for name,relative in m.RELEASE_FILES.items():
        p=release/family/relative;p.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix=='.npz':archive(p)
        else:p.write_text('{}')
        artifacts[family+'/'+relative]={'sha256':m.sha(p)}
    dump(release/'manifest.json',dict(artifacts=artifacts))
    predicate=root/'outputs/predicate.json';dump(predicate,dict(thresholds={'version':'task10b_functional_guard_v1'}))
    rows=[]
    source=release/family/'reference.npz';model=release/family/'model/human.xml'
    common=dict(family=family,source_reference_sha256=m.sha(source),source_model_path=str(model),source_model_sha256=m.sha(model),
        kinematic_admission=True,rejection_reasons=[],split='d0_development',launch_tick=40,outward_end_tick=300,
        duration_s=2.,episode_duration_s=3.,phase_denominator_s=3.,immutable_annotation='retained')
    rows.append(dict(**common,candidate_id='cross_original',kind='original_control',requested_ttc_s=None,
        reference_path=str(source),reference_sha256=m.sha(source),legacy_runtime_path=str(source),legacy_runtime_sha256=m.sha(source)))
    for t in m.BINS[family]:
        ref=root/f'outputs/ref{t}.npz';archive(ref,t)
        rows.append(dict(**common,candidate_id='cross_uniform_'+f'{t:.3f}'.replace('.','p'),kind='uniform',requested_ttc_s=t,
            reference_path=str(ref),reference_sha256=m.sha(ref),legacy_runtime_path=str(ref),legacy_runtime_sha256=m.sha(ref)))
    catalog=root/'outputs/catalog.json';dump(catalog,dict(schema=m.CATALOG,complete=True,predicate_sha256=m.sha(predicate),cells=rows))
    video=root/'outputs/video_review.json';dump(video,dict(schema=m.VIDEO_SCHEMA,status='reviewed',predeclared_selection=True,
        families=[family],all_flagged_failures_reviewed=True,movement_exploit_detected=False))
    route=root/'outputs/route_review.json';dump(route,dict(schema=m.ROUTE_SCHEMA,status='reviewed',scope='d0_development',
        catalog_sha256=m.sha(catalog),video_review_sha256=m.sha(video),final_test_used=False,final_recipe_domain_frozen=False,
        families={family:dict(training_candidate_ids=[r['candidate_id'] for r in rows[1:3]],recipe_version='synthetic_recipe_v1',
            selection_reason='Synthetic-only validation fixture',authorize_transferred_pilot=True,explicit_nonuniform_training_ids=[])}))
    plan=m.create_plan(catalog,release,predicate,route,video,family)
    return root,plan


def rebind_catalog(plan,catalog):
    path=Path(plan['catalog']['path']);dump(path,catalog);plan['catalog']=m.binding(path)
    routepath=Path(plan['route_receipt']['path']);route=m.read(routepath);route['catalog_sha256']=m.sha(path);dump(routepath,route)
    plan['route_receipt']=m.binding(routepath)
    for row in catalog['cells']:
        if row['candidate_id'] in plan['selected_row_sha256']:plan['selected_row_sha256'][row['candidate_id']]=m.fingerprint(row)


def test_training_and_validation_engines_are_original_functions_not_copies():
    assert m.train_pilot is original.train_pilot and m.fresh_policy is original.fresh_policy
    assert m.PilotBudget is original.PilotBudget and m.NativeSpecialistEnv is original.NativeSpecialistEnv
    assert m.run_cell_panel is clock.run_cell_panel


def test_exact_budget_original_validation_only_and_no_actual_plan_created(fixture):
    root,plan=fixture;paths,rows,budget=m.validate_plan(plan)
    assert budget.environments==128 and budget.rollout_steps==64 and budget.maximum_updates==16 and budget.transition_cap==131072
    assert plan['validation_updates']==[0,8,16] and plan['planned_validation_trials']==72
    assert rows[0]['kind']=='original_control' and all(r['kind']!='original_control' for r in rows[1:])
    assert plan['seed']==21021 and plan['validation_seed']==22021 and not plan['actual_route_freeze']
    assert not list(root.rglob('launch.json'))


@pytest.mark.parametrize('mutation',['environments','steps','updates','cap','cadence','seed','wall','stagnation','ancestor','resume','source'])
def test_caps_clocks_original_ancestry_and_source_changes_fail(fixture,mutation):
    _,plan=fixture
    if mutation=='environments':plan['budget']['environments']=256
    if mutation=='steps':plan['budget']['rollout_steps']=32
    if mutation=='updates':plan['budget']['maximum_updates']=17
    if mutation=='cap':plan['fresh_transition_cap']=2000000
    if mutation=='cadence':plan['validation_updates']=[0,16]
    if mutation=='seed':plan['seed']=21022
    if mutation=='wall':plan['application_wall_seconds']=float('nan')
    if mutation=='stagnation':plan['budget']['no_improvement_checks']=3
    if mutation=='ancestor':Path(plan['release_inputs']['checkpoint']['path']).write_text('changed weights')
    if mutation=='resume':plan['ancestry']['prior_training_resumed']=True
    if mutation=='source':Path(plan['sources'][m.SOURCES[2]]['path']).write_text('changed trainer')
    with pytest.raises(ValueError):m.validate_plan(plan)


@pytest.mark.parametrize('mutation',['interior','timing','duplicate_timing','denominator','launch','duration','family','rejection','source_reference'])
def test_selected_row_science_checks_not_only_row_hash(fixture,mutation):
    _,plan=fixture;catalog=m.read(Path(plan['catalog']['path']));row=catalog['cells'][1]
    if mutation=='interior':row['kind']='uniform_interior';row['training_eligible']=False
    if mutation=='timing':row['requested_ttc_s']=.675
    if mutation=='duplicate_timing':row['requested_ttc_s']=catalog['cells'][2]['requested_ttc_s']
    if mutation=='denominator':row['phase_denominator_s']=2.
    if mutation=='launch':row['launch_tick']=41
    if mutation=='duration':row.update(duration_s=2.5,episode_duration_s=3.5,phase_denominator_s=3.5)
    if mutation=='family':row['family']='jab'
    if mutation=='rejection':row['kinematic_admission']=False;row['rejection_reasons']=['unsupported']
    if mutation=='source_reference':row['source_reference_sha256']='0'*64
    rebind_catalog(plan,catalog)
    with pytest.raises(ValueError):m.validate_plan(plan)


def test_nonuniform_false_marker_needs_exact_new_authorization_and_is_preserved(fixture):
    _,plan=fixture;catalog=m.read(Path(plan['catalog']['path']));row=catalog['cells'][1]
    row.update(kind='nonuniform_contact_anchored',training_eligible=False,timing_role='bounded_frozen_policy_route_comparison')
    rebind_catalog(plan,catalog)
    with pytest.raises(ValueError,match='exact new per-row'):m.validate_plan(plan)
    routepath=Path(plan['route_receipt']['path']);route=m.read(routepath)
    route['families']['cross']['explicit_nonuniform_training_ids']=[row['candidate_id']];dump(routepath,route);plan['route_receipt']=m.binding(routepath)
    before=Path(plan['catalog']['path']).read_bytes();_,rows,_=m.validate_plan(plan)
    assert rows[1]['training_eligible'] is False and Path(plan['catalog']['path']).read_bytes()==before
    row['training_eligible']=True;rebind_catalog(plan,catalog)
    with pytest.raises(ValueError,match='Do not rewrite'):m.validate_plan(plan)


@pytest.mark.parametrize('mutation',['not_reviewed','final_test','missing_video','exploit','unreviewed_failure','wrong_catalog','missing_recipe'])
def test_incomplete_or_wrong_review_cannot_open_training(fixture,mutation):
    _,plan=fixture;routepath=Path(plan['route_receipt']['path']);route=m.read(routepath)
    videopath=Path(plan['video_receipt']['path']);video=m.read(videopath)
    if mutation=='not_reviewed':route['status']='proposed'
    if mutation=='final_test':route['final_test_used']=True
    if mutation=='missing_video':video['predeclared_selection']=False
    if mutation=='exploit':video['movement_exploit_detected']=True
    if mutation=='unreviewed_failure':video['all_flagged_failures_reviewed']=False
    if mutation=='wrong_catalog':route['catalog_sha256']='0'*64
    if mutation=='missing_recipe':route['families']['cross']['recipe_version']=''
    dump(videopath,video);plan['video_receipt']=m.binding(videopath);route['video_review_sha256']=m.sha(videopath)
    dump(routepath,route);plan['route_receipt']=m.binding(routepath)
    with pytest.raises(ValueError):m.validate_plan(plan)


def test_three_finite_bins_allowed_but_original_is_never_training(fixture):
    _,plan=fixture;routepath=Path(plan['route_receipt']['path']);route=m.read(routepath)
    catalog=m.read(Path(plan['catalog']['path']));selected=[r['candidate_id'] for r in catalog['cells'][1:]]
    route['families']['cross']['training_candidate_ids']=selected;dump(routepath,route)
    plan=m.create_plan(plan['catalog']['path'],Path(plan['release']['path']).parent,plan['predicate']['path'],
        routepath,plan['video_receipt']['path'],'cross')
    assert plan['planned_validation_trials']==96 and len(plan['training_candidate_ids'])==3
    route['families']['cross']['training_candidate_ids']=['cross_original',selected[0]];dump(routepath,route)
    with pytest.raises(ValueError):m.create_plan(plan['catalog']['path'],Path(plan['release']['path']).parent,plan['predicate']['path'],
        routepath,plan['video_receipt']['path'],'cross')


def test_owned_paths_and_finite_memory_preflight(fixture,tmp_path):
    root,plan=fixture;plan['resident_reference_byte_cap']=1
    with pytest.raises(ValueError):m.validate_plan(plan)
    plan['resident_reference_byte_cap']=m.RESIDENT_BYTES
    outside=tmp_path/'external';outside.write_text('{}');alias=root/'escape';alias.symlink_to(outside)
    plan['predicate']={'path':str(alias),'sha256':m.sha(outside)}
    with pytest.raises(ValueError,match='inside human-punching'):m.validate_plan(plan)


def test_allocation_guard_precedes_output_or_native_construction(fixture,monkeypatch):
    root,plan=fixture;path=root/'outputs/plan.json';dump(path,plan)
    def refuse():raise RuntimeError('counted allocation required')
    monkeypatch.setattr(m,'require_allocation',refuse)
    with pytest.raises(RuntimeError,match='allocation'):m.execute(path,root/'outputs/task10b/uncreated')
    assert not (root/'outputs/task10b/uncreated').exists()


def test_launch_metadata_supplies_independent_auditor_and_renderer_inputs_without_relabeling(fixture):
    root,plan=fixture;path=root/'outputs/committed_plan.json';dump(path,plan)
    progress={str(u):dict(disposition='unattempted',planned_trials=plan['planned_trials_per_panel']) for u in (0,8,16)}
    runtime={'synthetic':True};clock={'schema':'task10b_phase2_d0_postprocessing_clock_v2'}
    launch=m.launch_receipt(path,plan,runtime=runtime,clock=clock,slurm_job_id='synthetic_no_allocation',progress=progress)
    assert launch['schema']==m.VERSION and launch['mode']=='timing_pilot'
    assert launch['catalog']==plan['catalog']['path'] and launch['catalog_sha256']==m.sha(Path(launch['catalog']))
    assert launch['family']=='cross' and launch['release_inputs']==plan['release_inputs']
    assert launch['planned_cells']==plan['validation_candidate_ids'] and launch['diagnostic_bank_ids']==list(range(8))
    assert launch['training_candidate_ids']==plan['training_candidate_ids'] and 'cross_original' not in launch['training_candidate_ids']
    assert launch['postprocessing_clock_adapter']==clock and launch['plan']==m.binding(path)
    for source,digest in launch['sources'].items():assert m.sha(Path(source))==digest
    progress['0']['disposition']='modified_after_receipt'
    plan['release_inputs']['checkpoint']['sha256']='0'*64
    assert launch['planned_validation']['0']['disposition']=='unattempted'
    assert launch['release_inputs']['checkpoint']['sha256']!='0'*64


def test_launch_metadata_rejects_unbound_plan_or_missing_job_identity(fixture):
    root,plan=fixture;path=root/'outputs/committed_plan.json';dump(path,plan)
    kwargs=dict(runtime={},clock={},slurm_job_id='synthetic_no_allocation',progress={})
    changed=deepcopy(plan);changed['family']='jab'
    with pytest.raises(ValueError,match='committed'):m.launch_receipt(path,changed,**kwargs)
    with pytest.raises(ValueError,match='job identity'):m.launch_receipt(path,plan,**{**kwargs,'slurm_job_id':''})


@pytest.mark.parametrize('parity',[None,{},dict(parameters_equal=False,actions_equal=True,maximum_action_error=0.),
    dict(parameters_equal=True,actions_equal=False,maximum_action_error=.01),
    dict(parameters_equal=True,actions_equal=True,maximum_action_error=float('nan'))])
def test_final_wall_stop_checkpoint_cannot_receive_credit_with_missing_or_failed_parity(parity):
    with pytest.raises(RuntimeError,match='Final checkpoint'):m.require_final_checkpoint_parity({'reload_parity':parity})
    m.require_final_checkpoint_parity({'reload_parity':dict(parameters_equal=True,actions_equal=True,maximum_action_error=0.)})
