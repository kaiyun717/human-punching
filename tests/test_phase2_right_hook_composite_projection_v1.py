"""Saved-byte history composition; no model, dynamics or reference construction."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from human_punching import phase2_right_hook_composite_projection_v1 as projection
from human_punching.wbc.provenance import sha256_file
from test_phase2_right_hook_full_projection_v1 import fixture_files, write


def binding(path):return dict(path=str(path),sha256=sha256_file(path))


def component_refresh(d):
    write(d.catalog_path,d.catalog)
    d.sidecar['manifest_sha256']=sha256_file(d.catalog_path)
    write(d.run/d.sidecar_name,d.sidecar)
    for key,path in dict(packet=d.packet_path,plan=d.plan_path,manifest=d.catalog_path,
                         execution=d.execution_path,accounting=d.terminal_path).items():
        d.audit[key+'_path']=str(path);d.audit[key+'_sha256']=sha256_file(path)
    d.audit['generation']={k:d.catalog[v] for k,v in dict(planned='planned_cells',executed='executed_cells',
        generated='generated_cells',admitted='admitted_candidate_cells',unattempted='unattempted_cells').items()}
    write(d.audit_path,d.audit)


@pytest.fixture
def data(tmp_path,monkeypatch):
    prior=fixture_files(tmp_path,monkeypatch)
    monkeypatch.setattr(projection,'ROOT',prior.root)
    # Semantic resolution is tested against the real immutable metadata below;
    # this fixture isolates packet/history/path checks using opaque byte files.
    monkeypatch.setattr(projection,'_resolve_suffix',lambda plan:None)
    original_rows=deepcopy(prior.catalog['cells'])
    timeout=deepcopy(original_rows[54])
    for key in ('reference_path','legacy_runtime_path'):timeout[key]=None
    failure=prior.run/'rh_target_54/failure.json'
    timeout.update(generation_disposition='timeout',kinematic_admission=False,
        rejection_reasons=['bounded_timeout'],details=dict(physical_failure=False,
        reason='serialization_reserve',boundary='target_screen_chunk'),
        generation_failure_path=str(prior.compute/failure.relative_to(prior.root)))
    write(failure,dict(prior_timeout=True));timeout['generation_failure_sha256']=sha256_file(failure)
    prior.catalog.update(cells=original_rows[:54]+[timeout],complete=False,executed_cells=55,
        generated_cells=54,admitted_candidate_cells=54,unattempted_cells=prior.plan['cells'][55:],
        exception=dict(type='BoundedExecutionInterruption',message='serialization_reserve at target_screen_chunk'))
    prior.execution.update(status='failed',returncode=1);write(prior.execution_path,prior.execution)
    prior.terminal.update(state='FAILED',exit_code='2:0');write(prior.terminal_path,prior.terminal)
    prior.sidecar_name='right_hook_full_entry.json'
    component_refresh(prior)
    root=prior.root;folder=root/'outputs/task10b/suffix';run=folder/'run';run.mkdir(parents=True)
    plan=deepcopy(prior.plan)
    paths=dict(prior_plan=prior.plan_path,prior_manifest=prior.catalog_path,prior_packet=prior.packet_path,
        prior_execution=prior.execution_path,prior_accounting=prior.terminal_path,prior_audit=prior.audit_path)
    plan.update(schema='task10b_phase2_right_hook_suffix_catalog_plan_v2',maximum_cells=10,
        maximum_child_seconds=90,output_byte_cap=64*1024**2,cells=deepcopy(prior.plan['cells'][54:]),
        **{k:binding(p) for k,p in paths.items()})
    for name in (projection.SUFFIX_WRAPPER,'scripts/task10b_phase2_right_hook_suffix_v2.py',projection.SUFFIX_AUDITOR):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# synthetic suffix source\n')
    plan_path=root/'suffix_plan.json';write(plan_path,plan)
    rows=[]
    for old in original_rows[54:]:
        row=deepcopy(old);cell=run/row['candidate_id'];cell.mkdir()
        for field,filename in [('reference_path','reference.npz'),('legacy_runtime_path','reference.npz'),('diagnostic_screen_path','screen.json')]:
            path=cell/filename
            if not path.exists():path.write_bytes(b'unchanged saved synthetic bytes')
            row[field]=str(prior.compute/path.relative_to(root))
            if field!='diagnostic_screen_path':row[field[:-5]+'_sha256']=sha256_file(path)
        row['kinematic_admission']=True;row['rejection_reasons']=[]
        rows.append(row)
    files=dict(prior.packet['files'])
    for path in [plan_path,*paths.values(),root/projection.SUFFIX_WRAPPER,
                 root/'scripts/task10b_phase2_right_hook_suffix_v2.py',root/projection.SUFFIX_AUDITOR]:
        files[str(path.relative_to(root))]=sha256_file(path)
    archive=folder/'source_snapshot'
    for name in files:
        destination=archive/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,destination)
    packet=deepcopy(prior.packet);packet.update(schema=projection.SUFFIX_SCHEMA,
        mode='right-hook-suffix-target-references-v2',allocation_wall_seconds=180,application_wall_seconds=150,
        sequential_child_wall_seconds=90,output=str(run.relative_to(root)),files=files,inputs={'plan':str(plan_path)})
    packet_path=folder/'packet.json';write(packet_path,packet)
    execution=deepcopy(prior.execution);execution.update(status='executed',returncode=0,job_id='124',
        packet_sha256=sha256_file(packet_path),native_command=[str(prior.compute/'.venv-task10b/bin/python'),
        '-B','-m','human_punching.phase2_right_hook_suffix_catalog_v2','--plan',str(prior.compute/'suffix_plan.json'),
        '--output',str(prior.compute/run.relative_to(root)),'--wall-seconds','90.0'])
    execution_path=folder/'execution.json';write(execution_path,execution)
    terminal=deepcopy(prior.terminal);terminal.update(state='COMPLETED',exit_code='0:0',job_id='124',packet_sha256=sha256_file(packet_path))
    terminal_path=folder/'terminal_accounting.json';write(terminal_path,terminal)
    inputs=dict(prior.catalog['input_hashes'],**{k:v['sha256'] for k,v in plan.items() if k in projection.PRIOR_KEYS})
    catalog=dict(schema=projection.CATALOG,path_solver_version=projection.SOLVER_VERSION,output_byte_cap=64*1024**2,
        complete=True,exception=None,planned_cells=10,executed_cells=10,cells=rows,generated_cells=10,
        admitted_candidate_cells=10,unattempted_cells=[],input_hashes=inputs,plan_sha256=sha256_file(plan_path),
        source_sha256=files[projection.GENERATOR],recipe_sha256=plan['recipe']['sha256'],physics_executed=False,
        physical_qualification=False,learned_trajectory_generator=False)
    catalog_path=run/'manifest.json';write(catalog_path,catalog)
    sidecar=dict(schema='task10b_phase2_right_hook_suffix_catalog_v2',plan_sha256=sha256_file(plan_path),
        wrapper_source_sha256=files[projection.SUFFIX_WRAPPER],frozen_generator_source_sha256=files[projection.GENERATOR],
        manifest_sha256=sha256_file(catalog_path),family='right_hook',split='train',child_wall_seconds=90,
        engineering_cap_rad=.205,solver_changed=False,physical_qualification=False,native_frozen_control_required_before_learning=True,
        ancestry={k:plan[k] for k in projection.PRIOR_KEYS},recipe_sha256=plan['recipe']['sha256'],
        retained_prefix_ids=[r['candidate_id'] for r in prior.plan['cells'][:54]],
        regenerated_original_ids=[r['candidate_id'] for r in plan['cells']],prior_outcomes_rewritten=False,automatic_further_retry=False)
    audit_path=root/'suffix_audit.json'
    audit=dict(schema='task10b_phase2_rh_suffix_saved_target_reference_audit_v2',passed=True,family='right_hook',active_hand_index=1,
        split='train',job_id='124',physics_steps=0,project_reference_generation=False,reference_generation=False,
        source_sha256=sha256_file(root/projection.SUFFIX_AUDITOR),cells=[dict(candidate_id=r['candidate_id'],all_checks_passed=True) for r in rows],
        historical_outcomes_preserved=True,prior_inputs={k:plan[k] for k in projection.PRIOR_KEYS},
        original_plan_suffix_start=54,retained_prefix_ids=sidecar['retained_prefix_ids'])
    d=SimpleNamespace(root=root,compute=prior.compute,folder=folder,run=run,plan=plan,plan_path=plan_path,
        packet=packet,packet_path=packet_path,execution=execution,execution_path=execution_path,
        terminal=terminal,terminal_path=terminal_path,catalog=catalog,catalog_path=catalog_path,
        audit=audit,audit_path=audit_path,sidecar=sidecar,sidecar_name='right_hook_suffix_entry_v2.json',
        prior=prior,probe_path=prior.probe_path,archive=archive)
    component_refresh(d)
    return d


def build(d):return projection.build_projection(d.catalog_path,d.execution_path,d.probe_path,saved_audit_path=d.audit_path)


def test_exact54_plus10_preserves_original_failure_and_every_reference_byte(data):
    before={p:sha256_file(p) for p in data.root.rglob('*') if p.is_file()}
    view,receipt=build(data)
    assert len(view['cells'])==64 and view['generated_cells']==64 and view['aggregate_native_attempts']==65
    assert view['complete'] is True and 'executed_cells' not in view
    assert [r['candidate_id'] for r in view['cells']]==[r['candidate_id'] for r in data.prior.plan['cells']]
    history=view['construction_history']
    assert history['prior']['scheduler_state']=='FAILED' and history['prior']['executed']==55
    assert history['prior']['timeout_row']['generation_disposition']=='timeout'
    assert history['prior']['unattempted_cells']==data.prior.plan['cells'][55:]
    assert history['original_outcomes_rewritten'] is False
    for name,original in [('prior',data.prior.catalog),('suffix',data.catalog)]:
        rows=view['cells'][:54] if name=='prior' else view['cells'][54:]
        for i,row in enumerate(rows):
            restored=deepcopy(row)
            for change in receipt['components'][name]['changes']:
                _,index,key=change['json_location'].split('/')
                if int(index)==i:restored[key]=change['old']
            assert restored==original['cells'][i]
    assert all(sha256_file(p)==h for p,h in before.items())


def test_roundtrip_closes_both_archives_and_nested_original_inputs(data,monkeypatch):
    nested=data.root/'nested.json';write(nested,{'scope':'exact full and suffix dependencies'})
    calls=[]
    def semantic(path,kind):
        calls.append(kind);return {'nested.json':sha256_file(nested)}
    monkeypatch.setattr(projection,'_semantic_files',semantic)
    projection.project(data.catalog_path,data.execution_path,data.probe_path,data.root/'view',saved_audit_path=data.audit_path)
    path=data.root/'view/manifest.json';view,receipt=projection.verify_projection(path);closure=projection.files(path)
    assert calls==['prior','suffix'] and closure['nested.json']==sha256_file(nested)
    for d in (data.prior,data):
        for name,digest in d.packet['files'].items():assert closure[str((d.archive/name).relative_to(data.root))]==digest
    assert receipt['historical_timeout_and_unattempted_preserved'] is True
    with pytest.raises(ValueError,match='Fresh owned'):
        projection.project(data.catalog_path,data.execution_path,data.probe_path,data.root/'view',saved_audit_path=data.audit_path)


@pytest.mark.parametrize('change',['prefix_row','timeout_hidden','history_state','drop_unattempted','swap_suffix','remove_origin','claim_qualified'])
def test_composite_tampering_rejected(data,change):
    projection.project(data.catalog_path,data.execution_path,data.probe_path,data.root/'view',saved_audit_path=data.audit_path)
    path=data.root/'view/manifest.json';view=json.loads(path.read_text())
    if change=='prefix_row':view['cells'][0]['target_w'][2]+=.001
    elif change=='timeout_hidden':view['construction_history']['prior']['timeout_row']['generation_disposition']='generated'
    elif change=='history_state':view['construction_history']['prior']['scheduler_state']='COMPLETED'
    elif change=='drop_unattempted':view['construction_history']['prior']['unattempted_cells'].pop()
    elif change=='swap_suffix':view['cells'][54],view['cells'][55]=view['cells'][55],view['cells'][54]
    elif change=='remove_origin':view['right_hook_composite_catalog_projection']['row_origins'].pop()
    else:view['physical_qualification']=True
    write(path,view)
    with pytest.raises(ValueError):projection.validate(path)


@pytest.mark.parametrize('change',['old_suffix_schema','late_child','wrong_job','zero_gpu','audit_wrong_prior','audit_no_history','sidecar_wrong_ids','archive','reference','nonce','duplicate'])
def test_native_component_evidence_is_mandatory(data,change):
    if change=='old_suffix_schema':data.packet['schema']='task10b_phase2_right_hook_suffix_packet_v1';write(data.packet_path,data.packet)
    elif change=='late_child':data.execution['native_command'][-1]='89.0';write(data.execution_path,data.execution)
    elif change=='wrong_job':data.terminal['job_id']='999';write(data.terminal_path,data.terminal)
    elif change=='zero_gpu':data.terminal['allocated_tres']['gres/gpu']='0';write(data.terminal_path,data.terminal)
    elif change=='audit_wrong_prior':data.audit['prior_inputs']['prior_manifest']['sha256']='a'*64;write(data.audit_path,data.audit)
    elif change=='audit_no_history':data.audit['historical_outcomes_preserved']=False;write(data.audit_path,data.audit)
    elif change=='sidecar_wrong_ids':data.sidecar['retained_prefix_ids'].pop();write(data.run/data.sidecar_name,data.sidecar)
    elif change=='archive':(data.prior.archive/projection.GENERATOR).write_text('changed')
    elif change=='reference':Path(data.prior.catalog['cells'][0]['reference_path']).write_bytes(b'changed')
    elif change=='nonce':write(data.prior.nonce_path,{'different':'nonce'})
    elif change=='duplicate':data.catalog['cells'][1]=deepcopy(data.catalog['cells'][0]);component_refresh(data)
    with pytest.raises(ValueError):build(data)


def test_suffix_geometry_failure_retained_as_negative_not_repaired(data):
    row=data.catalog['cells'][0]
    row.update(reference_path=None,legacy_runtime_path=None,generation_disposition='failed',kinematic_admission=False,
        rejection_reasons=['bounded_geometry_failure'])
    failure=data.run/row['candidate_id']/'failed.json';write(failure,{'failed':True})
    row.update(generation_failure_path=str(data.compute/failure.relative_to(data.root)),generation_failure_sha256=sha256_file(failure))
    data.catalog.update(generated_cells=9,admitted_candidate_cells=9)
    component_refresh(data)
    view,_=build(data)
    assert view['complete'] is True and view['generated_cells']==63 and view['admitted_candidate_cells']==63
    assert view['cells'][54]['generation_disposition']=='failed'
    assert view['construction_history']['prior']['timeout_row']['generation_disposition']=='timeout'


def test_real_saved_prior_component_and_corrected_suffix_metadata_only():
    from human_punching.wbc.provenance import PROJECT_ROOT
    from human_punching.phase2_right_hook_suffix_catalog_v2 import resolve_cells
    p=PROJECT_ROOT/'outputs/task10b/p2rhsuffixprep02/plan.json';plan=json.loads(p.read_text())
    assert len(resolve_cells(plan)[1])==10
    prior,receipt=projection._component(plan['prior_manifest']['path'],plan['prior_execution']['path'],
        PROJECT_ROOT/'outputs/task10b/p2hostalias01/probe.json',saved_audit_path=plan['prior_audit']['path'],kind='prior')
    assert prior['executed_cells']==55 and prior['generated_cells']==54
    assert prior['cells'][54]['generation_disposition']=='timeout' and len(prior['unattempted_cells'])==9
    assert receipt['packet_schema']==projection.FULL_SCHEMA
