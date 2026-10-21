"""Synthetic binding/ledger tests; no project reference generation or physics."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import human_punching.phase2_interiors as m


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'human-punching';root.mkdir();monkeypatch.setattr(m,'ROOT',root)
    release=root/'outputs/four-punch-milestone';bundle=root/'outputs/task10/release_baseline_02'
    dump(release/'manifest.json',{'test':'accepted-source-synthetic'})
    dump(bundle/'manifest.json',{'test':'bundle-synthetic'})
    for name in m.SOURCES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# synthetic binding '+name)
    for family in m.INTERIORS:
        for name in m.ANCESTRY:
            path=release/family/name;path.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix=='.npz':np.savez(path,time_s=np.arange(3)*.005,qpos=np.zeros((3,55)),qvel=np.zeros((3,54)))
            else:path.write_text('{}')
    predicate=root/'outputs/task10b/predicate.json';dump(predicate,{'thresholds':{'version':'task10b_functional_guard_v1'}})
    domain=root/'outputs/task10b/domain.json';dump(domain,dict(schema='Task10BPhase2Domain/v1',candidate_timings_s=m.BINS,
        provisional_final_timings_s={f:[m.BINS[f][0],t,m.BINS[f][1]] for f,t in m.INTERIORS.items()}))
    rows=[]
    for family,ttcs in m.BINS.items():
        source=release/family/'reference.npz';model=release/family/'model/human.xml'
        common=dict(family=family,source_reference_sha256=m.sha(source),source_model_path=str(model),source_model_sha256=m.sha(model),
            reference_path=str(source),reference_sha256=m.sha(source),legacy_runtime_path=str(source),legacy_runtime_sha256=m.sha(source),
            kinematic_admission=True,rejection_reasons=[],retained_arbitrary_parent_annotation={'preserve':True})
        rows.append(dict(**common,candidate_id=family+'_original',kind='original_control',requested_ttc_s=None))
        rows.extend(dict(**common,candidate_id=family+'_uniform_'+f'{ttc:.3f}'.replace('.','p'),kind='uniform',requested_ttc_s=ttc) for ttc in ttcs)
    catalog=root/'outputs/task10b/catalog.json';dump(catalog,dict(schema=m.CATALOG,complete=True,cells=rows,predicate_sha256=m.sha(predicate)))
    plan=m.create_plan(catalog,domain,predicate,release,bundle)
    return root,plan


def test_factory_only_uses_four_already_declared_interiors(fixture):
    root,plan=fixture
    assert [(r['family'],r['requested_ttc_s']) for r in plan['cells']]==list(m.INTERIORS.items())
    assert all(not r['training_eligible'] and r['timing_role']=='prespecified_untrained_interior' for r in plan['cells'])
    assert len(plan['source_ancestry'])==4 and len(plan['generator_sources'])==len(m.SOURCES)
    assert not list(root.rglob('attempt_started.json'))


@pytest.mark.parametrize('mutation',['ttc','duplicate','role','source','source_ancestry','predicate','domain','parent'])
def test_hash_or_scientific_request_mutation_is_rejected(fixture,mutation):
    root,plan=fixture;bad=deepcopy(plan)
    if mutation=='ttc':bad['cells'][0]['requested_ttc_s']=.68
    if mutation=='duplicate':bad['cells'][1]=deepcopy(bad['cells'][0])
    if mutation=='role':bad['cells'][0]['training_eligible']=True
    if mutation=='source':Path(bad['generator_sources'][m.SOURCES[0]]['path']).write_text('changed code')
    if mutation=='source_ancestry':Path(bad['source_ancestry']['jab']['checkpoint.pt']['path']).write_text('changed weights')
    if mutation in ('predicate','domain','parent'):
        key='parent_catalog' if mutation=='parent' else mutation
        Path(bad[key]['path']).write_text('{}')
    with pytest.raises(ValueError):m.validate_plan(bad)


def test_declared_middle_cannot_be_rebound_to_nearest_existing_bin(fixture):
    _,plan=fixture;domain=Path(plan['domain']['path']);data=json.loads(domain.read_text())
    data['provisional_final_timings_s']['jab'][1]=.75;dump(domain,data);plan['domain']=m.binding(domain)
    with pytest.raises(ValueError,match='no substitution'):m.validate_plan(plan)


@pytest.mark.parametrize('key,value',[('application_wall_seconds',float('nan')),('application_wall_seconds',841.),
    ('cell_reference_byte_cap',m.CELL_BYTES+1),('reference_byte_cap',m.TOTAL_BYTES+1),
    ('cell_product_byte_cap',0),('resident_byte_cap',16),('maximum_search_trials',19),('screen_chunk_frames',17)])
def test_finite_execution_and_memory_bounds(fixture,key,value):
    _,plan=fixture;plan[key]=value
    with pytest.raises(ValueError):m.validate_plan(plan)


def test_input_symlink_cannot_escape_owned_root(fixture,tmp_path):
    root,plan=fixture;outside=tmp_path/'external.json';outside.write_text('{}')
    alias=root/'escape.json';alias.symlink_to(outside)
    plan['domain']={'path':str(alias),'sha256':m.sha(outside)}
    with pytest.raises(ValueError,match='inside human-punching'):m.validate_plan(plan)


def test_serialized_paths_keep_prepared_root_across_owned_runtime_alias(fixture,tmp_path,monkeypatch):
    root,plan=fixture;prepared=tmp_path/'prepared-human';prepared.symlink_to(root,target_is_directory=True)
    plan['logical_project_root']=str(prepared)
    # Simulate preparation and execution importing the same project through two
    # owned names. Serialized references remain in the preparation namespace.
    relative=Path('outputs/task10b/future/interior/reference.npz')
    monkeypatch.setattr(m,'ROOT',prepared)
    from_preparation=m.logical_path(plan,relative)
    monkeypatch.setattr(m,'ROOT',root)
    from_runtime=m.logical_path(plan,root/relative)
    assert from_preparation==from_runtime==str(prepared/relative)
    m.validate_plan(plan)
    plan['logical_project_root']=str(tmp_path)
    with pytest.raises(ValueError,match='logical root'):m.validate_plan(plan)


def test_relative_bindings_are_owned_project_relative_not_cwd(fixture,tmp_path,monkeypatch):
    root,plan=fixture;record=deepcopy(plan['domain']);record['path']=str(Path(record['path']).relative_to(root))
    monkeypatch.chdir(tmp_path)
    assert m.bound(record)==root/record['path']


def test_total_payload_reserves_all_four_interiors_before_generation(fixture):
    _,plan=fixture;plan['reference_byte_cap']=4*plan['cell_reference_byte_cap']
    with pytest.raises(ValueError,match='reservation exceeds'):m.validate_plan(plan)


def test_parent_rows_cannot_be_duplicated(fixture):
    _,plan=fixture;path=Path(plan['parent_catalog']['path']);parent=json.loads(path.read_text());parent['cells'][1]=parent['cells'][0]
    dump(path,parent);plan['parent_catalog']=m.binding(path)
    with pytest.raises(ValueError,match='substituted, duplicated'):m.validate_plan(plan)


def test_parent_control_cannot_be_relabeled_as_an_interior(fixture):
    _,plan=fixture;path=Path(plan['parent_catalog']['path']);parent=json.loads(path.read_text());parent['cells'][0]['kind']='uniform_interior'
    dump(path,parent);plan['parent_catalog']=m.binding(path)
    with pytest.raises(ValueError,match='relabeled'):m.validate_plan(plan)


def configure_mock_generation(fixture,monkeypatch,behavior=None):
    root,plan=fixture;path=root/'outputs/task10b/plan.json';dump(path,plan)
    monkeypatch.setattr(m,'require_allocation',lambda:None)
    monkeypatch.setattr(m,'_native_context',lambda paths:None)
    def fake(plan,entry,paths,inputs,folder,checkpoint):
        checkpoint('synthetic-only')
        if behavior is not None:behavior(entry,folder)
        return dict(**entry,kind='uniform_interior',split='d0_development',kinematic_admission=True,
            generation_disposition='generated',reference_uncompressed_array_bytes=10,physical_qualification=False,fine_tuned=False)
    monkeypatch.setattr(m,'_generate_cell',fake)
    return root,plan,path


def test_complete_catalog_retains_all_parent_rows_without_mutation(fixture,monkeypatch):
    root,plan,path=configure_mock_generation(fixture,monkeypatch)
    parent=Path(plan['parent_catalog']['path']);before=parent.read_bytes()
    result=m.generate(path,root/'outputs/task10b/generated',wall_seconds=300.)
    assert result['complete'] and result['executed_new_cells']==4 and len(result['cells'])==20
    assert result['cells'][:16]==json.loads(before)['cells'] and parent.read_bytes()==before
    assert all(row['generation_source_sha256']==plan['generator_sources'][m.SOURCES[0]]['sha256']
        and row['parent_catalog_sha256']==plan['parent_catalog']['sha256'] for row in result['cells'][16:])
    assert not result['physics_executed'] and not result['final_test_executed']
    assert result['unattempted_cells']==[]
    with pytest.raises(FileExistsError):m.generate(path,root/'outputs/task10b/generated')


def test_scientific_rejection_is_preserved_and_other_families_remain_attemptable(fixture,monkeypatch):
    def fail(entry,folder):
        if entry['family']=='cross':raise ValueError('fixed source path has no contact')
    root,plan,path=configure_mock_generation(fixture,monkeypatch,fail)
    result=m.generate(path,root/'outputs/task10b/negative')
    assert result['complete'] and result['executed_new_cells']==4 and result['exception'] is None
    failed=result['cells'][17];assert failed['generation_disposition']=='failed' and not failed['kinematic_admission']
    assert 'no contact' in failed['error']['message']
    assert result['cells'][18]['generation_disposition']=='generated'


def test_timeout_retains_partial_artifact_and_every_unattempted_cell(fixture,monkeypatch):
    def stop(entry,folder):
        (folder/'reference.npz').write_bytes(b'partial synthetic output')
        raise m.BoundedExecutionInterruption(dict(reason='test_watchdog',boundary='synthetic'))
    root,plan,path=configure_mock_generation(fixture,monkeypatch,stop)
    result=m.generate(path,root/'outputs/task10b/partial')
    assert not result['complete'] and result['executed_new_cells']==1 and len(result['unattempted_cells'])==3
    failed=result['cells'][16];assert failed['generation_disposition']=='timeout'
    assert Path(failed['retained_partial_reference_path']).read_bytes()==b'partial synthetic output'
    assert result['reference_bytes_measurement_is_lower_bound'] and len(result['unknown_partial_reference_payloads'])==1
    assert not failed['kinematic_admission'] and result['exception']
    started=json.loads((root/'outputs/task10b/partial/progress_001.json').read_text())
    assert started['cells'][0]['generation_disposition']=='generation_started'
    assert 'unknown generation extent' in started['interruption_semantics']


def test_constructor_failure_is_a_failed_attempt_and_not_unattempted(fixture,monkeypatch):
    root,plan,path=configure_mock_generation(fixture,monkeypatch)
    def fail(_):raise RuntimeError('synthetic constructor failure')
    monkeypatch.setattr(m,'_native_context',fail)
    result=m.generate(path,root/'outputs/task10b/constructor_failure')
    assert result['executed_new_cells']==1 and len(result['unattempted_cells'])==3
    assert result['cells'][16]['generation_disposition']=='failed'


def test_execution_wall_cannot_silently_change(fixture,monkeypatch):
    root,plan,path=configure_mock_generation(fixture,monkeypatch)
    with pytest.raises(ValueError,match='immutable plan'):m.generate(path,root/'outputs/task10b/not_created',wall_seconds=400.)
    assert not (root/'outputs/task10b/not_created').exists()


def test_cpu_generation_rejected_before_artifact_creation(fixture,monkeypatch):
    root,plan=fixture;path=root/'outputs/task10b/plan.json';dump(path,plan)
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='counted Slurm CUDA'):m.generate(path,root/'outputs/task10b/not_created')
    assert not (root/'outputs/task10b/not_created').exists()


def test_archive_rejects_object_nonfinite_and_payload_overflow(fixture):
    root,_=fixture
    for value in (np.array([object()],object),np.array([np.nan]),np.zeros(50)):
        with pytest.raises(ValueError):m._archive(root/'rejected.npz',{'a':value},10)
    assert not (root/'rejected.npz').exists()


def test_native_factory_reuses_exact_solver_and_original_joint_feedforward(fixture,monkeypatch):
    from types import SimpleNamespace
    import mujoco
    import human_punching.phase2_reference as reference
    import human_punching.wbc.release_baseline as baseline
    root,plan=fixture;paths,_,_=m.validate_plan(plan);entry=plan['cells'][0];folder=root/'synthetic_worker';folder.mkdir()
    model=SimpleNamespace(nq=55,nv=54,nu=48)
    monkeypatch.setattr(mujoco.MjModel,'from_xml_path',lambda path:model)
    calls=[];t=np.arange(25)*.005
    events=dict(launch_tick=4,launch_time_s=.02,outward_end_tick=16,outward_completion_time_s=.08)
    candidate=dict(t=t,time_s=t,qpos=np.zeros((25,55)),qvel=np.zeros((25,54)),qacc=np.zeros((25,54)),
        head_center=np.zeros(3),active_hand_index=0,guard_hold_start=.1,duration_scale=1.,events=events)
    contact=dict(timing_valid=True,inactive_hand_first_or_simultaneous=False,any_contact_before_launch=False)
    def solver(used_model,source,requested,maximum_trials):
        calls.append(('solve',used_model is model,requested,maximum_trials));return deepcopy(candidate),dict(selected_contact=contact)
    monkeypatch.setattr(reference,'solve_uniform_timing',solver)
    def feedforward(used_model,source,dofs):
        calls.append(('original_feedforward',used_model is model,len(dofs)));return np.full((25,48),3.)
    monkeypatch.setattr(baseline,'frozen_source',lambda release,family:(SimpleNamespace(support_feedforward=feedforward),None))
    def screen(path,chunk,caps):
        n=len(chunk['t']);calls.append(('screen',n))
        summary={key:0. for key in ('joint_limit_excursion_max_rad','selected_self_penetration_max_m','floor_penetration_max_m',
            'joint_velocity_max_rad_s','joint_acceleration_max_rad_s2','maximum_minimax_effort_ratio','maximum_base_force_residual_n','maximum_base_moment_residual_nm')}
        summary.update(frames=n,within_effort_caps_frames=0,over_effort_caps_frames=0,support_status_counts={'support_infeasible':n},physics_executed=False)
        return summary,dict(time_s=chunk['t'],joint_torque=np.full((n,48),9.),minimum_peak_effort_ratio=np.full(n,np.nan))
    monkeypatch.setattr(reference,'screen_diagnostics',screen)
    inputs={'jab':{'schema':{'dof_addresses':list(range(6,54)),'torque_limits':[[-10,10]]*48}}}
    result=m._generate_cell(plan,entry,paths,inputs,folder,lambda stage:None)
    assert calls[:2]==[('solve',True,.675,18),('original_feedforward',True,48)]
    assert [x for x in calls if x[0]=='screen']==[('screen',16),('screen',9)]
    with np.load(folder/'reference.npz',allow_pickle=False) as saved:
        assert np.all(saved['torque']==3.)  # Optimized support torque9 never enters executed reference.
    assert result['launch_tick']==4 and result['phase_denominator_s']==pytest.approx(1.12)
    assert result['training_eligible'] is False and not result['physical_qualification']
    summary=json.loads((folder/'screen.json').read_text());assert summary['support_status_counts']=={'support_infeasible':25}
