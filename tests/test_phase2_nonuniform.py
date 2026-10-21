"""Synthetic scalar-path, provenance and interruption checks; no project generation."""
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

import human_punching.phase2_nonuniform as m
import human_punching.phase2_interiors as ownership


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


def test_pchip_preserves_postcontact_durations_and_is_c1_not_claimed_c2():
    old=np.array([0.,.2,.9,1.6,1.9,2.5]);curve,receipt=m.clock_contract(old,.85)
    new=np.asarray(receipt['final_knots_s'])
    np.testing.assert_allclose(curve(new),old,atol=1e-14)
    np.testing.assert_allclose(np.diff(new[2:]),np.diff(old[2:]),atol=1e-14)
    assert receipt['local_source_rate_min']>=.5 and receipt['local_source_rate_max']<=1.5
    assert receipt['maximum_clock_velocity_jump']<1e-12
    assert receipt['maximum_clock_acceleration_jump']>0
    # Analytic rate bound covers interval extrema, not only knots.
    sampled=curve.derivative()(np.linspace(0,new[-1],10001))
    assert sampled.min()>=receipt['local_source_rate_min']-1e-12
    assert sampled.max()<=receipt['local_source_rate_max']+1e-12


@pytest.mark.parametrize('old,new',[( [0,.2,.9,1.6,1.9,2.5],.21),([0,.2,.9,.9,1.9,2.5],.85),
    ([0,.2,.9,1.6,1.9,2.5],float('nan')),([0,.2,.9,1.6,1.9,2.5],2.8)])
def test_bad_or_out_of_envelope_clocks_rejected(old,new):
    with pytest.raises(ValueError):m.clock_contract(old,new)


@pytest.fixture
def synthetic():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
    <geom type="sphere" size=".1" mass="1"/><body><joint axis="0 0 1"/>
    <geom type="sphere" size=".1" mass="1"/></body></body></worldbody></mujoco>''')
    t=np.arange(251)*.01;q=np.tile(model.qpos0,(len(t),1));motion=np.clip((t-.2)/1.4,0,1)
    q[:,0]=.1*motion;angle=.2*motion;q[:,3]=np.cos(angle/2);q[:,6]=np.sin(angle/2);q[:,7]=.3*motion
    source=dict(time_s=t,qpos=q,launch_time=.2,return_start=1.6,guard_hold_start=1.9,
        head_center=np.zeros(3),active_hand_index=0)
    return model,source,np.array([0.,.2,.9,1.6,1.9,2.5])


def test_fresh_tangent_derivatives_complete_path_and_actual_event_semantics(synthetic):
    model,source,knots=synthetic
    # Annotation intentionally differs from the physical source's .20s onset.
    source['launch_time']=knots[1]=.213
    result,clock=m.retime_contact_anchored(model,source,knots,.852)
    t=result['t'];q=result['qpos'];v=result['qvel'];a=result['qacc']
    np.testing.assert_allclose(np.diff(t),.005,rtol=0,atol=1e-14)
    np.testing.assert_allclose(q[0],source['qpos'][0]);np.testing.assert_allclose(q[-1],source['qpos'][-1])
    independently=np.empty_like(v)
    for i in range(len(t)):
        lo=max(0,i-1);hi=min(len(t)-1,i+1)
        mujoco.mj_differentiatePos(model,independently[i],t[hi]-t[lo],q[lo],q[hi])
    np.testing.assert_allclose(v,independently,rtol=0,atol=1e-12)
    np.testing.assert_allclose(a,np.gradient(independently,t,axis=0,edge_order=2),rtol=0,atol=1e-10)
    assert np.max(abs(v[:,5]))>0
    events=result['events'];assert events['launch_tick']%4==0
    assert events['launch_time_s']!=result['launch_time'] # finite-difference motion precedes annotation
    assert result['return_start']==pytest.approx(.852+1.6-.9)
    assert events['outward_completion_time_s']>=result['return_start']
    assert events['outward_completion_time_s']-result['return_start']<.005+1e-12
    assert result['terminal_padding_s']<.005 and result['t'][-1]>=result['unrounded_duration_s']


def test_endpoint_motion_is_never_silently_zeroed(synthetic):
    model,source,knots=synthetic;source['qpos'][-1,7]+=.01
    candidate,_=m.retime_contact_anchored(model,source,knots,.85)
    assert candidate['qvel'][-1,-1]>0


def test_scalar_solve_uses_actual_launch_without_policy_lag_compensation(synthetic,monkeypatch):
    model,source,knots=synthetic
    def gloves(model,q):
        out=np.ones((len(q),2,3));out[:,0,1:]=0
        # Contact at source .9, exactly the source landmark.
        out[:,0,0]=.64-10*q[:,0]
        return out
    monkeypatch.setattr(m.reference,'glove_positions',gloves)
    candidate,search=m.solve_contact_anchored(model,source,.70)
    assert search['executed_trials']<=18 and search['solver_residual_within_1ms']
    assert not search['reference_contact_compensation_for_policy_lag']
    assert search['selected_contact']['timing_valid']
    assert abs(search['selected_contact']['timing_error_s'])<=.001
    assert search['contact_anchor_bounds_s']==pytest.approx([.86,.94])
    assert candidate['events']['launch_tick']%4==0


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'human-punching';root.mkdir();monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(ownership,'ROOT',root)
    release=root/'outputs/release';bundle=root/'outputs/bundle'
    dump(release/'manifest.json',{});dump(bundle/'manifest.json',{})
    for name in m.SOURCES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# synthetic '+name)
    for family in m.BINS:
        for name in m.ANCESTRY:
            path=release/family/name;path.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix=='.npz':np.savez(path,time_s=np.arange(3)*.005,qpos=np.zeros((3,55)),qvel=np.zeros((3,54)))
            else:path.write_text('{}')
    predicate=root/'outputs/predicate.json';dump(predicate,{'thresholds':{'version':'task10b_functional_guard_v1'}})
    domain=root/'outputs/domain.json';dump(domain,dict(schema='Task10BPhase2Domain/v1',candidate_timings_s=m.BINS))
    assessment=root/'outputs/assessment.json'
    measured={ 'failed_cell_ids':[r['parent_uniform_candidate_id'] for r in m.cell_plan()] }
    for prefix in ('cells','error_localization'):
        evidence=root/f'outputs/{prefix}.json';dump(evidence,{'scope':'synthetic-only'})
        measured.update({prefix+'_path':str(evidence),prefix+'_sha256':m.sha(evidence)})
    dump(assessment,dict(schema='task10b_phase2_uniform_nonuniform_route_assessment_v1',measured=measured))
    rows=[]
    for family,ttcs in m.BINS.items():
        source=release/family/'reference.npz';model=release/family/'model/human.xml'
        common=dict(family=family,source_reference_sha256=m.sha(source),source_model_path=str(model),source_model_sha256=m.sha(model),
            reference_path=str(source),reference_sha256=m.sha(source),legacy_runtime_path=str(source),legacy_runtime_sha256=m.sha(source),
            kinematic_admission=True,rejection_reasons=[],arbitrary_parent_annotation={'preserve':True})
        rows.append(dict(**common,candidate_id=family+'_original',kind='original_control',requested_ttc_s=None))
        rows.extend(dict(**common,candidate_id=family+'_uniform_'+f'{t:.3f}'.replace('.','p'),kind='uniform',requested_ttc_s=t) for t in ttcs)
    parent=root/'outputs/catalog.json';dump(parent,dict(schema=m.CATALOG,complete=True,cells=rows,predicate_sha256=m.sha(predicate)))
    return root,m.create_plan(parent,domain,predicate,release,bundle,assessment)


def test_exact_six_cells_and_finite_comparison_are_predeclared(fixture):
    root,plan=fixture
    assert [(r['family'],r['requested_ttc_s']) for r in plan['cells']]==list(m.FAILED_CELLS)
    assert all(r['bank_ids']==list(range(8)) and not r['training_eligible'] for r in plan['cells'])
    assert plan['comparison']['planned_cycles']==48 and plan['comparison']['aggregate_reserved_ceiling_seconds']==1740
    assert not list(root.rglob('attempt_started.json'))


@pytest.mark.parametrize('mutation',['request','role','seed','wall','rate','source','ancestry','parent','assessment','bytes'])
def test_bound_contract_mutations_fail_closed(fixture,mutation):
    _,plan=fixture;plan=deepcopy(plan)
    if mutation=='request':plan['cells'][0]['requested_ttc_s']=.74
    if mutation=='role':plan['cells'][0]['training_eligible']=True
    if mutation=='seed':plan['comparison']['seed']=42
    if mutation=='wall':plan['application_wall_seconds']=241
    if mutation=='rate':plan['local_clock_rate_bounds']=[.35,3]
    if mutation=='source':Path(plan['generator_sources'][m.SOURCES[0]]['path']).write_text('changed')
    if mutation=='ancestry':Path(plan['source_ancestry']['cross']['checkpoint.pt']['path']).write_text('changed')
    if mutation in ('parent','assessment'):
        key='parent_catalog' if mutation=='parent' else mutation;Path(plan[key]['path']).write_text('{}')
    if mutation=='bytes':plan['reference_byte_cap']=6*plan['cell_reference_byte_cap']
    with pytest.raises(ValueError):m.validate_plan(plan)


def test_owned_alias_paths_stay_in_prepared_logical_namespace(fixture,tmp_path):
    root,plan=fixture;alias=tmp_path/'logical';alias.symlink_to(root,target_is_directory=True)
    plan['logical_project_root']=str(alias)
    m.validate_plan(plan)
    assert m.logical_path(plan,root/'outputs/task10b/new/reference.npz')==str(alias/'outputs/task10b/new/reference.npz')
    plan['logical_project_root']=str(tmp_path)
    with pytest.raises(ValueError):m.validate_plan(plan)


def mock_generation(fixture,monkeypatch,failure=None):
    root,plan=fixture;path=root/'outputs/plan.json';dump(path,plan)
    monkeypatch.setattr(m,'require_allocation',lambda:None);monkeypatch.setattr(m,'_native_context',lambda paths:None)
    def fake(plan,entry,paths,inputs,folder,checkpoint):
        checkpoint('synthetic_only')
        if failure:failure(entry,folder)
        return dict(**entry,generation_disposition='generated',kinematic_admission=True,reference_uncompressed_array_bytes=0)
    monkeypatch.setattr(m,'_generate_cell',fake)
    return root,plan,path


def test_original_sixteen_rows_preserved_and_new_catalog_never_overwritten(fixture,monkeypatch):
    root,plan,path=mock_generation(fixture,monkeypatch);before=Path(plan['parent_catalog']['path']).read_bytes()
    result=m.generate(path,root/'outputs/task10b/new',wall_seconds=240.)
    assert result['complete'] and result['executed_new_cells']==6 and len(result['cells'])==22
    assert result['cells'][:16]==json.loads(before)['cells'] and Path(plan['parent_catalog']['path']).read_bytes()==before
    assert not result['physics_executed'] and not result['physical_qualification']
    with pytest.raises(FileExistsError):m.generate(path,root/'outputs/task10b/new')


def test_rejection_retained_without_silent_replacement(fixture,monkeypatch):
    def reject(entry,folder):
        if entry['family']=='cross':raise ValueError('clock unsupported')
    root,plan,path=mock_generation(fixture,monkeypatch,reject)
    result=m.generate(path,root/'outputs/task10b/rejected')
    assert result['complete'] and result['executed_new_cells']==6
    assert [r['generation_disposition'] for r in result['cells'][16:]]==['failed']*2+['generated']*4
    assert result['exception'] is None and not result['cells'][16]['kinematic_admission']


def test_interruption_preserves_unknown_extent_and_all_unattempted_cells(fixture,monkeypatch):
    def fail(entry,folder):
        (folder/'reference.npz').write_bytes(b'partial synthetic archive')
        raise m.BoundedExecutionInterruption(dict(reason='test_stop',boundary='synthetic'))
    root,plan,path=mock_generation(fixture,monkeypatch,fail)
    result=m.generate(path,root/'outputs/task10b/partial')
    assert not result['complete'] and result['executed_new_cells']==1 and len(result['unattempted_cells'])==5
    assert result['reference_bytes_measurement_is_lower_bound'] and len(result['unknown_partial_reference_payloads'])==1
    progress=json.loads((root/'outputs/task10b/partial/progress_001.json').read_text())
    assert progress['cells'][0]['generation_disposition']=='generation_started'
    assert 'unknown generation extent' in progress['interruption_semantics']
    assert Path(result['cells'][16]['retained_partial_reference_path']).read_bytes()==b'partial synthetic archive'


def test_allocation_guard_runs_before_output_creation(fixture,monkeypatch):
    root,plan=fixture;path=root/'outputs/plan.json';dump(path,plan)
    def refuse():raise RuntimeError('counted allocation required')
    monkeypatch.setattr(m,'require_allocation',refuse)
    with pytest.raises(RuntimeError,match='allocation'):m.generate(path,root/'outputs/task10b/no')
    assert not (root/'outputs/task10b/no').exists()
