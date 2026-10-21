"""Synthetic in-memory shared geometry fixtures; no project references or physics."""
from copy import deepcopy

import mujoco
import numpy as np
import pytest

from human_punching.phase2_phases import PhaseSchedule
from human_punching.phase2_shared_reference import (LOWER_JOINT_NAMES,
    SharedReferenceRejection, array_digest, compose_shared_reference,
    evaluate_quintic, polynomial_joint_bounds, polynomial_derivative_bounds,
    quintic_coefficients, resolve_shared_joints)


@pytest.fixture
def fixture():
    upper = ''
    for i in range(34):
        upper += f'<body name="upper{i}"><joint name="upper_{i}" range="-2 2"/><geom size=".02" mass=".1"/>'
        if i<2:upper += '<site name="'+('Left' if i==0 else 'Right')+'Hand_center" pos=".15 0 0"/>'
        upper += '</body>'
    legs = ''
    for side in ('Right','Left'):
        names = [name for name in LOWER_JOINT_NAMES if name.startswith(side)]
        legs += ''.join(f'<body name="{name}"><joint name="{name}" range="-2 2"/><geom size=".02" mass=".1"/>' for name in names)
        legs += ''.join(f'<site name="{side}Foot_sole_{i}" pos="{.08 if i<2 else -.08} {.04 if i%2 else -.04} -.1"/>' for i in range(4))+'</body>'*7
    model = mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/>'
        '<worldbody><body><freejoint/><geom size=".1" mass="1"/>'+upper+legs+'</body></worldbody></mujoco>')
    t = np.arange(201)*.005
    q = np.tile(model.qpos0, (len(t),1)); q[:,0] = .2
    mask = resolve_shared_joints(model)
    q[:,mask['lower_qpos']] = .1
    x = np.clip((t-.2)/.6,0,1)
    q[:,mask['upper_qpos'][0]] = .04+.12*(64*x**3*(1-x)**3)
    source = dict(time_s=t,qpos=q,head_center=np.array([.5,0.,1.]),active_hand_index=0)
    jab = model.qpos0.copy(); jab[2] = .9
    start = jab.copy();start[mask['upper_qpos'][0]] += .01
    velocity = np.zeros(model.nv);velocity[mask['upper_dof'][0]] = .04
    support = dict(evidence_id='synthetic_support_fixture_not_execution',
        state_qpos_qvel_sha256=array_digest(start,velocity), supported_feet=[True,True],
        finite_upright=True,disallowed_body_floor_contact=False,state_time_s=2.,support_interval_end_s=2.)
    return dict(model=model,source=source,accepted_jab_qpos=jab,start_qpos=start,start_qvel=velocity,
        source_schedule=PhaseSchedule(0,40,160,180,200),preparation_s=.4,support_evidence=support)


def test_connector_exact_start_velocity_c2_join_and_no_input_mutation(fixture):
    before = {name:value.copy() for name,value in fixture.items() if isinstance(value,np.ndarray)}
    source_before = fixture['source']['qpos'].copy()
    candidate, receipt = compose_shared_reference(**fixture)
    mask = resolve_shared_joints(fixture['model']);qa,va = mask['upper_qpos'],mask['upper_dof']
    np.testing.assert_array_equal(candidate['qpos'][0],fixture['start_qpos'])
    np.testing.assert_array_equal(candidate['qvel'][0],fixture['start_qvel'])
    connector_end = evaluate_quintic(candidate['connector_coefficients'],np.array([.4]),.4)
    for field, endpoint in zip(('qpos','qvel','qacc'),connector_end):
        addresses=candidate['connector_qpos_addresses' if field=='qpos' else 'connector_dof_addresses']
        np.testing.assert_allclose(candidate[field][80,addresses],endpoint[0],atol=3e-13)
    np.testing.assert_array_equal(candidate['qpos'][80:,qa],source_before[:,qa])
    np.testing.assert_array_equal(fixture['source']['qpos'],source_before)
    for name,value in before.items():np.testing.assert_array_equal(fixture[name],value)
    assert not receipt['state_or_history_modified']
    assert not receipt['initial_physical_qacc_continuity_qualified']
    assert receipt['finite_difference_crosscheck']['velocity_max_abs']>0
    assert not receipt['existing_linear_reference_library_compatible']
    for index in (0,39,40,45,80,180):
        endpoints=evaluate_quintic(candidate['source_segment_coefficients'][index],np.array([0.,.005]),.005)
        for field, values in zip(('qpos','qvel','qacc'),endpoints):
            np.testing.assert_allclose(candidate[field][80+index:82+index,qa if field=='qpos' else va],values,atol=1e-9)


def test_protected_reference_root_and_fourteen_named_legs_stay_exact(fixture):
    candidate, receipt = compose_shared_reference(**fixture)
    mask = resolve_shared_joints(fixture['model'])
    qa = np.r_[np.arange(7),mask['lower_qpos']]
    va = np.r_[np.arange(6),mask['lower_dof']]
    np.testing.assert_array_equal(candidate['qpos'][:,qa],np.tile(fixture['accepted_jab_qpos'][qa],(281,1)))
    assert not np.any(candidate['qvel'][:,va]) and not np.any(candidate['qacc'][:,va])
    assert len(receipt['lower_joint_names'])==14
    assert receipt['sole_geometry_preserved_by_verified_kinematic_ancestry']
    assert not receipt['dynamics_or_support_from_geometry_qualified']


def test_explicit_launch_follows_preparation_and_preserves_source_holds(fixture):
    candidate,receipt = compose_shared_reference(**fixture)
    schedule = candidate['schedule']
    assert candidate['qvel'][0].max()>0
    assert (schedule.preparation_complete_tick,schedule.launch_tick,schedule.outward_complete_tick)==(80,120,240)
    assert schedule.reference_end_tick==280 and schedule.phase_denominator_s==pytest.approx(2.4)
    assert candidate['time_s'][80:].shape==fixture['source']['time_s'].shape
    np.testing.assert_allclose(candidate['time_s'][80:]-.4,fixture['source']['time_s'],atol=1e-15)
    assert not receipt['launch_inferred_from_motion']
    assert receipt['source_clock_and_all_hold_knots_preserved']
    for first,last in receipt['constant_source_hold_windows_ticks']:
        assert not np.any(candidate['qvel'][80+first:80+last+1])
        assert not np.any(candidate['qacc'][80+first:80+last+1])
        for index in range(first,last):
            coefficients=candidate['source_segment_coefficients'][index]
            assert not np.any(coefficients[1:])


def test_original_glove_world_corridor_retained_and_target_never_relabelled(fixture):
    candidate,receipt = compose_shared_reference(**fixture)
    assert not np.allclose(candidate['glove_pos'][80:],candidate['source_original_glove_pos'])
    np.testing.assert_array_equal(candidate['head_center'],fixture['source']['head_center'])
    assert not receipt['original_world_glove_path_preserved']
    assert 'unresolved' in receipt['target_geometry_status']
    assert not receipt['physics_integrated'] and not receipt['physical_qualification']


@pytest.mark.parametrize('kind',['root_pose','lower_pose','root_velocity','lower_velocity','upper_pose','upper_velocity'])
def test_unsupported_perturbed_starts_rejected_never_zeroed(fixture,kind):
    mask = resolve_shared_joints(fixture['model'])
    field,index,delta = {
        'root_pose':('start_qpos',0,.001),'lower_pose':('start_qpos',mask['lower_qpos'][0],.016),
        'root_velocity':('start_qvel',0,1e-10),'lower_velocity':('start_qvel',mask['lower_dof'][0],.061),
        'upper_pose':('start_qpos',mask['upper_qpos'][0],.016),'upper_velocity':('start_qvel',mask['upper_dof'][0],.061)}[kind]
    fixture[field][index] += delta
    unchanged=fixture[field].copy()
    fixture['support_evidence']['state_qpos_qvel_sha256']=array_digest(fixture['start_qpos'],fixture['start_qvel'])
    with pytest.raises(SharedReferenceRejection):compose_shared_reference(**fixture)
    np.testing.assert_array_equal(fixture[field],unchanged)


@pytest.mark.parametrize('changes,reason',[
    ({'supported_feet':[True,False]},'unsupported'),
    ({'state_qpos_qvel_sha256':'wrong'},'does_not_bind'),
    ({'support_interval_end_s':1.9},'stale'),
    ({'disallowed_body_floor_contact':True},'unsupported'),
])
def test_bound_current_support_required(fixture,changes,reason):
    fixture['support_evidence'].update(changes)
    with pytest.raises(SharedReferenceRejection,match=reason):compose_shared_reference(**fixture)


def test_continuous_quintic_extrema_detect_between_sample_limit_overshoot():
    coefficients = quintic_coefficients([0.],[10.],[0.],[0.],[-10.],[0.],1.)
    lo,hi = polynomial_joint_bounds(coefficients)
    assert lo[0]==pytest.approx(0.) and hi[0]>3.
    q,v,a=evaluate_quintic(coefficients,np.array([0.,1.]),1.)
    np.testing.assert_allclose(q,0.,atol=1e-13)
    np.testing.assert_allclose(v[:,0],[10.,-10.],atol=1e-13)
    np.testing.assert_allclose(a,0.,atol=1e-13)


def test_connector_limit_excursion_rejects_even_when_endpoints_legal(fixture):
    mask=resolve_shared_joints(fixture['model']);joint=mask['upper_joint_ids'][0]
    fixture['model'].jnt_range[joint]=[-.1,.01001]
    fixture['source']['qpos'][:,mask['upper_qpos'][0]]=0.
    with pytest.raises(SharedReferenceRejection,match='joint_limit_excursion') as caught:
        compose_shared_reference(**fixture)
    assert caught.value.disposition['phase']=='preparation'


def test_source_limits_and_unaligned_preparation_fail_closed(fixture):
    mask=resolve_shared_joints(fixture['model'])
    bad=deepcopy(fixture['source']);bad['qpos'][80,mask['upper_qpos'][0]]=3.
    with pytest.raises(SharedReferenceRejection,match='joint_limit_excursion'):
        compose_shared_reference(**(fixture|{'source':bad}))
    with pytest.raises(ValueError,match='20ms'):
        compose_shared_reference(**(fixture|{'preparation_s':.405}))


def test_incorrect_original_hold_annotation_rejected(fixture):
    fixture['source']['qpos'][10,7]+=.001
    with pytest.raises(SharedReferenceRejection,match='hold_not_constant'):
        compose_shared_reference(**fixture)


def test_moving_source_acceleration_differentiates_final_zero_hold_velocity(fixture):
    candidate,receipt=compose_shared_reference(**fixture)
    prep=candidate['source_start_tick'];t=fixture['source']['time_s']
    velocity=candidate['qvel'][prep:]
    expected=np.gradient(velocity,t,axis=0,edge_order=2)
    moving=np.ones(len(t),dtype=bool)
    for first,last in receipt['constant_source_hold_windows_ticks']:
        moving[first:last+1]=False
    np.testing.assert_array_equal(candidate['qacc'][prep:][moving],expected[moving])
    # Hermite coefficients still match the final prescribed derivatives at
    # both sides of the first moving interval, after the stencil correction.
    first=fixture['source_schedule'].launch_tick
    mask=resolve_shared_joints(fixture['model'])
    _,sample_v,sample_a=evaluate_quintic(candidate['source_segment_coefficients'][first],np.array([0.,.005]),.005)
    np.testing.assert_allclose(sample_v,velocity[first:first+2,mask['upper_dof']],atol=1e-10)
    np.testing.assert_allclose(sample_a,candidate['qacc'][prep+first:prep+first+2,mask['upper_dof']],atol=1e-10)
    assert candidate['derivative_version'].endswith('_v3')


def test_four_joint_perturbed_starts_keep_lower_state_and_explicit_prep(fixture):
    rng=np.random.default_rng(21021);mask=resolve_shared_joints(fixture['model'])
    qa=fixture['model'].jnt_qposadr[1:];va=fixture['model'].jnt_dofadr[1:]
    for start in range(4):
        q=fixture['accepted_jab_qpos'].copy();v=np.zeros(54)
        if start:
            q[qa]+=rng.uniform(-.015,.015,48);v[va]=rng.uniform(-.06,.06,48)
        support={**fixture['support_evidence'],'state_qpos_qvel_sha256':array_digest(q,v)}
        candidate,receipt=compose_shared_reference(**(fixture|dict(start_qpos=q,start_qvel=v,support_evidence=support)))
        np.testing.assert_array_equal(candidate['qpos'][0],q)
        np.testing.assert_array_equal(candidate['qvel'][0],v)
        np.testing.assert_array_equal(candidate['qpos'][:,:7],np.tile(q[:7],(len(candidate['time_s']),1)))
        assert not np.any(candidate['qvel'][:,:6]) and not np.any(candidate['qacc'][:,:6])
        np.testing.assert_array_equal(candidate['qpos'][80:,mask['lower_qpos']],
            np.tile(fixture['accepted_jab_qpos'][mask['lower_qpos']],(201,1)))
        assert not np.any(candidate['qvel'][80:,mask['lower_dof']])
        endpoints=evaluate_quintic(candidate['connector_coefficients'],np.array([0.,.4]),.4)
        for name,values,addresses in zip(('qpos','qvel','qacc'),endpoints,(qa,va,va)):
            np.testing.assert_allclose(candidate[name][[0,80]][:,addresses],values,atol=2e-12)
        assert receipt['lower_preparation_moves']==bool(start)
        assert not receipt['fixed_sole_during_preparation_claimed']
        assert receipt['source_sole_geometry_constant_after_preparation_by_verified_kinematic_ancestry']
        if start:
            assert receipt['sole_preparation_diagnostics']['maximum_sampled_displacement_from_initial_m']>0
            assert receipt['preparation_motion']['maximum_lower_speed_rad_s']>0
            assert not receipt['sole_geometry_preserved_by_verified_kinematic_ancestry']
        assert not receipt['physical_qualification'] and not receipt['initial_physical_qacc_continuity_qualified']


def test_short_fast_leg_preparation_rejected_by_explicit_analytic_bounds(fixture):
    mask=resolve_shared_joints(fixture['model'])
    fixture['start_qpos'][mask['lower_qpos'][0]]+=.015
    fixture['start_qvel'][mask['lower_dof'][0]]=.06
    fixture['support_evidence']['state_qpos_qvel_sha256']=array_digest(fixture['start_qpos'],fixture['start_qvel'])
    with pytest.raises(SharedReferenceRejection,match='lower_preparation_motion_outside_recipe'):
        compose_shared_reference(**(fixture|dict(preparation_s=.02)))


def test_derivative_extrema_bound_dense_quintic_samples():
    coefficients=quintic_coefficients([.015,-.015],[.06,-.06],[0.,0.],[0.,0.],[0.,0.],[0.,0.],.4)
    _,v,a=evaluate_quintic(coefficients,np.linspace(0,.4,2001),.4)
    for order,values in ((1,v),(2,a)):
        low,high=polynomial_derivative_bounds(coefficients,.4,order)
        assert np.all(values>=low-1e-12) and np.all(values<=high+1e-12)
        np.testing.assert_allclose(values.min(0),low,atol=2e-6)
        np.testing.assert_allclose(values.max(0),high,atol=2e-6)


def test_provisional_geometry_candidate_cannot_fabricate_native_support(fixture):
    support=dict(evidence_id='synthetic_geometry_only',evidence_kind='provisional_geometry_construction_only',
        native_support_established=False,native_execution_permitted=False,
        state_qpos_qvel_sha256=array_digest(fixture['start_qpos'],fixture['start_qvel']))
    _,receipt=compose_shared_reference(**(fixture|dict(support_evidence=support)))
    assert not receipt['native_support_established'] and not receipt['physical_qualification']
    for extra in ({'native_execution_permitted':True},{'supported_feet':[True,True]},
                  {'finite_upright':True},{'support_interval_end_s':0.}):
        with pytest.raises(SharedReferenceRejection,match='fabricate_physical_support'):
            compose_shared_reference(**(fixture|dict(support_evidence=support|extra)))


def test_outward_rounded_native_endpoint_is_rejected_without_state_coercion(fixture):
    address=resolve_shared_joints(fixture['model'])['upper_qpos'][0]
    fixture['accepted_jab_qpos'][address]=float(np.float32(.1))
    fixture['start_qpos'][address]=float(np.float32(fixture['accepted_jab_qpos'][address]+.015))
    assert fixture['start_qpos'][address]-fixture['accepted_jab_qpos'][address]>.015
    fixture['support_evidence']['state_qpos_qvel_sha256']=array_digest(fixture['start_qpos'],fixture['start_qvel'])
    with pytest.raises(SharedReferenceRejection,match='outside_local'):
        compose_shared_reference(**fixture)
