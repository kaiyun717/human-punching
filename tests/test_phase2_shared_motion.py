"""Focused CPU algebra/IK/event tests. No complete reference generation/native run."""
from dataclasses import asdict
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import mujoco

from human_punching import phase2_shared_motion as motion
from human_punching import phase2_shared_motion_reference as composer
from human_punching.phase2_shared_motion_target import refine_motion_target_derivatives
from human_punching.phase2_phases import PhaseSchedule
from human_punching.wbc.provenance import PROJECT_ROOT
from test_phase2_shape_derivatives import synthetic_shared


def quaternion(angle):return np.array([np.cos(angle/2),0,0,np.sin(angle/2)])


def test_relative_root_and_per_foot_motion_preserves_anchors_and_inputs():
    root=np.array([[1,2,3,*quaternion(x)] for x in (0,.1,.2)],float)
    root[:,0]+=[0,.01,.02];anchor=np.r_[4,5,6,quaternion(.4)]
    p=np.zeros((3,2,3));p[1:,:,0]=[[.001,.002],[.002,.004]]
    R=np.repeat(np.eye(3)[None,None],6,axis=0).reshape(3,2,3,3)
    p0=np.array([[.2,.3,.4],[-.2,-.3,.4]]);R0=R[0].copy();before=copy.deepcopy((root,anchor,p,R,p0,R0))
    mapped,feet,rotations=motion.relative_motion_geometry(root,anchor,p,R,p0,R0)
    assert np.array_equal(mapped[0],anchor) and np.array_equal(feet[0],p0)
    assert np.array_equal(rotations[0],R0)
    assert np.allclose(mapped[1,:3],anchor[:3]+np.array([.01*np.cos(.4),.01*np.sin(.4),0]))
    assert np.allclose(mapped[2,3:],quaternion(.6))
    assert np.allclose(feet[2,0]-p0[0],[.002*np.cos(.4),.002*np.sin(.4),0])
    for a,b in zip((root,anchor,p,R,p0,R0),before,strict=True):assert np.array_equal(a,b)


@pytest.mark.parametrize('bad',['quaternion','reflection','nonfinite','shape'])
def test_invalid_frame_payload_cannot_become_a_reference(bad):
    root=np.repeat(np.array([[0,0,1,1,0,0,0]],float),3,axis=0);anchor=root[0].copy()
    p=np.zeros((3,2,3));R=np.tile(np.eye(3),(3,2,1,1));p0=p[0].copy();R0=R[0].copy()
    if bad=='quaternion':root[1,3]=2
    elif bad=='reflection':R[1,0,0,0]=-1
    elif bad=='nonfinite':p[1,1,1]=np.nan
    else:root=np.array(1.)
    with pytest.raises(ValueError):motion.relative_motion_geometry(root,anchor,p,R,p0,R0)


@pytest.mark.parametrize('kwargs',[dict(maximum_iterations=41),dict(maximum_joint_difference_from_source_rad=.61),
    dict(maximum_iteration_joint_delta_rad=.026),dict(position_tolerance_m=.001),dict(orientation_tolerance_rad=.002)])
def test_engineering_boxes_cannot_enlarge_implicitly(kwargs):
    with pytest.raises(ValueError):motion.LowerMotionSettings(**kwargs)


def linear_rig(monkeypatch):
    qa=np.arange(7,21);va=np.arange(6,20);ids=np.arange(1,15)
    model=SimpleNamespace(nv=54,jnt_range=np.repeat([[-1.,1.]],49,axis=0),site=lambda name:SimpleNamespace(id=0 if name.startswith('Right') else 1))
    monkeypatch.setattr(motion,'resolve_shared_joints',lambda m:dict(lower_joint_ids=ids,lower_qpos=qa,lower_dof=va))
    def errors(model,data,q,sites,va,target_p,target_R):
        J=np.zeros((12,14));J[:6,:6]=np.eye(6);error=target_p.ravel()-q[qa[:6]]
        return J,np.r_[error,np.zeros(6)],[float(np.linalg.norm(error[:3])),float(np.linalg.norm(error[3:]))],[0.,0.]
    monkeypatch.setattr(motion,'_pose_errors_and_jacobian',errors)
    q=np.zeros(55);q[3]=1.;q[30]=.123
    return model,q,qa


def test_bounded_lower_ik_solves_without_changing_root_or_upper(monkeypatch):
    model,q,qa=linear_rig(monkeypatch);before=q.copy();p=np.full((2,3),.04);R=np.tile(np.eye(3),(2,1,1))
    out,receipt=motion.solve_lower_pose(model,q,q,p,R,data=object())
    assert receipt['iterations']>=2 and receipt['maximum_position_error_m']<=.0005
    fixed=np.ones(55,bool);fixed[qa]=False
    assert out[fixed].tobytes()==q[fixed].tobytes() and np.array_equal(q,before)
    assert np.max(abs(out[qa]))<=.60 and receipt['reference_only']


def test_unreachable_foot_target_rejects_without_clipped_success(monkeypatch):
    model,q,qa=linear_rig(monkeypatch)
    with pytest.raises(motion.LowerMotionRejection,match='bounded_lower_foot_motion_unsolved'):
        motion.solve_lower_pose(model,q,q,np.full((2,3),.7),np.tile(np.eye(3),(2,1,1)),data=object())
    assert np.all(q[qa]==0)


@pytest.mark.parametrize('kind',['physical','engineering'])
def test_lower_seed_outside_unchanged_bounds_is_not_coerced(monkeypatch,kind):
    model,q,qa=linear_rig(monkeypatch);seed=q.copy();seed[qa[0]]=1.1 if kind=='physical' else .61
    before=seed.copy()
    with pytest.raises(motion.LowerMotionRejection,match='lower_seed_outside'):
        motion.solve_lower_pose(model,seed,q,np.zeros((2,3)),np.tile(np.eye(3),(2,1,1)),data=object())
    assert np.array_equal(seed,before)


def test_actual_accepted_model_identity_foot_target_needs_no_ik_step():
    path=PROJECT_ROOT/'outputs/four-punch-milestone/jab/model/human.xml'
    if not path.exists():pytest.skip('Accepted read-only model unavailable')
    model=mujoco.MjModel.from_xml_path(str(path))
    with np.load(PROJECT_ROOT/'outputs/task10b/p2initialcaptureprep01/train_geometry.npz',allow_pickle=False) as z:q=z['qpos'][0].astype(float)
    p,R=motion.foot_pose_paths(model,q[None]);out,receipt=motion.solve_lower_pose(model,q,q,p[0],R[0])
    assert np.array_equal(out,q) and receipt['iterations']==0


def test_actual_accepted_model_known_reachable_local_foot_target_uses_its_jacobian():
    path=PROJECT_ROOT/'outputs/four-punch-milestone/jab/model/human.xml'
    if not path.exists():pytest.skip('Accepted read-only model unavailable')
    model=mujoco.MjModel.from_xml_path(str(path))
    with np.load(PROJECT_ROOT/'outputs/task10b/p2initialcaptureprep01/train_geometry.npz',allow_pickle=False) as z:q=z['qpos'][0].astype(float)
    truth=q.copy();joint=model.joint('RightLeg_x').id;address=model.jnt_qposadr[joint]
    truth[address]+=.01 if model.jnt_range[joint,1]-truth[address]>.02 else -.01
    p,R=motion.foot_pose_paths(model,truth[None]);out,receipt=motion.solve_lower_pose(model,q,q,p[0],R[0])
    assert receipt['iterations']>0 and receipt['maximum_position_error_m']<=.0005
    assert receipt['maximum_orientation_error_rad']<=.001
    mask=motion.resolve_shared_joints(model);fixed=np.ones(55,bool);fixed[mask['lower_qpos']]=False
    assert out[fixed].tobytes()==q[fixed].tobytes()


def test_all_four_saved_sources_keep_fresh_interval_launch_and_explicit_padding():
    import json
    p=PROJECT_ROOT/'outputs/task10b/p2sharedprobeprep01/parent_catalog.json'
    expected={'jab':(92,92,0),'cross':(100,100,0),'left_hook':(49,48,1),'right_hook':(51,48,3)}
    for row in json.loads(p.read_text())['cells']:
        with np.load(PROJECT_ROOT/row['reference_path'],allow_pickle=False) as z:q=z['qpos'].copy();v=z['qvel'].copy();t=z['time_s'].copy()
        source=PhaseSchedule(0,row['launch_tick'],row['outward_end_tick'],int(np.ceil(row['guard_time_s']/.005-1.e-10)),len(t)-1)
        prep=160;full=PhaseSchedule(prep,prep+source.launch_tick,prep+source.outward_complete_tick,prep+source.guard_hold_tick,prep+source.reference_end_tick)
        v[:source.launch_tick+1]=0.
        candidate=dict(qpos=np.concatenate((np.repeat(q[:1],prep,axis=0),q)),qvel=np.concatenate((np.zeros((prep,54)),v)),
            time_s=np.arange(len(q)+prep)*.005,source_time_s=t,schedule=full)
        events=composer.verify_motion_events(candidate,source);interval,boundary,padding=expected[row['family']]
        assert events['first_moving_reference_interval_tick']==prep+interval
        assert events['observed_first_outward_actor_boundary_tick']==prep+boundary
        assert events['leading_constant_physics_intervals_in_first_outward_actor_interval']==padding
        assert events['leading_constant_reference_time_in_first_outward_actor_interval_s']==padding*.005


def dynamic_candidate():
    model,candidate,metadata=synthetic_shared();candidate['source_time_s']=np.arange(17)*.005
    candidate['qpos'][9:13,0]=.01;candidate['qvel'][9:13,0]=.2;candidate['qacc'][9:13,0]=.3
    source=PhaseSchedule(0,4,8,12,16);metadata['source_schedule']=asdict(source)
    return model,candidate,metadata,source


def test_post_target_projection_preserves_nonzero_free_root_tangents_exactly():
    model,candidate,metadata,source=dynamic_candidate();before=copy.deepcopy(candidate)
    out,receipt=refine_motion_target_derivatives(model,candidate,metadata)
    assert out['qpos'].tobytes()==before['qpos'].tobytes()
    assert out['qvel'][:,:6].tobytes()==before['qvel'][:,:6].tobytes()
    assert out['qacc'][:,:6].tobytes()==before['qacc'][:,:6].tobytes()
    assert np.any(out['qvel'][:,:6]) and receipt['independent_polynomial_extrema_screen_passed']
    events=composer.verify_motion_events(out,source)
    assert events['first_moving_reference_interval_tick']==8 and events['observed_first_outward_actor_boundary_tick']==8
    assert events['phase_denominator_s']==1.1


@pytest.mark.parametrize('bad',['early_motion','early_velocity','changed_boundary','source_clock','proxy','no_motion'])
def test_new_motion_clock_mismatches_reject_without_silent_relabel(bad):
    model,candidate,metadata,source=dynamic_candidate()
    if bad=='early_motion':candidate['qpos'][7,0]=.001
    elif bad=='early_velocity':candidate['qvel'][7,0]=.001
    elif bad=='changed_boundary':candidate['qpos'][5,0]=.001
    elif bad=='source_clock':candidate['source_time_s'][3]+=.001
    elif bad=='proxy':source=PhaseSchedule(0,4,9,12,16)
    else:candidate['qpos'][4:]=candidate['qpos'][4]
    with pytest.raises(ValueError):composer.verify_motion_events(candidate,source)


def test_new_frame_and_ik_modules_never_integrate_or_apply_root_forces():
    import ast
    for name in ('phase2_shared_motion.py','phase2_shared_motion_reference.py','phase2_shared_motion_target.py'):
        tree=ast.parse((PROJECT_ROOT/'src/human_punching'/name).read_text())
        attributes={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}
        assert not attributes&{'mj_step','step','mj_forward','qfrc_applied','xfrc_applied','ctrl','reset','previous_action'}


def composed_fixture(monkeypatch):
    model,base,metadata,source_schedule=dynamic_candidate()
    # The static predecessor is supplied as a synthetic unit fixture. No full
    # accepted-source construction, targetIK, feedforward or screen is executed.
    base['qpos'][:,0]=0.;base['qvel'][:,:6]=0.;base['qacc'][:,:6]=0.
    source_q=base['qpos'][4:].copy();source_q[5:9,0]=.01
    source=dict(qpos=source_q,time_s=np.arange(17)*.005,head_center=np.array([.8,.2,1.2]),active_hand_index=0)
    base.update(glove_pos=np.zeros((21,2,3)),source_original_glove_pos=np.zeros((17,2,3)),
        head_center=source['head_center'],active_hand_index=0)
    lower_q=np.arange(41,55);lower_v=np.arange(40,54)
    monkeypatch.setattr(composer,'compose_shared_reference',lambda *a,**kw:(copy.deepcopy(base),dict(schedule=asdict(base['schedule']))))
    monkeypatch.setattr(composer,'resolve_shared_joints',lambda m:dict(lower_qpos=lower_q,lower_dof=lower_v))
    monkeypatch.setattr(composer,'foot_pose_paths',lambda m,q,**kw:(np.zeros((len(q),2,3)),np.tile(np.eye(3),(len(q),2,1,1))))
    monkeypatch.setattr(composer.mujoco,'MjData',lambda m:object())
    monkeypatch.setattr(composer,'solve_lower_pose',lambda m,q,src,p,R,**kw:(q.copy(),dict(iterations=0)))
    monkeypatch.setattr(composer,'_glove_positions',lambda m,q:np.zeros((len(q),2,3)))
    def tangent(model,t,q):
        out=np.zeros((len(q),54))
        for i in range(len(q)):
            left,right=max(0,i-1),min(len(q)-1,i+1)
            out[i,:3]=(q[right,:3]-q[left,:3])/(t[right]-t[left])
            out[i,6:]=(q[right,7:]-q[left,7:])/(t[right]-t[left])
        return out
    monkeypatch.setattr(composer,'tangent_velocities',tangent)
    return model,base,source,source_schedule


def test_motion_composition_preserves_prefix_and_hold_state_and_declares_new_root(monkeypatch):
    model,base,source,schedule=composed_fixture(monkeypatch);anchor=base['qpos'][4].copy();before=copy.deepcopy(base)
    out,receipt=composer.compose_shared_motion_reference(model,source,anchor,base['qpos'][0],base['qvel'][0],
        source_schedule=schedule,preparation_s=.02,support_evidence={})
    assert out['qpos'][:5].tobytes()==before['qpos'][:5].tobytes()
    assert out['qvel'][:5].tobytes()==before['qvel'][:5].tobytes()
    assert out['qacc'][:5].tobytes()==before['qacc'][:5].tobytes()
    assert np.any(out['qvel'][5:,:3]) and np.any(out['qpos'][5:,0])
    assert np.array_equal(out['qpos'][4:9],np.repeat(out['qpos'][4:5],5,axis=0))
    assert out['source_segment_coefficients'].shape==(16,6,48)
    assert receipt['lower_bound_is_engineering_reference_bound_not_physical_limit']
    assert receipt['motion_event_correspondence']['observed_first_outward_actor_boundary_tick']==8
    assert 'constant accepted-jab throughout' not in receipt['root_and_lower_reference_geometry']


def test_motion_composition_rejects_solver_noise_in_declared_source_hold(monkeypatch):
    model,base,source,schedule=composed_fixture(monkeypatch);anchor=base['qpos'][4].copy()
    def noisy(model,q,*args,**kwargs):
        out=q.copy();out[-1]+=.001;return out,dict(iterations=1)
    monkeypatch.setattr(composer,'solve_lower_pose',noisy)
    with pytest.raises(motion.LowerMotionRejection,match='constant_source_hold'):
        composer.compose_shared_motion_reference(model,source,anchor,base['qpos'][0],base['qvel'][0],
            source_schedule=schedule,preparation_s=.02,support_evidence={})


def test_motion_solver_failure_retains_exact_frame_and_unattempted_extent(monkeypatch):
    model,base,source,schedule=composed_fixture(monkeypatch);anchor=base['qpos'][4].copy();count=[]
    def fail_later(model,q,*args,**kwargs):
        count.append(True)
        if len(count)==6:raise motion.LowerMotionRejection('bounded_lower_foot_motion_unsolved',maximum_position_error_m=.01)
        return q.copy(),dict(iterations=0)
    monkeypatch.setattr(composer,'solve_lower_pose',fail_later)
    with pytest.raises(motion.LowerMotionRejection) as caught:
        composer.compose_shared_motion_reference(model,source,anchor,base['qpos'][0],base['qvel'][0],
            source_schedule=schedule,preparation_s=.02,support_evidence={})
    row=caught.value.disposition
    assert row['frame']==row['frames_completed']==5 and row['frames_unattempted']==11
    assert row['source_time_s']==.025 and row['maximum_position_error_m']==.01
