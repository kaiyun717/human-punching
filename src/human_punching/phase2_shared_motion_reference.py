"""New continuous shared reference with source-relative pelvis and foot motion."""
from dataclasses import asdict
import numpy as np
import mujoco

from .phase2_shared_reference_shape_v3 import (compose_shared_reference,resolve_shared_joints,
    SharedPreparationBounds,quintic_coefficients,_require_limits,_glove_positions)
from .phase2_shared_motion import (VERSION as MOTION_VERSION,LowerMotionSettings,
    LowerMotionRejection,foot_pose_paths,relative_motion_geometry,solve_lower_pose)
from .phase2_shape_derivatives import project_derivatives
from .wbc.references import tangent_velocities

VERSION='task10b_shared_relative_motion_analytic_connector_projected_hinges_v1'


def verify_motion_events(candidate,source_schedule):
    """Derive fresh moving interval/actor boundary and prove source-time mapping.

    The outward completion remains a separately declared conservative original
    withdrawal proxy. A time-preserving geometry map does not turn it into a
    new physical contact or arm-extremum event.
    """
    q=np.asarray(candidate['qpos']);v=np.asarray(candidate['qvel']);t=np.asarray(candidate['time_s'])
    schedule=candidate['schedule'];prep=schedule.preparation_complete_tick
    source_t=np.asarray(candidate.get('source_time_s'),float)
    if (q.ndim!=2 or q.shape[1]!=55 or v.shape!=(len(q),54) or t.shape!=(len(q),)
            or not np.allclose(t,np.arange(len(q))*.005,rtol=0,atol=1.e-12)
            or source_t.shape!=(len(q)-prep,)
            or not np.allclose(source_t,t[prep:]-t[prep],atol=1.e-12,rtol=0)):
        raise ValueError('Complete unchanged5ms source-time mapping required')
    expected=prep+source_schedule.launch_tick
    # Strict bit identity detects solver noise in a declared hold. It is not
    # the predicate for physical readiness; ClosedProtocolLaunchGuard owns that.
    moving=np.flatnonzero(np.any(q[prep+1:]!=q[prep:-1],axis=1))
    if not len(moving):raise LowerMotionRejection('no_outward_reference_motion')
    first_interval=prep+int(moving[0]);boundary=first_interval//4*4
    if (boundary!=expected or schedule.launch_tick!=boundary
            or not np.array_equal(q[prep:boundary+1],np.repeat(q[prep:prep+1],boundary-prep+1,axis=0))
            or np.any(v[prep:boundary+1])):
        raise LowerMotionRejection('fresh_launch_boundary_or_prefix_differs',first_moving_interval_tick=first_interval,
            observed_actor_boundary_tick=boundary,source_mapped_actor_boundary_tick=expected)
    if (schedule.outward_complete_tick!=prep+source_schedule.outward_complete_tick
            or schedule.guard_hold_tick!=prep+source_schedule.guard_hold_tick
            or schedule.reference_end_tick!=prep+source_schedule.reference_end_tick):
        raise LowerMotionRejection('source_phase_correspondence_changed')
    return dict(schema='task10b_shared_motion_event_correspondence_v1',first_moving_reference_interval_tick=first_interval,
        observed_first_outward_actor_boundary_tick=boundary,mapped_source_launch_actor_boundary_tick=expected,
        leading_constant_physics_intervals_in_first_outward_actor_interval=first_interval-boundary,
        leading_constant_reference_time_in_first_outward_actor_interval_s=(first_interval-boundary)*.005,
        actor_interval_padding_semantics='The first outward 20 ms actor interval contains 0..3 leading constant 5 ms reference intervals inherited from source phase alignment. No reference samples are added and no TTC or launch clock is shifted.',
        constant_initial_source_hold_verified=True,identity_source_time_mapping=True,
        outward_complete_tick=schedule.outward_complete_tick,
        outward_event_semantics='Original conservative source-playback proxy including withdrawal, transported only under identity source-time mapping; not a remeasured physical extremum/contact event',
        actual_launch_requires='Current positional/support readiness plus final immutable commitment at the derived 20 ms actor boundary; late or unready trials reject without shifting the reference or TTC',
        preparation_complete_tick=prep,phase_denominator_s=schedule.phase_denominator_s,physical_readiness_or_execution_qualified=False)


def compose_shared_motion_reference(model,source,accepted_jab_qpos,start_qpos,start_qvel,*,
        source_schedule,preparation_s,support_evidence,preparation_bounds=SharedPreparationBounds(),
        motion_settings=LowerMotionSettings(),checkpoint=None):
    """Keep the sealed analytic prefix; replace only the new source geometry."""
    if not isinstance(motion_settings,LowerMotionSettings):raise ValueError('Explicit bounded lower-motion settings required')
    base,receipt=compose_shared_reference(model,source,accepted_jab_qpos,start_qpos,start_qvel,
        source_schedule=source_schedule,preparation_s=preparation_s,support_evidence=support_evidence,
        preparation_bounds=preparation_bounds)
    q0=np.asarray(accepted_jab_qpos,float);original=np.asarray(source['qpos'],float)
    t=np.asarray(source['time_s'],float);prep=base['schedule'].preparation_complete_tick
    if not np.array_equal(original[source_schedule.guard_hold_tick],original[0]):
        raise LowerMotionRejection('source_guard_pose_not_initial_pose_for_reviewed_transport')
    p,R=foot_pose_paths(model,original,checkpoint=checkpoint)
    anchor_p,anchor_R=foot_pose_paths(model,q0[None],checkpoint=checkpoint)
    root,target_p,target_R=relative_motion_geometry(original[:,:7],q0[:7],p,R,anchor_p[0],anchor_R[0])
    mask=resolve_shared_joints(model);qa=mask['lower_qpos'];va=mask['lower_dof'];source_q=original.copy();source_q[:,:7]=root
    records=[];data=mujoco.MjData(model);previous=q0[qa].copy()
    for index in range(len(source_q)):
        if checkpoint is not None:checkpoint()
        in_hold=index<=source_schedule.launch_tick or index>=source_schedule.guard_hold_tick
        seed=source_q[index].copy();seed[qa]=q0[qa] if in_hold else previous
        try:
            solved,record=solve_lower_pose(model,seed,original[index],target_p[index],target_R[index],settings=motion_settings,data=data)
        except LowerMotionRejection as error:
            details={key:value for key,value in error.disposition.items()
                if key not in ('reason','motion_version','physical_qualification')}
            raise LowerMotionRejection(error.disposition['reason'],frame=index,source_time_s=float(t[index]),
                frames_completed=index,frames_unattempted=len(source_q)-index-1,**details) from error
        if in_hold and not np.array_equal(solved[qa],q0[qa]):
            raise LowerMotionRejection('constant_source_hold_lower_solution_differs',frame=index)
        source_q[index]=solved;previous=solved[qa].copy();records.append(dict(frame=index,**record))
    # The prefix's final q is already this exact source initial pose. Never
    # reset physical/controller state or overwrite a physical root here.
    if not np.array_equal(base['qpos'][prep],source_q[0]):
        raise LowerMotionRejection('motion_source_not_continuous_with_analytic_connector')
    out={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in base.items()}
    out['qpos'][prep:]=source_q
    velocity=tangent_velocities(model,out['time_s'],out['qpos']);velocity[:prep+1]=base['qvel'][:prep+1]
    holds=((prep,prep+source_schedule.launch_tick),(prep+source_schedule.guard_hold_tick,len(out['qpos'])-1))
    for left,right in holds:velocity[left:right+1]=0.
    acceleration=np.gradient(velocity,out['time_s'],axis=0,edge_order=2);acceleration[:prep+1]=base['qacc'][:prep+1]
    for left,right in holds:acceleration[left:right+1]=0.
    allqa=np.asarray(model.jnt_qposadr)[1:];allva=np.asarray(model.jnt_dofadr)[1:]
    projected=project_derivatives(source_q[:,allqa],velocity[prep:,allva],acceleration[prep:,allva],model.jnt_range[1:],
        joint_names=[model.joint(i).name for i in range(1,49)])
    for interval,c in enumerate(projected.coefficients):_require_limits(c,model.jnt_range[1:],phase='relative_motion_source',segment=interval)
    velocity[prep:,allva]=projected.velocity;acceleration[prep:,allva]=projected.acceleration
    if not np.array_equal(velocity[:prep+1],base['qvel'][:prep+1]) or not np.array_equal(acceleration[:prep+1],base['qacc'][:prep+1]):
        raise LowerMotionRejection('motion_derivatives_change_connector_endpoint')
    out.update(qvel=velocity,qacc=acceleration,source_segment_coefficients=projected.coefficients,
        source_segment_qpos_addresses=allqa.copy(),source_segment_dof_addresses=allva.copy(),
        glove_pos=_glove_positions(model,out['qpos']),derivative_version=VERSION)
    events=verify_motion_events(out,source_schedule)
    fresh_v=tangent_velocities(model,out['time_s'],out['qpos'])
    fresh_a=np.gradient(fresh_v,out['time_s'],axis=0,edge_order=2)
    info={**receipt,'schema':'task10b_shared_relative_motion_reference_v1','recipe':MOTION_VERSION,'motion_version':MOTION_VERSION,
        'derivative_version':VERSION,'motion_settings':asdict(motion_settings),'lower_ik_frames':records,
        'source_derivative_projection':projected.receipt,'motion_source_derivative_projection':projected.receipt,
        'motion_event_correspondence':events,
        'root_geometry':'Exact source-relative pelvis position/orientation about the shared jab initial root; reference geometry only, with no physical state/root assistance',
        'foot_geometry':'Each source-relative foot pose is anchored at the explicitly reached nominal jab sole pose. Source foot motion is retained; there is no universal stationary-foot requirement.',
        'root_derivatives':'Fresh original MuJoCo free-root tangent convention on the exact 5 ms grid. Only scalar hinges use the C2 Bezier projection. No new globally C2 quaternion interpolant is claimed.',
        'root_and_lower_reference_geometry':'Original bounded analytic preparation to shared nominal jab geometry, then source-relative pelvis and per-foot motion with 14-joint lower-body IK. No physical state, root or sole pose is swapped or constrained.',
        'source_derivatives':'Fresh original free-root tangent velocity/acceleration on final 5 ms source geometry; 48 scalar hinge derivatives projected into unchanged endpoint boxes and checked with full polynomial extrema. Original upper joint knots are preserved before target IK; lower joints and root reference geometry are newly constructed. No root interpolation or C2 claim beyond the stored knot contract.',
        'source_upper_path_between_knots':'Scalar hinges use explicit C2 quintic intervals through their declared knots. Root motion uses original tangent conventions and exact 5 ms playback; no analytic quaternion interpolant is introduced.',
        'runtime_sampling_required':'Exact integer 5 ms playback including +20/+50 future ticks is required. A future between-knot free-root interpolator needs separately reviewed derivative and phase semantics.',
        'constant_source_hold_windows_ticks':[list(window) for window in holds],
        'constant_source_hold_windows_clock':'full reset-relative 5 ms ticks',
        'finite_difference_crosscheck':dict(velocity_max_abs=float(abs(velocity-fresh_v).max()),
            acceleration_max_abs=float(abs(acceleration-fresh_a).max()),strict_finite_difference_only_audit_applicable=False,
            use='Diagnostics on the final motion reference only; analytic prefix and projected scalar hinge derivatives are not replaced by centered differences.'),
        'lower_reference_difference_bound_rad':motion_settings.maximum_joint_difference_from_source_rad,
        'lower_bound_is_engineering_reference_bound_not_physical_limit':True,'full_prefix_qva_preserved':True,
        'physical_qualification':False,'feedforward_recomputed':False}
    return out,info
