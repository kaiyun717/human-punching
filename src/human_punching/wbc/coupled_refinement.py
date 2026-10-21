"""Offline prototype coupling support forces, actuator caps and trajectory shape.

This module never integrates dynamics, actuates the root or clips a required
torque into apparent feasibility. Slack in the six free-base equations is an
explicit unresolved deficit. All output remains unqualified until independently
screened after hand-path adaptation, FK contact measurement and TTC retiming.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import mujoco
import numpy as np
from scipy import sparse
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeWarning, brentq, linprog

from .balance_refinement import (BalanceSettings,_collision_linearization,
    _minimum_pair_distance,_spline_basis,_whole_body_map)
from .effort_diagnostics import contact_edges,support_effort_solution,unconstrained_inverse
from .planning_refinement import _PoseCurve,_fk_contact,_unbiased_support_solve
from .references import tangent_velocities


@dataclass(frozen=True)
class CoupledSettings:
    control_points: int = 20
    outer_iterations: int = 3
    collocation_stride: int = 4
    trust_radius: float = .30
    hand_task_weight: float = .35
    maximum_glove_adaptation_m: float = .04
    source_character_weight: float = .002
    temporal_weight: float = .02

    def __post_init__(self):
        if self.control_points<8 or self.outer_iterations<1 or self.collocation_stride<1:
            raise ValueError('Positive iteration/grid counts and at least eight spline points required')
        if any(not np.isfinite(value) or value<=0 for value in (
                self.trust_radius,self.hand_task_weight,self.maximum_glove_adaptation_m,
                self.source_character_weight,self.temporal_weight)):
            raise ValueError('Finite positive coupled-refinement settings required')


def _screened_line_search(evaluate,physical_merit,coefficient,step):
    """Reject an invalid nonlinear trial without skipping the smaller trials.

    A violated support-clearance domain is not a valid force solution. Its exact
    reason is retained; it receives infinite merit, never relaxed contact rules.
    """
    trials=[];rejections=[]
    for fraction in (1.,.5,.25,0.):
        proposal=coefficient+fraction*step
        try:
            score,metrics,_=physical_merit(evaluate(proposal),proposal)
        except ValueError as error:
            score,metrics=np.inf,{'invalid_trial':str(error)}
            rejections.append({'step_fraction':fraction,'reason':str(error)})
        trials.append((score,fraction,proposal,metrics))
    best=min(trials,key=lambda row:row[0])
    if not np.isfinite(best[0]):
        raise ValueError('No valid nonlinear trial, including the unchanged reference')
    return best,rejections


def solve_coupled_subproblem(coefficient,world_wrench,world_jac,joint_required,joint_jac,
        contact_world,contact_joint,caps,hard_map,hard_lower,hard_upper,curvature,
        bound_low,bound_high,weight,settings):
    """Linearized forces with hard joint effort caps and explicit root slack.

    Variables are spline step, nonnegative friction-pyramid edge forces, positive
    and negative free-base residual, and absolute coefficient/curvature values.
    There is no root actuator variable. All 48 actuator inequalities use their
    existing caps; root residual is measured, penalized and never concealed.
    """
    frames,nforce=contact_joint.shape[0],contact_joint.shape[2]
    n=len(coefficient);nj=len(caps);nr=6*frames;nf=nforce*frames;nc=len(curvature)
    total=n+nf+2*nr+n+nc
    zero=lambda rows,columns:sparse.csr_matrix((rows,columns))
    normalizer=np.tile(np.r_[np.full(3,weight),np.full(3,.5*weight)],frames)
    world=sparse.block_diag([contact_world]*frames,format='csr').multiply(1/normalizer[:,None])
    jac=sparse.csr_matrix(world_jac.reshape(nr,n)/normalizer[:,None])
    equality=sparse.hstack((jac,-world,-sparse.eye(nr),sparse.eye(nr),zero(nr,n+nc)),format='csr')
    rhs=-world_wrench.ravel()/normalizer
    cap_vector=np.tile(caps,frames)
    bj=sparse.block_diag(list(contact_joint),format='csr').multiply(1/cap_vector[:,None])
    jt=sparse.csr_matrix(joint_jac.reshape(frames*nj,n)/cap_vector[:,None])
    torque=sparse.hstack((jt,-bj,zero(frames*nj,2*nr+n+nc)),format='csr')
    constraints=[torque,-torque]
    bounds=[1.-joint_required.ravel()/cap_vector,1.+joint_required.ravel()/cap_vector]
    hard=sparse.hstack((sparse.csr_matrix(hard_map),zero(len(hard_map),total-n)),format='csr')
    constraints.extend((hard,-hard));bounds.extend((hard_upper,-hard_lower))
    for sign in (1.,-1.):
        constraints.append(sparse.hstack((sign*sparse.eye(n),zero(n,nf+2*nr),-sparse.eye(n),zero(n,nc))))
        bounds.append(-sign*coefficient)
        constraints.append(sparse.hstack((sign*sparse.csr_matrix(curvature),zero(nc,nf+2*nr+n),-sparse.eye(nc))))
        bounds.append(-sign*(curvature@coefficient))
    objective=np.r_[np.zeros(n),np.full(nf,1e-5/(weight*frames)),
        np.full(2*nr,100./frames),np.full(n,settings.source_character_weight),
        np.full(nc,settings.temporal_weight)]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected.*threads.*',category=OptimizeWarning)
        result=linprog(objective,A_ub=sparse.vstack(constraints,format='csr'),b_ub=np.concatenate(bounds),
            A_eq=equality,b_eq=rhs,bounds=list(zip(bound_low,bound_high,strict=True))+[(0,None)]*(total-n),
            method='highs',options={'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if result.x is None:
        return result,None
    step=result.x[:n];forces=result.x[n:n+nf].reshape(frames,nforce)
    normalized_residual=(result.x[n+nf:n+nf+nr]-result.x[n+nf+nr:n+nf+2*nr]).reshape(frames,6)
    predicted=(joint_required+np.einsum('fjn,n->fj',joint_jac,step)
               -np.einsum('fjl,fl->fj',contact_joint,forces))
    audit={'root_residual_l1_mean_normalized':float(np.abs(normalized_residual).sum()/frames),
           'root_residual_max_normalized':float(np.abs(normalized_residual).max()),
           'predicted_peak_joint_effort_ratio':float(np.max(np.abs(predicted)/caps)),
           'contact_edge_loads':forces,'step':step}
    return result,audit


def coupled_refine_trajectory(model_path,times,poses,guard_time_s,caps,*,settings=None):
    """Bounded force-aware path prototype with soft intermediate glove tasks."""
    settings=settings or CoupledSettings()
    times,poses,caps=np.asarray(times),np.asarray(poses),np.asarray(caps)
    model=mujoco.MjModel.from_xml_path(str(model_path))
    if poses.shape!=(len(times),model.nq) or model.nv!=54 or caps.shape!=(48,):
        raise ValueError('Expected the named constrained-human trajectory and its 48 caps')
    if not np.allclose(np.diff(times),.005,atol=1e-12,rtol=0):
        raise ValueError('Coupled refinement requires a 200 Hz master')
    maps=np.array([_whole_body_map(model,pose,settings.hand_task_weight) for pose in poses])
    base_settings=BalanceSettings(control_points=settings.control_points,quiet_boundary_s=.06)
    basis=_spline_basis(times,guard_time_s,base_settings)
    scales=np.array([.03,.03,.025,.10,.10,.12,.10,.10,.12])
    n=basis.shape[1]*9
    indices=np.unique(np.r_[np.arange(0,len(times),settings.collocation_stride),len(times)-1])
    data=mujoco.MjData(model);data.qpos[:]=poses[0];mujoco.mj_forward(model,data)
    _,corners,mus=contact_edges(model,data)
    origin=poses[0,:3].copy();origin[2]=0.
    friction=float(mus.min())
    edge_directions=np.array([[friction,0,1],[-friction,0,1],[0,friction,1],[0,-friction,1]])
    world_edges=np.array([np.r_[force,np.cross(point-origin,force)] for point in corners for force in edge_directions]).T
    weight=float(model.body_mass.sum()*abs(model.opt.gravity[2]))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
    lower,upper=model.jnt_range[1:].T
    cache={}

    def evaluate(coefficients):
        key=coefficients.tobytes()
        if key in cache:return cache[key]
        latent=(basis@coefficients.reshape(-1,9))*scales
        delta=np.einsum('nij,nj->ni',maps,latent)
        q=poses.copy()
        for pose,change in zip(q,delta,strict=True):mujoco.mj_integratePos(model,pose,change,1.)
        v=tangent_velocities(model,times,q);a=np.gradient(v,times,axis=0,edge_order=2)
        required=[];wrenches=[];matrices=[]
        for index in indices:
            demand=unconstrained_inverse(model,data,q[index],v[index],a[index])
            matrix,_,_=contact_edges(model,data)
            rotation=data.xmat[model.body('Hips').id].reshape(3,3)
            moment=rotation@demand[3:6]+np.cross(q[index,:3]-origin,demand[:3])
            required.append(demand);wrenches.append(np.r_[demand[:3],moment]);matrices.append(matrix)
        value=(q,np.asarray(required),np.asarray(wrenches),np.asarray(matrices))
        cache[key]=value
        return value

    def physical_merit(value,coefficients):
        q,required,_,matrices=value
        excursion=max(float((lower-q[:,7:]).max()),float((q[:,7:]-upper).max()),0.)
        if excursion>1e-8 or _minimum_pair_distance(model,q)<-.001:
            return np.inf,{},None
        residuals=[];overloads=[];forces=[];failed=0
        for demand,matrix in zip(required,matrices,strict=True):
            result=support_effort_solution(demand,matrix,caps)
            forces.append(result['edge_loads'])
            residual=result['base_residual']/np.r_[np.full(3,weight),np.full(3,.5*weight)]
            residuals.append(np.abs(residual).sum())
            overloads.append(max(0.,float(np.max(np.abs(result['joint_torque'])/caps))-1.))
            failed+=int(not result['support_feasible'])
        merit=100*np.mean(residuals)+25*np.mean(overloads)+settings.source_character_weight*np.abs(coefficients).sum()
        return float(merit),{'collocation_support_failed':failed,'mean_root_residual':float(np.mean(residuals)),
            'mean_effort_overload':float(np.mean(overloads))},np.asarray(forces)

    coefficient=np.zeros(n);ledger=[]
    curvature=np.kron(np.diff(np.eye(basis.shape[1]),n=2,axis=0),np.eye(9))
    for iteration in range(settings.outer_iterations):
        cache.clear();base=evaluate(coefficient)
        merit,before,force0=physical_merit(base,coefficient)
        if force0 is None:
            raise ValueError('Source prototype violates hard geometry before coupled refinement')
        if before['collocation_support_failed']==0 and before['mean_effort_overload']<=1e-8:
            ledger.append({'iteration':iteration,'already_within_collocation_support_and_effort':True});break
        q,required,wrench,matrices=base
        world_jac=np.empty((len(indices),6,n));joint_jac=np.empty((len(indices),48,n))
        base_net=required[:,6:]-np.einsum('fjl,fl->fj',matrices[:,6:],force0)
        for column in range(n):
            test=coefficient.copy();test[column]+=.002
            _,r,w,b=evaluate(test)
            world_jac[:,:,column]=(w-wrench)/.002
            joint_jac[:,:,column]=(r[:,6:]-np.einsum('fjl,fl->fj',b[:,6:],force0)-base_net)/.002
        hard=np.einsum('nc,njk,k->njck',basis,maps[:,6:],scales).reshape(-1,n)
        hard_low=(lower-q[:,7:]).ravel();hard_high=(upper-q[:,7:]).ravel()
        collision_map,distances=_collision_linearization(model,q,maps,basis,scales)
        hard=np.vstack((hard,collision_map));hard_low=np.r_[hard_low,.001-distances]
        hard_high=np.r_[hard_high,np.full(len(distances),100.)]
        result,linear_audit=solve_coupled_subproblem(coefficient,wrench,world_jac,required[:,6:],joint_jac,
            world_edges,matrices[:,6:],caps,hard,hard_low,hard_high,curvature,
            np.maximum(-settings.trust_radius,-1.-coefficient),np.minimum(settings.trust_radius,1.-coefficient),weight,settings)
        if linear_audit is None:
            ledger.append({'iteration':iteration,'solver_success':False,'solver_status':int(result.status),
                           'solver_message':result.message});break
        (score,fraction,proposal,metrics),rejections=_screened_line_search(
            evaluate,physical_merit,coefficient,linear_audit['step'])
        ledger.append({'iteration':iteration,'solver_success':bool(result.success),'solver_status':int(result.status),
            'solver_message':result.message,'before':before,'after':metrics,'accepted_step_fraction':fraction,
            'linear_root_residual_l1_mean_normalized':linear_audit['root_residual_l1_mean_normalized'],
            'linear_root_residual_max_normalized':linear_audit['root_residual_max_normalized'],
            'predicted_peak_joint_effort_ratio':linear_audit['predicted_peak_joint_effort_ratio'],
            'merit_before':merit,'merit_after':score,'invalid_nonlinear_trials':rejections})
        coefficient=proposal
        if fraction==0:break
    proposed=evaluate(coefficient)[0]
    # The normal-contact model is separate from constraint-disabled inverse
    # dynamics. Preserve the adapted hand path while correcting exact foot tasks.
    model=mujoco.MjModel.from_xml_path(str(model_path));data=mujoco.MjData(model)
    data.qpos[:]=poses[0];mujoco.mj_forward(model,data)
    feet={side:(data.site_xpos[model.site(side+'Foot_sole').id].copy(),
                data.site_xmat[model.site(side+'Foot_sole').id].reshape(3,3).copy()) for side in ('Left','Right')}
    corrected=[]
    for index,prior in enumerate(proposed):
        if times[index]<=.06 or times[index]>=guard_time_s-.06:
            corrected.append(poses[index].copy());continue
        data.qpos[:]=prior;mujoco.mj_forward(model,data)
        points=[('body','Hips',prior[:3].copy(),3.)]
        rotations=[('body','Chest',data.xmat[model.body('Chest').id].reshape(3,3).copy(),1.)]
        for side in ('Left','Right'):
            points.append(('site',side+'Hand_center',data.site_xpos[model.site(side+'Hand_center').id].copy(),3.))
            points.append(('site',side+'Foot_sole',feet[side][0],100.))
            rotations.append(('site',side+'Foot_sole',feet[side][1],30.))
        temporal=prior.copy()
        if corrected:
            delta=np.zeros(model.nv);mujoco.mj_differentiatePos(model,delta,1.,proposed[index-1],prior)
            temporal=corrected[-1].copy();mujoco.mj_integratePos(model,temporal,delta,1.)
        corrected.append(_unbiased_support_solve(model,prior,temporal,points,rotations,16))
    corrected=np.asarray(corrected)
    glove_changes=[]
    for old,new in zip(poses,corrected,strict=True):
        data.qpos[:]=old;mujoco.mj_forward(model,data)
        original=np.array([data.site_xpos[model.site(side+'Hand_center').id] for side in ('Left','Right')])
        data.qpos[:]=new;mujoco.mj_forward(model,data)
        changed=np.array([data.site_xpos[model.site(side+'Hand_center').id] for side in ('Left','Right')])
        glove_changes.append(np.linalg.norm(changed-original,axis=1))
    glove_changes=np.asarray(glove_changes)
    if glove_changes.max()>settings.maximum_glove_adaptation_m:
        raise ValueError('Prototype exceeds its declared maximum intermediate glove-path adaptation')
    return corrected,{'schema':'human_wbc_coupled_refinement_v1','settings':asdict(settings),
        'physics_executed':False,'qualified_for_training':False,'actuator_constraints':'Hard existing joint effort caps inside each force-allocation LP; no torque clipping.',
        'freebase_constraints':'Six linearized required wrench equations with explicit penalized residual; no base actuator.',
        'intermediate_glove_path':'Soft source-relative task; final first contact and TTC must be remeasured.',
        'maximum_glove_adaptation_m':float(glove_changes.max()),
        'rms_glove_adaptation_m':float(np.sqrt(np.mean(glove_changes**2))),
        'coefficients':coefficient.reshape(-1,9).tolist(),'iteration_ledger':ledger}


def retime_first_contact(model_path,times,poses,target,active_hand,requested_ttc_s,outward_end_s,guard_time_s):
    """Remeasure changed-path first contact, then rebuild clock and derivatives."""
    model=mujoco.MjModel.from_xml_path(str(model_path));curve=_PoseCurve(times,poses)
    site=model.site(('Left' if active_hand==0 else 'Right')+'Hand_center').id
    contact=_fk_contact(model,curve,np.asarray(target),site,outward_end_s)
    old=np.array([0.,contact,guard_time_s,times[-1]])
    new=np.array([0.,requested_ttc_s,requested_ttc_s+guard_time_s-contact,
                  requested_ttc_s+times[-1]-contact])
    if np.any(np.diff(old)<=0) or np.any(np.diff(new)<=0):
        raise ValueError('Invalid changed-path contact/guard ordering')
    source_clock=PchipInterpolator(new,old)
    mapped_outward=float(brentq(lambda t:float(source_clock(t))-outward_end_s,0.,new[2]))
    end=float(np.ceil((new[-1]-1e-10)/.02)*.02)
    final_time=np.arange(round(end/.005)+1)*.005
    final_pose=curve(source_clock(np.clip(final_time,0.,new[-1])))
    final_pose[0]=poses[0];final_pose[final_time>=new[2]]=poses[-1]
    velocity=tangent_velocities(model,final_time,final_pose)
    acceleration=np.gradient(velocity,final_time,axis=0,edge_order=2)
    if np.max(abs(velocity[[0,-1]]))>1e-10 or np.max(abs(acceleration[[0,-1]]))>1e-8:
        raise ValueError('Retiming did not retain quiet sampled endpoint derivatives')
    return final_time,final_pose,velocity,{'source_first_contact_s':contact,
        'requested_ttc_s':requested_ttc_s,'source_knots_s':old.tolist(),'final_knots_s':new.tolist(),
        'outward_end_s':mapped_outward,'guard_time_s':float(new[2]),'all_derivatives_recomputed':True}
