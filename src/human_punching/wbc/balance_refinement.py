"""Experimental trajectory-level centroidal support correction, offline only.

Small pelvis/chest spline variables drive a whole-body Jacobian map preserving
both feet and glove paths. Sequential convex subproblems constrain the complete
required free-base wrench to a conservative contact cone. Positive slack remains
an explicit failure, and all output needs independent dense FK/LP qualification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import mujoco
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import OptimizeWarning, linprog, minimize
from scipy import sparse
from scipy.spatial import ConvexHull

from .effort_diagnostics import contact_edges, unconstrained_inverse
from .planning_refinement import _unbiased_support_solve
from .references import tangent_velocities


@dataclass(frozen=True)
class BalanceSettings:
    control_points: int = 16
    outer_iterations: int = 3
    collocation_stride: int = 4
    finite_difference_step: float = .002
    trust_radius: float = .35
    cop_margin_m: float = .003
    quiet_boundary_s: float = .06
    source_character_weight: float = .002
    temporal_weight: float = .01
    projection_iterations: int = 16
    subproblem_solver: str = 'slsqp'

    def __post_init__(self):
        if self.control_points < 8 or self.outer_iterations < 1 or self.collocation_stride < 1:
            raise ValueError('Positive iteration/grid counts and at least eight spline points required')
        if any(not np.isfinite(value) or value <= 0 for value in (
                self.finite_difference_step,self.trust_radius,self.cop_margin_m,
                self.quiet_boundary_s,self.source_character_weight,self.temporal_weight)):
            raise ValueError('Finite positive balance-refinement settings required')
        if self.subproblem_solver not in ('slsqp','linear_program'):
            raise ValueError('Unknown sequential convex subproblem solver')


def _linear_subproblem(coefficient,jac,violation,joint_map,joint_low,joint_high,
                       bound_low,bound_high,curvature,facets,settings):
    """Sparse LP: mean positive per-frame wrench deficit plus maximum deficit.

    Absolute spline size and curvature retain source character. Every slack is
    explicit; this is a correction search, never permission to call it feasible.
    """
    n=len(coefficient);frames=len(violation)//facets;nc=len(curvature)
    # [coefficient step, max slack, frame slacks, |coefficients|, |curvature|]
    total=n+1+frames+n+nc
    j=sparse.csr_matrix(jac)
    assignment=sparse.csr_matrix((np.ones(len(violation)),
        (np.arange(len(violation)),np.repeat(np.arange(frames),facets))),shape=(len(violation),frames))
    zeros=lambda rows,cols:sparse.csr_matrix((rows,cols))
    matrices=[sparse.hstack((j,zeros(len(jac),1),-assignment,zeros(len(jac),n+nc))),
        sparse.hstack((zeros(frames,n),-np.ones((frames,1)),sparse.eye(frames),zeros(frames,n+nc)))]
    rhs=[-violation,np.zeros(frames)]
    jm=sparse.csr_matrix(joint_map)
    matrices.extend((sparse.hstack((jm,zeros(len(joint_map),total-n))),
                     sparse.hstack((-jm,zeros(len(joint_map),total-n)))))
    rhs.extend((joint_high,-joint_low))
    for sign in (1.,-1.):
        matrices.append(sparse.hstack((sign*sparse.eye(n),zeros(n,1+frames),-sparse.eye(n),zeros(n,nc))))
        rhs.append(-sign*coefficient)
        matrices.append(sparse.hstack((sign*sparse.csr_matrix(curvature),zeros(nc,1+frames+n),-sparse.eye(nc))))
        rhs.append(-sign*(curvature@coefficient))
    objective=np.r_[np.zeros(n),1.,np.full(frames,100./frames),
                     np.full(n,settings.source_character_weight),np.full(nc,settings.temporal_weight)]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected.*threads.*',category=OptimizeWarning)
        result=linprog(objective,A_ub=sparse.vstack(matrices,format='csr'),b_ub=np.concatenate(rhs),
            bounds=list(zip(bound_low,bound_high,strict=True))+[(0.,None)]*(total-n),
            method='highs',options={'threads':1,'primal_feasibility_tolerance':1e-9,
                                    'dual_feasibility_tolerance':1e-9})
    if result.x is not None:
        result.x=np.r_[result.x[:n],result.x[n]]
    return result


def contact_wrench_halfspaces(corners: np.ndarray, mu: float,
                             origin: np.ndarray) -> np.ndarray:
    """Cone of all nonnegative foot-corner friction-pyramid edge forces.

    Slice at total Fz=1 gives a bounded 5-D polytope. Its facet inequalities
    homogenize back into six-component world-wrench halfspaces A*w <= 0.
    This matches the conservative force allocation used by the LP diagnostic.
    """
    corners,origin=np.asarray(corners),np.asarray(origin)
    if corners.shape!=(8,3) or origin.shape!=(3,) or not np.isfinite(mu) or mu<=0:
        raise ValueError('Eight support corners, a world origin and positive friction required')
    forces=np.array([[mu,0,1],[-mu,0,1],[0,mu,1],[0,-mu,1]])
    rays=np.array([np.r_[force,np.cross(point-origin,force)] for point in corners for force in forces])
    section=rays[:,[0,1,3,4,5]]
    hull=ConvexHull(section)
    equations=hull.equations
    matrix=np.column_stack((equations[:,0],equations[:,1],equations[:,5],
                            equations[:,2],equations[:,3],equations[:,4]))
    matrix/=np.linalg.norm(matrix,axis=1)[:,None]
    matrix=np.unique(np.round(matrix,12),axis=0)
    # Explicit positive normal-force halfspace protects the homogenization.
    return np.vstack((matrix,[0,0,-1,0,0,0]))


def _whole_body_map(model,pose,glove_weight=100.):
    data=mujoco.MjData(model)
    data.qpos[:]=pose
    mujoco.mj_forward(model,data)
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    rows,targets=[],[]
    def append(jac,weight,columns=None):
        target=np.zeros((3,9))
        if columns is not None:
            target[:,columns]=np.eye(3)
        rows.append(weight*jac.copy());targets.append(weight*target)
    for side in ('Left','Right'):
        mujoco.mj_jacSite(model,data,jp,jr,model.site(side+'Foot_sole').id)
        append(jp,300.);append(jr,150.)
        mujoco.mj_jacSite(model,data,jp,jr,model.site(side+'Hand_center').id)
        append(jp,glove_weight)
    mujoco.mj_jacBody(model,data,jp,jr,model.body('Hips').id)
    append(jp,10.,slice(0,3));append(jr,10.,slice(3,6))
    mujoco.mj_jacBody(model,data,jp,jr,model.body('Chest').id)
    append(jr,8.,slice(6,9))
    jac=np.vstack(rows)
    return np.linalg.solve(jac.T@jac+.15**2*np.eye(model.nv),jac.T@np.vstack(targets))


def _spline_basis(times,guard_time,settings):
    start,end=settings.quiet_boundary_s,guard_time-settings.quiet_boundary_s
    if end-start <= .5:
        raise ValueError('Insufficient motion interval for quiet balance splines')
    n=settings.control_points
    knots=np.r_[np.full(4,start),np.linspace(start,end,n-2)[1:-1],np.full(4,end)]
    full=BSpline(knots,np.eye(n),3,extrapolate=False)(np.clip(times,start,end))
    # Three zero endpoint controls give zero pose/velocity/acceleration there.
    basis=full[:,3:-3]
    basis[(times<=start)|(times>=end)]=0.
    return basis


def _collision_linearization(model,poses,maps,basis,scales):
    """Named model-pair distance constraints, even with constraint solving off."""
    data=mujoco.MjData(model)
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    rows,distances=[],[]
    for index,pose in enumerate(poses):
        data.qpos[:]=pose;mujoco.mj_forward(model,data)
        for g1,g2 in zip(model.pair_geom1,model.pair_geom2,strict=True):
            points=np.zeros(6)
            distance=float(mujoco.mj_geomDistance(model,data,int(g1),int(g2),.12,points))
            if distance>=.119999 or abs(distance)<1e-12:
                continue
            normal=(points[3:]-points[:3])/distance
            mujoco.mj_jac(model,data,jp,jr,points[:3],int(model.geom_bodyid[g1]))
            first=jp.copy()
            mujoco.mj_jac(model,data,jp,jr,points[3:],int(model.geom_bodyid[g2]))
            latent=(normal@(jp-first))@maps[index]*scales
            row=np.outer(basis[index],latent).ravel()
            if np.max(abs(row))>1e-12:
                rows.append(row);distances.append(distance)
    return np.asarray(rows).reshape(-1,basis.shape[1]*9),np.asarray(distances)


def _minimum_pair_distance(model,poses):
    data=mujoco.MjData(model)
    minimum=.12
    for pose in poses:
        data.qpos[:]=pose;mujoco.mj_forward(model,data)
        for g1,g2 in zip(model.pair_geom1,model.pair_geom2,strict=True):
            minimum=min(minimum,float(mujoco.mj_geomDistance(model,data,int(g1),int(g2),.12,None)))
    return minimum


def refine_balance_trajectory(model_path,times,poses,guard_time_s,*,settings=None):
    """Return an unqualified trajectory prototype with solver/constraint ledger.

    The source path, timing and body stay fixed. Nine named latent channels are
    world pelvis translation/orientation and world chest orientation. Whole-body
    IK distributes corrections over limbs and torso; this is not arm-only aiming.
    """
    settings=settings or BalanceSettings()
    times,poses=np.asarray(times),np.asarray(poses)
    if poses.shape!=(len(times),55) or not np.allclose(np.diff(times),.005,rtol=0,atol=1e-12):
        raise ValueError('A complete 200 Hz constrained-human master is required')
    model=mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
    data=mujoco.MjData(model)
    data.qpos[:]=poses[0]
    mujoco.mj_forward(model,data)
    _,corners,mus=contact_edges(model,data)
    origin=poses[0,:3].copy();origin[2]=0.
    cone=contact_wrench_halfspaces(corners,float(mus.min()),origin)
    hull=ConvexHull((corners-origin)[:,:2])
    # These rows are physical CoP clearance (metres after division by weight).
    cop=np.array([[0,0,c+settings.cop_margin_m,b,-a,0] for a,b,c in hull.equations])
    inequalities=np.vstack((cone,cop))
    basis=_spline_basis(times,guard_time_s,settings)
    maps=np.array([_whole_body_map(model,pose) for pose in poses])
    scales=np.array([.03,.03,.025,.10,.10,.12,.10,.10,.12])
    variables=basis.shape[1]*9
    indices=np.unique(np.r_[np.arange(0,len(times),settings.collocation_stride),len(times)-1])
    weight=float(model.body_mass.sum()*abs(model.opt.gravity[2]))
    cache={}

    def evaluate(coefficients):
        key=coefficients.tobytes()
        if key in cache:
            return cache[key]
        latent=(basis@coefficients.reshape(-1,9))*scales
        delta=np.einsum('nij,nj->ni',maps,latent)
        output=poses.copy()
        for pose,change in zip(output,delta,strict=True):
            mujoco.mj_integratePos(model,pose,change,1.)
        velocity=tangent_velocities(model,times,output)
        acceleration=np.gradient(velocity,times,axis=0,edge_order=2)
        wrenches=[]
        for index in indices:
            required=unconstrained_inverse(model,data,output[index],velocity[index],acceleration[index])
            rotation=data.xmat[model.body('Hips').id].reshape(3,3)
            moment=rotation@required[3:6]+np.cross(output[index,:3]-origin,required[:3])
            wrenches.append(np.r_[required[:3],moment])
        violation=np.asarray(wrenches)@inequalities.T/weight
        result=(output,violation.ravel())
        cache[key]=result
        return result

    coefficient=np.zeros(variables)
    ledger=[]
    # Penalize curve magnitude and coefficient curvature to preserve source
    # whole-body character, alongside a minimized explicit constraint slack.
    second=np.diff(np.eye(basis.shape[1]),n=2,axis=0)
    character=settings.source_character_weight*np.eye(variables)
    temporal=settings.temporal_weight*np.kron(second.T@second,np.eye(9))
    quadratic=character+temporal
    lower,upper=model.jnt_range[1:].T
    for iteration in range(settings.outer_iterations):
        cache.clear()
        base_pose,violation=evaluate(coefficient)
        if violation.max()<=1e-6:
            ledger.append({'iteration':iteration,'solver_success':True,
                'solver_status':0,'solver_message':'Current collocation wrenches satisfy all halfspaces',
                'maximum_wrench_violation_before':float(violation.max()),
                'maximum_wrench_violation_after':float(violation.max()),'accepted_step_fraction':0.})
            break
        jac=np.empty((len(violation),variables))
        for column in range(variables):
            test=coefficient.copy();test[column]+=settings.finite_difference_step
            jac[:,column]=(evaluate(test)[1]-violation)/settings.finite_difference_step
        # Joint positions are exactly additive for the scalar hinge coordinates.
        joint_indices=np.arange(len(times)) if settings.subproblem_solver=='linear_program' else indices
        joint_map=np.einsum('nc,njk,k->njck',basis[joint_indices],maps[joint_indices,6:],scales).reshape(-1,variables)
        joint_now=base_pose[joint_indices,7:].ravel()
        joint_low=np.tile(lower,len(joint_indices))-joint_now
        joint_high=np.tile(upper,len(joint_indices))-joint_now
        if settings.subproblem_solver=='linear_program':
            collision_map,distances=_collision_linearization(model,base_pose,maps,basis,scales)
            joint_map=np.vstack((joint_map,collision_map))
            joint_low=np.r_[joint_low,.001-distances]
            joint_high=np.r_[joint_high,np.full(len(distances),100.)]
        bound_low=np.maximum(-settings.trust_radius,-1.-coefficient)
        bound_high=np.minimum(settings.trust_radius,1.-coefficient)
        linear=np.vstack((np.column_stack((-jac,np.ones(len(jac)))),
            np.column_stack((joint_map,np.zeros(len(joint_map)))),
            np.column_stack((-joint_map,np.zeros(len(joint_map))))))
        offset=np.r_[-violation,-joint_low,joint_high]
        def objective(x):
            current=coefficient+x[:-1]
            return 10.*x[-1]+.5*current@quadratic@current
        def gradient(x):
            return np.r_[quadratic@(coefficient+x[:-1]),10.]
        initial=np.r_[np.zeros(variables),max(0.,float(violation.max()))+1e-8]
        if settings.subproblem_solver=='linear_program':
            solution=_linear_subproblem(coefficient,jac,violation,joint_map,joint_low,joint_high,
                bound_low,bound_high,np.kron(second,np.eye(9)),len(inequalities),settings)
        else:
            solution=minimize(objective,initial,jac=gradient,method='SLSQP',
                bounds=list(zip(bound_low,bound_high,strict=True))+[(0.,None)],
                constraints={'type':'ineq','fun':lambda x:offset+linear@x,'jac':lambda x:linear},
                options={'maxiter':100,'ftol':1e-9})
        if solution.x is None:
            ledger.append({'iteration':iteration,'solver_success':False,'solver_status':int(solution.status),
                'solver_message':str(solution.message),'accepted_step_fraction':0.})
            break
        candidates=[]
        trial_ledger=[]
        for fraction in (1.,.5,.25,0.):
            proposal=coefficient+fraction*solution.x[:-1]
            candidate_pose,actual=evaluate(proposal)
            excursion=max(float(np.max(lower-candidate_pose[:,7:])),float(np.max(candidate_pose[:,7:]-upper)),0.)
            if settings.subproblem_solver=='linear_program':
                per_frame=actual.reshape(-1,len(inequalities)).max(axis=1)
                merit=float(max(0.,actual.max())+100*np.maximum(per_frame,0.).mean())
                merit+=float(.5*proposal@quadratic@proposal)
                if excursion>1e-8:
                    merit=np.inf
                minimum_distance=_minimum_pair_distance(model,candidate_pose) if np.isfinite(merit) else None
                if minimum_distance is not None and minimum_distance<-.001:
                    merit=np.inf
            else:
                merit=float(actual.max())+100*excursion
                minimum_distance=None
            candidates.append((merit,fraction,proposal,actual,excursion))
            trial_ledger.append({'fraction':fraction,'merit':float(merit) if np.isfinite(merit) else None,
                'joint_excursion_rad':excursion,'minimum_pair_distance_m':minimum_distance})
        _,fraction,proposal,actual,excursion=min(candidates,key=lambda row:row[0])
        ledger.append({'iteration':iteration,'solver_success':bool(solution.success),'solver_status':int(solution.status),
            'solver_message':str(solution.message),'linear_slack':float(solution.x[-1]),
            'maximum_wrench_violation_before':float(violation.max()),
            'maximum_wrench_violation_after':float(actual.max()),'accepted_step_fraction':fraction,
            'maximum_dense_joint_limit_excursion_rad':excursion,'line_search':trial_ledger})
        coefficient=proposal
        if fraction==0 or actual.max()<=1e-6:
            break
    proposed,final_violation=evaluate(coefficient)
    # Restore exact foot/hand tasks after the first-order map, preserving the
    # newly optimized pelvis/chest priors. No unconstrained smoothing follows.
    # Constraint-disabled inverse dynamics also omits contact records. Use a
    # separate unchanged compiled model so the IK collision terms remain active.
    model=mujoco.MjModel.from_xml_path(str(model_path))
    data=mujoco.MjData(model)
    corrected=[]
    foot_targets={}
    data.qpos[:]=poses[0];mujoco.mj_forward(model,data)
    for side in ('Left','Right'):
        index=model.site(side+'Foot_sole').id
        foot_targets[side]=(data.site_xpos[index].copy(),data.site_xmat[index].reshape(3,3).copy())
    for index,prior in enumerate(proposed):
        if times[index]<=settings.quiet_boundary_s or times[index]>=guard_time_s-settings.quiet_boundary_s:
            corrected.append(poses[index].copy());continue
        data.qpos[:]=poses[index];mujoco.mj_forward(model,data)
        points=[('site',side+'Hand_center',data.site_xpos[model.site(side+'Hand_center').id].copy(),15.) for side in ('Left','Right')]
        data.qpos[:]=prior;mujoco.mj_forward(model,data)
        points.append(('body','Hips',prior[:3].copy(),3.))
        orientations=[('body','Chest',data.xmat[model.body('Chest').id].reshape(3,3).copy(),1.)]
        for side in ('Left','Right'):
            point,rotation=foot_targets[side]
            points.append(('site',side+'Foot_sole',point,100.))
            orientations.append(('site',side+'Foot_sole',rotation,30.))
        temporal=prior.copy()
        if corrected:
            delta=np.zeros(model.nv)
            mujoco.mj_differentiatePos(model,delta,1.,proposed[index-1],prior)
            temporal=corrected[-1].copy();mujoco.mj_integratePos(model,temporal,delta,1.)
        corrected.append(_unbiased_support_solve(model,prior,temporal,points,orientations,settings.projection_iterations))
    return np.asarray(corrected),{'schema':'human_wbc_balance_refinement_prototype_v1',
        'physics_executed':False,'qualified_for_training':False,'settings':asdict(settings),
        'latent_channels':['pelvis_world_xyz_m','pelvis_world_rotation_rad','chest_world_rotation_rad'],
        'coefficient_scales':scales.tolist(),'coefficients':coefficient.reshape(-1,9).tolist(),
        'wrench_halfspaces':len(inequalities),'collocation_frames':len(indices),
        'iteration_ledger':ledger,'maximum_preprojection_wrench_violation':float(final_violation.max()),
        'qualification':'Positive slack is an unresolved constraint violation. Dense postprojection FK, timing, derivatives and native support LP remain mandatory.'}
