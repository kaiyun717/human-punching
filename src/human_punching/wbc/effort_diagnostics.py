"""Offline inverse-dynamics and conservative contact-force diagnostics.

The free base is never actuated. Ground forces must satisfy its six equations.
Neither LP feasibility nor overload proves physical execution or impossibility;
the test concerns one exact reference under idealized declared support contacts.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import warnings

import mujoco
import numpy as np
from scipy.optimize import OptimizeWarning, linprog, lsq_linear

from .provenance import writable_path
from .references import sha256, tangent_velocities
from .runtime.bundle import load_runtime_bundle


def support_effort_solution(required: np.ndarray, generalized_edges: np.ndarray,
                            effort_caps: np.ndarray) -> dict:
    """Minimize largest joint |torque|/cap over nonnegative pyramid edge loads.

    generalized_edges maps edge coefficients (N) to generalized ground force.
    Its first six rows are unactuated free-base coordinates. Each contact edge
    contributes one unit normal force; no unconstrained base-wrench variable is
    permitted. An approximate nonnegative fallback is reported only as residual.
    """
    required, matrix, caps = map(lambda x:np.asarray(x,dtype=float),
                                (required,generalized_edges,effort_caps))
    if (required.ndim!=1 or matrix.ndim!=2 or matrix.shape[0]!=len(required)
            or len(required)!=len(caps)+6 or len(caps)<1 or matrix.shape[1]<1
            or not all(np.isfinite(x).all() for x in (required,matrix,caps))
            or np.any(caps<=0)):
        raise ValueError('Finite free-base demand, contact edges and positive actuator caps required')
    columns=matrix.shape[1]
    objective=np.r_[np.zeros(columns),1.]
    # required_joint - B_joint*lambda is the joint actuator torque.
    inequalities=np.vstack((np.column_stack((-matrix[6:],-caps)),
                             np.column_stack((matrix[6:],-caps))))
    upper=np.r_[-required[6:],required[6:]]
    equality=np.column_stack((matrix[:6],np.zeros(6)))
    # SciPy passes the supported native HiGHS thread option through with a
    # wrapper warning. Restrict this tiny LP to one thread in shared test nodes.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message="Unrecognized options detected.*threads.*",category=OptimizeWarning)
        result=linprog(objective,A_ub=inequalities,b_ub=upper,A_eq=equality,b_eq=required[:6],
                       bounds=[(0,None)]*(columns+1),method='highs',options={'threads':1})
    exact=bool(result.success)
    if exact:
        loads=result.x[:columns]
    else:
        # Moments are converted to force-equivalent units with a declared 0.5 m
        # scale only for this residual diagnostic, never for feasibility claims.
        scale=np.r_[np.ones(3),np.full(3,2.)]
        approximate=lsq_linear(matrix[:6]*scale[:,None],required[:6]*scale,
                               bounds=(0,np.inf),tol=1e-10,max_iter=500,method='bvls')
        loads=approximate.x
    supported=matrix@loads
    residual=required[:6]-supported[:6]
    torque=required[6:]-supported[6:]
    if exact and (np.linalg.norm(residual[:3])>1e-5 or np.linalg.norm(residual[3:])>1e-5):
        exact=False
        status='numerical_residual_failure'
    else:
        status='feasible' if exact else ('support_infeasible' if result.status==2 else 'solver_failure')
    ratio=float(np.max(np.abs(torque)/caps))
    return {'status':status,'support_feasible':exact,'effort_within_caps':exact and ratio<=1.+1e-6,
            'minimum_peak_effort_ratio':ratio if exact else None,'edge_loads':loads,
            'joint_torque':torque,'base_residual':residual,'solver_status':int(result.status),
            'solver_message':result.message}


def contact_edges(model: mujoco.MjModel,data: mujoco.MjData) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """World forces at projected foot-box corners and their full Jacobians.

    Uses the inscribed four-edge pyramid |fx|+|fy| <= mu*fz. This conservative
    diagnostic is not advertised as identical to every simulator cone convention.
    Foot corners are projected down their sub-millimetre reference clearance to
    z=0; these are hypothetical declared support contacts, not measured forces.
    """
    floor=model.geom('floor').id
    matrices,points,mus=[],[],[]
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    for side in ('Left','Right'):
        geom=model.geom('physical/'+side+'Foot').id
        if model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError('Expected solid foot boxes')
        mu=min(float(model.geom_friction[floor,0]),float(model.geom_friction[geom,0]))
        if mu<=0:
            raise ValueError('Contact friction must be positive')
        pyramid=np.array([[mu,-mu,0.,0.],[0.,0.,mu,-mu],[1.,1.,1.,1.]])
        size=model.geom_size[geom]
        rotation=data.geom_xmat[geom].reshape(3,3)
        for sx,sy in product((-1,1),repeat=2):
            point=data.geom_xpos[geom]+rotation@np.array([sx*size[0],sy*size[1],-size[2]])
            if not -.001 <= point[2] <= .003:
                raise ValueError('Reference foot corner is outside declared support clearance')
            point[2]=0.
            mujoco.mj_jac(model,data,jp,jr,point,int(model.geom_bodyid[geom]))
            matrices.append(jp.T@pyramid)
            points.append(point.copy())
            mus.append(mu)
    return np.concatenate(matrices,axis=1),np.asarray(points),np.asarray(mus)


def unconstrained_inverse(model: mujoco.MjModel,data: mujoco.MjData,
                          qpos: np.ndarray,qvel: np.ndarray,qacc: np.ndarray) -> np.ndarray:
    """Raw required generalized external force, excluding all constraint forces.

    Only the temporary CPU diagnostic model disables constraints. Asset XML,
    masses, inertias, joint limits, actuators, and execution settings are unchanged.
    Native inverse dynamics and the caller's derivative convention are continuous;
    no simulator integration step or hidden substep occurs here.
    """
    if not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONSTRAINT:
        raise ValueError('Diagnostic inverse requires constraints explicitly disabled')
    data.qpos[:],data.qvel[:],data.qacc[:]=qpos,qvel,qacc
    mujoco.mj_inverse(model,data)
    required=data.qfrc_inverse.copy()
    if not np.isfinite(required).all():
        raise ValueError('Nonfinite native inverse dynamics')
    return required


def diagnose_bundle(bundle_path: Path,output_root: Path, *, phase_sidecar: Path | None = None) -> dict:
    bundle=load_runtime_bundle(bundle_path)
    phases={}
    if phase_sidecar is not None:
        phase_data=json.loads(Path(phase_sidecar).read_text())
        if phase_data['bundle_manifest_sha256']!=sha256(bundle.root/'manifest.json'):
            raise ValueError('Phase sidecar belongs to a different reference bundle')
        phases={row['reference_id']:row for row in phase_data['references']}
    output=writable_path(output_root)
    if output.exists():
        raise FileExistsError('Choose a new effort diagnostic directory')
    model=mujoco.MjModel.from_xml_path(str(bundle.model_path))
    if model.nv!=54 or model.nu!=48 or model.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Expected constrained 48-actuator human with free base')
    caps=np.asarray(bundle.body['effort'],dtype=float)
    if (not np.array_equal(model.actuator_trnid[:,0],np.arange(1,49))
            or not np.allclose(model.actuator_gear[:,0],1.)
            or not np.allclose(model.actuator_forcerange,np.column_stack((-caps,caps)))):
        raise ValueError('Native joint actuator mapping/caps differ from the controller contract')
    original_disable=int(model.opt.disableflags)
    original_enable=int(model.opt.enableflags)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    if hasattr(mujoco.mjtEnableBit,'mjENBL_INVDISCRETE'):
        model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
    output.mkdir(parents=True)
    rows=[]
    for clip in bundle.clips:
        with np.load(bundle.path(clip['master_path']),allow_pickle=False) as source:
            times,poses,velocities=(source[key].copy() for key in ('time_s','qpos','qvel'))
        fresh=tangent_velocities(model,times,poses)
        if not np.allclose(fresh,velocities,atol=1e-8,rtol=1e-8):
            raise ValueError('Stored velocities disagree with fresh reference differentiation')
        acceleration=np.gradient(velocities,times,axis=0,edge_order=2)
        data=mujoco.MjData(model)
        static_required=unconstrained_inverse(model,data,poses[0],np.zeros(model.nv),np.zeros(model.nv))
        static_edges,_,_=contact_edges(model,data)
        static_guard=support_effort_solution(static_required,static_edges,caps)
        needed,torques,residuals,loads,ratios,statuses,corner_points,friction=[],[],[],[],[],[],[],[]
        for pose,velocity,accel in zip(poses,velocities,acceleration,strict=True):
            required=unconstrained_inverse(model,data,pose,velocity,accel)
            matrix,points,mus=contact_edges(model,data)
            solution=support_effort_solution(required,matrix,caps)
            needed.append(required);torques.append(solution['joint_torque'])
            residuals.append(solution['base_residual']);loads.append(solution['edge_loads'])
            ratios.append(solution['minimum_peak_effort_ratio'] if solution['support_feasible'] else np.nan)
            statuses.append(solution['status']);corner_points.append(points);friction.append(mus)
        needed,torques,residuals,loads,ratios,corner_points,friction=map(np.asarray,
            (needed,torques,residuals,loads,ratios,corner_points,friction))
        statuses=np.asarray(statuses)
        feasible=statuses=='feasible'
        foot_forces=np.empty((len(times),8,3))
        for k in range(8):
            lam=loads[:,4*k:4*k+4]
            foot_forces[:,k]=np.column_stack((friction[:,k]*(lam[:,0]-lam[:,1]),
                friction[:,k]*(lam[:,2]-lam[:,3]),lam.sum(axis=1)))
        force_z=foot_forces[:,:,2]
        friction_use=np.divide(np.abs(foot_forces[:,:,:2]).sum(axis=2),friction*force_z,
            out=np.zeros_like(force_z),where=force_z>1e-8)
        worst=np.unravel_index(np.argmax(np.abs(torques)/caps),torques.shape)
        artifact=output/(clip['reference_id']+'_effort.npz')
        np.savez_compressed(artifact,time_s=times,qacc=acceleration,
            required_generalized_force=needed,required_joint_torque_with_support=torques,
            root_wrench_residual_generalized=residuals,contact_edge_loads=loads,
            hypothetical_corner_forces_w=foot_forces,projected_support_corners_w=corner_points,
            friction_mu=friction,friction_pyramid_utilization=friction_use,
            support_feasible=feasible,support_status=statuses,minimum_peak_effort_ratio=ratios,
            joint_names=np.asarray(bundle.joint_names),effort_caps=caps)
        finite=ratios[feasible]
        phase=phases.get(clip['reference_id'],clip)
        outward_end=phase.get('outward_end_s')
        phase_counts={}
        if outward_end is not None:
            if phase.get('master_sha256')!=clip['master_sha256']:
                raise ValueError('Phase master hash mismatch')
            masks={'first_100_ms':times<.1,'outward_after_100_ms':(times>=.1)&(times<outward_end),
                   'recovery':(times>=outward_end)&(times<clip['guard_time_s']),
                   'guard_hold':times>=clip['guard_time_s']}
            phase_counts={name:{'frames':int(mask.sum()),'support_infeasible_frames':int(np.sum(mask&~feasible)),
                               'over_cap_frames':int(np.sum(mask&feasible&(ratios>1.+1e-6)))}
                          for name,mask in masks.items()}
        rows.append({'family':clip['family'],'reference_id':clip['reference_id'],
            'master_sha256':clip['master_sha256'],'frames':len(times),
            'support_feasible_frames':int(feasible.sum()),
            'support_infeasible_frames':int((statuses=='support_infeasible').sum()),
            'solver_failure_frames':int(np.isin(statuses,['solver_failure','numerical_residual_failure']).sum()),
            'within_caps_frames':int(np.sum(feasible&(ratios<=1.+1e-6))),
            'over_cap_frames':int(np.sum(feasible&(ratios>1.+1e-6))),
            'maximum_minimax_effort_ratio':None if not len(finite) else float(finite.max()),
            'median_minimax_effort_ratio':None if not len(finite) else float(np.median(finite)),
            'maximum_base_force_residual_n':float(np.linalg.norm(residuals[:,:3],axis=1).max()),
            'maximum_base_moment_residual_nm':float(np.linalg.norm(residuals[:,3:],axis=1).max()),
            'maximum_friction_pyramid_utilization':float(friction_use.max()),
            'maximum_required_freebase_force_n':float(np.linalg.norm(needed[:,:3],axis=1).max()),
            'maximum_required_freebase_moment_nm':float(np.linalg.norm(needed[:,3:6],axis=1).max()),
            'worst_joint':bundle.joint_names[worst[1]],'worst_joint_time_s':float(times[worst[0]]),
            'worst_joint_required_nm':float(torques[worst]),'worst_joint_cap_nm':float(caps[worst[1]]),
            'guard_start_effort_ratio':None if not feasible[0] else float(ratios[0]),
            'static_guard_support_feasible':static_guard['support_feasible'],
            'static_guard_minimax_effort_ratio':static_guard['minimum_peak_effort_ratio'],
            'reference_start_root_linear_velocity_m_s':velocities[0,:3].tolist(),
            'reference_start_root_linear_acceleration_m_s2':acceleration[0,:3].tolist(),
            'reference_start_peak_joint_acceleration_rad_s2':float(abs(acceleration[0,6:]).max()),
            'worst_joint_scope':'Includes approximate contact allocation on support-infeasible frames; not a feasible required-actuator solution.',
            'phase_counts':phase_counts,
            'artifact':artifact.name,'artifact_sha256':sha256(artifact)})
    report={'schema':'human_wbc_offline_effort_diagnostics_v1','physics_executed':False,
        'qualified_for_training':False,'status':'diagnostic_only',
        'bundle_manifest_sha256':sha256(bundle.root/'manifest.json'),
        'model_sha256':sha256(bundle.model_path),'controller_sha256':bundle.controller_sha256(),
        'diagnostic_source_sha256':sha256(Path(__file__)),
        'mass_kg':float(model.body_mass.sum()),'gravity_w':model.opt.gravity.tolist(),
        'source_disableflags':original_disable,'source_enableflags':original_enable,
        'diagnostic_disableflags':int(model.opt.disableflags),'diagnostic_enableflags':int(model.opt.enableflags),
        'derivatives':'Native tangent qvel checked against master; qacc is per-clip second-order finite difference of qvel at 5 ms.',
        'support_assumption':'Eight declared foot-box corners projected to z=0; unilateral forces in conservative |fx|+|fy|<=mu*fz pyramids. Corners must be within [-1,3] mm of floor.',
        'freebase_equations':'Native required generalized force[:6] = contact Jacobian transpose times world forces; no base actuator or applied support wrench.',
        'objective':'Minimize peak absolute joint torque divided by its existing effort cap while satisfying all six free-base equations.',
        'solver_threads':1,
        'interpretation':'Feasibility is for this exact imposed path and idealized contact model only. Overload/infeasibility is diagnostic, not proof that no physical punch exists. No forces or feedforward are applied or added to references.',
        'families':rows}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
