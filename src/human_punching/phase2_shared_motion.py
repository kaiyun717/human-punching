"""Explicit source-relative root/foot geometry and bounded lower-body IK.

All arrays are offline reference geometry. No simulation state, force, actor,
controller history or source artifact is changed. A physical execution remains
required even when the kinematic construction succeeds.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .phase2_shared_reference_shape_v3 import resolve_shared_joints

VERSION="task10b_shared_source_relative_pelvis_and_per_foot_motion_v1"
SIDES=("Right","Left")


@dataclass(frozen=True)
class LowerMotionSettings:
    maximum_iterations:int=40
    maximum_iteration_joint_delta_rad:float=.025
    maximum_joint_difference_from_source_rad:float=.60
    position_tolerance_m:float=.0005
    orientation_tolerance_rad:float=.001

    def __post_init__(self):
        values=asdict(self)
        if (type(self.maximum_iterations) is not int or not 1<=self.maximum_iterations<=40
                or any(isinstance(v,bool) or not np.isfinite(v) or v<=0 for k,v in values.items()
                    if k!="maximum_iterations")
                or self.maximum_iteration_joint_delta_rad>.025
                or self.maximum_joint_difference_from_source_rad>.60
                or self.position_tolerance_m>.0005 or self.orientation_tolerance_rad>.001):
            raise ValueError("Explicit bounded lower-body reference IK settings required")


class LowerMotionRejection(ValueError):
    def __init__(self,reason,**details):
        self.disposition=dict(reason=reason,motion_version=VERSION,physical_qualification=False,**details)
        super().__init__(reason)


def _multiply(a,b):
    return np.r_[a[0]*b[0]-a[1:]@b[1:],a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]


def _matrix(q):
    # Normalize only the algebraic rotation matrix, never the retained q knots
    # or any physical state. Native float32 anchor quaternions carry roundoff.
    out=np.empty(9);mujoco.mju_quat2Mat(out,q/np.linalg.norm(q));return out.reshape(3,3)


def relative_motion_geometry(source_root,anchor_root,source_foot_positions,source_foot_rotations,
                             anchor_foot_positions,anchor_foot_rotations):
    """Pure reference frame map; preserve every relative root/foot motion."""
    root,anchor,p,R,p0,R0=(np.asarray(x,dtype=np.float64) for x in
        (source_root,anchor_root,source_foot_positions,source_foot_rotations,
         anchor_foot_positions,anchor_foot_rotations))
    n=root.shape[0] if root.ndim==2 else 0
    if (root.shape!=(n,7) or not 3<=n<=6001 or anchor.shape!=(7,)
            or p.shape!=(n,2,3) or R.shape!=(n,2,3,3) or p0.shape!=(2,3) or R0.shape!=(2,3,3)
            or any(not np.isfinite(x).all() for x in (root,anchor,p,R,p0,R0))
            or not np.allclose(np.linalg.norm(root[:,3:7],axis=1),1,rtol=0,atol=1.e-7)
            or not np.isclose(np.linalg.norm(anchor[3:7]),1,rtol=0,atol=1.e-7)
            or not np.allclose(R.swapaxes(-1,-2)@R,np.eye(3),rtol=0,atol=1.e-7)
            or not np.allclose(R0.swapaxes(-1,-2)@R0,np.eye(3),rtol=0,atol=1.e-7)
            or np.any(np.linalg.det(R)<0) or np.any(np.linalg.det(R0)<0)):
        raise ValueError("Finite accepted root quaternions and explicit two-foot rigid poses required")
    inverse=root[0,3:7]*np.array([1,-1,-1,-1])/(root[0,3:7]@root[0,3:7])
    transform=_multiply(anchor[3:7],inverse);rotation=_matrix(transform)
    mapped=np.column_stack((anchor[:3]+(root[:,:3]-root[0,:3])@rotation.T,
        np.array([_multiply(transform,value) for value in root[:,3:7]])))
    same=np.all(root==root[0],axis=1);mapped[same]=anchor
    target_p=p0+(p-p[:1])@rotation.T
    target_R=np.einsum('sij,sjk,nskl->nsil',R0,R[0].swapaxes(-1,-2),R)
    # Explicit endpoint numbers avoid cancellation at the unchanged source hold.
    foot_same=np.all(p==p[:1],axis=(-1,-2))&np.all(R==R[:1],axis=(-1,-2,-3))
    target_p[foot_same]=p0;target_R[foot_same]=R0
    return mapped,target_p,target_R


def foot_pose_paths(model,poses,*,checkpoint=None):
    q=np.asarray(poses,float)
    if q.ndim!=2 or q.shape[1]!=55 or not np.isfinite(q).all():raise ValueError("Complete finite reference poses required")
    resolve_shared_joints(model)
    sites=np.array([model.site(side+'Foot_sole_0').id for side in SIDES])
    data=mujoco.MjData(model);p=[];R=[]
    for pose in q:
        if checkpoint is not None:checkpoint()
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        p.append(data.site_xpos[sites].copy());R.append(data.site_xmat[sites].reshape(2,3,3).copy())
    return np.array(p),np.array(R)


def _pose_errors_and_jacobian(model,data,pose,sites,va,target_p,target_R):
    data.qpos[:]=pose;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
    rows=[];rhs=[];positions=[];orientations=[]
    for side,site in enumerate(sites):
        dp=target_p[side]-data.site_xpos[site]
        dr=Rotation.from_matrix(target_R[side]@data.site_xmat[site].reshape(3,3).T).as_rotvec()
        jp=np.zeros((3,model.nv));jr=np.zeros((3,model.nv));mujoco.mj_jacSite(model,data,jp,jr,int(site))
        rows.extend((jp[:,va],.1*jr[:,va]));rhs.extend((dp,.1*dr))
        positions.append(float(np.linalg.norm(dp)));orientations.append(float(np.linalg.norm(dr)))
    return np.vstack(rows),np.concatenate(rhs),positions,orientations


def solve_lower_pose(model,pose,source_pose,target_positions,target_rotations,*,settings=LowerMotionSettings(),data=None):
    """Change lower14 reference joints only; reject instead of clipping outputs."""
    if not isinstance(settings,LowerMotionSettings):raise ValueError("Versioned lower-motion settings required")
    mask=resolve_shared_joints(model);ids=mask['lower_joint_ids'];qa=mask['lower_qpos'];va=mask['lower_dof']
    original=np.asarray(source_pose,float);candidate=np.asarray(pose,float).copy()
    target_p=np.asarray(target_positions,float);target_R=np.asarray(target_rotations,float)
    if (candidate.shape!=(55,) or original.shape!=(55,) or target_p.shape!=(2,3) or target_R.shape!=(2,3,3)
            or any(not np.isfinite(x).all() for x in (candidate,original,target_p,target_R))
            or not np.allclose(target_R.swapaxes(-1,-2)@target_R,np.eye(3),atol=1.e-7,rtol=0)
            or np.any(np.linalg.det(target_R)<0)):
        raise ValueError("Finite accepted pose/source and two-foot rigid targets required")
    low=np.maximum(model.jnt_range[ids,0],original[qa]-settings.maximum_joint_difference_from_source_rad)
    high=np.minimum(model.jnt_range[ids,1],original[qa]+settings.maximum_joint_difference_from_source_rad)
    if np.any(low>high) or np.any(candidate[qa]<low) or np.any(candidate[qa]>high):
        raise LowerMotionRejection('lower_seed_outside_current_physical_or_recipe_box')
    fixed=np.ones(55,bool);fixed[qa]=False;before=candidate.copy()
    sites=np.array([model.site(side+'Foot_sole_0').id for side in SIDES]);data=data if data is not None else mujoco.MjData(model)
    for iteration in range(settings.maximum_iterations+1):
        J,error,position,orientation=_pose_errors_and_jacobian(model,data,candidate,sites,va,target_p,target_R)
        if max(position)<=settings.position_tolerance_m and max(orientation)<=settings.orientation_tolerance_rad:
            if not np.array_equal(candidate[fixed],before[fixed]):raise AssertionError("Lower IK altered root/upper geometry")
            if np.any(candidate[qa]<model.jnt_range[ids,0]) or np.any(candidate[qa]>model.jnt_range[ids,1]):
                raise LowerMotionRejection('lower_solution_outside_physical_range')
            return candidate,dict(iterations=iteration,maximum_position_error_m=max(position),
                maximum_orientation_error_rad=max(orientation),maximum_joint_difference_from_source_rad=float(abs(candidate[qa]-original[qa]).max()),
                root_and_upper_unchanged=True,physical_joint_ranges_enlarged=False,reference_only=True)
        if iteration==settings.maximum_iterations:break
        lo=np.maximum(low-candidate[qa],-settings.maximum_iteration_joint_delta_rad)
        hi=np.minimum(high-candidate[qa],settings.maximum_iteration_joint_delta_rad)
        free=hi-lo>1.e-12
        if not free.any():break
        augmented=np.vstack((J,.002*np.eye(len(qa))));rhs=np.r_[error,-.002*(candidate[qa]-original[qa])]
        result=lsq_linear(augmented[:,free],rhs,bounds=(lo[free],hi[free]),method='bvls',tol=1.e-10,max_iter=60)
        if not result.success or not np.isfinite(result.x).all():break
        candidate[qa[free]]+=result.x
        if np.any(candidate[qa]<low) or np.any(candidate[qa]>high):
            raise LowerMotionRejection('lower_numerical_step_outside_box',iteration=iteration)
    raise LowerMotionRejection('bounded_lower_foot_motion_unsolved',iterations=iteration,
        maximum_position_error_m=max(position),maximum_orientation_error_rad=max(orientation),settings=asdict(settings))
