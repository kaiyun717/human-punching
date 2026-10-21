"""DRAFT private GPU FK workspace. No integration, forward, collision or solve.

Only public mujoco_warp kinematics/com_pos/com_vel are dispatched. Full-model
state is copied into private Data; the shared model is read-only. Full Data with
zero contact/constraint capacities is the initial memory qualification candidate,
not a claim that its other solver workspaces are free. All quaternions are wxyz,
including MJWarp's internally named wp.quat arrays.
"""
from __future__ import annotations

from dataclasses import dataclass,fields,is_dataclass
from typing import Mapping
import numpy as np
import torch
from .streams import current_native_stream

KINEMATIC_DATA_FIELDS=(
    'qpos','qvel','mocap_pos','mocap_quat','xpos','xquat','xanchor','xaxis',
    'xmat','xipos','ximat','geom_xpos','geom_xmat','site_xpos','site_xmat',
    'subtree_com','cinert','cdof','cvel','cdof_dot')


@dataclass(frozen=True)
class EntityBindings:
    prefix: str
    joint_names: tuple[str,...]
    root_body: str='Hips'
    chest_body: str='Chest'
    glove_sites: tuple[str,str]=('LeftHand_center','RightHand_center')
    foot_bodies: tuple[str,str]=('LeftFoot','RightFoot')
    sole_sites: tuple[str,str]=('LeftFoot_sole','RightFoot_sole')
    anchor_local: tuple[float,float,float]=(0.,-.185,-.05)


@dataclass(frozen=True)
class ResolvedBindings:
    spec: EntityBindings
    nq: int
    nv: int
    nmocap: int
    body_ids: tuple[int,...]  # root, chest, left foot, right foot
    body_root_ids: tuple[int,...]
    site_ids: tuple[int,...]  # left glove, right glove, left sole, right sole
    site_body_ids: tuple[int,...]
    site_root_ids: tuple[int,...]
    joint_qpos_addresses: tuple[int,...]
    joint_dof_addresses: tuple[int,...]
    root_qpos_address: int


def resolve_bindings(model,spec: EntityBindings) -> ResolvedBindings:
    import mujoco
    if (not spec.joint_names or len(set(spec.joint_names))!=len(spec.joint_names)
            or len(spec.glove_sites)!=2 or len(spec.foot_bodies)!=2 or len(spec.sole_sites)!=2
            or np.asarray(spec.anchor_local).shape!=(3,) or not np.isfinite(spec.anchor_local).all()):
        raise ValueError('Distinct named scalar joints, two hands/feet and a finite material anchor required')
    def named(kind,name):
        try:return int(getattr(model,kind)(spec.prefix+name).id)
        except (KeyError,ValueError) as error:raise ValueError(f'Missing entity {kind}: {spec.prefix+name}') from error
    root=named('body',spec.root_body)
    if root<=0 or model.body_jntnum[root]!=1:
        raise ValueError('Human root must own one free joint')
    root_joint=int(model.body_jntadr[root])
    if model.jnt_type[root_joint]!=mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Human root must be free and unactuated')
    direct_joint=np.isin(model.actuator_trntype,(mujoco.mjtTrn.mjTRN_JOINT,mujoco.mjtTrn.mjTRN_JOINTINPARENT))
    if np.any(direct_joint&(model.actuator_trnid[:,0]==root_joint)):
        raise ValueError('Unexpected actuator attached to human root')
    def owned(body):
        cursor=int(body)
        while cursor!=0 and cursor!=root:cursor=int(model.body_parentid[cursor])
        if cursor!=root:raise ValueError('Named binding belongs to a different entity')
    bodies=(root,named('body',spec.chest_body),*(named('body',n) for n in spec.foot_bodies))
    sites=tuple(named('site',n) for n in (*spec.glove_sites,*spec.sole_sites))
    site_bodies=tuple(int(model.site_bodyid[i]) for i in sites)
    joints=tuple(named('joint',n) for n in spec.joint_names)
    for body in (*bodies,*site_bodies,*(int(model.jnt_bodyid[j]) for j in joints)):owned(body)
    if any(model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE) for j in joints):
        raise ValueError('Actuated joint bindings must be scalar hinge/slide joints')
    return ResolvedBindings(spec,int(model.nq),int(model.nv),int(model.nmocap),bodies,
        tuple(int(model.body_rootid[b]) for b in bodies),sites,site_bodies,
        tuple(int(model.body_rootid[b]) for b in site_bodies),
        tuple(int(model.jnt_qposadr[j]) for j in joints),tuple(int(model.jnt_dofadr[j]) for j in joints),
        int(model.jnt_qposadr[root_joint]))


def tensor_bindings(bindings: ResolvedBindings,device,dtype=torch.float32) -> dict[str,torch.Tensor]:
    result={name:torch.tensor(getattr(bindings,name),dtype=torch.long,device=device) for name in
        ('body_ids','body_root_ids','site_ids','site_body_ids','site_root_ids','joint_qpos_addresses','joint_dof_addresses')}
    result['anchor_local']=torch.tensor(bindings.spec.anchor_local,dtype=dtype,device=device)
    return result


def _quat_multiply(a,b):
    av,bv=a[...,1:],b[...,1:]
    return torch.cat((a[...,:1]*b[...,:1]-(av*bv).sum(-1,keepdim=True),
        a[...,:1]*bv+b[...,:1]*av+torch.linalg.cross(av,bv)),dim=-1)


def _quat_conjugate(q):
    return torch.cat((q[...,:1],-q[...,1:]),dim=-1)


OUTPUT_SHAPES={
    'root_position_w':(3,),'root_quat_wxyz':(4,),
    'root_linear_velocity_w':(3,),'root_angular_velocity_w':(3,),
    'chest_position_w':(3,),'chest_quat_wxyz':(4,),
    'chest_linear_velocity_w':(3,),'chest_angular_velocity_w':(3,),
    'foot_position_w':(2,3),'foot_quat_wxyz':(2,4),
    'foot_linear_velocity_w':(2,3),'foot_angular_velocity_w':(2,3),
    'glove_position_w':(2,3),'glove_quat_wxyz':(2,4),
    'glove_linear_velocity_w':(2,3),'glove_angular_velocity_w':(2,3),
    'sole_position_w':(2,3),'sole_quat_wxyz':(2,4),'sole_linear_velocity_w':(2,3),
    'support_anchor_position_w':(2,3),'support_anchor_velocity_w':(2,3),
    'glove_position_guard_frame':(2,3),'glove_quat_guard_frame':(2,4),
    'glove_velocity_guard_frame':(2,3),
}


def allocate_outputs(worlds,joints,device,dtype=torch.float32):
    result={name:torch.empty((worlds,*shape),device=device,dtype=dtype) for name,shape in OUTPUT_SHAPES.items()}
    result.update({name:torch.empty((worlds,joints),device=device,dtype=dtype) for name in ('joint_position','joint_velocity')})
    result['finite_kinematics']=torch.empty(worlds,dtype=torch.bool,device=device)
    return result


def extract_named_kinematics(arrays: Mapping[str,torch.Tensor],index: Mapping[str,torch.Tensor],outputs):
    """Tensor-only point transport, shared by CPU interface tests and GPU helper.

    cvel is [world angular, world linear at the tree's subtree COM], not body
    origin velocity. Site/body/material-point velocities explicitly shift origin.
    Return buffers are owned by the caller and never views of live simulation.
    """
    body=index['body_ids'];sites=index['site_ids'];site_bodies=index['site_body_ids']
    pos=arrays['xpos'].index_select(1,body);quat=arrays['xquat'].index_select(1,body)
    rotation=arrays['xmat'].index_select(1,body)
    cv=arrays['cvel'].index_select(1,body);omega=cv[...,:3]
    com=arrays['subtree_com'].index_select(1,index['body_root_ids'])
    velocity=cv[...,3:]+torch.linalg.cross(omega,pos-com)
    point=arrays['site_xpos'].index_select(1,sites)
    site_cv=arrays['cvel'].index_select(1,site_bodies)
    site_com=arrays['subtree_com'].index_select(1,index['site_root_ids'])
    point_velocity=site_cv[...,3:]+torch.linalg.cross(site_cv[...,:3],point-site_com)
    local=arrays['site_local_quat_wxyz'].index_select(1,sites)
    if local.shape[0]!=1 and local.shape[0]!=point.shape[0]:
        raise ValueError('Site quaternion model batches must be shared or one per world')
    point_quat=_quat_multiply(arrays['xquat'].index_select(1,site_bodies),local)
    anchor_offset=(rotation[:,2:]@index['anchor_local'].reshape(1,1,3,1)).squeeze(-1)
    anchors=pos[:,2:]+anchor_offset
    anchor_velocity=velocity[:,2:]+torch.linalg.cross(omega[:,2:],anchor_offset)
    chest_inverse=rotation[:,1].transpose(-2,-1)
    glove_offset=point[:,:2]-pos[:,1,None,:]
    relative_position=(chest_inverse[:,None]@glove_offset[...,None]).squeeze(-1)
    relative_velocity=(chest_inverse[:,None]@(point_velocity[:,:2]-velocity[:,1,None,:]
        -torch.linalg.cross(omega[:,1,None,:].expand_as(glove_offset),glove_offset))[...,None]).squeeze(-1)
    relative_quat=_quat_multiply(_quat_conjugate(quat[:,1,None,:]).expand(-1,2,-1),point_quat[:,:2])
    values={
        'root_position_w':pos[:,0],'root_quat_wxyz':quat[:,0],
        'root_linear_velocity_w':velocity[:,0],'root_angular_velocity_w':omega[:,0],
        'chest_position_w':pos[:,1],'chest_quat_wxyz':quat[:,1],
        'chest_linear_velocity_w':velocity[:,1],'chest_angular_velocity_w':omega[:,1],
        'foot_position_w':pos[:,2:],'foot_quat_wxyz':quat[:,2:],
        'foot_linear_velocity_w':velocity[:,2:],'foot_angular_velocity_w':omega[:,2:],
        'glove_position_w':point[:,:2],'glove_quat_wxyz':point_quat[:,:2],
        'glove_linear_velocity_w':point_velocity[:,:2],'glove_angular_velocity_w':site_cv[:,:2,:3],
        'sole_position_w':point[:,2:],'sole_quat_wxyz':point_quat[:,2:],
        'sole_linear_velocity_w':point_velocity[:,2:],
        'support_anchor_position_w':anchors,'support_anchor_velocity_w':anchor_velocity,
        'glove_position_guard_frame':relative_position,'glove_quat_guard_frame':relative_quat,
        'glove_velocity_guard_frame':relative_velocity,
        'joint_position':arrays['qpos'].index_select(1,index['joint_qpos_addresses']),
        'joint_velocity':arrays['qvel'].index_select(1,index['joint_dof_addresses']),
    }
    for name,value in values.items():outputs[name].copy_(value)
    finite=torch.isfinite(arrays['qpos']).all(-1)&torch.isfinite(arrays['qvel']).all(-1)
    for value in (point,velocity,relative_velocity,point_quat):
        finite=finite&torch.isfinite(value).flatten(1).all(-1)
    finite=finite&(torch.abs(torch.linalg.vector_norm(quat,dim=-1)-1.)<1e-3).all(-1)
    outputs['finite_kinematics'].copy_(finite)
    return outputs


def storage_ranges(value,path='root',seen=None):
    """Metadata-only memory ranges for alias checks; no GPU copy/synchronization."""
    seen=set() if seen is None else seen
    if id(value) in seen:return []
    seen.add(id(value))
    if isinstance(value,torch.Tensor):
        storage=value.untyped_storage();start=int(storage.data_ptr());size=storage.nbytes()
        return [(str(value.device),start,start+size,path)] if start and size else []
    if hasattr(value,'ptr') and hasattr(value,'capacity') and hasattr(value,'device'):
        start=int(value.ptr or 0);size=int(value.capacity)
        return [(str(value.device),start,start+size,path)] if start and size else []
    if is_dataclass(value):children=((f.name,getattr(value,f.name)) for f in fields(value))
    elif isinstance(value,Mapping):children=value.items()
    elif hasattr(value,'__dict__'):children=vars(value).items()
    elif isinstance(value,(list,tuple)):children=enumerate(value)
    else:return []
    return [r for key,child in children for r in storage_ranges(child,f'{path}.{key}',seen)]


def assert_storage_disjoint(private,live):
    for dev,start,end,path in storage_ranges(private):
        for other_dev,other_start,other_end,other_path in storage_ranges(live):
            if dev==other_dev and max(start,other_start)<min(end,other_end):
                raise ValueError(f'Private/live writable storage alias: {path} and {other_path}')


def allocated_bytes(value):
    ranges={(dev,start,end) for dev,start,end,_ in storage_ranges(value)}
    return sum(end-start for _,start,end in ranges)


class ScratchKinematics:
    """Reusable CUDA workspace, intended for the post-integration metric hook.

    Construction/metadata inspection are outside graph capture. update has no
    per-world Python loop, host tensor read, forward/contact solve or live write.
    Output storage is overwritten on the next update: an event accumulator must
    copy any previous endpoint it needs, as explicitly documented in its schema.
    """
    def __init__(self,cpu_model,warp_model,bindings: EntityBindings,worlds:int,device,*,live_data):
        import warp as wp
        import mujoco_warp as mjw
        self.device=torch.device(device)
        if self.device.type!='cuda' or not isinstance(worlds,int) or worlds<1:
            raise ValueError('Scratch GPU FK requires CUDA and a positive world count')
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        self.bindings=resolve_bindings(cpu_model,bindings);self.worlds=worlds
        self._wp,self._mjw,self._model=wp,mjw,warp_model
        self._indices=tensor_bindings(self.bindings,self.device)
        stream=current_native_stream(self.device)
        with wp.ScopedDevice(str(self.device)),wp.ScopedStream(stream):
            self._data=mjw.make_data(cpu_model,nworld=worlds,nconmax=0,nccdmax=0,njmax=0,
                njmax_nnz=0,naconmax=0,naccdmax=0)
        self._arrays={name:wp.to_torch(getattr(self._data,name)) for name in
            ('qpos','qvel','mocap_pos','mocap_quat','xpos','xquat','xmat','site_xpos','subtree_com','cvel')}
        self._arrays['site_local_quat_wxyz']=wp.to_torch(warp_model.site_quat)
        self.outputs=allocate_outputs(worlds,len(bindings.joint_names),self.device)
        assert_storage_disjoint(self._data,live_data)
        assert_storage_disjoint(self.outputs,live_data)
        required={name:getattr(self._data,name) for name in KINEMATIC_DATA_FIELDS}
        self.memory={'full_private_data_bytes':allocated_bytes(self._data),
            'kinematic_required_array_bytes':allocated_bytes(required),'private_output_bytes':allocated_bytes(self.outputs),
            'shared_model_excluded':True,'contact_capacity':int(self._data.naconmax),
            'constraint_capacity_per_world':int(self._data.njmax),'worlds':worlds}

    def update(self,qpos,qvel,*,mocap_pos=None,mocap_quat=None):
        expected=((qpos,(self.worlds,self.bindings.nq)),(qvel,(self.worlds,self.bindings.nv)))
        if self.bindings.nmocap:
            expected+=((mocap_pos,(self.worlds,self.bindings.nmocap,3)),
                       (mocap_quat,(self.worlds,self.bindings.nmocap,4)))
        for value,shape in expected:
            if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or value.dtype!=torch.float32 or value.device!=self.device:
                raise ValueError('Full-model FP32 state tensors on the declared CUDA device are required')
        self._arrays['qpos'].copy_(qpos);self._arrays['qvel'].copy_(qvel)
        if self.bindings.nmocap:
            self._arrays['mocap_pos'].copy_(mocap_pos);self._arrays['mocap_quat'].copy_(mocap_quat)
        stream=current_native_stream(self.device)
        with self._wp.ScopedStream(stream):
            self._mjw.kinematics(self._model,self._data)
            self._mjw.com_pos(self._model,self._data)
            self._mjw.com_vel(self._model,self._data)
        return extract_named_kinematics(self._arrays,self._indices,self.outputs)
