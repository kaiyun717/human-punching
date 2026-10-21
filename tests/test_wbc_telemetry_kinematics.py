"""CPU interface/origin/ownership tests, not GPU kernel qualification."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
import torch
from human_punching.wbc.telemetry import kinematics as draft


XML='''<mujoco><worldbody>
<body name="other" pos="2 0 1"><freejoint name="other_free"/><geom type="sphere" size=".1" mass="1"/></body>
<body name="robot/Hips" pos="0 0 1"><freejoint name="robot/root"/><geom type="sphere" size=".1" mass="2"/>
 <body name="robot/Chest" pos="0 0 .3"><joint name="robot/torso" type="hinge" axis="0 0 1"/><geom type="sphere" size=".1" mass="1"/>
  <body name="robot/LeftHand" pos=".4 .2 0"><geom type="sphere" size=".05" mass=".1"/><site name="robot/LeftHand_center" pos=".05 0 0" quat=".9238795325 .3826834324 0 0"/></body>
  <body name="robot/RightHand" pos=".4 -.2 0"><geom type="sphere" size=".05" mass=".1"/><site name="robot/RightHand_center" pos=".05 0 0"/></body>
 </body>
 <body name="robot/LeftFoot" pos="0 .15 -.8"><geom type="box" size=".1 .2 .05" mass="1"/><site name="robot/LeftFoot_sole" pos="0 -.1 -.05"/></body>
 <body name="robot/RightFoot" pos="0 -.15 -.8"><geom type="box" size=".1 .2 .05" mass="1"/><site name="robot/RightFoot_sole" pos="0 -.1 -.05"/></body>
</body></worldbody></mujoco>'''


def fixture():
    model=mujoco.MjModel.from_xml_string(XML)
    spec=draft.EntityBindings('robot/',('torso',));bindings=draft.resolve_bindings(model,spec)
    rng=np.random.default_rng(123);worlds=7
    qpos=np.tile(model.qpos0,(worlds,1));qvel=rng.normal(0,.7,(worlds,model.nv))
    address=bindings.root_qpos_address
    qpos[:,address:address+3]+=rng.normal(0,2.,(worlds,3))
    quats=rng.normal(size=(worlds,4));quats/=np.linalg.norm(quats,axis=1,keepdims=True)
    qpos[:,address+3:address+7]=quats;qpos[:,bindings.joint_qpos_addresses[0]]=np.linspace(-.7,.7,worlds)
    arrays={name:[] for name in ('xpos','xquat','xmat','site_xpos','subtree_com','cvel')}
    for q,v in zip(qpos,qvel,strict=True):
        data=mujoco.MjData(model);data.qpos[:]=q;data.qvel[:]=v
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_comVel(model,data)
        for name in arrays:arrays[name].append(getattr(data,name).copy())
    tensors={name:torch.tensor(np.asarray(value)) for name,value in arrays.items()}
    tensors['xmat']=tensors['xmat'].reshape(worlds,model.nbody,3,3)
    tensors.update(qpos=torch.tensor(qpos),qvel=torch.tensor(qvel),site_local_quat_wxyz=torch.tensor(model.site_quat[None].copy()))
    return model,bindings,tensors


def test_named_ownership_addresses_are_not_standalone_slices():
    model,bindings,arrays=fixture()
    assert bindings.root_qpos_address==7
    assert bindings.joint_qpos_addresses==(14,) and bindings.joint_dof_addresses==(12,)
    outputs=draft.allocate_outputs(7,1,'cpu',torch.float64)
    draft.extract_named_kinematics(arrays,draft.tensor_bindings(bindings,'cpu',torch.float64),outputs)
    np.testing.assert_array_equal(outputs['joint_position'],arrays['qpos'][:,[14]])
    np.testing.assert_array_equal(outputs['joint_velocity'],arrays['qvel'][:,[12]])
    with pytest.raises(ValueError,match='Missing entity'):
        draft.resolve_bindings(model,draft.EntityBindings('missing/',('torso',)))
    with pytest.raises(ValueError,match='different entity'):
        draft.resolve_bindings(model,draft.EntityBindings('',('other_free',),root_body='robot/Hips',
            chest_body='robot/Chest',glove_sites=('robot/LeftHand_center','robot/RightHand_center'),
            foot_bodies=('robot/LeftFoot','robot/RightFoot'),sole_sites=('robot/LeftFoot_sole','robot/RightFoot_sole')))


def test_com_origin_transport_and_rotating_chest_match_independent_native_jacobians(monkeypatch):
    for name in ('mj_forward','mj_step','mj_collision'):
        monkeypatch.setattr(mujoco,name,lambda *a:(_ for _ in ()).throw(AssertionError('No dynamics/contact solving')))
    model,bindings,arrays=fixture();outputs=draft.allocate_outputs(7,1,'cpu',torch.float64)
    draft.extract_named_kinematics(arrays,draft.tensor_bindings(bindings,'cpu',torch.float64),outputs)
    jp=np.empty((3,model.nv));jr=np.empty_like(jp)
    for k in range(7):
        data=mujoco.MjData(model);data.qpos[:]=arrays['qpos'][k];data.qvel[:]=arrays['qvel'][k]
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        velocity=data.qvel;root,chest,left_foot,right_foot=bindings.body_ids
        for key,body in (('root',root),('chest',chest)):
            mujoco.mj_jacBody(model,data,jp,jr,body)
            np.testing.assert_allclose(outputs[key+'_linear_velocity_w'][k],jp@velocity,atol=1e-12)
            np.testing.assert_allclose(outputs[key+'_angular_velocity_w'][k],jr@velocity,atol=1e-12)
        mujoco.mj_jacBody(model,data,jp,jr,chest);chest_v=jp@velocity;chest_w=jr@velocity
        rotation=data.xmat[chest].reshape(3,3)
        for hand,site in enumerate(bindings.site_ids[:2]):
            mujoco.mj_jacSite(model,data,jp,jr,site);offset=data.site_xpos[site]-data.xpos[chest]
            np.testing.assert_allclose(outputs['glove_linear_velocity_w'][k,hand],jp@velocity,atol=1e-12)
            np.testing.assert_allclose(outputs['glove_angular_velocity_w'][k,hand],jr@velocity,atol=1e-12)
            np.testing.assert_allclose(outputs['glove_velocity_guard_frame'][k,hand],
                rotation.T@(jp@velocity-chest_v-np.cross(chest_w,offset)),atol=1e-12)
            quaternion=np.empty(4);mujoco.mju_mat2Quat(quaternion,(rotation.T@data.site_xmat[site].reshape(3,3)).ravel())
            measured=outputs['glove_quat_guard_frame'][k,hand].numpy()
            assert abs(abs(np.dot(measured,quaternion))-1)<1e-12
        for side,body in enumerate((left_foot,right_foot)):
            point=data.xpos[body]+data.xmat[body].reshape(3,3)@bindings.spec.anchor_local
            mujoco.mj_jac(model,data,jp,jr,point,body)
            np.testing.assert_allclose(outputs['support_anchor_position_w'][k,side],point,atol=1e-12)
            np.testing.assert_allclose(outputs['support_anchor_velocity_w'][k,side],jp@velocity,atol=1e-12)
    assert outputs['finite_kinematics'].all()


def test_private_outputs_nonalias_input_and_storage_overlap_fails_closed():
    _,bindings,arrays=fixture();original={k:v.clone() for k,v in arrays.items()}
    outputs=draft.allocate_outputs(7,1,'cpu',torch.float64)
    draft.extract_named_kinematics(arrays,draft.tensor_bindings(bindings,'cpu',torch.float64),outputs)
    draft.assert_storage_disjoint(outputs,arrays)
    outputs['glove_position_w'].fill_(999)
    for name,value in arrays.items():assert torch.equal(value,original[name])
    live=SimpleNamespace(qpos=arrays['qpos'],contact=SimpleNamespace(force=torch.ones(20,6)))
    independent=SimpleNamespace(qpos=arrays['qpos'].clone(),contact=SimpleNamespace(force=torch.ones(20,6)))
    draft.assert_storage_disjoint(independent,live)
    independent.contact.force=live.contact.force[2:5]
    with pytest.raises(ValueError,match='writable storage alias'):draft.assert_storage_disjoint(independent,live)
    assert draft.allocated_bytes(outputs)>0


def test_nonfinite_state_flag_and_public_kernel_surface():
    _,bindings,arrays=fixture();outputs=draft.allocate_outputs(7,1,'cpu',torch.float64)
    arrays['qvel'][3,0]=float('nan')
    draft.extract_named_kinematics(arrays,draft.tensor_bindings(bindings,'cpu',torch.float64),outputs)
    assert outputs['finite_kinematics'].tolist()==[True,True,True,False,True,True,True]
    tree=ast.parse(Path(draft.__file__).read_text())
    calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
    assert not set(calls)&{'forward','step','step1','step2','collision','make_constraint','solve','mj_forward','mj_step'}
    assert all(calls.count(name)==1 for name in ('kinematics','com_pos','com_vel'))
