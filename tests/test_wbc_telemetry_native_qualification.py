"""Offline qualification bookkeeping; native execution is a separate GPU job."""
from dataclasses import fields
from types import SimpleNamespace
import numpy as np
import torch
from human_punching.wbc.telemetry import native_qualification as q


def traces():
    return {key:np.zeros((5,2,3),np.float32) for key in ('qpos','qvel','ctrl','time','actuator_force','qfrc_actuator')}


def test_comparison_requires_bitwise_not_just_close_or_same_float_value():
    a=traces();b=traces()
    assert q.compare_rollouts(a,b)['passed']
    b['qpos'][1,0,0]=1e-8
    assert not q.compare_rollouts(a,b)['passed']
    b=traces();b['ctrl'][1,0,0]=-0.
    result=q.compare_rollouts(a,b)
    assert not result['passed'] and result['fields']['ctrl']['maximum_absolute_difference']==0


def test_live_byte_witness_detects_one_element_and_preserves_nan_payload():
    audit=q.NonmutationAudit()
    audit.arrays={'qpos':torch.tensor([[0.,float('nan')]]),'ctrl':torch.zeros(1,2)}
    audit.before={key:value.clone() for key,value in audit.arrays.items()}
    audit.bad={key:torch.tensor(False) for key in audit.arrays}
    audit.verify(None)
    assert audit.report()['passed']
    audit.capture(None);audit.arrays['ctrl'][0,0]=-0.
    audit.verify(None)
    assert audit.report()['mutated_fields']=={'qpos':False,'ctrl':True}
    assert audit.report()['checks']==2


def test_enabled_native_metric_is_sandwiched_by_witnesses_and_default_off_is_untouched():
    from human_punching.wbc.runtime.action import PunchJointPositionActionCfg
    old=PunchJointPositionActionCfg(entity_name='robot',actuator_names=('j',),command_name='motion',scale=.25)
    cfg=SimpleNamespace(actions={'joint_pos':old},metrics={'original':object()})
    assert q.instrument_cfg(cfg,enabled=False,auditor=q.NonmutationAudit(),recorded_states=[]) is cfg
    assert list(cfg.metrics)==['original'] and cfg.actions['joint_pos'] is old
    q.instrument_cfg(cfg,enabled=True,auditor=q.NonmutationAudit(),recorded_states=[])
    assert list(cfg.metrics)[:3]==['observer_nonmutation_before','human_native_substep_valid','observer_nonmutation_after']
    assert isinstance(cfg.actions['joint_pos'],q.AuditedActionCfg)
    assert {f.name:getattr(cfg.actions['joint_pos'],f.name) for f in fields(old)}==vars(old)


def test_cpu_contact_parity_uses_interval_start_anchor_and_ignores_initial_force_row(monkeypatch):
    monkeypatch.setattr(q,'ENVIRONMENTS',1)
    expected={'support_anchor_position_w':np.stack([np.full((2,3),i,dtype=float) for i in range(3)])}
    monkeypatch.setattr(q,'derive_kinematics',lambda *args:expected)
    seen=[]
    def classify(model,contact,anchors,bindings):
        seen.append(anchors.copy())
        return dict(support_contact=np.ones(2,bool),support_patch_contact=np.ones(2,bool),
                    forbidden_collision=False,human_g1_collision=False)
    monkeypatch.setattr(q,'classify_contacts',classify)
    raw=dict(qpos=np.zeros((3,1,55)),qvel=np.zeros((3,1,54)),contact_offsets=np.zeros(4,int),
        contact_worldid=np.zeros(0,int),contact_geom=np.zeros((0,2),int),contact_pos=np.zeros((0,3)),
        contact_dist=np.zeros(0),contact_force=np.zeros((0,6)))
    states=dict(support_anchor_position_w=expected['support_anchor_position_w'][1:,None],
        support_contact=np.ones((2,1,2),bool),support_patch_contact=np.ones((2,1,2),bool),
        forbidden_collision=np.zeros((2,1),bool),human_g1_collision=np.zeros((2,1),bool))
    result=q.offline_parity(None,raw,states)
    assert result['passed']
    assert [a[0,0] for a in seen]==[0.,1.]
    states['support_contact'][1,0,0]=False
    assert not q.offline_parity(None,raw,states)['passed']


def test_actual_warp_constraint_force_layout_is_nested_and_witness_reads_it(monkeypatch):
    from mujoco_warp._src.types import Data,Constraint
    import warp as wp
    assert 'efc' in Data.__annotations__ and 'efc_force' not in Data.__annotations__
    assert 'force' in Constraint.__annotations__
    monkeypatch.setattr(wp,'to_torch',lambda value:value)
    native=SimpleNamespace(**{key:torch.zeros(1,3) for key in q.PHYSICAL_FIELDS})
    contacts=SimpleNamespace(**{key:torch.zeros(1,3) for key in ('geom','worldid','pos','dist','dim','efc_address')})
    constraint=SimpleNamespace(force=torch.tensor([[3.,4.,5.]]))
    wp_data=SimpleNamespace(efc=constraint,contact=contacts,nacon=torch.zeros(1,dtype=torch.int32),overflow=torch.zeros(1,dtype=torch.int32))
    env=SimpleNamespace(sim=SimpleNamespace(data=native,wp_data=wp_data),device='cpu')
    audit=q.NonmutationAudit();audit.capture(env)
    assert audit.arrays['efc.force'] is constraint.force
    audit.verify(env);assert audit.report()['passed']
    constraint.force[0,1]=6.;audit.verify(env)
    assert audit.report()['mutated_fields']['efc.force']


def test_native_run_audits_reuse_nonzero_command_and_rotation_verifiers(monkeypatch):
    import mujoco
    seen = []
    model = object()
    monkeypatch.setattr(mujoco, 'MjModel', SimpleNamespace(from_binary_path=lambda path: seen.append(('model', path)) or model))
    monkeypatch.setattr(q, 'command_drive_audit', lambda native,bundle,kind:
        seen.append(('controller',native,bundle,kind)) or {'status':'passed'})
    monkeypatch.setattr(q, 'offline_audit', lambda m,raw,observed:
        seen.append(('parity',m,raw,observed)) or {'passed':True})
    bundle, raw, native = object(), object(), object()
    observed = {'root_position_w':np.zeros((1200,16,3))}
    controller, parity = q.audit_recorded_run(bundle,'punch',native,raw,observed,
        'compiled.mjb',policy_steps=300,enabled=True)
    assert controller == {'status':'passed'} and parity == {'passed':True}
    assert seen[0] == ('controller',native,bundle,'punch')
    assert seen[1] == ('model','compiled.mjb') and seen[2][0:3] == ('parity',model,raw)
    seen.clear()
    assert q.audit_recorded_run(bundle,'guard',native,raw,{},'absent.mjb',
        policy_steps=0,enabled=True) == ({'status':'unexecuted'},None)
    assert not seen
