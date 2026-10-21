"""Bridge harness contracts and offline audits, without native integration."""
from types import SimpleNamespace
from contextlib import contextmanager
import mujoco
import numpy as np
import pytest
import torch

from human_punching.wbc.cycle_trace import TraceBindings, derive_kinematics
from human_punching.wbc.telemetry import cycle_qualification as q
from human_punching.wbc.telemetry import snapshot_providers as p


def test_native_stream_orders_producers_and_cleanup_even_on_failure(monkeypatch):
    import warp as wp
    events=[]
    native=SimpleNamespace(wait_stream=lambda other:events.append(('native_waits',other)))
    caller=SimpleNamespace(wait_stream=lambda other:events.append(('caller_waits',other)))
    monkeypatch.setattr(wp,'get_stream',lambda device:SimpleNamespace(cuda_stream=123))
    monkeypatch.setattr(torch.cuda,'ExternalStream',lambda pointer,device:native)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:caller)
    @contextmanager
    def scoped(stream):
        assert stream is native;events.append('enter')
        try:yield
        finally:events.append('leave')
    monkeypatch.setattr(torch.cuda,'stream',scoped)
    with pytest.raises(RuntimeError,match='test failure'):
        with q.native_stream_scope():
            events.append('native_work');raise RuntimeError('test failure')
    assert events==[('native_waits',caller),'enter','native_work','leave',('caller_waits',native)]


def test_declared_nominal_is_master_guard_fk_without_assumed_load(task10_common_bundle, monkeypatch):
    bundle = task10_common_bundle
    model = mujoco.MjModel.from_xml_path(str(bundle.model_path))
    for name in ('mj_step', 'mj_forward', 'mj_collision'):
        monkeypatch.setattr(mujoco, name, lambda *a: pytest.fail('No dynamics or force solve'))
    binding = TraceBindings(prefix='', floor_geoms=('floor',), human_geom_prefix='physical/')
    table = p.declared_guard_table(model, bundle, bundle.clips, bindings=binding)
    assert table['joint_position'].shape == (4, 48)
    assert not table['support_contact'].any() and not table['support_patch_contact'].any()
    assert not table['joint_velocity'].any() and not table['effort'].any()
    with np.load(bundle.path(bundle.clips[0]['master_path'])) as master:
        nominal = derive_kinematics(model, master['qpos'][:1], np.zeros((1, 54)), binding)
        np.testing.assert_array_equal(table['glove_position_w'][0], nominal['glove_position_w'][0])


def test_providers_translate_declared_world_fields_and_assign_reset_identity(task10_common_bundle, monkeypatch):
    bundle = task10_common_bundle; n = 2
    model = mujoco.MjModel.from_xml_path(str(bundle.model_path))
    table = p.declared_guard_table(model, bundle, bundle.clips,
        bindings=TraceBindings(prefix='', floor_geoms=('floor',), human_geom_prefix='physical/'))
    monkeypatch.setattr(p, 'declared_guard_table', lambda *a: table)
    origins = torch.tensor([[0., 0., 0.], [4., -3., 0.]])
    env = SimpleNamespace(device='cpu', num_envs=n, sim=SimpleNamespace(mj_model=model), scene=SimpleNamespace(env_origins=origins))
    cmd = SimpleNamespace(cfg=SimpleNamespace(bundle_path=bundle.root), clips=bundle.clips,
        num_envs=n, device='cpu', _env=env, trajectory_ids=torch.tensor([0, 1]),
        targets=torch.tensor([c['target_w'] for c in bundle.clips]),
        ttcs=torch.tensor([c['requested_ttc_s'] for c in bundle.clips]),
        family_ids=torch.tensor([('jab','cross','left_hook','right_hook').index(c['family']) for c in bundle.clips]),
        g1_context=torch.tensor([c['launch_features'] for c in bundle.clips]))
    env.command_manager = SimpleNamespace(get_term=lambda _: cmd)
    observer = SimpleNamespace(reset_count=torch.tensor([1, 3]))
    monkeypatch.setattr(p, 'get_observer', lambda _: observer)
    providers = p.make_snapshot_providers(env, engineering_admitted=True)
    nominal = providers['nominal_provider']()
    torch.testing.assert_close(nominal['root_position_w'][1], torch.tensor(table['root_position_w'][1], dtype=torch.float32)+origins[1])
    torch.testing.assert_close(nominal['glove_position_guard_frame'][1], torch.tensor(table['glove_position_guard_frame'][1], dtype=torch.float32))
    assert providers['request_ids']().tolist() == [2, 7]
    packet = providers['packet_provider']()
    torch.testing.assert_close(packet['target_w'], cmd.targets[cmd.trajectory_ids]+origins)
    assert not nominal['support_contact'].any()
    with pytest.raises(ValueError, match='explicit engineering'):
        p.make_snapshot_providers(env, engineering_admitted=False)


def test_probe_config_exercises_auto_resets_and_new_task_rewards(task10_common_bundle):
    cfg = q.build_cfg(task10_common_bundle, q.QualificationRecorder(), readiness=q.ReadinessConfig())
    assert cfg.auto_reset and not cfg.events
    assert set(cfg.terminations) == {'cycle_complete'}
    assert isinstance(cfg.commands['motion'], q.QualificationCommandCfg)
    assert cfg.commands['motion'].require_physical_qualification is False
    assert isinstance(cfg.actions['joint_pos'], q.QualificationActionCfg)
    assert cfg.observations['actor'].terms['punch_condition'].func is q.cycle_condition
    assert 'human_guard_quiet' in cfg.rewards and 'timed_target' not in cfg.rewards
    assert cfg.sim.mujoco.timestep == .005 and cfg.decimation == 4


def fixture(monkeypatch):
    n, count = 2, 4; monkeypatch.setattr(q, 'ENVIRONMENTS', n)
    a = np.zeros((count, n, 55)); a[..., 3] = 1
    quaternion = a[..., 3:7]
    pre = dict(qpos=a.copy(), qvel=np.zeros((count,n,54)), time=np.arange(count)[:,None]*.005+np.zeros((count,n)),
        qref=np.zeros((count,n,48)), vref=np.zeros((count,n,48)), feedforward=np.zeros((count,n,48)),
        guard=np.zeros((count,n,48)), future12=np.zeros((count,n,48)), future24=np.zeros((count,n,48)),
        reference_tick=np.zeros((count,n),int), launched=np.zeros((count,n),bool),
        launch_tick=np.full((count,n),-1), episode_tick=np.broadcast_to(np.arange(count)[:,None],(count,n)))
    native = dict(qpos=a, qvel=pre['qvel'], raw_action=np.zeros((count,n,48)),
        joint_position_target=pre['qref'].copy(), joint_velocity_target=pre['vref'].copy(),
        joint_effort_target=pre['feedforward'].copy(), ctrl=pre['qref'].copy(), actuator_force=pre['qref'].copy(),
        time=pre['time']+.005, sim_step_index=np.arange(1,count+1), policy_step_index=np.zeros(count,int),
        qfrc_applied=np.zeros((count,n,54)), xfrc_applied=np.zeros((count,n,2,6)),
        qfrc_actuator=np.zeros((count,n,54)), overflow=np.zeros((count,n)))
    endpoint={'root_position_w':a[...,:3], 'root_quat_wxyz':quaternion,
        'ready_endpoint':np.zeros((count,n),bool),'quiet_intervals':np.zeros((count,n),int),
        'support_contact':np.zeros((count,n,2),bool)}
    monkeypatch.setattr(q,'derive_kinematics',lambda model,pos,vel:{'root_position_w':pos[...,:3], 'root_quat_wxyz':pos[...,3:7]})
    policy=dict(qpos=a[-1:], root_position=a[-1:,:,:3], root_quaternion=quaternion[-1:], reward=np.zeros((1,n)),
        observation_finite=np.ones((1,n),bool), launched=np.zeros((1,n),bool), pending_reset=np.zeros((1,n),bool),
        request_id=np.zeros((1,n),int), target=np.zeros((1,n,3)), actor_context=np.zeros((1,n,77)),
        launch_tick=np.full((1,n),-1),episode_tick=np.full((1,n),4))
    bundle=SimpleNamespace(body={'kp':[1.]*48,'kd':[1.]*48,'effort':[10.]*48,'joint_limits_rad':[[-1.,1.]]*48})
    model=SimpleNamespace(opt=SimpleNamespace(timestep=.005))
    def audit():return q.audit_arrays(native,pre,endpoint,policy,bundle,model,steps=1,writes=np.array([2,2]),resets=np.array([2,2]))
    return native, pre, policy, audit


def test_unlaunched_audit_has_no_fabricated_launch_qualification(monkeypatch):
    _,pre,_,audit=fixture(monkeypatch)
    result=audit();assert result['status']=='passed'
    assert result['launch_qualification']=='not_tested_unready'
    pre['launch_tick'][0,0]=0
    assert not audit()['checks']['no_invented_launch']


def test_audit_detects_hidden_future_progression_and_cancelling_wrenches(monkeypatch):
    native,pre,policy,audit=fixture(monkeypatch)
    pre['future12'][2,0,0]=.01
    assert not audit()['checks']['future_held_before_launch']
    native['xfrc_applied'][1,0,0,0]=1
    native['xfrc_applied'][1,0,1,0]=-1
    assert not audit()['checks']['no_external_forces']
    policy['root_position']=policy['root_position'].copy();policy['root_position'][0,0,0]+=.005
    assert not audit()['checks']['fresh_actor_anchor']
