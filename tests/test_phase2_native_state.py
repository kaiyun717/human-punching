"""Full-state copy/restore tests with CPU tensor storage; no physics rollouts."""
from dataclasses import replace
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from human_punching.phase2_native_state import (
    STATIC_FIELDS,NativeComparisonSnapshot,SPEC,model_inventory,model_identity,
    copy_native_integration,write_native_integration,capture_comparison,restore_comparison,
    save_snapshot,load_snapshot,native_parity,
)
from human_punching.wbc.provenance import PROJECT_ROOT


@pytest.fixture
def model():
    # Compiling an existing model and copying MjData performs no integration.
    return mujoco.MjModel.from_xml_path(str(PROJECT_ROOT/'outputs/four-punch-milestone/jab/model/human.xml'))


def native_data(model,n=2):
    q=torch.as_tensor(model.qpos0.copy(),dtype=torch.float32).repeat(n,1)
    return SimpleNamespace(time=torch.full((n,),1.25),qpos=q,qvel=torch.arange(n*54,dtype=torch.float32).reshape(n,54)/1024,
        qacc_warmstart=torch.full((n,54),.125),ctrl=torch.full((n,48),.25),qfrc_applied=torch.full((n,54),.5),
        xfrc_applied=torch.full((n,24,6),.75),act=torch.empty(n,0),history=torch.empty(n,0),
        eq_active=torch.empty(n,0,dtype=torch.bool),mocap_pos=torch.empty(n,0,3),
        mocap_quat=torch.empty(n,0,4),userdata=torch.empty(n,0),plugin_state=torch.empty(n,0))


class FakeEnv(SimpleNamespace):
    @property
    def duration(self):return self.library.durations[self.ids_ref]


def fake_env(model,n=2,*,source=True):
    env=FakeEnv(**{key:None for key in STATIC_FIELDS})
    env.model=model;env.n=n;env.dt=.005;env.decimation=4;env.solver='accepted';env.device='cpu'
    env.ids=np.arange(1,49);env.qadr=np.arange(7,55);env.vadr=np.arange(6,54)
    for name in ('kp','kd','low','high','joint_low','joint_high'):setattr(env,name,torch.ones(48))
    env.torque_limits=np.ones((48,2))
    data=native_data(model,n)
    def forward():data.qacc_warmstart.fill_(999.)
    env.sim=SimpleNamespace(data=data,forward=forward,_sensor_context=None,_expanded_fields=set(),
        _world_to_variant={},_per_world_default_fields=set(),nan_guard=SimpleNamespace(enabled=False))
    env.library=SimpleNamespace(rows=[dict(reference_sha256='1'*64),dict(reference_sha256='2'*64)],
        durations=torch.tensor([2.,3.]))
    env.ids_ref=torch.zeros(n,dtype=torch.long);env.ticks=torch.full((n,),12 if source else 0,dtype=torch.long)
    env.last_action=torch.full((n,48),.125 if source else 0.);env.last_support=torch.ones(n,2,dtype=torch.bool)
    env.initial_up=torch.tensor([[0.,0.,1.]]).repeat(n,1);env.reset_counts=torch.ones(n,dtype=torch.long)
    env.rejected=torch.zeros(n,dtype=torch.bool);env.launched=torch.zeros(n,dtype=torch.bool)
    env.completed=env.fallen=0;env.physics_steps_executed=8 if source else 0;env.actor_updates_executed=2 if source else 0
    env.hard_faults=[]
    if source:env._last_physical_fallen=torch.zeros(n,dtype=torch.bool)
    return env


def snapshot(env):
    return capture_comparison(env,held_raw_action=env.last_action.numpy(),reference_phase='preparation',
        support_interval_end_s=env.sim.data.time.numpy())


def test_accepted_complete_state_inventory_and_model_identity_include_solver(model):
    report=model_inventory(model)
    assert report['native_state_spec']==16383 and report['native_state_size']==410
    assert report['zero_sized_fields_checked']
    clone=mujoco.MjModel.from_xml_path(str(PROJECT_ROOT/'outputs/four-punch-milestone/jab/model/human.xml'))
    assert model_identity(model)==model_identity(clone)
    clone.opt.iterations+=1
    assert model_identity(model)!=model_identity(clone)


@pytest.mark.parametrize('field',['na','nhistory','neq','nmocap','nuserdata','nplugin','npluginstate'])
def test_nonempty_unimplemented_native_fields_fail_closed(model,field):
    proxy=SimpleNamespace(**{key:getattr(model,key) for key in ('nq','nv','nu','na','nhistory','neq','nmocap','nuserdata','nplugin','npluginstate','nbody')})
    setattr(proxy,field,1)
    with pytest.raises(ValueError,match='Nonempty'):model_inventory(proxy)


def test_pack_layout_matches_cpu_full_integration_and_preserves_external_forces(model):
    data=native_data(model);native=copy_native_integration(model,data,2)
    cpu=mujoco.MjData(model)
    for world in range(2):
        cpu.time=float(data.time[world])
        for name in ('qpos','qvel','qacc_warmstart','ctrl','qfrc_applied','xfrc_applied'):
            getattr(cpu,name)[:]=getattr(data,name)[world].numpy()
        expected=np.empty(410);mujoco.mj_getState(model,cpu,expected,SPEC)
        assert np.array_equal(native[world],expected)
    destination=native_data(model)
    for value in vars(destination).values():value.zero_()
    pointers={name:value.data_ptr() for name,value in vars(destination).items()}
    write_native_integration(model,destination,native)
    assert np.array_equal(copy_native_integration(model,destination,2),native)
    assert pointers=={name:value.data_ptr() for name,value in vars(destination).items()}


def test_restore_rejects_precision_loss_before_mutating_destination(model):
    data=native_data(model);values=copy_native_integration(model,data,2);original=values.copy()
    values[0,0]=1.25+1.e-12
    with pytest.raises(ValueError,match='represented exactly'):write_native_integration(model,data,values)
    assert np.array_equal(copy_native_integration(model,data,2),original)


def test_missing_positive_size_and_surprising_empty_storage_fail_closed(model):
    data=native_data(model);del data.qfrc_applied
    with pytest.raises(ValueError,match='Missing native'):copy_native_integration(model,data,2)
    data=native_data(model);data.act=torch.zeros(2,1)
    with pytest.raises(ValueError,match='zero-sized'):copy_native_integration(model,data,2)


def test_restore_preserves_full_native_state_history_and_support_interval(model):
    source=fake_env(model);saved=snapshot(source);destination=fake_env(model,source=False)
    receipt=restore_comparison(destination,saved,trial_id='synthetic_independent_comparison',reference_ids=np.array([1,1]))
    assert receipt['complete_native_integration_equal'] and receipt['physical_state_history_preserved']
    assert np.array_equal(copy_native_integration(model,destination.sim.data,2),copy_native_integration(model,source.sim.data,2))
    assert torch.equal(destination.last_action,source.last_action)
    assert torch.equal(destination.initial_up,source.initial_up)
    assert torch.equal(destination.ticks,source.ticks)
    assert torch.equal(destination._last_physical_fallen,source._last_physical_fallen)
    assert destination.physics_steps_executed==8 and destination.actor_updates_executed==2
    assert destination.ids_ref.tolist()==[1,1] and destination.duration.tolist()==[3.,3.]
    recaptured=snapshot(destination)
    assert saved.physical_controller_identity==recaptured.physical_controller_identity
    assert saved.states[0].digest()!=recaptured.states[0].digest()


def test_restore_is_impossible_during_an_existing_execution_or_twice(model):
    saved=snapshot(fake_env(model));destination=fake_env(model)
    before=copy_native_integration(model,destination.sim.data,2)
    with pytest.raises(ValueError,match='fresh independent'):restore_comparison(destination,saved,trial_id='bad',reference_ids=np.array([0,0]))
    assert np.array_equal(before,copy_native_integration(model,destination.sim.data,2))
    destination=fake_env(model,source=False)
    restore_comparison(destination,saved,trial_id='first',reference_ids=np.array([0,0]))
    with pytest.raises(ValueError,match='fresh independent'):restore_comparison(destination,saved,trial_id='second',reference_ids=np.array([0,0]))


def test_unknown_adapter_state_or_stateful_simulation_extensions_are_rejected(model):
    env=fake_env(model);env.hidden_controller_history=torch.ones(2,48)
    with pytest.raises(ValueError,match='Uninventoried'):snapshot(env)
    env=fake_env(model);env.sim._sensor_context=object()
    with pytest.raises(ValueError,match='Sensor'):snapshot(env)


def test_capture_requires_true_completed_action_and_support_clock(model):
    env=fake_env(model)
    with pytest.raises(ValueError,match='completed held action'):
        capture_comparison(env,held_raw_action=np.zeros((2,48)),reference_phase='preparation',support_interval_end_s=np.array([1.25,1.25]))
    with pytest.raises(ValueError,match='current support'):
        capture_comparison(env,held_raw_action=env.last_action.numpy(),reference_phase='preparation',support_interval_end_s=np.array([0.,0.]))
    env.physics_steps_executed=9
    with pytest.raises(ValueError,match='complete four-step'):snapshot(env)


def test_snapshot_serialization_is_immutable_hash_checked_and_bounded(model,tmp_path):
    saved=snapshot(fake_env(model));path=save_snapshot(saved,tmp_path/'snapshot')
    loaded=load_snapshot(path)
    assert loaded.identity==saved.identity and loaded.physical_controller_identity==saved.physical_controller_identity
    with pytest.raises(ValueError):loaded.states[0].native_integration_state.flags.writeable=True
    record=json.loads(path.read_text());record['worlds'][0]['reference_clock_s']+=.005
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='identity differs'):load_snapshot(path)


def test_physical_controller_identity_excludes_only_explicit_reference_metadata(model):
    saved=snapshot(fake_env(model));left=saved.states[0]
    changed=replace(left,reference_id='3'*64,adapter_state={**left.adapter_state,'ids_ref':np.array(1),
        'phase_denominator_s':np.array(9.,np.float32)})
    reference_variant=NativeComparisonSnapshot((changed,*saved.states[1:]),saved.inventory_json)
    assert saved.physical_controller_identity==reference_variant.physical_controller_identity
    changed=replace(left,previous_raw_action=np.ones(48))
    history_variant=NativeComparisonSnapshot((changed,*saved.states[1:]),saved.inventory_json)
    assert saved.physical_controller_identity!=history_variant.physical_controller_identity


def test_native_parity_never_executes_without_counted_gpu(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='counted CUDA'):
        native_parity('unused','jab','unused',tmp_path/'never_created',300.)
    assert not (tmp_path/'never_created').exists()


def test_all_sixteen_worlds_roundtrip_without_reducing_complete_state_inventory(model,tmp_path):
    saved=snapshot(fake_env(model,n=16));path=save_snapshot(saved,tmp_path/'sixteen_worlds')
    assert len(load_snapshot(path).states)==16
    assert load_snapshot(path).physical_controller_identity==saved.physical_controller_identity


def test_static_controller_gain_change_is_rejected_before_restore(model):
    saved=snapshot(fake_env(model));destination=fake_env(model,source=False)
    destination.kp[0]+=1.
    before=copy_native_integration(model,destination.sim.data,2)
    with pytest.raises(ValueError,match='Static native drive'):restore_comparison(destination,saved,trial_id='gain_tamper',reference_ids=np.array([0,0]))
    assert np.array_equal(before,copy_native_integration(model,destination.sim.data,2))


def test_inconsistent_saved_clocks_are_rejected_before_any_native_mutation(model):
    saved=snapshot(fake_env(model));first=replace(saved.states[0],actor_update_index=3)
    bad=NativeComparisonSnapshot((first,*saved.states[1:]),saved.inventory_json)
    destination=fake_env(model,source=False);before=copy_native_integration(model,destination.sim.data,2)
    with pytest.raises(ValueError,match='clocks are inconsistent'):
        restore_comparison(destination,bad,trial_id='clock_tamper',reference_ids=np.array([0,0]))
    assert np.array_equal(before,copy_native_integration(model,destination.sim.data,2))


def test_changed_archive_hash_or_extra_payload_never_restores(model,tmp_path):
    path=save_snapshot(snapshot(fake_env(model)),tmp_path/'tampered')
    record=json.loads(path.read_text());archive=tmp_path/'tampered/state.npz'
    with np.load(archive,allow_pickle=False) as data:arrays={key:value.copy() for key,value in data.items()}
    arrays['extra_ignored_state']=np.ones(1)
    with archive.open('wb') as stream:np.savez_compressed(stream,**arrays)
    with pytest.raises(ValueError,match='Changed or unbounded'):load_snapshot(path)
    # Rehashing the archive cannot make an unlisted controller field disappear.
    from human_punching.wbc.provenance import sha256_file
    record['archive_sha256']=sha256_file(archive);path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='exact state inventory'):load_snapshot(path)


def test_public_native_access_requires_counted_allocation_before_touching_cuda(model,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    env=fake_env(model);env.device='cuda:0'
    with pytest.raises(RuntimeError,match='counted CUDA'):snapshot(env)
