"""CPU checks for the single learned-cross development comparison."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import inspect

import pytest

from human_punching import phase2_learned_cross_target_control_v2 as control

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def plan():
    return control.create_plan('outputs/task10b/p2familycatalog01c/manifest.json',
        'outputs/task10b/p2familytargetaudit01c/audit.json')


def test_exact_learned_checkpoint_and_old_target_references(plan):
    assert plan['evaluation_checkpoint']['sha256']=='e3d720096c46de135c5e512b6db5a952032d8d32ad677ec22b00c23e07a6a2ce'
    assert plan['accepted_task09_ancestor']['sha256']=='61aa4d27f5a2ec815bedfc7326b120cfd8f3b0aa1e6bbd03e534476c33f11984'
    assert plan['catalog']['sha256']=='dc52182c9227e3ca8542ceca9635dee8eef6f5104b40f72657afde030b044c1b'
    admitted=[r for r in plan['requests'] if '_c4251_' in r['candidate_id']]
    assert len(admitted)==16
    assert {r['reference_sha256'] for r in admitted}=={
        '3007b1f888f166665fd60f35ab078ce5f3241385eb6e0dcddbaa2857dc9fd331',
        '64f00089c84b1402bccc382a5ab9303370e2c014ea56c739060be2697dc2a78e'}
    assert all(r['checkpoint_sha256']==plan['evaluation_checkpoint']['sha256'] for r in plan['requests'])


def test_full40_denominator_and_zero_credit_before_any_attempt(plan):
    ledger=control._original_ledger()+control.trial_ledger(plan)
    assert len(ledger)==40 and len({r['trial_id'] for r in ledger})==40
    assert sum(r['status']=='reference_unavailable' for r in ledger)==16
    assert sum(r['native_execution_admitted'] for r in ledger)==24
    assert not any(r['numeric_complete_cycle'] or r['execution_complete'] for r in ledger)
    assert plan['original_regression']['row']['requested_ttc_s'] is None
    assert plan['original_regression']['planned_actor_padded_physics_steps']==960
    assert plan['original_regression']['stop_before_targets_on_any_numeric_or_dynamics_failure'] is True
    assert plan['training'] is False and plan['fresh_transition_cap']==0
    assert 'cross_original_bank_03' in plan['video_selection']['presampled_trial_ids']


@pytest.mark.parametrize('key,value',[('planned_trials',39),('native_attempt_capacity',25),('family','left_hook')])
def test_plan_tampering_is_rejected(plan,key,value):
    altered=deepcopy(plan);altered[key]=value
    with pytest.raises(ValueError):control.validate_plan(altered)


def panel():
    return dict(status='executed',hard_faults=[],trials=[dict(trial_id=f'cross_original_bank_{i:02d}',
        numeric_complete_cycle=True,disposition='executed') for i in range(8)])


def test_original_regression_requires_every_trial_and_exact_counter():
    valid=panel()
    assert control.regression_passed(valid,dict(passed=True),960)
    assert not control.regression_passed(valid,dict(passed=True),964)
    assert not control.regression_passed(valid,dict(passed=False),960)
    invalid=panel();invalid['hard_faults']=[dict(type='effort_cap_violation')]
    assert not control.regression_passed(invalid,dict(passed=True),960)
    invalid=panel();invalid['trials'].pop()
    assert not control.regression_passed(invalid,dict(passed=True),960)


@pytest.mark.parametrize('index',range(8))
def test_each_failed_original_trial_stops_target_admission(index):
    invalid=panel();invalid['trials'][index]['numeric_complete_cycle']=False
    assert not control.regression_passed(invalid,dict(passed=True),960)


def test_selected_checkpoint_restores_actual_weights_on_cpu(plan):
    import torch
    paths={'checkpoint':ROOT/plan['evaluation_checkpoint']['path']}
    policy=control.route.fresh_policy(paths,'cpu')
    parity=control.route.restored_policy_parity(paths['checkpoint'],policy,torch.linspace(-1,1,369)[None])
    assert parity==dict(parameters_equal=True,actions_equal=True,maximum_action_error=0.)
    ancestor=control.route.fresh_policy({'checkpoint':ROOT/plan['accepted_task09_ancestor']['path']},'cpu')
    assert control.route.policy_digest(policy)!=control.route.policy_digest(ancestor)


def test_original_stage_uses_actual_deadline_wrapper_and_strict_native_signature(monkeypatch):
    """Exercise real wrapper argument consumption before any GPU constructor."""
    cls=control.DeadlineNativeSpecialistEnv
    native=cls.__mro__[1]
    actual=inspect.signature(native.__init__)
    clock_native=inspect.signature(control.route.NativeSpecialistEnv.__init__)
    assert actual==clock_native
    assert 'deadline' not in actual.parameters
    sentinel=object();args=object();rows=[object()];seen=[]
    with pytest.raises(TypeError):actual.bind(None,args,rows,num_envs=8,deadline=sentinel)
    def strict_intercept(*values,**kwargs):
        bound=actual.bind(*values,**kwargs)
        seen.append(bound.arguments)
    monkeypatch.setattr(native,'__init__',strict_intercept)
    instance=cls(args,rows,num_envs=8,deadline=sentinel)
    assert instance.execution_deadline is sentinel
    assert len(seen)==1 and seen[0]['args'] is args and seen[0]['rows'] is rows and seen[0]['num_envs']==8
    assert 'deadline' not in seen[0]
    source=inspect.getsource(control._run_original_regression)
    assert 'env=DeadlineNativeSpecialistEnv(args,[row],num_envs=8,deadline=deadline)' in source
    assert 'env=route.NativeSpecialistEnv(' not in source


def test_driver_dispatch_and_unshortened_child(plan):
    source=ROOT/'scripts/task10b_phase2_learned_cross_target_control_v2.py'
    spec=importlib.util.spec_from_file_location('learned_cross_driver_test',source)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    record=dict(mode=driver.MODE,inputs={'plan':'outputs/task10b/p2familycontrolprep01/plan_c.json'},
        application_wall_seconds=300,allocation_wall_seconds=360,target_control_version=control.VERSION,
        target_control_trials=40,target_control_cells=4,target_control_child_wall_seconds=210,
        target_control_environments=8,fresh_transition_cap=0,target_control_native_attempt_capacity=24,
        target_control_target_requests=32,target_control_original_regressions=8,
        target_control_media_reserve_seconds=300,physical_qualification=False,output='outputs/task10b/not_executed/run')
    command=driver.native_command(record,210)
    assert command[2:4]==['-m','human_punching.phase2_learned_cross_target_control_v2']
    assert command[-2:]==['--wall-seconds','210.0']
    for wall in (0,209,271):
        with pytest.raises(RuntimeError):driver.native_command(record,wall)
    record['target_control_media_reserve_seconds']=0
    with pytest.raises(ValueError):driver.native_command(record,270)
