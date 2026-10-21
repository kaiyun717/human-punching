"""CPU tests of real adapter ordering/contracts, no physical integration."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from human_punching.wbc.telemetry.runtime_telemetry import (DT, FLAGS, FourTickWindow, ObserverCore, ObserverOptions,
                               applied_force_flags, configure_observer, window_metric,
                               ATTRIBUTE, TelemetryMetric)


def state(n=2, j=3):
    x = {key: torch.zeros(n, dtype=torch.bool) for key in FLAGS if key != "clock_failure"}
    x.update(support_anchor_position_w=torch.zeros(n, 2, 3),
        support_anchor_velocity_w=torch.zeros(n, 2, 3), joint_position=torch.zeros(n,j),
        joint_velocity=torch.zeros(n,j), root_linear_velocity_w=torch.zeros(n,3),
        root_angular_velocity_w=torch.zeros(n,3), glove_velocity_guard_frame=torch.zeros(n,2,3),
        effort=torch.zeros(n,j), foot_normal_force_n=torch.zeros(n,2),
        support_contact=torch.ones(n,2,dtype=torch.bool), support_patch_contact=torch.ones(n,2,dtype=torch.bool),
        finite_kinematics=torch.ones(n,dtype=torch.bool))
    return x


def window(n=2, j=3):
    return FourTickWindow(n, torch.tensor([[-1.,1.]]*j), torch.tensor([10.]*j))


def run(w, x, modify=None):
    w.begin(x, torch.zeros(w.n,dtype=torch.int64), torch.zeros(w.n))
    for i in range(1,5):
        if modify: modify(i,x)
        w.consume(x,torch.full((w.n,),i,dtype=torch.int64),torch.full((w.n,),i*DT))
    return w.snapshot()


def test_true_joint_and_force_peak_survives_quiet_final_tick():
    x=state();w=window()
    def mutate(i,x):
        x['joint_position'][:,0] = 1.08 if i==2 else 0.
        x['effort'][:,1] = 12. if i==3 else 0.
    result=run(w,x,mutate)
    torch.testing.assert_close(result['joint_limit_excess_rad'],torch.full((2,),.08))
    assert result['effort_peak_nm'].tolist()==[12.,12.]
    assert result['effort_excess_nm'].tolist()==[2.,2.]
    assert result['double_support_fraction'].tolist()==[1.,1.]
    assert not result['clock_failure'].any()
    assert not result['numerical_failure'].any()


def test_interval_start_slip_and_anchor_buffer_are_cloned():
    w=window();x=state()
    x['support_anchor_velocity_w'][...,0] = 2.
    w.begin(x,torch.zeros(2,dtype=torch.int64),torch.zeros(2))
    x['support_anchor_position_w'].fill_(999.)
    x['support_anchor_velocity_w'].fill_(0.)
    assert not w.previous_anchor.any()
    x['support_patch_contact'][0,1]=False
    for i in range(1,5):
        w.consume(x,torch.full((2,),i,dtype=torch.int64),torch.full((2,),i*DT))
    torch.testing.assert_close(w.snapshot()['loaded_patch_slip_m'],torch.tensor([[.01,0.],[.01,.01]]))


@pytest.mark.parametrize('fault', ['skipped_tick','clock','nan_joint','nan_force','nan_time','nan_anchor'])
def test_actual_faults_are_not_hidden(fault):
    w=window();x=state();w.begin(x,torch.zeros(2,dtype=torch.int64),torch.zeros(2))
    for i in range(1,5):
        tick=torch.full((2,),i,dtype=torch.int64);time=torch.full((2,),i*DT)
        if i==2:
            if fault=='skipped_tick':tick[0]+=1
            if fault=='clock':time[0]+=.002
            if fault=='nan_time':time[0]=float('nan')
            if fault=='nan_joint':x['joint_position'][0,0]=float('nan')
            if fault=='nan_force':x['effort'][0,0]=float('nan')
            if fault=='nan_anchor':x['support_anchor_position_w'][0,0,0]=float('nan')
        w.consume(x,tick,time)
    result=w.snapshot()
    name='clock_failure' if fault in ('skipped_tick','clock','nan_time') else 'numerical_failure'
    assert result[name].tolist()==[True,False]


def test_four_interval_gate_and_snapshot_immutability():
    w=window();x=state()
    with pytest.raises(RuntimeError):w.snapshot()
    w.begin(x,torch.zeros(2,dtype=torch.int64),torch.zeros(2))
    with pytest.raises(RuntimeError):w.begin(x,torch.zeros(2,dtype=torch.int64),torch.zeros(2))
    for i in range(1,5):w.consume(x,torch.full((2,),i,dtype=torch.int64),torch.full((2,),i*DT))
    result=w.snapshot()
    with pytest.raises(RuntimeError):w.consume(x,torch.full((2,),5,dtype=torch.int64),torch.full((2,),.025))
    result['effort_peak_nm'].fill_(999.)
    assert not w.values['effort_peak_nm'].any()


def test_all_body_wrenches_and_qfrc_cannot_cancel():
    q=torch.zeros(3,6);x=torch.zeros(3,8,6);root=torch.zeros(3,6)
    x[0,1,0]=4.;x[0,2,0]=-4.
    q[1,1]=2.;q[1,2]=-2.
    root[2,0]=1.
    external,root_act,numeric=applied_force_flags(q,x,root)
    assert external.tolist()==[True,True,False]
    assert root_act.tolist()==[False,False,True]
    assert not numeric.any()
    x[2,7,5]=float('nan')
    assert applied_force_flags(q,x,root)[2].tolist()==[False,False,True]


def test_reused_fk_contacts_use_previous_anchors_before_new_endpoint_and_no_live_write():
    # Simulated backend methods read only. The test advances its native fixture
    # itself; observer callbacks can only read the actual state/clock.
    x=state();q=torch.zeros(2,4);ctrl=torch.zeros(2,3);tick=torch.zeros(2,dtype=torch.int64);time=torch.zeros(2)
    order=[];seen=[]
    def fk():
        order.append('fk');x['support_anchor_position_w'][...,0]=q[:,0,None]
        return x
    def contact(anchors):
        order.append('contacts');seen.append(anchors.clone())
        return {key:x[key] for key in ('support_contact','support_patch_contact','foot_normal_force_n',
                                       'forbidden_collision','human_g1_collision','contact_overflow','numerical_failure')}
    def force():
        return dict(effort=x['effort'],external_support=x['external_support'],root_actuation=x['root_actuation'],
                    nonfinite_force=torch.zeros(2,dtype=torch.bool))
    obs=ObserverCore(window(),fresh_kinematics=fk,interval_contacts=contact,force_state=force,clock_state=lambda:(tick,time))
    obs.begin_policy_window();before=[v.clone() for v in (q,ctrl,tick,time)]
    for a,b in zip(before,(q,ctrl,tick,time)):assert torch.equal(a,b)
    for i in range(1,5):
        q[:,0]=i;tick.fill_(i);time.fill_(i*DT)
        before=[v.clone() for v in (q,ctrl,tick,time)]
        obs.after_native_substep()
        assert all(torch.equal(a,b) for a,b in zip(before,(q,ctrl,tick,time)))
    assert order==['fk','contacts','fk','contacts','fk','contacts','fk','contacts','fk']
    assert [v[0,0,0].item() for v in seen]==[0.,1.,2.,3.]
    assert obs.actual_samples==4
    assert not hasattr(obs,'launch_tick')
    obs.record_reset(torch.tensor([1]),terminated=torch.tensor([False,True]),timed_out=torch.tensor([False,False]))
    assert obs.reset_count.tolist()==[0,1]
    assert obs.reset_terminated.tolist()==[False,True]
    q[:,0]=0.;tick.zero_();time.zero_();obs.begin_policy_window()
    assert not obs.window.previous_anchor.any()


def test_defaults_are_inert_without_importing_native_backend():
    cfg=SimpleNamespace(anything=object())
    assert configure_observer(cfg) is cfg
    assert vars(cfg).keys()=={'anything'}
    assert ObserverOptions().enabled is False


def test_published_config_hook_preserves_action_semantics_and_uses_per_policy_maxima():
    from human_punching.wbc.runtime.action import PunchJointPositionActionCfg,PunchJointPositionAction
    old=PunchJointPositionActionCfg(entity_name='robot',actuator_names=('j0','j1'),command_name='motion',scale=.25,preserve_order=True)
    cfg=SimpleNamespace(actions={'joint_pos':old},metrics={})
    configure_observer(cfg,ObserverOptions(enabled=True))
    new=cfg.actions['joint_pos']
    assert {f.name:getattr(new,f.name) for f in __import__('dataclasses').fields(old)}==vars(old)
    # Build substitutes only local subclass; it inherits the reviewed drive law.
    from human_punching.wbc.telemetry.runtime_telemetry import ObserverAction as action_class
    assert action_class.apply_actions is PunchJointPositionAction.apply_actions
    substep=cfg.metrics['human_native_substep_valid'];assert substep.per_substep
    assert substep.func is TelemetryMetric
    assert all(not term.per_substep and term.reduce=='max' for name,term in cfg.metrics.items() if name!='human_native_substep_valid')
    with pytest.raises(ValueError):configure_observer(cfg,ObserverOptions(enabled=True))


def test_window_metric_never_averages_away_substep_violation():
    w=window();x=state();x['joint_position'][0,0]=1.1
    run(w,x)
    env=SimpleNamespace(**{ATTRIBUTE:SimpleNamespace(window=w)})
    torch.testing.assert_close(window_metric(env,'joint_limit_excess_rad'),torch.tensor([.1,0.]))
    w.count=3
    with pytest.raises(RuntimeError):window_metric(env,'effort_peak_nm')


def test_draft_contains_no_physics_calls_or_launch_fabrication():
    from human_punching.wbc.telemetry import runtime_telemetry
    tree=ast.parse(Path(runtime_telemetry.__file__).read_text())
    forbidden={'forward','step','reset','write_data_to_sim','write_root_pose_to_sim','write_joint_state_to_sim','launch'}
    calls={node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
    assert not calls&forbidden


def test_real_mjlab_torcharray_is_unwrapped_without_cast_copy_or_mutation():
    from mjlab.sim.sim_data import TorchArray
    from human_punching.wbc.telemetry.runtime_telemetry import native_tensor
    proxy=object.__new__(TorchArray)
    proxy._tensor=torch.arange(12,dtype=torch.float32).reshape(2,6)
    snapshot=proxy._tensor.clone()
    result=native_tensor(proxy)
    assert isinstance(result,torch.Tensor) and result.data_ptr()==proxy._tensor.data_ptr()
    assert native_tensor(result) is result
    assert torch.equal(snapshot,proxy._tensor)
    with pytest.raises(TypeError):native_tensor([1.,2.])


def test_nominal_geometry_preserved_until_real_selected_reset():
    from human_punching.wbc.telemetry.runtime_telemetry import get_observer
    x=state();w=window();tick=torch.zeros(2,dtype=torch.int64);time=torch.zeros(2)
    obs=ObserverCore(w,fresh_kinematics=lambda:x,interval_contacts=lambda anchor:x,
        force_state=lambda:dict(nonfinite_force=torch.zeros(2,dtype=torch.bool)),clock_state=lambda:(tick,time))
    with pytest.raises(RuntimeError):obs.nominal_guard_state()
    with pytest.raises(RuntimeError):obs.endpoint_state()
    obs.begin_policy_window()
    for i in range(1,5):
        tick.fill_(i);time.fill_(i*DT);x['support_anchor_position_w'].fill_(i)
        obs.after_native_substep()
    assert not obs.nominal_guard_state()['support_anchor_position_w'].any()
    assert obs.endpoint_state()['support_anchor_position_w'][0,0,0]==4
    assert obs.window_summary()['double_support_fraction'][0]==1
    obs.record_reset(torch.tensor([1]),terminated=torch.tensor([False,True]),timed_out=torch.zeros(2,dtype=torch.bool))
    x['support_anchor_position_w'][1]=9.
    tick[1]=0;time[1]=0
    obs.begin_policy_window()
    assert not obs.nominal_guard_state()['support_anchor_position_w'][0].any()
    assert torch.all(obs.nominal_guard_state()['support_anchor_position_w'][1]==9.)
    assert get_observer(SimpleNamespace(**{ATTRIBUTE:obs})) is obs
    with pytest.raises(RuntimeError):get_observer(SimpleNamespace())


def test_optional_lifecycle_callbacks_preserve_real_initialization_and_terminal_order():
    x=state();w=window();tick=torch.zeros(2,dtype=torch.int64);time=torch.zeros(2)
    obs=ObserverCore(w,fresh_kinematics=lambda:x,interval_contacts=lambda anchors:x,
        force_state=lambda:dict(nonfinite_force=torch.zeros(2,dtype=torch.bool)),clock_state=lambda:(tick,time))
    log=[]
    obs.after_pending_reset=lambda ids,state,tick,time:log.append(('initial',ids.tolist(),tick.tolist()))
    obs.begin_sink=lambda:log.append(('begin',))
    obs.endpoint_sink=lambda tick,state:log.append(('endpoint',tick.tolist()))
    obs.reset_sink=lambda ids,term,timeout:log.append(('terminal',ids.tolist(),term[ids].tolist()))
    obs.begin_policy_window()
    for i in range(1,5):tick.fill_(i);time.fill_(i*DT);obs.after_native_substep()
    obs.record_reset(torch.tensor([0]),terminated=torch.tensor([True,False]),timed_out=torch.zeros(2,dtype=torch.bool))
    tick[0]=0;time[0]=0
    obs.begin_policy_window()
    assert log[:2]==[('initial',[0,1],[0,0]),('begin',)]
    assert log[2:6]==[('endpoint',[i,i]) for i in range(1,5)]
    assert log[6:]==[('terminal',[0],[True]),('initial',[0],[0,4]),('begin',)]
