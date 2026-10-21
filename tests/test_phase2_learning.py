"""Focused CPU contracts for Phase 2 original PPO, not native qualification."""
from pathlib import Path
from types import SimpleNamespace
import json

import mujoco
import numpy as np
import pytest
import torch

from human_punching.phase2_learning import (
    PilotBudget, ReferenceLibrary, NativeSpecialistEnv, first_crossing, first_glove_crossing,
    timing_result, audit_trace, policy_digest, save_policy,
    restored_policy_parity, validation_score, family_arguments, validation_without_rng_side_effects,
)
from human_punching.jab_learning import ActorCritic
from human_punching.wbc.provenance import sha256_file
from human_punching.wbc.release_baseline_probe import legacy_observations


def test_pilot_budget_is_finite_and_starts_original_geometry():
    budget = PilotBudget()
    assert budget.transition_cap == 262144
    assert list(range(0, budget.maximum_updates+1, budget.validation_every)) == [0, 8, 16, 24, 32]
    for kwargs in ({"environments": 256}, {"rollout_steps": 128}, {"maximum_updates": 33},
                   {"maximum_wall_s": 14401}, {"validation_every": 4}):
        with pytest.raises(ValueError):
            PilotBudget(**kwargs)


def test_first_contact_cannot_be_repaired_by_later_crossing():
    times = np.array([0., .49, .50, .60, .61])
    distances = np.array([.3, .13, .3, .13, .1])
    first = first_crossing(times, distances)
    assert first < .50
    result = timing_result(first, 0., .60)
    assert result["hit"] and result["early_hit"] and not result["timing_pass"]
    assert timing_result(None, 0., .60)["actual_ttc_s"] is None
    assert not timing_result(.6, None, .60)["hit"]


def test_first_geometric_crossing_detects_pass_through_between_saved_frames():
    glove = np.zeros((2,2,3))
    glove[:,0,0] = [-.2,.2]
    glove[:,1,0] = 10.
    value = first_glove_crossing([.49,.495],glove,[0.,0.,0.],0)
    assert value == pytest.approx(.49075)
    assert first_glove_crossing([.49,.495],glove,[0.,0.,0.],1) is None


@pytest.mark.parametrize("requested,actual,passed", [(.5,.499,False),(.5,.50,True),
    (.6,.579,False),(.6,.58,True),(.6,.62,True),(.6,.621,False)])
def test_timing_uses_early_floor_and_twenty_ms_tolerance(requested, actual, passed):
    assert timing_result(actual+1., 1., requested)["timing_pass"] is passed


@pytest.fixture
def small_library(tmp_path):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="Chest"><freejoint/><geom type="sphere" size=".1" mass="1"/>
    <site name="LeftHand_center" pos=".4 0 0"/><site name="RightHand_center" pos="-.4 0 0"/>
    <body><joint name="a" axis="0 0 1"/><geom type="sphere" size=".1" mass="1"/></body>
    </body></worldbody><actuator><motor joint="a" ctrlrange="-10 10"/></actuator></mujoco>''')
    rows = []
    for index, duration in enumerate((.04, .08)):
        t = np.linspace(0., duration, 5)
        q = np.tile(model.qpos0, (5, 1))
        q[:, 0] = np.linspace(0, 1+index, 5)
        v = np.zeros((5, model.nv))
        path = tmp_path/f"ref{index}.npz"
        np.savez(path, t=t, qpos=q, qvel=v, torque=np.zeros((5, 1)),
            head_center=np.array([1., 0., 1.]), active_hand_index=0,
            guard_hold_start=duration, attack_window_reference_s=np.array([0., duration]))
        rows.append(dict(candidate_id=f"cell{index}", family="jab", kind="uniform",
            reference_path=str(path), reference_sha256=sha256_file(path),
            legacy_runtime_path=str(path), legacy_runtime_sha256=sha256_file(path),
            phase_denominator_s=duration+1., episode_duration_s=duration+1.,
            launch_tick=0, outward_end_tick=4, kinematic_admission=True,
            requested_ttc_s=.6+.15*index, target_w=[1., 0., 1.]))
    return model, rows, ReferenceLibrary(model, rows)


def test_reference_batching_and_per_world_full_duration(small_library):
    _, _, library = small_library
    ids = torch.tensor([1, 0, 1, 0])
    clock = torch.tensor([.04, .02, 0., 100.])
    before = library.content_digest()
    q = library.sample("qpos", ids, clock)
    np.testing.assert_allclose(q[:, 0].numpy(), [1., .5, 0., 1.])
    np.testing.assert_allclose(library.durations[ids].numpy(), [1.08, 1.04, 1.08, 1.04])
    q[:] = 123  # Sampling never lends mutable resident storage to the caller.
    assert library.content_digest() == before
    assert library.sample("guard_glove_chest_pos", ids, clock).shape == (4, 2, 3)
    assert library.sample("episode_duration", ids, clock).shape == (4,)


def test_reference_rejections_and_identity_are_enforced(small_library):
    model, rows, _ = small_library
    with pytest.raises(ValueError, match="Rejected"):
        ReferenceLibrary(model, [dict(rows[0], kinematic_admission=False)])
    with pytest.raises(ValueError, match="actor-boundary"):
        ReferenceLibrary(model, [dict(rows[0], launch_tick=1)])
    with pytest.raises(ValueError, match="changed"):
        ReferenceLibrary(model, [dict(rows[0], reference_sha256="wrong")])
    with pytest.raises(ValueError, match="denominator"):
        ReferenceLibrary(model, [dict(rows[0], phase_denominator_s=99.)])
    with pytest.raises(ValueError, match="resident-byte"):
        ReferenceLibrary(model, [rows[0]], byte_limit=1)


def test_original_observation_schema_previous_raw_and_future_offsets():
    env = NativeSpecialistEnv.__new__(NativeSpecialistEnv)
    n = env.n = 3
    env.device, env.dt = "cpu", .005
    env.qadr, env.vadr = np.arange(7, 55), np.arange(6, 54)
    env.ticks = torch.tensor([0, 4, 9])
    env.ids_ref = torch.zeros(n, dtype=torch.long)
    env.rejected = torch.zeros(n, dtype=torch.bool)
    env.last_action = torch.arange(n*48, dtype=torch.float32).reshape(n, 48)/100.
    q = torch.zeros((n, 55)); q[:, 3] = 1.; q[:, 2] = 1.
    v = torch.zeros((n, 54))
    gloves = torch.zeros((n, 2, 3))
    env.hands = [0, 1]
    env.sim = SimpleNamespace(data=SimpleNamespace(qpos=q, qvel=v, site_xpos=gloves))
    times = torch.tensor([0., .5, 1.])
    qref = torch.zeros((3, 55)); qref[:, 3] = 1.; qref[:, 2] = 1.; qref[:, 7] = torch.tensor([0., .1, .2])
    reference = {"t": times, "qpos": qref, "qvel": torch.zeros((3, 54)), "glove_pos": torch.zeros((3, 2, 3))}
    library = ReferenceLibrary.__new__(ReferenceLibrary)
    library.arrays, library.rows, library.durations = [reference], [{}], torch.tensor([2.])
    env.library = library
    before = env.last_action.clone()
    expected = legacy_observations(q, v, gloves, env.last_action, env.clock, reference, 2.)
    assert torch.equal(env.observe(), expected)
    assert torch.equal(env.last_action, before)
    # Rejection holds reference future features and preserves current history.
    env.rejected[1] = True
    assert torch.equal(env.ref("qpos", .25)[1], env.ref("qpos", 0.)[1])
    assert torch.equal(env.last_action, before)


def synthetic_trace():
    intervals, worlds, joints = 8, 2, 48
    env = SimpleNamespace(qadr=np.arange(7,55), vadr=np.arange(6,54),
        kp=torch.ones(joints)*2, kd=torch.ones(joints)*3,
        torque_limits=np.tile([-10., 10.], (joints,1)))
    q = np.zeros((intervals+1,worlds,55)); q[:,:,3] = 1
    v = np.zeros((intervals+1,worlds,54))
    actions = np.array([np.zeros((worlds,joints)), np.ones((worlds,joints))*2])
    a = dict(qpos=q,qvel=v,pre_qpos=q[:-1].copy(),pre_qvel=v[:-1].copy(),
        pre_drive=np.ones((intervals,worlds,joints)),actuator_force=np.ones((intervals+1,worlds,joints)),
        ctrl=np.ones((intervals+1,worlds,joints)),qfrc_actuator=np.zeros((intervals+1,worlds,54)),
        qfrc_applied=np.zeros((intervals+1,worlds,54)),xfrc_applied=np.zeros((intervals+1,worlds,3,6)),
        time=np.repeat((np.arange(intervals+1)*.005)[:,None],worlds,axis=1),
        reset_count=np.ones((intervals+1,worlds)), policy_episode_tick=np.array([0,4]),
        policy_previous_raw_action=np.array([np.zeros((worlds,joints)),actions[0]]),
        policy_action=actions, pre_raw_action=np.repeat(actions,4,axis=0))
    return env,a


def test_independent_trace_audits_pair_forces_clock_and_history():
    env, arrays = synthetic_trace()
    assert audit_trace(arrays,env)["passed"]
    arrays["policy_previous_raw_action"][1,0,0] = .1
    assert not audit_trace(arrays,env)["checks"]["raw_previous_action"]
    env, arrays = synthetic_trace()
    arrays["actuator_force"][1,0,0] = 11.
    result = audit_trace(arrays,env)
    assert not result["passed"] and not result["checks"]["caps"]
    env, arrays = synthetic_trace()
    arrays["pre_qpos"][1,0,7] = .1
    assert not audit_trace(arrays,env)["checks"]["interval_state_pairing"]


def test_physical_selection_keeps_rejections_and_uses_no_ppo_reward():
    panel = dict(status="executed",hard_faults=[],trials=[
        dict(numeric_complete_cycle=True,timing={"timing_pass":True},outward_active_glove_rms_m=.01),
        dict(numeric_complete_cycle=False,disposition="offline_rejection")])
    assert validation_score(panel) == (.5,.5,-.01)
    panel["hard_faults"] = ["nonfinite"]
    assert validation_score(panel) is None


def test_checkpoint_roundtrip_covers_parameters_and_actor_actions(tmp_path):
    torch.manual_seed(17)
    policy = ActorCritic(369,48)
    with torch.no_grad():
        policy.actor[-1].weight.normal_(0,.01)
    optimizer = torch.optim.Adam(policy.parameters(),lr=1.e-4)
    assert not optimizer.state  # Fresh optimizer, no inherited moments.
    obs = torch.randn((9,369))
    library = SimpleNamespace(commitment="test-only",rows=[])
    before = policy_digest(policy)
    checkpoint = save_policy(tmp_path/"policy.pt",policy,optimizer,ancestry={"seed":17},
                             updates=0,fresh_transitions=0,library=library)
    assert policy_digest(policy) == before
    parity = restored_policy_parity(checkpoint["path"],policy,obs)
    assert parity == dict(parameters_equal=True,actions_equal=True,maximum_action_error=0.)
    with pytest.raises(FileExistsError):
        save_policy(tmp_path/"policy.pt",policy,optimizer,ancestry={},updates=0,fresh_transitions=0,library=library)


def test_family_hyperparameters_are_accepted_release_values():
    root = Path(__file__).resolve().parents[1]/"outputs/four-punch-milestone"
    expected = {"jab":(.2,0.,0.),"cross":(.1,.65,.3),"left_hook":(.05,.65,0.),"right_hook":(.05,.65,.3)}
    for family, values in expected.items():
        args, paths = family_arguments(root,family)
        assert (args.joint_limit_penalty,args.attack_tracking_weight,args.guard_joint_weight) == values
        assert args.learning_rate == 1.e-4 and args.target_kl == .03
        assert paths["checkpoint"].is_file()


def test_validation_does_not_rewind_training_seed_stream():
    torch.manual_seed(317)
    np.random.seed(317)
    torch_before, numpy_before = torch.get_rng_state().clone(), np.random.get_state()
    def callback(policy, update):
        torch.manual_seed(444)
        np.random.seed(444)
        torch.rand(10)
        np.random.randn(10)
        return update
    assert validation_without_rng_side_effects(callback,None,8,device="cpu") == 8
    assert torch.equal(torch.get_rng_state(),torch_before)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1],numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
