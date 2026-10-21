"""State hashing and mutation detection only; no physics or learning."""
from types import SimpleNamespace as NS
import pytest
import torch

from human_punching.wbc import training_witness as m
from human_punching.wbc.telemetry import cycle_runtime, runtime_telemetry
from human_punching.wbc.runtime.training_initialization import TrainingInitializer, TrainingInitializationCfg


def fixture(monkeypatch):
    import warp
    monkeypatch.setattr(warp, "to_torch", lambda value: value)
    from human_punching.wbc.telemetry.native_qualification import PHYSICAL_FIELDS
    data = NS(**{name: torch.zeros(2, 3) for name in (*PHYSICAL_FIELDS, "qacc_warmstart")})
    contact = NS(**{name: torch.zeros(2, 3) for name in
                    ("geom", "worldid", "pos", "dist", "dim", "efc_address")})
    native = NS(efc=NS(force=torch.zeros(2, 3)), contact=contact,
                nacon=torch.zeros(2), overflow=torch.zeros(2))
    command = NS(physics_steps=torch.tensor([4, 8]), trajectory_ids=torch.tensor([0, 1]))
    action = NS(_raw_actions=torch.zeros(2, 48))
    history = NS(_buffer=torch.arange(12.).reshape(2, 2, 3), _pointer=0)
    obs = NS(_obs_buffer={"actor": torch.zeros(2, 398)},
        _group_obs_term_history_buffer={"actor": {"test": history}},
        _group_obs_term_delay_buffer={"actor": {}})
    observer = NS(actual_samples=8, window=NS(count=4, values={"finite": torch.ones(2)}))
    bridge = NS(readiness=NS(phase=torch.zeros(2)), events=NS(tick=torch.zeros(2)),
                pending_reset=torch.zeros(2, dtype=torch.bool),
                training_initializer=TrainingInitializer(2, "cpu", TrainingInitializationCfg(.75, 4)))
    bridge.training_initializer.packet = {"target_w": torch.zeros(2, 3),
                                          "plan_digest": torch.zeros(2, 4, dtype=torch.int64)}
    env = NS(sim=NS(data=data, wp_data=native), common_step_counter=2,
        command_manager=NS(get_term=lambda _: command),
        action_manager=NS(get_term=lambda _: action), observation_manager=obs)
    setattr(env, cycle_runtime.ATTRIBUTE, bridge)
    setattr(env, runtime_telemetry.ATTRIBUTE, observer)
    env.step = env.reset = obs.compute = lambda *a, **kw: pytest.fail("No execution method permitted")
    return env, command, action, history


@pytest.mark.parametrize("field", ["warmstart", "contact", "command", "action", "history", "observer"])
def test_witness_detects_mutable_training_buffers(monkeypatch, field):
    env, command, action, history = fixture(monkeypatch)
    before = m.training_state_witness(env)
    tensor = {"warmstart": env.sim.data.qacc_warmstart,
              "contact": env.sim.wp_data.contact.pos, "command": command.physics_steps,
              "action": action._raw_actions, "history": history._buffer,
              "observer": getattr(env, runtime_telemetry.ATTRIBUTE).window.values["finite"]}[field]
    tensor.flatten()[0] += 1
    with pytest.raises(RuntimeError, match="changed training state"):
        m.assert_training_state_unchanged(before, m.training_state_witness(env))


def test_hashing_does_not_advance_rng_or_mutate_state(monkeypatch):
    env, *_ = fixture(monkeypatch)
    env.sim.wp_data.contact.pos[0, 0] = float("nan")  # inactive native slots may contain NaN
    torch.manual_seed(17)
    rng = torch.get_rng_state().clone()
    before, after = m.training_state_witness(env), m.training_state_witness(env)
    m.assert_training_state_unchanged(before, after)
    assert torch.equal(rng, torch.get_rng_state())
    assert before["gpu_state_clones_retained"] is False
    assert "qacc_warmstart" in before["fields"]["native"]


def test_full_learner_and_counter_state_is_covered(monkeypatch):
    env, *_ = fixture(monkeypatch)
    weight = torch.ones(3)
    runner = NS(alg=NS(save=lambda: {"actor_state_dict": {"weight": weight},
                   "optimizer_state_dict": {"state": {0: {"step": torch.tensor(20.)}}}}),
                current_learning_iteration=0, logger=NS(tot_timesteps=196608, tot_time=1.))
    before = m.training_state_witness(env, runner=runner)
    weight[0] = 2
    with pytest.raises(RuntimeError, match="learner"):
        m.assert_training_state_unchanged(before, m.training_state_witness(env, runner=runner))


def test_witness_refuses_missing_solver_state_and_tampered_receipt(monkeypatch):
    env, *_ = fixture(monkeypatch)
    before = m.training_state_witness(env)
    changed = {**before, "sha256": "0"*64}
    with pytest.raises(ValueError, match="identity"):
        m.assert_training_state_unchanged(before, changed)
    del env.sim.data.qacc_warmstart
    with pytest.raises(ValueError, match="warmstart"):
        m.training_state_witness(env)


@pytest.mark.parametrize("field", ["selected", "prepared", "support_seen", "grace_ticks",
    "grace_total_ticks", "reference_starts_count", "packet", "configuration"])
def test_witness_detects_training_initialization_mutations(monkeypatch, field):
    env, *_ = fixture(monkeypatch)
    initializer = getattr(env, cycle_runtime.ATTRIBUTE).training_initializer
    before = m.training_state_witness(env)
    if field == "packet":
        initializer.packet["target_w"][0, 0] = 1.
    elif field == "configuration":
        initializer.config = TrainingInitializationCfg(.5, 4)
    else:
        getattr(initializer, field)[0] = 1
    with pytest.raises(RuntimeError, match="training_initializer"):
        m.assert_training_state_unchanged(before, m.training_state_witness(env))
