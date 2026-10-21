"""CPU contracts for explicit training frame-zero exposure; no native steps."""
from types import SimpleNamespace

import pytest
import torch

from task10_telemetry_helpers import state
from human_punching.wbc.runtime.training_initialization import (
    TrainingInitializationCfg, TrainingInitializer, require_strict_initialization,
)
from human_punching.wbc.telemetry import cycle_runtime as m
from human_punching.wbc.telemetry.readiness_clock import Phase, PlanStatus, ReadinessConfig


def fixture(probability=1., n=4, grace=0):
    s = state(n, dtype=torch.float32)
    packet = {"target_w": torch.tensor([[0., 0., 1.]]*n),
        "g1_launch_state": torch.arange(n*7, dtype=torch.float32).reshape(n, 7),
        "requested_ttc_s": torch.full((n,), .1), "outward_end_s": torch.full((n,), .2),
        "family_id": torch.arange(n)%4, "plan_handle": torch.arange(n),
        "plan_digest": torch.ones((n, 32), dtype=torch.uint8),
        "status": torch.full((n,), PlanStatus.ADMITTED, dtype=torch.int64),
        "plan_matches_guard_context": torch.ones(n, dtype=torch.bool),
        "actor_g1_context": torch.arange(n*77, dtype=torch.float32).reshape(n, 77)}
    receipts = []
    bridge = m.CycleRuntime(n, torch.tensor([[-1., 1.]]*3), torch.full((3,), 10.),
        context_width=7, packet_provider=lambda: packet, nominal_provider=lambda: s,
        request_ids=lambda: torch.arange(n), reference_gloves=lambda: s["glove_position_w"],
        receipt_sink=receipts.append, config=ReadinessConfig(quiet_ticks=2, maximum_settling_ticks=12),
        training_initialization=TrainingInitializationCfg(probability, grace))
    command = SimpleNamespace(physics_steps=torch.zeros(n, dtype=torch.int64),
        robot_anchor_quat_w=s["root_quat_wxyz"], robot_anchor_pos_w=s["root_position_w"])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    setattr(env, m.ATTRIBUTE, bridge)
    command._env = env
    return bridge, s, packet, receipts, env, command


def reset(bridge, s, ids=None):
    if ids is None:
        ids = torch.arange(bridge.readiness.n)
    bridge.prepare_training_reset(ids)
    tick = bridge.readiness.tick.clone(); tick[ids] = 0
    bridge.after_pending_reset(ids, s, tick, torch.zeros(bridge.readiness.n, dtype=torch.float64))


def window(bridge, s):
    bridge.begin_window()
    for _ in range(4):
        bridge.after_substep(bridge.readiness.tick+1, s)
    bridge.command_boundary()


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_training_probability(value):
    with pytest.raises(ValueError):
        TrainingInitializationCfg(value)


@pytest.mark.parametrize("probability,ticks", [(0., 1), (1., -1), (1., 5), (1., True), (1., 1.0)])
def test_support_grace_is_explicit_bounded_and_training_only(probability, ticks):
    with pytest.raises(ValueError, match="support grace"):
        TrainingInitializationCfg(probability, ticks)


def test_strict_default_consumes_no_sampling_rng_or_immediate_launch():
    b, s, _, _, _, _ = fixture(0.)
    torch.manual_seed(83); before = torch.get_rng_state().clone()
    reset(b, s)
    assert torch.equal(before, torch.get_rng_state())
    assert not b.readiness.launched.any()
    assert not b.events.armed.any()
    window(b, s)
    assert b.readiness.launch_tick.tolist() == [4]*4
    assert b.training_initializer.reference_starts_count.tolist() == [0]*4


def test_first_action_condition_and_reference_agree_before_and_after_native_reset():
    b, s, packet, _, env, command = fixture()
    b.prepare_training_reset(torch.arange(4))
    before = m.cycle_condition(env)
    tick, active = m.CycleMotionCommand._cycle_clock(command)
    assert tick.tolist() == [0]*4 and active.all()
    assert before.shape == (4, 91)
    assert before[:, -4:].argmax(-1).tolist() == [1]*4
    assert torch.equal(before[:, 10:87], packet["actor_g1_context"])
    assert before[:, 8].tolist() == [0.]*4
    assert m.reference_indices(tick, active, torch.arange(4),
        torch.tensor([0, 100, 200, 300]), torch.full((4,), 100), 12)[0].tolist() == [12, 112, 212, 312]
    b.after_pending_reset(torch.arange(4), s, torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.float64))
    assert torch.equal(before, m.cycle_condition(env))
    assert b.readiness.launch_tick.tolist() == [0]*4
    assert b.readiness.quiet_intervals.tolist() == [0]*4
    assert not b.events.previous_guard.any()
    assert b.events.armed.all()
    window(b, s)
    assert b.readiness.reference_tick.tolist() == [4]*4


def test_initial_floor_gap_is_diagnostic_not_fabricated_support():
    b, s, _, _, _, _ = fixture()
    unsupported = {k: v.clone() for k, v in s.items()}
    unsupported["support_contact"][:] = False
    unsupported["support_patch_contact"][:] = False
    reset(b, unsupported)
    assert b.readiness.launched.all()
    assert not b.events.previous_guard.any()
    assert not b.readiness.last_state["support_contact"].any()
    assert not b.readiness.previous_ready.any()


def unsupported(s):
    result = {k: v.clone() for k, v in s.items()}
    result["support_contact"][:] = False
    result["support_patch_contact"][:] = False
    return result


def test_no_support_at_fifth_tick_fails_after_exactly_four_grace_ticks():
    b, s, _, _, _, _ = fixture(grace=4)
    gap = unsupported(s); saved = {k: v.clone() for k, v in gap.items()}
    reset(b, gap); window(b, gap)
    assert not b.events.flags["fall"].any()
    assert b.training_initializer.grace_ticks.tolist() == [4]*4
    assert not b.training_initializer.support_seen.any()
    assert all(torch.equal(v, gap[k]) for k, v in saved.items())
    b.begin_window(); b.after_substep(b.readiness.tick+1, gap)
    assert b.events.flags["fall"].all()
    assert b.events.first_failure_tick.tolist() == [5]*4
    assert b.training_initializer.grace_ticks.tolist() == [4]*4
    assert (b.readiness.phase == Phase.FAILED).all()


def test_first_actual_support_ends_grace_permanently():
    b, s, _, _, _, _ = fixture(grace=4)
    gap = unsupported(s); reset(b, gap); b.begin_window()
    b.after_substep(b.readiness.tick+1, gap)
    b.after_substep(b.readiness.tick+1, s)
    assert b.training_initializer.support_seen.all()
    b.after_substep(b.readiness.tick+1, gap)
    assert b.events.flags["fall"].all()
    assert b.events.first_failure_tick.tolist() == [3]*4
    assert b.training_initializer.grace_ticks.tolist() == [1]*4


@pytest.mark.parametrize("fault", ["height", "tilt", "joint_limit", "external_support", "forbidden_collision"])
def test_grace_never_suppresses_other_failures(fault):
    b, s, _, _, _, _ = fixture(grace=4)
    gap = unsupported(s); reset(b, gap); b.begin_window()
    bad = {k: v.clone() for k, v in gap.items()}
    if fault == "height": bad["root_position_w"][:, 2] *= .5
    elif fault == "tilt": bad["root_quat_wxyz"][:] = torch.tensor([.5, .5, .5, .5])
    elif fault == "joint_limit": bad["joint_position"][:, 0] = 1.03
    else: bad[fault][:] = True
    b.after_substep(b.readiness.tick+1, bad)
    assert b.events.first_failure_tick.tolist() == [1]*4
    assert (b.readiness.phase == Phase.FAILED).all()


def test_zero_grace_still_fails_on_first_unsupported_endpoint():
    b, s, _, _, _, _ = fixture(grace=0)
    gap = unsupported(s); reset(b, gap); b.begin_window()
    b.after_substep(b.readiness.tick+1, gap)
    assert b.events.flags["fall"].all()
    assert not b.training_initializer.grace_ticks.any()


def test_actual_grace_counts_are_reported_without_inventing_guard_dwell():
    b, s, _, receipts, _, _ = fixture(grace=4)
    gap = unsupported(s); reset(b, gap); window(b, gap)
    b.record_reset(torch.arange(4), torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    item = receipts[-1]["initialization"]
    assert item["initial_support_grace_ticks_limit"] == 4
    assert item["initial_support_grace_ticks_used"].tolist() == [4]*4
    assert item["initial_support_grace_ticks_cumulative"].tolist() == [4]*4
    assert not item["initial_support_seen"].any()
    assert not b.events.guard_intervals.any()
    reset(b, gap)
    assert not b.training_initializer.grace_ticks.any()
    assert b.training_initializer.grace_total_ticks.tolist() == [4]*4


def test_missing_reset_preparation_is_rejected_before_first_training_interval():
    b, s, _, _, _, _ = fixture()
    with pytest.raises(ValueError, match="prepared before the first observation"):
        b.after_pending_reset(torch.arange(4), s, torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.float64))
    assert not b.events.armed.any()


def test_training_start_never_accepts_unsafe_or_unclear_initial_state():
    for change in ("external_support", "joint_position", "glove_position_w"):
        b, s, _, _, _, _ = fixture()
        bad = {k: v.clone() for k, v in s.items()}
        if change == "external_support": bad[change][0] = True
        elif change == "joint_position": bad[change][0, 0] = 1.1
        else: bad[change][0, 0] = torch.tensor([0., 0., 1.])
        with pytest.raises(ValueError, match="finite clear frame-zero guard"):
            reset(b, bad)
        assert not b.events.armed.any()


def test_frozen_pending_packet_and_poststart_context_cannot_chase():
    b, s, p, _, _, _ = fixture()
    b.prepare_training_reset(torch.arange(4))
    wanted = {k: v.clone() for k, v in p.items()}
    p["target_w"].add_(10); p["actor_g1_context"].add_(100)
    b.after_pending_reset(torch.arange(4), s, torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.float64))
    assert torch.equal(b.readiness.committed["target_w"], wanted["target_w"])
    assert torch.equal(b.actor_context, wanted["actor_g1_context"])
    window(b, s)
    assert torch.equal(b.readiness.committed["target_w"], wanted["target_w"])
    assert torch.equal(b.actor_context, wanted["actor_g1_context"])


def test_asynchronous_subset_reset_preserves_other_clock_and_tags_receipt():
    b, s, _, receipts, _, _ = fixture()
    reset(b, s); window(b, s)
    b.record_reset(torch.tensor([1]), torch.tensor([False, True, False, False]), torch.zeros(4, dtype=torch.bool))
    item = receipts[-1]
    assert item["acceptance_eligible"] is False
    assert item["initialization"]["start_type_codes"].tolist() == [1]
    assert item["initialization"]["strict_ready_launch"].tolist() == [False]
    assert item["initialization"]["training_reference_start"].tolist() == [True]
    reset(b, s, torch.tensor([1]))
    assert b.readiness.reference_tick.tolist() == [4, 0, 4, 4]
    assert b.training_initializer.reference_starts_count.tolist() == [1, 2, 1, 1]
    window(b, s)
    assert b.readiness.reference_tick.tolist() == [8, 4, 8, 8]


def test_mixture_is_seeded_and_partial_reset_does_not_resample_other_rows():
    a = TrainingInitializer(1000, "cpu", TrainingInitializationCfg(.5))
    p = {"value": torch.arange(1000)}
    torch.manual_seed(713); a.prepare(torch.arange(1000), p)
    first = a.selected.clone()
    assert 400 < int(first.sum()) < 600
    a.prepare(torch.tensor([4]), p)
    assert torch.equal(first[5:], a.selected[5:])
    b = TrainingInitializer(1000, "cpu", TrainingInitializationCfg(.5))
    torch.manual_seed(713); b.prepare(torch.arange(1000), p)
    assert torch.equal(first, b.selected)


def test_mixed_rows_have_distinct_real_start_clocks_and_explicit_launch_counts():
    b, s, _, _, env, command = fixture(.5, n=64)
    torch.manual_seed(77)
    b.prepare_training_reset(torch.arange(64))
    selected = b.training_initializer.selected.clone()
    assert selected.any() and (~selected).any()
    condition = m.cycle_condition(env)
    assert torch.equal(condition[:, -4:].argmax(-1), selected.long())
    _, active = m.CycleMotionCommand._cycle_clock(command)
    assert torch.equal(active, selected)
    b.after_pending_reset(torch.arange(64), s, torch.zeros(64, dtype=torch.int64), torch.zeros(64, dtype=torch.float64))
    assert torch.equal(b.readiness.launched, selected)
    window(b, s)
    assert (b.readiness.launch_tick[selected] == 0).all()
    assert (b.readiness.launch_tick[~selected] == 4).all()
    tags = b.training_initializer.receipt(torch.arange(64), b.readiness)
    assert torch.equal(tags["training_reference_start"], selected)
    assert torch.equal(tags["strict_ready_launch"], ~selected)
    assert torch.equal(tags["reference_start_counts"], selected.long())
    assert torch.equal(tags["strict_ready_launch_counts"], (~selected).long())


@pytest.mark.parametrize("old_tick", [88, 172, 184])
def test_pending_strict_reset_after_training_start_has_frame_zero_clock(old_tick):
    """Regression for ti01's asynchronous training-to-strict reset boundary."""
    b, s, _, _, env, command = fixture()
    reset(b, s)
    b.readiness.tick[:] = old_tick
    b.readiness.phase[:3] = Phase.FAILED
    b.pending_reset[:] = torch.tensor([True, True, True, False])
    b.training_initializer.selected[:] = torch.tensor([False, True, False, True])
    # Row 1 restarts a training reference; rows 0/2 return to strict guard;
    # row 3 continues its existing cycle and must retain its elapsed time.
    tick, active = m.CycleMotionCommand._cycle_clock(command)
    assert tick.tolist() == [0, 0, 0, old_tick]
    assert active.tolist() == [False, True, False, True]
    assert b.readiness.reference_tick.tolist() == [old_tick]*4
    condition = m.cycle_condition(env)
    assert condition[:, -4:].argmax(-1).tolist() == [0, 1, 0, 1]
    assert condition[:3, 8].tolist() == [0., 0., 0.]
    assert condition[3, 8].item() == pytest.approx(old_tick*.005)
    indices, stationary = m.reference_indices(tick, active,
        torch.zeros(4, dtype=torch.int64), torch.tensor([10]), torch.tensor([1000]), 12)
    assert indices.tolist() == [10, 22, 10, 10+old_tick+12]
    assert stationary.tolist() == [True, False, True, False]


def test_strict_evaluation_rejects_training_mode():
    require_strict_initialization(TrainingInitializationCfg())
    with pytest.raises(ValueError, match="Strict evaluation"):
        require_strict_initialization(TrainingInitializationCfg(.5))
    b, _, _, _, _, _ = fixture()
    with pytest.raises(ValueError, match="acceptance"):
        m.CycleRuntime(4, b.readiness.limits, b.readiness.caps, context_width=7,
            packet_provider=b.packet_provider, nominal_provider=b.nominal_provider,
            request_ids=b.request_ids, reference_gloves=b.reference_gloves,
            receipt_sink=list().append, acceptance_eligible=True,
            training_initialization=TrainingInitializationCfg(1.))


def test_actual_strict_panel_builder_rejects_enabled_initialization(monkeypatch):
    from human_punching.wbc import capacity_evidence, engineering_cycle_panel
    configured = SimpleNamespace(commands={"motion": SimpleNamespace(
        training_initialization=TrainingInitializationCfg(.5))})
    monkeypatch.setattr(capacity_evidence, "configure_engineering_capacity", lambda *a, **k: configured)
    with pytest.raises(ValueError, match="Strict evaluation"):
        engineering_cycle_panel.build_engineering_panel_cfg(None, [object()], None, engineering_admission=None)


def test_reference_end_holds_last_frame_without_reset_or_offset_fabrication():
    index, stationary = m.reference_indices(torch.tensor([999]), torch.tensor([True]),
        torch.tensor([0]), torch.tensor([10]), torch.tensor([100]), 24)
    assert index.tolist() == [109] and stationary.tolist() == [True]
