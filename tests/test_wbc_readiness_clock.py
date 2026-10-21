"""CPU launch reducer tests. No environment construction or physics integration."""

import pytest
import torch

from human_punching.wbc.telemetry.readiness_clock import Failure, Phase, PlanStatus, ReadinessClock, ReadinessConfig
from task10_telemetry_helpers import state


def setup(n=2, quiet=2, maximum=12, dtype=torch.float64):
    s = state(n, dtype=dtype)
    limits = torch.tensor([[-1., 1.]]*3, dtype=dtype)
    caps = torch.full((3,), 10., dtype=dtype)
    r = ReadinessClock(n, limits, caps, context_width=7, config=ReadinessConfig(quiet, maximum))
    r.start_episode(list(range(n)), s, nominal_guard=s,
        request_id=torch.arange(n), time_origin_s=torch.full((n,), 100., dtype=torch.float64))
    p = {"target_w": torch.tensor([[0., 0., 1.]]*n, dtype=dtype),
        "g1_launch_state": torch.arange(n*7, dtype=dtype).reshape(n, 7),
        "requested_ttc_s": torch.full((n,), .1, dtype=dtype),
        "outward_end_s": torch.full((n,), .2, dtype=dtype),
        "family_id": torch.arange(n, dtype=torch.int64)%4,
        "plan_handle": torch.arange(n, dtype=torch.int64)+10,
        "plan_digest": torch.ones((n, 32), dtype=torch.uint8),
        "status": torch.full((n,), int(PlanStatus.ADMITTED), dtype=torch.int64),
        "plan_matches_guard_context": torch.ones(n, dtype=torch.bool)}
    return r, s, p


def step(r, s, p, *, cycle_failed=None, recovered=None, tick=None):
    false = torch.zeros(r.n, dtype=torch.bool)
    r.after_substep(r.tick+1 if tick is None else tick, s, prospective_target_w=p["target_w"],
        cycle_failed=false if cycle_failed is None else cycle_failed,
        recovery_confirmed=false if recovered is None else recovered)


def window(r, s, p, **kwargs):
    for _ in range(4): step(r, s, p, **kwargs)
    return r.boundary(p)


def test_initialization_is_not_launch_or_ready_duration():
    r, s, p = setup()
    assert not r.boundary(p).any()
    assert not r.launched.any() and not r.reference_tick.any()
    assert r.tick.tolist() == [0, 0] and r.quiet_intervals.tolist() == [0, 0]
    assert torch.isnan(r.snapshot()["launch_time_s"]).all()


def test_qualifying_guard_launches_on_boundary_before_first_strike_integration():
    r, s, p = setup(); r.boundary(p)
    for _ in range(3): step(r, s, p)
    assert r.quiet_intervals.tolist() == [2, 2] and not r.launched.any()
    with pytest.raises(ValueError, match="20 ms boundary"): r.boundary(p)
    step(r, s, p)
    assert r.boundary(p).all()
    assert r.launch_tick.tolist() == [4, 4] and r.reference_tick.tolist() == [0, 0]
    assert r.snapshot()["launch_time_s"].tolist() == [100.02, 100.02]
    step(r, s, p)
    assert r.reference_tick.tolist() == [1, 1]


def test_environments_settle_independently_and_keep_their_own_ttc_origin():
    r, s, p = setup(); unready = {k: v.clone() for k, v in s.items()}; unready["joint_velocity"][1, 0] = 1.
    assert window(r, unready, p).tolist() == [True, False]
    assert r.reference_tick.tolist() == [0, 0]
    assert window(r, s, p).tolist() == [False, True]
    assert r.launch_tick.tolist() == [4, 8] and r.reference_tick.tolist() == [4, 0]
    assert r.launch_count.tolist() == [1, 1]


def test_initial_submillimetre_floor_gap_must_physically_settle_without_pose_fix():
    r, s, p = setup()
    unsupported = {k: v.clone() for k, v in s.items()}
    unsupported["support_contact"][:] = False
    unsupported["support_patch_contact"][:] = False
    unsupported["support_anchor_position_w"][:, :, 2] = .0004
    before = {k: v.clone() for k, v in unsupported.items()}
    assert not window(r, unsupported, p).any()
    assert r.phase.tolist() == [Phase.SETTLING]*2 and not r.failure_count.any()
    assert all(torch.equal(v, before[k]) for k, v in unsupported.items())
    assert window(r, s, p).all()
    assert r.launch_tick.tolist() == [8, 8]


def test_no_support_times_out_with_no_invented_launch_and_no_dropped_rows():
    r, s, p = setup(); s["support_contact"][:] = False
    for _ in range(3): window(r, s, p)
    report = r.terminal_receipt()
    assert report["failure_category"].tolist() == [Failure.GUARD_TIMEOUT]*2
    assert report["failure_tick"].tolist() == [12, 12]
    assert report["launch_tick"].tolist() == [-1, -1]
    assert torch.isnan(report["launch_time_s"]).all()
    assert report["counts"]["failed"].item() == 2
    assert sum(value.item() for value in report["counts"].values()) == 2
    assert report["failure_count"].tolist() == [1, 1]
    window(r, s, p)
    assert r.failure_count.tolist() == [1, 1]


def test_ready_launch_at_exact_settling_deadline_wins_over_timeout():
    r, s, p = setup(quiet=10, maximum=12)
    assert not window(r, s, p).any(); assert not window(r, s, p).any()
    assert window(r, s, p).all()
    assert r.launch_tick.tolist() == [12, 12] and not r.failure_count.any()


@pytest.mark.parametrize("status,code", [(PlanStatus.REJECTED, Failure.REQUEST_REJECTED),
    (PlanStatus.FAILED, Failure.PLANNER_FAILED)])
def test_explicit_planner_rejections_are_counted_without_waiting_for_guard(status, code):
    r, s, p = setup(); p["status"][:] = status
    assert not r.boundary(p).any()
    assert r.failure.tolist() == [code]*2 and r.failure_tick.tolist() == [0, 0]
    assert not r.launched.any()


@pytest.mark.parametrize("missing,code", [(True, Failure.PLAN_UNAVAILABLE), (False, Failure.PLAN_CONTEXT_MISMATCH)])
def test_ready_but_unavailable_or_stale_plan_has_separate_failure(missing, code):
    r, s, p = setup()
    if missing: p["status"][:] = PlanStatus.PREPARING
    else: p["plan_matches_guard_context"][:] = False
    for _ in range(3): window(r, s, p)
    assert r.failure.tolist() == [code]*2 and not r.launched.any()


def test_launch_rechecks_exact_target_clearance_at_boundary():
    r, s, p = setup()
    for _ in range(4): step(r, s, p)
    p["target_w"][:, 0] = .2
    assert not r.boundary(p).any()
    assert not r.launched.any()


def test_frozen_commitment_ignores_subsequent_g1_target_or_plan_changes():
    r, s, p = setup(); assert window(r, s, p).all()
    committed = {k: v.clone() for k, v in r.committed.items()}
    for name in ("target_w", "g1_launch_state", "requested_ttc_s", "outward_end_s"):
        p[name].fill_(100.)
    p["plan_handle"] += 100; p["plan_digest"][:] = 7; p["family_id"][:] = 3
    p["status"][:] = PlanStatus.REJECTED
    assert not window(r, s, p).any()
    assert all(torch.equal(v, committed[k]) for k, v in r.committed.items())
    assert r.phase.tolist() == [Phase.STRIKE]*2 and r.launch_count.tolist() == [1, 1]


def test_readiness_uses_nominal_guard_and_cannot_redefine_a_drifted_pose_as_ready():
    r, s, p = setup()
    drifted = {k: v.clone() for k, v in s.items()}; drifted["root_position_w"][:, 0] = .04
    for _ in range(3): window(r, drifted, p)
    assert r.failure.tolist() == [Failure.GUARD_TIMEOUT]*2


def test_return_goal_keeps_nominal_posture_but_actual_ready_footprint():
    r, s, p = setup()
    ready = {k: v.clone() for k, v in s.items()}
    ready["root_position_w"][:, 0] = .01
    ready["joint_position"][:, 0] = .01
    ready["support_anchor_position_w"][:, :, 0] += .02
    assert window(r, ready, p).all()
    goal = r.recovery_goal()
    assert torch.equal(goal["root_position_w"], s["root_position_w"])
    assert torch.equal(goal["joint_position"], s["joint_position"])
    assert torch.equal(goal["support_anchor_position_w"], ready["support_anchor_position_w"])
    assert torch.equal(r.launch_guard["root_position_w"], ready["root_position_w"])


@pytest.mark.parametrize("kind,code", [("speed", Failure.GUARD_TIMEOUT), ("foot", Failure.FOOT_REPOSITION),
    ("joint", Failure.JOINT_LIMIT), ("effort", Failure.EFFORT_LIMIT), ("height", Failure.BALANCE),
    ("external_support", Failure.EXTERNAL_SUPPORT), ("forbidden_collision", Failure.FORBIDDEN_COLLISION),
    ("nan", Failure.INVALID_TELEMETRY)])
def test_readiness_failures_are_distinct_and_never_drop_attempts(kind, code):
    r, s, p = setup()
    if kind == "speed": s["joint_velocity"][:, 0] = .6
    elif kind == "foot": s["support_anchor_position_w"][:, 0, 0] += .04
    elif kind == "joint": s["joint_position"][:, 0] = 1.03
    elif kind == "effort": s["effort"][:, 0] = 11.
    elif kind == "height": s["root_position_w"][:, 2] = .7
    elif kind == "nan": s["root_position_w"][:, 0] = torch.nan
    else: s[kind][:] = True
    for _ in range(3): window(r, s, p)
    assert r.failure.tolist() == [code]*2
    assert r.failure_count.tolist() == [1, 1] and r.attempt_count.tolist() == [1, 1]
    assert not r.launched.any()


def test_readiness_quiet_interval_resets_after_one_noisy_tick():
    r, s, p = setup(quiet=5, maximum=16)
    window(r, s, p)
    noisy = {k: v.clone() for k, v in s.items()}; noisy["joint_velocity"][:] = .6
    step(r, noisy, p)
    assert not r.quiet_intervals.any()
    for _ in range(3): step(r, s, p)
    assert not r.boundary(p).any()
    assert window(r, s, p).all()
    assert r.launch_tick.tolist() == [12, 12]


def test_reference_clock_stays_zero_until_launch_even_during_long_settling():
    r, s, p = setup(quiet=9, maximum=16)
    for _ in range(2):
        assert not window(r, s, p).any()
        assert not r.reference_tick.any()
    assert window(r, s, p).all()
    assert r.reference_tick.tolist() == [0, 0]


def test_recovery_phase_and_reported_completion_do_not_relaunch():
    r, s, p = setup(); window(r, s, p)
    for _ in range(10): window(r, s, p)
    assert r.reference_tick.tolist() == [40, 40] and r.phase.tolist() == [Phase.RECOVERY]*2
    true = torch.ones(2, dtype=torch.bool)
    window(r, s, p, recovered=true)
    assert r.phase.tolist() == [Phase.RECOVERY]*2  # impossible <1 s dwell does not complete
    for _ in range(50): window(r, s, p, recovered=true)
    assert r.phase.tolist() == [Phase.COMPLETE]*2
    assert r.complete_count.tolist() == [1, 1] and r.launch_count.tolist() == [1, 1]


def test_missing_recovery_times_out_separately_from_guard_readiness():
    r, s, p = setup(); window(r, s, p)
    for _ in range(206): window(r, s, p)
    assert r.failure.tolist() == [Failure.RECOVERY_TIMEOUT]*2
    assert r.launched.all() and r.failure_count.tolist() == [1, 1]


def test_cycle_failure_retains_launch_identity_and_first_failure_tick():
    r, s, p = setup(); window(r, s, p)
    true = torch.ones(2, dtype=torch.bool)
    step(r, s, p, cycle_failed=true)
    assert r.failure.tolist() == [Failure.CYCLE_FAILURE]*2 and r.failure_tick.tolist() == [5, 5]
    assert r.launch_tick.tolist() == [4, 4]
    for _ in range(3): step(r, s, p)
    r.boundary(p)
    assert r.failure_tick.tolist() == [5, 5] and r.failure_count.tolist() == [1, 1]


def test_boundary_packet_invalid_and_duplicate_calls_fail_closed():
    r, s, p = setup(); malformed = {**p, "target_w": p["target_w"].float()}
    with pytest.raises(ValueError, match="Malformed launch packet"): r.boundary(malformed)
    assert not r.boundary(p).any()  # malformed shape check did not consume boundary
    with pytest.raises(ValueError, match="Exactly one"): r.boundary(p)


def test_admitted_packet_requires_finite_bound_plan_identity():
    r, s, p = setup(); p["plan_digest"][:] = 0
    r.boundary(p)
    assert r.failure.tolist() == [Failure.INVALID_PLAN_PACKET]*2


def test_partial_episode_restart_requires_receipt_and_preserves_other_rows():
    r, s, p = setup(); p["status"][0] = PlanStatus.REJECTED
    r.boundary(p)
    kwargs = dict(nominal_guard=s, request_id=torch.tensor([10, 1]), time_origin_s=torch.full((2,), 101., dtype=torch.float64))
    with pytest.raises(ValueError, match="preserved prior receipt"): r.start_episode([0], s, **kwargs)
    receipt = r.terminal_receipt(); r.start_episode([0], s, **kwargs)
    assert r.phase.tolist() == [Phase.SETTLING]*2 and r.attempt_count.tolist() == [2, 1]
    assert receipt["failure_category"].tolist() == [Failure.REQUEST_REJECTED, Failure.NONE]
    assert r.request_id.tolist() == [10, 1]
    assert not r.nominal["joint_position"].any()


def test_no_host_tensor_reads_at_5ms_or_20ms_hot_callbacks(monkeypatch):
    r, s, p = setup()
    def forbidden(*args, **kwargs): raise AssertionError("hot-path host read")
    with monkeypatch.context() as m:
        for name in ("item", "tolist", "cpu", "numpy", "__bool__"):
            m.setattr(torch.Tensor, name, forbidden)
        r.boundary(p)
        for _ in range(4): step(r, s, p)
        launched = r.boundary(p)
    assert launched.all()


def test_clock_fault_cannot_be_repaired_into_later_launch():
    r, s, p = setup(); step(r, s, p, tick=torch.tensor([2, 1]))
    for _ in range(3): step(r, s, p)
    assert r.boundary(p).tolist() == [False, True]
    assert r.failure.tolist() == [Failure.CLOCK, Failure.NONE]


def test_float32_native_dtype_preserves_boundary_event_counts():
    r, s, p = setup(dtype=torch.float32); assert window(r, s, p).all()
    assert r.launch_tick.tolist() == [4, 4] and r.reference_tick.tolist() == [0, 0]


def test_inherited_termination_counts_unready_request_before_ordinary_reset():
    r, s, p = setup(); s["support_contact"][:] = False
    window(r, s, p)
    r.record_environment_end(torch.tensor([True, False]), torch.tensor([False, True]))
    receipt = r.terminal_receipt()
    assert receipt["failure_category"].tolist() == [Failure.ENVIRONMENT_TERMINATION, Failure.ENVIRONMENT_TIMEOUT]
    assert receipt["failure_tick"].tolist() == [4, 4] and receipt["launch_tick"].tolist() == [-1, -1]
    assert receipt["environment_terminated"].tolist() == [True, False]
    r.start_episode([0], s, nominal_guard=s, request_id=torch.tensor([4, 1]),
        time_origin_s=torch.full((2,), 101., dtype=torch.float64))
    assert r.phase.tolist() == [Phase.SETTLING, Phase.FAILED]
    assert r.failure_count.tolist() == [1, 1] and r.attempt_count.tolist() == [2, 1]


def test_inherited_end_preserves_first_physical_failure_and_is_idempotent():
    r, s, p = setup(); s["external_support"][:] = True
    window(r, s, p)
    for _ in range(2): r.record_environment_end(torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    assert r.failure.tolist() == [Failure.EXTERNAL_SUPPORT]*2
    assert r.failure_tick.tolist() == [1, 1] and r.failure_count.tolist() == [1, 1]
    assert r.environment_terminated.all() and r.environment_timed_out.all()


def test_inherited_reset_receipt_cannot_hide_mid_window_reset():
    r, s, p = setup(); step(r, s, p)
    with pytest.raises(ValueError, match="policy boundary"):
        r.record_environment_end(torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))


def test_readiness_event_bridge_uses_separate_t0_without_fake_settling_contact():
    from human_punching.wbc.telemetry.event_state import PunchEventState
    ready, nominal, packet = setup()
    events = PunchEventState(2, ready.limits, ready.caps)
    ready.boundary(packet)
    for physical_tick in range(1, 13):
        if (physical_tick-1) % 4 == 0:
            events.begin_window()
        observed = {name: value.clone() for name, value in nominal.items()}
        if physical_tick <= 4:
            observed["joint_velocity"][1, 0] = 1.
        if physical_tick == 5:
            observed["glove_position_w"][0, 0, 0] = .14
        if physical_tick == 9:
            observed["glove_position_w"][1, 1, 0] = .14
        cycle_tick = torch.where(ready.launched, ready.reference_tick+1, 0)
        events.step(cycle_tick, observed)
        step(ready, observed, packet)
        if physical_tick % 4 == 0:
            launched = ready.boundary(packet)
            ids = launched.nonzero().flatten()
            if ids.numel():
                c = ready.committed
                events.launch(ids, observed, target_w=c["target_w"],
                    active_hand=((c["family_id"] == 1) | (c["family_id"] == 3)).long(),
                    ttc_s=c["requested_ttc_s"], outward_end_s=c["outward_end_s"], nominal_guard=ready.nominal)
        if physical_tick == 4:
            assert ready.launch_tick.tolist() == [4, -1]
            assert events.first_contact.tolist() == [-1, -1]
        if physical_tick == 8:
            assert ready.launch_tick.tolist() == [4, 8]
            assert events.first_contact.tolist() == [1, -1]
    assert events.first_contact.tolist() == [1, 1]
    assert events.tick.tolist() == [8, 4]
    assert ready.reference_tick.tolist() == [8, 4]
