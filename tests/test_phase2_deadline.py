"""Cooperative deadline boundaries preserve evidence; no native/GPU execution."""
import signal

import pytest

from human_punching.phase2_deadline import BoundedExecutionInterruption, ExecutionDeadline


def test_startup_uses_original_deadline_and_preserves_serialization_reserve():
    now = [120.]
    deadline = ExecutionDeadline(390., serialization_reserve_s=90.,
        session_started_monotonic=100., clock=lambda: now[0])
    assert deadline.checkpoint("after_compile")["elapsed_wall_s"] == 20.
    now[0] = 400.
    with pytest.raises(BoundedExecutionInterruption) as interrupted:
        deadline.checkpoint("actor_boundary", progress={"executed_physics_steps": 64})
    receipt = interrupted.value.receipt
    assert receipt["reason"] == "serialization_reserve"
    assert receipt["remaining_application_wall_s"] == 90.
    assert receipt["progress"] == {"executed_physics_steps": 64}
    assert not receipt["physical_failure"]


def test_signal_handler_only_sets_flags_and_stop_waits_for_explicit_boundary():
    calls = []
    def clock():
        calls.append("clock")
        return 10.
    deadline = ExecutionDeadline(390., clock=clock)
    calls.clear()
    deadline._signal_handler(signal.SIGTERM, None)
    assert calls == []
    assert deadline.stop_requested and deadline.signal_number == signal.SIGTERM
    with pytest.raises(BoundedExecutionInterruption) as interrupted:
        deadline.checkpoint("rollout_actor_boundary", progress={"fresh_transitions": 128})
    assert interrupted.value.receipt["reason"] == "signal_request"
    assert interrupted.value.receipt["progress"]["fresh_transitions"] == 128
    assert not interrupted.value.receipt["serialization_completion_guaranteed"]


def test_real_signal_is_deferred_and_original_handler_restored():
    original = signal.getsignal(signal.SIGTERM)
    deadline = ExecutionDeadline(390.)
    with deadline.signal_handlers(signals=(signal.SIGTERM,)):
        signal.raise_signal(signal.SIGTERM)
        assert deadline.stop_requested
        with pytest.raises(BoundedExecutionInterruption):
            deadline.checkpoint("after_complete_actor_interval")
    assert signal.getsignal(signal.SIGTERM) == original


def test_handler_restore_also_happens_on_body_exception_and_nesting_rejected():
    original = signal.getsignal(signal.SIGTERM)
    deadline = ExecutionDeadline(390.)
    with pytest.raises(LookupError):
        with deadline.signal_handlers(signals=(signal.SIGTERM,)):
            with pytest.raises(RuntimeError, match="Nested"):
                with deadline.signal_handlers(signals=(signal.SIGTERM,)):
                    pass
            raise LookupError("body failure")
    assert signal.getsignal(signal.SIGTERM) == original


def test_hard_deadline_overrun_is_explicit_and_cannot_claim_full_reserve():
    now = [100.]
    deadline = ExecutionDeadline(120., serialization_reserve_s=90., clock=lambda: now[0])
    now[0] = 230.
    with pytest.raises(BoundedExecutionInterruption) as interrupted:
        deadline.checkpoint("long_uninterruptible_native_call_returned")
    assert interrupted.value.receipt["remaining_application_wall_s"] == 0.
    assert interrupted.value.receipt["hard_deadline_already_passed"]
    assert not interrupted.value.receipt["reserve_remaining_in_full"]


@pytest.mark.parametrize("wall,reserve", [(90.,90.), (14401.,90.), (390.,0.), (float("nan"),90.), (390.,float("inf"))])
def test_invalid_bounds_rejected(wall, reserve):
    with pytest.raises(ValueError):
        ExecutionDeadline(wall, serialization_reserve_s=reserve)


def test_future_start_and_unnamed_boundary_rejected():
    with pytest.raises(ValueError, match="Session start"):
        ExecutionDeadline(390., session_started_monotonic=2., clock=lambda: 1.)
    deadline = ExecutionDeadline(390.)
    with pytest.raises(ValueError, match="explicit"):
        deadline.checkpoint("")
