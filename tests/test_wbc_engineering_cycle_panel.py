"""Engineering-panel identity and independently reduced outcomes; no dynamics."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from task10_telemetry_helpers import state
from human_punching.wbc import engineering_cycle_panel as m
from human_punching.wbc.telemetry.cycle_runtime import CycleRuntime, JsonReceiptSink
from human_punching.wbc.telemetry.readiness_clock import ReadinessConfig, PlanStatus, Phase


def expectation(**changes):
    args = {name: "a"*64 for name in (
        "policy_sha256", "normalization_sha256", "checkpoint_contract_sha256",
        "reference_manifest_sha256", "recipe_manifest_sha256", "engineering_admission_sha256",
        "measurement_contract_sha256")}
    args.update(completed_iterations=250, training_seed=101)
    args.update(changes)
    return m.EngineeringPanelExpectation(**args)


@pytest.mark.parametrize("changes", [dict(completed_iterations=True), dict(training_seed=-1),
    dict(episodes_per_family=0), dict(policy_sha256="0"*63), dict(scope="production")])
def test_panel_requires_exact_scoped_frozen_identities(changes):
    with pytest.raises(ValueError):
        expectation(**changes)


def fixture_arrays():
    n = 4
    initial = state(n, dtype=torch.float32)
    packet = {"target_w": torch.tensor([[0., 0., 1.]]*n),
        "g1_launch_state": torch.zeros(n, 84), "requested_ttc_s": torch.full((n,), .1),
        "outward_end_s": torch.full((n,), .2), "family_id": torch.arange(n),
        "plan_handle": torch.arange(n), "plan_digest": torch.ones(n, 32, dtype=torch.uint8),
        "status": torch.full((n,), PlanStatus.ADMITTED, dtype=torch.int64),
        "plan_matches_guard_context": torch.ones(n, dtype=torch.bool), "actor_g1_context": torch.zeros(n, 77)}
    body = {"joint_limits_rad": [[-1., 1.]]*3, "effort": [10.]*3}
    receipts = []
    config = ReadinessConfig(quiet_ticks=2, maximum_settling_ticks=12)
    live = CycleRuntime(n, torch.tensor(body["joint_limits_rad"]), torch.tensor(body["effort"]),
        context_width=84, nominal_provider=lambda: initial, packet_provider=lambda: packet,
        request_ids=lambda: torch.arange(n)+n, reference_gloves=lambda: torch.zeros(n, 2, 3),
        receipt_sink=lambda row: receipts.append(JsonReceiptSink._plain(row)), config=config)
    live.after_pending_reset(torch.arange(n), initial, torch.zeros(n, dtype=torch.int64),
                             torch.full((n,), 100., dtype=torch.float64))
    endpoints = []
    for tick in range(1, 249):
        if (tick-1) % 4 == 0:
            live.begin_window()
        current = {name: value.clone() for name, value in initial.items()}
        # Row zero never becomes ready; its later glove motion must not invent
        # a punch event. The remaining rows launch at the real boundary tick 4.
        current["root_linear_velocity_w"][0, 0] = .2
        if tick >= 13:
            current["glove_position_w"][0, 0] = torch.tensor([.13, 0., 1.])
        if tick == 24:
            current["glove_position_w"][1, 1] = torch.tensor([.13, 0., 1.])
        if tick == 5:
            current["glove_position_w"][2, 1] = torch.tensor([.13, 0., 1.])  # Wrong hand.
            current["joint_position"][3, 0] = 1.03
        live.after_substep(torch.full((n,), tick, dtype=torch.int64), current)
        endpoints.append({name: value.numpy().copy() for name, value in current.items()})
        if tick % 4 == 0:
            live.command_boundary()
    arrays = {"initial": {k: v.numpy().copy() for k, v in initial.items()},
        "nominal": {k: v.numpy().copy() for k, v in initial.items()},
        "packet": {k: v.numpy().copy() for k, v in packet.items()},
        "pre_interval": {"time": np.full((248, n), 100.)},
        "endpoint": m.stack(endpoints)}
    recipes = [{"family": family, "recipe_id": family} for family in m.FAMILIES]
    return live, arrays, body, receipts, recipes, config


def test_native_endpoint_replay_preserves_first_failures_and_real_launches():
    live, arrays, body, receipts, recipes, config = fixture_arrays()
    independent, replayed = m.replay_attempts(arrays, body, readiness=config)
    assert m._receipt_decisions(receipts) == m._receipt_decisions(replayed)
    assert torch.equal(live.readiness.phase, independent.readiness.phase)
    summary = m.summarize_attempts(independent, arrays, recipes)
    assert summary["attempts"] == 4 and summary["all_terminal"]
    assert summary["launched"] == 3 and summary["strict_cycle_success"] == 1
    unready, success, wrong_hand, limit = summary["episodes"]
    assert not unready["launched"] and unready["launch_tick"] is None
    assert unready["failure_category"] == "GUARD_TIMEOUT"
    assert unready["first_contact_cycle_tick"] is None
    assert unready["contact_ttc_error_s"] is unready["outward_minimum_target_distance_m"] is None
    assert unready["terminal_physics_tick"] == 12
    assert unready["post_terminal_diagnostic_ticks"] == 236
    assert success["launch_tick"] == 4 and success["launch_time_s"] == pytest.approx(100.020)
    assert success["first_contact_cycle_tick"] == 20
    assert success["contact_ttc_error_s"] == pytest.approx(0., abs=1e-8)
    assert wrong_hand["failure_category"] == limit["failure_category"] == "CYCLE_FAILURE"
    assert summary["families"]["jab"]["contact_ttc_error_s"]["observed_count"] == 0
    assert summary["families"]["jab"]["contact_ttc_error_s"]["mean"] is None
    assert summary["families"]["jab"]["launch_rate"] == 0.
    assert summary["families"]["cross"]["strict_cycle_success_rate"] == 1.


def test_independent_replay_detects_changed_first_contact_endpoint():
    _, arrays, body, receipts, _, config = fixture_arrays()
    arrays["endpoint"]["glove_position_w"][23, 1, 1, 0] = .3
    _, replayed = m.replay_attempts(arrays, body, readiness=config)
    assert m._receipt_decisions(receipts) != m._receipt_decisions(replayed)


def test_terminal_receipts_cannot_duplicate_or_drop_recipe_ownership():
    _, _, _, receipts, _, _ = fixture_arrays()
    with pytest.raises(ValueError, match="Duplicate terminal"):
        m._receipt_decisions([receipts[0], receipts[0]])


def test_observation_contract_checks_shared_semantics_and_order():
    original = {"schema": "test", "previous_action": "clipped", "groups": {
        "actor": {"shape": [398], "terms": ["body", "phase"]}, "critic": {"shape": [800]}}}
    m.verify_panel_observations(original, deepcopy(original))
    for changed in (dict(original, previous_action="raw"),
                    dict(original, groups={"actor": {"shape": [398], "terms": ["phase", "body"]}})):
        with pytest.raises(ValueError, match="observation schema"):
            m.verify_panel_observations(changed, original)
    changed = deepcopy(original); changed["groups"]["critic"]["shape"] = [900]
    m.verify_panel_observations(changed, original)  # Not an actor input.


def test_conservative_memory_retains_resident_training_allocations_and_sampling_caveat():
    samples = [{"total_bytes": 1000, "free_bytes": 300, "reserved_bytes": 500,
                "torch_peak_reserved_bytes": 550},
               {"total_bytes": 1000, "free_bytes": 400, "reserved_bytes": 350,
                "torch_peak_reserved_bytes": 600}]
    result = m.memory_summary(samples)
    assert result["conservative_peak_used_bytes"] == 850
    assert result["maximum_sampled_non_torch_bytes"] == 250
    assert result["headroom_fraction"] == .15
    assert result["caller_allocations_remain_resident"]
    assert result["driver_sampling_is_not_continuous_peak_measurement"]


def test_no_recipe_can_physically_initialize_twice():
    command = m.PlannedCommand.__new__(m.PlannedCommand)
    command.physical_resets = np.asarray([1, 0])
    with pytest.raises(RuntimeError, match="second physical initialization"):
        command.reset(torch.tensor([0]))


def test_reward_capture_retains_original_pre_sanitizer_values_without_recomputation(monkeypatch):
    from human_punching.wbc.runtime.imitation_rewards import CACHE_ATTRIBUTE, UPSTREAM_NAMES
    from human_punching.wbc.runtime.task_rewards import DEFAULT_WEIGHTS
    imitation = SimpleNamespace(serial=4, raw_values={name: torch.ones(2) for name in UPSTREAM_NAMES})
    task = SimpleNamespace(serial=4, cache={name: torch.zeros(2) for name in DEFAULT_WEIGHTS})
    env = SimpleNamespace(**{CACHE_ATTRIBUTE: imitation})
    monkeypatch.setattr(m, "get_observer", lambda _: SimpleNamespace(actual_samples=4))
    monkeypatch.setattr(m, "get_cycle_runtime", lambda _: SimpleNamespace(_task_reward_cache=task))
    result = m.original_fresh_rewards(env)
    assert result.shape == (2, 25)
    assert torch.equal(result[:, :8], torch.ones(2, 8))
    imitation.raw_values[UPSTREAM_NAMES[0]][0] = float("nan")
    assert torch.isnan(m.original_fresh_rewards(env)[0, 0])  # Audit must see the fault.
    task.serial = 8
    with pytest.raises(RuntimeError, match="original fresh rewards"):
        m.original_fresh_rewards(env)


def test_exception_receipt_preserves_failure_location_with_bounded_traceback():
    try:
        raise RuntimeError("native stream diagnostic")
    except RuntimeError as error:
        receipt = m.exception_receipt(error, stage="native_execution")
    assert receipt["type"] == "RuntimeError" and receipt["stage"] == "native_execution"
    assert "test_exception_receipt" in receipt["traceback"]
    assert "native stream diagnostic" in receipt["traceback"]
    assert not receipt["traceback_truncated"]
    large = m.exception_receipt(RuntimeError("x"*20000), stage="offline_verification")
    assert len(large["traceback"]) == large["traceback_character_limit"] == 16384
    assert large["traceback_truncated"]
