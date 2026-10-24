"""Pinned inference, source-time recovery and actual delayed-history semantics."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from human_punching.task10d.g1 import execution_dependencies
from human_punching.task10d.task08_policy_diagnostic import (
    FrozenTask08Policy, H1_FINAL_SHA256, Task08ActorHistory,
    Task08EpisodeOutcomes, checkpoint_payload,
)


@pytest.fixture(scope="module")
def policy():
    manifest, _, _ = checkpoint_payload()
    action = manifest["identity"]["action"]
    plant = SimpleNamespace(device="cpu", nominal=execution_dependencies().nominal,
        scale=torch.tensor(action["native_scale"]), default=torch.tensor(action["native_offset"]),
        ctrl_lower=torch.tensor(action["physical_target_lower"]),
        ctrl_upper=torch.tensor(action["physical_target_upper"]))
    return FrozenTask08Policy(plant)


def test_real_selected_checkpoint_identity_and_original_composition(policy):
    assert policy.identity["sha256"] == H1_FINAL_SHA256
    assert policy.identity["selection"] == "user-selected warm H1 final"
    assert policy.identity["archived_pitch_up_deg"] == -15.
    assert policy.identity["diagnostic_pitch_up_deg"] == 25.
    actor = policy.actor
    obs = torch.randn(7, 948, generator=torch.Generator().manual_seed(710)) * .05
    with torch.no_grad():
        nominal = actor.nominal(actor.standing_observation(obs))
        residual = torch.tanh(actor.residual(obs))
        expected = actor.limit(actor.native_offset + actor.native_scale
                               * (nominal + actor.b_delta_diagonal * residual))
    torch.testing.assert_close(policy.action(obs), expected, atol=0, rtol=0)
    assert policy.inference_health(obs)["passed"]
    assert policy.assert_unchanged()["passed"]
    assert all(not p.requires_grad for p in actor.parameters())


def test_checkpoint_pin_is_mandatory():
    with pytest.raises(ValueError, match="SHA256"):
        checkpoint_payload(sha256="0" * 64)


def test_history_uses_delivered_packets_at_original_lags_and_keeps_raw_actions():
    from boxing.perception.latency import VisualDelayBuffer
    count = 2
    history = Task08ActorHistory(count, "cpu")
    delay = VisualDelayBuffer(count, fixed_delay_ticks=3)
    mask = torch.ones(count, dtype=torch.bool)
    plant = SimpleNamespace(actor99=torch.zeros(count, 99), clean111=torch.zeros(count, 111),
        data=SimpleNamespace(qpos=torch.ones(count, 36)))
    valid = torch.ones(count, 9, 16, dtype=torch.bool)
    for tick in range(22):
        image = torch.full((count, 9, 16), tick / 40.)
        times = torch.full((count,), tick * .02, dtype=torch.float64)
        (delay.reset if tick == 0 else delay.append)(mask, image, image, valid, valid, times)
        packet = delay.delivered(times)
        plant.actor99.copy_(torch.arange(99)[None] + tick * 100.)
        plant.actor99[:, 96:] = 0  # actual command channels stay zero
        before = plant.actor99.clone()
        exposure = dict(capture_timestamps=times, delivered_normalized=packet["corrupted"],
            delivered_clean_normalized=packet["clean"], delivered_frame_ids=packet["frame_ids"],
            delivered_capture_timestamps=packet["capture_timestamps"], padding=packet["padding"],
            delay_ticks=packet["delay_ticks"])
        result = history.acquire(plant, exposure, tick * .02, reset=tick == 0)
        torch.testing.assert_close(plant.actor99, before, atol=0, rtol=0)
        ids = torch.tensor([max(0, tick - 3 - lag) for lag in (0, 3, 8, 18)])
        torch.testing.assert_close(result["O_rsa"][:, 372:].reshape(count, 4, 144)[0, :, 0],
                                   ids.float() / 40., atol=0, rtol=0)
        torch.testing.assert_close(result["visual_frame_ids"][0], ids)
        torch.testing.assert_close(result["visual_timestamps"][0], ids.double() * .02)
        torch.testing.assert_close(result["O_stand"][:, 64:93], plant.actor99[:, 67:96])
        assert torch.equal(result["O_stand"][:, 93:], torch.zeros(count, 3))
        assert "O_rsa_GT" not in result
    latest_raw = result["O_rsa"][:, 256:372].reshape(count, 4, 29)[0]
    torch.testing.assert_close(latest_raw, torch.stack([torch.arange(67, 96) + k * 100.
                                                     for k in (18, 19, 20, 21)]))
    # source frame0 at tick3 is real, not pre-episode; only lag18 still used padding.
    assert result["visual_padding"][0].tolist() == [False, False, False, False]


def geometry(count, energy=51.):
    return dict(head_position=torch.zeros(count, 3, dtype=torch.float64),
        mechanically_safe=torch.ones(count, dtype=torch.bool), finite=torch.ones(count, dtype=torch.bool),
        stance_energy=torch.full((count,), energy, dtype=torch.float64),
        height=torch.ones(count, dtype=torch.float64), self_collision=torch.zeros(count, dtype=torch.bool),
        joint_margin=torch.ones(count, 29, dtype=torch.float64))


def test_safety_at_reset_and_recovery_is_only_an_arrival_gate():
    reducer = Task08EpisodeOutcomes([1., 1., 1.], "cpu")
    geom = geometry(3, 40.)
    fists = torch.full((3, 2, 3), .5, dtype=torch.float64)
    fists[0, 1] = torch.tensor([.15, 0., 0.])
    first = reducer.update(geom, fists, 0.)
    assert first["stopped"].tolist() == [True, False, False]
    assert reducer.reason.tolist() == [2, 0, 0]
    # Safe and low-energy before source recovery cannot count as arrival.
    reducer.update(geom, fists, .5)
    assert reducer.reason.tolist() == [2, 0, 0]
    geom["stance_energy"][1] = 51.
    reducer.update(geom, fists, 1.)
    assert reducer.reason.tolist() == [2, 0, 1]
    geom["stance_energy"][1] = 50.
    reducer.update(geom, fists, 1.005)
    assert reducer.reason.tolist() == [2, 1, 1]
    assert reducer.event_time.tolist() == [0., 1.005, 1.]
    assert reducer.records()[0]["observed_hit"] is False  # .15 safety failure, no .14 contact


def test_shared_safety_priority_and_sampled_return_plus_three_seconds():
    reducer = Task08EpisodeOutcomes([.303, .303, .303], "cpu")
    geom = geometry(3)
    fists = torch.full((3, 2, 3), .5, dtype=torch.float64)
    reducer.update(geom, fists, .300)
    reducer.update(geom, fists, .305)
    reducer.update(geom, fists, 3.300)
    assert reducer.active.all()
    fists[0, 0] = torch.tensor([.139, 0., 0.])
    geom["stance_energy"][2] = 50.
    reducer.update(geom, fists, 3.305)
    assert reducer.reason.tolist() == [2, 6, 1]  # failure/arrival beat simultaneous cutoff
    assert reducer.hit.tolist() == [True, False, False]
    assert [row["disposition"] for row in reducer.records()] == ["fist_clearance", "return_plus_3s_cutoff", "arrival"]
    # Later states cannot overwrite first event or physical diagnostics.
    reducer.update(geometry(3, 0.), torch.zeros_like(fists), 4.)
    assert reducer.reason.tolist() == [2, 6, 1]
    assert reducer.hit.tolist() == [True, False, False]


def test_nonfinite_and_watchdog_remain_distinct_from_safe_arrival():
    reducer = Task08EpisodeOutcomes([40., 40.], "cpu")
    geom = geometry(2, 0.)
    geom["finite"][0] = False
    fists = torch.full((2, 2, 3), .5, dtype=torch.float64)
    reducer.update(geom, fists, 30.)
    assert reducer.reason.tolist() == [10, 9]
