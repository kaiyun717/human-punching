"""Bounded counters must never label reset-start training as strict launches."""
import copy
import json

import pytest
import torch

from human_punching.wbc.telemetry.attempt_receipts import AggregateAttemptSink
from human_punching.wbc.telemetry.readiness_clock import Phase, Failure


def receipt():
    return {"sequence": 0, "acceptance_eligible": False,
        "environment_ids": torch.arange(4), "requested_family": torch.arange(4),
        "readiness": {"phase": torch.tensor([Phase.COMPLETE, Phase.FAILED, Phase.FAILED, Phase.COMPLETE]),
            "launched": torch.tensor([True, False, True, True]),
            "failure_category": torch.tensor([Failure.NONE, Failure.GUARD_TIMEOUT, Failure.CYCLE_FAILURE, Failure.NONE])},
        "events": {"has_contact": torch.tensor([True, False, True, True])},
        "initialization": {"schema": "human_wbc_training_initialization_v1",
            "config": {"reference_start_probability": .75, "initial_support_grace_ticks": 4},
            "start_type_codes": torch.tensor([0, 0, 1, 1]),
            "strict_launch_eligible": torch.tensor([True, True, False, False]),
            "strict_ready_launch": torch.tensor([True, False, False, False]),
            "training_reference_start": torch.tensor([False, False, True, True]),
            "initial_support_grace_ticks_used": torch.tensor([0, 0, 3, 1]),
            "initial_support_grace_ticks_limit": 4, "physical_reset_only": True, "interior_rsi": False}}


def test_mode_counts_split_attempts_launches_contacts_recovery_and_failure(tmp_path):
    sink = AggregateAttemptSink(tmp_path, shared_identity={"fixture": "mixed"}, flush_every_batches=1)
    sink(receipt())
    report = json.loads((tmp_path/"aggregate.json").read_text())
    modes = report["terminal_attempts_by_start_type"]
    strict, training = modes["strict_guard_settling"], modes["training_reference_frame_zero"]
    assert strict["attempts"] == training["attempts"] == 2
    assert [strict[key] for key in ("reference_starts", "contacts", "recoveries", "failures")] == [1, 1, 1, 1]
    assert [training[key] for key in ("reference_starts", "contacts", "recoveries", "failures")] == [2, 2, 1, 1]
    assert strict["failure_categories"]["guard_timeout"] == 1
    assert training["failure_categories"]["cycle_failure"] == 1
    assert report["strict_ready_launch_attempts"] == 1
    assert report["training_reference_start_attempts"] == 2
    assert report["launched_attempts"] == 3
    assert report["initial_support_grace_ticks_used"] == 4
    assert report["initial_support_grace_attempts"] == 2
    assert not report["acceptance_eligible"]
    assert report["initialization_metadata_missing_attempts"] == 0


def test_legacy_missing_metadata_remains_unclassified_not_strict(tmp_path):
    sink = AggregateAttemptSink(tmp_path, shared_identity={"fixture": "legacy"})
    old = receipt(); del old["initialization"]
    sink(old)
    report = sink.report()
    assert report["launched_attempts"] == 3
    assert report["unclassified_launch_attempts"] == 3
    assert report["initialization_metadata_missing_attempts"] == 4
    assert report["strict_ready_launch_attempts"] == report["training_reference_start_attempts"] == 0
    assert report["terminal_attempts_by_start_type"]["unclassified_legacy"]["contacts"] == 3


@pytest.mark.parametrize("fault", ["strict_label", "mode", "probability", "grace", "scope", "acceptance"])
def test_incoherent_initialization_rejected_without_consuming_batch(tmp_path, fault):
    sink = AggregateAttemptSink(tmp_path, shared_identity={"fixture": fault})
    item = receipt(); bad = copy.deepcopy(item)
    if fault == "strict_label": bad["initialization"]["strict_ready_launch"][2] = True
    elif fault == "mode": bad["initialization"]["start_type_codes"][2] = 7
    elif fault == "probability": bad["initialization"]["config"] = {"reference_start_probability": 0., "initial_support_grace_ticks": 0}
    elif fault == "grace": bad["initialization"]["initial_support_grace_ticks_used"][0] = 1
    elif fault == "scope": bad["initialization"]["interior_rsi"] = True
    else: bad["acceptance_eligible"] = True
    with pytest.raises(ValueError):
        sink(bad)
    assert sink.report()["terminal_attempts"] == 0 and sink.last_sequence == -1
    sink(item)
    assert sink.report()["terminal_attempts"] == 4
